#!/usr/bin/env python3
"""Compare a predicted protein structure (mmCIF) to a reference structure by Ca RMSD.

Purpose
-------
Validate AlphaFold 3 predictions (e.g. produced on an Apple Silicon / MPS
machine) against an experimental reference structure (e.g. an RCSB PDB crystal
mmCIF) by computing C-alpha (Ca) RMSD after optimal rigid-body (Kabsch)
superposition.

Why sequence alignment (not residue-index) matching
---------------------------------------------------
Crystal structures routinely have missing/disordered residues, expression tags,
and residue-numbering offsets relative to a model. Matching Ca atoms by naive
residue number silently pairs the wrong residues and inflates RMSD. This tool
instead establishes residue correspondence by a GLOBAL SEQUENCE ALIGNMENT
(Needleman-Wunsch) between each chain pair, and only uses aligned positions
whose amino-acid identity is IDENTICAL in both structures. Every matched Ca pair
is therefore verified to be the same residue type.

What it reports
---------------
* Per-chain Ca RMSD: for each predicted chain, the best-matching reference chain
  (lowest RMSD among candidate reference chains), with the number of matched Ca.
* Symmetry-aware whole-complex Ca RMSD: for a homodimer (two predicted chains vs
  two reference chains) it evaluates BOTH chain assignments
  (A->A,B->B and A->B,B->A) using a SINGLE global Kabsch superposition over all
  matched Ca atoms from both chains together, and reports the lower RMSD.

Method
------
Kabsch superposition (numpy): translate both point sets to their centroids,
compute the covariance matrix H = P^T Q, take its SVD (H = U S V^T), correct for
a possible reflection using d = sign(det(V U^T)), form the optimal rotation
R = V diag(1,1,d) U^T, then RMSD = sqrt(mean(||R*P_c - Q_c||^2)).

Dependencies
------------
numpy + Python standard library only. No Biopython, no external mmCIF parser.

Usage
-----
    python compare_to_crystal.py --pred MODEL.cif --ref REFERENCE.cif

    # restrict / order chains explicitly (default: all protein chains, sorted)
    python compare_to_crystal.py --pred MODEL.cif --ref REF.cif \
        --pred-chains A,B --ref-chains A,B

    # force a fixed chain mapping (pred:ref pairs) instead of symmetry search
    python compare_to_crystal.py --pred MODEL.cif --ref REF.cif \
        --map A:A,B:B

    # emit machine-readable JSON as well as the text report
    python compare_to_crystal.py --pred MODEL.cif --ref REF.cif --json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Residue name -> one-letter code (standard 20 amino acids).
# Only standard residues are treated as protein for sequence building; this
# keeps hetero-residues (ligands, ions, phospho-residues such as TPO/SEP, water)
# out of the alignment unless they are one of the standard 20.
# ---------------------------------------------------------------------------
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


class Residue:
    """A single protein residue's Ca record."""

    __slots__ = ("seq_id", "resname", "one_letter", "coord")

    def __init__(self, seq_id: str, resname: str, one_letter: str,
                 coord: np.ndarray) -> None:
        self.seq_id = seq_id            # auth_seq_id as string (numbering may differ)
        self.resname = resname          # 3-letter residue name
        self.one_letter = one_letter    # 1-letter code
        self.coord = coord              # np.array([x, y, z])


# Ordered chain -> list[Residue]
Chains = Dict[str, List[Residue]]


# ---------------------------------------------------------------------------
# mmCIF parsing
# ---------------------------------------------------------------------------
def parse_mmcif_ca(path: str, model_num: Optional[str] = None) -> Chains:
    """Parse Ca atoms of protein residues from an mmCIF ``atom_site`` loop.

    The loop's column order is read from its header, so this works for both RCSB
    depositions and AlphaFold 3 output (which use different column subsets).

    Rules applied:
      * only ``group_PDB == 'ATOM'`` records (HETATM ligands/ions/water skipped);
      * only ``label_atom_id == 'CA'`` atoms;
      * only standard-amino-acid residues (via ``label_comp_id``);
      * one Ca per (chain, residue): the FIRST occurrence wins, so alternate
        conformations (altloc 'A'/'B') do not double-count;
      * only the first model encountered (or ``model_num`` if given).

    Chains are keyed by ``auth_asym_id`` (author chain id) so that chain labels
    match the conventional PDB chain letters (A, B, C, ...). Residues are kept in
    file order, which for these files is ascending residue number.
    """
    with open(path, "r") as fh:
        lines = fh.readlines()

    # Locate the atom_site loop header and record column indices.
    col_index: Dict[str, int] = {}
    i = 0
    n = len(lines)
    in_header = False
    data_start = None
    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()
        if stripped == "loop_":
            # Peek: is the next non-blank a _atom_site. tag?
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].strip().startswith("_atom_site."):
                in_header = True
                col_index = {}
                k = j
                while k < n and lines[k].strip().startswith("_atom_site."):
                    tag = lines[k].strip().split(".", 1)[1]
                    col_index[tag] = len(col_index)
                    k += 1
                data_start = k
                break
        i += 1

    if data_start is None:
        raise ValueError(f"No _atom_site loop found in {path}")

    required = ["group_PDB", "label_atom_id", "label_comp_id",
                "Cartn_x", "Cartn_y", "Cartn_z"]
    for tag in required:
        if tag not in col_index:
            raise ValueError(f"{path}: _atom_site missing required column {tag}")

    # Prefer author chain / residue ids; fall back to label ids if absent.
    chain_col = col_index.get("auth_asym_id", col_index.get("label_asym_id"))
    seq_col = col_index.get("auth_seq_id", col_index.get("label_seq_id"))
    model_col = col_index.get("pdbx_PDB_model_num")
    altloc_col = col_index.get("label_alt_id")

    chains: Chains = {}
    seen: Dict[Tuple[str, str], bool] = {}
    chosen_model: Optional[str] = model_num

    for line in lines[data_start:]:
        s = line.strip()
        if s == "" or s == "#" or s == "loop_" or s.startswith("_"):
            # End of the atom_site data block.
            break
        fields = s.split()
        if len(fields) < len(col_index):
            # Defensive: skip malformed / wrapped lines.
            continue
        if fields[col_index["group_PDB"]] != "ATOM":
            continue
        if fields[col_index["label_atom_id"]] != "CA":
            continue
        resname = fields[col_index["label_comp_id"]]
        one = THREE_TO_ONE.get(resname)
        if one is None:
            continue  # non-standard residue: not part of the protein sequence

        if model_col is not None:
            mnum = fields[model_col]
            if chosen_model is None:
                chosen_model = mnum
            if mnum != chosen_model:
                continue

        chain = fields[chain_col]
        seq_id = fields[seq_col]
        key = (chain, seq_id)
        if key in seen:
            continue  # first altloc / occurrence already taken
        seen[key] = True

        try:
            coord = np.array([
                float(fields[col_index["Cartn_x"]]),
                float(fields[col_index["Cartn_y"]]),
                float(fields[col_index["Cartn_z"]]),
            ], dtype=float)
        except ValueError:
            continue

        chains.setdefault(chain, []).append(
            Residue(seq_id, resname, one, coord))

    return chains


def chain_sequence(residues: Sequence[Residue]) -> str:
    """One-letter sequence of a chain (order = file order)."""
    return "".join(r.one_letter for r in residues)


# ---------------------------------------------------------------------------
# Global sequence alignment (Needleman-Wunsch, identity scoring)
# ---------------------------------------------------------------------------
def needleman_wunsch(seq1: str, seq2: str,
                     match: float = 1.0, mismatch: float = -1.0,
                     gap: float = -1.0) -> List[Tuple[Optional[int], Optional[int]]]:
    """Global alignment of two sequences.

    Returns a list of (i, j) index pairs where i indexes seq1 and j indexes
    seq2; a gap is represented by ``None`` on the corresponding side.
    """
    n, m = len(seq1), len(seq2)
    # Score matrix and traceback.
    F = np.zeros((n + 1, m + 1), dtype=float)
    F[:, 0] = np.arange(n + 1) * gap
    F[0, :] = np.arange(m + 1) * gap
    # 0 = diag, 1 = up (gap in seq2), 2 = left (gap in seq1)
    tb = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb[1:, 0] = 1
    tb[0, 1:] = 2

    for i in range(1, n + 1):
        s1 = seq1[i - 1]
        Fi = F[i]
        Fim1 = F[i - 1]
        tbi = tb[i]
        for j in range(1, m + 1):
            diag = Fim1[j - 1] + (match if s1 == seq2[j - 1] else mismatch)
            up = Fim1[j] + gap
            left = Fi[j - 1] + gap
            best = diag
            move = 0
            if up > best:
                best = up
                move = 1
            if left > best:
                best = left
                move = 2
            Fi[j] = best
            tbi[j] = move

    # Traceback from (n, m).
    aln: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = tb[i, j]
        if i > 0 and j > 0 and move == 0:
            aln.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and move == 1:
            aln.append((i - 1, None))
            i -= 1
        else:
            aln.append((None, j - 1))
            j -= 1
    aln.reverse()
    return aln


def match_ca_pairs(pred_res: Sequence[Residue], ref_res: Sequence[Residue]
                   ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Match Ca atoms of two chains via global sequence alignment.

    Only alignment columns where BOTH sides have a residue and the residue
    identities are IDENTICAL become matched Ca pairs (identity-verified).

    Returns (P, Q, n_matched, n_aligned_but_mismatched) where P and Q are
    (N, 3) coordinate arrays for pred and ref respectively.
    """
    seq_p = chain_sequence(pred_res)
    seq_r = chain_sequence(ref_res)
    aln = needleman_wunsch(seq_p, seq_r)

    P: List[np.ndarray] = []
    Q: List[np.ndarray] = []
    mismatched = 0
    for i, j in aln:
        if i is None or j is None:
            continue
        if pred_res[i].one_letter != ref_res[j].one_letter:
            mismatched += 1
            continue
        P.append(pred_res[i].coord)
        Q.append(ref_res[j].coord)

    if P:
        return np.vstack(P), np.vstack(Q), len(P), mismatched
    return np.zeros((0, 3)), np.zeros((0, 3)), 0, mismatched


def sequence_identity(pred_res: Sequence[Residue], ref_res: Sequence[Residue]
                      ) -> Tuple[float, int, int]:
    """% identity of the alignment between two chains.

    Returns (percent_identity, n_identical, n_aligned_columns) where aligned
    columns counts positions where both sides have a residue.
    """
    aln = needleman_wunsch(chain_sequence(pred_res), chain_sequence(ref_res))
    aligned = 0
    identical = 0
    for i, j in aln:
        if i is None or j is None:
            continue
        aligned += 1
        if pred_res[i].one_letter == ref_res[j].one_letter:
            identical += 1
    pct = (100.0 * identical / aligned) if aligned else 0.0
    return pct, identical, aligned


# ---------------------------------------------------------------------------
# Kabsch superposition
# ---------------------------------------------------------------------------
def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD between point sets P and Q after optimal rigid-body superposition.

    P is superposed onto Q. Both are (N, 3). Reflections are corrected so the
    result is a proper rotation.
    """
    if P.shape[0] == 0:
        return float("nan")
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_rot = Pc @ R.T
    diff = P_rot - Qc
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


# ---------------------------------------------------------------------------
# High-level comparisons
# ---------------------------------------------------------------------------
def per_chain_report(pred: Chains, ref: Chains,
                     pred_chains: Sequence[str],
                     ref_chains: Sequence[str]) -> List[dict]:
    """For each predicted chain, find the best-matching reference chain."""
    rows = []
    for pc in pred_chains:
        best = None
        for rc in ref_chains:
            P, Q, n, mm = match_ca_pairs(pred[pc], ref[rc])
            if n == 0:
                continue
            rmsd = kabsch_rmsd(P, Q)
            pct, _, _ = sequence_identity(pred[pc], ref[rc])
            cand = {
                "pred_chain": pc, "ref_chain": rc, "rmsd": rmsd,
                "n_matched": n, "n_mismatched": mm, "identity_pct": pct,
            }
            if best is None or rmsd < best["rmsd"]:
                best = cand
        if best is not None:
            rows.append(best)
    return rows


def complex_rmsd_for_mapping(pred: Chains, ref: Chains,
                             mapping: Sequence[Tuple[str, str]]
                             ) -> Tuple[float, int]:
    """Single global Kabsch RMSD over all matched Ca across a chain mapping."""
    Ps, Qs = [], []
    for pc, rc in mapping:
        P, Q, n, _mm = match_ca_pairs(pred[pc], ref[rc])
        if n:
            Ps.append(P)
            Qs.append(Q)
    if not Ps:
        return float("nan"), 0
    Pall = np.vstack(Ps)
    Qall = np.vstack(Qs)
    return kabsch_rmsd(Pall, Qall), Pall.shape[0]


def symmetry_aware_complex(pred: Chains, ref: Chains,
                           pred_chains: Sequence[str],
                           ref_chains: Sequence[str]
                           ) -> dict:
    """Try all one-to-one chain assignments; return the lowest-RMSD mapping.

    For a homodimer this evaluates both A->A,B->B and A->B,B->A. Generalizes to
    any equal-size chain sets via permutations of the reference chains.
    """
    best = None
    all_maps = []
    if len(pred_chains) != len(ref_chains):
        # Unequal chain counts: map each predicted chain to its best ref chain
        # independently (no permutation search).
        mapping = []
        for pc in pred_chains:
            cand = None
            for rc in ref_chains:
                P, Q, n, _ = match_ca_pairs(pred[pc], ref[rc])
                if n and (cand is None or kabsch_rmsd(P, Q) < cand[1]):
                    cand = (rc, kabsch_rmsd(P, Q))
            if cand:
                mapping.append((pc, cand[0]))
        rmsd, n = complex_rmsd_for_mapping(pred, ref, mapping)
        return {"best_mapping": mapping, "rmsd": rmsd, "n_matched": n,
                "all_mappings": [{"mapping": mapping, "rmsd": rmsd,
                                  "n_matched": n}]}

    for perm in itertools.permutations(ref_chains):
        mapping = list(zip(pred_chains, perm))
        rmsd, n = complex_rmsd_for_mapping(pred, ref, mapping)
        entry = {"mapping": mapping, "rmsd": rmsd, "n_matched": n}
        all_maps.append(entry)
        if best is None or (rmsd == rmsd and rmsd < best["rmsd"]):  # nan-safe
            best = entry
    return {"best_mapping": best["mapping"], "rmsd": best["rmsd"],
            "n_matched": best["n_matched"], "all_mappings": all_maps}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_chain_list(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [c.strip() for c in s.split(",") if c.strip()]


def parse_mapping(s: Optional[str]) -> Optional[List[Tuple[str, str]]]:
    if not s:
        return None
    out = []
    for pair in s.split(","):
        pc, rc = pair.split(":")
        out.append((pc.strip(), rc.strip()))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ca RMSD of a predicted mmCIF vs a reference mmCIF "
                    "(sequence-alignment matched, Kabsch superposed).")
    ap.add_argument("--pred", required=True, help="predicted mmCIF path")
    ap.add_argument("--ref", required=True, help="reference mmCIF path")
    ap.add_argument("--pred-chains", help="comma list, e.g. A,B "
                    "(default: all protein chains, sorted)")
    ap.add_argument("--ref-chains", help="comma list, e.g. A,B "
                    "(default: all protein chains, sorted)")
    ap.add_argument("--map", dest="mapping",
                    help="fixed pred:ref chain mapping, e.g. A:A,B:B "
                    "(skips symmetry search)")
    ap.add_argument("--json", action="store_true",
                    help="also print a JSON block")
    args = ap.parse_args(argv)

    pred = parse_mmcif_ca(args.pred)
    ref = parse_mmcif_ca(args.ref)

    pred_chains = parse_chain_list(args.pred_chains) or sorted(pred.keys())
    ref_chains = parse_chain_list(args.ref_chains) or sorted(ref.keys())

    for c in pred_chains:
        if c not in pred:
            ap.error(f"predicted chain {c!r} not found (have {sorted(pred)})")
    for c in ref_chains:
        if c not in ref:
            ap.error(f"reference chain {c!r} not found (have {sorted(ref)})")

    print("=" * 72)
    print("Ca RMSD comparison")
    print(f"  pred: {args.pred}")
    print(f"  ref : {args.ref}")
    print("=" * 72)
    print("Chain sizes (resolved Ca):")
    for c in pred_chains:
        print(f"  pred {c}: {len(pred[c]):4d} residues")
    for c in ref_chains:
        print(f"  ref  {c}: {len(ref[c]):4d} residues")
    print()

    result: dict = {"pred": args.pred, "ref": args.ref}

    # Per-chain best match.
    print("Per-chain best match (predicted chain -> best reference chain):")
    pc_rows = per_chain_report(pred, ref, pred_chains, ref_chains)
    result["per_chain"] = pc_rows
    for r in pc_rows:
        print(f"  {r['pred_chain']} -> {r['ref_chain']}: "
              f"Ca RMSD = {r['rmsd']:.3f} A over {r['n_matched']} atoms "
              f"(identity {r['identity_pct']:.1f}%, "
              f"{r['n_mismatched']} aligned mismatches skipped)")
    print()

    # Whole-complex.
    if args.mapping:
        mapping = parse_mapping(args.mapping)
        rmsd, n = complex_rmsd_for_mapping(pred, ref, mapping)
        result["complex"] = {"best_mapping": mapping, "rmsd": rmsd,
                             "n_matched": n, "all_mappings": [
                                 {"mapping": mapping, "rmsd": rmsd,
                                  "n_matched": n}]}
        print("Whole-complex (fixed mapping):")
    else:
        result["complex"] = symmetry_aware_complex(
            pred, ref, pred_chains, ref_chains)
        print("Whole-complex (symmetry-aware, single global superposition):")

    comp = result["complex"]
    for entry in comp["all_mappings"]:
        mp = ",".join(f"{a}->{b}" for a, b in entry["mapping"])
        tag = "  <-- best" if entry["mapping"] == comp["best_mapping"] else ""
        print(f"  [{mp}] Ca RMSD = {entry['rmsd']:.3f} A "
              f"over {entry['n_matched']} atoms{tag}")
    print()
    best_mp = ",".join(f"{a}->{b}" for a, b in comp["best_mapping"])
    print(f"BEST complex Ca RMSD = {comp['rmsd']:.3f} A "
          f"({best_mp}, {comp['n_matched']} atoms)")

    if args.json:
        print()
        print("JSON:")
        print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
