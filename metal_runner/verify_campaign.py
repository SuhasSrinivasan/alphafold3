#!/usr/bin/env python3
"""Post-campaign verification of AlphaFold 3 predictions against experimental references.

Purpose
=======
Run AFTER a large AF3 benchmark campaign (e.g. produced on an Apple Silicon / MPS
machine) to score every predicted structure against its experimental PDB
reference (an RCSB mmCIF). It computes and reports, per target:

  1. PROTEIN accuracy    - per-chain Ca RMSD (sequence-aligned, identity-checked)
                           and a symmetry-aware whole-complex Ca RMSD that tries
                           all sensible chain-assignment permutations for homomers
                           and reports the best. Matched-atom counts included.
  2. INTERFACE metrics   - interface Ca RMSD, fraction of native contacts (fnat),
                           and a DockQ-style score, for each contacting chain pair.
  3. NUCLEIC accuracy    - DNA/RNA backbone RMSD over C1' (and P) atoms, matched by
                           residue-type-verified sequence alignment.
  4. LIGAND accuracy     - after superposing the protein pocket, ligand heavy-atom
                           RMSD (atoms matched by name within the CCD residue), plus
                           whether the ligand landed in the correct pocket.
  5. DETERMINISM / seed  - RMSD spread and confidence spread across the
     spread              seed-N_sample-M models of a target (signal vs sampling noise).
  6. CONFIDENCE vs        - pairs AF3 confidence (ranking_score, pTM, ipTM, mean
     ACCURACY calibration  pLDDT) against the measured RMSD, per target.

Design principles / robustness
==============================
* Correspondence between predicted and reference residues is ALWAYS established
  by GLOBAL SEQUENCE ALIGNMENT (Needleman-Wunsch) and every matched pair is
  identity-verified. Naive residue-index matching is never used, so missing
  residues, residue-numbering offsets, and expression tags are handled correctly.
* Altloc atoms are de-duplicated (first occurrence per atom name wins).
* Modified residues (e.g. MSE, TPO, SEP, PTR) are mapped to their parent
  one-letter code so they participate in alignment and matching.
* Only the first model of a multi-model mmCIF is read.
* Extra chains / peptides / ligands / waters present in the crystal but not in
  the prediction are ignored for protein RMSD and reported as caveats.
* Ligands share author chain letters with protein in many PDB files, so ligand
  instances are grouped by label_asym_id (unique per non-polymer copy), while
  polymer chains are grouped by auth_asym_id.

Dependencies
============
numpy + Python standard library only. No Biopython, no external mmCIF parser.
The mmCIF parser, Needleman-Wunsch aligner, and Kabsch superposition are
generalised from metal_runner/compare_to_crystal.py.

Usage
=====
    # single target
    python verify_campaign.py \
        --target 5exa_query_only \
        --pred-dir /path/to/5exa_ab_query_only \
        --ref-cif  /path/to/5EXA.cif \
        --out-dir  ./verify_out

    # whole campaign from a manifest
    python verify_campaign.py --manifest campaign.json --out-dir ./verify_out

Manifest format (JSON)
======================
    {
      "targets": [
        {
          "name": "5exa_query_only",
          "prediction_dir": "/.../5exa_ab_query_only",
          "reference_cif_path": "/.../5EXA.cif",   # OR "reference_pdb_id": "5EXA"
          "chain_map": {"A": "A", "B": "B"},        # optional pred->ref chain map
          "molecule_types": {"protein": ["A","B"]}, # optional, informational
          "ref_protein_chains": ["A", "B"]          # optional restriction of ref chains
        }
      ]
    }

A bare JSON list of target objects is also accepted.
"""

from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ===========================================================================
# Residue chemistry tables
# ===========================================================================
# Standard amino acids.
AA_STD: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
# Common modified amino acids -> parent one-letter code. Mapping to the parent
# lets a modified residue align with (and match) its unmodified counterpart in
# the other structure (e.g. crystal TPO vs a predicted THR).
AA_MOD: Dict[str, str] = {
    "MSE": "M",  # selenomethionine
    "SEP": "S",  # phosphoserine
    "TPO": "T",  # phosphothreonine
    "PTR": "Y",  # phosphotyrosine
    "CSO": "C", "CSD": "C", "CME": "C", "OCS": "C", "CAS": "C",
    "KCX": "K", "MLY": "K", "M3L": "K", "ALY": "K",
    "HYP": "P", "PCA": "E", "CGU": "E", "LLP": "K",
    "SNN": "N", "AYA": "A", "FME": "M", "NLE": "L",
    "DAL": "A", "DAR": "R", "DSN": "S",
}
# Standard ribonucleotides and deoxyribonucleotides -> one-letter.
RNA_STD: Dict[str, str] = {"A": "A", "C": "C", "G": "G", "U": "U", "I": "I",
                           "N": "N"}
DNA_STD: Dict[str, str] = {"DA": "A", "DC": "C", "DG": "G", "DT": "T",
                           "DU": "U", "DI": "I", "DN": "N"}
WATER = {"HOH", "DOD", "WAT", "H2O"}

# Molecule kinds.
PROTEIN, RNA, DNA, LIGAND, WATER_KIND = "protein", "rna", "dna", "ligand", "water"


def classify_comp(comp_id: str) -> Tuple[str, Optional[str]]:
    """Return (kind, one_letter) for a residue component id.

    one_letter is None for ligands/water. Classification is by component id only
    (so it is independent of the ATOM/HETATM flag, which is essential because
    modified polymer residues such as MSE/TPO are deposited as HETATM).
    """
    c = comp_id.upper()
    if c in AA_STD:
        return PROTEIN, AA_STD[c]
    if c in AA_MOD:
        return PROTEIN, AA_MOD[c]
    if c in DNA_STD:
        return DNA, DNA_STD[c]
    if c in RNA_STD:
        return RNA, RNA_STD[c]
    if c in WATER:
        return WATER_KIND, None
    return LIGAND, None


# ===========================================================================
# Structure model
# ===========================================================================
class Residue:
    """One residue (or ligand/ion/water molecule) and its atoms."""

    __slots__ = ("label_asym", "auth_asym", "comp_id", "auth_seq",
                 "ins_code", "kind", "one_letter", "atoms")

    def __init__(self, label_asym: str, auth_asym: str, comp_id: str,
                 auth_seq: str, ins_code: str) -> None:
        self.label_asym = label_asym
        self.auth_asym = auth_asym
        self.comp_id = comp_id
        self.auth_seq = auth_seq
        self.ins_code = ins_code
        self.kind, self.one_letter = classify_comp(comp_id)
        # atom_name -> (coord ndarray[3], element str)
        self.atoms: Dict[str, Tuple[np.ndarray, str]] = {}

    def ca(self) -> Optional[np.ndarray]:
        a = self.atoms.get("CA")
        return a[0] if a is not None else None

    def atom(self, name: str) -> Optional[np.ndarray]:
        a = self.atoms.get(name)
        return a[0] if a is not None else None

    def heavy_coords(self) -> np.ndarray:
        """(N,3) array of non-hydrogen atom coordinates."""
        out = [c for c, el in self.atoms.values() if el not in ("H", "D")]
        return np.vstack(out) if out else np.zeros((0, 3))


class Structure:
    """Parsed structure: polymer chains, nucleic chains, and ligand instances."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.residues: List[Residue] = []          # all, in file order
        self.protein_chains: Dict[str, List[Residue]] = {}
        self.nucleic_chains: Dict[str, List[Residue]] = {}
        self.ligands: Dict[str, List[Residue]] = {}  # label_asym -> residues
        self.n_water = 0


def _tokenize_cif(s: str) -> List[str]:
    """Tokenise one mmCIF data line, honouring single/double quoting.

    Necessary because atom names containing a prime (e.g. C1', O3', OP1) and any
    value with spaces are quoted in mmCIF. A naive ``str.split()`` would retain
    the surrounding quotes and silently fail to match those atoms. Per the CIF
    rules, a quote opens a value only at a whitespace boundary and closes when
    the same quote is followed by whitespace or end-of-line.
    """
    tokens: List[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t":
            i += 1
            continue
        if c in "'\"":
            q = c
            i += 1
            start = i
            while i < n and not (s[i] == q and (i + 1 >= n or s[i + 1] in " \t")):
                i += 1
            tokens.append(s[start:i])
            i += 1  # skip the closing quote
        else:
            start = i
            while i < n and s[i] not in " \t":
                i += 1
            tokens.append(s[start:i])
    return tokens


def _element_from(name: str, type_symbol: Optional[str]) -> str:
    if type_symbol:
        return type_symbol.upper()
    # Fallback: first alphabetic char of the atom name.
    for ch in name:
        if ch.isalpha():
            return ch.upper()
    return "X"


def parse_mmcif(path: str, model_num: Optional[str] = None) -> Structure:
    """Parse an mmCIF ``atom_site`` loop into a Structure (all atoms).

    Column order is read from the loop header, so this works for both RCSB
    depositions and AlphaFold 3 output. Only the first model is used. Altloc
    atoms are de-duplicated (first occurrence per residue+atom-name wins).
    """
    with open(path, "r") as fh:
        lines = fh.readlines()

    col: Dict[str, int] = {}
    n = len(lines)
    i = 0
    data_start = None
    while i < n:
        if lines[i].strip() == "loop_":
            j = i + 1
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].strip().startswith("_atom_site."):
                col = {}
                k = j
                while k < n and lines[k].strip().startswith("_atom_site."):
                    tag = lines[k].strip().split(".", 1)[1]
                    col[tag] = len(col)
                    k += 1
                data_start = k
                break
        i += 1
    if data_start is None:
        raise ValueError(f"No _atom_site loop found in {path}")

    for req in ("group_PDB", "label_atom_id", "label_comp_id",
                "Cartn_x", "Cartn_y", "Cartn_z"):
        if req not in col:
            raise ValueError(f"{path}: _atom_site missing required column {req}")

    c_atom = col["label_atom_id"]
    c_comp = col["label_comp_id"]
    c_x, c_y, c_z = col["Cartn_x"], col["Cartn_y"], col["Cartn_z"]
    c_auth_asym = col.get("auth_asym_id", col.get("label_asym_id"))
    c_label_asym = col.get("label_asym_id", col.get("auth_asym_id"))
    c_auth_seq = col.get("auth_seq_id", col.get("label_seq_id"))
    c_ins = col.get("pdbx_PDB_ins_code")
    c_alt = col.get("label_alt_id")
    c_model = col.get("pdbx_PDB_model_num")
    c_elem = col.get("type_symbol")

    struct = Structure(path)
    res_by_key: Dict[Tuple[str, str, str, str], Residue] = {}
    chosen_model: Optional[str] = model_num

    ncols = len(col)
    for line in lines[data_start:]:
        s = line.strip()
        if s == "" or s == "#" or s == "loop_" or s.startswith("_"):
            break
        f = _tokenize_cif(s)
        if len(f) < ncols:
            continue
        if c_model is not None:
            m = f[c_model]
            if chosen_model is None:
                chosen_model = m
            if m != chosen_model:
                continue

        comp = f[c_comp]
        label_asym = f[c_label_asym]
        auth_asym = f[c_auth_asym]
        auth_seq = f[c_auth_seq] if c_auth_seq is not None else "."
        ins = f[c_ins] if c_ins is not None else "."
        atom_name = f[c_atom]

        key = (label_asym, auth_seq, ins, comp)
        res = res_by_key.get(key)
        if res is None:
            res = Residue(label_asym, auth_asym, comp, auth_seq, ins)
            res_by_key[key] = res
            struct.residues.append(res)
        if atom_name in res.atoms:
            continue  # altloc / duplicate atom already recorded
        try:
            coord = np.array([float(f[c_x]), float(f[c_y]), float(f[c_z])],
                             dtype=float)
        except ValueError:
            continue
        element = _element_from(atom_name, f[c_elem] if c_elem is not None else None)
        res.atoms[atom_name] = (coord, element)

    # Bucket residues by kind.
    for r in struct.residues:
        if r.kind == PROTEIN:
            struct.protein_chains.setdefault(r.auth_asym, []).append(r)
        elif r.kind in (RNA, DNA):
            struct.nucleic_chains.setdefault(r.auth_asym, []).append(r)
        elif r.kind == LIGAND:
            struct.ligands.setdefault(r.label_asym, []).append(r)
        elif r.kind == WATER_KIND:
            struct.n_water += 1
    return struct


# ===========================================================================
# Sequence alignment (Needleman-Wunsch, identity scoring)
# ===========================================================================
def seq_of(residues: Sequence[Residue]) -> str:
    return "".join(r.one_letter or "X" for r in residues)


def needleman_wunsch(seq1: str, seq2: str, match: float = 1.0,
                     mismatch: float = -1.0, gap: float = -1.0
                     ) -> List[Tuple[Optional[int], Optional[int]]]:
    """Global alignment; returns list of (i, j) index pairs (None == gap)."""
    n, m = len(seq1), len(seq2)
    F = np.zeros((n + 1, m + 1), dtype=float)
    F[:, 0] = np.arange(n + 1) * gap
    F[0, :] = np.arange(m + 1) * gap
    tb = np.zeros((n + 1, m + 1), dtype=np.int8)
    tb[1:, 0] = 1
    tb[0, 1:] = 2
    for i in range(1, n + 1):
        s1 = seq1[i - 1]
        Fim1 = F[i - 1]
        Fi = F[i]
        tbi = tb[i]
        for j in range(1, m + 1):
            diag = Fim1[j - 1] + (match if s1 == seq2[j - 1] else mismatch)
            up = Fim1[j] + gap
            left = Fi[j - 1] + gap
            best, move = diag, 0
            if up > best:
                best, move = up, 1
            if left > best:
                best, move = left, 2
            Fi[j] = best
            tbi[j] = move
    aln: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = tb[i, j]
        if i > 0 and j > 0 and move == 0:
            aln.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and move == 1:
            aln.append((i - 1, None)); i -= 1
        else:
            aln.append((None, j - 1)); j -= 1
    aln.reverse()
    return aln


def aligned_index_pairs(pred_res: Sequence[Residue], ref_res: Sequence[Residue]
                        ) -> List[Tuple[int, int]]:
    """Identity-verified (pred_idx, ref_idx) pairs from a global alignment."""
    aln = needleman_wunsch(seq_of(pred_res), seq_of(ref_res))
    pairs = []
    for i, j in aln:
        if i is None or j is None:
            continue
        if (pred_res[i].one_letter or "X") == (ref_res[j].one_letter or "Y"):
            pairs.append((i, j))
    return pairs


def match_atoms(pred_res: Sequence[Residue], ref_res: Sequence[Residue],
                atom_name: str = "CA"
                ) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Match a named atom (default CA) between two chains via alignment.

    Returns (P, Q, n_matched, n_aligned_but_mismatched).
    """
    aln = needleman_wunsch(seq_of(pred_res), seq_of(ref_res))
    P, Q, mismatched = [], [], 0
    for i, j in aln:
        if i is None or j is None:
            continue
        if (pred_res[i].one_letter or "X") != (ref_res[j].one_letter or "Y"):
            mismatched += 1
            continue
        pa = pred_res[i].atom(atom_name)
        qa = ref_res[j].atom(atom_name)
        if pa is None or qa is None:
            continue
        P.append(pa); Q.append(qa)
    if P:
        return np.vstack(P), np.vstack(Q), len(P), mismatched
    return np.zeros((0, 3)), np.zeros((0, 3)), 0, mismatched


def sequence_identity(pred_res: Sequence[Residue], ref_res: Sequence[Residue]
                      ) -> Tuple[float, int, int]:
    aln = needleman_wunsch(seq_of(pred_res), seq_of(ref_res))
    aligned = identical = 0
    for i, j in aln:
        if i is None or j is None:
            continue
        aligned += 1
        if (pred_res[i].one_letter or "X") == (ref_res[j].one_letter or "Y"):
            identical += 1
    return (100.0 * identical / aligned if aligned else 0.0), identical, aligned


# ===========================================================================
# Kabsch superposition
# ===========================================================================
def kabsch(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (R, p_centroid, q_centroid) superposing P onto Q.

    A point x in P's frame maps to (x - p_centroid) @ R.T + q_centroid.
    """
    p0 = P.mean(axis=0)
    q0 = Q.mean(axis=0)
    H = (P - p0).T @ (Q - q0)
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, p0, q0


def apply_transform(X: np.ndarray, R: np.ndarray, p0: np.ndarray,
                    q0: np.ndarray) -> np.ndarray:
    return (X - p0) @ R.T + q0


def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """RMSD of P superposed onto Q."""
    if P.shape[0] == 0:
        return float("nan")
    R, p0, q0 = kabsch(P, Q)
    diff = apply_transform(P, R, p0, q0) - Q
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


def rmsd_no_fit(P: np.ndarray, Q: np.ndarray) -> float:
    if P.shape[0] == 0:
        return float("nan")
    diff = P - Q
    return float(np.sqrt((diff * diff).sum() / P.shape[0]))


# ===========================================================================
# Protein accuracy
# ===========================================================================
def best_ref_chain_for(pred_res, ref_struct: Structure, ref_chains):
    best = None
    for rc in ref_chains:
        P, Q, nn, mm = match_atoms(pred_res, ref_struct.protein_chains[rc])
        if nn == 0:
            continue
        r = kabsch_rmsd(P, Q)
        if best is None or r < best[1]:
            best = (rc, r, nn, mm)
    return best


def per_chain_report(pred: Structure, ref: Structure,
                     pred_chains, ref_chains) -> List[dict]:
    rows = []
    for pc in pred_chains:
        b = best_ref_chain_for(pred.protein_chains[pc], ref, ref_chains)
        if b is None:
            continue
        rc, r, nn, mm = b
        pct, _, _ = sequence_identity(pred.protein_chains[pc],
                                      ref.protein_chains[rc])
        rows.append({"pred_chain": pc, "ref_chain": rc, "rmsd": r,
                     "n_matched": nn, "n_mismatched": mm, "identity_pct": pct})
    return rows


def complex_rmsd_for_mapping(pred: Structure, ref: Structure,
                             mapping) -> Tuple[float, int]:
    Ps, Qs = [], []
    for pc, rc in mapping:
        P, Q, nn, _ = match_atoms(pred.protein_chains[pc], ref.protein_chains[rc])
        if nn:
            Ps.append(P); Qs.append(Q)
    if not Ps:
        return float("nan"), 0
    Pall, Qall = np.vstack(Ps), np.vstack(Qs)
    return kabsch_rmsd(Pall, Qall), Pall.shape[0]


def greedy_chain_assignment(pred: Structure, ref: Structure,
                            pred_chains, ref_chains) -> list:
    """Optimal-ish 1:1 chain assignment via greedy minimum-RMSD matching.

    Used instead of the O(n!) permutation search when there are too many
    interchangeable chains (see ``symmetry_aware_complex``). Builds a per-chain
    pair Ca RMSD cost matrix (each pair superposed independently) and greedily
    assigns the globally lowest-cost pairs first. For homomers this reliably
    recovers the correct chain correspondence in O(n^2 log n) instead of O(n!).
    """
    pairs = []
    for i, pc in enumerate(pred_chains):
        for j, rc in enumerate(ref_chains):
            P, Q, nn, _ = match_atoms(pred.protein_chains[pc],
                                      ref.protein_chains[rc])
            if nn:
                pairs.append((kabsch_rmsd(P, Q), i, j))
    pairs.sort(key=lambda t: t[0])
    used_p, used_r, mapping = set(), set(), []
    for _c, i, j in pairs:
        if i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        mapping.append((pred_chains[i], ref_chains[j]))
    return mapping


def symmetry_aware_complex(pred: Structure, ref: Structure,
                           pred_chains, ref_chains) -> dict:
    """Try all one-to-one chain assignments; return lowest-RMSD mapping.

    A single global Kabsch superposition is used per mapping over all matched Ca.
    For unequal chain counts each predicted chain is mapped to its best ref chain
    independently (no permutation search).
    """
    if len(pred_chains) != len(ref_chains):
        mapping = []
        for pc in pred_chains:
            b = best_ref_chain_for(pred.protein_chains[pc], ref, ref_chains)
            if b is not None:
                mapping.append((pc, b[0]))
        rmsd, nn = complex_rmsd_for_mapping(pred, ref, mapping)
        return {"best_mapping": mapping, "rmsd": rmsd, "n_matched": nn,
                "all_mappings": [{"mapping": mapping, "rmsd": rmsd, "n_matched": nn}]}
    # Exact permutation search is O(n!): exhaustive and fine for a handful of
    # interchangeable chains, but it explodes for large homomers (8 chains =
    # 40320 superpositions, 9 = 362880). Cap it and fall back to a greedy
    # minimum-RMSD assignment beyond 7 chains.
    if math.factorial(len(ref_chains)) > 5040:
        mapping = greedy_chain_assignment(pred, ref, pred_chains, ref_chains)
        rmsd, nn = complex_rmsd_for_mapping(pred, ref, mapping)
        return {"best_mapping": mapping, "rmsd": rmsd, "n_matched": nn,
                "assignment": "greedy",
                "all_mappings": [{"mapping": mapping, "rmsd": rmsd,
                                  "n_matched": nn}]}
    best, all_maps = None, []
    for perm in itertools.permutations(ref_chains):
        mapping = list(zip(pred_chains, perm))
        rmsd, nn = complex_rmsd_for_mapping(pred, ref, mapping)
        entry = {"mapping": mapping, "rmsd": rmsd, "n_matched": nn}
        all_maps.append(entry)
        if best is None or (not math.isnan(rmsd) and rmsd < best["rmsd"]):
            best = entry
    return {"best_mapping": best["mapping"], "rmsd": best["rmsd"],
            "n_matched": best["n_matched"], "all_mappings": all_maps,
            "assignment": "exact"}


# ===========================================================================
# Interface metrics (fnat, interface RMSD, DockQ)
# ===========================================================================
BACKBONE = ("N", "CA", "C", "O")


def residue_contact_matrix(res_i: Sequence[Residue], res_j: Sequence[Residue],
                           cutoff: float) -> np.ndarray:
    """Boolean [ni, nj]: True where any heavy-atom pair is within ``cutoff``.

    Vectorised: atom-atom distances between the two chains' heavy atoms are
    thresholded, then reduced to residue-pair contacts by boolean matmul with
    residue membership selectors.
    """
    coords_i, owner_i = [], []
    for k, r in enumerate(res_i):
        h = r.heavy_coords()
        if h.shape[0]:
            coords_i.append(h)
            owner_i.append(np.full(h.shape[0], k))
    coords_j, owner_j = [], []
    for k, r in enumerate(res_j):
        h = r.heavy_coords()
        if h.shape[0]:
            coords_j.append(h)
            owner_j.append(np.full(h.shape[0], k))
    if not coords_i or not coords_j:
        return np.zeros((len(res_i), len(res_j)), dtype=bool)
    Ci = np.vstack(coords_i); Cj = np.vstack(coords_j)
    oi = np.concatenate(owner_i); oj = np.concatenate(owner_j)
    # atom-atom squared distances
    d2 = ((Ci[:, None, :] - Cj[None, :, :]) ** 2).sum(-1)
    close = d2 <= cutoff * cutoff                       # [Ai, Aj]
    Si = np.zeros((len(res_i), Ci.shape[0]))            # residue membership
    Si[oi, np.arange(Ci.shape[0])] = 1.0
    Sj = np.zeros((Cj.shape[0], len(res_j)))
    Sj[np.arange(Cj.shape[0]), oj] = 1.0
    contacts = Si @ close.astype(float) @ Sj           # [ni, nj], count of close atom pairs
    return contacts > 0


def _chain_index_maps(pred_chain, ref_chain):
    """pred_idx->ref_idx and ref_idx->pred_idx from identity-verified alignment."""
    pairs = aligned_index_pairs(pred_chain, ref_chain)
    p2r = {i: j for i, j in pairs}
    r2p = {j: i for i, j in pairs}
    return p2r, r2p


def interface_metrics(pred: Structure, ref: Structure, mapping,
                      contact_cutoff: float = 5.0,
                      iface_cutoff: float = 10.0) -> List[dict]:
    """Per contacting chain-pair: fnat, interface Ca RMSD, and DockQ.

    Contacts (fnat): native residue pairs with any heavy-atom distance <=
    ``contact_cutoff`` (default 5 A). Interface residues (iRMS): residues with
    any heavy-atom distance <= ``iface_cutoff`` (default 10 A) across the
    interface in the native structure. DockQ uses the Basu & Wallner (2016)
    formulation with backbone atoms (N, CA, C, O).
    """
    md = dict(mapping)  # pred_chain -> ref_chain
    pred_chs = list(md.keys())
    results = []
    for a, b in itertools.combinations(pred_chs, 2):
        ra, rb = md[a], md[b]
        ref_a, ref_b = ref.protein_chains[ra], ref.protein_chains[rb]
        pred_a, pred_b = pred.protein_chains[a], pred.protein_chains[b]

        # Native contacts (reference).
        nat = residue_contact_matrix(ref_a, ref_b, contact_cutoff)
        n_native = int(nat.sum())
        if n_native == 0:
            continue  # these chains do not form an interface in the crystal

        # Model contacts (prediction).
        mdl = residue_contact_matrix(pred_a, pred_b, contact_cutoff)

        # Map native contacts onto predicted residue indices.
        _, r2p_a = _chain_index_maps(pred_a, ref_a)
        _, r2p_b = _chain_index_maps(pred_b, ref_b)
        recovered = 0
        for i, j in zip(*np.where(nat)):
            pi = r2p_a.get(int(i)); pj = r2p_b.get(int(j))
            if pi is None or pj is None:
                continue
            if mdl[pi, pj]:
                recovered += 1
        fnat = recovered / n_native

        # Interface residues for iRMS (native, 10 A heavy-atom).
        nat_iface = residue_contact_matrix(ref_a, ref_b, iface_cutoff)
        iface_a = set(np.where(nat_iface.any(axis=1))[0].tolist())
        iface_b = set(np.where(nat_iface.any(axis=0))[0].tolist())

        # Backbone atom pairs over interface residues.
        Pbb, Qbb, Pca, Qca = [], [], [], []
        for ref_chain, pred_chain, iface in ((ref_a, pred_a, iface_a),
                                             (ref_b, pred_b, iface_b)):
            _, r2p = _chain_index_maps(pred_chain, ref_chain)
            for ridx in sorted(iface):
                pidx = r2p.get(int(ridx))
                if pidx is None:
                    continue
                for name in BACKBONE:
                    pa = pred_chain[pidx].atom(name)
                    qa = ref_chain[ridx].atom(name)
                    if pa is not None and qa is not None:
                        Pbb.append(pa); Qbb.append(qa)
                pca = pred_chain[pidx].atom("CA")
                qca = ref_chain[ridx].atom("CA")
                if pca is not None and qca is not None:
                    Pca.append(pca); Qca.append(qca)
        irms = kabsch_rmsd(np.vstack(Pbb), np.vstack(Qbb)) if Pbb else float("nan")
        irms_ca = kabsch_rmsd(np.vstack(Pca), np.vstack(Qca)) if Pca else float("nan")

        # Ligand RMSD (DockQ): superpose on receptor (larger chain) backbone,
        # then RMSD of the ligand (smaller chain) backbone.
        len_a = sum(1 for r in ref_a if r.one_letter)
        len_b = sum(1 for r in ref_b if r.one_letter)
        if len_a >= len_b:
            rec_pred, rec_ref, lig_pred, lig_ref = pred_a, ref_a, pred_b, ref_b
        else:
            rec_pred, rec_ref, lig_pred, lig_ref = pred_b, ref_b, pred_a, ref_a
        lrms = dockq_ligand_rmsd(rec_pred, rec_ref, lig_pred, lig_ref)

        dockq = None
        if not any(math.isnan(x) for x in (fnat, irms, lrms)):
            dockq = (fnat + 1.0 / (1.0 + (irms / 1.5) ** 2)
                     + 1.0 / (1.0 + (lrms / 8.5) ** 2)) / 3.0

        results.append({
            "pred_pair": [a, b], "ref_pair": [ra, rb],
            "n_native_contacts": n_native, "n_recovered": recovered,
            "fnat": fnat, "interface_ca_rmsd": irms_ca,
            "interface_backbone_rmsd": irms, "ligand_rmsd_dockq": lrms,
            "dockq": dockq,
            "n_iface_res": len(iface_a) + len(iface_b),
            "contact_cutoff": contact_cutoff, "iface_cutoff": iface_cutoff,
        })
    return results


def dockq_ligand_rmsd(rec_pred, rec_ref, lig_pred, lig_ref) -> float:
    """DockQ L_rms: fit on receptor backbone, RMSD of ligand backbone."""
    Prec, Qrec, _, _ = match_backbone(rec_pred, rec_ref)
    if Prec.shape[0] < 3:
        return float("nan")
    R, p0, q0 = kabsch(Prec, Qrec)
    Plig, Qlig, _, _ = match_backbone(lig_pred, lig_ref)
    if Plig.shape[0] == 0:
        return float("nan")
    moved = apply_transform(Plig, R, p0, q0)
    return rmsd_no_fit(moved, Qlig)


def match_backbone(pred_res, ref_res):
    """Match N, CA, C, O backbone atoms across two chains via alignment."""
    aln = needleman_wunsch(seq_of(pred_res), seq_of(ref_res))
    P, Q = [], []
    for i, j in aln:
        if i is None or j is None:
            continue
        if (pred_res[i].one_letter or "X") != (ref_res[j].one_letter or "Y"):
            continue
        for name in BACKBONE:
            pa = pred_res[i].atom(name)
            qa = ref_res[j].atom(name)
            if pa is not None and qa is not None:
                P.append(pa); Q.append(qa)
    if P:
        return np.vstack(P), np.vstack(Q), len(P), 0
    return np.zeros((0, 3)), np.zeros((0, 3)), 0, 0


# ===========================================================================
# Nucleic acid accuracy
# ===========================================================================
def nucleic_report(pred: Structure, ref: Structure) -> List[dict]:
    """Per predicted nucleic chain: backbone RMSD over C1' and P atoms."""
    rows = []
    ref_chains = list(ref.nucleic_chains.keys())
    for pc, pred_res in pred.nucleic_chains.items():
        best = None
        for rc in ref_chains:
            ref_res = ref.nucleic_chains[rc]
            metrics = {}
            for atom in ("C1'", "P"):
                P, Q, nn, _ = match_atoms(pred_res, ref_res, atom)
                metrics[atom] = (kabsch_rmsd(P, Q), nn) if nn else (float("nan"), 0)
            # Combined backbone (C1' + P) single superposition.
            Ps, Qs = [], []
            for atom in ("C1'", "P"):
                P, Q, nn, _ = match_atoms(pred_res, ref_res, atom)
                if nn:
                    Ps.append(P); Qs.append(Q)
            if Ps:
                bb_rmsd = kabsch_rmsd(np.vstack(Ps), np.vstack(Qs))
                bb_n = sum(x.shape[0] for x in Ps)
            else:
                bb_rmsd, bb_n = float("nan"), 0
            pct, _, _ = sequence_identity(pred_res, ref_res)
            cand = {"pred_chain": pc, "ref_chain": rc,
                    "c1prime_rmsd": metrics["C1'"][0], "c1prime_n": metrics["C1'"][1],
                    "p_rmsd": metrics["P"][0], "p_n": metrics["P"][1],
                    "backbone_rmsd": bb_rmsd, "backbone_n": bb_n,
                    "identity_pct": pct}
            if best is None or (not math.isnan(bb_rmsd) and
                                (math.isnan(best["backbone_rmsd"]) or
                                 bb_rmsd < best["backbone_rmsd"])):
                best = cand
        if best is not None:
            rows.append(best)
    return rows


# ===========================================================================
# Ligand accuracy
# ===========================================================================
def _pocket_ref_residues(ref: Structure, lig_res: Residue,
                         pocket_cutoff: float):
    """Reference protein residues with any heavy atom within cutoff of ligand."""
    lig_h = lig_res.heavy_coords()
    pocket = []  # (chain, ref_idx, residue)
    if lig_h.shape[0] == 0:
        return pocket
    for ch, residues in ref.protein_chains.items():
        for idx, r in enumerate(residues):
            h = r.heavy_coords()
            if h.shape[0] == 0:
                continue
            d2 = ((h[:, None, :] - lig_h[None, :, :]) ** 2).sum(-1)
            if d2.min() <= pocket_cutoff * pocket_cutoff:
                pocket.append((ch, idx, r))
    return pocket


def ligand_report(pred: Structure, ref: Structure, chain_map: Dict[str, str],
                  pocket_cutoff: float = 6.0) -> List[dict]:
    """Ligand heavy-atom RMSD after superposing the (matched) protein pocket.

    For each reference ligand, the surrounding protein pocket (heavy atoms within
    ``pocket_cutoff``) is identified, mapped to the predicted structure by
    sequence alignment, and used for a Ca superposition. A predicted ligand of
    the same CCD component is then transformed and its heavy atoms matched by
    name to the reference ligand.
    """
    rows = []
    # Predicted ligands available, grouped by comp id.
    pred_ligs: Dict[str, List[Residue]] = {}
    for residues in pred.ligands.values():
        for r in residues:
            pred_ligs.setdefault(r.comp_id, []).append(r)
    ref2pred = chain_map  # ref chain -> pred chain (inverse of pred->ref)

    for lasym, residues in ref.ligands.items():
        for lig_ref in residues:
            comp = lig_ref.comp_id
            entry = {"comp_id": comp, "ref_label_asym": lasym,
                     "ref_auth_asym": lig_ref.auth_asym,
                     "ref_auth_seq": lig_ref.auth_seq,
                     "predicted": False, "rmsd": None,
                     "n_atoms_matched": 0, "correct_pocket": None,
                     "centroid_dist": None, "note": ""}
            pocket = _pocket_ref_residues(ref, lig_ref, pocket_cutoff)
            if comp not in pred_ligs:
                entry["note"] = "ligand present in crystal but not in prediction"
                rows.append(entry)
                continue
            # Build pocket Ca correspondence.
            Pca, Qca = [], []
            for ch, ridx, rres in pocket:
                pch = ref2pred.get(ch)
                if pch is None or pch not in pred.protein_chains:
                    continue
                _, r2p = _chain_index_maps(pred.protein_chains[pch],
                                           ref.protein_chains[ch])
                pidx = r2p.get(ridx)
                if pidx is None:
                    continue
                pca = pred.protein_chains[pch][pidx].atom("CA")
                qca = rres.atom("CA")
                if pca is not None and qca is not None:
                    Pca.append(pca); Qca.append(qca)
            if len(Pca) < 3:
                entry["note"] = ("insufficient matched pocket residues for "
                                 "superposition")
                rows.append(entry)
                continue
            R, p0, q0 = kabsch(np.vstack(Pca), np.vstack(Qca))
            # Choose the predicted ligand copy that lands closest to this ref copy.
            ref_atoms = {n: c for n, (c, _e) in lig_ref.atoms.items()}
            ref_centroid = lig_ref.heavy_coords().mean(axis=0)
            best = None
            for lig_pred in pred_ligs[comp]:
                names = [n for n in lig_pred.atoms if n in ref_atoms]
                if not names:
                    continue
                Pl = np.vstack([lig_pred.atoms[n][0] for n in names])
                Ql = np.vstack([ref_atoms[n] for n in names])
                moved = apply_transform(Pl, R, p0, q0)
                rmsd = rmsd_no_fit(moved, Ql)
                cdist = float(np.linalg.norm(
                    apply_transform(lig_pred.heavy_coords(), R, p0, q0).mean(0)
                    - ref_centroid))
                if best is None or rmsd < best[0]:
                    best = (rmsd, len(names), cdist)
            if best is None:
                entry["note"] = ("predicted ligand of same component has no "
                                 "matching atom names")
                rows.append(entry)
                continue
            rmsd, natoms, cdist = best
            entry.update({"predicted": True, "rmsd": rmsd,
                          "n_atoms_matched": natoms, "centroid_dist": cdist,
                          "correct_pocket": bool(cdist <= 8.0),
                          "note": ("pocket superposed on %d Ca" % len(Pca))})
            rows.append(entry)
    return rows


# ===========================================================================
# AF3 output discovery + confidence parsing
# ===========================================================================
def find_top_model(pred_dir: str) -> Optional[str]:
    """The best-ranked top-level model (``*_model.cif`` directly in pred_dir)."""
    cands = [p for p in glob.glob(os.path.join(pred_dir, "*_model.cif"))]
    return sorted(cands)[0] if cands else None


def find_sample_models(pred_dir: str) -> List[dict]:
    """All seed-N_sample-M models with parsed (seed, sample) tags."""
    out = []
    for cif in sorted(glob.glob(os.path.join(pred_dir, "seed-*_sample-*",
                                             "*_model.cif"))):
        d = os.path.basename(os.path.dirname(cif))
        seed = sample = None
        for tok in d.split("_"):
            if tok.startswith("seed-"):
                seed = tok[len("seed-"):]
            elif tok.startswith("sample-"):
                sample = tok[len("sample-"):]
        out.append({"seed": seed, "sample": sample, "model": cif,
                    "dir": os.path.dirname(cif)})
    return out


def _first(paths):
    return sorted(paths)[0] if paths else None


def load_summary_confidence(directory: str) -> dict:
    p = _first(glob.glob(os.path.join(directory, "*_summary_confidences.json")))
    if not p:
        return {}
    try:
        with open(p) as fh:
            d = json.load(fh)
    except Exception:
        return {}
    return {k: d.get(k) for k in ("ranking_score", "ptm", "iptm",
                                  "has_clash", "fraction_disordered",
                                  "chain_ptm", "chain_iptm")}


def load_mean_plddt(directory: str) -> Optional[float]:
    p = _first(glob.glob(os.path.join(directory, "*_confidences.json")))
    if not p:
        return None
    try:
        with open(p) as fh:
            d = json.load(fh)
        pl = d.get("atom_plddts")
        return float(np.mean(pl)) if pl else None
    except Exception:
        return None


def load_ranking_csv(pred_dir: str) -> Dict[Tuple[str, str], float]:
    p = _first(glob.glob(os.path.join(pred_dir, "*_ranking_scores.csv")))
    out: Dict[Tuple[str, str], float] = {}
    if not p:
        return out
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[(row["seed"], row["sample"])] = float(row["ranking_score"])
            except (KeyError, ValueError):
                continue
    return out


# ===========================================================================
# Reference resolution (optional RCSB download via stdlib urllib)
# ===========================================================================
def resolve_reference(tgt: dict, cache_dir: str, allow_download: bool) -> str:
    if tgt.get("reference_cif_path"):
        path = os.path.expanduser(tgt["reference_cif_path"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"reference_cif_path not found: {path}")
        return path
    pdb_id = tgt.get("reference_pdb_id")
    if not pdb_id:
        raise ValueError("target needs reference_cif_path or reference_pdb_id")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, f"{pdb_id.upper()}.cif")
    if os.path.exists(dst):
        return dst
    if not allow_download:
        raise FileNotFoundError(
            f"{pdb_id}: not cached at {dst} and downloads are disabled "
            f"(pass --allow-download or provide reference_cif_path)")
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif"
    urllib.request.urlretrieve(url, dst)
    return dst


# ===========================================================================
# Chain selection
# ===========================================================================
def select_ref_protein_chains(pred: Structure, ref: Structure,
                              pred_chains, explicit=None,
                              identity_threshold: float = 90.0):
    """Pick reference protein chains that correspond to the predicted chains.

    If ``explicit`` is provided it is used verbatim. Otherwise, a reference chain
    qualifies if it aligns to some predicted chain with >= ``identity_threshold``
    identity AND comparable length; this excludes co-crystallised peptides,
    fusion partners, and unrelated chains from the whole-complex RMSD.
    """
    if explicit:
        return [c for c in explicit if c in ref.protein_chains]
    qualifying = []
    for rc, ref_res in ref.protein_chains.items():
        for pc in pred_chains:
            pred_res = pred.protein_chains[pc]
            pct, _, _ = sequence_identity(pred_res, ref_res)
            lr = len(ref_res) / max(1, len(pred_res))
            if pct >= identity_threshold and 0.5 <= lr <= 2.0:
                qualifying.append(rc)
                break
    return sorted(qualifying)


# ===========================================================================
# Per-target verification
# ===========================================================================
def verify_target(tgt: dict, cache_dir: str, allow_download: bool) -> dict:
    name = tgt.get("name") or os.path.basename(tgt["prediction_dir"].rstrip("/"))
    pred_dir = os.path.expanduser(tgt["prediction_dir"])
    ref_path = resolve_reference(tgt, cache_dir, allow_download)

    top_model = find_top_model(pred_dir)
    if top_model is None:
        raise FileNotFoundError(f"no *_model.cif in {pred_dir}")

    pred = parse_mmcif(top_model)
    ref = parse_mmcif(ref_path)

    caveats: List[str] = []
    result: dict = {
        "target": name, "prediction_dir": pred_dir,
        "top_model": top_model, "reference": ref_path,
        "molecule_types_declared": tgt.get("molecule_types"),
        "caveats": caveats,
    }

    # --- chain selection -------------------------------------------------
    pred_chains = sorted(pred.protein_chains.keys())
    chain_map_pred_ref = tgt.get("chain_map")  # pred -> ref (optional)
    ref_prot_chains = select_ref_protein_chains(
        pred, ref, pred_chains, tgt.get("ref_protein_chains"))
    result["pred_protein_chains"] = pred_chains
    result["ref_protein_chains_used"] = ref_prot_chains

    # Report crystal-only content as caveats.
    ignored_prot = [c for c in ref.protein_chains if c not in ref_prot_chains]
    if ignored_prot:
        caveats.append(
            "reference protein chains ignored for protein RMSD (low identity / "
            "different entity): " + ", ".join(sorted(ignored_prot)))
    if ref.ligands:
        caveats.append("reference contains %d ligand instance(s): %s"
                       % (len(ref.ligands),
                          ", ".join(sorted({r.comp_id for rs in ref.ligands.values()
                                            for r in rs}))))
    if ref.n_water:
        caveats.append("reference waters (%d) ignored" % ref.n_water)

    # --- 1. protein accuracy --------------------------------------------
    if pred_chains and ref_prot_chains:
        result["per_chain"] = per_chain_report(pred, ref, pred_chains,
                                               ref_prot_chains)
        if chain_map_pred_ref:
            mapping = [(pc, rc) for pc, rc in chain_map_pred_ref.items()
                       if pc in pred.protein_chains and rc in ref.protein_chains]
            rmsd, nn = complex_rmsd_for_mapping(pred, ref, mapping)
            result["complex"] = {"best_mapping": mapping, "rmsd": rmsd,
                                 "n_matched": nn,
                                 "all_mappings": [{"mapping": mapping,
                                                   "rmsd": rmsd, "n_matched": nn}]}
        else:
            result["complex"] = symmetry_aware_complex(
                pred, ref, pred_chains, ref_prot_chains)
        best_map = result["complex"]["best_mapping"]

        # --- 2. interface metrics ---------------------------------------
        if len(best_map) >= 2:
            result["interfaces"] = interface_metrics(pred, ref, best_map)
        else:
            result["interfaces"] = []

        # ref chain -> pred chain map for ligand pocket work.
        ref2pred = {rc: pc for pc, rc in best_map}
    else:
        result["per_chain"] = []
        result["complex"] = {"best_mapping": [], "rmsd": float("nan"),
                             "n_matched": 0, "all_mappings": []}
        result["interfaces"] = []
        ref2pred = {}
        caveats.append("no corresponding protein chains found")

    # --- 3. nucleic accuracy --------------------------------------------
    result["nucleic"] = nucleic_report(pred, ref) if pred.nucleic_chains else []
    if ref.nucleic_chains and not pred.nucleic_chains:
        caveats.append("reference has nucleic chains but prediction does not")

    # --- 4. ligand accuracy ---------------------------------------------
    result["ligands"] = (ligand_report(pred, ref, ref2pred)
                         if ref.ligands else [])

    # --- 5. determinism / seed spread -----------------------------------
    result["determinism"] = determinism_report(pred_dir, ref, pred_chains,
                                               ref_prot_chains,
                                               chain_map_pred_ref)

    # --- 6. confidence of the ranked model ------------------------------
    conf = load_summary_confidence(pred_dir)
    conf["mean_plddt"] = load_mean_plddt(pred_dir)
    result["confidence"] = conf

    # --- 6b. confidence-vs-accuracy pairing -----------------------------
    result["calibration"] = {
        "ranking_score": conf.get("ranking_score"),
        "ptm": conf.get("ptm"), "iptm": conf.get("iptm"),
        "mean_plddt": conf.get("mean_plddt"),
        "complex_ca_rmsd": result["complex"]["rmsd"],
    }
    return result


def determinism_report(pred_dir: str, ref: Structure, pred_chains,
                       ref_prot_chains, chain_map_pred_ref) -> dict:
    samples = find_sample_models(pred_dir)
    ranking = load_ranking_csv(pred_dir)
    rows = []
    parsed: List[Optional[Structure]] = []  # one parse per sample, reused below
    rmsds, ranks, ptms, iptms, plddts = [], [], [], [], []
    for s in samples:
        try:
            ps = parse_mmcif(s["model"])
        except Exception as e:  # pragma: no cover - defensive
            parsed.append(None)
            rows.append({**s, "error": str(e)})
            continue
        parsed.append(ps)
        if ref_prot_chains and pred_chains:
            if chain_map_pred_ref:
                mapping = [(pc, rc) for pc, rc in chain_map_pred_ref.items()
                           if pc in ps.protein_chains and rc in ref.protein_chains]
                rmsd, nn = complex_rmsd_for_mapping(ps, ref, mapping)
            else:
                comp = symmetry_aware_complex(ps, ref, sorted(ps.protein_chains),
                                              ref_prot_chains)
                rmsd, nn = comp["rmsd"], comp["n_matched"]
        else:
            rmsd, nn = float("nan"), 0
        conf = load_summary_confidence(s["dir"])
        mp = load_mean_plddt(s["dir"])
        rank = ranking.get((s["seed"], s["sample"]), conf.get("ranking_score"))
        row = {"seed": s["seed"], "sample": s["sample"],
               "complex_ca_rmsd": rmsd, "n_matched": nn,
               "ranking_score": rank, "ptm": conf.get("ptm"),
               "iptm": conf.get("iptm"), "mean_plddt": mp}
        rows.append(row)
        if not math.isnan(rmsd):
            rmsds.append(rmsd)
        for lst, v in ((ranks, rank), (ptms, conf.get("ptm")),
                       (iptms, conf.get("iptm")), (plddts, mp)):
            if v is not None:
                lst.append(v)

    def spread(vals):
        if not vals:
            return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
        a = np.array(vals, dtype=float)
        return {"n": len(vals), "mean": float(a.mean()), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    # Pairwise model-to-model RMSD (sampling noise, reference-free); reuses the
    # structures already parsed above.
    pair_rmsds = []
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            if parsed[i] is None or parsed[j] is None:
                continue
            chs = sorted(set(parsed[i].protein_chains) & set(parsed[j].protein_chains))
            if not chs:
                continue
            comp = symmetry_aware_complex(parsed[i], parsed[j], chs, chs)
            if not math.isnan(comp["rmsd"]):
                pair_rmsds.append(comp["rmsd"])

    return {"n_samples": len(samples), "per_sample": rows,
            "rmsd_to_ref_spread": spread(rmsds),
            "pairwise_model_rmsd_spread": spread(pair_rmsds),
            "ranking_score_spread": spread(ranks),
            "ptm_spread": spread(ptms), "iptm_spread": spread(iptms),
            "mean_plddt_spread": spread(plddts)}


# ===========================================================================
# Reporting
# ===========================================================================
def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def print_report(res: dict) -> None:
    L = print
    L("=" * 74)
    L(f"TARGET: {res['target']}")
    L(f"  model    : {res['top_model']}")
    L(f"  reference: {res['reference']}")
    L("=" * 74)

    comp = res["complex"]
    L("[1] PROTEIN ACCURACY")
    L(f"  pred protein chains : {', '.join(res['pred_protein_chains']) or '(none)'}")
    L(f"  ref  protein chains : {', '.join(res['ref_protein_chains_used']) or '(none)'}")
    for r in res["per_chain"]:
        L(f"    chain {r['pred_chain']} -> ref {r['ref_chain']}: "
          f"Ca RMSD = {_fmt(r['rmsd'])} A over {r['n_matched']} atoms "
          f"(id {_fmt(r['identity_pct'],1)}%, {r['n_mismatched']} mismatches skipped)")
    if comp["all_mappings"]:
        L("  whole-complex (symmetry-aware, single global superposition):")
        for e in comp["all_mappings"]:
            mp = ",".join(f"{a}->{b}" for a, b in e["mapping"])
            tag = "  <-- best" if e["mapping"] == comp["best_mapping"] else ""
            L(f"    [{mp}] Ca RMSD = {_fmt(e['rmsd'])} A over {e['n_matched']} atoms{tag}")
    best_mp = ",".join(f"{a}->{b}" for a, b in comp["best_mapping"])
    L(f"  BEST complex Ca RMSD = {_fmt(comp['rmsd'])} A "
      f"({best_mp or 'n/a'}, {comp['n_matched']} atoms)")

    L("[2] INTERFACE METRICS")
    if res["interfaces"]:
        for it in res["interfaces"]:
            L(f"    {it['pred_pair'][0]}-{it['pred_pair'][1]} "
              f"(ref {it['ref_pair'][0]}-{it['ref_pair'][1]}): "
              f"fnat = {_fmt(it['fnat'])} ({it['n_recovered']}/{it['n_native_contacts']}), "
              f"iRMS(Ca) = {_fmt(it['interface_ca_rmsd'])} A, "
              f"iRMS(bb) = {_fmt(it['interface_backbone_rmsd'])} A, "
              f"L_rms = {_fmt(it['ligand_rmsd_dockq'])} A, "
              f"DockQ = {_fmt(it['dockq'])}")
    else:
        L("    (no protein-protein interface)")

    L("[3] NUCLEIC ACID ACCURACY")
    if res["nucleic"]:
        for r in res["nucleic"]:
            L(f"    chain {r['pred_chain']} -> ref {r['ref_chain']}: "
              f"backbone RMSD = {_fmt(r['backbone_rmsd'])} A over {r['backbone_n']} atoms "
              f"(C1' {_fmt(r['c1prime_rmsd'])}/{r['c1prime_n']}, "
              f"P {_fmt(r['p_rmsd'])}/{r['p_n']}, id {_fmt(r['identity_pct'],1)}%)")
    else:
        L("    (no nucleic acid chains in prediction)")

    L("[4] LIGAND ACCURACY")
    if res["ligands"]:
        for r in res["ligands"]:
            if r["predicted"]:
                L(f"    {r['comp_id']} (ref {r['ref_auth_asym']}{r['ref_auth_seq']}): "
                  f"heavy-atom RMSD = {_fmt(r['rmsd'])} A over {r['n_atoms_matched']} atoms, "
                  f"centroid dist = {_fmt(r['centroid_dist'])} A, "
                  f"correct pocket = {r['correct_pocket']}")
            else:
                L(f"    {r['comp_id']} (ref {r['ref_auth_asym']}{r['ref_auth_seq']}): "
                  f"{r['note']}")
    else:
        L("    (no ligands in reference)")

    d = res["determinism"]
    L(f"[5] DETERMINISM / SEED SPREAD ({d['n_samples']} samples)")
    for r in d["per_sample"]:
        L(f"    seed {r.get('seed')} sample {r.get('sample')}: "
          f"complex Ca RMSD = {_fmt(r.get('complex_ca_rmsd'))} A, "
          f"rank = {_fmt(r.get('ranking_score'))}, "
          f"pTM = {_fmt(r.get('ptm'))}, ipTM = {_fmt(r.get('iptm'))}, "
          f"pLDDT = {_fmt(r.get('mean_plddt'),2)}")
    sr = d["rmsd_to_ref_spread"]; pr = d["pairwise_model_rmsd_spread"]
    L(f"    RMSD-to-ref spread : mean {_fmt(sr['mean'])} std {_fmt(sr['std'])} "
      f"[{_fmt(sr['min'])}, {_fmt(sr['max'])}] A")
    L(f"    model-model spread : mean {_fmt(pr['mean'])} std {_fmt(pr['std'])} "
      f"[{_fmt(pr['min'])}, {_fmt(pr['max'])}] A")
    rs = d["ranking_score_spread"]
    L(f"    ranking_score spread: mean {_fmt(rs['mean'])} std {_fmt(rs['std'])} "
      f"[{_fmt(rs['min'])}, {_fmt(rs['max'])}]")

    c = res["confidence"]
    L("[6] CONFIDENCE vs ACCURACY (ranked model)")
    L(f"    ranking_score = {_fmt(c.get('ranking_score'))}, "
      f"pTM = {_fmt(c.get('ptm'))}, ipTM = {_fmt(c.get('iptm'))}, "
      f"mean pLDDT = {_fmt(c.get('mean_plddt'),2)}, "
      f"has_clash = {_fmt(c.get('has_clash'))}")
    L(f"    measured complex Ca RMSD = {_fmt(comp['rmsd'])} A")

    if res["caveats"]:
        L("CAVEATS")
        for cav in res["caveats"]:
            L(f"    - {cav}")
    L("")


# ===========================================================================
# CSV summary
# ===========================================================================
CSV_FIELDS = [
    "target", "top_model", "reference",
    "n_pred_protein_chains", "complex_ca_rmsd", "complex_n_atoms",
    "best_mapping", "mean_per_chain_rmsd",
    "interface_pair", "fnat", "interface_ca_rmsd", "dockq",
    "n_nucleic_chains", "nucleic_backbone_rmsd",
    "n_ref_ligands", "n_pred_ligands_matched", "best_ligand_rmsd",
    "ranking_score", "ptm", "iptm", "mean_plddt",
    "seed_rmsd_mean", "seed_rmsd_std", "model_model_rmsd_std",
]


def summary_row(res: dict) -> dict:
    comp = res["complex"]
    per_chain = res["per_chain"]
    mean_pc = (float(np.mean([r["rmsd"] for r in per_chain
                              if not math.isnan(r["rmsd"])]))
               if per_chain else float("nan"))
    ifaces = res["interfaces"]
    iface0 = ifaces[0] if ifaces else {}
    nucleic = res["nucleic"]
    ligs = [r for r in res["ligands"] if r.get("predicted")]
    best_lig = (min((r["rmsd"] for r in ligs if r["rmsd"] is not None),
                    default=None))
    d = res["determinism"]
    c = res["confidence"]
    return {
        "target": res["target"], "top_model": res["top_model"],
        "reference": res["reference"],
        "n_pred_protein_chains": len(res["pred_protein_chains"]),
        "complex_ca_rmsd": _fmt(comp["rmsd"]),
        "complex_n_atoms": comp["n_matched"],
        "best_mapping": ";".join(f"{a}->{b}" for a, b in comp["best_mapping"]),
        "mean_per_chain_rmsd": _fmt(mean_pc),
        "interface_pair": ("%s-%s" % tuple(iface0["pred_pair"])
                           if iface0 else ""),
        "fnat": _fmt(iface0.get("fnat")) if iface0 else "",
        "interface_ca_rmsd": _fmt(iface0.get("interface_ca_rmsd")) if iface0 else "",
        "dockq": _fmt(iface0.get("dockq")) if iface0 else "",
        "n_nucleic_chains": len(nucleic),
        "nucleic_backbone_rmsd": (_fmt(nucleic[0]["backbone_rmsd"])
                                  if nucleic else ""),
        "n_ref_ligands": len(res["ligands"]),
        "n_pred_ligands_matched": len(ligs),
        "best_ligand_rmsd": _fmt(best_lig) if best_lig is not None else "",
        "ranking_score": _fmt(c.get("ranking_score")),
        "ptm": _fmt(c.get("ptm")), "iptm": _fmt(c.get("iptm")),
        "mean_plddt": _fmt(c.get("mean_plddt"), 2),
        "seed_rmsd_mean": _fmt(d["rmsd_to_ref_spread"]["mean"]),
        "seed_rmsd_std": _fmt(d["rmsd_to_ref_spread"]["std"]),
        "model_model_rmsd_std": _fmt(d["pairwise_model_rmsd_spread"]["std"]),
    }


# ===========================================================================
# CLI
# ===========================================================================
def load_manifest(path: str) -> List[dict]:
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "targets" in data:
        return data["targets"]
    if isinstance(data, list):
        return data
    raise ValueError("manifest must be a list or an object with a 'targets' key")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Post-campaign verification of AF3 predictions vs references.")
    ap.add_argument("--manifest", help="JSON manifest of targets")
    ap.add_argument("--target", help="single-target name")
    ap.add_argument("--pred-dir", help="single-target prediction directory")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--ref-cif", help="single-target reference mmCIF path")
    g.add_argument("--ref-pdb-id", help="single-target reference PDB id")
    ap.add_argument("--chain-map", help="single-target pred:ref map, e.g. A:A,B:B")
    ap.add_argument("--ref-protein-chains",
                    help="single-target ref chain restriction, e.g. A,B")
    ap.add_argument("--out-dir", default="verify_out",
                    help="directory for per-target JSON + summary CSV")
    ap.add_argument("--ref-cache-dir", default=None,
                    help="cache dir for downloaded references "
                    "(default: <out-dir>/ref_cache)")
    ap.add_argument("--allow-download", action="store_true",
                    help="permit downloading references from RCSB by PDB id")
    args = ap.parse_args(argv)

    if args.manifest:
        targets = load_manifest(args.manifest)
    elif args.pred_dir and (args.ref_cif or args.ref_pdb_id):
        t = {"name": args.target, "prediction_dir": args.pred_dir}
        if args.ref_cif:
            t["reference_cif_path"] = args.ref_cif
        else:
            t["reference_pdb_id"] = args.ref_pdb_id
        if args.chain_map:
            t["chain_map"] = {p.split(":")[0]: p.split(":")[1]
                              for p in args.chain_map.split(",")}
        if args.ref_protein_chains:
            t["ref_protein_chains"] = [c.strip() for c in
                                       args.ref_protein_chains.split(",")]
        targets = [t]
    else:
        ap.error("provide --manifest, or --pred-dir with --ref-cif/--ref-pdb-id")

    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.ref_cache_dir or os.path.join(args.out_dir, "ref_cache")

    all_results, rows = [], []
    for tgt in targets:
        try:
            res = verify_target(tgt, cache_dir, args.allow_download)
        except Exception as e:
            print(f"!! target {tgt.get('name', tgt.get('prediction_dir'))} "
                  f"failed: {e}", file=sys.stderr)
            continue
        all_results.append(res)
        print_report(res)
        name = res["target"]
        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as fh:
            json.dump(res, fh, indent=2, default=_json_default)
        rows.append(summary_row(res))

    if rows:
        csv_path = os.path.join(args.out_dir, "summary.csv")
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote summary CSV: {csv_path}")
        print(f"Wrote {len(all_results)} per-target JSON file(s) to {args.out_dir}")
    return 0


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


if __name__ == "__main__":
    sys.exit(main())
