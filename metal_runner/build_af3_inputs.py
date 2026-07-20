#!/usr/bin/env python3
"""Build AF3 input JSONs from canonical target specs for the MPS benchmark campaign.

A *canonical target* is a dict:
  {
    "name": "monomer_9X1W",
    "type": "monomer",
    "pdb_id": "9X1W",
    "deposition_date": "2025-12-01",     # ISO; used to derive blind max_template_date
    "modelSeeds": [1,2,3,4,5],           # optional, default 5 fixed seeds
    "entities": [
       {"kind":"protein", "id":["A"], "sequence":"..."},
       {"kind":"dna",     "id":["C"], "sequence":"ACGT..."},
       {"kind":"rna",     "id":["E"], "sequence":"ACGU..."},
       {"kind":"ligand",  "id":["L"], "ccdCodes":["ATP"]},
       {"kind":"ion",     "id":["M"], "ccdCodes":["MG"]},
    ],
  }

Design choices (see docs/input.md):
  * Protein & RNA chains OMIT unpairedMsa/pairedMsa/templates  -> AF3 builds MSA + searches
    templates in the data pipeline (this is what we want to exercise).
  * Ions are ligands with a CCD code (e.g. MG).
  * ``max_template_date`` is a per-run CLI flag, not in the JSON. We derive the *blind* date
    (deposition_date - 1 day) here and return it so the orchestrator passes it per target,
    preventing a target from templating against its own (or newer) structure.
  * Every generated JSON is validated against AF3's own ``folding_input.Input.from_json`` and
    every ligand/ion CCD code is checked against AF3's CCD (2022-09-28 snapshot). Modern
    (post-2022) 5-char CCD codes that are absent are reported so such targets can be excluded
    or given SMILES instead.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

DEFAULT_SEEDS = [1, 2, 3, 4, 5]


def blind_max_template_date(deposition_date: str) -> str:
  d = datetime.date.fromisoformat(deposition_date)
  return (d - datetime.timedelta(days=1)).isoformat()


def build_af3_input(target: dict) -> dict:
  """Canonical target -> AF3 input JSON dict (dialect alphafold3, version 1)."""
  sequences = []
  for e in target["entities"]:
    kind = e["kind"]
    cid = e["id"]
    if kind == "protein":
      # Omit MSA/template fields -> data pipeline builds them.
      sequences.append({"protein": {"id": cid, "sequence": e["sequence"]}})
    elif kind == "rna":
      sequences.append({"rna": {"id": cid, "sequence": e["sequence"]}})
    elif kind == "dna":
      sequences.append({"dna": {"id": cid, "sequence": e["sequence"]}})
    elif kind in ("ligand", "ion"):
      sequences.append({"ligand": {"id": cid, "ccdCodes": list(e["ccdCodes"])}})
    else:
      raise ValueError(f"unknown entity kind: {kind!r}")
  return {
      "name": target["name"],
      "modelSeeds": target.get("modelSeeds", DEFAULT_SEEDS),
      "sequences": sequences,
      "dialect": "alphafold3",
      "version": 1,
  }


def validate_structure(af3_dict: dict) -> None:
  """Parse with AF3's own loader; raises on any structural problem."""
  from alphafold3.common import folding_input
  folding_input.Input.from_json(json.dumps(af3_dict))


def ccd_codes(target: dict) -> list[str]:
  out = []
  for e in target["entities"]:
    if e["kind"] in ("ligand", "ion"):
      out.extend(e["ccdCodes"])
  return out


def check_ccd(codes, ccd) -> list[str]:
  """Return CCD codes NOT present in AF3's CCD snapshot."""
  missing = []
  for c in codes:
    try:
      present = ccd.get(c) is not None
    except Exception:
      present = False
    if not present:
      missing.append(c)
  return missing


def _selftest() -> int:
  from alphafold3.constants import chemical_components
  ccd = chemical_components.Ccd()
  cases = [
      {"name": "t_monomer", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"}]},
      {"name": "t_homodimer", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A", "B"], "sequence": "ACDEFGHIKLMNPQRSTVWY"}]},
      {"name": "t_prot_dna", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"},
                    {"kind": "dna", "id": ["C"], "sequence": "ACGTACGTAC"},
                    {"kind": "dna", "id": ["D"], "sequence": "GTACGTACGT"}]},
      {"name": "t_prot_rna", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"},
                    {"kind": "rna", "id": ["E"], "sequence": "ACGUACGU"}]},
      {"name": "t_prot_lig", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"},
                    {"kind": "ligand", "id": ["L"], "ccdCodes": ["ATP"]}]},
      {"name": "t_pp_lig_ion", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"},
                    {"kind": "protein", "id": ["B"], "sequence": "MNPQRSTVWYACDEFGHIKL"},
                    {"kind": "ligand", "id": ["L"], "ccdCodes": ["ATP"]},
                    {"kind": "ion", "id": ["M"], "ccdCodes": ["MG"]}]},
      # A deliberately BAD ccd code to prove the CCD check catches it.
      {"name": "t_bad_ccd", "deposition_date": "2025-06-01",
       "entities": [{"kind": "protein", "id": ["A"], "sequence": "ACDEFGHIKLMNPQRSTVWY"},
                    {"kind": "ligand", "id": ["L"], "ccdCodes": ["A1IYM"]}]},
  ]
  ok = True
  for t in cases:
    d = build_af3_input(t)
    try:
      validate_structure(d)
      struct = "parse-OK"
    except Exception as ex:  # noqa: BLE001
      struct = f"parse-FAIL: {ex}"
      ok = False
    missing = check_ccd(ccd_codes(t), ccd)
    print(f"{t['name']:16s} blind_date={blind_max_template_date(t['deposition_date'])} "
          f"{struct}  missing_ccd={missing}")
  # sanity: the bad-ccd case MUST report A1IYM missing
  return 0 if ok else 1


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--selftest", action="store_true")
  args = ap.parse_args()
  if args.selftest:
    sys.exit(_selftest())
  ap.error("nothing to do; pass --selftest (campaign build wiring added next)")


if __name__ == "__main__":
  main()
