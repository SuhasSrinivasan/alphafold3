# SPDX-License-Identifier: Apache-2.0
"""Verifies database-free, query-only MSA featurisation for a protein dimer."""

from __future__ import annotations

import argparse

from _common import DEFAULT_SEQUENCE_A
from _common import DEFAULT_SEQUENCE_B
from _common import featurise_msa_free_dimer
from _common import positive_int


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sequence-a', default=DEFAULT_SEQUENCE_A)
  parser.add_argument('--sequence-b', default=DEFAULT_SEQUENCE_B)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--bucket', type=positive_int, default=18)
  args = parser.parse_args()
  featurise_msa_free_dimer(
      name='mps_msa_free_dimer_gate',
      seed=args.seed,
      sequence_a=args.sequence_a,
      sequence_b=args.sequence_b,
      bucket=args.bucket,
  )


if __name__ == '__main__':
  main()
