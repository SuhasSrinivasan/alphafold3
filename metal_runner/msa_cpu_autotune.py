#!/usr/bin/env python3
"""Auto-tune HMMER (MSA) thread count for Apple Silicon.

AlphaFold 3's data pipeline launches several genetic searches *concurrently*
(for a protein chain: 4 jackhmmer searches -- UniRef90, MGnify, UniProt, BFD).
Each search uses ``--jackhmmer_n_cpu`` / ``--nhmmer_n_cpu`` worker threads, so
the true thread demand is ``concurrency * per_search``. On Apple Silicon we want
that product to equal the number of *performance* (P) cores, so the heavy MSA
work stays on the P-cluster and does not oversubscribe or spill onto the
efficiency (E) cores.

    per_search_threads = max(1, P_cores // concurrency)

Detection uses ``sysctl hw.perflevel0.logicalcpu`` (the P-cluster on Apple
Silicon), with fallbacks for non-Apple hosts. Prints the per-search thread count
to stdout so shell scripts can capture it; ``--verbose`` prints the derivation
to stderr.

Examples (concurrency=4, the protein default):
    M2 Ultra  16 P-cores -> 4     M4 Max 10-12 P -> 2-3
    M2 Max     8 P-cores -> 2     M2      4 P    -> 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def _sysctl_int(key: str) -> int | None:
  try:
    out = subprocess.check_output(
        ['sysctl', '-n', key], stderr=subprocess.DEVNULL
    )
    return int(out.decode().strip())
  except Exception:
    return None


def performance_cores() -> int:
  """Number of performance cores (Apple Silicon P-cluster), with fallbacks."""
  for key in ('hw.perflevel0.logicalcpu', 'hw.perflevel0.physicalcpu'):
    v = _sysctl_int(key)
    if v:
      return v
  # Non-Apple-Silicon or older sysctl: fall back to physical/logical count.
  return _sysctl_int('hw.physicalcpu') or _sysctl_int('hw.logicalcpu') or 8


def per_search_threads(concurrency: int) -> int:
  return max(1, performance_cores() // max(1, concurrency))


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument(
      '--concurrency',
      type=int,
      default=4,
      help='Number of genetic searches AF3 runs concurrently (protein=4).',
  )
  ap.add_argument('--verbose', action='store_true')
  args = ap.parse_args()

  p = performance_cores()
  per = per_search_threads(args.concurrency)
  if args.verbose:
    e = _sysctl_int('hw.perflevel1.logicalcpu')
    total = _sysctl_int('hw.logicalcpu')
    print(
        f'performance_cores={p} efficiency_cores={e} total_logical={total} '
        f'concurrency={args.concurrency} per_search_threads={per} '
        f'total_worker_threads={per * args.concurrency}',
        file=sys.stderr,
    )
  print(per)


if __name__ == '__main__':
  main()
