# SPDX-License-Identifier: Apache-2.0
"""Compares two saved AF3 diagnostic prediction archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _compare_array(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
  result: dict[str, Any] = {
      'reference_shape': list(reference.shape),
      'candidate_shape': list(candidate.shape),
      'reference_dtype': str(reference.dtype),
      'candidate_dtype': str(candidate.dtype),
  }
  compatible = (
      reference.shape == candidate.shape
      and reference.dtype == candidate.dtype
  )
  result['compatible'] = compatible
  if not compatible:
    result['exact'] = False
    return result
  result['exact'] = bool(np.array_equal(reference, candidate, equal_nan=True))
  result['byte_identical'] = reference.tobytes(order='C') == candidate.tobytes(
      order='C'
  )
  if not np.issubdtype(reference.dtype, np.number):
    return result
  comparison_dtype = (
      np.complex128
      if np.issubdtype(reference.dtype, np.complexfloating)
      else np.float64
  )
  reference_numeric = reference.astype(comparison_dtype)
  candidate_numeric = candidate.astype(comparison_dtype)
  finite_pairs = np.isfinite(reference_numeric) & np.isfinite(candidate_numeric)
  if np.issubdtype(reference.dtype, np.complexfloating):
    matching_real = (
        reference_numeric.real == candidate_numeric.real
    ) | (
        np.isnan(reference_numeric.real) & np.isnan(candidate_numeric.real)
    )
    matching_imaginary = (
        reference_numeric.imag == candidate_numeric.imag
    ) | (
        np.isnan(reference_numeric.imag) & np.isnan(candidate_numeric.imag)
    )
    matching_nonfinite = ~finite_pairs & matching_real & matching_imaginary
  else:
    matching_nonfinite = ~finite_pairs & (
        (reference_numeric == candidate_numeric)
        | (np.isnan(reference_numeric) & np.isnan(candidate_numeric))
    )
  nonfinite_mismatch_count = int(
      reference.size - np.count_nonzero(finite_pairs | matching_nonfinite)
  )
  if np.any(finite_pairs):
    absolute = np.abs(
        candidate_numeric[finite_pairs] - reference_numeric[finite_pairs]
    )
    denominator = np.maximum(
        np.abs(reference_numeric[finite_pairs]), np.finfo(np.float64).tiny
    )
    relative = absolute / denominator
    maximum_absolute = float(np.max(absolute))
    mean_absolute = float(np.mean(absolute))
    maximum_relative = float(np.max(relative))
  else:
    maximum_absolute = 0.0
    mean_absolute = 0.0
    maximum_relative = 0.0
  result.update(
      {
          'finite_pair_count': int(np.count_nonzero(finite_pairs)),
          'matching_nonfinite_count': int(np.count_nonzero(matching_nonfinite)),
          'nonfinite_mismatch_count': nonfinite_mismatch_count,
          'max_absolute_difference': maximum_absolute,
          'mean_absolute_difference': mean_absolute,
          'max_relative_difference': maximum_relative,
      }
  )
  return result


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('reference', type=Path)
  parser.add_argument('candidate', type=Path)
  parser.add_argument('--output', type=Path)
  parser.add_argument('--require-exact', action='store_true')
  args = parser.parse_args()
  reference_path = args.reference.expanduser().resolve()
  candidate_path = args.candidate.expanduser().resolve()
  with (
      np.load(reference_path, allow_pickle=False) as reference,
      np.load(candidate_path, allow_pickle=False) as candidate,
  ):
    reference_keys = set(reference.files)
    candidate_keys = set(candidate.files)
    shared_keys = sorted(reference_keys & candidate_keys)
    arrays = {
        key: _compare_array(reference[key], candidate[key])
        for key in shared_keys
    }
  exact = (
      reference_keys == candidate_keys
      and all(value['exact'] for value in arrays.values())
  )
  comparison = {
      'reference': str(args.reference),
      'candidate': str(args.candidate),
      'reference_sha256': _sha256(reference_path),
      'candidate_sha256': _sha256(candidate_path),
      'reference_only_keys': sorted(reference_keys - candidate_keys),
      'candidate_only_keys': sorted(candidate_keys - reference_keys),
      'shared_keys': shared_keys,
      'all_arrays_exact': exact,
      'arrays': arrays,
  }
  rendered = json.dumps(comparison, indent=2, sort_keys=True) + '\n'
  if args.output is not None:
    output_path = args.output.expanduser().resolve()
    if not output_path.parent.is_dir():
      raise SystemExit(f'Output directory does not exist: {output_path.parent}')
    output_path.write_text(rendered)
  print(rendered, end='')
  if args.require_exact and not exact:
    return 2
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
