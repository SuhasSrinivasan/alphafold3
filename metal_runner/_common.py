# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the experimental Apple Silicon GPU runners."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import run_alphafold  # pylint: disable=g-import-not-at-top
from alphafold3.common import folding_input  # pylint: disable=g-import-not-at-top
from alphafold3.constants import (  # pylint: disable=g-import-not-at-top
    chemical_components,
)
from alphafold3.data import featurisation  # pylint: disable=g-import-not-at-top


DEFAULT_SEQUENCE_A = 'ACDEFGHIK'
DEFAULT_SEQUENCE_B = 'LMNPQRSTV'


def emit(event: str, **values: Any) -> None:
  """Emits one stable, machine-readable progress record."""
  print(
      json.dumps({'event': event, **values}, default=str, sort_keys=True),
      flush=True,
  )


def nonnegative_int(value: str) -> int:
  parsed = int(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError(f'expected a non-negative integer: {value}')
  return parsed


def positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError(f'expected a positive integer: {value}')
  return parsed


def add_model_arguments(
    parser: argparse.ArgumentParser, *, default_attention: str
) -> None:
  parser.add_argument(
      '--model-dir',
      default=os.environ.get('AF3_MODEL_DIR'),
      help=(
          'Directory containing the licensed AF3 weights. Defaults to the '
          'AF3_MODEL_DIR environment variable.'
      ),
  )
  parser.add_argument(
      '--attention',
      choices=('xla', 'xla_chunked'),
      default=default_attention,
      help='Portable attention implementation used for the MPS experiment.',
  )
  parser.add_argument(
      '--monitor-warmup-seconds',
      type=nonnegative_int,
      default=10,
      help='Delay before loading weights so run_with_mactop.py can attach.',
  )
  parser.add_argument(
      '--cooldown-seconds',
      type=nonnegative_int,
      default=0,
      help='Optional delay after the gate for final monitoring samples.',
  )


def require_model_dir(value: str | None) -> Path:
  if not value:
    raise SystemExit('Pass --model-dir or set AF3_MODEL_DIR.')
  model_dir = Path(value).expanduser().resolve()
  if not model_dir.is_dir():
    raise SystemExit(f'Model directory does not exist: {model_dir}')
  return model_dir


def wait_for_monitor(seconds: int) -> None:
  if seconds:
    emit('monitor_warmup', seconds=seconds)
    time.sleep(seconds)


def select_mps_device(attention: str) -> jax.Device:
  """Selects and validates the explicitly requested MPS device."""
  device = run_alphafold._select_inference_device(
      use_cpu_only=False,
      accelerator_backend='mps',
      accelerator_device_index=0,
  )
  run_alphafold._validate_inference_device(device, attention)
  emit('device', device=str(device), platform=device.platform)
  return device


def make_msa_free_dimer(
    *, name: str, seed: int, sequence_a: str, sequence_b: str
) -> folding_input.Input:
  payload = {
      'name': name,
      'modelSeeds': [seed],
      'sequences': [
          {
              'protein': {
                  'id': 'A',
                  'sequence': sequence_a,
                  'unpairedMsa': '',
                  'pairedMsa': '',
                  'templates': [],
              }
          },
          {
              'protein': {
                  'id': 'B',
                  'sequence': sequence_b,
                  'unpairedMsa': '',
                  'pairedMsa': '',
                  'templates': [],
              }
          },
      ],
      'dialect': folding_input.JSON_DIALECT,
      'version': folding_input.JSON_VERSION,
  }
  return folding_input.Input.from_json(json.dumps(payload))


def featurise_msa_free_dimer(
    *,
    name: str,
    seed: int,
    sequence_a: str,
    sequence_b: str,
    bucket: int,
) -> tuple[folding_input.Input, dict[str, Any]]:
  num_tokens = len(sequence_a) + len(sequence_b)
  if bucket < num_tokens:
    raise SystemExit(
        f'Bucket {bucket} is smaller than the {num_tokens}-token dimer.'
    )
  fold_input = make_msa_free_dimer(
      name=name,
      seed=seed,
      sequence_a=sequence_a,
      sequence_b=sequence_b,
  )
  batches = featurisation.featurise_input(
      fold_input=fold_input,
      ccd=chemical_components.Ccd(),
      buckets=[bucket],
      verbose=True,
  )
  if len(batches) != 1:
    raise AssertionError(f'Expected one featurised batch, got {len(batches)}.')
  batch = batches[0]
  active_tokens = int(batch['seq_mask'].sum())
  active_msa_rows = int(batch['msa_mask'].any(axis=1).sum())
  active_templates = int(
      batch['template_atom_mask'].any(axis=(1, 2)).sum()
  )
  if active_tokens != num_tokens:
    raise AssertionError((active_tokens, num_tokens))
  if int(batch['num_alignments']) != 1 or active_msa_rows != 1:
    raise AssertionError('MSA-free input must contain only the query row.')
  if active_templates != 0:
    raise AssertionError('Template-free input unexpectedly has templates.')
  emit(
      'no_msa_batch',
      chains=len(fold_input.protein_chains),
      tokens=active_tokens,
      bucket=bucket,
      msa_shape=batch['msa'].shape,
      active_msa_rows=active_msa_rows,
      num_alignments=int(batch['num_alignments']),
      template_shape=batch['template_aatype'].shape,
      active_templates=active_templates,
  )
  return fold_input, batch


def block_and_report_parameters(model_params: Any, *, started_at: float) -> None:
  jax.tree.map(lambda array: array.block_until_ready(), model_params)
  elapsed = time.monotonic() - started_at
  leaves = jax.tree.leaves(model_params)
  emit(
      'parameter_load_ok',
      leaves=len(leaves),
      bytes=sum(array.nbytes for array in leaves),
      devices=sorted({str(array.device) for array in leaves}),
      seconds=round(elapsed, 3),
  )


def validate_and_report_result(
    *,
    result: dict[str, Any],
    runner: run_alphafold.ModelRunner,
    batch: dict[str, Any],
    target_name: str,
    num_tokens: int,
    elapsed: float,
) -> None:
  float_arrays = []
  for value in jax.tree.leaves(result):
    if not hasattr(value, 'dtype'):
      continue
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
      float_arrays.append(array)
  if not float_arrays or not all(
      np.isfinite(value).all() for value in float_arrays
  ):
    raise AssertionError('Inference returned no finite floating-point arrays.')

  expected_shapes = {
      ('diffusion_samples', 'atom_positions'): (1, num_tokens, 24, 3),
      ('diffusion_samples', 'mask'): (1, num_tokens, 24),
      ('predicted_lddt',): (1, num_tokens, 24),
      ('full_pae',): (1, num_tokens, num_tokens),
      ('full_pde',): (1, num_tokens, num_tokens),
      ('distogram', 'contact_probs'): (num_tokens, num_tokens),
  }
  for path, expected_shape in expected_shapes.items():
    value: Any = result
    for key in path:
      value = value[key]
    if value.shape != expected_shape:
      raise AssertionError((path, value.shape, expected_shape))

  inference_results = runner.extract_inference_results(
      batch=batch, result=result, target_name=target_name
  )
  if len(inference_results) != 1:
    raise AssertionError(len(inference_results))
  emit(
      'inference_ok',
      seconds=round(elapsed, 3),
      float_arrays=len(float_arrays),
      ranking_score=inference_results[0].metadata['ranking_score'],
      result_keys=sorted(result),
  )
