# SPDX-License-Identifier: Apache-2.0
"""Runs a reduced weighted MPS smoke test with AF3's trusted fixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import time

import jax

from _common import REPO_ROOT
from _common import add_model_arguments
from _common import block_and_report_parameters
from _common import emit
from _common import nonnegative_int
from _common import positive_int
from _common import require_model_dir
from _common import run_alphafold
from _common import select_mps_device
from _common import validate_and_report_result
from _common import wait_for_monitor


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  add_model_arguments(parser, default_attention='xla_chunked')
  parser.add_argument(
      '--fixture',
      type=Path,
      default=REPO_ROOT / 'src/alphafold3/test_data/featurised_example.pkl',
  )
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num-recycles', type=nonnegative_int, default=1)
  parser.add_argument('--diffusion-steps', type=positive_int, default=4)
  args = parser.parse_args()
  model_dir = require_model_dir(args.model_dir)

  emit('process', pid=os.getpid())
  device = select_mps_device(args.attention)
  config = run_alphafold.make_model_config(
      flash_attention_implementation=args.attention,
      num_diffusion_samples=1,
      num_recycles=args.num_recycles,
      return_embeddings=False,
      return_distogram=False,
  )
  config.heads.diffusion.eval.steps = args.diffusion_steps
  runner = run_alphafold.ModelRunner(
      config=config, device=device, model_dir=model_dir
  )

  fixture_path = args.fixture.expanduser().resolve()
  featurised_examples = pickle.loads(fixture_path.read_bytes())
  if len(featurised_examples) != 1:
    raise AssertionError(len(featurised_examples))
  batch = featurised_examples[0]
  num_tokens = int(batch['seq_mask'].sum())
  emit(
      'fixture',
      path=fixture_path,
      tokens=num_tokens,
      msa_shape=batch['msa'].shape,
  )

  wait_for_monitor(args.monitor_warmup_seconds)
  emit('parameter_load_begin')
  start = time.monotonic()
  model_params = runner.model_params
  block_and_report_parameters(model_params, started_at=start)

  emit('inference_begin')
  start = time.monotonic()
  result = runner.run_inference(batch, jax.random.PRNGKey(args.seed))
  validate_and_report_result(
      result=result,
      runner=runner,
      batch=batch,
      target_name='mps_weighted_fixture_smoke',
      num_tokens=num_tokens,
      elapsed=time.monotonic() - start,
  )
  if args.cooldown_seconds:
    time.sleep(args.cooldown_seconds)


if __name__ == '__main__':
  main()
