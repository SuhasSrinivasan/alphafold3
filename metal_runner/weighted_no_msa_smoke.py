# SPDX-License-Identifier: Apache-2.0
"""Runs a reduced weighted MPS smoke test on a database-free protein dimer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import jax

from _common import DEFAULT_SEQUENCE_A
from _common import DEFAULT_SEQUENCE_B
from _common import add_model_arguments
from _common import block_and_report_parameters
from _common import emit
from _common import featurise_msa_free_dimer
from _common import nonnegative_int
from _common import parse_runner_args
from _common import positive_int
from _common import require_model_dir
from _common import run_alphafold
from _common import save_result_artifact
from _common import select_mps_device
from _common import validate_and_report_result
from _common import wait_for_monitor


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  add_model_arguments(parser, default_attention='xla')
  parser.add_argument('--sequence-a', default=DEFAULT_SEQUENCE_A)
  parser.add_argument('--sequence-b', default=DEFAULT_SEQUENCE_B)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--bucket', type=positive_int, default=18)
  parser.add_argument('--num-recycles', type=nonnegative_int, default=0)
  parser.add_argument('--diffusion-steps', type=positive_int, default=4)
  parser.add_argument('--result-npz', type=Path)
  args = parse_runner_args(parser)
  model_dir = require_model_dir(args.model_dir)

  emit('process', pid=os.getpid())
  device = select_mps_device(args.attention)
  fold_input, batch = featurise_msa_free_dimer(
      name='mps_msa_free_dimer_weighted_smoke',
      seed=args.seed,
      sequence_a=args.sequence_a,
      sequence_b=args.sequence_b,
      bucket=args.bucket,
  )
  num_tokens = len(args.sequence_a) + len(args.sequence_b)

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
      target_name=fold_input.name,
      num_tokens=num_tokens,
      elapsed=time.monotonic() - start,
  )
  if args.result_npz is not None:
    save_result_artifact(result=result, output_path=args.result_npz)
  if args.cooldown_seconds:
    time.sleep(args.cooldown_seconds)


if __name__ == '__main__':
  main()
