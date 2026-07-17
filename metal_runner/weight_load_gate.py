# SPDX-License-Identifier: Apache-2.0
"""Loads licensed AF3 parameters onto MPS without running inference."""

from __future__ import annotations

import argparse
import os
import time

from _common import add_model_arguments
from _common import block_and_report_parameters
from _common import emit
from _common import require_model_dir
from _common import run_alphafold
from _common import select_mps_device
from _common import wait_for_monitor


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  add_model_arguments(parser, default_attention='xla_chunked')
  args = parser.parse_args()
  model_dir = require_model_dir(args.model_dir)

  emit('process', pid=os.getpid())
  device = select_mps_device(args.attention)
  config = run_alphafold.make_model_config(
      flash_attention_implementation=args.attention,
      num_diffusion_samples=1,
      num_recycles=1,
  )
  runner = run_alphafold.ModelRunner(
      config=config, device=device, model_dir=model_dir
  )
  wait_for_monitor(args.monitor_warmup_seconds)
  emit('parameter_load_begin')
  start = time.monotonic()
  model_params = runner.model_params
  block_and_report_parameters(model_params, started_at=start)
  if args.cooldown_seconds:
    time.sleep(args.cooldown_seconds)


if __name__ == '__main__':
  main()
