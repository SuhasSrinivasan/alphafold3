# SPDX-License-Identifier: Apache-2.0
"""Runs dependency, device, transfer, BF16, and control-flow MPS gates."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import time

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from _common import emit
from _common import nonnegative_int
from _common import positive_int
from _common import select_mps_device
from _common import wait_for_monitor


def _version(distribution: str) -> str:
  try:
    return importlib.metadata.version(distribution)
  except importlib.metadata.PackageNotFoundError:
    return 'not-installed'


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--matrix-size',
      type=positive_int,
      default=4096,
      help='Square BF16 matmul size. The default is intentionally memory-safe.',
  )
  parser.add_argument(
      '--control-flow-steps',
      type=positive_int,
      default=128,
      help='Iteration count for scan and while-loop compatibility gates.',
  )
  parser.add_argument(
      '--monitor-warmup-seconds', type=nonnegative_int, default=10
  )
  args = parser.parse_args()

  emit(
      'environment',
      python=platform.python_version(),
      jax=_version('jax'),
      jaxlib=_version('jaxlib'),
      jax_mps=_version('jax-mps'),
      tokamax=_version('tokamax'),
      dm_haiku=_version('dm-haiku'),
  )
  device = select_mps_device('xla')
  wait_for_monitor(args.monitor_warmup_seconds)

  host_input = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
  device_input = jax.device_put(host_input, device)
  elementwise = jax.jit(lambda x: jnp.sin(x) + jnp.cos(x))
  elementwise_result = elementwise(device_input)
  elementwise_result.block_until_ready()
  roundtrip = np.asarray(jax.device_get(elementwise_result))
  if roundtrip.shape != host_input.shape or not np.isfinite(roundtrip).all():
    raise AssertionError('JIT/transfer gate returned invalid output.')
  emit('jit_transfer_ok', shape=roundtrip.shape, dtype=str(roundtrip.dtype))

  matrix = jax.device_put(
      np.ones((args.matrix_size, args.matrix_size), dtype=np.float32), device
  ).astype(jnp.bfloat16)
  matmul = jax.jit(lambda x: x @ x)
  start = time.monotonic()
  matmul_result = matmul(matrix)
  matmul_result.block_until_ready()
  matmul_seconds = time.monotonic() - start
  matmul_value = float(jax.device_get(matmul_result[0, 0]))
  if not np.isfinite(matmul_value):
    raise AssertionError(matmul_value)
  emit(
      'bf16_matmul_ok',
      matrix_size=args.matrix_size,
      dtype=str(matmul_result.dtype),
      sample=matmul_value,
      seconds=round(matmul_seconds, 3),
  )

  scan_input = jax.device_put(
      np.arange(args.control_flow_steps, dtype=np.float32), device
  )

  def run_scan(values):
    return lax.scan(lambda carry, x: (carry + x, carry), 0.0, values)[0]

  scan_result = jax.jit(run_scan)(scan_input)
  scan_result.block_until_ready()

  def run_while(value):
    initial = (jnp.asarray(0, dtype=jnp.int32), value)
    return lax.while_loop(
        lambda state: state[0] < args.control_flow_steps,
        lambda state: (state[0] + 1, state[1] + 1.0),
        initial,
    )[1]

  while_result = jax.jit(run_while)(jax.device_put(np.float32(0), device))
  while_result.block_until_ready()
  emit(
      'control_flow_ok',
      steps=args.control_flow_steps,
      scan=float(jax.device_get(scan_result)),
      while_loop=float(jax.device_get(while_result)),
  )


if __name__ == '__main__':
  main()
