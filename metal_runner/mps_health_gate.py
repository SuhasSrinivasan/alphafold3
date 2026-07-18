# SPDX-License-Identifier: Apache-2.0
"""Runs a tiny fresh-process MPS health check after a monitored workload."""

from __future__ import annotations

import json
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np


def _emit(event: str, **values: object) -> None:
  print(json.dumps({'event': event, **values}, sort_keys=True), flush=True)


def main() -> None:
  started = time.monotonic()
  devices = jax.local_devices(backend='mps')
  if len(devices) != 1:
    raise AssertionError(f'Expected one MPS device, got {devices!r}.')
  device = devices[0]
  host = np.arange(256 * 256, dtype=np.float32).reshape(256, 256)
  value = jax.device_put(host, device).astype(jnp.bfloat16)
  result = jax.jit(lambda x: jnp.tanh(x @ x.T))(value)
  result.block_until_ready()
  sample = np.asarray(jax.device_get(result[:2, :2]), dtype=np.float32)
  if not np.isfinite(sample).all():
    raise AssertionError(sample)
  _emit(
      'mps_health_ok',
      python=platform.python_version(),
      jax=jax.__version__,
      device=str(device),
      dtype=str(result.dtype),
      elapsed_seconds=round(time.monotonic() - started, 3),
  )


if __name__ == '__main__':
  main()
