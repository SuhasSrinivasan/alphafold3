# SPDX-License-Identifier: Apache-2.0
"""Runs one gate with 1-second mactop and process-RSS capture."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading


def _positive_float(value: str) -> float:
  parsed = float(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError(f'expected a positive value: {value}')
  return parsed


def _rss_monitor(
    *,
    workload: subprocess.Popen[str],
    output_path: Path,
    interval_seconds: float,
    max_rss_bytes: int,
    stop: threading.Event,
    limit_exceeded: threading.Event,
) -> None:
  with output_path.open('w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(('timestamp_utc', 'rss_bytes'))
    while not stop.is_set():
      result = subprocess.run(
          ('ps', '-o', 'rss=', '-p', str(workload.pid)),
          capture_output=True,
          check=False,
          text=True,
      )
      value = result.stdout.strip()
      if value:
        rss_bytes = int(value) * 1024
        writer.writerow(
            (datetime.datetime.now(datetime.UTC).isoformat(), rss_bytes)
        )
        output.flush()
        if rss_bytes > max_rss_bytes and workload.poll() is None:
          limit_exceeded.set()
          workload.terminate()
          return
      elif workload.poll() is not None:
        return
      stop.wait(interval_seconds)


def _stop_process(process: subprocess.Popen[object]) -> None:
  if process.poll() is not None:
    return
  process.terminate()
  try:
    process.wait(timeout=5)
  except subprocess.TimeoutExpired:
    process.kill()
    process.wait(timeout=5)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--output-dir',
      type=Path,
      required=True,
      help='New directory for workload.log, mactop.csv, rss.csv, and metadata.',
  )
  parser.add_argument('--interval-ms', type=int, default=1000)
  parser.add_argument('--max-rss-gib', type=_positive_float, default=32.0)
  parser.add_argument('command', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.command[1:] if args.command[:1] == ['--'] else args.command
  if not command:
    parser.error('provide a workload command after --')
  if args.interval_ms <= 0:
    parser.error('--interval-ms must be positive')

  mactop = shutil.which('mactop')
  if not mactop:
    raise SystemExit('mactop is required but was not found on PATH.')
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=False)

  started_at = datetime.datetime.now(datetime.UTC)
  stop = threading.Event()
  limit_exceeded = threading.Event()
  workload_log_path = output_dir / 'workload.log'
  mactop_path = output_dir / 'mactop.csv'
  mactop_stderr_path = output_dir / 'mactop.stderr.log'
  rss_path = output_dir / 'rss.csv'

  with (
      workload_log_path.open('w', buffering=1) as workload_log,
      mactop_path.open('w') as mactop_output,
      mactop_stderr_path.open('w') as mactop_stderr,
  ):
    workload = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    monitor = subprocess.Popen(
        (
            mactop,
            '--headless',
            '--format',
            'csv',
            '--interval',
            str(args.interval_ms),
            '--count',
            '0',
            '--pid',
            str(workload.pid),
        ),
        stdout=mactop_output,
        stderr=mactop_stderr,
    )
    rss_thread = threading.Thread(
        target=_rss_monitor,
        kwargs={
            'workload': workload,
            'output_path': rss_path,
            'interval_seconds': args.interval_ms / 1000,
            'max_rss_bytes': int(args.max_rss_gib * 1024**3),
            'stop': stop,
            'limit_exceeded': limit_exceeded,
        },
        daemon=True,
    )
    rss_thread.start()
    try:
      assert workload.stdout is not None
      for line in workload.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        workload_log.write(line)
      workload_code = workload.wait()
    except KeyboardInterrupt:
      _stop_process(workload)
      workload_code = 130
    finally:
      stop.set()
      rss_thread.join(timeout=5)
      _stop_process(monitor)

  metadata = {
      'command': command,
      'started_at_utc': started_at.isoformat(),
      'finished_at_utc': datetime.datetime.now(datetime.UTC).isoformat(),
      'workload_exit_code': workload_code,
      'mactop_exit_code': monitor.returncode,
      'interval_ms': args.interval_ms,
      'max_rss_gib': args.max_rss_gib,
      'rss_limit_exceeded': limit_exceeded.is_set(),
  }
  (output_dir / 'metadata.json').write_text(
      json.dumps(metadata, indent=2, sort_keys=True) + '\n'
  )
  print(json.dumps(metadata, sort_keys=True), flush=True)
  if limit_exceeded.is_set():
    print('RSS safety limit exceeded; workload was terminated.', file=sys.stderr)
    return 3
  if monitor.returncode not in (0, -15):
    print(
        f'mactop failed with exit code {monitor.returncode}; see '
        f'{mactop_stderr_path}.',
        file=sys.stderr,
    )
    return 4
  return workload_code if workload_code >= 0 else 1


if __name__ == '__main__':
  raise SystemExit(main())
