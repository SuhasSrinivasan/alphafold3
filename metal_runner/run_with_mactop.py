# SPDX-License-Identifier: Apache-2.0
"""Runs one gate with synchronized mactop, console, and memory capture."""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any


_TIMEOUT_PATTERN = re.compile(
    r'kIOGPUCommandBufferCallbackErrorTimeout|Caused GPU Timeout Error'
)
_ENV_PREFIXES = ('JAX_', 'MLX_', 'XLA_', 'MTL_', 'MPS_')
_ENV_NAMES = {
    'OMP_NUM_THREADS',
    'PYTHONHASHSEED',
    'VECLIB_MAXIMUM_THREADS',
}
_HELD_EXEC = """
import os
import sys

fd = int(sys.argv[1])
try:
  token = os.read(fd, 1)
finally:
  os.close(fd)
if token != b'1':
  raise SystemExit('workload release token was not received')
os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
"""


def _now() -> datetime.datetime:
  return datetime.datetime.now(datetime.UTC)


def _positive_float(value: str) -> float:
  parsed = float(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError(f'expected a positive value: {value}')
  return parsed


def _positive_int(value: str) -> int:
  parsed = int(value)
  if parsed <= 0:
    raise argparse.ArgumentTypeError(f'expected a positive integer: {value}')
  return parsed


def _nonnegative_float(value: str) -> float:
  parsed = float(value)
  if parsed < 0:
    raise argparse.ArgumentTypeError(f'expected a non-negative value: {value}')
  return parsed


def _write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _run_text(command: list[str], *, timeout: float = 20) -> dict[str, Any]:
  try:
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=timeout,
    )
  except (OSError, subprocess.TimeoutExpired) as error:
    return {'command': command, 'error': repr(error)}
  return {
      'command': command,
      'exit_code': result.returncode,
      'stdout': result.stdout.rstrip(),
      'stderr': result.stderr.rstrip(),
  }


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _is_python_executable(value: str) -> bool:
  return Path(value).name.startswith('python')


def _record_static_metadata(
    *, output_dir: Path, command: list[str], mactop: str
) -> None:
  cwd = Path.cwd()
  system = {
      'platform': platform.platform(),
      'machine': platform.machine(),
      'python_running_wrapper': sys.version,
      'sw_vers': _run_text(['sw_vers']),
      'uname': _run_text(['uname', '-a']),
      'hardware_memory': _run_text(['sysctl', '-n', 'hw.memsize']),
      'hardware_model': _run_text(['sysctl', '-n', 'machdep.cpu.brand_string']),
      'mactop': _run_text([mactop, '--version']),
  }
  _write_json(output_dir / 'system.json', system)
  git = {
      'head': _run_text(['git', 'rev-parse', 'HEAD']),
      'branch': _run_text(['git', 'branch', '--show-current']),
      'status': _run_text(['git', 'status', '--porcelain=v1']),
  }
  _write_json(output_dir / 'git.json', git)
  _write_json(
      output_dir / 'command.json',
      {'argv': command, 'cwd': str(cwd), 'wrapper_argv': sys.argv},
  )
  allowed_environment = {
      key: value
      for key, value in sorted(os.environ.items())
      if key in _ENV_NAMES or key.startswith(_ENV_PREFIXES)
  }
  _write_json(output_dir / 'environment.json', allowed_environment)

  if not _is_python_executable(command[0]):
    return
  package_result = _run_text(
      [command[0], '-m', 'pip', 'freeze', '--all'], timeout=60
  )
  _write_json(output_dir / 'packages.json', package_result)
  version_script = """
import importlib.metadata as metadata
import json

names = [
    'alphafold3',
    'jax',
    'jaxlib',
    'jax-mps',
    'tokamax',
    'dm-haiku',
    'numpy',
    'scipy',
    'ml-dtypes',
]
versions = {}
for name in names:
  try:
    versions[name] = metadata.version(name)
  except metadata.PackageNotFoundError:
    versions[name] = None
print(json.dumps(versions))
"""
  versions = _run_text([command[0], '-c', version_script])
  if versions.get('exit_code') == 0:
    try:
      versions['parsed'] = json.loads(versions['stdout'])
    except json.JSONDecodeError:
      pass
  _write_json(output_dir / 'package_versions.json', versions)
  dylib_script = (
      'from pathlib import Path; import jax_plugins.mps as m; '
      "print(Path(m.__file__).parent / 'lib' / 'libpjrt_plugin_mps.dylib')"
  )
  dylib_result = _run_text([command[0], '-c', dylib_script])
  dylib_metadata: dict[str, Any] = {'lookup': dylib_result}
  if dylib_result.get('exit_code') == 0:
    dylib_path = Path(dylib_result['stdout'].strip())
    if dylib_path.is_file():
      dylib_metadata.update(
          {
              'filename': dylib_path.name,
              'bytes': dylib_path.stat().st_size,
              'sha256': _sha256(dylib_path),
          }
      )
  _write_json(output_dir / 'jax_mps_dylib.json', dylib_metadata)


def _spawn_held_workload(
    command: list[str], *, environment: dict[str, str]
) -> tuple[subprocess.Popen[str], int]:
  read_fd, write_fd = os.pipe()
  helper = [sys.executable, '-c', _HELD_EXEC, str(read_fd), *command]
  try:
    process = subprocess.Popen(
        helper,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
        pass_fds=(read_fd,),
        start_new_session=True,
    )
  except Exception:
    os.close(read_fd)
    os.close(write_fd)
    raise
  os.close(read_fd)
  return process, write_fd


def _release_workload(write_fd: int) -> None:
  try:
    os.write(write_fd, b'1')
  finally:
    os.close(write_fd)


def _discard_release_fd(write_fd: int | None) -> None:
  if write_fd is None:
    return
  try:
    os.close(write_fd)
  except OSError:
    pass


def _stop_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = 5
) -> None:
  if process.poll() is not None:
    return
  try:
    os.killpg(process.pid, signal.SIGTERM)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=grace_seconds)
  except subprocess.TimeoutExpired:
    try:
      os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
      return
    process.wait(timeout=grace_seconds)


def _stop_process(process: subprocess.Popen[Any]) -> None:
  if process.poll() is not None:
    return
  process.terminate()
  try:
    process.wait(timeout=5)
  except subprocess.TimeoutExpired:
    process.kill()
    process.wait(timeout=5)


def _drain_stream(
    *,
    stream: Any,
    stream_name: str,
    raw_path: Path,
    console_output: Any,
    console_events: Any,
    console_lock: threading.Lock,
) -> None:
  with raw_path.open('w', buffering=1) as raw_output:
    for line in stream:
      raw_output.write(line)
      raw_output.flush()
      event = {
          'timestamp_utc': _now().isoformat(),
          'stream': stream_name,
          'line': line.rstrip('\n'),
      }
      with console_lock:
        console_events.write(json.dumps(event, sort_keys=True) + '\n')
        console_events.flush()
        console_output.write(line)
        console_output.flush()


def _process_group_memory(pgid: int) -> tuple[int, int, int]:
  result = subprocess.run(
      ('ps', '-axo', 'pid=,pgid=,rss='),
      capture_output=True,
      check=False,
      text=True,
  )
  if result.returncode != 0:
    raise RuntimeError(
        f'ps failed while sampling process memory: {result.stderr.strip()}'
    )
  group_rss_kib = 0
  root_rss_kib = 0
  process_count = 0
  for line in result.stdout.splitlines():
    fields = line.split()
    if len(fields) != 3:
      continue
    try:
      pid, process_group, rss_kib = map(int, fields)
    except ValueError:
      continue
    if process_group != pgid:
      continue
    group_rss_kib += rss_kib
    process_count += 1
    if pid == pgid:
      root_rss_kib = rss_kib
  return root_rss_kib * 1024, group_rss_kib * 1024, process_count


def _memory_monitor(
    *,
    workload: subprocess.Popen[str],
    output_path: Path,
    interval_seconds: float,
    terminate_rss_bytes: int,
    absolute_rss_bytes: int,
    stop: threading.Event,
    limit_exceeded: threading.Event,
    statistics: dict[str, int],
    errors: list[str],
) -> None:
  with output_path.open('w', newline='') as output:
    writer = csv.writer(output)
    writer.writerow(
        (
            'timestamp_utc',
            'root_rss_bytes',
            'process_group_rss_bytes',
            'process_count',
        )
    )
    while not stop.is_set():
      try:
        root_rss, group_rss, process_count = _process_group_memory(workload.pid)
      except Exception as error:  # Surface safety-monitor failure to main.
        errors.append(repr(error))
        _stop_process_group(workload)
        return
      if process_count:
        writer.writerow(
            (_now().isoformat(), root_rss, group_rss, process_count)
        )
        output.flush()
        statistics['peak_root_rss_bytes'] = max(
            statistics.get('peak_root_rss_bytes', 0), root_rss
        )
        statistics['peak_process_group_rss_bytes'] = max(
            statistics.get('peak_process_group_rss_bytes', 0), group_rss
        )
        threshold = min(terminate_rss_bytes, absolute_rss_bytes)
        if group_rss >= threshold and workload.poll() is None:
          limit_exceeded.set()
          _stop_process_group(workload)
          return
      elif workload.poll() is not None:
        return
      stop.wait(interval_seconds)


def _mactop_samples(path: Path) -> int:
  if not path.exists():
    return 0
  try:
    with path.open(errors='replace') as source:
      return max(sum(bool(line.strip()) for line in source) - 1, 0)
  except OSError:
    return 0


def _wait_for_monitor(
    *,
    monitor: subprocess.Popen[Any],
    mactop_path: Path,
    required_samples: int,
    timeout_seconds: float,
) -> int:
  deadline = time.monotonic() + timeout_seconds
  while time.monotonic() < deadline:
    samples = _mactop_samples(mactop_path)
    if samples >= required_samples:
      return samples
    if monitor.poll() is not None:
      raise RuntimeError(
          f'mactop exited before baseline capture (exit {monitor.returncode})'
      )
    time.sleep(0.25)
  raise TimeoutError(
      f'mactop did not produce {required_samples} samples within '
      f'{timeout_seconds} seconds'
  )


def _run_health_check(
    *,
    python: str,
    script: Path,
    output_dir: Path,
    environment: dict[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
  health_dir = output_dir / 'health'
  health_dir.mkdir()
  command = [python, str(script)]
  health_environment = environment.copy()
  for name in (
      'JAX_MPS_GPU_CAPTURE',
      'JAX_MPS_GPU_CAPTURE_DISPATCHES',
      'JAX_MPS_DUMP_OPTIMIZED_IR',
      'MTL_CAPTURE_ENABLED',
  ):
    health_environment.pop(name, None)
  started = _now()
  try:
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        env=health_environment,
        timeout=timeout_seconds,
    )
    stdout = result.stdout
    stderr = result.stderr
    exit_code: int | None = result.returncode
    timed_out = False
  except subprocess.TimeoutExpired as error:
    stdout = error.stdout or ''
    stderr = error.stderr or ''
    if isinstance(stdout, bytes):
      stdout = stdout.decode(errors='replace')
    if isinstance(stderr, bytes):
      stderr = stderr.decode(errors='replace')
    exit_code = None
    timed_out = True
  (health_dir / 'stdout.log').write_text(stdout)
  (health_dir / 'stderr.log').write_text(stderr)
  metadata = {
      'command': command,
      'started_at_utc': started.isoformat(),
      'finished_at_utc': _now().isoformat(),
      'exit_code': exit_code,
      'timed_out': timed_out,
      'passed': exit_code == 0 and not timed_out,
  }
  _write_json(health_dir / 'result.json', metadata)
  return metadata


def _parse_timestamp(value: str) -> datetime.datetime | None:
  try:
    parsed = datetime.datetime.fromisoformat(value)
  except ValueError:
    return None
  if parsed.tzinfo is None:
    return parsed.replace(tzinfo=datetime.UTC)
  return parsed.astimezone(datetime.UTC)


def _numeric_summary(values: list[float]) -> dict[str, float] | None:
  if not values:
    return None
  return {
      'minimum': min(values),
      'maximum': max(values),
      'mean': sum(values) / len(values),
  }


def _summarize_mactop(
    *,
    source_path: Path,
    selected_path: Path,
    process_selected_path: Path,
    monitored_pid: int | None,
    released_at: datetime.datetime | None,
    workload_finished_at: datetime.datetime | None,
) -> dict[str, Any]:
  selected_fields = (
      'Timestamp',
      'GPU_Usage',
      'GPU_Freq_MHz',
      'GPU_Active_Percent',
      'Mem_Used',
      'Mem_Total',
      'Swap_Used',
      'Total_Power',
      'CPU_Temp',
      'GPU_Temp',
      'Thermal_State',
      'DRAM_Read_BW_GBs',
      'DRAM_Write_BW_GBs',
      'DRAM_BW_Combined_GBs',
  )
  rows: list[dict[str, str]] = []
  process_rows: list[dict[str, Any]] = []
  with source_path.open(newline='', errors='replace') as source:
    reader = csv.DictReader(source)
    for row in reader:
      if row:
        rows.append(row)
        if monitored_pid is None:
          continue
        try:
          processes = json.loads(row.get('Processes_JSON', 'null')) or []
        except (json.JSONDecodeError, TypeError):
          continue
        for process in processes:
          if process.get('pid') != monitored_pid:
            continue
          process_rows.append(
              {
                  'Timestamp': row.get('Timestamp', ''),
                  'pid': process.get('pid'),
                  'cpu_percent': process.get('cpu_percent'),
                  'gpu_ms_per_sec': process.get('gpu_ms_per_sec'),
                  'memory_percent': process.get('memory_percent'),
                  'rss_kb': process.get('rss_kb'),
              }
          )
  with selected_path.open('w', newline='') as selected:
    writer = csv.DictWriter(selected, fieldnames=selected_fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({field: row.get(field, '') for field in selected_fields})
  process_fields = (
      'Timestamp',
      'pid',
      'cpu_percent',
      'gpu_ms_per_sec',
      'memory_percent',
      'rss_kb',
  )
  with process_selected_path.open('w', newline='') as selected:
    writer = csv.DictWriter(selected, fieldnames=process_fields)
    writer.writeheader()
    writer.writerows(process_rows)

  def subset(name: str) -> list[dict[str, str]]:
    if name == 'all' or released_at is None:
      return rows
    output = []
    for row in rows:
      timestamp = _parse_timestamp(row.get('Timestamp', ''))
      if timestamp is None:
        continue
      if name == 'baseline' and timestamp < released_at:
        output.append(row)
      elif (
          name == 'workload'
          and timestamp >= released_at
          and (workload_finished_at is None or timestamp <= workload_finished_at)
      ):
        output.append(row)
      elif (
          name == 'recovery'
          and workload_finished_at is not None
          and timestamp > workload_finished_at
      ):
        output.append(row)
    return output

  result: dict[str, Any] = {}
  for name in ('baseline', 'workload', 'recovery', 'all'):
    current = subset(name)
    fields: dict[str, Any] = {'samples': len(current)}
    for field in (
        'GPU_Usage',
        'GPU_Freq_MHz',
        'Mem_Used',
        'Swap_Used',
        'GPU_Temp',
        'DRAM_BW_Combined_GBs',
    ):
      values = []
      for row in current:
        try:
          values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
          continue
      fields[field] = _numeric_summary(values)
    fields['thermal_states'] = sorted(
        {row.get('Thermal_State', '') for row in current}
    )
    result[name] = fields
  process_summary: dict[str, Any] = {'samples': len(process_rows)}
  for field in ('cpu_percent', 'gpu_ms_per_sec', 'memory_percent', 'rss_kb'):
    process_summary[field] = _numeric_summary(
        [
            float(row[field])
            for row in process_rows
            if row.get(field) is not None
        ]
    )
  process_summary['gpu_active_samples'] = sum(
      float(row.get('gpu_ms_per_sec') or 0) > 0 for row in process_rows
  )
  result['monitored_process'] = process_summary
  return result


def _write_manifest(output_dir: Path) -> None:
  manifest_path = output_dir / 'manifest.sha256'
  lines = []
  for path in sorted(output_dir.rglob('*')):
    if not path.is_file() or path == manifest_path:
      continue
    lines.append(f'{_sha256(path)}  {path.relative_to(output_dir)}')
  manifest_path.write_text('\n'.join(lines) + '\n')


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--output-dir',
      type=Path,
      required=True,
      help='New directory for all private diagnostic artifacts.',
  )
  parser.add_argument('--interval-ms', type=_positive_int, default=1000)
  parser.add_argument('--max-rss-gib', type=_positive_float, default=32.0)
  parser.add_argument(
      '--terminate-rss-gib',
      type=_positive_float,
      help=(
          'Early-stop threshold; defaults to min(30, --max-rss-gib) and must '
          'not exceed --max-rss-gib.'
      ),
  )
  parser.add_argument('--baseline-samples', type=_positive_int, default=3)
  parser.add_argument(
      '--monitor-ready-timeout-seconds', type=_positive_float, default=30
  )
  parser.add_argument('--recovery-seconds', type=_nonnegative_float, default=10)
  parser.add_argument('--health-timeout-seconds', type=_positive_float, default=30)
  parser.add_argument(
      '--health-script',
      type=Path,
      default=Path(__file__).with_name('mps_health_gate.py'),
  )
  parser.add_argument('command', nargs=argparse.REMAINDER)
  args = parser.parse_args()
  command = args.command[1:] if args.command[:1] == ['--'] else args.command
  if not command:
    parser.error('provide a workload command after --')
  terminate_rss_gib = (
      min(30.0, args.max_rss_gib)
      if args.terminate_rss_gib is None
      else args.terminate_rss_gib
  )
  if terminate_rss_gib > args.max_rss_gib:
    parser.error('--terminate-rss-gib must not exceed --max-rss-gib')
  if not _is_python_executable(command[0]):
    parser.error('the monitored command must begin with a Python executable')

  mactop = shutil.which('mactop')
  if not mactop:
    raise SystemExit('mactop is required but was not found on PATH.')
  health_script = args.health_script.expanduser().resolve()
  if not health_script.is_file():
    raise SystemExit(f'health script does not exist: {health_script}')
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=False)
  (output_dir / 'mactop_state').mkdir()

  started_at = _now()
  _write_json(
      output_dir / 'run.partial.json',
      {
          'status': 'preparing',
          'started_at_utc': started_at.isoformat(),
          'command': command,
      },
  )
  _record_static_metadata(
      output_dir=output_dir, command=command, mactop=mactop
  )

  child_environment = os.environ.copy()
  child_environment.setdefault('PYTHONUNBUFFERED', '1')
  child_environment.setdefault('PYTHONFAULTHANDLER', '1')
  mactop_environment = os.environ.copy()
  mactop_environment['XDG_STATE_HOME'] = str(output_dir / 'mactop_state')

  mactop_path = output_dir / 'mactop.csv'
  mactop_stderr_path = output_dir / 'mactop.stderr.log'
  memory_path = output_dir / 'process_memory.csv'
  console_path = output_dir / 'console.jsonl'
  stdout_path = output_dir / 'workload.stdout.log'
  stderr_path = output_dir / 'workload.stderr.log'
  stop_memory = threading.Event()
  limit_exceeded = threading.Event()
  external_signal = threading.Event()
  received_signals: list[int] = []
  memory_statistics: dict[str, int] = {}
  memory_errors: list[str] = []
  workload: subprocess.Popen[str] | None = None
  monitor: subprocess.Popen[Any] | None = None
  release_fd: int | None = None
  workload_code: int | None = None
  released_at: datetime.datetime | None = None
  workload_finished_at: datetime.datetime | None = None
  baseline_samples = 0
  health: dict[str, Any] | None = None
  run_error: str | None = None
  drain_threads: list[threading.Thread] = []
  previous_handlers: dict[int, Any] = {}

  def handle_signal(signum: int, _frame: Any) -> None:
    received_signals.append(signum)
    external_signal.set()

  try:
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
      previous_handlers[signum] = signal.signal(signum, handle_signal)
    with (
        mactop_path.open('w', buffering=1) as mactop_output,
        mactop_stderr_path.open('w', buffering=1) as mactop_stderr,
        console_path.open('w', buffering=1) as console_events,
    ):
      workload, release_fd = _spawn_held_workload(
          command, environment=child_environment
      )
      console_lock = threading.Lock()
      assert workload.stdout is not None
      assert workload.stderr is not None
      drain_threads = [
          threading.Thread(
              target=_drain_stream,
              kwargs={
                  'stream': workload.stdout,
                  'stream_name': 'stdout',
                  'raw_path': stdout_path,
                  'console_output': sys.stdout,
                  'console_events': console_events,
                  'console_lock': console_lock,
              },
              daemon=True,
          ),
          threading.Thread(
              target=_drain_stream,
              kwargs={
                  'stream': workload.stderr,
                  'stream_name': 'stderr',
                  'raw_path': stderr_path,
                  'console_output': sys.stderr,
                  'console_events': console_events,
                  'console_lock': console_lock,
              },
              daemon=True,
          ),
      ]
      for thread in drain_threads:
        thread.start()

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
          env=mactop_environment,
      )
      memory_thread = threading.Thread(
          target=_memory_monitor,
          kwargs={
              'workload': workload,
              'output_path': memory_path,
              'interval_seconds': min(args.interval_ms / 1000, 0.5),
              'terminate_rss_bytes': int(terminate_rss_gib * 1024**3),
              'absolute_rss_bytes': int(args.max_rss_gib * 1024**3),
              'stop': stop_memory,
              'limit_exceeded': limit_exceeded,
              'statistics': memory_statistics,
              'errors': memory_errors,
          },
          daemon=True,
      )
      memory_thread.start()
      baseline_samples = _wait_for_monitor(
          monitor=monitor,
          mactop_path=mactop_path,
          required_samples=args.baseline_samples,
          timeout_seconds=args.monitor_ready_timeout_seconds,
      )
      released_at = _now()
      _write_json(
          output_dir / 'run.partial.json',
          {
              'status': 'running',
              'started_at_utc': started_at.isoformat(),
              'released_at_utc': released_at.isoformat(),
              'baseline_samples': baseline_samples,
              'command': command,
          },
      )
      _release_workload(release_fd)
      release_fd = None
      while workload.poll() is None:
        if monitor.poll() is not None:
          raise RuntimeError(
              'mactop exited during the workload '
              f'(exit {monitor.returncode}); stopping the run'
          )
        if external_signal.is_set():
          _stop_process_group(workload)
          break
        time.sleep(0.2)
      workload_code = workload.wait()
      workload_finished_at = _now()
      stop_memory.set()
      memory_thread.join(timeout=5)
      if memory_errors:
        raise RuntimeError(
            f'process memory monitor failed: {memory_errors[0]}'
        )
      for thread in drain_threads:
        thread.join(timeout=5)
      time.sleep(args.recovery_seconds)
      health = _run_health_check(
          python=command[0],
          script=health_script,
          output_dir=output_dir,
          environment=child_environment,
          timeout_seconds=args.health_timeout_seconds,
      )
  except Exception as error:  # Preserve partial evidence and clean up below.
    run_error = repr(error)
  finally:
    _discard_release_fd(release_fd)
    stop_memory.set()
    if workload is not None:
      _stop_process_group(workload)
      if workload_code is None:
        workload_code = workload.poll()
    for thread in drain_threads:
      thread.join(timeout=5)
    if monitor is not None:
      _stop_process(monitor)
    for signum, handler in previous_handlers.items():
      signal.signal(signum, handler)

  finished_at = _now()
  timeout_detected = False
  for log_path in (stdout_path, stderr_path):
    if log_path.exists() and _TIMEOUT_PATTERN.search(
        log_path.read_text(errors='replace')
    ):
      timeout_detected = True
  try:
    mactop_summary = _summarize_mactop(
        source_path=mactop_path,
        selected_path=output_dir / 'mactop.selected.csv',
        process_selected_path=output_dir / 'mactop.process.csv',
        monitored_pid=workload.pid if workload is not None else None,
        released_at=released_at,
        workload_finished_at=workload_finished_at,
    )
  except Exception as error:
    mactop_summary = {'error': repr(error)}
  _write_json(output_dir / 'mactop_summary.json', mactop_summary)

  monitor_code = monitor.returncode if monitor is not None else None
  metadata = {
      'status': 'complete' if run_error is None else 'wrapper_error',
      'command': command,
      'started_at_utc': started_at.isoformat(),
      'released_at_utc': released_at.isoformat() if released_at else None,
      'workload_finished_at_utc': (
          workload_finished_at.isoformat() if workload_finished_at else None
      ),
      'finished_at_utc': finished_at.isoformat(),
      'workload_exit_code': workload_code,
      'mactop_exit_code': monitor_code,
      'baseline_samples': baseline_samples,
      'interval_ms': args.interval_ms,
      'max_rss_gib': args.max_rss_gib,
      'terminate_rss_gib': terminate_rss_gib,
      'rss_limit_exceeded': limit_exceeded.is_set(),
      'memory_statistics': memory_statistics,
      'metal_timeout_detected': timeout_detected,
      'received_signals': received_signals,
      'health': health,
      'wrapper_error': run_error,
  }
  _write_json(output_dir / 'run.json', metadata)
  _write_manifest(output_dir)
  print(json.dumps(metadata, sort_keys=True), flush=True)

  if run_error is not None:
    print(f'Monitoring wrapper failed: {run_error}', file=sys.stderr)
    return 4
  if limit_exceeded.is_set():
    print('RSS safety threshold exceeded; workload was stopped.', file=sys.stderr)
    return 3
  if health is None or not health['passed']:
    print('Postflight MPS health check failed; stop further tests.', file=sys.stderr)
    return 5
  if workload_code is None:
    return 4
  return workload_code if workload_code >= 0 else 1


if __name__ == '__main__':
  raise SystemExit(main())
