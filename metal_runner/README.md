# Experimental AF3 inference on Apple Silicon GPU

This directory contains the minimal tooling retained from the Apple Silicon
feasibility study. JAX 0.10.2, jaxlib 0.10.2, and stable jax-mps 0.10.9 ran
reduced AlphaFold 3 weighted inference on an Apple GPU. This remains an
experimental backend and has not been numerically validated against CUDA. A
complete MSA-free `run_alphafold.py` test also passed for the 460-residue 5EXA
A/B homodimer using the current AF3 defaults of 10 recycles and 5 diffusion
samples.

Start with [AI_AGENT_SETUP.md](AI_AGENT_SETUP.md). It gives a reproducible
native macOS source build, patched HMMER and database setup, monitored smoke
tests, complete CLI gates, and a full MSA/template pipeline workflow.
[RESULTS.md](RESULTS.md) records the evidence, the bounded explanation of the
old Metal timeout, and the remaining validation work.

## Retained files

- `requirements/backend-jax-0.10.txt`: the selected stable backend versions.
- `backend_gate.py`: MPS discovery, JIT, transfers, BF16 matrix multiplication,
  `lax.scan`, and `lax.while_loop`.
- `weighted_no_msa_smoke.py` and `_common.py`: a reduced, weighted,
  database-free protein-dimer inference test. It uses AF3's real
  `ModelRunner`, parameters, featurisation, inference, and result extraction.
- `run_with_mactop.py` and `mps_health_gate.py`: synchronized workload logs,
  one-second `mactop` telemetry, process-group RSS limits, recovery sampling,
  postflight GPU health, environment metadata, and a SHA-256 manifest.
- `compare_results.py`: array-level comparison of saved MPS and CUDA prediction
  archives.
- `examples/msa_free_dimer.json`: minimal query-only JSON for the fast CLI
  gate.
- `examples/full_pipeline_5exa_ab.json`: full 5EXA A/B homodimer input with
  MSA/template fields deliberately omitted so the normal database pipeline
  performs genetic and template searches.

The old 0.9 backend pin, development build pin, standalone MSA/weight gates,
and test-fixture-only runner were intentionally removed after stable 0.10.9
passed the combined weighted gate.

## Safety and evidence

The monitor refuses to reuse an output directory. Its default absolute
process-group RSS ceiling is 32 GiB, with an early stop at 30 GiB. Unified GPU
allocations are not necessarily fully represented by process RSS, so inspect
both `run.json` and `mactop_summary.json` before increasing model size or memory
limits.

`mactop.csv` can contain process, volume, display, network, and other host
fields. Treat it as private. `mactop.selected.csv` and `mactop.process.csv` are
the reduced files intended for review or an upstream issue.

Local artifacts belong under `metal_runner/results/`, which is gitignored.
Never delete or overwrite an earlier run; use a new run name.

## Scope boundary

No Docker image is used on macOS because Linux containers do not expose the
Apple Metal GPU to JAX. The validated path installs the AF3 Python package and
chemical-component data natively, replaces the CUDA JAX backend with jax-mps,
and invokes repository scripts with that environment's Python.

The reduced weighted smoke test remains useful because it uses four diffusion
steps and can use zero recycles. The official `run_alphafold.py` path has now
also passed locally: query-only 5EXA chains A/B (230 residues each), one input
seed, 10 default recycles, 5 default diffusion samples, the normal 200
diffusion steps per sample, and an automatic 512-token bucket. It wrote the
standard AF3 output tree in 165.17 seconds, peaked at 5.57 GiB process-group
RSS, and completed without a Metal timeout or swap. See `RESULTS.md` for the
scientific and telemetry audit. CUDA agreement and broader reliability remain
open validation work.
