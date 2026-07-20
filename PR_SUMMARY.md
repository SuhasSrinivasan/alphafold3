# Add experimental Apple Silicon GPU (Metal / MPS) inference support

## Summary

This PR enables AlphaFold 3 **inference on the Apple Silicon integrated GPU** via the community [`jax-mps`](https://pypi.org/project/jax-mps/) Metal plugin. It is an **opt-in, experimental** path: the default backend stays `gpu` (NVIDIA CUDA), and the CUDA code path is untouched. It complements the existing CPU-only macOS mode by giving Mac users a much faster GPU option for small/medium structures. All changes are additive and gated behind a new flag.

> **Native build — no Docker.** Docker Desktop on macOS runs a Linux VM that cannot access the Metal GPU, so the containerized recipe is CPU-only on Mac by construction. This path builds AlphaFold 3 **natively** (the same `scikit-build`/CMake build already supported on `darwin arm64`, plus `build_data`) and installs the MPS backend directly — no Docker required. See `docs/installation_apple_silicon_gpu.md`.

> Motivation: the current docs state Mac users must fall back to CPU because "jax-metal is unfinished." `jax-mps` is a different, working Metal plugin — this PR wires it in and backs it with a 52-target validation campaign (below).

## What changed (5 files, +650 / −56)

| File | Change |
|---|---|
| `run_alphafold.py` | New `--jax_backend {gpu,mps}` (default `gpu`); new `xla_chunked` attention option; device selection/validation refactored into `_select_inference_device` / `_validate_inference_device` |
| `run_alphafold_device_test.py` | **New** — 31 unit tests for backend selection + validation |
| `requirements/apple-silicon-gpu.txt` | **New** — pinned Metal backend stack |
| `docs/installation_apple_silicon_gpu.md` | **New** — native (non-Docker) macOS GPU install + run guide |
| `docs/installation.md` | +7 lines — pointer to the new guide from the macOS section |

## Backward compatibility (nothing NVIDIA changes)

| Aspect | Status |
|---|---|
| Default backend | `gpu` (CUDA) — unchanged |
| CUDA compute-capability checks (≥6.0, 7.x `XLA_FLAGS` workaround) | Preserved, byte-for-byte behavior |
| Model / weights / data pipeline | Untouched |
| `triton` / `cudnn` attention | Unchanged; still required on NVIDIA |
| Unit tests | 31/31 pass, incl. explicit CUDA-path-preservation cases |

## New flags / usage

| Flag | Values | Notes |
|---|---|---|
| `--jax_backend` | `gpu` (default), `mps` | `mps` requires `xla`/`xla_chunked`; `--use_cpu_only` overrides |
| `--flash_attention_implementation` | `triton`, `cudnn`, `xla`, **`xla_chunked`** (new) | `xla_chunked` = portable, linear-memory, slower |

MPS invocation: `--jax_backend=mps --flash_attention_implementation=xla`

## Backend dependency stack

Installed as a post-install override of the base `jax==0.9.1` pin (one expected `pip check` warning); no change to `uv.lock` or the default install.

| Package | Version |
|---|---|
| jax | 0.10.2 |
| jaxlib | 0.10.2 |
| jax-mps | 0.10.9 |
| Python | 3.11–3.13 (tested 3.13) |

## Validation campaign

| Item | Value |
|---|---|
| Hardware | M2 Ultra Mac Studio — 76-GPU-core, 192 GB unified memory, macOS 15 |
| Targets | 52 (all 12 molecule-type classes × 4 + 4 token-scaling 1.5k–2.2k) |
| Pipeline | Full HMMER MSA + templates → 5-seed MPS inference |
| Result | 52/52 completed, 0 failures, 40.7 h wall-time, 0 timeouts / OOM-kills |
| Reference | Verified vs experimental PDB structures |

## Accuracy (best of 5 seeds, vs PDB)

**Overall (n = 52)**

| Metric | Result |
|---|---|
| Per-chain Cα RMSD (fold quality) | median **1.23 Å** — 21 sub-1 Å, **42 sub-2 Å**, 46 sub-3 Å |
| Complex Cα RMSD (quaternary) | median 1.62 Å |
| Ligand RMSD (pocket-aligned) | median **0.32 Å** |
| Nucleic-acid backbone RMSD | median **0.94 Å** |
| AF3 `has_clash` | **0 / 52** |
| pTM / ipTM | median 0.88 / 0.87 |
| Confidence calibration (pTM vs complex RMSD) | **r = −0.72** |
| Determinism (same-seed re-run) | **bit-identical, 0.000 Å** |

**By molecule type** (median). Read *per-chain* as fold quality; a high *complex* value with low per-chain = quaternary arrangement of a flexible assembly, not a fold error.

| Type | n | per-chain Cα (Å) | complex Cα (Å) | DockQ | Note |
|---|--:|--:|--:|--:|---|
| hetero-trimer | 4 | 0.56 | 1.65 | 0.96 | |
| protein-DNA-ligand | 4 | 0.70 | 0.79 | 0.95 | |
| polymer-ligand | 4 | 0.81 | 0.93 | 0.88 | |
| protein-RNA | 4 | 0.94 | 12.28 | 0.05 | RNA placement harder (subunits fold well) |
| hetero-dimer | 4 | 0.99 | 1.51 | 0.89 | |
| protein-protein-ligand-ion | 4 | 1.18 | 7.38 | 0.41 | |
| monomer-ligand | 4 | 1.23 | 1.23 | – | |
| homo-polymer | 4 | 1.28 | 16.79 | 0.03 | quaternary/oligomer arrangement |
| protein-RNA-ligand | 4 | 1.33 | 1.32 | 0.50 | |
| hetero-tetramer | 4 | 1.48 | 16.34 | 0.79 | quaternary |
| protein-DNA | 4 | 1.93 | 1.93 | – | |
| large (1.5–2.2k tok) | 4 | 2.10 | 9.48 | 0.86 | |
| monomer | 4 | 11.18 | 11.18 | – | 2/4 genuine misses, both low-pTM (AF3-flagged) |

*Highlight:* nitrogenase `3U7Q` (2,177 tokens, FeMoco metal cluster) — per-chain 0.42 Å, complex 0.47 Å, DockQ 0.98, ligand 0.13 Å, pTM 0.96.

## Performance and memory (M2 Ultra, 192 GB)

| Padded tokens | MSA time¹ | Inference / seed² | Peak unified memory (GB)² |
|--:|--:|--:|--:|
| 256 | ~7 min | ~29 s | 36 |
| 512 | ~13 min | ~90 s | 50 |
| 768 | ~13 min | ~3.9 min | 73 |
| 1,024 | ~7 min | ~8.6 min | 101 |
| 1,536 | ~19 min | ~26.6 min | 155 |
| ~2,180 | ~20 min | ~3.9 h | 167 |

¹ MSA / data pipeline is CPU-bound and driven by the number of **unique chains and RNA presence, not token count** — hence non-monotonic across sizes (observed range 7–35 min). ² Inference time and peak memory are set by the padded token bucket. Inference scales ~polynomially (exponent ~1.7 → ~2.8) with a steeper Metal-specific penalty above ~1,536 tokens; unified memory (not process RSS) is the capacity limit — practical ceiling ≈ 2,200 tokens on 192 GB.

## Known limitations

| Limitation | Detail |
|---|---|
| Experimental backend | `jax-mps` Metal plugin; not numerically certified |
| Attention | Only `xla` / `xla_chunked` on MPS (`triton`/`cudnn` are NVIDIA-only) |
| JAX version | MPS needs jax 0.10.2 vs the base 0.9.1 pin (documented override) |
| Compile cache | `--jax_compilation_cache_dir` does not persist on Metal |
| Memory / speed | ~2,200-token ceiling on 192 GB; steep time cost above ~1,500 tokens |

## Test plan

```sh
# CPU-only, no GPU needed — verifies selection logic + CUDA-path preservation
python run_alphafold_device_test.py        # 31 tests

# On Apple Silicon (native build, no Docker), after installing the MPS backend:
python run_alphafold.py --jax_backend=mps --flash_attention_implementation=xla ...
```
