# Apple Silicon GPU feasibility results

This ledger records controlled backend comparisons. It is not a claim that
AlphaFold 3 inference is supported on Apple Silicon GPU.

## Baseline: JAX 0.9.1 / jax-mps 0.9.13

Tested with Python 3.13.14, jaxlib 0.9.1, Tokamax 0.0.11, and dm-haiku
0.0.16. `mactop` 2.1.5 was configured with a 1000 ms sampling interval.

| Gate | Result |
|---|---|
| MPS discovery, JIT, transfers, BF16 | Pass |
| Representative AF3 modules with `xla` and `xla_chunked` | Pass |
| Two-chain, 18-token, database-free featurisation | Pass |
| 405 parameter arrays (1,146,752,808 bytes) loaded onto `MPS:0` | Pass |
| 150-token weighted fixture; one sample/recycle; four diffusion steps | Metal watchdog timeout |
| 18-token MSA-free weighted dimer; zero recycles; four diffusion steps | Metal watchdog timeout |
| MSA-free retry with `MLX_MAX_OPS_PER_BUFFER=1` | Metal watchdog timeout |

The weighted failures reported
`kIOGPUCommandBufferCallbackErrorTimeout`. Monitoring showed real GPU activity,
zero swap, nominal thermal state, and a peak process RSS of approximately
5.78 GiB. The failure was therefore not the 32 GiB experiment limit or an
out-of-memory condition. A small BF16 MPS operation passed after each failed
process, confirming that the GPU recovered.

## Candidate: JAX 0.10.2 / jax-mps 0.10.9

Status: not tested. Use a separate environment and keep async dispatch disabled
for the first comparison. Run the same scripts, input, seed, attention setting,
and reduced model configuration before changing any other variable.
