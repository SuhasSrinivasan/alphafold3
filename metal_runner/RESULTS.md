# Apple Silicon GPU feasibility results

## Decision

Use stable jax-mps 0.10.9 with JAX/jaxlib 0.10.2 for the next AlphaFold 3
campaign. Reduced weighted AF3 inference completed on an Apple M4 Max, whereas
the same cases timed out on jax-mps 0.9.13. This proves technical feasibility
for the tested model graphs. A complete MSA-free `run_alphafold.py` inference
also completed for the 5EXA A/B homodimer with the current default recycles and
samples. This does not yet establish production reliability or numerical
equivalence with the supported CUDA path.

## Controlled results

Both environments used Python 3.13.14, Tokamax 0.0.11, dm-haiku 0.0.16, the
same AF3 source commit and licensed parameters, and portable XLA attention.
`mactop` 2.1.5 sampled at a requested one-second interval.

| Gate | JAX 0.9.1 / jax-mps 0.9.13 | JAX 0.10.2 / jax-mps 0.10.9 |
|---|---:|---:|
| MPS discovery, JIT, transfers, BF16 | Pass | Pass |
| `lax.scan` and `lax.while_loop` | Pass | Pass |
| 18-token, two-chain, database-free featurisation | Pass | Pass |
| 405 parameter arrays (1,146,752,808 bytes) on `MPS:0` | Pass | Pass |
| 18-token weighted dimer; zero recycles; four diffusion steps | Metal watchdog timeout | Pass |
| Trusted 150-token fixture; one recycle; four diffusion steps | Metal watchdog timeout | Pass |
| Fresh-process MPS health after every monitored run | Pass | Pass |

The old failures emitted `kIOGPUCommandBufferCallbackErrorTimeout`. They were
not memory-limit events: peak process RSS was about 5.78 GiB, swap was zero,
thermals were nominal, and a fresh BF16 MPS operation passed after each failed
process.

With stable 0.10.9, one corrected dimer run completed in 18.017 seconds; a
later cache-warm run completed in 2.083 seconds and must not be treated as a
controlled benchmark. The trusted fixture completed in 5.646 seconds. Peak
process-group RSS was approximately 5.91 GiB for the dimer and 2.61 GiB for the
fixture, with zero swap and nominal thermals.

Telemetry confirmed Apple GPU execution. During the saved stable dimer run,
GPU frequency reached 994 MHz, combined DRAM bandwidth reached 104.91 GB/s,
and the process-attributed metric peaked at 473.76 GPU-ms/s.

## End-to-end CLI validation

Two native `run_alphafold.py` runs used RCSB 5EXA chains A and B with empty
paired/unpaired MSAs, no templates, `--norun_data_pipeline`, MPS, and portable
XLA attention.

The initial 100+100-residue truncation passed the CLI and output writer with
one recycle and one sample, but was a structural negative control: ipTM 0.19,
pTM 0.32, complex C-alpha RMSD 17.82 angstrom, and none of the 27 reference
8-angstrom C-alpha interface contacts recovered. Truncating the 14-3-3 fold at
residue 100 removed essential structural context.

The full 230+230-residue test used one JSON model seed and deliberately omitted
the recycle, sample, and bucket flags. The checked-out AF3 defaults therefore
applied: 10 recycles, 5 diffusion samples, 200 diffusion steps per sample, and
an automatic 512-token bucket. The run:

- completed with workload exit code zero in 165.17 seconds;
- wrote all five sample structures and the standard top-level AF3 files;
- peaked at 5,981,798,400 bytes (5.57 GiB) process-group RSS with zero swap;
- had no Metal timeout or RSS-limit event and passed the postflight MPS gate;
- recorded 130 process-attributed GPU-active samples, up to 941.65 GPU-ms/s,
  100% system GPU use, 1578 MHz, and 332.69 GB/s combined DRAM bandwidth; and
- produced five complex C-alpha RMSDs of 0.90--2.51 angstrom to the crystal
  homodimer, with all five recovering all 27 reference interface contacts.

AF3 ranked the lowest-RMSD sample first. That sample had ranking score 0.7368,
ipTM 0.73, pTM 0.78, and symmetry-aware complex C-alpha RMSD 0.90 angstrom over
459 matched atoms. Its top-level model and confidence files are byte-identical
to the corresponding sample-4 files.

The sustained full run reached a reported GPU temperature of 100.78 C;
`mactop` recorded nominal, moderate, and heavy thermal states. The workload
completed and the host later returned to nominal at approximately 41 C, but
long campaigns should include cooldown checks and stop for persistent heavy
thermal state or throttling.

## Bounded timeout conclusion

High confidence: the old timeout came from behavior in the earlier
jax-mps/embedded-MLX execution path, not from AF3 weights, missing MSA data,
input length, the 32 GiB safety policy, or an intrinsic inability to execute
AF3 on Metal. The inputs, weights, device, AF3 configuration, and attention
implementation were held fixed while only the JAX backend stack changed.

The exact correcting commit was not proven because stable inference passed and
a version bisect or faulting Metal trace was no longer necessary. Public
control-flow changes relevant to AF3's repeated Haiku stacks include:

- [PR #137](https://github.com/tillahoffmann/jax-mps/pull/137), which reverted
  an initial while-loop implementation because of deadlocks;
- [PR #142](https://github.com/tillahoffmann/jax-mps/pull/142), which introduced
  a bounded synchronous replacement;
- [PR #176](https://github.com/tillahoffmann/jax-mps/pull/176), which addressed
  nested control-flow/event deadlocks; and
- [PR #194](https://github.com/tillahoffmann/jax-mps/pull/194), which redesigned
  counted `scan`/`fori_loop` execution.

A control-flow deadlock or pathological command-buffer construction is the
leading explanation, but attributing the fix to one change requires a backend
version bisect.

## Stable-versus-development consistency check

The exact 18-token dimer was also run once with jax-mps 0.10.10.dev800 because
it included later fused-loop and synchronization fixes. Stable 0.10.9 and that
development build produced byte-identical values for all 11 saved arrays:

- no missing or additional keys;
- zero maximum absolute or relative difference;
- identical NPZ SHA-256
  `19e082752a7a0c3ce95029ed6d5e027a1a3a8f8969a217be701481916d717dd6`;
- identical ranking score `0.5042238409028527`.

This was a narrow backend-version check. The development pin is no longer
retained because the selected campaign backend is stable 0.10.9.

## What has and has not been validated

Validated:

- native AF3 package import and compiled extensions on macOS arm64;
- generated AF3 chemical-component data;
- explicit MPS selection and portable attention validation;
- database-free AF3 input construction and featurisation;
- licensed parameter loading to MPS;
- real reduced model inference and AF3 result extraction;
- complete `run_alphafold.py` inference and standard output writing on MPS;
- one 460-residue homodimer with 10 recycles, 5 samples, and normal diffusion;
- synchronized GPU, memory, console, package, and postflight-health evidence.

Not yet validated:

- local HMMER/database pipeline operation on macOS;
- numerical agreement with a matched Linux CUDA reference;
- proteins across lengths and oligomeric states, nucleic acids, ligands,
  templates, real MSAs, multiple seeds, substantially larger inputs, and
  repeated-run reliability.

An `int64` featurisation iota was lowered to `int32` by the successful 0.10
backend. A matched CUDA run is needed before deciding whether that warning is
benign.

No issue against current stable jax-mps or MLX is justified by the resolved
reduced cases. If a production configuration times out, preserve the monitored
artifact set and then isolate target embedding, trunk/recycles, diffusion,
confidence, and distogram before filing upstream.
