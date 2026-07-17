# Experimental Apple Silicon GPU runners

These runners reproduce the Apple Silicon GPU feasibility gates without
modifying AlphaFold 3 model code. They are diagnostic tools, not a statement
that MPS inference is supported or scientifically validated. The current
baseline reaches the GPU but does not complete weighted inference; see
[`RESULTS.md`](RESULTS.md).

## Layout

- `backend_gate.py`: reports dependency versions and tests MPS discovery, JIT,
  transfers, BF16 matmul, `lax.scan`, and `lax.while_loop`.
- `no_msa_gate.py`: constructs and featurises a two-protein query-only MSA input
  without sequence databases.
- `weight_load_gate.py`: loads licensed parameters onto `MPS:0` without running
  inference.
- `weighted_fixture_smoke.py`: runs a reduced weighted test using AF3's trusted
  150-token fixture.
- `weighted_no_msa_smoke.py`: runs the smallest database-free weighted dimer
  used in the initial investigation.
- `run_with_mactop.py`: records workload output, per-process RSS, and `mactop`
  CSV samples requested at a one-second interval. It terminates the workload if
  process RSS exceeds the configured safety limit.
- `requirements/`: exact backend triplets for the known baseline and next A/B
  candidate. These files intentionally do not install AF3 or its licensed
  weights.

All scripts emit JSON progress records so results from different dependency
stacks can be compared mechanically. Run each Metal watchdog experiment in a
fresh process.

## Inputs and safety

Set the model directory to the directory containing `af3.bin.zst`, not the
compressed file itself:

```bash
export AF3_MODEL_DIR=/path/to/weights
```

The monitor refuses to reuse an output directory, preserving earlier evidence.
Its default process-RSS limit is 32 GiB. GPU allocations use unified memory and
may not all be attributed to process RSS, so also inspect `mactop.csv` before
increasing token counts or model settings.

## Recommended gate order

Run the database-free featurisation gate first:

```bash
python metal_runner/no_msa_gate.py
```

Run hardware gates under monitoring:

```bash
python metal_runner/run_with_mactop.py \
  --output-dir metal_runner/results/jax-0.9-backend \
  --max-rss-gib 32 \
  -- python metal_runner/backend_gate.py

python metal_runner/run_with_mactop.py \
  --output-dir metal_runner/results/jax-0.9-weights \
  --max-rss-gib 32 \
  -- python metal_runner/weight_load_gate.py
```

The weighted runners are expected to abort with a Metal watchdog timeout on the
recorded 0.9 baseline. They should only be retried intentionally:

```bash
python metal_runner/run_with_mactop.py \
  --output-dir metal_runner/results/jax-0.10-no-msa \
  --max-rss-gib 32 \
  -- python metal_runner/weighted_no_msa_smoke.py
```

Use `--help` on every script for configurable sequences, bucket, attention,
recycles, diffusion steps, fixture path, and monitoring delays.

## Next JAX A/B environment

Keep the working JAX 0.9.1/jax-mps 0.9.13 environment unchanged. Create a
separate Python 3.13 environment, install AF3 and its normal dependencies, then
override only the backend triplet with
`requirements/backend-jax-0.10.txt`. AF3 currently pins JAX 0.9.1, so this is an
intentional compatibility experiment rather than a supported dependency update.

For an interpretable comparison:

1. Keep `JAX_MPS_ASYNC_DISPATCH` disabled initially.
2. Run `backend_gate.py`, `no_msa_gate.py`, and `weight_load_gate.py`.
3. Run the exact 18-token weighted MSA-free case with the same seed and config.
4. If it completes, increase diffusion steps before increasing token count.
5. If it still times out, capture a Metal trace and isolate target embedding,
   one Evoformer recycle, one diffusion step, and one confidence sample before
   attempting invasive layer-stack changes.

For later numerical validation, run Linux CUDA and MPS against the same
pipeline-augmented `*_data.json`, seed, weights, and model configuration.
