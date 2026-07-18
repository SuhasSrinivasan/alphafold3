# AI-agent runbook: native AF3 on Apple Silicon GPU

This runbook is for a fresh Apple Silicon Mac and the `codex/review-1` branch.
It selects stable jax-mps 0.10.9. The backend is experimental and the first
priority is to preserve reproducible evidence, not to optimize performance.

Treat the exact checkout as the source of truth. This native recipe maps the
official [installation guide](../docs/installation.md),
[Dockerfile](../docker/Dockerfile), [Python build metadata](../pyproject.toml),
[database fetcher](../fetch_databases.sh), and [input format](../docs/input.md)
onto macOS, changing only the container/CUDA backend and paths required for
Metal. Re-audit those files before applying the recipe to a newer AF3 commit.

## 1. Operating rules

An AI agent following this document must:

1. Ask for explicit approval before installing or upgrading any package,
   program, library, or environment.
2. Never use `sudo` and never delete or overwrite an existing environment,
   output directory, prediction, or diagnostic artifact.
3. Record the exact commit, host metadata, dependency versions, command, and
   exit status for every weighted run.
4. Keep the model weights private. The expected input is a licensed
   `af3.bin.zst`; it does not need to be decompressed.
5. Start inference with the 32 GiB monitor policy below. A full genetic-search
   pipeline can legitimately require more memory; obtain explicit approval for
   both its higher hard and early-stop limits before launching it.
6. Stop after a Metal timeout, an RSS-limit event, or a failed postflight MPS
   health check. Inspect the evidence before starting another run.

## 2. What replaces the Docker image

The official image is a reproducible Linux/NVIDIA bundle, not a requirement of
the AF3 Python program. Linux containers on Docker Desktop do not expose the
Apple Metal GPU, so MPS inference must run natively on macOS.

| Official image component | Native macOS equivalent used here |
|---|---|
| Ubuntu and system Python | macOS arm64 and an isolated Conda Python 3.13 environment |
| CUDA base image and CUDA JAX plugin | JAX 0.10.2, jaxlib 0.10.2, and jax-mps 0.10.9 |
| GCC/CMake/Ninja build layer | Apple Command Line Tools/Clang plus PEP 517 build dependencies from `pyproject.toml` |
| `uv sync` of the AF3 package | `python -m pip install <repository>` |
| `uv run build_data` | the installed `build_data` entry point |
| Patched HMMER and sequence databases | native HMMER 3.4 plus the repository patch and databases for the full pipeline; omitted only for inference-only gates |
| Mounted model directory | a native directory containing licensed `af3.bin.zst` |
| CUDA-specific XLA environment variables | omitted; portable `xla` or `xla_chunked` attention is selected |

The completed feasibility tests imported the natively installed `alphafold3`
package and used the repository's `run_alphafold.ModelRunner`. They exercised
real weights, featurisation, model execution, and inference-result extraction.
They did **not** build a container. Subsequent local testing also passed the
complete Abseil CLI/output-writing path, including a 460-residue homodimer with
default recycles and samples. Section 7 is the reproducible entry-level CLI
gate before repeating or expanding that test on another host.

## 3. Read-only preflight

First identify the host and existing tools without changing anything:

```bash
uname -m
sw_vers
sysctl -n hw.memsize
xcode-select -p
clang --version
git --version
make --version
patch --version
conda --version
mactop --version
command -v wget
command -v zstd
git status --short --branch
git rev-parse HEAD
```

Required results are `arm64`, working Apple Command Line Tools (Apple Clang,
`make`, and `patch`), a Conda-family environment manager, and the
`codex/review-1` branch. Full Xcode is not needed to build or run AF3; it is
needed only to inspect Instruments/Metal trace files.

If Conda, Command Line Tools, or `mactop` is absent, report that fact and ask
for approval for the specific installation. A common non-privileged `mactop`
installation is `brew install mactop`.

Choose absolute paths and verify them. Do not place weights or results in Git:

```bash
export AF3_REPO=/absolute/path/to/alphafold3
export AF3_MODEL_DIR=/absolute/path/to/weights
export AF3_HMMER_ROOT=/absolute/path/to/af3-tools/hmmer-3.4
export AF3_DB_DIR=/absolute/path/to/public_databases
export AF3_EVIDENCE_ROOT=/absolute/path/to/af3-metal-evidence
export AF3_PIPELINE_OUTPUT_ROOT=/absolute/path/to/af3-pipeline-outputs
export AF3_PREDICTION_ROOT=/absolute/path/to/af3-metal-predictions
test -f "$AF3_MODEL_DIR/af3.bin.zst"
```

Weights, databases, HMMER, evidence, and predictions must be outside the Git
checkout. Before writing to any selected directory, verify the exact path,
available space, ownership, and whether it is empty. Never reuse a previous
run directory.

## 4. Build AF3 and its native dependencies

### 4.1 Installation authorization

Show the user these exact mutations and wait for approval. Then create a new
environment; never repurpose a CUDA, CPU, or older jax-mps environment.

The full setup can require three separately approved installations:

1. a new Conda environment and Python packages;
2. a user-local HMMER 3.4 source build; and
3. `wget` and `zstd` command-line tools if they are absent.

Never use `sudo`. Apple Command Line Tools are a host prerequisite; if they are
missing, stop and ask the user to install them. One non-privileged way to
provide the database tools inside the approved environment is:

```bash
conda install --name af3-mps-stable --channel conda-forge wget zstd -y
```

Do not execute that command unless the user approved it.

### 4.2 Build and install the AF3 Python package

Use the intended branch and record its commit before building:

```bash
cd "$AF3_REPO"
git status --short --branch
git rev-parse HEAD
```

After installation approval, create the isolated environment and install this
checkout as a non-editable native package:

```bash
conda create --name af3-mps-stable python=3.13 pip -y
conda activate af3-mps-stable
python -m pip install "$AF3_REPO"
python -m pip install --upgrade \
  -r "$AF3_REPO/metal_runner/requirements/backend-jax-0.10.txt"
build_data
```

Why the two Python installation steps: AF3's current package metadata pins
`jax==0.9.1`. Installing AF3 normally builds its native extensions and installs
all regular dependencies; the second command deliberately replaces only the
JAX backend triplet with the tested versions. Consequently, `pip check` is
expected to report exactly one mismatch: AF3 asks for JAX 0.9.1 while JAX
0.10.2 is installed. Any other mismatch is a stop condition.

`build_data` creates the CCD and chemical-component lookup files in the
installed package and uses roughly 0.5 GiB. It is required even when sequence
databases are skipped.

The first `pip install` is also the AF3 source build. PEP 517 creates an
isolated build environment from `pyproject.toml`, requiring
`scikit-build-core`, Pybind11, CMake 3.28 or newer, Ninja, and NumPy. CMake uses
Apple Clang in C++20 mode, fetches the repository-pinned Abseil, Pybind11,
Pybind11-Abseil, libcifpp, and DSSP revisions, and builds the
`alphafold3.cpp` extension. Network access is therefore required during a
fresh source build even if Python wheels are cached.

Runtime dependencies installed from `pyproject.toml` include Abseil Python,
Haiku 0.0.16, etils, NumPy, RDKit 2025.9.4, Tokamax 0.0.11, tqdm, and
zstandard. The second installation command replaces the package's JAX 0.9.1
pin with the tested MPS triplet. Record the resolved environment after all
overrides:

```bash
python -m pip freeze
python -m pip check
```

If AF3 source changes after installation, rebuild it only after confirming the
working tree and obtaining approval for the package mutation:

```bash
python -m pip install --no-deps --force-reinstall "$AF3_REPO"
build_data
python -m pip install --upgrade \
  -r "$AF3_REPO/metal_runner/requirements/backend-jax-0.10.txt"
```

The `--no-deps` rebuild avoids silently downgrading the validated JAX stack.

### 4.3 Build the patched HMMER 3.4 suite

HMMER is not needed for `--norun_data_pipeline`, but it is required for the
normal protein/RNA MSA and protein-template pipeline. For equivalence with the
official Docker recipe, prefer a user-local source build of HMMER 3.4 with
`docker/jackhmmer_seq_limit.patch`. The patch lets Jackhmmer truncate output
before writing very large redundant alignments, reducing peak memory and I/O.
Current AF3 can run with an unpatched binary, but that is not the preferred
full-pipeline configuration.

After explicit build approval, choose a new, empty source directory outside
the repository and run:

```bash
export AF3_HMMER_SOURCE=/absolute/path/to/af3-tools/hmmer-source
test ! -e "$AF3_HMMER_SOURCE/hmmer-3.4"
mkdir -p "$AF3_HMMER_SOURCE" "$AF3_HMMER_ROOT"
cd "$AF3_HMMER_SOURCE"
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz" | shasum -a 256 -c
tar -xzf hmmer-3.4.tar.gz
patch -p0 < "$AF3_REPO/docker/jackhmmer_seq_limit.patch"
cd hmmer-3.4
./configure --prefix="$AF3_HMMER_ROOT"
make -j "$(sysctl -n hw.logicalcpu)"
make install
cd easel
make install
```

No command above requires elevated privileges when both selected directories
are user-owned. Put the built binaries first on `PATH` for every AF3 session:

```bash
export PATH="$AF3_HMMER_ROOT/bin:$PATH"
for binary in jackhmmer nhmmer hmmalign hmmsearch hmmbuild; do
  command -v "$binary"
done
jackhmmer -h
```

Confirm that `jackhmmer -h` lists `--seq_limit`. Preserve the HMMER tarball
hash, configure output, build log, binary paths, and `jackhmmer -h` in the
environment evidence. A Conda/Bioconda HMMER 3.4 package is a possible
compatibility fallback, but it normally lacks this repository patch and must
be recorded as a deviation.

## 5. Validate the installation and backend

Run these read-only checks from the repository root:

```bash
cd "$AF3_REPO"
python -m pip check
python -c "import importlib.metadata as m; print({n: m.version(n) for n in ('alphafold3', 'jax', 'jaxlib', 'jax-mps', 'tokamax', 'dm-haiku')})"
python -c "import alphafold3.cpp as c; print(c.__file__)"
python -c "import jax; print(jax.local_devices(backend='mps'))"
python -c "from alphafold3.constants import chemical_components; print(len(chemical_components.Ccd()))"
python run_alphafold.py --helpshort
```

Expected backend versions are JAX 0.10.2, jaxlib 0.10.2, and jax-mps 0.10.9.
There should be one device with platform `mps`. Leave experimental controls at
their defaults for the baseline:

```bash
unset JAX_MPS_ASYNC_DISPATCH
unset JAX_MPS_GPU_CAPTURE
unset MLX_MAX_OPS_PER_BUFFER
```

If HMMER was built, validate the included miniature-database data pipeline
before touching the 630 GB database set:

```bash
python run_alphafold_data_test.py
```

This test requires all five HMMER programs on `PATH` but does not require model
weights or the external databases. Preserve the complete test log. A failure
here is a stop condition for the full pipeline, even if inference-only gates
pass.

Use a new evidence directory name for every command. The wrapper deliberately
fails if the directory already exists.

```bash
python metal_runner/run_with_mactop.py \
  --output-dir "$AF3_EVIDENCE_ROOT/backend-gate-001" \
  --max-rss-gib 32 \
  -- python metal_runner/backend_gate.py \
  --monitor-warmup-seconds 0
```

Require a zero exit code, `rss_limit_exceeded: false`,
`metal_timeout_detected: false`, and a passing health check in `run.json`.
This gate's matrix operation can finish between one-second samples, so its
process-specific GPU counter may remain zero. Confirm the selected `MPS:0`
device in the workload log and inspect workload-versus-baseline system GPU and
DRAM fields. The longer weighted gate must show process-attributed GPU work.

## 6. Run reduced weighted inference

This is the fast gate that previously distinguished the broken 0.9 backend
from stable 0.10.9. It uses an 18-token query-only dimer, one diffusion sample,
four diffusion steps, and zero recycles.

```bash
python metal_runner/run_with_mactop.py \
  --output-dir "$AF3_EVIDENCE_ROOT/weighted-smoke-001" \
  --max-rss-gib 32 \
  -- python metal_runner/weighted_no_msa_smoke.py \
  --model-dir "$AF3_MODEL_DIR" \
  --attention xla \
  --monitor-warmup-seconds 0 \
  --result-npz "$AF3_EVIDENCE_ROOT/weighted-smoke-001/prediction.npz"
```

Do not interpret this reduced result as a scientific prediction. Its purpose
is backend compatibility and controlled comparison. Preserve the NPZ, its JSON
summary, `run.json`, both workload logs, the selected telemetry CSVs, package
metadata, and `manifest.sha256`. Require at least one process-attributed
GPU-active sample; if none is captured, repeat with a longer model setting
instead of inferring GPU use from system-wide utilization alone.

## 7. Run the actual AF3 CLI without sequence databases

`examples/msa_free_dimer.json` explicitly supplies empty paired/unpaired MSAs
and an empty template list for both proteins. This is query-only MSA inference.
`--norun_data_pipeline` prevents construction of database paths and HMMER
workers; no 600+ GiB sequence database is needed.

The official CLI enforces at least one recycle and does not expose the reduced
runner's four-step diffusion override. This gate therefore executes more model
work and writes the standard AF3 output tree.

```bash
python metal_runner/run_with_mactop.py \
  --output-dir "$AF3_EVIDENCE_ROOT/official-cli-001" \
  --max-rss-gib 32 \
  -- python run_alphafold.py \
  --json_path "$AF3_REPO/metal_runner/examples/msa_free_dimer.json" \
  --output_dir "$AF3_PREDICTION_ROOT/official-cli-001" \
  --model_dir "$AF3_MODEL_DIR" \
  --norun_data_pipeline \
  --run_inference \
  --jax_backend mps \
  --gpu_device 0 \
  --flash_attention_implementation xla \
  --buckets 18 \
  --num_recycles 1 \
  --num_diffusion_samples 1
```

Success means the wrapper and workload both exit zero, GPU telemetry is
present, the health check passes, and the prediction directory contains AF3's
mmCIF, confidence JSON, ranking CSV, and processed input JSON. This is the gate
that establishes official CLI/output-writing compatibility. It passed on the
local M4 Max and should be repeated on the remote M2 Ultra before expansion.

## 8. Repeat the full default-setting homodimer, then expand

The local full-chain 5EXA A/B test passed with one JSON seed and no explicit
`--num_recycles`, `--num_diffusion_samples`, or `--buckets` flags. On this
checkout that means 10 recycles, 5 diffusion samples, 200 diffusion steps per
sample, and the automatic 512-token bucket. Use the full 230-residue YWHAZ
sequence for two homomeric chains, with both MSA fields empty and no templates.
Do not infer that the defaults are five recycles: verify the flag defaults in
the exact checkout before every campaign.

On the local M4 Max this run completed in 165.17 seconds at 5.57 GiB peak
process-group RSS, but reached a reported GPU temperature of 100.78 C and a
heavy thermal state. Preserve the same telemetry and health evidence on the
remote host, include cooldown checks between sustained tests, and stop for a
persistent heavy state or evidence of throttling.

This query-only reproduction is the final gate before enabling the external
database pipeline.

## 9. Prepare and validate the full database set

The normal AF3 data pipeline uses these repository-pinned database snapshots:

- `bfd-first_non_consensus_sequences.fasta`;
- `mgy_clusters_2022_05.fa`;
- `uniref90_2022_05.fa`;
- `uniprot_all_2021_04.fa`;
- `pdb_seqres_2022_09_28.fasta`;
- `nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta`;
- `rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta`;
- `rnacentral_active_seq_id_90_cov_80_linclust.fasta`; and
- `mmcif_files/`, containing approximately 200,000 PDB mmCIF files.

The official download is about 252 GB compressed and 630 GB after expansion.
Use a fast local SSD and leave ample additional space for temporary files,
inputs, generated MSA JSON, predictions, logs, and filesystem overhead. On the
8 TB remote host, confirm the selected volume with `df -h` before download.

`fetch_databases.sh` requires `wget`, `tar`, and the `zstd` executable. It
starts multiple downloads and decompressions concurrently and writes directly
to the target. It does not provide a resumable transaction or an upstream
checksum manifest for all expanded files. Therefore:

1. obtain explicit approval for the approximately 630 GB write and network
   transfer;
2. use a new empty directory outside the repository;
3. preserve the complete console log; and
4. never rerun it over a partial or populated directory without explicit user
   direction.

After approval, the repository-supported download command is:

```bash
cd "$AF3_REPO"
mkdir -p "$AF3_DB_DIR"
test -z "$(ls -A "$AF3_DB_DIR")"
./fetch_databases.sh "$AF3_DB_DIR"
```

If the user populated the databases separately, do not run the fetch script.
Validate the supplied layout read-only:

```bash
for database_file in \
  bfd-first_non_consensus_sequences.fasta \
  mgy_clusters_2022_05.fa \
  uniref90_2022_05.fa \
  uniprot_all_2021_04.fa \
  pdb_seqres_2022_09_28.fasta \
  nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta \
  rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta \
  rnacentral_active_seq_id_90_cov_80_linclust.fasta; do
  test -r "$AF3_DB_DIR/$database_file" && test -s "$AF3_DB_DIR/$database_file"
done
test -d "$AF3_DB_DIR/mmcif_files"
find "$AF3_DB_DIR/mmcif_files" -type f | wc -l
du -sh "$AF3_DB_DIR"
df -h "$AF3_DB_DIR"
```

Do not respond to a permission failure with `sudo` or a broad recursive
`chmod`. Report the exact unreadable path and ask the user to correct ownership
or permissions. The database snapshots are static scientific inputs; record
their names, apparent sizes, and source script commit in the run evidence.

## 10. Run the full typical AF3 pipeline

`examples/full_pipeline_5exa_ab.json` contains the full 230+230-residue 5EXA
A/B homodimer but deliberately omits `unpairedMsa`, `pairedMsa`, and
`templates`. Omitted or explicit `null` fields instruct AF3 to search the
protein databases and templates. Empty strings plus `templates: []` do the
opposite and are only appropriate for the query-only gates.

For auditability, run the normal pipeline in two stages. This executes the same
genetic search, template search, featurisation, and model inference as a single
default CLI invocation, while preserving a reusable processed JSON and
separating CPU/database failures from MPS failures.

### 10.1 Database, MSA, and template stage

The official documentation recommends at least 64 GB RAM for genetic search.
The monitor defaults to a 30 GiB early stop, so a higher pair of limits must be
passed explicitly. On a 192 GB host, 160 GiB hard and 150 GiB early-stop limits
leave operating-system headroom, but they are examples rather than implicit
authorization: obtain the user's approval for those exact values first.

Use a new output and evidence directory:

```bash
export AF3_FULL_INPUT="$AF3_REPO/metal_runner/examples/full_pipeline_5exa_ab.json"
export AF3_DATA_RUN_ROOT="$AF3_PIPELINE_OUTPUT_ROOT/5exa-data-001"

python metal_runner/run_with_mactop.py \
  --output-dir "$AF3_EVIDENCE_ROOT/5exa-data-001" \
  --max-rss-gib 160 \
  --terminate-rss-gib 150 \
  -- python run_alphafold.py \
  --json_path "$AF3_FULL_INPUT" \
  --output_dir "$AF3_DATA_RUN_ROOT" \
  --db_dir "$AF3_DB_DIR" \
  --run_data_pipeline \
  --norun_inference \
  --jackhmmer_binary_path "$AF3_HMMER_ROOT/bin/jackhmmer" \
  --nhmmer_binary_path "$AF3_HMMER_ROOT/bin/nhmmer" \
  --hmmalign_binary_path "$AF3_HMMER_ROOT/bin/hmmalign" \
  --hmmsearch_binary_path "$AF3_HMMER_ROOT/bin/hmmsearch" \
  --hmmbuild_binary_path "$AF3_HMMER_ROOT/bin/hmmbuild" \
  --max_template_date 2021-09-30
```

AF3 defaults each HMMER program to `min(cpu_count, 8)` CPUs and launches four
protein searches concurrently. Record any CPU-count override; changing it is a
performance choice, not a scientific model change.

Require workload exit code zero, no RSS event, intact logs and manifest, and a
processed input at:

```text
$AF3_DATA_RUN_ROOT/5exa_ab_full_pipeline/5exa_ab_full_pipeline_data.json
```

Audit MSA/template presence without printing the private or potentially large
alignment strings:

```bash
export AF3_PROCESSED_JSON="$AF3_DATA_RUN_ROOT/5exa_ab_full_pipeline/5exa_ab_full_pipeline_data.json"
python - "$AF3_PROCESSED_JSON" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
print('processed_json_bytes', path.stat().st_size)
for sequence_entry in data['sequences']:
  if 'protein' not in sequence_entry:
    continue
  protein = sequence_entry['protein']
  print({
      'id': protein['id'],
      'unpaired_msa_chars': len(protein.get('unpairedMsa') or ''),
      'paired_msa_chars': len(protein.get('pairedMsa') or ''),
      'template_count': len(protein.get('templates') or []),
      'unpaired_is_null': protein.get('unpairedMsa') is None,
      'paired_is_null': protein.get('pairedMsa') is None,
  })
PY
```

Both MSA fields must no longer be `null`; template count can legitimately be
zero if every hit is filtered. Preserve this processed JSON as the exact input
for MPS and CUDA comparisons.

### 10.2 Default MPS inference stage

Run inference from the processed JSON in a new prediction directory. Omit
`--num_recycles`, `--num_diffusion_samples`, and `--buckets`; this checkout
then uses 10 recycles, 5 diffusion samples, and automatic bucketing.

```bash
python metal_runner/run_with_mactop.py \
  --output-dir "$AF3_EVIDENCE_ROOT/5exa-full-pipeline-inference-001" \
  --max-rss-gib 32 \
  --terminate-rss-gib 30 \
  -- python run_alphafold.py \
  --json_path "$AF3_PROCESSED_JSON" \
  --output_dir "$AF3_PREDICTION_ROOT/5exa-full-pipeline-inference-001" \
  --model_dir "$AF3_MODEL_DIR" \
  --norun_data_pipeline \
  --run_inference \
  --jax_backend mps \
  --gpu_device 0 \
  --flash_attention_implementation xla
```

Require workload exit code zero, no Metal timeout or RSS event, nonzero
process-attributed GPU samples, zero swap, passing postflight MPS health, and
all five sample output directories plus the ranked top-level output. Inspect
thermal history before starting another weighted run; allow the machine to
return to nominal if it entered a heavy state.

## 11. Compare, expand, and preserve provenance

After the complete 5EXA reproduction, increase one dimension at a time:
additional seeds, token count, entity diversity, real heteromers, RNA/DNA,
ligands, and repeated-run reliability. Keep the seed, weights, processed input
JSON, bucket, attention implementation, and model configuration fixed when
comparing backends. For CUDA comparison, save the same diagnostic arrays with
the reduced runner or add an equivalent non-invasive capture to the CUDA run,
then use:

```bash
python metal_runner/compare_results.py \
  /absolute/path/to/cuda-prediction.npz \
  /absolute/path/to/mps-prediction.npz \
  --output /absolute/path/to/comparison.json
```

Cross-backend results are not expected to be byte-identical. Define tolerances
and compare aligned structures, confidence values, ranking, determinism,
failure rate, runtime, and peak unified memory over a diverse test matrix
before making scientific-equivalence claims.

For every full-pipeline run, preserve the raw input, processed input, model
parameters hash or private path metadata, database snapshot names, exact Git
commit, `pip freeze`, HMMER build/hash, command, complete console streams,
`run.json`, telemetry summaries, selected telemetry CSVs, output ranking CSV,
confidence JSON, structures, postflight health result, and SHA-256 manifest.
Treat raw `mactop.csv`, model weights, generated MSAs, and biological inputs as
private unless the user explicitly approves publication.
