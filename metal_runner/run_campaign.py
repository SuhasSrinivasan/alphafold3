#!/usr/bin/env python3
"""Autonomous AF3-on-MPS benchmark campaign orchestrator.

Consumes a manifest of targets and runs, per target, two monitored stages:
  A) data pipeline  (MSA + templates)  -- CPU, auto-tuned HMMER threads, blind
     --max_template_date, high RSS cap
  B) inference      (5 seeds)          -- MPS/Metal GPU, xla attention, low RSS cap

Every stage is wrapped in run_with_mactop.py (CPU/GPU/mem/temp telemetry + RSS cap +
postflight MPS health + SHA-256 manifest). The orchestrator is:
  * resumable  -- skips targets already completed (status file + output existence),
                  retries an incomplete target with a fresh evidence dir suffix;
  * durable    -- predictions/evidence live on disk under --pred-root/--evidence-root;
                  a status JSON + append-only log are updated after every stage;
  * fault-tolerant -- a per-target failure is logged and the campaign continues;
  * template-contrast aware -- for flagged targets it also runs a no-template inference
                  from a stripped copy of the processed JSON.

Manifest schema (JSON): {"targets": [
  {"name","type","pdb_id","input_json_path","max_template_date",
   "reference_pdb_id","molecule_types":[...],"template_contrast":bool,
   "token_estimate":int}]}
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time


def _now() -> str:
  # Wall-clock stamp; campaign runs are long so second precision is plenty.
  return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class Campaign:
  def __init__(self, args):
    self.a = args
    self.repo = pathlib.Path(args.repo)
    self.manifest = json.loads(pathlib.Path(args.manifest).read_text())["targets"]
    self.status_path = pathlib.Path(args.status_file)
    self.log_path = pathlib.Path(args.log_file)
    self.status = self._load_status()
    self.py = sys.executable
    self.threads = self._autotune_threads()

  # ---- persistence -------------------------------------------------------
  def _load_status(self) -> dict:
    if self.status_path.exists():
      return json.loads(self.status_path.read_text())
    return {"started_at": _now(), "targets": {}}

  def _save_status(self) -> None:
    tmp = self.status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(self.status, indent=2))
    tmp.replace(self.status_path)

  def log(self, msg: str) -> None:
    line = f"[{_now()}] {msg}"
    print(line, flush=True)
    with self.log_path.open("a") as fh:
      fh.write(line + "\n")

  # ---- helpers -----------------------------------------------------------
  def _autotune_threads(self) -> int:
    try:
      out = subprocess.check_output(
          [self.py, str(self.repo / "metal_runner" / "msa_cpu_autotune.py")]
      )
      return int(out.decode().strip())
    except Exception:
      return 4

  def _run_monitored(self, evidence_dir: pathlib.Path, max_rss: int,
                     term_rss: int, workload: list[str]) -> int:
    """Invoke run_with_mactop around a workload; return workload exit code proxy."""
    cmd = [
        self.py, str(self.repo / "metal_runner" / "run_with_mactop.py"),
        "--output-dir", str(evidence_dir),
        "--max-rss-gib", str(max_rss),
        "--terminate-rss-gib", str(term_rss),
        "--",
    ] + workload
    self.log("  $ " + " ".join(workload))
    proc = subprocess.run(cmd)
    return proc.returncode

  def _evidence_dir(self, name: str, tag: str) -> pathlib.Path:
    """Fresh, non-colliding evidence dir (run_with_mactop refuses to reuse)."""
    base = pathlib.Path(self.a.evidence_root) / f"{name}-{tag}"
    d, n = base, 1
    while d.exists():
      n += 1
      d = pathlib.Path(f"{base}-retry{n}")
    return d

  # ---- stages ------------------------------------------------------------
  def _stage_data(self, t: dict) -> tuple[int, str]:
    name = t["name"]
    out_root = pathlib.Path(self.a.pred_root) / f"{name}-data"
    processed = out_root / self._job_name(t) / f"{self._job_name(t)}_data.json"
    if processed.exists():
      self.log(f"  [data] already present: {processed}")
      return 0, str(processed)
    hb = pathlib.Path(self.a.hmmer_root) / "bin"
    workload = [
        self.py, str(self.repo / "run_alphafold.py"),
        "--json_path", t["input_json_path"],
        "--output_dir", str(out_root),
        "--db_dir", self.a.db_dir,
        "--run_data_pipeline", "--norun_inference",
        "--jackhmmer_binary_path", str(hb / "jackhmmer"),
        "--nhmmer_binary_path", str(hb / "nhmmer"),
        "--hmmalign_binary_path", str(hb / "hmmalign"),
        "--hmmsearch_binary_path", str(hb / "hmmsearch"),
        "--hmmbuild_binary_path", str(hb / "hmmbuild"),
        "--jackhmmer_n_cpu", str(self.threads),
        "--nhmmer_n_cpu", str(self.threads),
        "--max_template_date", t["max_template_date"],
    ]
    ev = self._evidence_dir(name, "data")
    rc = self._run_monitored(ev, self.a.max_rss_data, self.a.term_rss_data, workload)
    ok = rc == 0 and processed.exists()
    return (0 if ok else (rc or 1)), str(processed)

  def _stage_infer(self, t: dict, processed_json: str, tag: str,
                   out_name: str) -> int:
    out_dir = pathlib.Path(self.a.pred_root) / out_name
    workload = [
        self.py, str(self.repo / "run_alphafold.py"),
        "--json_path", processed_json,
        "--output_dir", str(out_dir),
        "--model_dir", self.a.model_dir,
        "--norun_data_pipeline", "--run_inference",
        "--jax_backend", "mps", "--gpu_device", "0",
        "--flash_attention_implementation", "xla",
        "--num_diffusion_samples", str(self.a.num_diffusion_samples),
        "--jax_compilation_cache_dir", self.a.jax_cache,
    ]
    ev = self._evidence_dir(t["name"], tag)
    return self._run_monitored(ev, self.a.max_rss_inf, self.a.term_rss_inf, workload)

  def _job_name(self, t: dict) -> str:
    # AF3 sanitises the JSON "name"; our input builder sets name == t["name"].
    return t["name"]

  def _strip_templates(self, processed_json: str) -> str:
    """Write a no-template copy of the processed JSON for template contrast."""
    p = pathlib.Path(processed_json)
    data = json.loads(p.read_text())
    for s in data.get("sequences", []):
      if "protein" in s:
        s["protein"]["templates"] = []
    out = p.with_name(p.stem + "_notmpl.json")
    out.write_text(json.dumps(data))
    return str(out)

  # ---- driver ------------------------------------------------------------
  def run(self) -> None:
    self.log(f"=== campaign start: {len(self.manifest)} targets, "
             f"threads/search={self.threads} ===")
    for t in self.manifest:
      name = t["name"]
      st = self.status["targets"].setdefault(name, {"state": "pending"})
      if st.get("state") == "done":
        self.log(f"[{name}] skip (done)")
        continue
      self.log(f"[{name}] type={t.get('type')} tokens={t.get('token_estimate')} START")
      st.update(state="running", started_at=_now(), type=t.get("type"))
      self._save_status()
      try:
        # Stage A: data pipeline
        rc, processed = self._stage_data(t)
        st["processed_json"] = processed
        if rc != 0:
          st.update(state="failed", stage="data", rc=rc, ended_at=_now())
          self.log(f"[{name}] DATA FAILED rc={rc}")
          self._save_status()
          continue
        st["data_done_at"] = _now()
        self._save_status()
        # Stage B: default (with-template) inference
        rc = self._stage_infer(t, processed, "infer", f"{name}-infer")
        st["infer_rc"] = rc
        if rc != 0:
          st.update(state="failed", stage="infer", rc=rc, ended_at=_now())
          self.log(f"[{name}] INFER FAILED rc={rc}")
          self._save_status()
          continue
        # Optional: template-contrast (no-template) inference
        if t.get("template_contrast"):
          try:
            notmpl = self._strip_templates(processed)
            rc2 = self._stage_infer(t, notmpl, "infer-notmpl", f"{name}-infer-notmpl")
            st["infer_notmpl_rc"] = rc2
            self.log(f"[{name}] template-contrast rc={rc2}")
          except Exception as ex:  # noqa: BLE001
            self.log(f"[{name}] template-contrast ERROR: {ex}")
        st.update(state="done", ended_at=_now())
        self.log(f"[{name}] DONE")
      except Exception as ex:  # noqa: BLE001
        st.update(state="failed", stage="exception", error=str(ex), ended_at=_now())
        self.log(f"[{name}] EXCEPTION: {ex}")
      self._save_status()
    done = sum(1 for v in self.status["targets"].values() if v.get("state") == "done")
    failed = sum(1 for v in self.status["targets"].values() if v.get("state") == "failed")
    self.status["finished_at"] = _now()
    self._save_status()
    self.log(f"=== campaign end: {done} done, {failed} failed, "
             f"{len(self.manifest)} total ===")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--manifest", required=True)
  ap.add_argument("--status-file", required=True)
  ap.add_argument("--log-file", required=True)
  ap.add_argument("--repo", default="/Users/suhas/repositories/alphafold3")
  ap.add_argument("--evidence-root", required=True)
  ap.add_argument("--pred-root", required=True)
  ap.add_argument("--model-dir", default="/Users/suhas/projects/alphafold/weights")
  ap.add_argument("--db-dir", default="/Users/suhas/projects/alphafold/databases")
  ap.add_argument("--hmmer-root",
                  default="/Users/suhas/projects/alphafold/af3-tools/hmmer-3.4")
  ap.add_argument("--jax-cache", default="/Users/suhas/projects/alphafold/jax_cache")
  ap.add_argument("--num-diffusion-samples", type=int, default=1,
                  dest="num_diffusion_samples",
                  help="Diffusion samples per seed (5 JSON seeds x this = structures/target)")
  ap.add_argument("--max-rss-data", type=int, default=160, dest="max_rss_data")
  ap.add_argument("--term-rss-data", type=int, default=155, dest="term_rss_data")
  ap.add_argument("--max-rss-inf", type=int, default=32, dest="max_rss_inf")
  ap.add_argument("--term-rss-inf", type=int, default=30, dest="term_rss_inf")
  args = ap.parse_args()
  for d in (args.evidence_root, args.pred_root):
    pathlib.Path(d).mkdir(parents=True, exist_ok=True)
  Campaign(args).run()


if __name__ == "__main__":
  main()
