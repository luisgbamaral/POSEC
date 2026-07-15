"""reproduce.py — single entry point to reproduce all POSEC results.

For each experiment set it runs the POSEC calibration (run_probabilistic) and the
residual spatial diagnostics (run_spatial_diag) on the trained backbones, writing
the result CSVs + figures under ./results/.

Protocol per experiment set (the per-cell gate is always judged on a validation
block disjoint from dose selection, gate_frac=1/3):
  main / chicago (DAILY): n_his=7, chronological last-110 test / prev-110 val;
                          the 110-day validation splits ~73 dose / ~37 gate.
  weekly (7-day-ahead):   n_his=6 (monthly memory), chronological 60/20/10/10
                          (train / dose-val / gate-val / test).

Prerequisites (see README): an activated env with the deps (environment.yml;
TensorFlow needs the GPU only for --train); datasets in ./data and trained
checkpoints in ./checkpoints — unless you pass --build-data / --train.

Usage (from repo root):
  python scripts/reproduce.py                 # eval + diagnostics, all experiments
  python scripts/reproduce.py --only weekly   # one experiment set
  python scripts/reproduce.py --train         # (re)train the backbones first (GPU)
  python scripts/reproduce.py --build-data    # (re)build the datasets first
  python scripts/reproduce.py --no-diag       # skip the spatial diagnostics
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable   # the activated env's python (GPU DLLs already on its PATH)
# every subprocess (train + eval) must resolve `import posec`
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH", "")]))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# experiment set -> config. split is ("abs", n_val, n_test) or ("frac", val_frac, test_frac).
EXPERIMENTS = {
    "main":    dict(datasets=["SP_CRIME", "POA_CRIME", "BA_LESIONES"],
                    n_his=7, split=("abs", 110, 110), gate_frac=1.0 / 3.0, out="results/probabilistic"),
    "chicago": dict(datasets=["CHI_CRIME"],
                    n_his=7, split=("abs", 110, 110), gate_frac=1.0 / 3.0, out="results/chi_daily"),
    "weekly":  dict(datasets=["SP_CRIME_7D", "POA_CRIME_7D", "BA_LESIONES_7D", "CHI_CRIME_7D"],
                    n_his=6, split=("frac", 0.30, 0.10), gate_frac=1.0 / 3.0, out="results/all_7d"),
}
_CSVS = ["als_master.csv", "gw_dm_tests.csv", "calibration.csv", "per_node.csv", "spatial_diag.csv"]


def _v(ds):
    return ROOT / "data" / f"{ds}_V.csv"


def n_route(ds):
    with open(_v(ds)) as f:
        return len(f.readline().split(","))


def n_steps(ds):
    with open(_v(ds)) as f:
        return sum(1 for _ in f)


def split_days(ds, split):
    """(n_val, n_test) in time steps; 'frac' splits are per-dataset (series lengths differ)."""
    kind, a, b = split
    if kind == "abs":
        return int(a), int(b)
    T = n_steps(ds)
    return int(round(a * T)), int(round(b * T))


def run(cmd, env=None):
    print(">>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, **(env or {})}, check=False)


def train_backbones(ds, n_his, n_val, n_test, epochs):
    """Train STGCN + Graph-WaveNet + STHSL on one dataset (size-aware memory settings)."""
    N = n_route(ds)
    big = N > 500                                   # SP / Chicago: low-memory settings
    common = ["--dataset", ds, "--n_route", str(N), "--n_his", str(n_his),
              "--batch_size", "8" if big else "16", "--epoch", str(epochs),
              "--save", str(min(100, epochs)),
              "--n_val_days", str(n_val), "--n_test_days", str(n_test)]
    run([PY, "scripts/train_stgcn.py", *common, "--n_pred", "1", *(["--small_model"] if big else [])])
    run([PY, "scripts/train_gwavenet.py", *common, "--n_pred", "1", "--loss", "mse",
         *(["--res_ch", "16", "--skip_ch", "64", "--end_ch", "128"] if big else [])])
    run([PY, "scripts/train_sthsl.py", *common, "--loss", "mse"])


def _merge_parts(parts, outdir):
    """Concatenate the per-group CSVs into outdir and collect the figures."""
    for fn in _CSVS:
        dfs = [pd.read_csv(p / fn) for p in parts if (p / fn).exists()]
        if dfs:
            pd.concat(dfs, ignore_index=True).to_csv(outdir / fn, index=False)
    figdst = outdir / "figs"; figdst.mkdir(exist_ok=True)
    for p in parts:
        if (p / "figs").exists():
            for f in (p / "figs").glob("*"):
                shutil.move(str(f), str(figdst / f.name))
        if (p / "README.md").exists() and not (outdir / "README.md").exists():
            shutil.copyfile(p / "README.md", outdir / "README.md")
    for p in parts:
        shutil.rmtree(p, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Reproduce all POSEC results.")
    ap.add_argument("--only", choices=list(EXPERIMENTS), help="run a single experiment set")
    ap.add_argument("--train", action="store_true", help="(re)train the backbones first (GPU)")
    ap.add_argument("--build-data", action="store_true", help="(re)build the datasets first")
    ap.add_argument("--no-diag", action="store_true", help="skip the residual spatial diagnostics")
    ap.add_argument("--epochs", type=int, default=300, help="training epochs (with --train)")
    args = ap.parse_args()

    if args.build_data:
        run([PY, "data_prep/prepare_chicago.py"])
        run([PY, "data_prep/make_weekly.py"])

    for name in ([args.only] if args.only else list(EXPERIMENTS)):
        E = EXPERIMENTS[name]
        present = [ds for ds in E["datasets"] if _v(ds).exists()]
        if not present:
            print(f"[skip] {name}: no datasets present"); continue
        print(f"\n########## {name}  (n_his={E['n_his']}, gate_frac={E['gate_frac']:.3f})  ->  {E['out']} ##########", flush=True)
        splits = {ds: split_days(ds, E["split"]) for ds in present}
        if args.train:
            for ds in present:
                nv, nt = splits[ds]
                train_backbones(ds, E["n_his"], nv, nt, args.epochs)
        # datasets that share a (n_val, n_test) can be scored together; others need
        # separate runs (run_probabilistic reads one split from the env) -> merge after.
        groups = {}
        for ds in present:
            groups.setdefault(splits[ds], []).append(ds)
        outdir = ROOT / E["out"]; outdir.mkdir(parents=True, exist_ok=True)
        parts = []
        for gi, ((nv, nt), dss) in enumerate(groups.items()):
            tmp = outdir / f"_g{gi}"
            env = {"POSEC_CITIES": ",".join(f"{ds}:{n_route(ds)}" for ds in dss),
                   "POSEC_NVAL": str(nv), "POSEC_NTEST": str(nt),
                   "POSEC_GATE_FRAC": str(E["gate_frac"]),
                   "POSEC_OUT": str(tmp), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
            run([PY, "scripts/run_probabilistic.py"], env)
            if not args.no_diag:
                run([PY, "scripts/run_spatial_diag.py"], env)
            parts.append(tmp)
        _merge_parts(parts, outdir)

    print("\nDone. Results under ./results/.")


if __name__ == "__main__":
    main()
