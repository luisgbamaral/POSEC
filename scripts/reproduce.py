"""reproduce.py — single entry point to reproduce all POSEC results.

For each experiment set it runs the POSEC calibration (run_probabilistic) and the
residual spatial diagnostics (run_spatial_diag) on the trained backbones, writing
the result CSVs + figures under ./results/.

Prerequisites (see README):
  * an activated env with the deps (environment.yml); TensorFlow needs the GPU
    only for --train (evaluation also runs on CPU, just slower).
  * datasets in ./data and trained checkpoints in ./checkpoints — both provided
    separately — unless you pass --build-data / --train.

Usage (from repo root):
  python scripts/reproduce.py                 # eval + diagnostics, all experiments
  python scripts/reproduce.py --only chicago  # one experiment set
  python scripts/reproduce.py --train         # (re)train backbones first (GPU, hours)
  python scripts/reproduce.py --build-data    # (re)build the datasets first
  python scripts/reproduce.py --no-diag       # skip the spatial diagnostics
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable   # the activated env's python (GPU DLLs already on its PATH)

# experiment set -> (datasets, n_val, n_test, output dir)
EXPERIMENTS = {
    "main":    (["SP_CRIME", "POA_CRIME", "BA_LESIONES"],                          110, 110, "results/probabilistic"),
    "chicago": (["CHI_CRIME"],                                                     110, 110, "results/chi_daily"),
    "weekly":  (["SP_CRIME_7D", "POA_CRIME_7D", "BA_LESIONES_7D", "CHI_CRIME_7D"],  16,  16, "results/all_7d"),
}


def n_route(ds):
    """Number of nodes = column count of the V.csv."""
    with open(ROOT / "data" / f"{ds}_V.csv") as f:
        return len(f.readline().split(","))


def has_data(ds):
    return (ROOT / "data" / f"{ds}_V.csv").exists()


def run(cmd, env=None):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, **(env or {})}, check=False)


def train_backbones(ds, n_val, n_test, epochs, n_his):
    """Train STGCN + Graph-WaveNet + STHSL on one dataset (size-aware memory settings)."""
    N = n_route(ds)
    big = N > 500                                   # SP / Chicago: use low-memory settings
    common = ["--dataset", ds, "--n_route", str(N), "--n_his", str(n_his),
              "--batch_size", "8" if big else "16", "--epoch", str(epochs),
              "--save", str(min(100, epochs)),
              "--n_val_days", str(n_val), "--n_test_days", str(n_test)]
    run([PY, "scripts/train_stgcn.py", *common, "--n_pred", "1", *(["--small_model"] if big else [])])
    run([PY, "scripts/train_gwavenet.py", *common, "--n_pred", "1", "--loss", "mse",
         *(["--res_ch", "16", "--skip_ch", "64", "--end_ch", "128"] if big else [])])
    run([PY, "scripts/train_sthsl.py", *common, "--loss", "mse"])


def main():
    ap = argparse.ArgumentParser(description="Reproduce all POSEC results.")
    ap.add_argument("--only", choices=list(EXPERIMENTS), help="run a single experiment set")
    ap.add_argument("--train", action="store_true", help="(re)train the backbones first (GPU, hours)")
    ap.add_argument("--build-data", action="store_true", help="(re)build the datasets first")
    ap.add_argument("--no-diag", action="store_true", help="skip the residual spatial diagnostics")
    ap.add_argument("--epochs", type=int, default=300, help="training epochs (with --train)")
    ap.add_argument("--n-his", type=int, default=7, help="history window length")
    args = ap.parse_args()

    if args.build_data:
        run([PY, "data_prep/prepare_chicago.py"])
        run([PY, "data_prep/make_weekly.py"])

    for name in ([args.only] if args.only else list(EXPERIMENTS)):
        datasets, n_val, n_test, out = EXPERIMENTS[name]
        present = [ds for ds in datasets if has_data(ds)]
        if not present:
            print(f"[skip] {name}: no datasets present"); continue
        print(f"\n########## experiment: {name}  ->  {out} ##########")
        if args.train:
            for ds in present:
                train_backbones(ds, n_val, n_test, args.epochs, args.n_his)
        env = {"POSEC_CITIES": ",".join(f"{ds}:{n_route(ds)}" for ds in present),
               "POSEC_NVAL": str(n_val), "POSEC_NTEST": str(n_test),
               "POSEC_OUT": str(ROOT / out), "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        run([PY, "scripts/run_probabilistic.py"], env)
        if not args.no_diag:
            run([PY, "scripts/run_spatial_diag.py"], env)

    print("\nDone. Results under ./results/.")


if __name__ == "__main__":
    main()
