"""
make_weekly.py
--------------
Build 7-day (weekly) aggregated variants of the crime datasets, for the
"predict next week as a single step" experiment. Non-overlapping 7-day blocks
of daily counts are SUMMED into a weekly series; the spatial graph is unchanged.

For each dataset X in the list, reads ./data/X_{V,W,W2}.csv + X_mask{,2}.npy and
writes ./data/X_7D_{V,W,W2}.csv + X_7D_mask{,2}.npy .

  X_7D_V.csv  : (floor(T/7), N)  weekly summed counts
  X_7D_W.csv  : identical to X_W.csv  (spatial adjacency does not change)
  ... (W2, mask, mask2 likewise copied)

Usage (from repo root):
  python data_prep/make_weekly.py
  python data_prep/make_weekly.py --datasets SP_CRIME POA_CRIME BA_LESIONES CHI_CRIME
"""
import argparse
import os
import shutil
import numpy as np
import pandas as pd

DATA = "./data"


def weekly_sum(V, k=7):
    """(T, N) daily -> (T//k, N) summed over non-overlapping k-day blocks (drop remainder)."""
    T, N = V.shape
    nb = T // k
    return V[: nb * k].reshape(nb, k, N).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["SP_CRIME", "POA_CRIME", "BA_LESIONES", "CHI_CRIME"])
    ap.add_argument("--k", type=int, default=7, help="block size in days (default 7)")
    args = ap.parse_args()

    for ds in args.datasets:
        v_path = f"{DATA}/{ds}_V.csv"
        if not os.path.exists(v_path):
            print(f"[SKIP] {ds}: {v_path} not found")
            continue
        V = pd.read_csv(v_path, header=None).values.astype(np.float64)
        Vw = weekly_sum(V, args.k)
        out = f"{ds}_7D"
        pd.DataFrame(Vw).to_csv(f"{DATA}/{out}_V.csv", header=False, index=False)
        # spatial graph unchanged -> copy W/W2/mask/mask2 under the _7D name
        for suf in ("_W.csv", "_W2.csv"):
            if os.path.exists(f"{DATA}/{ds}{suf}"):
                shutil.copyfile(f"{DATA}/{ds}{suf}", f"{DATA}/{out}{suf}")
        for suf in ("_mask.npy", "_mask2.npy"):
            if os.path.exists(f"{DATA}/{ds}{suf}"):
                shutil.copyfile(f"{DATA}/{ds}{suf}", f"{DATA}/{out}{suf}")
        print(f">> {ds}: {V.shape} daily -> {out} {Vw.shape} weekly "
              f"(total={int(Vw.sum()):,}, mean/cell/week={Vw.mean():.3f})")


if __name__ == "__main__":
    main()
