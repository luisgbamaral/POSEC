"""
prepare_chicago.py
------------------
Build a CHI_CRIME dataset in the SAME format as SP_CRIME / POA_CRIME / BA_LESIONES
(single-channel daily counts on a ~1km spatial grid + gaussian-kernel adjacency),
from the raw "Crimes - 2001 to Present" Chicago CSV. ALL crime types are aggregated
into a single count per cell per day ("any crime").

Outputs to ./data/ :
  CHI_CRIME_V.csv     (T, N)  daily counts, no header
  CHI_CRIME_W.csv     (N, N)  gaussian-kernel weighted adjacency, no header
  CHI_CRIME_W2.csv    (N, N)  denser adjacency (for saea=structural2)
  CHI_CRIME_mask.npy  (N, N)  1 where NO edge in W  (saea=structural)
  CHI_CRIME_mask2.npy (N, N)  1 where NO edge in W2 (saea=none/structural2)
  CHI_CRIME_cells.csv (N, 4)  node_id, row, col, lat, lon  (reference, not used by the pipeline)

Date range matches SP_CRIME exactly: 2023-01-01 .. 2025-12-31 (1096 days).

Gaussian kernel identical in spirit to prepare_crime_data.py / math_graph.weight_matrix:
  W[i,j] = exp(-(d/10000)^2 / sigma2) * (W >= epsilon) * no_self_loop     (d in metres)
sigma2/epsilon are tuned to the ~1km grid so a cell connects to its ~2-3 km
neighbourhood (reported at the end; adjust with the flags if desired).

Usage (from repo root):
  python data_prep/prepare_chicago.py
  python data_prep/prepare_chicago.py --cell_km 1.0 --sigma2 0.1 --epsilon 0.5
"""
import argparse
import os
import numpy as np
import pandas as pd

# Public source: City of Chicago "Crimes - 2001 to Present" (data.cityofchicago.org).
# Download the CSV and point --raw-csv at it (default below).
DEFAULT_RAW_CSV = "./raw/chicago_crimes_2001_present.csv"
START, END = "2023-01-01", "2025-12-31"          # inclusive, matches SP_CRIME (1096 days)


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised haversine distance in metres between two (broadcastable) lat/lon arrays."""
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def gaussian_kernel(D, sigma2, epsilon):
    """exp(-(d/10000)^2 / sigma2) thresholded at epsilon, zero diagonal (matches SP)."""
    n = D.shape[0]
    Ds = D / 10_000.0
    W = np.exp(-(Ds ** 2) / sigma2)
    W = W * (W >= epsilon) * (np.ones((n, n)) - np.eye(n))
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", dest="raw_csv", default=DEFAULT_RAW_CSV,
                    help="path to the Chicago crimes CSV (default ./raw/...)")
    ap.add_argument("--out-dir", dest="out_dir", default="./data", help="output dir (default ./data)")
    ap.add_argument("--cell_km", type=float, default=1.0, help="grid cell size in km (default 1.0)")
    ap.add_argument("--sigma2", type=float, default=0.1, help="gaussian kernel sigma2 for W")
    ap.add_argument("--epsilon", type=float, default=0.5, help="edge threshold for W")
    ap.add_argument("--sigma2_dense", type=float, default=0.3, help="sigma2 for the denser W2")
    ap.add_argument("--epsilon_dense", type=float, default=0.1, help="threshold for W2")
    ap.add_argument("--clip_pct", type=float, default=0.999,
                    help="keep points within this central lat/lon quantile (drops geocoding outliers)")
    args = ap.parse_args()
    RAW_CSV, OUT_DIR = args.raw_csv, args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. read only the needed columns; pre-filter by Year, then by exact date ──
    print(f">> reading {RAW_CSV} (this is ~2.3 GB, a minute or two) ...")
    df = pd.read_csv(RAW_CSV, usecols=["Date", "Latitude", "Longitude", "Year"],
                     dtype={"Latitude": str, "Longitude": str})
    df = df[df["Year"].isin([2023, 2024, 2025])].copy()
    print(f"   rows in 2023-2025: {len(df):,}")

    # lat/lon use comma decimal separator in this export
    df["lat"] = pd.to_numeric(df["Latitude"].str.replace(",", ".", regex=False), errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude"].str.replace(",", ".", regex=False), errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    df["date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    days = pd.date_range(START, END, freq="D")
    df = df[(df["date"] >= days[0]) & (df["date"] <= days[-1])]
    print(f"   rows with valid date+coords in range: {len(df):,}")

    # ── 2. drop geocoding outliers via central quantiles ─────────────────────────
    q = (1 - args.clip_pct) / 2
    lo_la, hi_la = df["lat"].quantile([q, 1 - q])
    lo_lo, hi_lo = df["lon"].quantile([q, 1 - q])
    df = df[(df["lat"].between(lo_la, hi_la)) & (df["lon"].between(lo_lo, hi_lo))]

    # ── 3. build the ~cell_km grid ───────────────────────────────────────────────
    lat0, lat1 = df["lat"].min(), df["lat"].max()
    lon0, lon1 = df["lon"].min(), df["lon"].max()
    mid_lat = (lat0 + lat1) / 2
    dlat = args.cell_km / 111.0                                   # deg per km (lat)
    dlon = args.cell_km / (111.0 * np.cos(np.radians(mid_lat)))   # deg per km (lon)
    row = np.floor((df["lat"].values - lat0) / dlat).astype(int)
    col = np.floor((df["lon"].values - lon0) / dlon).astype(int)
    df["cell"] = list(zip(row, col))

    # keep only cells with >=1 crime; assign contiguous node ids (sorted for determinism)
    cells = sorted(df["cell"].unique())
    cell_to_id = {c: i for i, c in enumerate(cells)}
    N = len(cells)
    df["node"] = df["cell"].map(cell_to_id)
    print(f">> grid: cell~{args.cell_km}km  ->  {N} populated cells (of {(row.max()+1)*(col.max()+1)} grid slots)")

    # cell centres (lat/lon)
    centres = np.zeros((N, 2))
    for c, i in cell_to_id.items():
        r, k = c
        centres[i, 0] = lat0 + (r + 0.5) * dlat
        centres[i, 1] = lon0 + (k + 0.5) * dlon

    # ── 4. daily counts V (T, N) ─────────────────────────────────────────────────
    day_idx = {d: t for t, d in enumerate(days)}
    df["t"] = df["date"].map(day_idx)
    V = np.zeros((len(days), N), dtype=np.float64)
    np.add.at(V, (df["t"].values, df["node"].values), 1.0)
    print(f">> V: shape {V.shape}  total_crimes={int(V.sum()):,}  "
          f"mean/cell/day={V.mean():.3f}  max={int(V.max())}")

    # ── 5. adjacency via haversine + gaussian kernel ─────────────────────────────
    la = centres[:, 0][:, None]; lo = centres[:, 1][:, None]
    D = haversine_m(la, lo, la.T, lo.T)                           # (N, N) metres
    W = gaussian_kernel(D, args.sigma2, args.epsilon)
    W2 = gaussian_kernel(D, args.sigma2_dense, args.epsilon_dense)
    deg, deg2 = (W > 0).sum(1), (W2 > 0).sum(1)
    print(f">> W  degree: min/med/mean/max = {deg.min()}/{int(np.median(deg))}/{deg.mean():.1f}/{deg.max()}")
    print(f">> W2 degree: min/med/mean/max = {deg2.min()}/{int(np.median(deg2))}/{deg2.mean():.1f}/{deg2.max()}")

    mask = (W == 0).astype(np.float32)     # 1 where NO edge
    mask2 = (W2 == 0).astype(np.float32)

    # ── 6. write, matching the other datasets' file conventions ──────────────────
    pd.DataFrame(V).to_csv(f"{OUT_DIR}/CHI_CRIME_V.csv", header=False, index=False)
    pd.DataFrame(W).to_csv(f"{OUT_DIR}/CHI_CRIME_W.csv", header=False, index=False)
    pd.DataFrame(W2).to_csv(f"{OUT_DIR}/CHI_CRIME_W2.csv", header=False, index=False)
    np.save(f"{OUT_DIR}/CHI_CRIME_mask.npy", mask)
    np.save(f"{OUT_DIR}/CHI_CRIME_mask2.npy", mask2)
    ref = pd.DataFrame({"node": np.arange(N), "lat": centres[:, 0], "lon": centres[:, 1]})
    ref.to_csv(f"{OUT_DIR}/CHI_CRIME_cells.csv", index=False)
    print(f">> wrote CHI_CRIME_{{V,W,W2}}.csv + mask{{,2}}.npy + cells.csv to {OUT_DIR}/  (N={N})")
    print(f">> USE n_route = {N} for the training/eval commands.")


if __name__ == "__main__":
    main()
