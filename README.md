# POSEC — Pareto-Optimal Spatial Error Calibration via Poisson Regression

Post-hoc probabilistic calibration of frozen spatio-temporal neural backbones
(STGCN, Graph-WaveNet, STHSL) for crime forecasting. **POSEC** fits a per-cell
Poisson-GLM that treats the backbone prediction as a covariate, then chooses how
much spatial-lag correction each cell receives at the **per-node Pareto knee** of
(validation loss × local spatial autocorrelation |LISA|), behind a per-cell
**gate**; the calibrated mean is scored as a Negative-Binomial distribution
(`guardia-lisac+NB`, the proposed model). Evaluated against the raw backbone
(`base+Poisson`, `base+NB`) on four crime datasets — São Paulo, Porto Alegre,
Bahía, Chicago — with a unified discrete log score (ALS), point error (MAE/RMSE),
PAI hotspot concentration, Moran's I and residual cross-sectional-dependence
diagnostics. A weekly (7-day-ahead, single-step) variant of every dataset is also
provided.

**This README is the replication guide.** For the code architecture and the
module ↔ method mapping, see [ARCHITECTURE.md](ARCHITECTURE.md). For the method
and experimental protocol in short-paper form (with formulas), see
[docs/method_guardia.md](docs/method_guardia.md). For the results tables and
analysis, see [docs/RESULTS.md](docs/RESULTS.md).

---

## 1. Setup

```bash
conda env create -f environment.yml   # python 3.9, TF 2.10 + CUDA 11.2/cuDNN 8.1
conda activate posec
pip install -e .                       # makes `import posec` resolve
```

Activating the env puts the CUDA DLLs on `PATH`, so TensorFlow 2.10 (`tf.compat.v1`
graph mode) sees the GPU with no extra setup — **needed for training and for the
backbone inference used in evaluation.** All commands run **from the repo root**.

## 2. Data

Datasets are **not tracked in git**; place them under `data/` as
`{CITY}_{V,W,W2}.csv` + `{CITY}_mask{,2}.npy`, where `V` = daily counts
(T days × N cells), `W`/`W2` = gaussian-kernel spatial adjacencies, `mask{,2}` =
no-edge masks. Cities: `SP_CRIME` (N=1445), `POA_CRIME` (94), `BA_LESIONES` (74),
`CHI_CRIME` (653). Each has a weekly variant `{CITY}_7D_*` (non-overlapping 7-day
sums). Download: _TBD (release/Zenodo)_.

Rebuild from raw sources (`data_prep/`):

```bash
python data_prep/prepare_chicago.py --raw-csv ./raw/chicago_crimes_2001_present.csv  # CHI_CRIME (public data)
python data_prep/make_weekly.py                                                       # all *_7D variants
```
`prepare_chicago.py` uses the public City-of-Chicago "Crimes 2001–Present" CSV
(all types aggregated, 2023-01-01…2025-12-31, ~1 km grid). The SP/POA/BA prep
scripts document provenance from non-redistributable sources — use the provided
built `data/` for those.

## 3. Checkpoints

Trained backbones go under `checkpoints/` (git-ignored). Download: _TBD (Zenodo)_,
or retrain via `reproduce.py --train` (§5) / the per-model scripts, e.g.:

```bash
python scripts/train_stgcn.py    --dataset SP_CRIME --n_route 1445 --n_his 7 --n_pred 1 --batch_size 8 --epoch 300 --small_model
python scripts/train_gwavenet.py --dataset SP_CRIME --n_route 1445 --n_his 7 --n_pred 1 --batch_size 8 --loss mse --res_ch 16 --skip_ch 64 --end_ch 128
python scripts/train_sthsl.py    --dataset SP_CRIME --n_route 1445 --n_his 7 --batch_size 8 --epoch 300 --loss mse
```

## 4. Smoke test (~15 s)

```bash
SMOKE=1 python scripts/run_probabilistic.py          # POA_CRIME / stgcn only
```
PowerShell: `$env:SMOKE=1; python scripts/run_probabilistic.py; $env:SMOKE=$null`

## 5. Reproduce everything — one command

```bash
python scripts/reproduce.py                 # eval + spatial diagnostics, all experiments
python scripts/reproduce.py --only chicago  # a single experiment set (main | chicago | weekly)
python scripts/reproduce.py --train         # (re)train the 3 backbones first (GPU, hours)
python scripts/reproduce.py --build-data    # (re)build the datasets first
```

`reproduce.py` runs, per experiment set, `run_probabilistic.py` (POSEC calibration)
and `run_spatial_diag.py` (residual diagnostics), writing to `results/`:

| Experiment | Datasets | Output |
|---|---|---|
| `main`    | SP / POA / BA (daily)              | `results/probabilistic/` |
| `chicago` | Chicago (daily)                   | `results/chi_daily/` |
| `weekly`  | SP / POA / BA / Chicago (7-day)   | `results/all_7d/` |

Each produces `als_master.csv` (ALS/MAE/RMSE/MI/PAI per method), `gw_dm_tests.csv`
(Giacomini-White / Diebold-Mariano, HAC), `calibration.csv` (PIT + 80/95% coverage),
`per_node.csv` (per-cell errors + POSEC doses/gates for maps), and — from the
diagnostics — `spatial_diag.csv` + `figs/` (Pesaran CD, hop correlograms, ECM maps).
A column-by-column `README.md` is written next to the CSVs. **Methods scored (3):**
`base+Poisson`, `base+NB`, `guardia-lisac+NB` (proposed).

## 6. Determinism

- Stochastic steps are seeded (`np.random.seed(0)` for the randomized PIT and the
  `spatial_diag` permutation nulls); the GLM-IRLS and grid-MLE estimators are
  deterministic given the data.
- GPU backbone inference uses TensorFloat-32, non-deterministic at ~1e-6 — far
  below the 3–4 significant figures reported. The golden-regression test
  (`tests/test_golden.py`) therefore locks results with `atol=1e-3`.

## 7. Configuration

Everything lives in **`posec/config.py`**: cities, backbones, split sizes, NB
dispersion grid, POSEC gate loss (`GUARDIA_GATE`), EB pool size, parallel jobs,
paths. Env overrides (used by `reproduce.py`): `POSEC_CITIES="NAME:N,..."`,
`POSEC_NVAL`, `POSEC_NTEST`, `POSEC_OUT`, `POSEC_BACKBONES`, and `SMOKE=1`.

## 8. Tests

```bash
pytest tests/ -q
```
Fast numerical invariants (`test_math.py`: PMF mass, PIT uniformity, NB-MLE
recovery, LISA consistency, HAC power) + the golden-regression lock
(`test_golden.py`: the SMOKE run reproduces `tests/fixtures/` within `atol=1e-3`;
auto-skips if the POA_CRIME data/checkpoint are absent).

## License

MIT — see [LICENSE](LICENSE).
