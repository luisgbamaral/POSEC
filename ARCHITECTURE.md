# POSEC — Architecture

How the code is organized, which equation lives in which module, and how to
extend it. For installation and replication commands, see [README.md](README.md).

## 1. Data flow

```
data/{CITY}_V.csv ──► posec.data.data_utils ──► z-scored windows (train/val/test)
                                 │
checkpoints/ ──► posec.models.infer ──► backbone predictions ŷ  (frozen weights)
                                 │
                                 │  POSEC / GUARD IA
                                 │  posec.hybrid.glm       (per-cell Poisson calibration)
                                 │  posec.hybrid.guardia   (dose sweep + gate)
                                 │  posec.hybrid.predictive (discrete distributions)
                                 │
                  posec.eval.probabilistic  (scores all methods)
                  posec.eval.metrics        (ALS/MAE/RMSE/Moran/PAI/HAC)
                  posec.eval.spatial_diag   (CD / correlogram / ECM: base vs POSEC)
                                 ▼
                  results/probabilistic/*.csv + figs/
```

## 2. Module ↔ method map

### `posec/config.py` — the experiment
Single source of truth: cities (SP/POA/BA/Chicago + weekly `*_7D`), backbones,
split sizes, NB grid, POSEC settings (gate loss, EB pool min nodes, parallel
jobs), paths. Env overrides drive `reproduce.py`: `POSEC_CITIES`, `POSEC_NVAL`,
`POSEC_NTEST`, `POSEC_OUT`, `POSEC_BACKBONES`; `SMOKE=1` restricts to POA/stgcn.

### `posec/models/` — backbones
- `base_model.py`, `layers.py`, `trainer.py`, `tester.py` — STGCN and the SAEA
  variants (`--saea none|sparse|structural`), TF1-style graphs.
- Graph-WaveNet and STHSL are self-contained in `scripts/train_*.py`.
- `infer.py` — the backbone catalogue (`_MODEL_SPECS`: checkpoint prefix +
  output tensor per model) and frozen-weight inference. **Adding a backbone =
  registering it here** + a checkpoint dir following `_DEFAULT_SUBDIR`.

### `posec/hybrid/` — the calibration method
- **`glm.py`** — the GUARD IA engine: per-cell Poisson calibration GLM

      log E[y_it] = β0 + α·log(ŷ_it) + β1·ε_{i,t−1} + β2·(Wε)_{i,t−1}

  (ŷ as a *free* covariate: α≠1 absorbs systematic bias). Training data only.
  `fit_one_node` returns the MLE params + val/test design matrices for the sweep.
- **`guardia.py`** — dose and gate on top of the GLM: sweep c ∈ [0,2] scaling
  β2, record per-node validation loss L[c,i] and |LISA| A[c,i]; select c per
  node at the **Pareto knee of (L[:,i], A[:,i])** (`cstar_lisa`, the proposed
  dose), with degenerate cells falling back to the global-knee `best_c`; a
  per-node gate keeps the backbone wherever calibration does not beat it on
  validation. Returns the `lisapareto` prediction (`mu_te, mu_va, s_i`).
- **`predictive.py`** — every method is scored as a discrete distribution over
  integer counts. `Predictive` (shared: pmf via cdf differences, log score,
  randomized PIT, central-interval coverage); `CountPredictive` (native
  Poisson/NB2); `nb_alpha_mle` (NB2 dispersion by grid MLE on validation).

### `posec/eval/` — scoring and diagnostics
- **`metrics.py`** — Moran's I (+ analytical p per step, t-test across steps),
  LISA (mean and per-node), PAI@k, MAE/RMSE, Newey-West HAC test (the
  Giacomini-White / Diebold-Mariano statistic).
- **`probabilistic.py`** — the orchestrator: loads each backbone, runs POSEC and
  scores **three methods** — `base+Poisson`, `base+NB` (raw backbone baselines)
  and `guardia-lisac+NB` (the proposed model) — on the unified discrete log score
  + point/spatial metrics. Writes 4 CSVs. Sanity gates abort if any PMF loses
  mass or a log score is non-finite.
- **`spatial_diag.py`** — residual cross-sectional dependence: Pesaran CD,
  correlogram by graph hop with a node-permutation null band, λ_max vs the
  Marchenko-Pastur edge, and error-correlation-matrix figures (base vs POSEC).
- **`plotting.py`** — NeurIPS figure style (`set_style()`), one color per
  method family (`PALETTE`, `method_color()`), `save_fig()` (PDF+PNG).

### `scripts/` — entry points
- `reproduce.py` — **the single replication driver**: runs the calibration +
  spatial diagnostics for every experiment set (`main` / `chicago` / `weekly`),
  with `--train` / `--build-data` flags. Portable (uses `sys.executable`; no
  machine paths).
- `run_probabilistic.py`, `run_spatial_diag.py` — thin wrappers over the eval
  package. `train_{stgcn,gwavenet,sthsl}.py` — the SAEA-protocol backbone trainers.

### `data_prep/` — dataset construction
`prepare_chicago.py` (public Chicago CSV → `CHI_CRIME_*` on a ~1 km grid),
`make_weekly.py` (7-day-sum `*_7D` variants), and the SP/POA/BA provenance
scripts. All take input paths via CLI (defaults under `./raw/`).

### `tests/` — numerical invariants + golden lock
`test_math.py` (PMF mass, PIT uniformity, NB-MLE recovery, LISA consistency,
HAC power) and `test_golden.py` (the SMOKE run reproduces
`tests/fixtures/golden_smoke_poa_stgcn_als.csv` within `atol=1e-3`).

## 3. Design decisions

- **One proposed model.** `guardia-lisac+NB` is the method; `base+Poisson` /
  `base+NB` are the raw-backbone baselines. The backbone catalogue is a registry.
- **One config.** No hyperparameter lives outside `config.py`; the paper ↔
  code mapping is auditable in one screen.
- **Anti-leak by construction.** Estimation consumes train; dose/gate use
  validation; test enters only through `guardia_predict`, whose test-time
  regressors are lagged observed values.
- **Unified probabilistic ruler.** Every method is reduced to a PMF over the
  integers (Poisson / NB2) before scoring, so log scores are comparable.
- **Golden-regression development.** Every refactor is verified against the
  golden fixture (`test_golden.py`, `atol=1e-3` above the ~1e-6 TF32 GPU-inference
  noise) before landing.

## 4. How to extend

| Goal | Touch |
|---|---|
| New city | `data/{NAME}_{V,W,W2}.csv` + one line in `config.CITIES` |
| New backbone | train script + `_MODEL_SPECS`/`_DEFAULT_SUBDIR` in `models/infer.py` + `config.BACKBONES` |
| New count distribution | subclass `Predictive` (provide `cdf`, `ppf`) |
| New metric | function in `eval/metrics.py`, add to the row dict in `eval/probabilistic.py` |
| Different NB grid / gate loss / dose grid | `config.py` / `hybrid/guardia.py` |
