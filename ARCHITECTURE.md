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
Single source of truth: cities, backbones, split sizes, NB grid, GUARD IA
settings (gate loss, EB pool min nodes, parallel jobs), paths. `SMOKE=1` (env
var) restricts to POA/stgcn for fast regression checks.

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
  β2, record per-node validation loss L[c,i] and |LISA| A[c,i]; select c by
  `global` (Pareto knee on aggregate loss×|LISA|), `pernode` (argmin loss) or
  `lisapareto` (per-node Pareto knee); per-node gate keeps the backbone
  wherever calibration does not beat it on validation. The heavy per-cell GLM
  fit runs once, shared by all three dose modes.
- **`predictive.py`** — every method is scored as a discrete distribution over
  integer counts. `Predictive` (shared: pmf via cdf differences, log score,
  randomized PIT, central-interval coverage); `CountPredictive` (native
  Poisson/NB2); `TransformPredictive` (Gaussian in a Transform space, used for
  the `base+Gauss` baseline). `nb_alpha_mle` (NB2 dispersion by grid MLE) and
  `shrink_var` (per-node predictive variance shrinkage) also live here.
- **`transforms.py`** — variance-stabilizing spaces (`LEVEL`, `LOG1P`,
  `ANSCOMBE`); POSEC uses `LEVEL` for the Gaussian baseline. (Kept whole so the
  numerical unit tests remain shared with the sthyb repo.)

### `posec/eval/` — scoring and diagnostics
- **`metrics.py`** — Moran's I (+ analytical p per step, t-test across steps),
  LISA (mean and per-node), PAI@k, MAE/RMSE, Newey-West HAC test (the
  Giacomini-White / Diebold-Mariano statistic).
- **`probabilistic.py`** — the orchestrator: loads each backbone, runs GUARD IA
  (the three dose modes + NB wrappers) plus the base distributions, and scores
  the 9 methods on the unified discrete log score + point/spatial metrics.
  Writes 4 CSVs. Sanity gates abort the run if any PMF loses mass or a log score
  is non-finite.
- **`spatial_diag.py`** — residual cross-sectional dependence: Pesaran CD,
  correlogram by graph hop with a node-permutation null band, λ_max vs the
  Marchenko-Pastur edge, and error-correlation-matrix figures (base vs POSEC).
- **`plotting.py`** — NeurIPS figure style (`set_style()`), one color per
  method family (`PALETTE`, `method_color()`), `save_fig()` (PDF+PNG).

### `scripts/` — thin entry points
Each is ≤ 10 lines: parse nothing, import the package, run. The training
scripts (`train_stgcn/gwavenet/sthsl`) are the original SAEA-protocol trainers.

### `tests/` — numerical invariants
Six fast tests (no TF, no checkpoints): transform round-trip, PMF mass,
PIT uniformity, NB-MLE recovery, LISA consistency, HAC power.

## 3. Design decisions

- **Methods as data, not branches.** The backbone catalogue is a registry; the
  three dose modes share one GLM fit and one sweep.
- **One config.** No hyperparameter lives outside `config.py`; the paper ↔
  code mapping is auditable in one screen.
- **Anti-leak by construction.** Estimation consumes train; dose/gate use
  validation; test enters only through `guardia_predict`, whose test-time
  regressors are lagged observed values.
- **Unified probabilistic ruler.** Every method — Poisson, NB, Gaussian
  baseline — is reduced to a PMF over the integers before scoring, so log
  scores are comparable across families.
- **Golden-regression development.** Every refactor was verified byte-identical
  on the output CSVs (smoke + full 9-cell) before landing.

## 4. How to extend

| Goal | Touch |
|---|---|
| New city | `data/{NAME}_{V,W,W2}.csv` + one line in `config.CITIES` |
| New backbone | train script + `_MODEL_SPECS`/`_DEFAULT_SUBDIR` in `models/infer.py` + `config.BACKBONES` |
| New count distribution | subclass `Predictive` (provide `cdf`, `ppf`) |
| New metric | function in `eval/metrics.py`, add to the row dict in `eval/probabilistic.py` |
| Different NB grid / gate loss / dose grid | `config.py` / `hybrid/guardia.py` |
