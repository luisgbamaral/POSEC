# POSEC — Walkthrough: motivation, code, and experimental protocol

A guided tour of *why* POSEC exists, *how* the code produces every number, and the
*experimental protocol* behind the results. For the formal method (with formulas)
see [method_guardia.md](method_guardia.md); for the tables see [RESULTS.md](RESULTS.md).

---

## 1. Motivation — graph networks leave spatial structure in their residuals

Spatio-temporal graph neural networks (STGCN, Graph-WaveNet, STHSL) are trained to
minimise an **aggregate** point loss (MSE/MAE). That objective says nothing about
(i) the *shape of the predictive distribution* of a count, nor (ii) the *spatial
structure of the errors* the model makes. As a result, even an accurate backbone
leaves two things on the table:

1. **It is a poor probabilistic forecaster.** A point ŷ scored as a Poisson is
   over-confident on over-dispersed crime counts; there is no honest uncertainty.
2. **Its residuals stay spatially autocorrelated.** On the held-out test set the
   backbone residuals ε = y − ŷ are still clustered in space: **Moran's I and the
   Pesaran CD statistic are significant**, especially on the large graphs (São
   Paulo N=1445, Chicago N=653). Neighbouring cells are over- or under-predicted
   *together* — a signal the point model failed to absorb.

The second point is the crux: a leftover **spatially-structured error is exploitable**.
If cell *i*'s neighbours were collectively under-predicted yesterday, cell *i* is
likely under-predicted today. POSEC is a lightweight, post-hoc layer that (a) turns
the frozen backbone's point forecast into a **calibrated count distribution**, and
(b) uses the **neighbour residual lag (Wε)** to correct that spatial error — *per
cell*, and *only where it demonstrably helps on validation*. The diagnostics
(`spatial_diag`) then confirm POSEC **whitens** the residuals (lower CD / nearest-
neighbour correlation) that the backbone left behind.

Why a *calibration* layer instead of retraining the backbone with a spatial loss?
Because it is (i) **backbone-agnostic** — the same layer wraps STGCN, Graph-WaveNet
or STHSL with frozen weights; (ii) **cheap** — a per-cell Poisson GLM, no GPU; and
(iii) **safe** — a per-cell gate can always fall back to the backbone, so it never
makes a well-behaved cell worse.

---

## 2. The code, step by step

The data flows through the package in nine stages; each maps to one module.

**1. Datasets — `data_prep/` → `data/{CITY}_{V,W}.csv`.**
`V` is a T×N matrix of daily counts (all crime types aggregated); `W` is an N×N
gaussian-kernel spatial adjacency on cell-centre distances. `prepare_chicago.py`
builds Chicago on a ~1 km grid from the public city CSV; `make_weekly.py` sums
counts into non-overlapping 7-day blocks for the weekly experiment.

**2. Windows & splits — `posec/data/data_utils.py`.**
`data_gen_crime` cuts the series into overlapping `n_his+1` windows, z-scores using
**training statistics only**, and splits chronologically (last 110 days = test,
previous 110 = validation, rest = train; 16/16 weeks for the weekly variant).

**3. Frozen-backbone inference — `posec/models/infer.py`.**
`infer_split` restores a trained checkpoint and runs forward inference, returning
ŷ per split. The backbone catalogue (`_MODEL_SPECS`, `_DEFAULT_SUBDIR`) maps each
model name to its checkpoint prefix and output tensor. Predictions are de-normalised
back to counts and clamped ≥ 0. **Backbones are MSE-trained → mean-targeting.**

**4. Residuals — `posec/eval/probabilistic.py::load_backbone`.**
For train/val/test it returns `(y, ŷ)` in count space; the residual panel is
ε = y − ŷ and its spatial lag is (Wε) with `W` row-normalised.

**5. Per-cell calibration GLM — `posec/hybrid/glm.py`.**
For every cell *i*, `fit_one_node` fits by Poisson IRLS (**training data only**):

    log E[y_it] = β0 + α·log(ŷ_it) + β1·ε_{i,t−1} + β2·(Wε)_{i,t−1}

ŷ enters as a **free covariate** (α ≠ 1 absorbs multiplicative bias); the two
residual regressors are **lagged** (t−1), so a one-step-ahead forecast never needs
contemporaneous neighbour values. Cells whose GLM fails fall back to the
population-mean coefficients.

**6. Dose, Pareto knee, and gate — `posec/hybrid/guardia.py`.**
The spatial-lag coefficient β2 is scaled by a **dose** c ∈ [0, 2]. Sweeping c on
the **validation** set records, per cell, a gated loss curve `L[c,i]` and a local
Moran (|LISA|) curve `A[c,i]`. The proposed dose is the **per-node Pareto knee** of
(L[:,i], A[:,i]) — the point that best trades validation error against residual
spatial autocorrelation (`cstar_lisa`); degenerate cells fall back to the global
knee. A per-cell **gate** keeps the raw backbone wherever calibration does not beat
it on validation (`s_i = 0`), bounding the worst case.

**7. Predictive distribution — `posec/hybrid/predictive.py`.**
Every method is reduced to a **discrete distribution over integer counts** so log
scores are comparable. `CountPredictive` gives native Poisson / NB2; `nb_alpha_mle`
estimates the NB dispersion by grid MLE on validation. The calibrated mean is scored
as **NB2** (`guardia-lisac+NB`) because crime counts are over-dispersed. The base
class provides the discrete log score, randomized PIT, and central-interval coverage.

**8. Scoring — `posec/eval/probabilistic.py` + `metrics.py`.**
For each city × backbone it scores three methods — `base+Poisson`, `base+NB`
(raw-backbone baselines) and `guardia-lisac+NB` (proposed) — on ALS (mean discrete
log score, the headline), MAE/RMSE, PAI@k, and Moran's I of the residuals, plus
Giacomini-White / Diebold-Mariano significance with Newey-West HAC variance. Writes
`als_master.csv`, `gw_dm_tests.csv`, `calibration.csv`, `per_node.csv`.

**9. Residual diagnostics — `posec/eval/spatial_diag.py`.**
Compares the **base** vs **POSEC** test residuals with the Pesaran CD test, a graph-
hop correlogram (with a node-permutation null band), the largest eigenvalue vs the
Marchenko-Pastur edge, and error-correlation-matrix (ECM) heatmaps — the figures
that show POSEC whitening the spatial structure the backbone left behind.

**Orchestration — `scripts/reproduce.py`.** One command runs stages 3–9 for each
experiment set (`main` / `chicago` / `weekly`), with `--train` / `--build-data`.

---

## 3. Experimental protocol

**Datasets.** Daily crime counts on a spatial cell partition, all types aggregated:
São Paulo `SP_CRIME` (N=1445), Porto Alegre `POA_CRIME` (94), Bahía `BA_LESIONES`
(74), Chicago `CHI_CRIME` (653, ~1 km grid, 2023-01-01…2025-12-31). Spatial weights
`W` are a gaussian kernel on cell-centre distances (thresholded), matched per
dataset. A **weekly** variant of each (`*_7D`) sums counts into non-overlapping
7-day blocks — the model then predicts **next week as a single step** (7-day-ahead
aggregated forecast).

**Splits.** Strictly chronological, no shuffling: the last 110 days are test, the
preceding 110 are validation, the remainder is training (16/16 **weeks** for the
weekly variant). Normalisation statistics come from the training portion only.

**Backbones.** STGCN, Graph-WaveNet, STHSL, all **MSE-trained** (mean-targeting) on
the unified SAEA protocol (RMSProp, LR decay, `n_his=7`, `n_pred=1`), one-step-ahead.
POSEC wraps them with frozen weights — it never sees the backbone's training.

**Anti-leak by construction.** The calibration GLM is fit on **train** only; the
dose and gate are selected on **validation** only; **test** enters exclusively
through the lagged *observed* residuals ε_{t−1}, (Wε)_{t−1} (the val→test boundary
uses the last observed validation day). Nothing from the test period touches
estimation or model selection.

**Metrics.** (i) *Probabilistic:* **ALS**, the mean discrete log score on a unified
integer PMF (headline, lower = better); randomized-PIT histograms and 80/95%
central-interval coverage. (ii) *Point:* MAE, RMSE, and `f_worse` (fraction of cells
whose MAE worsens vs the backbone). (iii) *Spatial:* Moran's I of the test residuals
with analytical p-values; PAI@k hotspot concentration for k ∈ {1,5,10,25}%; and, in
the diagnostics, Pesaran CD, the hop correlogram, and the ECM.

**Significance.** Giacomini-White (on the ALS differential) and Diebold-Mariano (on
absolute and squared-error differentials), each with a Newey-West HAC variance
(truncation ⌊T^{1/3}⌋), computed on the per-day loss differences between the proposed
model and each baseline.

**Reproducibility.** All stochastic steps are seeded (`np.random.seed(0)`); the GLM-
IRLS and grid-MLE estimators are deterministic. GPU backbone inference uses
TensorFloat-32 and is non-deterministic at ~1e-6 — far below the reported precision;
`tests/test_golden.py` locks the pipeline against a fixture with `atol=1e-3`.

**Experiment sets** (`scripts/reproduce.py`): `main` (SP/POA/BA daily),
`chicago` (Chicago daily), `weekly` (all four datasets, 7-day).
