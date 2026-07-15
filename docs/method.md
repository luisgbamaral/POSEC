# POSEC: Pareto-Optimal Spatial Error Calibration via Poisson Regression

*Method and experimental protocol - companion to `posec/calib/{glm,calibration}.py`.*

## 1. Problem and idea

Let $y_{it} \in \mathbb{N}_0$ be the crime count in cell $i = 1,\dots,N$ on day
$t$, and $\hat{y}_{it} > 0$ the point forecast of a spatio-temporal neural backbone
(STGCN/SAEA, Graph-WaveNet or STHSL) trained on the same data. The backbones leave
two exploitable structures in the residuals $\varepsilon_{it} = y_{it} - \hat{y}_{it}$:
(i) systematic per-cell over/under-prediction, and (ii) short-lag spatio-temporal
dependence. POSEC corrects both with a **per-cell Poisson regression that treats the
backbone as a covariate**, and controls how much spatial correction each cell
receives through a **dose** tuned on validation and a per-cell **gate** that can
switch the correction off entirely. The output is natively probabilistic.

## 2. Per-cell calibration GLM

For each cell $i$, fit by maximum likelihood (IRLS, **training period only**):

$$\log \mathbb{E}[y_{it}] \;=\; \beta_{0i} \;+\; \alpha_i \log \hat{y}_{it}
\;+\; \beta_{1i}\,\varepsilon_{i,t-1} \;+\; \beta_{2i}\,(W\varepsilon)_{i,t-1},
\qquad y_{it}\sim\text{Poisson}. \tag{1}$$

Here $W$ is the row-normalised spatial-weight matrix and
$(W\varepsilon)_{i,t} = \sum_j w_{ij}\varepsilon_{jt}$. Two design decisions matter:

- **Free elasticity $\alpha_i$** - the backbone enters as a covariate, not a fixed
  offset; $\alpha_i \neq 1$ absorbs multiplicative bias (the fixed offset is
  recovered at $\alpha_i = 1$). Predictions use $\log\hat y$ floored at
  $\log(10^{-3})$ to keep the link finite.
- **Lagged regressors only** - both residual terms use $t-1$; no contemporaneous
  neighbour information enters Eq. (1), so the one-step-ahead forecast needs no
  simultaneity.

Cells whose GLM fails or returns invalid parameters fall back to the
population-mean coefficient vector $\bar\beta$ (mean over converged cells); the
method requires at least $20$ converged cells, otherwise it degrades to the raw
backbone.

## 3. Dose: scaling the spatial term

The spatial coefficient is scaled by a dose $c \ge 0$:

$$\hat\mu_{it}(c) \;=\; \exp\!\big(\beta_{0i} + \alpha_i \log\hat y_{it}
+ \beta_{1i}\varepsilon_{i,t-1} + c\,\beta_{2i}(W\varepsilon)_{i,t-1}\big). \tag{2}$$

$c$ is swept over the grid $\mathcal{C} = \{0, 0.1, \dots, 2.0\}$ on the
**validation** period. For each candidate $c$ two per-cell curves are recorded:

$$L_i(c) = \tfrac{1}{T_{va}}\textstyle\sum_t \big(y_{it} - \tilde\mu_{it}(c)\big)^2,
\qquad
A_i(c) = \tfrac{1}{T_{va}}\textstyle\sum_t \big|\,z_{it}\,(Wz_t)_i\,\big|, \tag{3}$$

where $\tilde\mu(c)$ is the **gated** prediction (§4), $z_t$ is the
cross-sectionally centred residual at time $t$, and $A_i$ is the mean absolute
**local Moran (LISA)** - a per-cell measure of residual spatial clustering. The
squared loss in (3) follows the configured gate loss (MSE here; MAE recovers the
original criterion).

## 4. Gate

For each cell and candidate dose, the calibrated prediction replaces the backbone
**only if it wins on validation**:

$$s_i(c) = \mathbb{1}\!\left[\,L_i^{\text{calib}}(c) < L_i^{\text{base}}\,\right],
\qquad
\tilde\mu_{it}(c) = s_i(c)\,\hat\mu_{it}(c) + (1 - s_i(c))\,\hat y_{it}. \tag{4}$$

The gate is the method's safety net: cells where calibration does not help keep the
backbone untouched, bounding the worst case.

**Independent gate (`gate_frac`).** The dose and the gate must not be selected on
the same data: choosing the dose to minimise validation loss and then gating on
that same loss is optimistic. We therefore split the validation chronologically
into a dose block (val1) and a **disjoint** gate block (val2): the dose is chosen on
val1 (Eq. 3), the gate $s_i$ is evaluated on val2 (Eq. 4), making the gate an
independent test. `gate_frac` is the fraction of validation held out for val2, set
to $1/3$ throughout (the daily 110-day validation splits into ~73 dose / ~37 gate;
the weekly 30% validation into 20% dose / 10% gate). `gate_frac=0` recovers the
single-validation gate.

## 5. Dose selection - per-node Pareto knee

The proposed dose is chosen **per cell** at the **Pareto knee** of the bi-objective
frontier $\big(L_i(\cdot), A_i(\cdot)\big)$ (validation loss vs. local spatial
autocorrelation): among the strictly non-dominated candidates, pick the one nearest
the ideal point after per-cell min-max normalisation. Degenerate cells (flat curves)
inherit the **global** knee - the same construction on the aggregate frontier
$\big(\sqrt{\bar L(c)},\, \bar A(c)\big)$. The GLM is fit once and the grid swept
once; selection uses validation only.

## 6. Predictive distribution

The calibrated mean $\tilde\mu_{it}$ is scored as a discrete distribution over
integer counts. Because crime counts are over-dispersed, the headline model wraps it
in a Negative-Binomial NB2, $\text{Var} = \mu + \hat\alpha\mu^2$, with the dispersion
$\hat\alpha$ estimated by grid MLE ($\alpha \in 10^{[-4,1]}$, 60 log-spaced points)
on the model's own validation predictions. The two baselines score the raw backbone
$\hat y$ as a Poisson (`base+Poisson`) and as an NB2 (`base+NB`).

## 7. Experimental protocol

**Data.** Daily counts on regular spatial cells, all crime types aggregated: São
Paulo ($N{=}1445$), Porto Alegre ($N{=}94$), Buenos Aires ($N{=}74$), Chicago ($N{=}653$),
each with a weekly (7-day-sum) variant. Backbones train on z-standardised data;
predictions are de-normalised and truncated at 0.

**Splits & memory.** Strictly chronological, no shuffling; the per-cell gate is
always judged on a validation block disjoint from dose selection ($\texttt{gate\_frac}=1/3$).
- *Daily:* $n_{his}=7$; last $110$ days = test, previous $110$ = validation (split
  ~73 dose / ~37 gate), rest = training. Data is abundant, so the training fraction
  stays large (~70-80%).
- *Weekly:* $n_{his}=6$ (monthly memory; STGCN requires the temporal output kernel
  $K_o>1$, i.e. $n_{his}\ge 6$); the series is short, so we rebalance to a
  chronological **60/20/10/10** split (train / dose-validation / gate-validation /
  test).

**Anti-leak.** Eq. (1) is estimated on train only; dose and gate use validation
only; the test regressors $\varepsilon_{i,t-1}, (W\varepsilon)_{i,t-1}$ are built
exclusively from *observed* past values (the val→test boundary uses the last
observed validation day). Nothing from the test period enters estimation or
selection. A `split_meta.json` written at training time is asserted against the
evaluation split.

**Evaluation.** (i) *Probabilistic:* mean discrete log score
$\text{ALS} = -\overline{\log P(Y = y_{it})}$ on a unified integer ruler; randomized
PIT histograms and 80/95% central-interval coverage. (ii) *Point:* MAE, RMSE and
$f_{\text{worse}}$ (fraction of cells whose MAE worsens vs. the backbone).
(iii) *Spatial:* Moran's $I$ of the test residuals with analytical per-step
$p$-values and a $t$-test across steps; PAI@$k$ for $k \in \{1,5,10,25\}\%$; and, in
the diagnostics, Pesaran CD, the hop correlogram, and the error-correlation matrix.
(iv) *Significance:* Giacomini-White (log scores) and Diebold-Mariano (absolute /
squared errors) on the per-time loss differentials, with a Newey-West HAC variance
(truncation $\lfloor T^{1/3} \rfloor$). Per-cell doses $c_i^\*$ and gates $s_i$ are
exported for spatial inspection (dose maps, gate-overlap analysis).
