"""
test_math.py — fast unit tests for the numerical core (no TF / no checkpoints).

Formalizes the ad-hoc checks used during development: PMF mass, randomized-PIT
uniformity, NB-MLE recovery, LISA per-node consistency, and the HAC (GW/DM)
test. Run:  pytest tests/ -q
"""
import numpy as np
from scipy.stats import nbinom

from posec.hybrid.predictive import CountPredictive, nb_alpha_mle
from posec.eval.metrics import mean_lisa_abs, lisa_abs_per_node, hac_test

rng = np.random.default_rng(0)


def test_count_pmf_mass():
    """Poisson/NB2 PMFs integrate to ~1 (cdf at large K)."""
    mu = rng.uniform(0.5, 5.0, size=(40, 15))
    assert np.all(CountPredictive('poisson', point=mu, lam=mu).cdf(4000) >= 0.999)


def test_pit_uniform_for_well_specified_poisson():
    mu = rng.uniform(0.5, 3.0, size=(200, 40))
    y = rng.poisson(mu).astype(int)
    pr = CountPredictive('poisson', point=mu, lam=np.maximum(mu, 1e-6))
    np.random.seed(0)
    h, _ = np.histogram(pr.pit(y), bins=10, range=(0, 1))
    assert h.std() / h.mean() < 0.15        # ~uniform


def test_nb_alpha_mle_recovers_dispersion():
    mu = rng.uniform(0.5, 3.0, size=(300, 40))
    a_true = 0.5
    n = 1.0 / a_true
    y = nbinom.rvs(n, n / (n + np.maximum(mu, 1e-6)), random_state=0).astype(int)
    assert 0.3 < nb_alpha_mle(y, mu) < 0.8


def test_lisa_per_node_matches_mean():
    N, T = 40, 50
    W = rng.random((N, N)); np.fill_diagonal(W, 0.0); W /= W.sum(1, keepdims=True)
    a, p = rng.random((T, N)), rng.random((T, N))
    assert np.isclose(lisa_abs_per_node(a, p, W).mean(), mean_lisa_abs(a, p, W), rtol=1e-5)


def test_hac_detects_nonzero_mean():
    d = rng.standard_normal(200) * 0.1 + 0.1
    m, t, pv = hac_test(d)
    assert m > 0 and pv < 0.05
