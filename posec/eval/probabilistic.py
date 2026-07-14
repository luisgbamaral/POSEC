"""probabilistic.py — POSEC probabilistic evaluation.

For every city x backbone in config, scores three methods on the held-out test:
  base+Poisson       the raw backbone as a Poisson distribution (naive baseline)
  base+NB            the raw backbone as an NB2 distribution   (fair baseline)
  guardia-lisac+NB   POSEC: per-cell Poisson calibration with a per-node
                     Pareto-optimal spatial-lag dose, wrapped in NB (proposed)

Outputs 4 CSVs to OUT_DIR (als_master, gw_dm_tests, calibration, per_node) —
see the README written alongside them.
Anti-leak: TEST never enters estimation; test-time lags come only from the
observed val->test past. Seed fixed for the randomized PIT.
"""
from os.path import join as pjoin
import numpy as np
import os
import pandas as pd
import sys
from posec.config import ALPHA_GRID, BACKBONES, BATCH, CITIES, GUARDIA_GATE, CKPT_DIR, DATA_DIR, N_TEST, N_VAL, OUT_DIR
from posec.data.data_utils import data_gen_crime
from posec.eval.metrics import hac_test, mae, rmse, spatial_metrics
from posec.hybrid.guardia import guardia_predict
from posec.hybrid.predictive import CountPredictive, nb_alpha_mle
from posec.models.infer import _DEFAULT_SUBDIR, _detect_n_his, _find_ckpt, infer_split
from posec.utils.math_utils import z_inverse

np.random.seed(0)   # reproducible randomized PIT

PROPOSED = 'guardia-lisac+NB'   # POSEC (per-node Pareto-optimal dose, NB-wrapped)


def load_backbone(ds, N, bk):
    """Load a frozen backbone and return {split: (y, p)} in count space (>=0)."""
    subdir = _DEFAULT_SUBDIR[bk].format(ds=ds); ckpt_dir = pjoin(CKPT_DIR, subdir)
    if not os.path.isdir(ckpt_dir):
        print(f"  [SKIP] {ds}/{bk}: dir not found ({ckpt_dir})"); return None
    ckpt = _find_ckpt(ckpt_dir, bk); n_his = _detect_n_his(ckpt) or 7
    PeMS = data_gen_crime(pjoin(DATA_DIR, f'{ds}_V.csv'), None, N_VAL, N_TEST, N, n_his + 1)
    st = PeMS.get_stats(); inv = lambda z: z_inverse(z, st['mean'], st['std'])
    out = {}
    for sp in ('train', 'val', 'test'):
        x = PeMS.get_data(sp)
        y = np.maximum(inv(x[:, n_his, :, 0]), 0.0)
        p = np.maximum(inv(infer_split(x, n_his, BATCH, ckpt, bk)), 0.0)
        out[sp] = (y.astype(np.float64), p.astype(np.float64))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    master, gwdm, calib, pernode = [], [], [], []

    for ds, N in CITIES:
        W = pd.read_csv(pjoin(DATA_DIR, f'{ds}_W.csv'), header=None).values.astype(np.float64)
        rs = W.sum(1); rs[rs == 0] = 1.0; Wr = W / rs[:, None]

        for bk in BACKBONES:
            print(f"\n=== {ds} / {bk} ===")
            data = load_backbone(ds, N, bk)
            if data is None: continue
            (y_tr, p_tr), (y_va, p_va), (y_te, p_te) = data['train'], data['val'], data['test']
            y_int = np.round(y_te).astype(int)
            yva_int = np.round(y_va).astype(int)

            # ── build predictives: two baselines + the proposed POSEC model ──────
            M = {}   # method -> (Predictive, point, s_i or None)
            mu_te = np.maximum(p_te, 1e-6)
            M['base+Poisson'] = (CountPredictive('poisson', p_te, lam=mu_te), p_te, None)
            a_nb = nb_alpha_mle(yva_int, np.maximum(p_va, 1e-6))
            if not (ALPHA_GRID[1] < a_nb < ALPHA_GRID[-2]):
                print(f"  [warn] base NB alpha*={a_nb:.4g} near grid edge")
            n_nb = 1.0 / a_nb
            M['base+NB'] = (CountPredictive('nb', p_te, n=n_nb, pp=n_nb/(n_nb+mu_te)), p_te, None)

            # POSEC: per-cell Poisson calibration + per-node Pareto dose, NB-wrapped
            gres = guardia_predict(y_tr, p_tr, y_va, p_va, y_te, p_te, Wr, N)
            mte, mva, si = gres['lisapareto']
            mte = np.maximum(mte, 1e-6)
            ag = nb_alpha_mle(yva_int, np.maximum(mva, 1e-6)); ng = 1.0 / ag
            M[PROPOSED] = (CountPredictive('nb', mte, n=ng, pp=ng/(ng+mte)), mte, si)
            cs = gres['cstar_lisa']
            print(f"  posec gate={GUARDIA_GATE} | best_c={gres['best_c']:.2f}  "
                  f"cstar_lisa min/med/max={cs.min():.2f}/{np.median(cs):.2f}/{cs.max():.2f}")

            # ── sanity: PMF mass ~1 and finite log score ─────────────────────────
            for nm, (pr, *_ ) in M.items():
                if not pr.mass_ok():
                    sys.exit(f"[ABORT] {ds}/{bk}/{nm}: PMF mass < 0.999.")
                if not np.all(np.isfinite(pr.logscore(y_int))):
                    sys.exit(f"[ABORT] {nm}: non-finite log score.")

            # ── master rows + per-node + calibration ─────────────────────────────
            mae_base_node = np.abs(y_te - p_te).mean(0)
            rmse_base_node = np.sqrt(((y_te - p_te) ** 2).mean(0))
            pn = {'node': np.arange(N), 'MAE_base': mae_base_node, 'RMSE_base': rmse_base_node}
            for nm, (pr, pt, si) in M.items():
                als = float(pr.logscore(y_int).mean())
                mae_i = np.abs(y_te - pt).mean(0)
                rmse_i = np.sqrt(((y_te - pt) ** 2).mean(0))
                frac = float(np.mean(mae_i > mae_base_node))      # f_worse vs base (MAE)
                row = dict(city=ds, backbone=bk, method=nm, ALS_discrete=als,
                           MAE=mae(y_te, pt), RMSE=rmse(y_te, pt),
                           f_worse=frac, **spatial_metrics(y_te, pt, Wr))
                master.append(row)
                pn[f'MAE_{nm}'] = mae_i; pn[f'RMSE_{nm}'] = rmse_i
                print(f"  {nm:18s} ALS={als:.4f}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  "
                      f"MI={row['MI']:.4f}(p={row['MI_pval']:.3f})  f_worse={frac:.2f}")
                u = pr.pit(y_int)
                hist, _ = np.histogram(u, bins=10, range=(0, 1))
                crow = dict(city=ds, backbone=bk, method=nm,
                            cov80=pr.coverage(y_int, 0.80), cov95=pr.coverage(y_int, 0.95))
                for b in range(10): crow[f'pit_b{b}'] = int(hist[b])
                calib.append(crow)
            pn['cstar_lisa'] = gres['cstar_lisa']
            pn['guardia_lisac_s_i'] = gres['lisapareto'][2]
            pn_df = pd.DataFrame(pn); pn_df.insert(0, 'backbone', bk); pn_df.insert(0, 'city', ds)
            pernode.append(pn_df)

            # ── GW / DM tests: the proposed model vs each baseline ───────────────
            ls = {k: M[k][0].logscore(y_int) for k in M}
            ae = {k: np.abs(y_te - M[k][1]) for k in M}
            se = {k: (y_te - M[k][1]) ** 2 for k in M}
            for B in [b for b in M if b != PROPOSED]:
                for metric, src in (('ALS', ls), ('absErr', ae), ('sqErr', se)):
                    d_t = (src[PROPOSED] - src[B]).mean(1)     # per-time loss differential
                    mdl, t, p = hac_test(d_t)
                    gwdm.append(dict(city=ds, backbone=bk, pair=f'{PROPOSED}_vs_{B}',
                                     metric=metric, mean_diff=mdl, t=t, p=p))

    # ── write ────────────────────────────────────────────────────────────────────
    mcols = ['city', 'backbone', 'method', 'ALS_discrete', 'MAE', 'RMSE',
             'f_worse', 'MI', 'MI_pct_sig', 'MI_pval', 'PAI1', 'PAI5', 'PAI10', 'PAI25']
    pd.DataFrame(master)[mcols].to_csv(pjoin(OUT_DIR, 'als_master.csv'), index=False)
    pd.DataFrame(gwdm).to_csv(pjoin(OUT_DIR, 'gw_dm_tests.csv'), index=False)
    pd.DataFrame(calib).to_csv(pjoin(OUT_DIR, 'calibration.csv'), index=False)
    pd.concat(pernode, ignore_index=True).to_csv(pjoin(OUT_DIR, 'per_node.csv'), index=False)
    _write_readme()
    print(f"\nSaved 4 CSVs + README -> {OUT_DIR}")


def _write_readme():
    txt = """# POSEC probabilistic evaluation outputs

Generated by: `python scripts/run_probabilistic.py` (cities/backbones in posec/config.py).

## Files
- **als_master.csv** — city x backbone x method: ALS_discrete (unified discrete
  log score over integer counts, lower = better), MAE, RMSE, f_worse (fraction of
  nodes with MAE worse than the backbone), MI/MI_pct_sig/MI_pval (Moran's I of the
  residuals + significance), PAI1/5/10/25 (hotspot concentration).
  Methods: base+Poisson / base+NB (raw backbone as a count distribution,
  baselines) and guardia-lisac+NB (POSEC — the proposed per-cell Poisson
  calibration with a per-node Pareto-optimal spatial-lag dose, NB-wrapped).
- **gw_dm_tests.csv** — Giacomini-White (ALS) and Diebold-Mariano (absErr, sqErr)
  tests with Newey-West HAC variance (lag floor(T^1/3)): the proposed model vs
  each baseline.
- **calibration.csv** — randomized-PIT histogram (10 bins) and central-interval
  coverage at 80%/95% per method.
- **per_node.csv** — per node: MAE/RMSE for base and every method, plus the POSEC
  per-node dose (cstar_lisa) and gate (guardia_lisac_s_i) for dose maps.

## Notes
- POSEC gate/Pareto loss is set by GUARDIA_GATE in config ('mse' here).
- NB dispersion is estimated by grid MLE on validation (ALPHA_GRID in config).
- Anti-leak: nothing from TEST enters estimation; test-time lags come only from
  the observed val->test past.
- Smoke run: `SMOKE=1 python scripts/run_probabilistic.py` (POA_CRIME/stgcn).
"""
    with open(pjoin(OUT_DIR, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(txt)
