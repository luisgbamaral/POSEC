"""
config.py — central experiment configuration for posec.

Single source of truth for datasets, backbones, lags, grids and paths.
Add a city / backbone / change a hyperparameter HERE — not in the scripts.
Set the env var SMOKE=1 to restrict to a fast POA_CRIME/stgcn smoke run.
"""
import os
import numpy as np

# ── datasets: (name, n_nodes) ────────────────────────────────────────────────
CITIES = [('SP_CRIME', 1445), ('POA_CRIME', 94), ('BA_LESIONES', 74)]

# ── backbones (MSE-trained → mean-targeting) ─────────────────────────────────
BACKBONES = ['stgcn', 'gwavenet_mse', 'sthsl_mse']

# ── train/val/test split sizes and inference batch ───────────────────────────
N_VAL, N_TEST, BATCH = 110, 110, 50

# ── residual-correction lags (own / spatial) ────────────────────────────────
OWN_LAGS, SP_LAGS = (1, 7, 14), (1, 7)

# ── NB2 dispersion MLE grid ──────────────────────────────────────────────────
ALPHA_GRID = np.logspace(-4, 1, 60)

# ── GUARD IA: EB pool min nodes, parallel jobs, gate/Pareto loss ─────────────
EB_MIN, GUARDIA_NJOBS, GUARDIA_GATE = 20, 4, 'mse'

# ── Anscombe offset ──────────────────────────────────────────────────────────
A_ANS = 0.375

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = './data'
CKPT_DIR  = './checkpoints'
OUT_DIR = './results/probabilistic'

# ── smoke test: isolated POA_CRIME/stgcn for fast golden-regression checks ───
if os.environ.get('SMOKE'):
    CITIES, BACKBONES = [('POA_CRIME', 94)], ['stgcn']

# ── env overrides for the overnight pipeline (run different dataset sets / splits) ──
# POSEC_CITIES="CHI_CRIME:1400,SP_CRIME_7D:1445"   POSEC_NVAL=16 POSEC_NTEST=16
if os.environ.get('POSEC_CITIES'):
    CITIES = [(p.split(':')[0], int(p.split(':')[1]))
              for p in os.environ['POSEC_CITIES'].split(',') if p]
if os.environ.get('POSEC_BACKBONES'):
    BACKBONES = os.environ['POSEC_BACKBONES'].split(',')
if os.environ.get('POSEC_NVAL'):
    N_VAL = int(os.environ['POSEC_NVAL'])
if os.environ.get('POSEC_NTEST'):
    N_TEST = int(os.environ['POSEC_NTEST'])
if os.environ.get('POSEC_OUT'):
    OUT_DIR = os.environ['POSEC_OUT']
