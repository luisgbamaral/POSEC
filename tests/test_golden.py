"""test_golden.py — golden-regression lock.

The SMOKE run (POA_CRIME / stgcn) must reproduce
tests/fixtures/golden_smoke_poa_stgcn_als.csv within GPU-noise tolerance
(the headline metrics are reported to 3-4 significant figures; TF32 backbone
inference is non-deterministic at ~1e-6, so we compare with atol=1e-3).

Skipped automatically if the POA_CRIME data/checkpoint are absent (both are
provided separately — see README).  Run:  pytest tests/test_golden.py -q
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "golden_smoke_poa_stgcn_als.csv"
ATOL = 1e-3


def _inputs_present():
    return (ROOT / "data" / "POA_CRIME_V.csv").exists() and (ROOT / "checkpoints" / "POA_CRIME").is_dir()


@pytest.mark.skipif(not _inputs_present(), reason="POA_CRIME data/checkpoint not present (see README)")
def test_golden_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "SMOKE": "1", "POSEC_OUT": tmp,
               "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"}
        subprocess.run([sys.executable, "scripts/run_probabilistic.py"],
                       cwd=str(ROOT), env=env, check=True)
        got = pd.read_csv(Path(tmp) / "als_master.csv").set_index("method")
    exp = pd.read_csv(FIXTURE).set_index("method")
    assert list(got.index) == list(exp.index), "method set changed"
    for col in ["ALS_discrete", "MAE", "RMSE", "MI", "PAI10"]:
        assert np.allclose(got[col].values, exp[col].values, atol=ATOL), f"{col} drifted"
