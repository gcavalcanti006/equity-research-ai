import numpy as np
import pandas as pd

from src import long_horizon as lh
from tests.synth import make_cfg


def _world(n=3000, seed=1, predictive=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=n)
    mkt = rng.normal(0, 0.01, n)
    oil = 0.2 * mkt + rng.normal(0, 0.02, n)
    X = pd.DataFrame({"ibov": mkt, "brent": oil}, index=idx)
    y = 0.9 * mkt + 0.3 * oil + rng.normal(0, 0.008, n)            # contemporaneous relationship
    if predictive:                                                 # last month's oil move also drives next month
        y = y + 0.03 * pd.Series(oil, index=idx).rolling(21).sum().shift(1).fillna(0).values
    return X, pd.Series(y, index=idx)


def _cfg(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.update({"long_step_days": 5, "long_min_train_days": 1000, "long_refit_every": 20, "long_ridge_alpha": 1.0})
    return cfg


def test_pairs_use_trailing_and_forward_windows():
    X, y = _world(200)
    p = lh.build_pairs(X, y, 21, 5)
    first = p.iloc[0]
    pos = int(first["pos"])
    assert np.isclose(first["f_brent"], X["brent"].iloc[pos - 20:pos + 1].sum())
    assert np.isclose(first["y_now"], y.iloc[pos - 20:pos + 1].sum())
    assert np.isclose(first["y_next"], y.iloc[pos + 1:pos + 22].sum())


def test_explains_but_does_not_predict_when_the_link_is_only_contemporaneous(tmp_path):
    X, y = _world()
    res, coefs = lh.run_stock(X, y, ["ibov"], ["brent"], _cfg(tmp_path), 21)
    assert res["explain_r2_full"] > 0.5
    assert res["predict_r2_full"] < 0.05 and res["predict_p_full_vs_mean"] > 0.01
    assert abs(coefs.loc["brent", "coef"] - 0.3) < 0.1
    assert "does not predict" in lh.verdict(res, 0.01)


def test_detects_a_real_predictive_signal(tmp_path):
    X, y = _world(n=6000, predictive=True)
    res, _ = lh.run_stock(X, y, ["ibov"], ["brent"], _cfg(tmp_path), 21)
    assert res["predict_r2_full"] > 0.02
    assert res["predict_p_full_vs_mean"] < 0.05


def test_too_little_history_is_reported_not_crashed(tmp_path):
    X, y = _world(400)
    res, coefs = lh.run_stock(X, y, ["ibov"], ["brent"], _cfg(tmp_path), 63)
    assert "error" in res and coefs is None
