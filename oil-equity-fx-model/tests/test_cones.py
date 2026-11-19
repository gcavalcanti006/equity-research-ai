import numpy as np
import pandas as pd
import pytest

from src import cones


def _returns(n=3000, seed=0, sigma=0.015, cluster=False):
    rng = np.random.default_rng(seed)
    vol = np.full(n, sigma)
    if cluster:                                     # calm years, then a wild year
        vol[1500:1800] = sigma * 3
    r = rng.normal(0, vol)
    return pd.Series(r, index=pd.bdate_range("2012-01-02", periods=n))


def test_prices_and_returns():
    p = pd.Series([100.0, 101.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    r = cones.log_returns(p)
    assert np.isclose(r.iloc[0], np.log(1.01)) and len(r) == 2


def test_simulation_shapes_zero_mean_and_reproducible():
    r = _returns()
    for method in ("bootstrap", "fhs"):
        a = cones.simulate_returns(r, 21, 4000, method=method, seed=3)
        b = cones.simulate_returns(r, 21, 4000, method=method, seed=3)
        assert a.shape == (4000, 21) and np.array_equal(a, b)
        assert abs(a.sum(axis=1).mean()) < 0.01
    with pytest.raises(ValueError):
        cones.simulate_returns(r, 5, 10, method="nope")


def test_cone_width_matches_the_volatility_and_grows_with_horizon():
    r = _returns(sigma=0.015)
    sims = cones.simulate_returns(r, 63, 6000, method="bootstrap", seed=1)
    assert abs(sims[:, :21].sum(axis=1).std() - 0.015 * np.sqrt(21)) < 0.015 * np.sqrt(21) * 0.1
    c = cones.cone(cones.price_paths(100.0, sims))
    w = c["p95"] - c["p5"]
    assert w.iloc[-1] > 2 * w.iloc[4] and abs(c["p50"].iloc[-1] - 100) < 3
    assert (c["p5"] < c["p25"]).all() and (c["p75"] < c["p95"]).all()


def test_fhs_widens_after_wild_days_and_narrows_after_calm_ones():
    wild = _returns(cluster=True).iloc[:1790]       # ends in the middle of the wild stretch
    calm = _returns(cluster=True).iloc[:1400]
    w = cones.simulate_returns(wild, 10, 4000, method="fhs", seed=1).sum(axis=1).std()
    c = cones.simulate_returns(calm, 10, 4000, method="fhs", seed=1).sum(axis=1).std()
    assert w > 1.5 * c
    # a plain bootstrap does not react that fast
    wb = cones.simulate_returns(wild, 10, 4000, method="bootstrap", seed=1).sum(axis=1).std()
    assert w > wb


def test_scenario_drift_moves_the_centre():
    r = _returns()
    sims = cones.simulate_returns(r, 22, 3000, method="bootstrap", seed=2)
    d = cones.scenario_drift(22, -0.10, 22)
    assert np.isclose(d.sum(), -0.10)
    paths = cones.price_paths(100.0, sims, d)
    assert 88 < np.median(paths[:, -1]) < 92
    probs = cones.end_probabilities(paths, 100.0)
    assert probs["below_0%"] > 0.8 and probs["above_+10%"] < 0.01


def test_calibration_hits_close_to_nominal_on_iid_data():
    r = _returns(n=2200, seed=4)
    cal = cones.calibrate(r, 21, "bootstrap", lookback=1000, step=10, n_paths=600)
    cov = cones.coverage(cal)
    assert 0.7 < cov["coverage_80"] < 0.9 and 0.4 < cov["coverage_50"] < 0.6 and cov["n_dates"] > 80


def test_calibration_of_fhs_beats_bootstrap_when_volatility_clusters():
    r = _returns(n=2600, seed=5, cluster=True)
    fhs = cones.coverage(cones.calibrate(r, 21, "fhs", lookback=1000, step=10, n_paths=600))
    boot = cones.coverage(cones.calibrate(r, 21, "bootstrap", lookback=1000, step=10, n_paths=600))
    assert fhs["coverage_80_turbulent"] >= boot["coverage_80_turbulent"] - 0.02
