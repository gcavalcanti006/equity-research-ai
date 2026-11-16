import numpy as np
import pandas as pd
import pytest

from src import factors, scenarios
from tests.synth import make_cfg, make_levels


def _setup(tmp_path, **kw):
    cfg = make_cfg(tmp_path)
    d = factors.make_design(make_levels(**kw), "petr4", cfg)
    return cfg, d, factors.rate_names(cfg, d["X"].columns)


def _rows(tmp_path, **kw):
    cfg, d, rates = _setup(tmp_path, **kw)
    rows = scenarios.run_scenarios(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg, cfg["scenarios"], rates)
    return cfg, d, pd.DataFrame(rows)


def test_shock_units():
    assert scenarios.shock_to_change("brent", -0.20, set()) == pytest.approx(np.log(0.8))
    assert scenarios.shock_to_change("us10y", 1.0, {"us10y"}) == 1.0          # +100bp stays +1.0 pp


def test_oil_shock_matches_the_true_beta_when_only_oil_moves(tmp_path):
    _, _, df = _rows(tmp_path, n=1500, brent_beta=0.30, noise=0.010)
    r = df[(df.scenario == "oil_down") & (df.reading == "only_shocked")].iloc[0]
    expected = np.expm1(0.30 * np.log(0.8)) * 100                              # about -6.5%
    assert r.impact_pct == pytest.approx(expected, abs=1.5)
    assert r.low_80_pct < r.impact_pct < r.high_80_pct


def test_correlated_factors_make_an_oil_drop_worse(tmp_path):
    # in the fake world Brent moves with the market, and the stock has a big market beta
    _, _, df = _rows(tmp_path, n=1500, brent_beta=0.30, noise=0.010)
    a = df[(df.scenario == "oil_down") & (df.reading == "only_shocked")].iloc[0].impact_pct
    b = df[(df.scenario == "oil_down") & (df.reading == "with_correlations")].iloc[0].impact_pct
    assert b < a < 0


def test_stock_ignores_scenarios_about_factors_it_does_not_use(tmp_path):
    _, _, df = _rows(tmp_path, n=800)
    assert "soy_down" not in set(df.scenario)               # petr4 has no soy factor
    assert {"oil_down", "brazil_stress", "us_rates_up"} <= set(df.scenario)


def test_horizon_changes_the_range_not_the_expected_impact(tmp_path):
    cfg, d, rates = _setup(tmp_path, n=1000)
    one = {"s": {"shocks": {"brent": -0.2}, "horizon_days": 1}}
    many = {"s": {"shocks": {"brent": -0.2}, "horizon_days": 25}}
    a = scenarios.run_scenarios(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg, one, rates)[0]
    b = scenarios.run_scenarios(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg, many, rates)[0]
    assert a["impact_pct"] == pytest.approx(b["impact_pct"], abs=1e-6)
    width_a = np.log1p(a["high_80_pct"] / 100) - np.log1p(a["low_80_pct"] / 100)
    width_b = np.log1p(b["high_80_pct"] / 100) - np.log1p(b["low_80_pct"] / 100)
    assert width_b / width_a == pytest.approx(5.0, rel=1e-3)                  # sqrt(25)


def test_a_huge_shock_is_flagged_as_beyond_history(tmp_path):
    cfg, d, rates = _setup(tmp_path, n=1000)
    spec = {"crash": {"shocks": {"brent": -0.90}, "horizon_days": 22}}
    row = scenarios.run_scenarios(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg, spec, rates)[0]
    assert "brent" in row["beyond_history"]


def test_proxy_substitution_is_reported(tmp_path):
    cfg, d, rates = _setup(tmp_path, n=800)
    spec = {"s": {"shocks": {"oil": -0.2}}}
    row = scenarios.run_scenarios(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg, spec, rates,
                                  substituted={"oil": "brent"})[0]
    assert "proxy brent" in row["note"] and row["impact_pct"] < 0


def test_auto_episodes_do_not_overlap_and_respect_history(tmp_path):
    cfg, d, _ = _setup(tmp_path, n=1200)
    eps = scenarios.auto_episodes(d["X"], "brent", 10, 2, min_train=300)
    assert len(eps) == 4
    pos = [(d["X"].index.get_loc(a), d["X"].index.get_loc(b)) for _, a, b in eps]
    assert all(a >= 300 for a, _ in pos)
    for i, (a1, b1) in enumerate(pos):
        for a2, b2 in pos[i + 1:]:
            assert b1 < a2 or b2 < a1


def test_backtest_never_uses_the_stocks_own_future(tmp_path):
    cfg, d, _ = _setup(tmp_path, n=1200)
    X, y = d["X"], d["y"]
    start, end = X.index[700], X.index[709]
    base = scenarios.backtest_episode(X, y, d["market_cols"], d["other_cols"], cfg, start, end, 300)
    y2 = y.copy()
    y2.iloc[700:] = y2.iloc[700:] + 0.5                       # wreck the stock's returns from the window on
    other = scenarios.backtest_episode(X, y2, d["market_cols"], d["other_cols"], cfg, start, end, 300)
    assert base["model_pct"] == pytest.approx(other["model_pct"]) and base["capm_pct"] == pytest.approx(other["capm_pct"])
    assert base["actual_pct"] != pytest.approx(other["actual_pct"])


def test_backtest_skips_windows_without_enough_history(tmp_path):
    cfg, d, _ = _setup(tmp_path, n=800)
    r = scenarios.backtest_episode(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg,
                                   d["X"].index[50], d["X"].index[60], 300)
    assert "skipped" in r


def test_backtest_on_a_world_with_a_real_oil_effect_beats_capm_and_covers_80(tmp_path):
    cfg, d, _ = _setup(tmp_path, n=2000, brent_beta=0.30, noise=0.010)
    X, y = d["X"], d["y"]
    eps = scenarios.auto_episodes(X, "brent", 10, 4, 400)
    res = [scenarios.backtest_episode(X, y, d["market_cols"], d["other_cols"], cfg, a, b, 400) for _, a, b in eps]
    summ = scenarios.summarize_backtest(pd.DataFrame(res))
    assert summ["episodes"] == 8
    assert summ["mae_model_pp"] < summ["mae_capm_pp"] < summ["mae_zero_pp"]
    assert summ["direction_hit_rate"] >= 0.75 and 0.5 <= summ["coverage_80"] <= 1.0


def test_reliability_needs_a_small_p_value_and_a_better_r2(tmp_path):
    from src import run_scenarios
    pd.DataFrame({"stock": ["a", "b", "c"], "p_factors_beats_capm": [0.001, 0.03, 0.001],
                  "oos_r2_factors": [0.3, 0.3, 0.1], "oos_r2_capm": [0.1, 0.1, 0.2]}
                 ).to_csv(tmp_path / "summary.csv", index=False)
    assert run_scenarios.reliability(tmp_path) == {"a": True, "b": False, "c": False}
    assert run_scenarios.reliability(tmp_path, 0.05)["b"] is True or run_scenarios.reliability(tmp_path, 0.05)["b"] == True  # noqa: E712
    assert run_scenarios.reliability(tmp_path / "nothing") == {}
