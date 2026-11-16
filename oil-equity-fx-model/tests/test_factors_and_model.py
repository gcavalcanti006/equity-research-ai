import numpy as np
import pandas as pd
import pytest

from src import factor_model as fm
from src import factors, run_factors
from tests.synth import make_cfg, make_levels


def test_align_does_not_create_zero_return_on_foreign_holiday():
    cal = pd.bdate_range("2024-01-01", periods=6)
    lv = pd.DataFrame({"sp500": [100, 101, np.nan, 103, 104, 105]}, index=cal)   # US closed on day 3
    al = factors.align_levels(lv, cal, max_ffill_days=2)
    assert al["sp500"].iloc[2] == 101                      # filled with the last known level
    ch = factors.to_changes(al, rate_series=set())
    assert ch["sp500"].iloc[2] == 0.0                       # the move shows up as a flat day...
    assert ch["sp500"].iloc[3] == pytest.approx(np.log(103 / 101))   # ...and the catch-up lands on day 4


def test_rates_are_differences_not_log_returns():
    cal = pd.bdate_range("2024-01-01", periods=3)
    lv = pd.DataFrame({"us10y": [4.20, 4.25, 4.10]}, index=cal)
    ch = factors.to_changes(lv, rate_series={"us10y"})
    assert ch["us10y"].iloc[1] == pytest.approx(0.05)
    assert ch["us10y"].iloc[2] == pytest.approx(-0.15)


def test_slope_is_built_from_the_two_yields():
    lv = make_levels(50)
    out = factors.add_derived_levels(lv)
    assert (out["us_slope"] == lv["us10y"] - lv["us2y"]).all()


def test_design_has_lags_and_reports_missing_factors(tmp_path):
    cfg, lv = make_cfg(tmp_path), make_levels(400)
    d = factors.make_design(lv, "petr4", cfg)
    assert "ibov" in d["market_cols"] and "ibov" not in d["other_cols"]
    assert "sp500_lag1" in d["other_cols"] and "us10y_lag1" in d["other_cols"]
    assert "us_slope" in d["other_cols"] and "brent" in d["other_cols"]
    assert d["missing"] == []
    d2 = factors.make_design(lv.drop(columns=["brent"]), "petr4", cfg)    # a failed download
    assert d2["missing"] == ["brent"] and "brent" not in d2["X"].columns


def test_design_errors_when_market_index_is_missing(tmp_path):
    cfg, lv = make_cfg(tmp_path), make_levels(400)
    assert "error" in factors.make_design(lv.drop(columns=["ibov"]), "petr4", cfg)


def _design(tmp_path, **kw):
    cfg = make_cfg(tmp_path)
    return cfg, factors.make_design(make_levels(**kw), "petr4", cfg)


def test_orthogonalisation_is_learned_on_training_only(tmp_path):
    cfg, d = _design(tmp_path, n=900)
    X, y = d["X"], d["y"]
    m = fm.FactorModel(d["market_cols"], d["other_cols"]).fit(X.iloc[:500], y.iloc[:500])
    Z = m._transform(X.iloc[:500])
    for c in d["other_cols"]:                              # cleaned factors have no market part in training
        assert abs(np.corrcoef(Z[c], X["ibov"].iloc[:500])[0, 1]) < 1e-8
    before = m.ortho_["brent"].copy()
    m.predict(X.iloc[500:])                                # predicting must not change what was learned
    assert (m.ortho_["brent"] == before).all()


def test_walk_forward_first_block_equals_manual_fit_on_the_past(tmp_path):
    cfg, d = _design(tmp_path, n=800)
    X, y = d["X"], d["y"]
    f = fm.default_models(d["market_cols"], d["other_cols"], cfg)
    preds = fm.walk_forward(X, y, f, min_train=300, step=21)
    first = preds[(preds.model == "factors")].iloc[:21]
    manual = f["factors"]().fit(X.iloc[:300], y.iloc[:300]).predict(X.iloc[300:321])
    assert np.allclose(first["pred"].values, manual)
    assert list(first["date"]) == list(y.index[300:321])


def test_finds_the_true_oil_effect_and_beats_capm_out_of_sample(tmp_path):
    cfg, d = _design(tmp_path, n=1200, brent_beta=0.30, noise=0.010)
    X, y = d["X"], d["y"]
    table, r2, r2_adj, _ = fm.full_sample_report(X, y, d["market_cols"], d["other_cols"], True)
    assert table.loc["brent", "coef"] == pytest.approx(0.30, abs=0.05)
    assert table.loc["brent", "p_value"] < 0.001
    ev = fm.evaluate(fm.walk_forward(X, y, fm.default_models(d["market_cols"], d["other_cols"], cfg), 300, 21))
    assert ev.loc["factors", "r2_oos_vs_mean"] > ev.loc["capm", "r2_oos_vs_mean"] > 0
    assert ev.loc["factors", "dm_p_vs_capm"] < 0.05 and ev.loc["factors", "dm_gain_vs_capm"] > 0


def test_pure_noise_does_not_look_like_a_win(tmp_path):
    cfg, d = _design(tmp_path, n=1200, signal=False, noise=0.015, seed=5)
    X, y = d["X"], d["y"]
    ev = fm.evaluate(fm.walk_forward(X, y, fm.default_models(d["market_cols"], d["other_cols"], cfg), 300, 21))
    # extra factors that carry no information must not beat the market-only model by a significant margin
    assert not (ev.loc["factors", "dm_gain_vs_capm"] > 0 and ev.loc["factors", "dm_p_vs_capm"] < 0.01)


def test_interval_coverage_is_roughly_80_percent(tmp_path):
    cfg, d = _design(tmp_path, n=1500)
    X, y = d["X"], d["y"]
    ev = fm.evaluate(fm.walk_forward(X, y, fm.default_models(d["market_cols"], d["other_cols"], cfg), 300, 21))
    assert 0.74 < ev.loc["factors", "coverage_80"] < 0.86


def test_rolling_coefficients_shape(tmp_path):
    cfg, d = _design(tmp_path, n=600)
    _, _, _, fitted = fm.full_sample_report(d["X"], d["y"], d["market_cols"], d["other_cols"], True)
    r = fm.rolling_coefficients(d["X"], d["y"], fitted, window=120, every=10)
    assert "brent" in r.columns and len(r) > 30


def test_run_stock_end_to_end_writes_all_outputs(tmp_path):
    cfg = make_cfg(tmp_path)
    row = run_factors.run_stock("petr4", make_levels(900), cfg, tmp_path / "out")
    assert "error" not in row and row["oos_r2_factors"] > row["oos_r2_capm"]
    for f in ("ols_full_sample.csv", "vif.csv", "rolling_coefficients.csv", "walk_forward_metrics.csv",
              "walk_forward_predictions.csv", "correlations.csv"):
        assert (tmp_path / "out" / "petr4" / f).exists()


def test_run_stock_reports_too_little_data(tmp_path):
    cfg = make_cfg(tmp_path)
    assert "error" in run_factors.run_stock("petr4", make_levels(200), cfg, tmp_path / "out")


def test_unknown_sector_gives_a_clear_error(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg["stocks"]["vale3"] = {"ticker": "VALE3.SA", "country": "BR", "sector": "mining"}
    lv = make_levels(400)
    lv["vale3"] = lv["petr4"]
    d = factors.make_design(lv, "vale3", cfg)
    assert "sector 'mining'" in d["error"]


def test_comparison_models_are_in_the_same_walk_forward(tmp_path):
    cfg, d = _design(tmp_path, n=1200, brent_beta=0.30, noise=0.010)
    X, y = d["X"], d["y"]
    models = fm.default_models(d["market_cols"], d["other_cols"], cfg, ["brent"], ["usdbrl"])
    assert set(models) == {"mean", "capm", "factors", "factors_nolag", "factors_ridge", "sector_fx"}
    ev = fm.evaluate(fm.walk_forward(X, y, models, 300, 21))
    assert ev.loc["sector_fx", "dm_gain_factors_over_it"] > 0 and ev.loc["sector_fx", "dm_p_factors_over_it"] < 0.05
    # in this fake world yesterday's values carry no information: dropping them changes little
    assert abs(ev.loc["factors_nolag", "r2_oos_vs_mean"] - ev.loc["factors", "r2_oos_vs_mean"]) < 0.03


def test_run_stock_summary_has_the_comparison_columns(tmp_path):
    cfg = make_cfg(tmp_path)
    row = run_factors.run_stock("petr4", make_levels(900), cfg, tmp_path / "out")
    for k in ("oos_r2_sector_fx", "oos_r2_factors_nolag", "p_factors_beats_capm",
              "p_factors_beats_sector_fx", "p_factors_beats_factors_nolag"):
        assert k in row
