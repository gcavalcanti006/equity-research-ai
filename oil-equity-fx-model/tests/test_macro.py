import numpy as np
import pandas as pd

from src import macro, macro_test, sources
from tests.synth import make_cfg


def test_us_yoy_and_br_yoy_values():
    idx = pd.date_range("2020-01-01", periods=24, freq="MS")
    cpi = pd.Series(100 * 1.002 ** np.arange(24), index=idx)
    assert np.isclose(macro.us_yoy(cpi).iloc[-1], 100 * (1.002 ** 12 - 1))
    assert macro.us_yoy(cpi).iloc[:12].isna().all()                       # needs a full year
    ipca = pd.Series(0.5, index=idx)
    assert np.isclose(macro.br_yoy(ipca).iloc[-1], 100 * (1.005 ** 12 - 1))
    assert macro.br_yoy(ipca).iloc[:11].isna().all()


def test_known_from_waits_for_the_release():
    m = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
    k = macro.known_from(m, 20)
    assert list(k.index) == [pd.Timestamp("2024-02-20"), pd.Timestamp("2024-03-20")]   # month end + 20 days


def test_inflation_levels_have_no_look_ahead():
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    cpi = pd.Series(100 * np.cumprod(1 + np.r_[0, np.random.default_rng(0).normal(0.003, 0.002, 29)]), index=idx)
    lv = macro.inflation_levels(us_cpi=cpi, lag_days=20)["us_infl"]
    yoy = macro.us_yoy(cpi)
    last_month = pd.Timestamp("2022-05-01")                                # its value must appear only after 2022-06-19
    assert lv.index.min() > pd.Timestamp("2021-01-31")
    assert pd.Timestamp("2022-06-19") not in lv.index or lv.loc[pd.Timestamp("2022-06-19")] != yoy.loc[last_month]
    assert np.isclose(lv.loc[pd.Timestamp("2022-06-20")], yoy.loc[last_month])


def test_election_flag():
    f = macro.election_flag(pd.to_datetime(["2017-05-01", "2018-05-01", "2019-01-01"]), [2018])
    assert list(f.values) == [0.0, 1.0, 0.0]


def test_holm_is_never_smaller_than_raw_and_monotone():
    p = np.array([0.001, 0.02, 0.04, 0.5])
    adj = macro_test.holm(p)
    assert (adj >= p).all() and np.isclose(adj[0], 0.004) and adj.max() <= 1


def test_bcb_is_fetched_in_chunks(monkeypatch):
    calls = []

    class R:
        def raise_for_status(self): pass
        def json(self): return [{"data": "01/01/2020", "valor": "0.5"}]
    monkeypatch.setattr(sources.requests, "get", lambda url, params, timeout: calls.append(params) or R())
    sources.fetch_bcb(433, "2005-01-01", "2024-12-31")
    assert len(calls) >= 3 and calls[0]["dataInicial"] == "01/01/2005"


def test_load_all_builds_inflation_levels(tmp_path, monkeypatch):
    idx = pd.date_range("2019-01-01", periods=40, freq="MS")
    monkeypatch.setattr(sources, "fetch_fred", lambda sid, s, e: pd.Series(100 * 1.002 ** np.arange(40), index=idx))
    monkeypatch.setattr(sources, "fetch_bcb", lambda c, s, e: pd.Series(0.4, index=idx))
    monkeypatch.setattr(sources, "fetch_yahoo", lambda t, s, e: pd.Series(
        [1.0] * 5, index=pd.bdate_range("2022-05-02", periods=5)))
    cfg = {"snapshot_dir": str(tmp_path / "snap"), "yahoo": {"x": "X"}, "fred_monthly": {"us_cpi_raw": "CPIAUCNS"},
           "bcb_monthly": {"br_ipca_raw": 433}, "inflation_release_lag_days": 20}
    levels, problems = sources.load_all(cfg, "2022-05-01", "2022-06-01")
    assert not problems and "us_cpi_raw" not in levels
    assert levels["us_infl"].notna().any() and levels["br_infl"].notna().any()
    assert levels["us_infl"].dropna().iloc[-1] > 0


def _world(n=3500, seed=3, infl_effect=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2008-01-02", periods=n)
    mkt = rng.normal(0, 0.01, n)
    oil = 0.2 * mkt + rng.normal(0, 0.02, n)
    infl = np.zeros(n)
    infl[::21] = rng.normal(0, 0.4, len(infl[::21]))                       # a release roughly every month
    y = 0.9 * mkt + 0.3 * oil + infl_effect * infl + rng.normal(0, 0.008, n)
    X = pd.DataFrame({"ibov": mkt, "brent": oil, "br_infl": infl}, index=idx)
    return X, pd.Series(y, index=idx)


def _cfg(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.update({"long_step_days": 5, "long_min_train_days": 1000, "long_refit_every": 20, "long_ridge_alpha": 1.0,
                "elections": {"BR": [2010, 2014, 2018], "US": []}})
    return cfg


def test_finds_a_real_inflation_effect_and_not_a_fake_election_effect(tmp_path):
    X, y = _world(infl_effect=0.3)
    r = macro_test.run_stock(X, y, ["ibov"], ["brent"], ["br_infl"], "BR", _cfg(tmp_path), 21)
    assert r["explain_gain_infl"] > 0 and r["explain_p_infl"] < 0.01
    assert r["explain_p_elec"] > 0.01 or r["explain_gain_elec"] <= 0
    assert abs(r["br_infl_coef"] - 0.3) < 0.1 and 0 < r["election_share"] < 1


def test_no_effect_means_no_gain(tmp_path):
    X, y = _world(infl_effect=0.0)
    r = macro_test.run_stock(X, y, ["ibov"], ["brent"], ["br_infl"], "BR", _cfg(tmp_path), 21)
    assert r["explain_p_infl"] > 0.01 or r["explain_gain_infl"] <= 0


def test_runs_without_inflation_columns(tmp_path):
    X, y = _world()
    r = macro_test.run_stock(X[["ibov", "brent"]], y, ["ibov"], ["brent"], [], "BR", _cfg(tmp_path), 21)
    assert "explain_p_elec" in r and "explain_p_infl" not in r


def test_too_little_history_is_reported(tmp_path):
    X, y = _world(400)
    assert "error" in macro_test.run_stock(X, y, ["ibov"], ["brent"], ["br_infl"], "BR", _cfg(tmp_path), 63)


def test_tidy_applies_holm():
    rows = [{"stock": "a", "country": "BR", "sector": "oil", "horizon_days": 21,
             "explain_r2_base": 0.3, "explain_r2_infl": 0.31, "explain_gain_infl": 1e-4, "explain_p_infl": 0.03,
             "predict_r2_base": 0.0, "predict_r2_infl": 0.0, "predict_gain_infl": -1e-5, "predict_p_infl": 0.6}]
    t = macro_test.tidy(rows, 0.01)
    assert len(t) == 2 and (t["p_holm"] >= t["p"]).all() and not t["helps_after_correction"].any()
    assert t["helps_p05"].sum() == 1
