import numpy as np
import pandas as pd

from src import earnings_dates as ed
from src import earnings_study as es
from tests.synth import make_cfg


def _world(n=2600, seed=5, jump=0.04, drift=0.0, pre=0.0, every=63):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2012-01-02", periods=n)
    mkt = rng.normal(0, 0.01, n)
    oil = 0.2 * mkt + rng.normal(0, 0.02, n)
    y = 0.9 * mkt + 0.3 * oil + rng.normal(0, 0.008, n)
    event_pos = np.arange(400, n - 40, every)
    for p in event_pos:
        J = rng.normal(0, jump)
        y[p] += J                                   # reaction on the release day
        y[p + 2:p + 12] += drift * J / 10           # drift after the reaction
        y[p - 10:p] += pre * J / 10                 # run-up before it
    X = pd.DataFrame({"ibov": mkt, "brent": oil}, index=idx)
    return X, pd.Series(y, index=idx), idx[event_pos]


def _cfg(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.update({"earnings_train_days": 750, "earnings_min_train_days": 250, "ridge_alpha": 1.0})
    return cfg


def test_reaction_is_bigger_than_normal_and_drift_is_found(tmp_path):
    X, y, dates = _world(drift=0.6)
    ev = es.event_table(X, y, ["ibov"], ["brent"], "brent", dates, _cfg(tmp_path))
    s = es.summarize_stock(ev)
    assert s["n_events"] > 25 and s["move_ratio"] > 2.0
    assert 0.3 < s["drift_slope"] < 0.9 and s["drift_p"] < 0.05
    assert abs(s["mean_car_react"]) < 0.02


def test_no_event_effect_means_move_ratio_near_one_and_no_drift(tmp_path):
    X, y, dates = _world(jump=0.0)
    s = es.summarize_stock(es.event_table(X, y, ["ibov"], ["brent"], "brent", dates, _cfg(tmp_path)))
    assert 0.7 < s["move_ratio"] < 1.4 and s["drift_p"] > 0.01


def test_abnormal_return_removes_what_the_factors_explain(tmp_path):
    X, y, dates = _world(jump=0.0)
    ev = es.event_table(X, y, ["ibov"], ["brent"], "brent", dates, _cfg(tmp_path))
    assert (ev["ret_react"].abs().mean()) > 0 and ev["car_react"].abs().mean() < ev["ret_react"].abs().mean() * 1.2
    assert ev["sigma_daily"].between(0.005, 0.012).all()


def test_model_uses_only_days_before_the_event(tmp_path):
    X, y, dates = _world(jump=0.0)
    y2 = y.copy()
    last = dates[-1]
    pos = y.index.get_loc(last)
    y2.iloc[pos - 5:] = y2.iloc[pos - 5:] + 0.5              # corrupt the days at/after the last event's pre-window
    a = es.event_table(X, y, ["ibov"], ["brent"], "brent", dates[:-1], _cfg(tmp_path))
    b = es.event_table(X, y2, ["ibov"], ["brent"], "brent", dates[:-1], _cfg(tmp_path))
    assert np.allclose(a["sigma_daily"], b["sigma_daily"]) and np.allclose(a["car_react"], b["car_react"])


def test_events_without_enough_history_or_outside_the_data_are_skipped(tmp_path):
    X, y, dates = _world()
    early, late = pd.Timestamp("2012-03-01"), pd.Timestamp("2030-01-01")
    ev = es.event_table(X, y, ["ibov"], ["brent"], "brent", [early, late], _cfg(tmp_path))
    assert ev.empty and es.summarize_stock(ev) == {"n_events": 0}


def test_correlation_before_and_after_is_computed(tmp_path):
    X, y, dates = _world()
    ev = es.event_table(X, y, ["ibov"], ["brent"], "brent", dates, _cfg(tmp_path))
    assert ev[["corr_mkt_pre", "corr_mkt_post", "r2_pre", "r2_post"]].notna().all().all()
    assert ev["corr_mkt_pre"].mean() > 0.5 and ev["corr_mkt_post"].mean() > 0.5


def test_pool_clusters_by_quarter(tmp_path):
    frames = []
    for k, name in enumerate(["a", "b", "c"]):
        X, y, dates = _world(seed=10 + k, drift=0.6)
        ev = es.event_table(X, y, ["ibov"], ["brent"], "brent", dates, _cfg(tmp_path))
        ev.insert(0, "stock", name); ev.insert(1, "country", "US"); ev.insert(2, "sector", "oil")
        frames.append(ev)
    pooled = es.pool(pd.concat(frames, ignore_index=True))
    allrow = pooled[pooled["group"] == "all"].iloc[0]
    assert allrow["n_events"] > 60 and allrow["drift_slope"] > 0.3 and allrow["drift_p"] < 0.05
    assert {"all", "country=US", "sector=oil"} <= set(pooled["group"])


def test_yahoo_table_is_cleaned():
    idx = pd.DatetimeIndex(pd.to_datetime(["2030-01-30 16:00", "2024-10-30 16:00", "2024-07-30 16:00"])).tz_localize("America/New_York")
    raw = pd.DataFrame({"EPS Estimate": [1.0, 1.1, 1.0], "Reported EPS": [np.nan, 1.2, 0.9], "Surprise(%)": [np.nan, 9.1, -10.0]}, index=idx)
    out = ed.normalize_yahoo(raw, today="2026-11-19")
    assert list(out["date"]) == [pd.Timestamp("2024-07-30"), pd.Timestamp("2024-10-30")]      # the future one is dropped
    assert list(out["surprise_pct"]) == [-10.0, 9.1]


def test_dates_snapshot_manual_merge_and_failure(tmp_path):
    cfg = {"earnings_dates_dir": str(tmp_path / "d"), "earnings_manual_csv": str(tmp_path / "m.csv")}
    (tmp_path / "m.csv").write_text("stock,date\nPETR4,2024-05-10\nxom,2024-05-03\npetr4,2024-08-09\n")
    boom = lambda t: (_ for _ in ()).throw(RuntimeError("no data"))
    t, note = ed.load_dates("petr4", "PETR4.SA", cfg, fetch=boom)
    assert len(t) == 2 and "yahoo failed" in note and "manual +2" in note
    t, note = ed.load_dates("xom", "XOM", cfg, fetch=lambda tk: pd.DataFrame({"date": pd.to_datetime(["2024-05-03", "2024-02-02"]),
                                                                           "eps_reported": [1.0, 1.0]}))
    assert len(t) == 2 and note.startswith("yahoo")                      # duplicate of the manual date merged
    t2, note2 = ed.load_dates("xom", "XOM", cfg, fetch=boom)
    assert note2.startswith("snapshot") and len(t2) == 2
    t3, _ = ed.load_dates("nope", "NOPE", cfg, fetch=boom)
    assert t3.empty
