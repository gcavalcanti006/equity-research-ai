"""Event study around earnings releases, with the project's factors taken into account.

For every release date we ask: how did the stock move compared with what the factor model expected?
  abnormal return (AR) = actual daily return - model prediction using the REAL moves of the market, FX, yields, commodities.
The model for one event is fitted only on days BEFORE the event window (never on the event or after it).

Windows, in trading days relative to t0 = the first trading day on or after the release date:
  pre      : t0-10 .. t0-1        (the run-up)
  reaction : t0 and t0+1          (two days, because we do not know if the release was before the open or after the close)
  post     : t0+2 .. t0+11        (the drift after the reaction)
CAR = cumulative abnormal return (sum of daily AR; returns are log returns).

Questions answered (see summarize_stock / pool):
  1. How big is the reaction compared with a normal 2-day move?     move_ratio  (1 = nothing special)
  2. Does the stock keep going the same way after the reaction?      drift slope of CAR_post on CAR_reaction (>0 = continues, <0 = reverses)
  3. Does it move before the release in the direction of the reaction?  corr(CAR_pre, CAR_reaction)
  4. Does the stock's link to the market / key factor change after the release?   corr and R2 in the 30 days before vs after
  5. (US only, when Yahoo has it) Does the EPS surprise line up with the reaction?   Spearman correlation
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.factor_model import FactorModel

PRE, POST, CORR_WIN = 10, 10, 30


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 5 and np.std(a) > 0 and np.std(b) > 0 else float("nan")


def event_table(X, y, market_cols, other_cols, key_col, dates, cfg):
    """One row per usable event. `X`,`y` share one daily index (the stock's own trading days)."""
    train_days = int(cfg.get("earnings_train_days", 750))
    min_train = int(cfg.get("earnings_min_train_days", 250))
    alpha = cfg.get("ridge_alpha", 10.0)
    cal = y.index
    rows = []
    for d in pd.to_datetime(pd.Series(dates)).sort_values().unique():
        t0 = int(cal.searchsorted(pd.Timestamp(d)))
        if t0 >= len(cal) or (cal[t0] - pd.Timestamp(d)).days > 4:       # release outside the data (or a long gap)
            continue
        fit_end = t0 - PRE - 2                                            # leave a margin before the pre window
        fit_start = max(0, fit_end - train_days)
        if fit_end - fit_start < min_train or t0 + 1 >= len(cal):
            continue
        model = FactorModel(market_cols, other_cols, orthogonalize=cfg["orthogonalize"], method="ridge",
                            alpha=alpha).fit(X.iloc[fit_start:fit_end], y.iloc[fit_start:fit_end])
        lo, hi = t0 - max(PRE, CORR_WIN + 1), min(len(cal), t0 + max(POST, CORR_WIN) + 2)
        seg_X, seg_y = X.iloc[lo:hi], y.iloc[lo:hi]
        ar = pd.Series(seg_y.values - model.predict(seg_X), index=seg_y.index)
        sigma = float(np.std(y.iloc[fit_start:fit_end].values - model.predict(X.iloc[fit_start:fit_end]), ddof=1))

        def win(a, b):                                                    # trading days t0+a .. t0+b (inclusive)
            return ar.loc[cal[t0 + a]:cal[min(t0 + b, len(cal) - 1)]]

        pre, react = win(-PRE, -1), win(0, 1)
        post = win(2, 1 + POST) if t0 + 1 + POST < len(cal) else pd.Series(dtype=float)
        row = {"date": pd.Timestamp(d), "t0_date": cal[t0], "sigma_daily": sigma,
               "car_pre": pre.sum(), "car_react": react.sum(),
               "car_post": post.sum() if len(post) == POST else np.nan,
               "z_pre": pre.sum() / (sigma * np.sqrt(PRE)), "z_react": react.sum() / (sigma * np.sqrt(2)),
               "ret_react": float(y.iloc[t0:t0 + 2].sum())}
        row["z_post"] = row["car_post"] / (sigma * np.sqrt(POST)) if pd.notna(row["car_post"]) else np.nan
        # 30 trading days before / after the release (the reaction days themselves are left out)
        for tag, (a, b) in (("pre", (-CORR_WIN - 1, -1)), ("post", (2, CORR_WIN + 1))):
            if t0 + b >= len(cal) or t0 + a < 0:
                row.update({f"corr_mkt_{tag}": np.nan, f"corr_key_{tag}": np.nan, f"r2_{tag}": np.nan})
                continue
            sl = slice(t0 + a, t0 + b + 1)
            yy, xx = y.iloc[sl], X.iloc[sl]
            res = ar.reindex(yy.index)
            row[f"corr_mkt_{tag}"] = _corr(yy.values, xx[market_cols[0]].values)
            row[f"corr_key_{tag}"] = _corr(yy.values, xx[key_col].values) if key_col in xx else np.nan
            ss_tot = float(((yy - yy.mean()) ** 2).sum())
            row[f"r2_{tag}"] = 1 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _t_mean(s):
    s = pd.Series(s).dropna()
    if len(s) < 5:
        return float("nan"), float("nan")
    r = stats.ttest_1samp(s, 0.0)
    return float(r.statistic), float(r.pvalue)


def _slope(x, y):
    d = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(d) < 8:
        return float("nan"), float("nan"), len(d)
    res = sm.OLS(d.iloc[:, 1], sm.add_constant(d.iloc[:, 0])).fit(cov_type="HC1")
    return float(res.params.iloc[1]), float(res.pvalues.iloc[1]), len(d)


def summarize_stock(ev):
    """All headline numbers of one stock (one row)."""
    if ev.empty:
        return {"n_events": 0}
    out = {"n_events": len(ev), "first_event": str(ev["date"].min().date()), "last_event": str(ev["date"].max().date())}
    out["mean_car_react"] = float(ev["car_react"].mean())
    out["t_mean_car_react"], out["p_mean_car_react"] = _t_mean(ev["car_react"])
    out["rms_z_react"], out["rms_z_pre"] = float(np.sqrt((ev["z_react"] ** 2).mean())), float(np.sqrt((ev["z_pre"] ** 2).mean()))
    out["move_ratio"] = out["rms_z_react"] / out["rms_z_pre"] if out["rms_z_pre"] > 0 else np.nan
    out["drift_slope"], out["drift_p"], out["n_drift"] = _slope(ev["car_react"], ev["car_post"])
    out["pre_slope"], out["pre_p"], _ = _slope(ev["car_pre"], ev["car_react"])
    for name in ("corr_mkt", "corr_key", "r2"):
        diff = ev[f"{name}_post"] - ev[f"{name}_pre"]
        out[f"{name}_pre"], out[f"{name}_post"] = float(ev[f"{name}_pre"].mean()), float(ev[f"{name}_post"].mean())
        out[f"{name}_change_t"], out[f"{name}_change_p"] = _t_mean(diff)
    if "surprise_pct" in ev and ev["surprise_pct"].notna().sum() >= 8:
        d = ev[["surprise_pct", "car_react"]].dropna()
        r = stats.spearmanr(d["surprise_pct"], d["car_react"])
        out["surprise_n"], out["surprise_spearman"], out["surprise_p"] = len(d), float(r.statistic), float(r.pvalue)
    return out


def _clustered(df, ycol, xcol=None):
    """Pooled regression with standard errors clustered by calendar quarter (all firms report in the same weeks, so
    their events are NOT independent). Returns (coef, p, n) of the slope, or of the mean when xcol is None."""
    cols = [ycol] + ([xcol] if xcol else []) + ["quarter"]
    d = df[cols].dropna()
    if len(d) < 20 or d["quarter"].nunique() < 4:
        return float("nan"), float("nan"), len(d)
    X = sm.add_constant(d[xcol]) if xcol else np.ones((len(d), 1))
    res = sm.OLS(d[ycol].values, X).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["quarter"])[0]})
    i = 1 if xcol else 0
    return float(np.asarray(res.params)[i]), float(np.asarray(res.pvalues)[i]), len(d)


def pool(events):
    """Pooled results per group (all, country, sector). events has columns stock, country, sector + event columns."""
    ev = events.copy()
    ev["quarter"] = ev["date"].dt.to_period("Q").astype(str)
    groups = {"all": ev}
    for c in ev["country"].unique():
        groups[f"country={c}"] = ev[ev["country"] == c]
    for s in ev["sector"].unique():
        groups[f"sector={s}"] = ev[ev["sector"] == s]
    rows = []
    for name, g in groups.items():
        m, mp, n = _clustered(g, "car_react")
        ds, dp, nd = _clustered(g, "car_post", "car_react")
        ps, pp, _ = _clustered(g, "car_react", "car_pre")
        g = g.assign(d_mkt=g["corr_mkt_post"] - g["corr_mkt_pre"], d_r2=g["r2_post"] - g["r2_pre"])
        cm, cmp_, _ = _clustered(g, "d_mkt")
        dr, drp, _ = _clustered(g, "d_r2")
        row = {"group": name, "n_events": n, "mean_car_react": m, "p_mean_car_react": mp,
               "move_ratio": float(np.sqrt((g["z_react"] ** 2).mean() / (g["z_pre"] ** 2).mean())),
               "drift_slope": ds, "drift_p": dp, "n_drift": nd, "pre_slope": ps, "pre_p": pp,
               "change_corr_market": cm, "change_corr_market_p": cmp_, "change_r2": dr, "change_r2_p": drp}
        if "surprise_pct" in g and g["surprise_pct"].notna().sum() >= 20:
            sl, sp, ns = _clustered(g.assign(sp=g["surprise_pct"].clip(-100, 100)), "car_react", "sp")
            row.update({"surprise_slope": sl, "surprise_p": sp, "n_surprise": ns})
        rows.append(row)
    return pd.DataFrame(rows)
