"""Longer horizons (1 and 3 months): what do the factors EXPLAIN, and what do they PREDICT?

Two different questions, kept apart on purpose:

  EXPLAIN  (contemporaneous): the stock's return over a window vs the factors' moves over the SAME window.
           "When Brent rose 15% over the quarter, how much did the stock usually rise?"  It is a relationship,
           not a forecast: you would have to know the factor moves in advance.
  PREDICT  (forward):  the factors' moves over the LAST window vs the stock's return over the NEXT window.
           This is the only one a trader could use. Expect it to be much weaker, usually close to nothing.

Both are scored walk-forward (the model only sees the past), against a running-mean benchmark and a market-only
benchmark. Windows overlap (we step by a few days), so every p-value uses HAC standard errors that allow for it.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.factor_model import FactorModel, MeanModel, dm_test


def build_pairs(X, y, horizon, step):
    """One row every `step` trading days. Columns: the factors' trailing `horizon`-day moves (prefix 'f_'),
    the stock's trailing return ('y_now') and the stock's NEXT `horizon`-day return ('y_next')."""
    trailing_x = X.rolling(horizon).sum()
    trailing_y = y.rolling(horizon).sum()
    forward_y = trailing_y.shift(-horizon)
    out = trailing_x.add_prefix("f_")
    out["y_now"], out["y_next"] = trailing_y, forward_y
    out["pos"] = np.arange(len(out))
    out = out.iloc[horizon - 1::step]
    return out.dropna(subset=list(trailing_x.columns.map(lambda c: f"f_{c}")) + ["y_now"])


def _factories(market_cols, other_cols, cfg):
    mk = [f"f_{c}" for c in market_cols]
    ot = [f"f_{c}" for c in other_cols]
    return {
        "mean": lambda: MeanModel(),
        "capm": lambda: FactorModel(mk, [], orthogonalize=False),
        "full": lambda: FactorModel(mk, ot, orthogonalize=cfg["orthogonalize"], method="ridge",
                                    alpha=cfg.get("long_ridge_alpha", 30.0)),
    }


def walk_forward_long(pairs, market_cols, other_cols, cfg, horizon, mode, factories=None):
    """mode 'explain' or 'predict'. Returns a table: pos, model, pred, actual, one row per test point.
    `factories` (optional) replaces the default models, e.g. to test extra variables (see macro_test.py)."""
    step_pos = int(pairs["pos"].iloc[1] - pairs["pos"].iloc[0]) if len(pairs) > 1 else 1
    min_train = int(cfg.get("long_min_train_days", 1260))
    refit = int(cfg.get("long_refit_every", 20))                    # in grid points
    feats = [c for c in pairs.columns if c.startswith("f_")]
    target = "y_now" if mode == "explain" else "y_next"
    usable = pairs.dropna(subset=[target]) if mode == "predict" else pairs
    factories = factories or _factories(market_cols, other_cols, cfg)
    rows = []
    idx = np.arange(len(usable))
    start = int(np.searchsorted(usable["pos"].values, min_train))
    for b in range(start, len(usable), refit):
        q0 = usable["pos"].iloc[b]
        # Only windows whose target was fully known (and, for 'explain', did not overlap the test window).
        known_until = q0 - horizon        # explain: no overlap with the test window; predict: target already known
        train = usable[usable["pos"] <= known_until]
        test = usable.iloc[b:b + refit]
        if len(train) < 30 or test.empty:
            continue
        for name, make in factories.items():
            m = make().fit(train[feats], train[target])
            rows.append(pd.DataFrame({"pos": test["pos"].values, "model": name,
                                      "pred": m.predict(test[feats]), "actual": test[target].values,
                                      "date": test.index}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["pos", "model", "pred", "actual", "date"])


def score(preds, hac_lags):
    """OOS R2 vs the running mean, hit rate, and DM p-values (full vs mean, full vs market-only)."""
    if preds.empty:
        return {}
    piv = {m: g.set_index("pos") for m, g in preds.groupby("model")}
    actual = piv["mean"]["actual"]
    sse_mean = float(((actual - piv["mean"]["pred"]) ** 2).sum())
    r2 = {m: 1 - float(((g["actual"] - g["pred"]) ** 2).sum()) / sse_mean for m, g in piv.items()}
    full, capm, mean = piv["full"], piv["capm"], piv["mean"]
    e_full, e_capm, e_mean = (full["actual"] - full["pred"], capm["actual"] - capm["pred"],
                              mean["actual"] - mean["pred"])
    gain_m, p_m = dm_test(e_full, e_mean, hac_lags)
    gain_c, p_c = dm_test(e_full, e_capm, hac_lags)
    nz = (full["pred"] != 0) & (full["actual"] != 0)
    return {"n_test": len(full), "r2_capm": r2["capm"], "r2_full": r2["full"],
            "p_full_vs_mean": p_m, "p_full_vs_capm": p_c, "gain_full_vs_capm": gain_c,
            "hit_rate_full": float((np.sign(full["pred"][nz]) == np.sign(full["actual"][nz])).mean())}


def coefficient_table(pairs, market_cols, other_cols, cfg, horizon, hac_lags):
    """Full-sample contemporaneous coefficients over `horizon` days (to READ, not to judge accuracy)."""
    feats = [f"f_{c}" for c in market_cols + other_cols]
    fm = FactorModel([f"f_{c}" for c in market_cols], [f"f_{c}" for c in other_cols],
                     orthogonalize=cfg["orthogonalize"]).fit(pairs[feats], pairs["y_now"])
    Z = fm._transform(pairs[feats])
    res = sm.OLS(pairs["y_now"], sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    t = pd.DataFrame({"coef": res.params, "std_err": res.bse, "t_stat": res.tvalues, "p_value": res.pvalues})
    t.index = [i.replace("f_", "") for i in t.index]
    return t, float(res.rsquared_adj)


def run_stock(X, y, market_cols, other_cols, cfg, horizon):
    """All numbers for one stock and one horizon (in trading days). Returns (summary dict, coefficient table)."""
    step = int(cfg.get("long_step_days", 5))
    pairs = build_pairs(X, y, horizon, step)
    min_train = int(cfg.get("long_min_train_days", 1260))
    if len(pairs) < 60 or pairs["pos"].max() < min_train + 2 * horizon:
        return {"error": f"too little history for horizon {horizon} (have {len(pairs)} windows)"}, None
    hac = max(2, horizon // step + 1)            # windows overlap by (horizon/step) grid steps
    out = {"horizon_days": horizon, "n_windows": len(pairs)}
    for mode in ("explain", "predict"):
        sc = score(walk_forward_long(pairs, market_cols, other_cols, cfg, horizon, mode), hac)
        out.update({f"{mode}_{k}": v for k, v in sc.items()})
    coefs, adj = coefficient_table(pairs, market_cols, other_cols, cfg, horizon, hac)
    out["r2_in_sample_adj"] = adj
    return out, coefs


def verdict(row, p_max):
    """Plain-language label from the numbers."""
    def good(prefix):
        p, r, c = row.get(f"{prefix}_p_full_vs_capm"), row.get(f"{prefix}_r2_full"), row.get(f"{prefix}_r2_capm")
        return bool(pd.notna(p) and p < p_max and r > c)
    explains, predicts = good("explain"), good("predict")
    pr2 = row.get("predict_r2_full")
    predicts_alone = bool(pd.notna(row.get("predict_p_full_vs_mean")) and row["predict_p_full_vs_mean"] < p_max and pr2 > 0)
    if predicts_alone:
        return "explains and PREDICTS (check carefully)" if explains else "predicts (check carefully)"
    return "explains, does not predict" if explains else "neither shown"
