"""Scenario engine and its historical test.

TWO different things live here:

1. SCENARIOS ("what if Brent falls 20%?")
   You choose shocks to some factors. The fitted factor model translates them into an expected
   move of the stock plus an 80% range. Two readings are always shown:
     - only_shocked      : only the factors you named move (everything else stays at 0)
     - with_correlations : the other factors move too, the way they usually do when the named ones move
                           (e.g. when Brent falls, the Ibovespa tends to fall a bit as well)
   Impact = prediction with the shock minus prediction with no shock (so the stock's average drift is excluded).

2. HISTORICAL TEST ("would this have worked in past episodes?")
   For a past window, the model is fitted ONLY on days before the window. We then feed it the factor moves
   that really happened inside the window and compare its answer with what the stock really did.
   It tests the translation factor -> stock. It does NOT test whether anyone could have forecast the factors.

Honest limits: the model is linear (a very large shock is an extrapolation), and the range uses the error size of turbulent days times `crisis_band_multiplier` (chosen so that past episodes
fall inside it ~80% of the time; picked on the same episodes, so expect real coverage a bit lower).
"""
import numpy as np
import pandas as pd

from src.factor_model import Z80, FactorModel


def base_columns(X):
    """Factor columns without the '_lag1' copies."""
    return [c for c in X.columns if not c.endswith("_lag1")]


def shock_to_change(factor, value, rate_cols):
    """You write -0.20 for '-20%' (prices) or +1.0 for '+100 basis points' (yields).
    Returned in the model's own unit: log return for prices, percentage points for yields."""
    return float(value) if factor in rate_cols else float(np.log1p(value))


def fit_for_scenarios(X, y, market_cols, other_cols, cfg):
    model = FactorModel(market_cols, other_cols, orthogonalize=cfg["orthogonalize"]).fit(X, y)
    sigma = float(np.std(y.values - model.predict(X), ddof=1))
    return model, sigma


def crisis_sigma(resid, market, quantile=0.8):
    """Daily error size to use for ranges. Errors are bigger on turbulent days (the 20% of days with the biggest
    market moves) than on average, so we take the larger of the two. Fixes ranges that were too narrow in crashes."""
    resid = pd.Series(np.asarray(resid), index=market.index)
    turbulent = market.abs() >= market.abs().quantile(quantile)
    return float(max(resid.std(ddof=1), resid[turbulent].std(ddof=1)))


def build_cumulative_vector(X, shocks, with_correlations):
    """Cumulative factor moves over the scenario. Returns (vector, applied_shocks) or (None, {})."""
    base = base_columns(X)
    applied = {f: v for f, v in shocks.items() if f in base}
    if not applied:
        return None, {}
    vec = pd.Series(0.0, index=base)
    for f, v in applied.items():
        vec[f] = v
    if with_correlations:
        given = list(applied)
        rest = [c for c in base if c not in applied]
        if rest:
            cov = X[base].cov()
            beta = cov.loc[rest, given].values @ np.linalg.pinv(cov.loc[given, given].values)
            vec[rest] = beta @ np.array([applied[f] for f in given])      # conditional expectation
    full = pd.Series(0.0, index=X.columns)
    for c in X.columns:
        full[c] = vec[c[:-5]] if c.endswith("_lag1") else vec[c]         # yesterday ~ same cumulative move
    return full, applied


def impact_of(model, sigma, X, vec, horizon):
    """Expected log-return impact and half-width of the 80% band over `horizon` days.
    The model is daily and linear, so we feed it the average daily move and multiply back."""
    zero = pd.DataFrame([np.zeros(len(X.columns))], columns=X.columns)
    avg = pd.DataFrame([vec.values / horizon], columns=X.columns)
    impact = horizon * (model.predict(avg)[0] - model.predict(zero)[0])
    return float(impact), float(Z80 * sigma * np.sqrt(horizon))


def beyond_history(X, applied, horizon):
    """Shocks bigger than any move of that factor over `horizon` days seen in the sample (= extrapolation)."""
    out = []
    for f, v in applied.items():
        seen = X[f].rolling(horizon).sum().dropna()
        if len(seen) and abs(v) > seen.abs().max():
            out.append(f)
    return out


def _pct(x):
    return float(np.expm1(x) * 100)


def run_scenarios(X, y, market_cols, other_cols, cfg, scenarios, rate_cols, substituted=None):
    """One row per scenario and reading. Scenarios whose shocked factors the stock does not use are skipped.
    `substituted` maps a factor to the proxy actually used (e.g. iron_ore -> iron_proxy): a shock named
    on the real factor is applied to the proxy, and the row says so."""
    model, sigma_avg = fit_for_scenarios(X, y, market_cols, other_cols, cfg)
    mult = float(cfg.get("crisis_band_multiplier", 1.25))
    sigma = mult * crisis_sigma(y.values - model.predict(X), X[market_cols[0]])
    substituted = substituted or {}
    rows = []
    for name, spec in scenarios.items():
        h = int(spec.get("horizon_days", cfg["scenario_horizon_days"]))
        asked = {substituted.get(f, f): shock_to_change(substituted.get(f, f), v, rate_cols)
                 for f, v in spec["shocks"].items()}
        value_of = {substituted.get(f, f): v for f, v in spec["shocks"].items()}
        note = "; ".join(f"{a} shock applied to proxy {b}" for a, b in substituted.items() if a in spec["shocks"])
        for reading, corr in (("only_shocked", False), ("with_correlations", True)):
            vec, applied = build_cumulative_vector(X, asked, corr)
            if vec is None:
                break
            impact, band = impact_of(model, sigma, X, vec, h)
            band_avg = Z80 * sigma_avg * np.sqrt(h)
            rows.append({"scenario": name, "reading": reading, "horizon_days": h,
                         "shocks_applied": ", ".join(f"{f}={value_of[f]:+g}" for f in applied),
                         "note": note,
                         "shocks_not_used_by_this_stock": ", ".join(f for f in asked if f not in applied),
                         "impact_pct": _pct(impact), "low_80_pct": _pct(impact - band),
                         "high_80_pct": _pct(impact + band),
                         "low_80_calm_pct": _pct(impact - band_avg), "high_80_calm_pct": _pct(impact + band_avg),
                         "beyond_history": ", ".join(beyond_history(X, applied, h))})
    return rows


# ---------------- historical test ----------------

def auto_episodes(X, key_factor, horizon, k, min_train):
    """The k worst and k best non-overlapping `horizon`-day moves of the stock's key factor
    (that still leave at least `min_train` days of history before them)."""
    if key_factor not in X:
        return []
    roll = X[key_factor].rolling(horizon).sum()
    pos = [i for i in range(len(roll)) if not np.isnan(roll.iloc[i]) and i - horizon + 1 >= min_train]
    episodes = []
    for label, reverse in (("worst", False), ("best", True)):
        chosen = []
        for i in sorted(pos, key=lambda i: roll.iloc[i], reverse=reverse):
            if all(abs(i - j) >= horizon for j in chosen):
                chosen.append(i)
            if len(chosen) == k:
                break
        for n, i in enumerate(sorted(chosen), 1):
            episodes.append((f"auto_{key_factor}_{label}{n}", X.index[i - horizon + 1], X.index[i]))
    return episodes


def backtest_episode(X, y, market_cols, other_cols, cfg, start, end, min_train):
    """Fit on days BEFORE `start`; feed the real factor moves of the window; compare with the real stock move."""
    i0 = int(X.index.searchsorted(pd.Timestamp(start)))
    i1 = int(X.index.searchsorted(pd.Timestamp(end), side="right")) - 1
    if i1 < i0 or i0 >= len(X):
        return {"skipped": "window outside the data"}
    if i0 < min_train:
        return {"skipped": f"only {i0} days of history before the window (need {min_train})"}
    h = i1 - i0 + 1
    Xtr, ytr = X.iloc[:i0], y.iloc[:i0]
    full = FactorModel(market_cols, other_cols, orthogonalize=cfg["orthogonalize"]).fit(Xtr, ytr)
    capm = FactorModel(market_cols, [], orthogonalize=False).fit(Xtr, ytr)
    sigma = float(np.std(ytr.values - full.predict(Xtr), ddof=1))
    sigma_c = float(cfg.get("crisis_band_multiplier", 1.25)) * crisis_sigma(ytr.values - full.predict(Xtr),
                                                                           Xtr[market_cols[0]])
    avg = pd.DataFrame([X.iloc[i0:i1 + 1].sum().values / h], columns=X.columns)
    pred_full, pred_capm = h * full.predict(avg)[0], h * capm.predict(avg)[0]
    actual = float(y.iloc[i0:i1 + 1].sum())
    band = Z80 * sigma_c * np.sqrt(h)
    band_calm = Z80 * sigma * np.sqrt(h)
    return {"start": X.index[i0].date(), "end": X.index[i1].date(), "days": h,
            "actual_pct": _pct(actual), "model_pct": _pct(pred_full), "capm_pct": _pct(pred_capm),
            "abs_err_model": abs(actual - pred_full), "abs_err_capm": abs(actual - pred_capm),
            "abs_err_zero": abs(actual),
            "inside_80": bool(abs(actual - pred_full) <= band),
            "inside_80_calm": bool(abs(actual - pred_full) <= band_calm),
            "direction_ok": bool(np.sign(pred_full) == np.sign(actual))}


def summarize_backtest(df):
    """Averages over episodes. 'model beats capm' = share of episodes where the full model's error is smaller."""
    ok = df[df["abs_err_model"].notna()]
    if ok.empty:
        return {}
    return {"episodes": len(ok),
            "mae_model_pp": 100 * ok["abs_err_model"].mean(),
            "mae_capm_pp": 100 * ok["abs_err_capm"].mean(),
            "mae_zero_pp": 100 * ok["abs_err_zero"].mean(),
            "share_model_beats_capm": float((ok["abs_err_model"] < ok["abs_err_capm"]).mean()),
            "direction_hit_rate": float(ok["direction_ok"].mean()),
            "coverage_80": float(ok["inside_80"].mean()),
            "coverage_80_calm": float(ok["inside_80_calm"].mean()) if "inside_80_calm" in ok else float("nan")}
