"""Probability cones: where could the price be in a few weeks or months, and how wide is the honest range?

This is NOT a forecast of direction. The factor model cannot predict direction (see the months-horizon study), so the
central path is neutral (zero expected log return). What a cone does well is show the SIZE of plausible moves, and it
is tested for that: `calibrate` checks, on past dates, how often the real outcome fell inside the cone.

Two ways to build the paths (both resample the stock's own past returns in blocks of a few days, which keeps
the clustering of calm and wild days):
  bootstrap : returns of the last `lookback_days`, as they were.
  fhs       : "filtered historical simulation": past returns are first divided by the volatility of their day
              (an EWMA), then rescaled by TODAY's volatility, which fades back to the long-run level over `halflife` days.
              The cone is wider after turbulent weeks and narrower after calm ones.
An optional scenario drift (e.g. the expected effect of "oil -20%" from the scenario engine) shifts the centre.
"""
import numpy as np
import pandas as pd

LAMBDA = 0.94                         # EWMA decay (RiskMetrics' daily value)
QUANTILES = (5, 10, 25, 50, 75, 90, 95)


def log_returns(prices):
    p = pd.Series(prices).dropna()
    p = p[p > 0]
    return np.log(p / p.shift(1)).dropna()


def ewma_variance(r, lam=LAMBDA):
    """Variance forecast for day t using only returns up to t-1 (index aligned to t)."""
    v = r.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return v.shift(1).fillna(r.pow(2).expanding().mean().shift(1)).bfill()


def _blocks(rng, n, horizon, n_paths, block):
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, max(n - block, 1), size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, nb * block)[:, :horizon]
    return np.minimum(idx, n - 1)


def simulate_returns(returns, horizon, n_paths=2000, block=5, method="fhs", halflife=30.0, seed=0):
    """Matrix (n_paths, horizon) of simulated daily log returns, zero mean."""
    r = pd.Series(returns).dropna()
    rng = np.random.default_rng(seed)
    idx = _blocks(rng, len(r), horizon, n_paths, block)
    if method == "bootstrap":
        x = (r - r.mean()).values
        return x[idx]
    if method != "fhs":
        raise ValueError("method must be 'bootstrap' or 'fhs'")
    var = ewma_variance(r)
    z = (r / np.sqrt(var)).values
    z = (z - z.mean()) / z.std()
    long_run = float(r.std())
    # today's volatility forecast = EWMA after the last observed return
    cur = float(np.sqrt(LAMBDA * var.iloc[-1] + (1 - LAMBDA) * r.iloc[-1] ** 2))
    h = np.arange(1, horizon + 1)
    sig = long_run + (cur - long_run) * np.exp(-h / halflife)
    return z[idx] * sig[None, :]


def scenario_drift(horizon, total_log_move, scenario_days):
    """Spread an expected total move evenly over `scenario_days`, then keep it (no further drift)."""
    d = np.zeros(horizon)
    k = min(horizon, max(int(scenario_days), 1))
    d[:k] = total_log_move / max(int(scenario_days), 1)
    return d


def price_paths(last_price, rets, drift=None):
    cum = np.cumsum(rets + (0 if drift is None else np.asarray(drift)[None, :]), axis=1)
    return last_price * np.exp(cum)


def cone(paths, quantiles=QUANTILES):
    """Table: one row per day ahead (1..H), one column per quantile of the simulated price."""
    q = np.percentile(paths, quantiles, axis=0)
    return pd.DataFrame(q.T, index=np.arange(1, paths.shape[1] + 1), columns=[f"p{x}" for x in quantiles])


def end_probabilities(paths, last_price, thresholds=(-0.10, 0.0, 0.10)):
    """Share of paths that END below -10%, below 0 and above +10% (from the starting price)."""
    end = paths[:, -1] / last_price - 1
    return {"below_-10%": float((end < -0.10).mean()), "below_0%": float((end < 0).mean()),
            "above_+10%": float((end > 0.10).mean())}


def calibrate(returns, horizon, method, lookback=1260, step=21, n_paths=1000, block=5, halflife=30.0, seed=1):
    """Walk-forward check. At each past date: build the cone from the preceding `lookback` days, see where the REAL
    cumulative return of the next `horizon` days fell. Returns one row per date."""
    r = pd.Series(returns).dropna()
    var_all = ewma_variance(r)
    rows = []
    for t in range(lookback, len(r) - horizon, step):
        hist = r.iloc[t - lookback:t]
        sims = simulate_returns(hist, horizon, n_paths, block, method, halflife, seed + t).sum(axis=1)
        real = float(r.iloc[t:t + horizon].sum())
        lo, hi = {}, {}
        for lvl in (50, 80, 90):
            a = (100 - lvl) / 2
            lo[lvl], hi[lvl] = np.percentile(sims, [a, 100 - a])
        rows.append({"date": r.index[t], "real": real, "pit": float((sims < real).mean()),
                     "vol": float(np.sqrt(var_all.iloc[t])),
                     **{f"in_{l}": bool(lo[l] <= real <= hi[l]) for l in (50, 80, 90)}})
    return pd.DataFrame(rows)


def coverage(cal, turbulent_share=0.2):
    """Overall and in the most turbulent `turbulent_share` of dates (when the cone is hardest to get right)."""
    if cal.empty:
        return {}
    out = {"n_dates": len(cal)}
    hot = cal["vol"] >= cal["vol"].quantile(1 - turbulent_share)
    for lvl in (50, 80, 90):
        out[f"coverage_{lvl}"] = float(cal[f"in_{lvl}"].mean())
        out[f"coverage_{lvl}_turbulent"] = float(cal.loc[hot, f"in_{lvl}"].mean())
    return out
