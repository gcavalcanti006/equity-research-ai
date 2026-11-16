"""Turn raw levels (prices and yields) into the return/change columns the model uses.

Key idea: every stock is evaluated on ITS OWN trading calendar. Factors from other markets are
forward-filled onto that calendar (at most `max_ffill_days`) BEFORE returns are computed, so a
Brazilian holiday never creates a fake zero-return day.

Kinds of series:
- price: log return  ln(P_t / P_{t-1})
- rate:  change in percentage points (a yield going from 4.20 to 4.25 is +0.05)
"""
import numpy as np
import pandas as pd


def add_derived_levels(levels):
    """Curve slopes (10y minus 2y), when both ends are available."""
    levels = levels.copy()
    for prefix in ("us", "br"):
        long_, short = f"{prefix}10y", f"{prefix}2y"
        if long_ in levels and short in levels:
            levels[f"{prefix}_slope"] = levels[long_] - levels[short]
    return levels


def align_levels(levels, calendar, max_ffill_days):
    """Put all series on `calendar`, filling short gaps (other markets' holidays)."""
    union = levels.index.union(calendar)
    filled = levels.reindex(union).sort_index().ffill(limit=max_ffill_days)
    return filled.reindex(calendar)


def to_changes(aligned, rate_series):
    """Price columns -> log returns, rate columns -> differences."""
    out = {}
    for col in aligned.columns:
        s = aligned[col]
        if col in rate_series:
            out[col] = s.diff()
        else:
            out[col] = np.log(s / s.shift(1))
    return pd.DataFrame(out)


def rate_names(cfg, columns):
    """Which columns are yields (changed by difference, not by log return)."""
    base = set(cfg.get("fred", {})) | set(cfg.get("bcb", {})) | set(cfg.get("rate_series", []))
    base |= {"br2y", "br10y", "us_slope", "br_slope"}
    return {c for c in columns if c in base}


def make_design(levels, stock, cfg):
    """Build y (stock return) and X (factor columns) for one stock.

    Returns a dict: X, y, market_cols, other_cols, missing (factors the config asked for that
    are not available, e.g. a download failed)."""
    info = cfg["stocks"][stock]
    if stock not in levels or levels[stock].dropna().empty:
        return {"error": f"no price data for {stock} ({info['ticker']})"}
    calendar = levels[stock].dropna().index
    levels = add_derived_levels(levels)
    aligned = align_levels(levels, calendar, cfg["max_ffill_days"])
    changes = to_changes(aligned, rate_names(cfg, aligned.columns))

    if info["sector"] not in cfg["sectors"]:
        return {"error": f"sector '{info['sector']}' of {stock} is not listed under 'sectors' in the config"}
    if info["country"] not in cfg["market"]:
        return {"error": f"country '{info['country']}' of {stock} is not listed under 'market' in the config"}
    market = cfg["market"][info["country"]]
    wanted = [market] + list(cfg["common_factors"].get(info["country"], [])) \
        + list(cfg["sectors"][info["sector"]])
    wanted = list(dict.fromkeys(wanted))               # remove duplicates, keep order
    missing = [f for f in wanted if f not in changes or changes[f].dropna().empty]
    used = [f for f in wanted if f not in missing]
    if market in missing:
        return {"error": f"market factor '{market}' is not available"}

    X = changes[used].copy()
    lagged = [f for f in cfg.get("lagged_factors", []) if f in used and f != market]
    for f in lagged:                                   # yesterday's value: captures time-zone gaps
        X[f"{f}_lag1"] = X[f].shift(1)
    data = pd.concat([changes[stock].rename("y"), X], axis=1).dropna()
    other = [c for c in data.columns if c not in ("y", market)]
    return {"X": data.drop(columns="y"), "y": data["y"], "market_cols": [market],
            "other_cols": other, "missing": missing}
