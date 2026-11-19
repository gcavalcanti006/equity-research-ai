"""Do inflation or election years add anything to the factor model at 1- and 3-month horizons?

For each stock and horizon we compare, out of sample (walk-forward, same rules as long_horizon.py):
    base   : market + the usual factors (the model already in the project)
    +infl  : base + change in year-on-year inflation over the window
    +elec  : base + a flag "this window ends in a presidential-election year"
    +both
Every variant is tested against base with a Diebold-Mariano test. 'gain' > 0 and a small p means the extra
variable helped; the runner then also applies a Holm correction because we run MANY tests and some will
look good by pure luck.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import long_horizon as lh
from src.factor_model import FactorModel, MeanModel, dm_test
from src.macro import election_flag

VARIANTS = ("infl", "elec", "both")


def _factories(market_cols, base_other, infl_cols, cfg, has_infl):
    mk = [f"f_{c}" for c in market_cols]
    base = [f"f_{c}" for c in base_other]
    inf = [f"f_{c}" for c in infl_cols]
    alpha, orth = cfg.get("long_ridge_alpha", 30.0), cfg["orthogonalize"]

    def full(extra):
        return lambda: FactorModel(mk, base + extra, orthogonalize=orth, method="ridge", alpha=alpha)

    fac = {"mean": lambda: MeanModel(), "base": full([]), "elec": full(["f_election"])}
    if has_infl:
        fac["infl"] = full(inf)
        fac["both"] = full(inf + ["f_election"])
    return fac


def _score(preds, hac):
    piv = {m: g.set_index("pos") for m, g in preds.groupby("model")}
    sse_mean = float(((piv["mean"]["actual"] - piv["mean"]["pred"]) ** 2).sum())
    out = {"n_test": len(piv["base"])}
    err_base = piv["base"]["actual"] - piv["base"]["pred"]
    out["r2_base"] = 1 - float((err_base ** 2).sum()) / sse_mean
    for v in VARIANTS:
        if v not in piv:
            continue
        err = piv[v]["actual"] - piv[v]["pred"]
        gain, p = dm_test(err, err_base, hac)            # positive gain = variant better than base
        out[f"r2_{v}"] = 1 - float((err ** 2).sum()) / sse_mean
        out[f"gain_{v}"], out[f"p_{v}"] = gain, p
    return out


def _extra_coefs(pairs, market_cols, base_other, infl_cols, cfg, hac):
    """In-sample coefficients of the extra variables (to read the sign and size, not to judge accuracy)."""
    feats = [f"f_{c}" for c in market_cols + base_other + infl_cols] + ["f_election"]
    fm = FactorModel([f"f_{c}" for c in market_cols], [f for f in feats if f[2:] not in market_cols],
                     orthogonalize=cfg["orthogonalize"]).fit(pairs[feats], pairs["y_now"])
    Z = fm._transform(pairs[feats])
    res = sm.OLS(pairs["y_now"], sm.add_constant(Z)).fit(cov_type="HAC", cov_kwds={"maxlags": hac})
    keep = [f"f_{c}" for c in infl_cols] + ["f_election"]
    return {f"{k[2:]}_{s}": float(v[k]) for k in keep for s, v in (("coef", res.params), ("t", res.tvalues))}


def run_stock(X, y, market_cols, base_other, infl_cols, country, cfg, horizon):
    """One stock, one horizon. X must already contain the inflation columns (if any).
    Returns a dict with one set of numbers for 'explain' and one for 'predict'."""
    step = int(cfg.get("long_step_days", 5))
    pairs = lh.build_pairs(X, y, horizon, step)
    min_train = int(cfg.get("long_min_train_days", 1260))
    if len(pairs) < 60 or pairs["pos"].max() < min_train + 2 * horizon:
        return {"error": f"too little history for horizon {horizon} (have {len(pairs)} windows)"}
    pairs = pairs.copy()
    pairs["f_election"] = election_flag(pairs.index, cfg["elections"][country]).values
    hac = max(2, horizon // step + 1)
    has_infl = bool(infl_cols)
    fac = _factories(market_cols, base_other, infl_cols, cfg, has_infl)
    out = {"horizon_days": horizon, "n_windows": len(pairs),
           "election_share": float(pairs["f_election"].mean())}
    for mode in ("explain", "predict"):
        preds = lh.walk_forward_long(pairs, market_cols, base_other, cfg, horizon, mode, factories=fac)
        out.update({f"{mode}_{k}": v for k, v in _score(preds, hac).items()})
    out.update(_extra_coefs(pairs, market_cols, base_other, infl_cols if has_infl else [], cfg, hac))
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (keeps the chance of ANY false alarm below the chosen level)."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def tidy(rows, p_max):
    """Long table: one row per (stock, horizon, mode, variant) with Holm-adjusted p and a plain verdict."""
    recs = []
    for r in rows:
        if "error" in r:
            continue
        for mode in ("explain", "predict"):
            for v in VARIANTS:
                if f"{mode}_p_{v}" not in r:
                    continue
                recs.append({"stock": r["stock"], "country": r["country"], "sector": r["sector"],
                             "horizon_days": r["horizon_days"], "mode": mode, "variant": v,
                             "r2_base": r[f"{mode}_r2_base"], "r2_variant": r[f"{mode}_r2_{v}"],
                             "gain": r[f"{mode}_gain_{v}"], "p": r[f"{mode}_p_{v}"]})
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["p_holm"] = holm(df["p"].values)
    df["helps_p05"] = (df["gain"] > 0) & (df["p"] < 0.05)       # small p with a NEGATIVE gain means it made things worse
    df["helps_alone"] = (df["gain"] > 0) & (df["p"] < p_max)
    df["helps_after_correction"] = (df["gain"] > 0) & (df["p_holm"] < 0.05)
    return df
