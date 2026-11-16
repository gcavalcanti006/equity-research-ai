"""Run the multi-factor model for every stock in config_multifactor.yaml.

Usage:  python -m src.run_factors [--refresh] [--only petr4] [config_multifactor.yaml]
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src import factor_model as fm
from src import factors, sources


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("yahoo", {})
    for name, info in cfg["stocks"].items():       # stocks are downloaded like any other price
        cfg["yahoo"][name] = info["ticker"]
    return cfg


def date_range(cfg):
    end = cfg.get("end_date") or date.today().isoformat()
    start = cfg.get("start_date") or (date.fromisoformat(end) - timedelta(days=365 * cfg["years"])).isoformat()
    return start, end


def run_stock(name, levels, cfg, out_dir):
    """Returns one summary row (dict) for the stock, or a row with 'error'."""
    design = factors.make_design(levels, name, cfg)
    if "error" in design:
        return {"stock": name, "error": design["error"]}
    X, y = design["X"], design["y"]
    market_cols, other_cols = design["market_cols"], design["other_cols"]
    if len(y) < cfg["min_train_days"] + cfg["step_days"]:
        return {"stock": name, "error": f"only {len(y)} usable days; need more than "
                                        f"{cfg['min_train_days'] + cfg['step_days']}"}

    stock_dir = Path(out_dir) / name
    stock_dir.mkdir(parents=True, exist_ok=True)

    table, r2, r2_adj, fitted = fm.full_sample_report(X, y, market_cols, other_cols,
                                                      cfg["orthogonalize"], cfg["hac_lags"])
    table.to_csv(stock_dir / "ols_full_sample.csv")
    fm.vif_table(X).to_csv(stock_dir / "vif.csv")
    fm.rolling_coefficients(X, y, fitted, cfg["rolling_window"]).to_csv(stock_dir / "rolling_coefficients.csv")
    X.assign(y=y).corr().to_csv(stock_dir / "correlations.csv")

    sector_cols = [c for c in cfg["sectors"][cfg["stocks"][name]["sector"]] if c in X.columns]
    fx_cols = ["usdbrl"] if "usdbrl" in X.columns else []
    models = fm.default_models(market_cols, other_cols, cfg, sector_cols, fx_cols)
    preds = fm.walk_forward(X, y, models, cfg["min_train_days"], cfg["step_days"])
    preds.to_csv(stock_dir / "walk_forward_predictions.csv", index=False)
    ev = fm.evaluate(preds, cfg["hac_lags"])
    ev.to_csv(stock_dir / "walk_forward_metrics.csv")

    row = {"stock": name, "days": len(y), "test_days": int(ev.loc["mean", "n_test"]),
           "factors": ",".join(market_cols + [c for c in other_cols if not c.endswith("_lag1")]),
           "missing_factors": ",".join(design["missing"]),
           "substituted": ",".join(f"{a}->{b}" for a, b in design["substituted"].items()),
           "r2_in_sample_adj": r2_adj}
    for model in ("capm", "sector_fx", "factors_nolag", "factors", "factors_ridge"):
        row[f"oos_r2_{model}"] = ev.loc[model, "r2_oos_vs_mean"] if model in ev.index else float("nan")
    # "p" = chance of seeing this gain by luck. Gain is in squared-return units (tiny numbers).
    for model in ("capm", "sector_fx", "factors_nolag"):
        if model in ev.index:
            row[f"p_factors_beats_{model}"] = ev.loc[model, "dm_p_factors_over_it"]
            row[f"gain_factors_over_{model}"] = ev.loc[model, "dm_gain_factors_over_it"]
    row["dm_p_factors_vs_capm"] = ev.loc["factors", "dm_p_vs_capm"]
    row["dm_gain_factors_vs_capm"] = ev.loc["factors", "dm_gain_vs_capm"]
    row["coverage_80_factors"] = ev.loc["factors", "coverage_80"]
    row["hit_rate_factors"] = ev.loc["factors", "hit_rate"]
    return row


def main(config_path="config_multifactor.yaml", refresh=False, only=None):
    cfg = load_config(config_path)
    start, end = date_range(cfg)
    out_dir = Path(cfg["paths"]["outputs"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading series {start} -> {end}")
    levels, problems = sources.load_all(cfg, start, end, refresh=refresh)
    if problems:
        print("\nProblems (these series are skipped):")
        for p in problems:
            print("  -", p)

    rows = []
    for name in cfg["stocks"]:
        if only and name != only:
            continue
        print(f"\n== {name} ==")
        row = run_stock(name, levels, cfg, out_dir)
        if "error" in row:
            print("  skipped:", row["error"])
        else:
            print(f"  days={row['days']} test_days={row['test_days']} "
                  f"OOS R2 capm={row['oos_r2_capm']:.3f} factors={row['oos_r2_factors']:.3f} "
                  f"(DM p={row['dm_p_factors_vs_capm']:.3f})")
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print("\nSummary saved to", out_dir / "summary.csv")
    if "oos_r2_capm" in summary:
        show = ["stock", "oos_r2_capm", "oos_r2_sector_fx", "oos_r2_factors_nolag", "oos_r2_factors",
                "p_factors_beats_capm", "p_factors_beats_sector_fx", "p_factors_beats_factors_nolag",
                "coverage_80_factors"]
        print("\nOut-of-sample R2 (higher is better) and the p-value that the full model beats each simpler one:")
        print(summary[[c for c in show if c in summary]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_multifactor.yaml")
    ap.add_argument("--refresh", action="store_true", help="download everything again")
    ap.add_argument("--only", help="run a single stock key, e.g. petr4")
    a = ap.parse_args()
    main(a.config, a.refresh, a.only)
