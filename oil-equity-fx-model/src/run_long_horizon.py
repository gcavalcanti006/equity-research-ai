"""Run the 1- and 3-month horizon analysis for every stock.

Usage:  python -m src.run_long_horizon [--only petr4] [config_multifactor.yaml]
Needs the data snapshots (run `python -m src.run_factors` once first, or this will download them).
Output: outputs/long_horizon/summary.csv  and  <stock>_h<days>_coefficients.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src import factors, long_horizon, sources
from src.run_factors import date_range, load_config


def main(config_path="config_multifactor.yaml", only=None):
    cfg = load_config(config_path)
    start, end = date_range(cfg)
    out = Path(cfg.get("long_outputs", "outputs/long_horizon"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading series {start} -> {end}")
    levels, problems = sources.load_all(cfg, start, end)
    for p in problems:
        print("  problem:", p)
    p_max = cfg.get("reliable_p_max", 0.01)
    rows = []
    for name, info in cfg["stocks"].items():
        if only and name != only:
            continue
        d = factors.make_design(levels, name, cfg)
        if "error" in d:
            print(f"  {name}: skipped ({d['error']})")
            continue
        for h in cfg.get("long_horizons", [21, 63]):
            res, coefs = long_horizon.run_stock(d["X"], d["y"], d["market_cols"], d["other_cols"], cfg, int(h))
            if "error" in res:
                print(f"  {name} h={h}: {res['error']}")
                rows.append({"stock": name, "country": info["country"], "sector": info["sector"],
                             "horizon_days": h, "error": res["error"]})
                continue
            res.update({"stock": name, "country": info["country"], "sector": info["sector"]})
            res["verdict"] = long_horizon.verdict(res, p_max)
            rows.append(res)
            coefs.to_csv(out / f"{name}_h{h}_coefficients.csv")
            print(f"  {name:7s} h={h:3d}  explain R2 {res['explain_r2_full']:+.2f} (market only {res['explain_r2_capm']:+.2f})"
                  f"   predict R2 {res['predict_r2_full']:+.2f} (p={res['predict_p_full_vs_mean']:.2f})   {res['verdict']}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    ok = df[df.get("error").isna()] if "error" in df else df
    if len(ok):
        print("\nVerdicts by horizon:")
        print(ok.groupby(["horizon_days", "verdict"]).size().to_string())
    print(f"\nSaved {out / 'summary.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_multifactor.yaml")
    ap.add_argument("--only")
    a = ap.parse_args()
    main(a.config, a.only)
