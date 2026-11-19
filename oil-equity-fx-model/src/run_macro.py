"""Test whether inflation and election years add anything to the factor model (1 and 3 months).

Usage:  python -m src.run_macro [--only petr4] [config_multifactor.yaml]
First run downloads US CPI (FRED) and Brazil IPCA (BCB) - run it on a computer with internet.
Output: outputs/macro/summary.csv (one row per stock and horizon), outputs/macro/tests.csv (every test, with Holm p).
"""
import argparse
from pathlib import Path

import pandas as pd

from src import factors, macro_test, sources
from src.run_factors import date_range, load_config


def main(config_path="config_multifactor.yaml", only=None):
    cfg = load_config(config_path)
    start, end = date_range(cfg)
    out = Path(cfg.get("macro_outputs", "outputs/macro"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading series {start} -> {end}")
    levels, problems = sources.load_all(cfg, start, end)
    for p in problems:
        print("  problem:", p)
    missing_cols = [c for c in ("us_infl", "br_infl") if c not in levels]
    if missing_cols:
        print(f"\nNo inflation series for {missing_cols}: only the election test will run for those.")
    p_max = cfg.get("reliable_p_max", 0.01)
    rows = []
    for name, info in cfg["stocks"].items():
        if only and name != only:
            continue
        country = info["country"]
        infl_cols = [c for c in cfg["macro_factors"].get(country, []) if c in levels]
        base = factors.make_design(levels, name, cfg)
        if "error" in base:
            print(f"  {name}: skipped ({base['error']})")
            continue
        ext_cfg = dict(cfg, common_factors={**cfg["common_factors"],
                                            country: list(cfg["common_factors"].get(country, [])) + infl_cols})
        d = factors.make_design(levels, name, ext_cfg)
        if "error" in d:
            print(f"  {name}: skipped ({d['error']})")
            continue
        base_other = [c for c in d["other_cols"] if c not in infl_cols]
        for h in cfg.get("long_horizons", [21, 63]):
            res = macro_test.run_stock(d["X"], d["y"], d["market_cols"], base_other, infl_cols, country, cfg, int(h))
            res.update({"stock": name, "country": country, "sector": info["sector"], "horizon_days": h})
            rows.append(res)
            if "error" in res:
                print(f"  {name} h={h}: {res['error']}")
            else:
                print(f"  {name:7s} h={h:3d}  explain R2 base {res['explain_r2_base']:+.3f}"
                      f"  +infl p={res.get('explain_p_infl', float('nan')):.2f}  +elec p={res['explain_p_elec']:.2f}"
                      f"   | predict +infl p={res.get('predict_p_infl', float('nan')):.2f}  +elec p={res['predict_p_elec']:.2f}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    tests = macro_test.tidy(rows, p_max)
    tests.to_csv(out / "tests.csv", index=False)
    if tests.empty:
        print("\nNo tests ran.")
        return
    print(f"\n{len(tests)} tests in total (stock x horizon x explain/predict x variant).")
    print(f"By pure luck about {0.025 * len(tests):.0f} would look helpful at p < 0.05 and {0.005 * len(tests):.0f} at p < 0.01 (half of the lucky ones have the wrong sign).")
    g = tests.groupby(["mode", "variant"]).agg(tests=("p", "size"), helps_p05=("helps_p05", "sum"),
                                              helps_p01=("helps_alone", "sum"),
                                              survive_holm=("helps_after_correction", "sum"),
                                              avg_r2_gain=("r2_variant", "mean"))
    g["avg_r2_gain"] = g["avg_r2_gain"] - tests.groupby(["mode", "variant"])["r2_base"].mean()   # average R2 change vs base
    print(g.to_string())
    print(f"\nSaved {out / 'summary.csv'} and {out / 'tests.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_multifactor.yaml")
    ap.add_argument("--only")
    a = ap.parse_args()
    main(a.config, a.only)
