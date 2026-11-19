"""Calibration of the probability cones for every stock (no downloads, no API: uses the saved price snapshots).

Usage:  python -m src.run_cones [--only petr4] [config_multifactor.yaml]
For each stock, horizon and method, it replays the past: at every date (every `cones_step_days`) the cone is built from the
preceding `cones_lookback_days` and compared with what really happened. Output: outputs/cones/calibration.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src import cones
from src.run_factors import load_config


def load_prices(cfg, stock):
    path = Path(cfg["snapshot_dir"]) / f"{stock}.csv"
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.dropna()


def main(config_path="config_multifactor.yaml", only=None):
    cfg = load_config(config_path)
    out = Path(cfg.get("cones_outputs", "outputs/cones"))
    out.mkdir(parents=True, exist_ok=True)
    lookback = int(cfg.get("cones_lookback_days", 1260))
    step = int(cfg.get("cones_step_days", 21))
    rows = []
    for name in cfg["stocks"]:
        if only and name != only:
            continue
        try:
            r = cones.log_returns(load_prices(cfg, name))
        except Exception as exc:
            print(f"  {name}: no prices ({type(exc).__name__})")
            continue
        for h in cfg.get("cones_horizons", [21, 63]):
            for method in ("bootstrap", "fhs"):
                cal = cones.calibrate(r, int(h), method, lookback, step, int(cfg.get("cones_calibration_paths", 1000)),
                                      int(cfg.get("cones_block_days", 5)), float(cfg.get("cones_halflife_days", 30)))
                cov = cones.coverage(cal)
                if not cov:
                    print(f"  {name} h={h}: too little history")
                    continue
                rows.append({"stock": name, "horizon_days": h, "method": method, **cov})
        last = [x for x in rows if x["stock"] == name and x["horizon_days"] == cfg.get("cones_horizons", [21, 63])[0]]
        if last:
            a = {x["method"]: x for x in last}
            print(f"  {name:7s} 21d 80%-cone hit rate: bootstrap {a['bootstrap']['coverage_80']:.0%} "
                  f"(turbulent {a['bootstrap']['coverage_80_turbulent']:.0%})   fhs {a['fhs']['coverage_80']:.0%} "
                  f"(turbulent {a['fhs']['coverage_80_turbulent']:.0%})")
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration.csv", index=False)
    if len(df):
        print("\nAverage hit rate over all stocks (a perfect 80% cone would show 80% everywhere):")
        print(df.groupby(["horizon_days", "method"])[["coverage_50", "coverage_80", "coverage_90",
                                                       "coverage_80_turbulent"]].mean().round(3).to_string())
    print(f"\nSaved {out / 'calibration.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_multifactor.yaml")
    ap.add_argument("--only")
    a = ap.parse_args()
    main(a.config, a.only)
