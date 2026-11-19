"""Earnings-release event study for every stock.

Usage:  python -m src.run_earnings [--refresh-dates] [--only xom] [config_multifactor.yaml]
Needs the data snapshots (run_factors) and internet for the release dates (Yahoo). Brazilian dates are often missing on Yahoo:
write them in data/earnings_manual.csv (columns stock,date) and run again.
Output: outputs/earnings/events.csv, summary.csv (one row per stock), pooled.csv (groups), dates_coverage.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from src import earnings_dates, earnings_study, factors, sources
from src.run_factors import date_range, load_config


def main(config_path="config_multifactor.yaml", only=None, refresh_dates=False):
    cfg = load_config(config_path)
    start, end = date_range(cfg)
    out = Path(cfg.get("earnings_outputs", "outputs/earnings"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading series {start} -> {end}")
    levels, problems = sources.load_all(cfg, start, end)
    for p in problems:
        print("  problem:", p)
    events, summaries, coverage = [], [], []
    print("\nEarnings dates and events:")
    for name, info in cfg["stocks"].items():
        if only and name != only:
            continue
        dates, note = earnings_dates.load_dates(name, info["ticker"], cfg, refresh=refresh_dates)
        d = factors.make_design(levels, name, cfg)
        if "error" in d:
            print(f"  {name:7s} skipped ({d['error']})")
            coverage.append({"stock": name, "dates": len(dates), "events_used": 0, "source": note})
            continue
        key = cfg["sectors"][info["sector"]][0]
        key = key if key in d["X"].columns else d["other_cols"][0]
        ev = earnings_study.event_table(d["X"], d["y"], d["market_cols"], d["other_cols"], key, dates["date"], cfg)
        if len(ev):
            ev = ev.merge(dates[["date", "eps_estimate", "eps_reported", "surprise_pct"]], on="date", how="left")
            ev.insert(0, "stock", name)
            ev.insert(1, "country", info["country"])
            ev.insert(2, "sector", info["sector"])
            events.append(ev)
        s = earnings_study.summarize_stock(ev)
        s.update({"stock": name, "country": info["country"], "sector": info["sector"]})
        summaries.append(s)
        coverage.append({"stock": name, "dates": len(dates), "events_used": len(ev), "source": note})
        print(f"  {name:7s} {len(dates):3d} dates, {len(ev):3d} events used   [{note}]")
    cov = pd.DataFrame(coverage)
    cov.to_csv(out / "dates_coverage.csv", index=False)
    if not events:
        print("\nNo events. Yahoo gave no dates: fill data/earnings_manual.csv (columns stock,date) and run again.")
        return
    ev_all = pd.concat(events, ignore_index=True)
    ev_all.to_csv(out / "events.csv", index=False)
    summ = pd.DataFrame(summaries)
    summ.to_csv(out / "summary.csv", index=False)
    pooled = earnings_study.pool(ev_all)
    pooled.to_csv(out / "pooled.csv", index=False)
    few = cov[cov["events_used"] < 8]["stock"].tolist()
    if few:
        print(f"\nFewer than 8 events (too few for conclusions): {', '.join(few)}")
    print("\nPooled results (standard errors clustered by quarter):")
    cols = ["group", "n_events", "mean_car_react", "p_mean_car_react", "move_ratio", "drift_slope", "drift_p",
            "pre_slope", "pre_p", "change_corr_market", "change_corr_market_p"]
    print(pooled[cols].round(3).to_string(index=False))
    print(f"\nSaved to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_multifactor.yaml")
    ap.add_argument("--only")
    ap.add_argument("--refresh-dates", action="store_true")
    a = ap.parse_args()
    main(a.config, a.only, a.refresh_dates)
