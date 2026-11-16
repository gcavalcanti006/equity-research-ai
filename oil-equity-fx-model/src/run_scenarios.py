"""Run the scenarios and the historical test for every stock.

Usage:  python -m src.run_scenarios [--only petr4] [config_multifactor.yaml]
Run `python -m src.run_factors` first: it saves the data snapshots and the evidence of which stocks the
factor model really explains (used to mark each scenario result as reliable or not).
"""
import argparse
from pathlib import Path

import pandas as pd

from src import factors, scenarios, sources
from src.run_factors import date_range, load_config


def reliability(out_dir, p_max=0.01):
    """stock -> True if the full model beat the market-only model out of sample (p < p_max).
    With ~30 stocks tested at once, a few would pass at 0.05 by luck alone, so the default is stricter."""
    path = Path(out_dir) / "summary.csv"
    if not path.exists():
        return {}
    s = pd.read_csv(path)
    if "p_factors_beats_capm" not in s:
        return {}
    ok = (s["p_factors_beats_capm"] < p_max) & (s["oos_r2_factors"] > s["oos_r2_capm"])
    return dict(zip(s["stock"], ok))


def main(config_path="config_multifactor.yaml", only=None):
    cfg = load_config(config_path)
    start, end = date_range(cfg)
    out = Path(cfg["scenario_outputs"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading series {start} -> {end}")
    levels, problems = sources.load_all(cfg, start, end)
    for p in problems:
        print("  problem:", p)
    reliable = reliability(cfg["paths"]["outputs"], cfg.get("reliable_p_max", 0.01))
    if not reliable:
        print("\nNote: outputs/multifactor/summary.csv not found, so 'reliable' cannot be shown. "
              "Run `python -m src.run_factors` first.")

    scen_rows, ep_rows = [], []
    for name, info in cfg["stocks"].items():
        if only and name != only:
            continue
        d = factors.make_design(levels, name, cfg)
        if "error" in d or len(d["y"]) < cfg["min_train_days"] + cfg["step_days"]:
            print(f"  {name}: skipped ({d.get('error', 'too little data')})")
            continue
        X, y, mk, ot = d["X"], d["y"], d["market_cols"], d["other_cols"]
        rate_cols = factors.rate_names(cfg, X.columns)
        for r in scenarios.run_scenarios(X, y, mk, ot, cfg, cfg["scenarios"], rate_cols, d["substituted"]):
            scen_rows.append({"stock": name, "country": info["country"], "sector": info["sector"],
                              "reliable": reliable.get(name), **r})

        episodes = [(k, v["start"], v["end"]) for k, v in cfg.get("episodes", {}).items()]
        first = cfg["sectors"][info["sector"]][0]
        key = d["substituted"].get(first, first)
        key = key if key in X.columns else None
        if key:
            episodes += scenarios.auto_episodes(X, key, cfg["auto_episode_days"], cfg["auto_episode_k"],
                                                cfg["min_train_days"])
        for label, s0, s1 in episodes:
            r = scenarios.backtest_episode(X, y, mk, ot, cfg, s0, s1, cfg["min_train_days"])
            ep_rows.append({"stock": name, "country": info["country"], "sector": info["sector"],
                            "reliable": reliable.get(name), "episode": label, **r})

    scen = pd.DataFrame(scen_rows)
    eps = pd.DataFrame(ep_rows)
    scen.to_csv(out / "scenario_results.csv", index=False)
    eps.to_csv(out / "backtest_episodes.csv", index=False)

    if not scen.empty:
        show = scen[(scen["reading"] == "with_correlations") & (scen["reliable"] == True)]   # noqa: E712
        for sc, g in show.groupby("scenario"):
            print(f"\n== {sc} ({g['horizon_days'].iloc[0]} days; stocks where the model beat the market-only model) ==")
            print(g[["stock", "impact_pct", "low_80_pct", "high_80_pct", "beyond_history"]]
                  .round(1).to_string(index=False))
        print(f"\nAll stocks and both readings: {out / 'scenario_results.csv'}")

    if not eps.empty and "abs_err_model" in eps:
        rows = []
        for (sector, country, rel), g in eps.groupby(["sector", "country", "reliable"], dropna=False):
            rows.append({"sector": sector, "country": country, "reliable": rel, **scenarios.summarize_backtest(g)})
        summ = pd.DataFrame(rows)
        summ.to_csv(out / "backtest_summary.csv", index=False)
        print("\nHistorical test (model fitted only on days before each episode; real factor moves fed in):")
        print(summ.round(2).to_string(index=False))
        skipped = eps["skipped"].notna().sum() if "skipped" in eps else 0
        print(f"\nEpisodes skipped for lack of earlier history: {skipped}. Details: {out / 'backtest_episodes.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config_multifactor.yaml")
    ap.add_argument("--only")
    a = ap.parse_args()
    main(a.config, a.only)
