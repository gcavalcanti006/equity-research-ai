"""Run the full oil / FX / equity analysis. Usage: python -m src.main [config.yaml]"""
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src import analysis, data, ml_model, plots, regression


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def date_range(cfg):
    end = cfg.get("end_date") or date.today().isoformat()
    start = cfg.get("start_date") or (date.fromisoformat(end)
                                      - timedelta(days=365 * cfg["years"])).isoformat()
    return start, end


def main(config_path="config.yaml"):
    cfg = load_config(config_path)
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)
    start, end = date_range(cfg)

    # 1. Data
    prices = data.download_prices(cfg["tickers"], start, end)
    prices, report = data.clean_prices(prices, cfg["max_ffill_days"])
    print(f"Data {start} to {end}: kept {report['rows_after']} days, "
          f"dropped {report['rows_dropped']} (missing in at least one series).")
    rets = data.add_oil_lag(data.log_returns(prices), cfg["oil_lag_days"])
    extra = [c for c in rets.columns if c.startswith("oil_lag")]
    x_cols = ["oil", "fx", *extra]
    print(f"Return observations: {len(rets)}")

    # 2. Correlation
    corr = analysis.correlation_matrix(rets)
    corr.to_csv(out / "correlation_matrix.csv")
    print("\nCorrelation matrix:\n", corr.round(3))
    roll = analysis.rolling_correlation(rets, cfg["rolling_window"])
    plots.plot_rolling_corr(roll, cfg["rolling_window"], out / "rolling_correlation.png")

    # 3. OLS
    res = regression.run_ols(rets, extra)
    table = regression.ols_table(res)
    table.to_csv(out / "ols_results.csv")
    print("\nOLS (HAC standard errors):\n", table.round(4))
    print(f"R-squared (in-sample): {res.rsquared:.4f}")

    # 4. ML with time split
    ev = ml_model.evaluate(rets, x_cols, cfg["model_type"],
                           cfg["test_fraction"], cfg["random_seed"])
    metrics = pd.DataFrame(ev["metrics"]).T
    metrics.to_csv(out / "test_metrics.csv")
    print(f"\nTest period: {ev['test'].index[0].date()} to {ev['test'].index[-1].date()} "
          f"({len(ev['test'])} days); train: {len(ev['train'])} days")
    print("Out-of-sample metrics:\n", metrics.round(5))
    print("Feature importances:", {k: round(v, 3) for k, v in ev["importances"].items()})

    # 5. Chart
    plots.plot_pred_vs_actual(ev["test"], ev["preds"], cfg["model_type"],
                              out / "predicted_vs_actual.png")
    print(f"\nCharts and tables saved in {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
