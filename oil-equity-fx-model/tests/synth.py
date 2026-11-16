"""Fake but realistic data for tests: a stock with KNOWN betas, so we can check the model finds them."""
import numpy as np
import pandas as pd


def make_cfg(tmp_path):
    return {
        "years": 5, "snapshot_dir": str(tmp_path / "snap"), "max_ffill_days": 2,
        "yahoo": {}, "fred": {}, "bcb": {}, "rate_series": [],
        "market": {"BR": "ibov", "US": "sp500"},
        "common_factors": {"BR": ["usdbrl", "sp500", "us10y", "us_slope", "vix"], "US": ["us10y"]},
        "sectors": {"oil": ["brent"], "agro": ["soy"]},
        "lagged_factors": ["sp500", "us10y"],
        "stocks": {"petr4": {"ticker": "PETR4.SA", "country": "BR", "sector": "oil"}},
        "scenario_horizon_days": 22, "auto_episode_days": 10, "auto_episode_k": 2,
        "scenario_outputs": str(tmp_path / "scen"), "factor_fallbacks": {},
        "scenarios": {
            "oil_down": {"shocks": {"brent": -0.20}},
            "brazil_stress": {"shocks": {"usdbrl": 0.10, "ibov": -0.10}},
            "us_rates_up": {"shocks": {"us10y": 1.0}},
            "soy_down": {"shocks": {"soy": -0.15}},
        },
        "min_train_days": 300, "step_days": 21, "hac_lags": 5,
        "orthogonalize": True, "ridge_alpha": 10.0, "rolling_window": 120,
        "paths": {"outputs": str(tmp_path / "out")},
    }


def make_levels(n=900, seed=0, brent_beta=0.30, noise=0.012, signal=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    ibov_r = rng.normal(0, 0.011, n)
    sp_r = 0.5 * ibov_r + rng.normal(0, 0.008, n)
    brent_r = 0.3 * ibov_r + rng.normal(0, 0.02, n)          # oil moves with the market a bit
    fx_r = -0.4 * ibov_r + rng.normal(0, 0.006, n)
    vix_r = -3 * sp_r + rng.normal(0, 0.03, n)
    stock_r = 0.9 * ibov_r + (brent_beta * brent_r if signal else 0) + rng.normal(0, noise, n)

    def price(r, p0):
        return pd.Series(p0 * np.exp(np.cumsum(r)), index=dates)

    us10y = pd.Series(4 + np.cumsum(rng.normal(0, 0.03, n)), index=dates)
    us2y = pd.Series(3.5 + np.cumsum(rng.normal(0, 0.03, n)), index=dates)
    levels = pd.DataFrame({"ibov": price(ibov_r, 100000), "sp500": price(sp_r, 4000),
                           "brent": price(brent_r, 80), "usdbrl": price(fx_r, 5.0),
                           "vix": price(vix_r, 18), "us10y": us10y, "us2y": us2y,
                           "petr4": price(stock_r, 30)})
    return levels
