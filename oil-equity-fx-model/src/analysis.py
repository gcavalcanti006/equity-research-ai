"""Correlation analysis."""
import pandas as pd


def correlation_matrix(returns):
    """Full-sample correlation between the return series."""
    return returns[["equity", "oil", "fx"]].corr()


def rolling_correlation(returns, window):
    """Rolling correlation of the equity with oil and with the exchange rate."""
    out = pd.DataFrame({
        "equity_vs_oil": returns["equity"].rolling(window).corr(returns["oil"]),
        "equity_vs_fx": returns["equity"].rolling(window).corr(returns["fx"]),
    })
    return out.dropna()
