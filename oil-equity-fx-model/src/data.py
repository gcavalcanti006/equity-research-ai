"""Download prices, align the three series on common dates and compute log returns."""
import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers, start, end):
    """Download daily closing prices for each ticker. Returns one column per series.

    Series: 'equity', 'oil', 'fx'. Raises an error if a download comes back empty
    (we never invent data).
    """
    series = {}
    for name, ticker in tickers.items():
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True,
                          progress=False)
        if raw is None or raw.empty:
            raise RuntimeError(f"No data returned for {ticker} ({name}). "
                               "Check the ticker, dates and internet connection.")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):      # newer yfinance returns a 1-column frame
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index)
        # Remove any time-zone so dates from different markets can be compared.
        if close.index.tz is not None:
            close.index = close.index.tz_localize(None)
        # Keep only the calendar date (drops intraday time stamps).
        close.index = close.index.normalize()
        series[name] = close[~close.index.duplicated(keep="last")]
    return pd.DataFrame(series)


def clean_prices(prices, max_ffill_days=2):
    """Align series on the union of dates, fill short gaps, drop the rest.

    Markets have different holidays (Brazil vs US). We forward-fill a missing price
    only for up to `max_ffill_days` days, then drop days where anything is missing.
    Returns the clean prices and a small report about what was removed.
    """
    prices = prices.sort_index()
    prices = prices[prices > 0]                       # prices must be positive
    n_before = len(prices)
    filled = prices.ffill(limit=max_ffill_days)
    clean = filled.dropna()
    report = {"rows_before": n_before, "rows_after": len(clean),
              "rows_dropped": n_before - len(clean)}
    return clean, report


def log_returns(prices):
    """Daily log return: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def add_oil_lag(returns, lag_days):
    """Optionally add yesterday's oil return as an extra column 'oil_lag1'."""
    returns = returns.copy()
    if lag_days and lag_days > 0:
        returns[f"oil_lag{lag_days}"] = returns["oil"].shift(lag_days)
        returns = returns.dropna()
    return returns
