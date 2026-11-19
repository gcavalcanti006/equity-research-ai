"""Inflation and election variables for the long-horizon tests.

Inflation (no look-ahead):
  - US: CPI index (FRED CPIAUCNS, not seasonally adjusted, never revised) -> year-on-year % change.
  - Brazil: IPCA monthly % change (BCB SGS 433, never revised) -> compounded over 12 months.
  A month's number is only "known" `lag_days` after the END of that month (US CPI and IPCA come out about
  10-15 days after month end; 20 is deliberately a bit late, so the model never sees a number early).
  The result is a daily LEVEL series (the latest known year-on-year %). The pipeline treats it like a yield:
  its daily change is zero except on release days, where it is the change in year-on-year inflation (pp).
  Summed over a window that gives "how much did inflation change during this window".

Elections: a 0/1 flag, 1 when the window ends in a presidential-election year of the stock's country.
There are very few elections (five per country in 20 years), so this variable can almost never be
"proven" - the test reports that honestly.
"""
import numpy as np
import pandas as pd


def us_yoy(cpi_index):
    """Year-on-year % change of a monthly index (dates on the 1st of each month)."""
    s = cpi_index.dropna().sort_index().asfreq("MS")
    return 100.0 * (s / s.shift(12) - 1.0)


def br_yoy(ipca_monthly_pct):
    """12-month compounded IPCA (%), from monthly % changes (dates on the 1st of each month)."""
    s = ipca_monthly_pct.dropna().sort_index().asfreq("MS")
    growth = 1.0 + s / 100.0
    return 100.0 * (growth.rolling(12, min_periods=12).apply(np.prod, raw=True) - 1.0)


def known_from(monthly, lag_days):
    """Re-index a monthly series by the date it becomes public: end of its month + lag_days."""
    s = monthly.dropna().copy()
    s.index = s.index + pd.offsets.MonthEnd(0) + pd.Timedelta(days=lag_days)
    return s[~s.index.duplicated(keep="last")].sort_index()


def inflation_levels(us_cpi=None, br_ipca=None, lag_days=20):
    """Dict {'us_infl': series, 'br_infl': series}; each has one value per release (to be forward-filled)."""
    out = {}
    if us_cpi is not None and len(us_cpi.dropna()) > 13:
        out["us_infl"] = known_from(us_yoy(us_cpi), lag_days).rename("us_infl")
    if br_ipca is not None and len(br_ipca.dropna()) > 13:
        out["br_infl"] = known_from(br_yoy(br_ipca), lag_days).rename("br_infl")
    return out


def election_flag(index, years):
    """1.0 for dates inside a listed election year, else 0.0."""
    return pd.Series(np.isin(pd.DatetimeIndex(index).year, list(years)).astype(float), index=index)
