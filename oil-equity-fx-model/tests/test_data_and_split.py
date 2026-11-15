import numpy as np
import pandas as pd

from src import data, ml_model


def test_log_return_value():
    prices = pd.DataFrame({"a": [100.0, 110.0]})
    r = data.log_returns(prices)
    assert np.isclose(r["a"].iloc[0], np.log(1.1))


def test_short_gap_filled_long_gap_dropped():
    idx = pd.date_range("2024-01-01", periods=6)
    p = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6.0],
                      "b": [1, np.nan, 3, np.nan, np.nan, np.nan]}, index=idx)
    clean, _ = data.clean_prices(p, max_ffill_days=1)
    # b is filled on day 2 (gap of 1) and day 4 (gap of 1); days 5-6 stay missing -> dropped
    assert len(clean) == 4


def test_time_split_has_no_lookahead():
    idx = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame({"equity": range(100)}, index=idx)
    train, test = ml_model.time_split(df, 0.2)
    assert len(train) == 80 and len(test) == 20
    assert train.index.max() < test.index.min()
