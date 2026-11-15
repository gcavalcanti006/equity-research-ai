# oil-equity-fx-model

Measures how much of a Brazilian oil stock's daily price movement is explained by
oil (Brent) and the exchange rate (USD/BRL).

## What it does
1. Downloads daily prices (yfinance), cleans and aligns them, computes log returns.
2. Correlation matrix and a rolling 60-day correlation chart.
3. OLS regression: `equity return = a + b1 * Brent return + b2 * USD/BRL return`.
4. A random forest (or gradient boosting) trained on the first 80% of days and tested on
   the last 20%, compared with OLS and with a naive "predict the training mean" baseline.
5. Chart of predicted vs actual returns on the test period.

## How to run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.main          # needs internet (yfinance)
.venv/bin/python -m pytest            # tests, no internet needed
```
Outputs go to `outputs/`: `rolling_correlation.png`, `predicted_vs_actual.png`,
`correlation_matrix.csv`, `ols_results.csv`, `test_metrics.csv`.

## How it works (plain English)
- **Log return** = ln(today's price / yesterday's price). It is close to the % change and
  adds up over time.
- Brazil and the US have different holidays, and the three series close at different
  times. We keep each series' calendar date, forward-fill gaps of at most
  `max_ffill_days` days, and drop days still missing. Returns are computed *after*
  alignment.
- **OLS** finds the straight-line relationship. Standard errors are HAC (Newey-West), which
  stay valid when returns have autocorrelation or changing volatility.
- **Random forest** can capture non-linear effects. The split is chronological: the model
  is fitted on the past only and scored on the future, so there is no look-ahead.
- **Out-of-sample R²** compares the model's errors with always predicting the training
  mean. It is 0 if no better, and negative if worse.

## Settings (`config.yaml`)
Tickers, years or explicit start/end dates, rolling window, `oil_lag_days` (adds
yesterday's Brent return as a regressor), test fraction, model type and seed.

## How to extend
- Another stock: change `tickers.equity` (e.g. `PRIO3.SA`).
- Another factor (e.g. iron ore, S&P 500): add a ticker in the config and one column in
  `data.py`, then add it to `x_cols` in `main.py`.
- Another model: add a branch in `ml_model.make_model`.

## Results (run on 2026-11-14, data 2021-11-15 to 2026-11-13, PETR4.SA, 1,299 daily returns)
| | |
|---|---|
| Correlation with Brent | 0.36 |
| Correlation with USD/BRL | 0.05 |
| OLS Brent coefficient | 0.28 (t = 10.96) |
| OLS USD/BRL coefficient | 0.15 (t = 2.43) |
| OLS R² (in-sample) | 0.133 |
| Test period | 2025-11-13 to 2026-11-13 (260 days) |
| Test R²: OLS / Random forest / baseline | 0.249 / 0.226 / -0.0004 |

What this says:
- A 1% move in Brent is associated with about a 0.28% same-day move in PETR4, holding
  USD/BRL fixed. A weaker real (USD/BRL up) is associated with a higher PETR4, which is
  consistent with Petrobras selling oil in dollars.
- **R² is low (13%).** About 87% of daily variation is something else: company-specific
  news, Brazilian politics and the government's influence on pricing and strategy,
  dividend announcements, governance concerns and general market moves.
- The random forest did **not** beat plain OLS (0.226 vs 0.249). With only two inputs and
  a weak signal, the extra flexibility adds nothing here.
- Test R² (0.25) is higher than the full-sample R² (0.13). That does not mean the model
  got better: in the last year, PETR4 simply followed oil more closely (rolling Brent
  correlation mostly 0.4-0.7). The rolling chart shows the relationship is not stable;
  the USD/BRL correlation even changes sign.

## Limitations
- **No causality.** These are statistical associations; oil, FX and the stock can all react
  to a third factor (e.g. global risk appetite).
- **Non-stationarity and regime changes.** The relationship moves over time (see rolling
  chart), so coefficients are averages over different regimes.
- **Small sample for ML.** About 1,000 training days with noisy returns.
- **Timing mismatch.** Brent futures, PETR4 and USD/BRL close at different times of day,
  so some same-day co-movement lands on the next day. `oil_lag_days: 1` tests this.
- **Forward-filled gaps** create a few zero returns on holidays.
- **Prices are dividend/split-adjusted** (yfinance `auto_adjust`), so returns include
  dividends.
- **Single train/test split.** One test year is one draw; results could differ with other
  periods.
- Brent futures (`BZ=F`) are a continuous contract: roll dates can create small jumps.
