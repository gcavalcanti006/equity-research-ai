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

## Multi-factor model (phase 1)
`config_multifactor.yaml` + `src/run_factors.py` extend the project from "oil + FX" to a general
multi-factor model for several stocks (Brazil and US; oil, agriculture, mining).

```bash
.venv/bin/python -m src.run_factors            # uses saved snapshots in data/snapshots/
.venv/bin/python -m src.run_factors --refresh  # downloads everything again
.venv/bin/python -m src.run_factors --only petr4
```
- **Data** (`src/sources.py`): Yahoo (stocks, indices, commodities, FX, VIX), FRED (US 2y/10y yields,
  no key needed), Banco Central (Selic). Every series is saved in `data/snapshots/`, so a run is
  reproducible. A series that fails to download is skipped and reported.
- **Factors** (`src/factors.py`): each stock is evaluated on its own trading calendar; prices become
  log returns and yields become changes in percentage points; US factors also get a 1-day lag
  (US markets close after B3).
- **Model** (`src/factor_model.py`): stock return = market index + FX + yields + curve slope + VIX +
  sector commodities. Optionally each factor is cleaned of its market part first, using training data only.
- **Evaluation**: expanding-window walk-forward. Every model is compared with (1) the running mean and
  (2) market-index-only (CAPM). Reported: out-of-sample R², Diebold-Mariano p-value of "factors vs CAPM",
  hit rate and the coverage of the 80% prediction interval (should be close to 80%).
- **Models compared in the same test**: `mean`, `capm` (market index only), `sector_fx` (the original
  project's idea: sector commodity + USD/BRL, no market index), `factors` (full), `factors_nolag`
  (full without the "yesterday" columns) and `factors_ridge`. The summary prints the p-value that the
  full model really beats each simpler one.
- **This measures explanation, not forecasting**: the factors are same-day returns. Out-of-sample R²
  says the relationship holds on data the model has not seen; it does not predict tomorrow.
- **Outputs** in `outputs/multifactor/`: `summary.csv` and, per stock, full-sample coefficients, VIF,
  rolling coefficients, correlations, walk-forward predictions and metrics.
- **Brazilian 2y/10y yields** are not downloaded automatically (no verified free daily source yet).
  Put your own CSV (`date,br2y,br10y`) in the path `br_yields_csv` and add `br10y` / `br_slope` to
  `common_factors.BR`.
- Iron ore is not on Yahoo in a reliable form, so copper is used as the mining/China proxy for now.

## Scenarios and historical test (phase 2)
```bash
.venv/bin/python -m src.run_factors      # first: data + evidence of which stocks the model explains
.venv/bin/python -m src.run_scenarios    # scenarios + historical test
```
- **Universe**: 30 stocks = 5 per sector (oil, agro, mining/steel) in Brazil and in the US, all in
  `config_multifactor.yaml`. 10 years of daily data.
- **Iron ore**: the run tries `TIO=F` (CME iron ore futures) on Yahoo. If it fails, `FMG.AX` (Fortescue, almost
  pure iron ore) is used as a **proxy** and the output says so. The only free iron ore series confirmed is
  monthly (FRED `PIORECRUSDM`), too slow for a daily model.
- **Scenarios** (`src/scenarios.py`): shocks to factors (prices as fractions, yields in percentage points) become an expected
  move of each stock with an 80% range. Two readings: `only_shocked`, and `with_correlations` (the other
  factors move as they usually do). Shocks bigger than anything in the sample are flagged `beyond_history`.
- **Historical test**: for past episodes (named ones in the config + the 3 worst and 3 best 10-day moves of each
  stock's key factor) the model is fitted only on days before the episode, fed the REAL factor moves of the
  episode, and compared with the real stock move, against market-only and "no move". It tests the translation
  factor -> stock, not the forecasting of the factors.
- **Reliable** = the full model beat market-only out of sample with p < `reliable_p_max` (0.01). Printed scenario
  tables only list reliable stocks; the CSV has all of them. With 30 stocks tested, some false positives are possible.
- Limits: linear model (large shocks are extrapolations), ranges assume independent daily errors (crashes have
  fatter tails), episode dates in the config were written from memory: check them.
