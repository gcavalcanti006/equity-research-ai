# equity-research-ai

Two research tools I built to answer a practical question: **what actually moves commodity-linked stocks, and can the
numbers in earnings releases be pulled out reliably and checked?** Python, tested, with a Streamlit dashboard on top.
The emphasis is on **honest testing**: every claim is checked out of sample, and the negative results are kept.

| Project | What it does |
|---|---|
| [`oil-equity-fx-model/`](oil-equity-fx-model/) | Multi-factor model of **29 stocks** (oil, agro, mining/steel; Brazil and US), 20 years of daily data. Factors: market index, USD/BRL, US yields, VIX, DXY, oil, grains, sugar, copper, iron ore. Scenario engine, 1-3 month horizon study, inflation and election tests, earnings-release event study, probability cones. |
| [`earnings-release-extractor/`](earnings-release-extractor/) | Reads earnings-release PDFs, extracts key metrics with the Anthropic API under a strict schema, **cross-checks them against each other** (e.g. net debt vs net debt/EBITDA) and compares with a hand-verified golden set. |
| [`dashboard/`](dashboard/) | Streamlit page that puts everything side by side (7 tabs). |

## What I found (plain English)
**Factor model**
- The factors explain a lot for **oil and mining/steel stocks** (19 of 29 stocks clearly beat a "market only" model out of sample, p < 0.01 and a higher R²).
  For most agro stocks the market alone explains almost everything.
- Over **1 and 3 months** the factors still *explain* what a stock did (in 13 of 29 stocks at one month), but **none of them predicts the next month or quarter**.
  This is the main result: the model is a tool to understand and to run "what if" scenarios, not a trading signal.
- **Inflation (US CPI, Brazil IPCA) and presidential-election years** were tested on top of the model in 324 out-of-sample tests. Nothing survived a correction for multiple testing.
- **Earnings releases** (1,536 events): stocks move about **2x more than a normal 2-day move** after a release, even after removing the factors' effect.
  The EPS surprise lines up with the reaction. There is **no detectable drift** afterwards.
- **Probability cones** (simulated price paths with a neutral centre) were checked against the past: the 80% cone built with today's volatility contained the real outcome
  79% of the time in turbulent periods, versus 72% for a plain 5-year cone.

**Extractor**
- On 11 company releases it matches **49 of 49** hand-checked values. False alarms fell from 17 to 6; the rest are cases where the document itself does not give the number.
- Built-in checks catch things like mixed currencies, different net-income definitions and ratios that do not add up. A value is never invented: when the release does not state it, the answer is "n/a".

## Honest limits
- Linear model; large shocks are extrapolations (flagged). Ranges for US oil stocks in past crashes were too narrow.
- The golden set is small (49 values) and was built after seeing the extractor's output: it catches regressions, it does not guarantee accuracy on other companies.
- Earnings dates come from Yahoo; some Brazilian dates may be off by a day or missing (RAIZ4 has none, AGRO3 has 4 events).
- Not investment advice.

## How to run
```bash
cd oil-equity-fx-model
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.run_factors          # downloads data, fits the model for every stock
.venv/bin/python -m src.run_scenarios        # scenarios + test on past episodes
.venv/bin/python -m src.run_long_horizon     # 1 and 3 months
.venv/bin/python -m src.run_macro            # inflation and elections
.venv/bin/python -m src.run_earnings         # earnings event study
.venv/bin/python -m src.run_cones            # calibration of the cones
.venv/bin/python -m pytest                   # tests, no internet needed
cd .. && oil-equity-fx-model/.venv/bin/pip install streamlit
oil-equity-fx-model/.venv/bin/streamlit run dashboard/app.py
```
The extractor needs an `ANTHROPIC_API_KEY` in your own terminal (see its README); the model and the dashboard need no key.

## Possible next steps
- Extractor: several quarters per company, OCR for scanned PDFs, a "always adjusted / always reported" setting for definitions.
- Link the two: use the extractor's margins and leverage next to the model's sensitivities.
- Fill in the missing Brazilian earnings dates by hand (`oil-equity-fx-model/data/earnings_manual.csv`) to firm up the Brazil results.
