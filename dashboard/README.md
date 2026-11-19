# Dashboard

One page that puts both projects side by side. It only reads their output files (plus a live scenario box that refits the
factor model locally from the saved data snapshots). No API key, no cost.

## Run

From the `equity-research-ai` folder:

```
oil-equity-fx-model/.venv/bin/pip install streamlit
oil-equity-fx-model/.venv/bin/streamlit run dashboard/app.py
```

(The first time, Streamlit may ask for an email in the Terminal: just press Enter.) The page opens in the browser.
To refresh the numbers, re-run `python -m src.run_factors` and `python -m src.run_scenarios` in `oil-equity-fx-model`,
and `python -m src.main` in `earnings-release-extractor`, then reload the page.

## Tabs

1. **Factor model**: which stocks the multi-factor model explains out of sample, and how each stock's sensitivities drifted.
2. **Scenarios**: preset and live "what if" shocks, and how the model did in past crises.
3. **Months horizon**: do the factors explain / predict 1- and 3-month moves.
4. **Earnings & macro**: inflation and election-year tests, and the earnings-release event study (reaction size, drift, EPS surprise).
5. **Fundamentals**: ratios from the earnings-release extractor, flags to check by hand, accuracy vs the hand-checked golden set.
6. **About / limits**: what the models do not do.
