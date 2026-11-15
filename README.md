# equity-research-ai

Two small, reusable equity-research tools in Python 3.11+. Both are driven by a `config.yaml`
(tickers, dates, model name, currency), so you can change inputs without editing code.

| Project | What it does |
|---|---|
| [`earnings-release-extractor/`](earnings-release-extractor/) | Reads earnings-release PDFs, extracts key metrics with the Anthropic API under a strict schema, validates them and writes CSV/Excel plus a list of flags to check by hand. |
| [`oil-equity-fx-model/`](oil-equity-fx-model/) | Measures how much of a Brazilian oil stock's daily returns is explained by Brent and USD/BRL (correlation, OLS, random forest with a time-based split). |

Each project has its own README (how to run, how it works, how to extend, limitations),
`requirements.txt` and tests. Each has its own virtual environment (`python3 -m venv .venv`).

## Status
- `oil-equity-fx-model`: run on real data; results are in its README.
- `earnings-release-extractor`: code and tests done; the real extraction still has to be run with an API key.

## Roadmap
1. Run the extractor on all PDFs, review the flag list against the PDFs, and tune the prompt.
2. Add OCR for scanned PDFs and `pdfplumber` table extraction for slide decks.
3. Extractor: compare several quarters over time (a quarter column and a trend table).
4. Oil model: more factors (iron ore, S&P 500, Selic), rolling-window regressions to track how the
   coefficients change, and walk-forward validation instead of one train/test split.
5. Apply the oil model to other Brazilian stocks (e.g. PRIO3, VALE3) by changing only the config.
6. Link the two: use extractor output (margins, leverage) alongside market data.
