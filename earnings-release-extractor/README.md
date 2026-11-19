# earnings-release-extractor

Reads earnings-release PDFs (text releases or slide decks), asks the Anthropic API to pull out
key metrics under a strict JSON schema, checks the result, and writes a clean table.

**Status:** run on 11 company releases (banks and non-financials). It matches 49 of 49 values of a hand-checked golden set (`golden/golden.csv`, scored by `python -m src.evaluate`). The tests need no API key (the API call is mocked).
The golden set is small and was built after seeing the output: it catches regressions, it is not a guarantee for other companies.

## How to run
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest                     # tests, no key needed

.venv/bin/python -m src.main --dry-run         # reads PDFs, shows pages/chunks, no API call

export ANTHROPIC_API_KEY=...                   # in YOUR terminal only
.venv/bin/python -m src.main                   # real run
.venv/bin/python -m src.main --only vale_3q26_usd.pdf   # one file
```
The key is read from the environment by the Anthropic SDK. The code never prints it or writes it
to a file. Answers are cached in `outputs/cache/`, so re-running does not pay twice; use
`--no-cache` to force new calls.

**Outputs** (`outputs/`): `results.csv`, `results.xlsx` (sheets: Non-financial, Banks, Sources,
Flags), `flags.csv`. The terminal shows a ratio-only comparison and the full list of flags.

## How it works (plain English)
1. **File name** `company_quarter_currency.pdf` gives company, quarter and currency
   (`filename_parser.py`).
2. **Text** is read page by page with pdfplumber and grouped into chunks of up to
   `chunk_pages` pages. Each page is labelled `=== PAGE n ===` (`pdf_reader.py`). A slide counts as a page.
3. **API call** per chunk with a JSON schema (`schema.py`, `extractor.py`). Every metric has a
   `found` flag; `found=false` becomes `null`. Every number carries unit, currency, basis
   (quarter / year-to-date / last 12 months / point in time), the document's own label, page and quote.
4. **Merge**: the first chunk that finds a metric wins. If a later chunk gives a clearly different
   value, a flag is raised.
5. **Validation** (`validator.py`) and **reports** (`report.py`).

## Rules the code follows
- No currency conversion, ever. Comparisons use ratios (margins, net debt/EBITDA, ROE).
- A "scale" change (millions vs billions) inside one currency is allowed for comparing two
  figures of the same company.
- Net income is labelled by definition (`reported`, `recurring_managerial`, `adjusted`), because
  these are not the same thing. A flag is raised if the table mixes definitions.
- Banks (listed in `config.yaml`) get no EBITDA; their metrics (net income, ROE, loan portfolio,
  efficiency ratio, CET1) go in the Banks sheet.
- A number without a page and quote is dropped and flagged.

## The flag list
Everything suspicious is printed at the end and saved in `flags.csv` (with page and quote where
available), so you can open the PDF page and check. Types: EBITDA greater than revenue; net
debt/EBITDA not matching net debt ÷ EBITDA; file name vs PDF mismatch (company, quarter, currency);
quote not found on the cited page; number without source; mixed currencies; bank classification;
bank reporting EBITDA; scanned PDF; differing net income definitions; conflicting values across
chunks; same company in two currencies with margins that differ; failed extraction.

## How to extend
- **New company**: copy the PDF into `data/raw_pdfs/` using the naming format. If it is a bank, add
  its name to `banks:` in `config.yaml`.
- **New metric**: add its name to `METRICS` in `schema.py`, to `BANK_METRICS` / `NON_BANK_METRICS` in
  `report.py`, and mention it in the prompt in `extractor.py`. Change `PROMPT_VERSION` to refresh the cache.
- **Different model**: change `model` in `config.yaml`.

## Limitations (read these)
- **Not yet run against the real API.** First real run may reveal problems I could not test: for
  example the API rejecting the schema as too complex, or the model leaving many fields null.
- **Quote check is partial.** It confirms the quoted sentence exists on the cited page (ignoring
  spaces). It does not prove the number was read correctly from that sentence. Check the flag list
  and the Sources sheet.
- **Net debt/EBITDA check is only like-for-like** when EBITDA and the ratio share a basis. A
  quarterly EBITDA against a last-12-months ratio is reported as "not checkable", not as a mismatch.
- **Tables flatten to text** with pdfplumber, so multi-column slide tables can be misread. Figures
  that only appear in charts are returned as null by design.
- **Scanned PDFs are skipped** (no OCR).
- The model is not told what the file name says, so the name check is independent, but a model
  that cannot find the company or period in a chunk simply leaves it empty (flagged as "cannot confirm").
- Refusals are handled as a failed chunk (flagged). I did not enable the API's server-side
  refusal fallback, to keep the call simple.
- `releases/` is an empty folder that was already there; the code does not use it.
