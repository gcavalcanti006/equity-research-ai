"""Send chunks of a PDF to the Anthropic API and merge the answers.

The API call lives in ONE small function (call_model) so tests can replace the client
with a fake one. The API key is read from the environment by the SDK; we never touch it.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import anthropic

from src.filename_parser import normalize_currency
from src.pdf_reader import split_into_chunks
from src.schema import CHUNK_SCHEMA, METRICS, PREFERRED_BASIS
from src.validator import (CHUNK_CONFLICT, EXTRACTION_FAILED, Flag, looks_like_total_capital,
                           same_value)

PROMPT_VERSION = "v2"   # change this to invalidate the cache after editing the prompt

SYSTEM_PROMPT = """You extract financial figures from pages of an earnings release (text or slide deck).

Strict rules:
1. Use ONLY what is written in the pages you are given. Never guess, estimate, or compute a value.
2. If a metric is not clearly stated in these pages, do NOT add an entry for it. A figure that only
   appears in a chart without a clear number must be left out.
3. Add entries to `figures`. For every entry give: the metric; the number exactly as written (a plain
   number, no thousands separators); unit (millions/billions/thousands for money, percent, times);
   currency (ISO code such as USD or BRL; "none" for ratios and percentages); basis; scope;
   definition_type; a short `definition` note; the label exactly as written in the document;
   the page number from the '=== PAGE n ===' marker; and the exact sentence or table row as quote.
4. BASIS must be stated explicitly: quarter (the current quarter alone), year_to_date (9 months etc.),
   last_12_months, or point_in_time (a balance at a date, such as net debt, loans or CET1).
   Revenue, EBITDA, net income, capex and dividends: extract the CURRENT-QUARTER figure from the
   CONSOLIDATED income statement or consolidated table, not the year-to-date, prior-quarter,
   prior-year or segment columns. Net debt, loans and CET1: the balance at the end of the current quarter.
5. SCOPE: consolidated for the whole company; segment for a business segment, division or subsidiary.
6. DEFINITION_TYPE: reported (as in the financial statements, IFRS/GAAP), adjusted (the company calls
   it adjusted / excluding one-off items), recurring_managerial (Brazilian banks' recurring managerial
   result), or other. If the document shows BOTH a reported and an adjusted figure for the same metric,
   add BOTH as separate entries. The `definition` note says exactly what the number is, for example
   "net income attributable to shareholders", "net income including non-controlling interests",
   "net debt = total debt less cash, excludes leases", "net debt including lease liabilities".
7. Never convert currencies or change units. Report what the document says.
8. Banks do not report EBITDA: leave it out. For banks capture net income, ROE, loan_portfolio,
   efficiency_ratio and cet1.
9. cet1 means ONLY Common Equity Tier 1 (in Portuguese "Capital Principal" or "CET1"). The total capital
   ratio ("Indice de Basileia", "Basel ratio", "total capital ratio", Tier 1 + Tier 2) is NOT CET1:
   leave cet1 out if only the total ratio is shown. Say in `definition` which ratio it is.
10. net_debt_ebitda and roe are ratios: unit 'times' for leverage, 'percent' for ROE and efficiency.
    Companies usually compute net debt/EBITDA on last-12-months EBITDA: set basis accordingly.
11. guidance: add one entry only for explicit forward-looking targets or ranges given by the
    company; otherwise leave the list empty.
12. management_tone: one entry (positive, neutral or cautious) with a short quote that supports it,
    or an empty list if the pages give no basis.
13. Header fields (company, period, currency): copy them as written in the pages, or use "" if the
    pages do not state them. is_bank is true only if the company itself is a bank.
14. An empty `figures` list is a valid answer when nothing is clearly stated."""


class ExtractionError(Exception):
    """One chunk could not be extracted (refusal, truncation, bad JSON...)."""


def describe_error(exc):
    """'ErrorType: message' for the terminal and flags.csv.

    Anything that looks like an API key is replaced, as a safety net (error messages from the
    API do not contain the key, but we never want one in a file or on screen)."""
    text = f"{type(exc).__name__}: {exc}"
    return re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[redacted]", text)


# ---------- the API call ----------

def call_model(client, cfg, chunk):
    """One API call for one chunk. Returns the parsed JSON dict (follows CHUNK_SCHEMA)."""
    response = client.messages.create(
        model=cfg["model"],
        max_tokens=cfg["max_tokens"],
        system=SYSTEM_PROMPT,
        output_config={"effort": cfg["effort"],
                       "format": {"type": "json_schema", "schema": CHUNK_SCHEMA}},
        messages=[{"role": "user", "content":
                   f"Pages {chunk.start_page} to {chunk.end_page} of the document follow.\n\n{chunk.text}"}],
    )
    if response.stop_reason == "refusal":
        raise ExtractionError("the model declined to answer this chunk (stop_reason=refusal)")
    if response.stop_reason == "max_tokens":
        raise ExtractionError("answer was cut off (max_tokens); lower chunk_pages or raise max_tokens")
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"answer was not valid JSON: {exc}") from exc


def _cache_file(cache_dir, stem, chunk, model):
    """Cache key includes model, prompt version, schema and the chunk text, so any change re-runs."""
    key = json.dumps([model, PROMPT_VERSION, CHUNK_SCHEMA, chunk.text], sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{stem}_chunk{chunk.index}_{digest}.json"


# ---------- cleaning and merging ----------

def _clean_figure(raw):
    """Turn one entry of the `figures` list into a tidy dict."""
    cur = raw.get("currency", "none")
    return {"value": raw["value"], "unit": raw["unit"],
            "currency": None if cur.strip().lower() in ("", "none") else (normalize_currency(cur) or cur.upper()),
            "basis": raw["basis"], "scope": raw.get("scope", "other"),
            "definition_type": raw.get("definition_type", "other"),
            "definition": raw.get("definition", ""), "label": raw.get("label", ""),
            "page": raw.get("page", 0), "quote": raw.get("quote", "")}


def clean_chunk(raw):
    """Turn the model's JSON into our internal shape.

    A chunk may list SEVERAL candidates for one metric (e.g. reported and adjusted net income);
    merge_chunks chooses between them."""
    candidates = {m: [] for m in METRICS}
    for entry in raw["figures"]:
        candidates[entry["metric"]].append(_clean_figure(entry))
    out = {"company": raw["company"].strip(), "period": raw["period"].strip(),
           "currency": raw["currency"].strip(), "is_bank": raw["is_bank"],
           "candidates": candidates, "guidance": None, "tone": None}
    if raw["guidance"]:
        g = raw["guidance"][0]
        out["guidance"] = {"text": g["text"], "page": g["page"], "quote": g["quote"]}
    if raw["management_tone"]:
        t = raw["management_tone"][0]
        out["tone"] = {"text": t["tone"], "page": t["page"], "quote": t["quote"]}
    return out


def _most_common(values):
    values = [v for v in values if v and v not in ("unknown", "not_stated")]
    return Counter(values).most_common(1)[0][0] if values else ""


def rank_key(metric, fig):
    """Lower is better. Order of preference when a metric appears more than once:
    1. consolidated figures over segment figures
    2. the basis we want for that metric (quarter for flows, point-in-time for balances, ...)
    3. reported figures over adjusted / recurring ones (the income statement is the reference)
    4. (CET1 only) a real CET1 over the total capital ratio
    Ties are broken by document order (earlier in the document wins)."""
    wanted = PREFERRED_BASIS.get(metric)
    return (metric == "cet1" and looks_like_total_capital(fig),
            fig["scope"] != "consolidated",
            wanted is not None and fig["basis"] != wanted,
            fig["definition_type"] != "reported")


def merge_chunks(stem, cleaned, rel_tol=0.01):
    """Combine chunk answers into one result. For each metric, choose the best candidate with
    rank_key. Flag a conflict ONLY when another candidate has the same basis, scope and
    definition type as the chosen one and still differs (after unit normalization)."""
    flags = []
    result = {"file": stem, "company": stem.split("_")[0],
              "pdf_company": _most_common([c["company"] for c in cleaned]),
              "pdf_period": _most_common([c["period"] for c in cleaned]),
              "pdf_currency": _most_common([c["currency"] for c in cleaned]),
              "is_bank": any(c["is_bank"] for c in cleaned),
              "metrics": {}, "guidance": None, "tone": None}
    for name in METRICS:
        candidates = [f for c in cleaned for f in c["candidates"][name]]   # document order
        if not candidates:
            result["metrics"][name] = None
            continue
        best = min(candidates, key=lambda f: rank_key(name, f))            # first of equal ranks
        result["metrics"][name] = best
        kind = (best["scope"], best["basis"], best["definition_type"])
        for other in candidates:
            if other is best or (other["scope"], other["basis"], other["definition_type"]) != kind:
                continue
            if name == "cet1" and looks_like_total_capital(other) and not looks_like_total_capital(best):
                continue        # a total capital ratio is not a rival CET1 value
            if not same_value(best, other, rel_tol):
                flags.append(Flag(stem, name, CHUNK_CONFLICT,
                                  f"kept {best['value']} {best['unit']} p.{best['page']} ({best['definition']}); "
                                  f"same basis/definition type but p.{other['page']} says "
                                  f"{other['value']} {other['unit']} ({other['definition']})",
                                  other["page"], other["quote"]))
                break
    ni = result["metrics"]["net_income"]
    result["net_income_definition"] = ni["definition_type"] if ni else "not_stated"
    for key in ("guidance", "tone"):
        found = [c[key] for c in cleaned if c[key]]
        result[key] = found[0] if found else None
    return result, flags


# ---------- one whole document ----------

def extract_document(client, pages, cfg, stem, cache_dir=None):
    """Extract one document (already read into `pages`). Returns (result, flags).

    A failed chunk becomes a flag; the other chunks still count. Authentication problems stop the run."""
    flags, cleaned = [], []
    for chunk in split_into_chunks(pages, cfg["chunk_pages"], cfg["max_chars_per_chunk"]):
        cache = _cache_file(cache_dir, stem, chunk, cfg["model"]) if cache_dir else None
        try:
            if cache and cache.exists():
                raw = json.loads(cache.read_text())
            else:
                raw = call_model(client, cfg, chunk)
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps(raw))
            cleaned.append(clean_chunk(raw))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError):
            raise   # a wrong key is not a per-document problem
        except (ExtractionError, anthropic.APIError, KeyError) as exc:
            reason = describe_error(exc)
            where = f"pages {chunk.start_page}-{chunk.end_page}"
            print(f"  ! {stem} {where} FAILED -> {reason}")      # show the real reason now
            flags.append(Flag(stem, where, EXTRACTION_FAILED, reason))
    if not cleaned:
        return None, flags
    result, merge_flags = merge_chunks(stem, cleaned, cfg.get("validation", {}).get("conflict_rel_tol", 0.01))
    return result, flags + merge_flags
