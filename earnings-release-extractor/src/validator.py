"""Validation checks. Each check returns a list of Flag objects (empty = nothing wrong).

A flag never changes a number silently: it only tells you what to look at in the PDF.
(The one exception is a number with no source, which we drop and flag.)
"""
import re
import unicodedata
from dataclasses import dataclass

from src.filename_parser import (normalize_currency, normalize_period, same_company)
from src.schema import METRICS

# Flag types (the 'flag_type' column in flags.csv)
EBITDA_GT_REVENUE = "EBITDA_GREATER_THAN_REVENUE"
RATIO_MISMATCH = "NET_DEBT_EBITDA_MISMATCH"
RATIO_NOT_CHECKABLE = "NET_DEBT_EBITDA_NOT_CHECKABLE"
NAME_MISMATCH = "FILENAME_PDF_MISMATCH"
QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND_ON_PAGE"
NO_SOURCE = "NUMBER_WITHOUT_SOURCE"
CURRENCY_MIX = "MIXED_CURRENCIES"
BANK_CLASS = "BANK_CLASSIFICATION"
BANK_EBITDA = "BANK_REPORTED_EBITDA"
SCANNED = "SCANNED_PDF_NO_TEXT"
DEFINITIONS = "NET_INCOME_DEFINITIONS_DIFFER"
CHUNK_CONFLICT = "CONFLICTING_VALUES_ACROSS_CHUNKS"
EXTRACTION_FAILED = "EXTRACTION_FAILED"
BAD_FILENAME = "BAD_FILENAME"
CET1_DEFINITION = "CET1_IS_TOTAL_CAPITAL_RATIO"
PAIR_DIFF = "SAME_COMPANY_RATIO_DIFFERS_ACROSS_CURRENCIES"

SCALE = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}


@dataclass
class Flag:
    company: str
    field: str
    flag_type: str
    message: str
    page: int | None = None
    quote: str = ""

    def as_dict(self):
        return {"company": self.company, "field": self.field, "flag_type": self.flag_type,
                "message": self.message, "page": self.page, "quote": self.quote}


def to_base(fig):
    """Value in plain units (millions -> x 1,000,000) so two figures can be compared.
    This is only a SCALE change inside one currency, never a currency conversion.
    Returns None for ratios/percentages/unknown units."""
    if fig is None or fig.get("unit") not in SCALE:
        return None
    return fig["value"] * SCALE[fig["unit"]]


def _fig_source(fig):
    return (fig.get("page"), fig.get("quote", "")) if fig else (None, "")


# ---------- individual checks ----------

def enforce_sources(result):
    """A number must have a page and a quote. If not, drop it (-> null) and flag it."""
    flags = []
    for name in METRICS:
        fig = result["metrics"].get(name)
        if fig is None:
            continue
        if not fig.get("page") or fig["page"] <= 0 or not fig.get("quote", "").strip():
            flags.append(Flag(result["file"], name, NO_SOURCE,
                              f"value {fig['value']} {fig['unit']} had no page/quote; set to null"))
            result["metrics"][name] = None
    return flags


def _plain(text):
    """Lower-case and remove accents."""
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _numbers(text):
    """Numbers in the text with thousands/decimal separators removed, so '1.234,5' and
    '1,234.5' both become '12345'."""
    return {re.sub(r"[.,]", "", n) for n in re.findall(r"\d+(?:[.,]\d+)*", _plain(text))}


def _words(text):
    return set(re.findall(r"[a-z]+", _plain(text)))


def quote_match(quote, page_text, min_words=0.8):
    """Is the quote on the page? Order does not matter (slide text is often scrambled).

    Pass if ALL numbers of the quote are on the page AND at least `min_words` (80%) of its
    words are on the page. Case, accents, spacing and punctuation are ignored.
    Returns (ok, reason)."""
    page_numbers, page_words = _numbers(page_text), _words(page_text)
    missing = sorted(_numbers(quote) - page_numbers)
    if missing:
        return False, f"numbers not on the page: {', '.join(missing)}"
    words = _words(quote)
    if words:
        share = len(words & page_words) / len(words)
        if share < min_words:
            return False, f"only {share:.0%} of the quote's words are on the page (need {min_words:.0%})"
    return True, ""


def check_quotes(result, pages, min_words=0.8):
    """Each quote must really appear on the cited page (guards against invented citations)."""
    flags = []
    items = [(n, result["metrics"].get(n)) for n in METRICS]
    items += [("guidance", result.get("guidance")), ("management_tone", result.get("tone"))]
    for name, fig in items:
        if not fig:
            continue
        page, quote = fig.get("page"), fig.get("quote", "")
        if not page or not (1 <= page <= len(pages)):
            flags.append(Flag(result["file"], name, QUOTE_NOT_FOUND,
                              f"cited page {page} does not exist (PDF has {len(pages)} pages)",
                              page, quote))
            continue
        ok, reason = quote_match(quote, pages[page - 1], min_words)
        if not ok:
            flags.append(Flag(result["file"], name, QUOTE_NOT_FOUND,
                              f"quote not found on page {page}: {reason}", page, quote))
    return flags


def same_value(a, b, rel_tol=0.01):
    """Do two extracted figures of the same metric agree, after putting them on the same scale?

    'R$ 10.3 billion' and 'R$ 10,312 million' agree. Different currencies never agree (no conversion).
    Differences smaller than rel_tol (default 1%, to allow rounding) are ignored."""
    if a["currency"] != b["currency"]:
        return False
    base_a, base_b = to_base(a), to_base(b)
    if base_a is not None and base_b is not None:
        x, y = base_a, base_b
    elif a["unit"] == b["unit"]:
        x, y = a["value"], b["value"]
    else:
        return False
    return abs(x - y) <= rel_tol * max(abs(x), abs(y), 1e-12)


def check_filename_vs_pdf(parsed, result):
    """Compare company, quarter and currency from the file name with what the PDF says."""
    flags, c = [], result["file"]
    pdf_company = result.get("pdf_company") or ""
    if not pdf_company:
        flags.append(Flag(c, "company", NAME_MISMATCH, "PDF content did not state the company name; cannot confirm"))
    elif not same_company(parsed.company, pdf_company):
        flags.append(Flag(c, "company", NAME_MISMATCH,
                          f"file name says '{parsed.company}', PDF says '{pdf_company}'"))

    pdf_period = result.get("pdf_period") or ""
    got = normalize_period(pdf_period)
    if got is None:
        flags.append(Flag(c, "period", NAME_MISMATCH,
                          f"could not read a quarter from PDF period '{pdf_period}'; cannot confirm {parsed.period}"))
    elif got != (parsed.quarter, parsed.year):
        flags.append(Flag(c, "period", NAME_MISMATCH,
                          f"file name says {parsed.period}, PDF says '{pdf_period}'"))

    pdf_cur = normalize_currency(result.get("pdf_currency"))
    if pdf_cur is None:
        flags.append(Flag(c, "currency", NAME_MISMATCH,
                          f"PDF currency '{result.get('pdf_currency')}' not recognised; cannot confirm {parsed.currency}"))
    elif pdf_cur != parsed.currency:
        flags.append(Flag(c, "currency", NAME_MISMATCH,
                          f"file name says {parsed.currency}, PDF says {pdf_cur}"))
    return flags


def check_ebitda_vs_revenue(result):
    rev, ebitda = result["metrics"].get("net_revenue"), result["metrics"].get("ebitda")
    if not rev or not ebitda:
        return []
    c = result["file"]
    if rev["currency"] != ebitda["currency"]:
        return [Flag(c, "ebitda", CURRENCY_MIX,
                     f"EBITDA in {ebitda['currency']} but revenue in {rev['currency']}; not compared")]
    r, e = to_base(rev), to_base(ebitda)
    if r is None or e is None:
        return []
    if e > r:
        note = "" if rev["basis"] == ebitda["basis"] else f" (bases differ: revenue {rev['basis']}, EBITDA {ebitda['basis']})"
        page, quote = _fig_source(ebitda)
        return [Flag(c, "ebitda", EBITDA_GT_REVENUE,
                     f"EBITDA {ebitda['value']:,.1f} {ebitda['unit']} > revenue {rev['value']:,.1f} {rev['unit']} {ebitda['currency']}{note}",
                     page, quote)]
    return []


def check_leverage(result, rel_tol, abs_tol):
    """Reported net debt/EBITDA vs net debt divided by EBITDA.

    Companies often report the ratio on last-12-months EBITDA while the extracted EBITDA is for
    one quarter. Then the check is not like-for-like, so we say so instead of raising a false alarm."""
    m = result["metrics"]
    nd, eb, ratio = m.get("net_debt"), m.get("ebitda"), m.get("net_debt_ebitda")
    if not (nd and eb and ratio):
        return []
    c = result["file"]
    page, quote = _fig_source(ratio)
    if nd["currency"] != eb["currency"]:
        return [Flag(c, "net_debt_ebitda", CURRENCY_MIX,
                     f"net debt in {nd['currency']} but EBITDA in {eb['currency']}; ratio not checked", page, quote)]
    n, e = to_base(nd), to_base(eb)
    if n is None or e is None or e <= 0:
        return []
    # Only compare like with like: the ratio and the EBITDA must have the SAME known basis.
    # (Reported ratios are usually on last-12-months EBITDA; a quarterly EBITDA would
    # produce a false mismatch.)
    known = {"quarter", "year_to_date", "last_12_months"}
    if not (eb["basis"] in known and eb["basis"] == ratio["basis"]):
        return [Flag(c, "net_debt_ebitda", RATIO_NOT_CHECKABLE,
                     f"EBITDA basis is '{eb['basis']}' but the ratio basis is '{ratio['basis']}': "
                     "not like-for-like, check by hand", page, quote)]
    adj = re.compile(r"adjusted|ajustad", re.I)
    ratio_txt = _plain(f"{ratio.get('label', '')} {ratio.get('definition', '')}")
    nd_txt = _plain(f"{nd.get('label', '')} {nd.get('definition', '')}")
    if adj.search(ratio_txt) and not adj.search(nd_txt):
        return [Flag(c, "net_debt_ebitda", RATIO_NOT_CHECKABLE,
                     "the reported ratio is an ADJUSTED leverage ratio (adjusted net debt) but the extracted net debt "
                     "is not adjusted: not like-for-like, check by hand", page, quote)]
    computed, reported = n / e, ratio["value"]
    if abs(computed - reported) > max(abs_tol, rel_tol * abs(reported)):
        return [Flag(c, "net_debt_ebitda", RATIO_MISMATCH,
                     f"reported {reported:.2f}x vs computed {computed:.2f}x "
                     f"(net debt {nd['value']:,.1f} / EBITDA {eb['value']:,.1f}, {nd['currency']} {nd['unit']})",
                     page, quote)]
    return []


_TOTAL_CAPITAL = re.compile(r"basileia|basel|total capital|capital total|tier 2|nivel 2", re.I)
_CET1 = re.compile(r"cet\s*1|common equity|capital principal", re.I)


def looks_like_total_capital(fig):
    """True if a figure filed as CET1 is described as the total capital (Basel) ratio instead."""
    text = _plain(f"{fig.get('label', '')} {fig.get('definition', '')}")
    return bool(_TOTAL_CAPITAL.search(text)) and not _CET1.search(text)


def has_cet1_evidence(fig):
    """True if the label, definition or quote actually says CET1 / common equity / capital principal."""
    text = _plain(f"{fig.get('label', '')} {fig.get('definition', '')} {fig.get('quote', '')}")
    return bool(_CET1.search(text))


def check_cet1(result):
    """CET1 is not the total capital ratio. If that is all we have, drop it and flag.
    Also drop it when nothing in the label, definition or quote says it is CET1 (a bare 'capital ratio'
    could be any of Tier 1, total capital...)."""
    fig = result["metrics"].get("cet1")
    if fig and not looks_like_total_capital(fig) and not has_cet1_evidence(fig):
        result["metrics"]["cet1"] = None
        return [Flag(result["file"], "cet1", CET1_DEFINITION,
                     f"nothing in '{fig.get('label', '')}' / definition / quote says this is CET1; set to null",
                     fig.get("page"), fig.get("quote", ""))]
    if fig and looks_like_total_capital(fig):
        result["metrics"]["cet1"] = None
        return [Flag(result["file"], "cet1", CET1_DEFINITION,
                     f"'{fig.get('label', '')}' looks like the total capital ratio, not CET1; set to null",
                     fig.get("page"), fig.get("quote", ""))]
    return []


def check_bank_rules(result, is_bank_in_config):
    """Banks have no EBITDA. Also flag a company the model thinks is a bank but config does not list."""
    c, flags = result["file"], []
    if not is_bank_in_config and result.get("is_bank"):
        flags.append(Flag(c, "is_bank", BANK_CLASS,
                          "model says this is a bank but it is not in config 'banks'; treated as a bank"))
    if (is_bank_in_config or result.get("is_bank")) and result["metrics"].get("ebitda"):
        fig = result["metrics"]["ebitda"]
        flags.append(Flag(c, "ebitda", BANK_EBITDA,
                          "banks do not report EBITDA; value removed (set to null)", fig.get("page"), fig.get("quote", "")))
        result["metrics"]["ebitda"] = None
    return flags


# ---------- checks across companies ----------

def check_currency_mix(rows):
    """One flag per run if the table mixes currencies: absolute values are not comparable."""
    currencies = sorted({r["currency"] for r in rows if r.get("currency")})
    if len(currencies) > 1:
        return [Flag("(all)", "currency", CURRENCY_MIX,
                     f"results mix {', '.join(currencies)}: no conversion done; compare ratios only")]
    return []


def check_definitions(results):
    """'Recurring managerial', 'adjusted' and 'reported' net income are different definitions."""
    defs = {}
    for r in results:
        d = r.get("net_income_definition")
        if r["metrics"].get("net_income") and d:
            defs.setdefault(d, []).append(r["file"])
    if len(defs) > 1:
        detail = "; ".join(f"{d}: {', '.join(cs)}" for d, cs in sorted(defs.items()))
        return [Flag("(all)", "net_income", DEFINITIONS,
                     f"net income definitions differ ({detail}); not strictly comparable")]
    return []


def _margin(result, numerator):
    rev, num = result["metrics"].get("net_revenue"), result["metrics"].get(numerator)
    if not rev or not num or rev["currency"] != num["currency"] or rev["basis"] != num["basis"]:
        return None
    r, n = to_base(rev), to_base(num)
    return None if not r or n is None else 100 * n / r


def ebitda_margin(result):
    return _margin(result, "ebitda")


def net_margin(result):
    return _margin(result, "net_income")


def check_pairs(results, tol_pp):
    """Same company in two currencies: margins should be close, because they are ratios."""
    by_company = {}
    for r in results:
        by_company.setdefault(r["company"], []).append(r)
    flags = []
    for company, group in by_company.items():
        if len(group) < 2:
            continue
        for name, fn in (("ebitda_margin", ebitda_margin), ("net_margin", net_margin)):
            vals = [(g["currency"], fn(g)) for g in group if fn(g) is not None]
            if len(vals) >= 2 and max(v for _, v in vals) - min(v for _, v in vals) > tol_pp:
                txt = ", ".join(f"{cur} {v:.1f}%" for cur, v in vals)
                flags.append(Flag(company, name, PAIR_DIFF,
                                  f"same company, different currency files: {txt} (> {tol_pp} pp apart)"))
    return flags


# ---------- everything for one company ----------

def validate_company(result, parsed, pages, cfg):
    """Run all per-company checks. Order matters: sources first, then everything that uses numbers."""
    v = cfg["validation"]
    is_bank = parsed.company in cfg["banks"]
    flags = []
    flags += check_filename_vs_pdf(parsed, result)
    flags += enforce_sources(result)
    flags += check_quotes(result, pages, v.get("quote_match", 0.8))
    flags += check_bank_rules(result, is_bank)
    flags += check_cet1(result)
    result["company"], result["currency"] = parsed.company, parsed.currency
    result["kind"] = "bank" if (is_bank or result.get("is_bank")) else "non_financial"
    flags += check_ebitda_vs_revenue(result)
    flags += check_leverage(result, v["ratio_rel_tol"], v["ratio_abs_tol"])
    return flags
