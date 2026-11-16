"""Read company, quarter and currency from a file name, and compare with PDF content.

File name format: company_quarter_currency.pdf   e.g. petrobras_3q26_usd.pdf
"""
import re
import unicodedata
from dataclasses import dataclass

# company (letters/digits, may contain hyphens) _ quarter like 3q26 _ 3-letter currency
_NAME = re.compile(
    r"^(?P<company>[a-z0-9]+(?:-[a-z0-9]+)*)_(?P<q>[1-4])q(?P<yy>\d{2})_(?P<cur>[a-z]{3})$",
    re.IGNORECASE)


@dataclass
class ParsedName:
    company: str      # e.g. "itau-unibanco"
    quarter: int      # 1-4
    year: int         # e.g. 2026
    currency: str     # e.g. "BRL"

    @property
    def period(self):
        return f"{self.quarter}Q{self.year % 100:02d}"


def parse_filename(filename):
    """Return a ParsedName, or raise ValueError if the name does not follow the format."""
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    m = _NAME.match(stem)
    if not m:
        raise ValueError(f"File name '{filename}' does not match company_quarter_currency.pdf")
    return ParsedName(company=m["company"].lower(), quarter=int(m["q"]),
                      year=2000 + int(m["yy"]), currency=m["cur"].upper())


def _year(text):
    n = int(text)
    return 2000 + n if n < 100 else n


_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4,
          "primeiro": 1, "segundo": 2, "terceiro": 3, "quarto": 4}


def _plain(text):
    """Lower-case, remove accents and ordinal signs ('3º' -> '3', 'trimestre' stays)."""
    text = text.replace("º", "").replace("°", "").replace("ª", "")
    # PDFs use "special" hyphens/dashes and non-breaking spaces (e.g. "Third\u2011quarter 2026"): make them plain.
    text = re.sub("[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    text = re.sub("[\u00a0\u2007\u202f\u2009\u200a]", " ", text)
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower().strip()


def normalize_period(text):
    """Turn a period written in English or Portuguese into (quarter, year).

    Understands: 3Q26, 3Q-26, Third-quarter 2026, 3Q 2026, Q3 2026, Q3'26, 3T26, 3T 2026, T3 2026,
    3rd quarter 2026, third quarter of 2026, 3º trimestre de 2026, terceiro trimestre de 2026.
    Returns None if we cannot read it (then we cannot compare, and we say so)."""
    if not text:
        return None
    t = _plain(text)
    m = re.search(r"\b([1-4])\s*[qt][\s'\-]*(\d{4}|\d{2})\b", t)             # 3q26, 3t26, 3t 2026
    if m:
        return int(m[1]), _year(m[2])
    m = re.search(r"\b[qt]\s*([1-4])\s*[\s/'\-]*\s*(\d{4}|\d{2})\b", t)     # q3 2026, t3'26
    if m:
        return int(m[1]), _year(m[2])
    m = re.search(r"\b([1-4])\s*(?:st|nd|rd|th|o|a)?[\s\-]+(?:quarter|trimestre|tri)\b"
                  r"[\s,]*(?:of|de|del)?\s*'?(\d{4}|\d{2})\b", t)             # 3rd quarter 2026, 3 trimestre de 2026
    if m:
        return int(m[1]), _year(m[2])
    m = re.search(r"\b(first|second|third|fourth|primeiro|segundo|terceiro|quarto)[\s\-]+"
                  r"(?:quarter|trimestre)[\s,]*(?:of|de|del)?\s*(\d{4})\b", t)  # third quarter of 2026
    if m:
        return _WORDS[m[1]], int(m[2])
    return None


_CURRENCY_WORDS = {"usd": "USD", "us$": "USD", "u$s": "USD", "us dollar": "USD",
                   "us dollars": "USD", "dollar": "USD", "dollars": "USD",
                   "brl": "BRL", "r$": "BRL", "reais": "BRL", "real": "BRL"}


def normalize_currency(text):
    """'US$', 'USD' -> 'USD'; 'R$', 'BRL' -> 'BRL'. Unknown 3-letter codes are upper-cased."""
    if not text:
        return None
    t = text.strip().lower()
    if t in _CURRENCY_WORDS:
        return _CURRENCY_WORDS[t]
    if re.fullmatch(r"[a-z]{3}", t):
        return t.upper()
    return None


def _squash(name):
    """Lower-case, remove accents, keep only letters and digits."""
    no_accents = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", no_accents.lower())


def same_company(file_company, pdf_company):
    """True if one name contains the other after cleaning
    ('itau-unibanco' vs 'Itaú Unibanco Holding S.A.')."""
    a, b = _squash(file_company), _squash(pdf_company)
    return bool(a) and bool(b) and (a in b or b in a)
