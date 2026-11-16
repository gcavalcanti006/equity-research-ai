import pytest

from src.filename_parser import (normalize_currency, normalize_period, parse_filename,
                                 same_company)


def test_parse_simple_name():
    p = parse_filename("petrobras_3q26_usd.pdf")
    assert (p.company, p.quarter, p.year, p.currency, p.period) == ("petrobras", 3, 2026, "USD", "3Q26")


def test_parse_hyphenated_company_and_brl():
    p = parse_filename("itau-unibanco_3q26_brl.pdf")
    assert p.company == "itau-unibanco" and p.currency == "BRL"


@pytest.mark.parametrize("bad", ["petrobras.pdf", "petrobras_3q26.pdf", "petrobras_5q26_usd.pdf",
                                 "petrobras_3q2026_usd.pdf", "petrobras_3q26_dollar.pdf"])
def test_bad_names_raise(bad):
    with pytest.raises(ValueError):
        parse_filename(bad)


@pytest.mark.parametrize("text,expected", [
    ("3Q26", (3, 2026)), ("Q3 2026", (3, 2026)), ("3Q 2026", (3, 2026)), ("Q3'26", (3, 2026)),
    ("Third quarter of 2026", (3, 2026)), ("third quarter 2026", (3, 2026)),
    ("", None), ("nine months 2026", None)])
def test_normalize_period(text, expected):
    assert normalize_period(text) == expected


def test_normalize_currency():
    assert normalize_currency("US$") == "USD"
    assert normalize_currency("R$") == "BRL"
    assert normalize_currency("eur") == "EUR"
    assert normalize_currency("") is None


def test_same_company_ignores_accents_and_suffixes():
    assert same_company("itau-unibanco", "Itaú Unibanco Holding S.A.")
    assert same_company("jpmorgan", "JPMorgan Chase & Co.")
    assert same_company("exxonmobil", "Exxon Mobil Corporation")
    assert not same_company("vale", "Petróleo Brasileiro S.A.")


@pytest.mark.parametrize("text", [
    "3T26", "3T 2026", "T3 2026", "3º trimestre de 2026", "3° trimestre de 2026",
    "3o trimestre de 2026", "3º Trimestre 2026", "Terceiro trimestre de 2026",
    "Resultados do 3T26", "Q3 2026", "3Q26", "3rd quarter 2026", "Third Quarter 2026",
    "third quarter of 2026", "Q3'26"])
def test_portuguese_and_english_periods_map_to_3q26(text):
    assert normalize_period(text) == (3, 2026)
    p = parse_filename("itau-unibanco_3q26_brl.pdf")
    assert normalize_period(text) == (p.quarter, p.year)


def test_other_quarters_and_unreadable_periods():
    assert normalize_period("2T25") == (2, 2025)
    assert normalize_period("4º trimestre de 2025") == (4, 2025)
    assert normalize_period("Primeiro trimestre de 2026") == (1, 2026)
    assert normalize_period("Resultados 2026") is None
    assert normalize_period("9M26") is None


@pytest.mark.parametrize("text", [
    "Third-quarter 2026", "third-quarter 2026", "Third-Quarter of 2026", "3rd-quarter 2026",
    "3Q-26", "Third–quarter 2026".replace("–", "-")])
def test_hyphenated_quarter_formats(text):
    assert normalize_period(text) == (3, 2026)


def test_period_with_special_hyphen_and_nbsp():
    # real PDFs use non-breaking hyphens / spaces (this made JPMorgan 'Third-quarter 2026' unreadable)
    from src.filename_parser import normalize_period
    assert normalize_period("Third‑quarter 2026") == (3, 2026)
    assert normalize_period("Third–quarter 2026") == (3, 2026)
