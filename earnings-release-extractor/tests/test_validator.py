from src import validator as v
from src.filename_parser import parse_filename
from tests.helpers import fig, result

CFG = {"banks": ["jpmorgan"], "validation": {"ratio_rel_tol": 0.10, "ratio_abs_tol": 0.10}}


def types(flags):
    return [f.flag_type for f in flags]


def test_ebitda_greater_than_revenue_flagged():
    r = result(net_revenue=fig(100), ebitda=fig(150))
    assert types(v.check_ebitda_vs_revenue(r)) == [v.EBITDA_GT_REVENUE]


def test_ebitda_less_than_revenue_ok_and_scales_compared():
    # 2 billions vs 1500 millions: EBITDA (2,000m) > revenue (1,500m) only after matching scales
    assert types(v.check_ebitda_vs_revenue(result(net_revenue=fig(1500), ebitda=fig(2, "billions")))) == [v.EBITDA_GT_REVENUE]
    assert v.check_ebitda_vs_revenue(result(net_revenue=fig(1500), ebitda=fig(1, "billions"))) == []


def test_ebitda_revenue_different_currency_is_not_compared():
    r = result(net_revenue=fig(100, currency="USD"), ebitda=fig(500, currency="BRL"))
    assert types(v.check_ebitda_vs_revenue(r)) == [v.CURRENCY_MIX]


def test_leverage_matches():
    r = result(net_debt=fig(300, basis="point_in_time"), ebitda=fig(100, basis="last_12_months"),
               net_debt_ebitda=fig(3.0, "times", None, "last_12_months"))
    assert v.check_leverage(r, 0.10, 0.10) == []


def test_leverage_mismatch_flagged():
    r = result(net_debt=fig(300, basis="point_in_time"), ebitda=fig(100, basis="last_12_months"),
               net_debt_ebitda=fig(1.5, "times", None, "last_12_months"))
    assert types(v.check_leverage(r, 0.10, 0.10)) == [v.RATIO_MISMATCH]


def test_leverage_quarterly_vs_ltm_is_not_a_false_alarm():
    r = result(net_debt=fig(300, basis="point_in_time"), ebitda=fig(75, basis="quarter"),
               net_debt_ebitda=fig(1.0, "times", None, "last_12_months"))
    assert types(v.check_leverage(r, 0.10, 0.10)) == [v.RATIO_NOT_CHECKABLE]


def test_leverage_skipped_when_something_missing():
    assert v.check_leverage(result(net_debt=fig(300)), 0.10, 0.10) == []


def test_number_without_source_is_dropped_and_flagged():
    r = result(net_income=fig(10, quote=""), capex=fig(5, page=0, quote="x"))
    flags = v.enforce_sources(r)
    assert types(flags) == [v.NO_SOURCE, v.NO_SOURCE]
    assert r["metrics"]["net_income"] is None and r["metrics"]["capex"] is None


def test_quote_must_be_on_the_cited_page():
    pages = ["Net income was 10 million", "Capex  of 5 million"]
    ok = result(net_income=fig(10, page=1, quote="net income was 10 million"))
    bad = result(net_income=fig(10, page=2, quote="net income was 10 million"))
    ghost = result(net_income=fig(10, page=9, quote="x"))
    assert v.check_quotes(ok, pages) == []
    assert types(v.check_quotes(bad, pages)) == [v.QUOTE_NOT_FOUND]
    assert types(v.check_quotes(ghost, pages)) == [v.QUOTE_NOT_FOUND]


def test_filename_vs_pdf_mismatch():
    parsed = parse_filename("acme_3q26_usd.pdf")
    assert v.check_filename_vs_pdf(parsed, result()) == []
    wrong = result()
    wrong.update(pdf_company="Other Inc", pdf_period="2Q26", pdf_currency="R$")
    flags = v.check_filename_vs_pdf(parsed, wrong)
    assert [f.field for f in flags] == ["company", "period", "currency"]
    assert set(types(flags)) == {v.NAME_MISMATCH}


def test_bank_ebitda_removed_and_unlisted_bank_flagged():
    r = result(ebitda=fig(10))
    flags = v.check_bank_rules(r, is_bank_in_config=True)
    assert types(flags) == [v.BANK_EBITDA] and r["metrics"]["ebitda"] is None
    r2 = result()
    r2["is_bank"] = True
    assert types(v.check_bank_rules(r2, is_bank_in_config=False)) == [v.BANK_CLASS]


def test_currency_mix_across_rows():
    assert types(v.check_currency_mix([{"currency": "USD"}, {"currency": "BRL"}])) == [v.CURRENCY_MIX]
    assert v.check_currency_mix([{"currency": "USD"}, {"currency": "USD"}]) == []


def test_definitions_differ():
    a, b = result(net_income=fig(1)), result(net_income=fig(2))
    a["net_income_definition"], b["net_income_definition"] = "adjusted", "recurring_managerial"
    b["file"] = "other_3q26_brl"
    assert types(v.check_definitions([a, b])) == [v.DEFINITIONS]
    b["net_income_definition"] = "adjusted"
    assert v.check_definitions([a, b]) == []


def test_pair_margin_difference_across_currencies():
    usd = result(net_revenue=fig(100), ebitda=fig(40))
    brl = result(net_revenue=fig(500, currency="BRL"), ebitda=fig(100, currency="BRL"))
    brl["currency"] = "BRL"
    assert types(v.check_pairs([usd, brl], 2.0)) == [v.PAIR_DIFF]   # 40% vs 20%
    brl["metrics"]["ebitda"] = fig(201, currency="BRL")
    assert v.check_pairs([usd, brl], 2.0) == []                     # 40% vs 40.2%


def test_validate_company_runs_all_checks_and_sets_kind():
    parsed = parse_filename("jpmorgan_3q26_usd.pdf")
    r = result(ebitda=fig(10, page=1, quote="ebitda 10"))
    r.update(file="jpmorgan_3q26_usd", pdf_company="JPMorgan Chase & Co.")
    flags = v.validate_company(r, parsed, ["EBITDA 10"], CFG)
    assert r["kind"] == "bank" and r["metrics"]["ebitda"] is None
    assert v.BANK_EBITDA in types(flags)


# ---------- tolerant quote check ----------

def test_quote_passes_with_scrambled_order_case_accents_and_punctuation():
    page = "LUCRO LÍQUIDO recorrente gerencial R$ 11.482 milhões | ROE 23,4% | Itaú Unibanco"
    quote = "itau unibanco: recorrente gerencial lucro liquido, R$ 11,482 milhoes"
    ok, _ = v.quote_match(quote, page)
    assert ok


def test_quote_passes_when_most_words_present_and_all_numbers_present():
    page = "net income 10 million capex 5 million guidance revenue growth outlook"
    quote = "Net income was 10 million in the quarter"   # 'was', 'in', 'the', 'quarter' missing: 4 of 7 words found? -> below 80%
    ok, reason = v.quote_match(quote, page)
    assert not ok and "words" in reason
    quote2 = "Net income 10 million capex 5 million outlook for revenue"   # 'for' missing: 7 of 8 = 88%
    assert v.quote_match(quote2, page)[0]


def test_quote_fails_when_a_number_is_not_on_the_page():
    page = "net income 10 million and capex 5 million"
    ok, reason = v.quote_match("net income 11 million and capex 5 million", page)
    assert not ok and "11" in reason


def test_quote_numbers_ignore_decimal_and_thousand_separators():
    assert v.quote_match("revenue 1,234.5 million", "revenue 1.234,5 million")[0]


def test_check_quotes_flags_only_real_problems():
    pages = ["Slide 1", "ROE 23,4% recorrente gerencial Itaú lucro R$ 11.482 milhões"]
    good = result(net_income=fig(11482, page=2, quote="lucro recorrente gerencial R$ 11.482 milhões ROE"))
    wrong_number = result(net_income=fig(99999, page=2, quote="lucro recorrente gerencial R$ 99.999 milhões"))
    assert v.check_quotes(good, pages) == []
    assert types(v.check_quotes(wrong_number, pages)) == [v.QUOTE_NOT_FOUND]


# ---------- comparing values across units ----------

def test_same_value_normalizes_units():
    assert v.same_value(fig(10.3, "billions", "BRL"), fig(10312, "millions", "BRL"))
    assert v.same_value(fig(5_000_000, "thousands", "BRL"), fig(5, "billions", "BRL"))
    assert not v.same_value(fig(10.3, "billions", "BRL"), fig(9000, "millions", "BRL"))


def test_same_value_never_equates_currencies_or_unit_kinds():
    assert not v.same_value(fig(10, "billions", "BRL"), fig(10, "billions", "USD"))
    assert not v.same_value(fig(10, "percent", None), fig(10, "millions", "BRL"))
    assert v.same_value(fig(23.4, "percent", None), fig(23.4, "percent", None))


# ---------- net debt / EBITDA is only compared like for like ----------

def _lev(eb_basis, ratio_basis, ratio_value):
    return result(net_debt=fig(300, basis="point_in_time"), ebitda=fig(100, basis=eb_basis),
                  net_debt_ebitda=fig(ratio_value, "times", None, ratio_basis))


def test_leverage_same_basis_is_compared():
    assert types(v.check_leverage(_lev("last_12_months", "last_12_months", 1.5), 0.1, 0.1)) == [v.RATIO_MISMATCH]
    assert v.check_leverage(_lev("last_12_months", "last_12_months", 3.0), 0.1, 0.1) == []


def test_leverage_different_or_unknown_basis_is_not_checkable_never_mismatch():
    for eb, ratio in [("quarter", "last_12_months"), ("other", "last_12_months"),
                      ("last_12_months", "other"), ("other", "other"), ("point_in_time", "point_in_time")]:
        assert types(v.check_leverage(_lev(eb, ratio, 1.5), 0.1, 0.1)) == [v.RATIO_NOT_CHECKABLE]


def test_cet1_without_cet1_evidence_is_dropped():
    r = result(cet1=fig(16.3, unit="percent", currency="none", basis="point_in_time",
                               label="Capital ratio", definition="capital ratio", quote="16,3%"))
    assert v.CET1_DEFINITION in types(v.check_cet1(r)) and r["metrics"]["cet1"] is None


def test_cet1_with_cet1_evidence_is_kept():
    r = result(cet1=fig(11.6, unit="percent", currency="none", basis="point_in_time",
                               label="CET1 ratio", definition="Common Equity Tier 1", quote="CET1 11,6%"))
    assert v.check_cet1(r) == [] and r["metrics"]["cet1"] is not None


def test_adjusted_leverage_vs_plain_net_debt_is_not_a_false_mismatch():
    r = result(
               net_debt=fig(15856, label="Net debt", definition="total debt less cash", basis="last_12_months"),
               ebitda=fig(3728, basis="last_12_months"),
               net_debt_ebitda=fig(1.9, unit="times", currency="none", basis="last_12_months",
                                   label="Adjusted Leverage Ratio", definition="adjusted net debt / adjusted EBITDA"))
    assert types(v.check_leverage(r, 0.10, 0.10)) == [v.RATIO_NOT_CHECKABLE]
