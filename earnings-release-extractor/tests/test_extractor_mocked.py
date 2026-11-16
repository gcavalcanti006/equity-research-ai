"""Extractor tests use a fake client: no API key, no network."""
import os

from src import extractor, report
from src.validator import Flag
from tests.helpers import FakeClient, empty_raw, raw_fig

CFG = {"model": "claude-sonnet-5-5", "effort": "medium", "max_tokens": 1000,
       "chunk_pages": 2, "max_chars_per_chunk": 10_000}
PAGES = ["Acme Corp 3Q26. Net revenue was 100", "EBITDA 40", "Net income 10", "Capex 5"]


def answer(**kw):
    raw = empty_raw()
    raw.update(company="Acme Corp", period="3Q26", currency="USD")
    raw.update(kw)
    return raw


def test_test_environment_has_no_key_needed():
    # tests must pass without the real key being present
    assert FakeClient([empty_raw()]).calls == []


def test_found_false_becomes_none_and_found_true_keeps_source():
    raw = answer(figures=[raw_fig("net_revenue", 100, page=1, quote="Net revenue was 100")])
    client = FakeClient([raw])
    result, flags = extractor.extract_document(client, PAGES, CFG, "acme_3q26_usd")
    assert flags == []
    assert result["metrics"]["net_revenue"]["value"] == 100
    assert result["metrics"]["net_revenue"]["page"] == 1
    assert result["metrics"]["ebitda"] is None          # never guessed
    assert result["pdf_company"] == "Acme Corp"
    assert len(client.calls) == 2                       # 4 pages / 2 per chunk


def test_request_uses_config_model_and_strict_schema():
    client = FakeClient([answer()])
    extractor.extract_document(client, PAGES, CFG, "acme_3q26_usd")
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert "=== PAGE 1 ===" in call["messages"][0]["content"]


def test_conflicting_values_across_chunks_are_flagged():
    a = answer(figures=[raw_fig("ebitda", 40, page=2, quote="EBITDA 40")])
    b = answer(figures=[raw_fig("ebitda", 55, page=4, quote="EBITDA 55")])
    result, flags = extractor.extract_document(FakeClient([a, b]), PAGES, CFG, "acme_3q26_usd")
    assert result["metrics"]["ebitda"]["value"] == 40
    assert [f.flag_type for f in flags] == ["CONFLICTING_VALUES_ACROSS_CHUNKS"]


def test_refusal_becomes_a_flag_not_a_crash():
    client = FakeClient([answer()], stop_reason="refusal")
    result, flags = extractor.extract_document(client, PAGES, CFG, "acme_3q26_usd")
    assert result is None
    assert {f.flag_type for f in flags} == {"EXTRACTION_FAILED"}


def test_cache_avoids_second_api_call(tmp_path):
    client = FakeClient([answer()])
    extractor.extract_document(client, PAGES, CFG, "acme_3q26_usd", tmp_path)
    first = len(client.calls)
    extractor.extract_document(client, PAGES, CFG, "acme_3q26_usd", tmp_path)
    assert len(client.calls) == first


def test_every_flag_is_printed(capsys):
    flags = [Flag("acme_3q26_usd", "ebitda", "EBITDA_GREATER_THAN_REVENUE", "msg one", 3, "the quote"),
             Flag("(all)", "currency", "MIXED_CURRENCIES", "msg two")]
    report.print_flags(flags)
    out = capsys.readouterr().out
    assert "FLAGS (2)" in out and "msg one" in out and "p.3" in out and "the quote" in out
    assert "msg two" in out


def test_no_flags_message(capsys):
    report.print_flags([])
    assert "No flags raised" in capsys.readouterr().out


def test_end_to_end_with_fake_client_writes_files(tmp_path, monkeypatch):
    """Whole pipeline on fake data in a temp folder (never touches the real outputs)."""
    import src.main as main_mod
    from src import pdf_reader

    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "acme_3q26_usd.pdf").write_bytes(b"not a real pdf")
    monkeypatch.setattr(pdf_reader, "read_pages", lambda p: PAGES + ["x" * 300])
    raw = answer(figures=[raw_fig("net_revenue", 100, page=1, quote="Net revenue was 100"),
                         raw_fig("ebitda", 150, page=2, quote="EBITDA 40")])     # EBITDA > revenue, wrong quote
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""
model: claude-sonnet-5-5
effort: medium
max_tokens: 1000
api_key_env: ANTHROPIC_API_KEY
chunk_pages: 12
max_chars_per_chunk: 60000
min_text_chars: 10
banks: []
validation: {{ratio_rel_tol: 0.1, ratio_abs_tol: 0.1, pair_margin_tol_pp: 2.0}}
paths: {{pdf_dir: "{pdf_dir}", outputs: "{tmp_path / 'out'}", cache: "{tmp_path / 'cache'}"}}
""")
    code = main_mod.main(["--config", str(cfg)], client=FakeClient([raw]))
    assert code == 0
    for name in ("results.csv", "results.xlsx", "flags.csv"):
        assert os.path.exists(tmp_path / "out" / name)
    flags_csv = (tmp_path / "out" / "flags.csv").read_text()
    assert "EBITDA_GREATER_THAN_REVENUE" in flags_csv
    assert "QUOTE_NOT_FOUND_ON_PAGE" not in flags_csv        # the quote really is on page 2
    import pandas as pd
    sheets = pd.ExcelFile(tmp_path / "out" / "results.xlsx").sheet_names
    assert sheets == ["Non-financial", "Banks", "Sources", "Flags"]


# ---------- failures and empty results ----------

def test_empty_results_write_headers_only_and_say_so(tmp_path, capsys):
    """If every chunk fails, results is empty: files must still be written, with the right columns."""
    import pandas as pd
    flags = [Flag("acme_3q26_usd", "pages 1-12", "EXTRACTION_FAILED", "BadRequestError: boom")]
    df = report.write_outputs([], flags, tmp_path)
    assert df.empty and "kind" in df.columns and "net_revenue" in df.columns
    csv = pd.read_csv(tmp_path / "results.csv")
    assert len(csv) == 0 and "kind" in csv.columns
    sheets = pd.ExcelFile(tmp_path / "results.xlsx").sheet_names
    assert sheets == ["Non-financial", "Banks", "Sources", "Flags"]
    assert "boom" in (tmp_path / "flags.csv").read_text()
    report.print_summary(df)
    assert "NOTHING WAS EXTRACTED" in capsys.readouterr().out


def test_failed_chunk_prints_and_saves_the_real_reason(capsys):
    import anthropic
    import httpx2 as httpx   # the anthropic SDK is built on httpx2

    class Failing:
        def __init__(self):
            self.messages = type("M", (), {"create": self.create})()

        def create(self, **kw):
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(400, request=req, json={})
            raise anthropic.BadRequestError("compiled grammar is too large", response=resp, body=None)

    result, flags = extractor.extract_document(Failing(), PAGES, CFG, "acme_3q26_usd")
    assert result is None and len(flags) == 2
    assert "BadRequestError" in flags[0].message and "compiled grammar is too large" in flags[0].message
    out = capsys.readouterr().out
    assert "FAILED" in out and "compiled grammar is too large" in out


def test_main_exit_code_and_message_when_nothing_extracted(tmp_path, monkeypatch, capsys):
    import src.main as main_mod
    from src import pdf_reader
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "acme_3q26_usd.pdf").write_bytes(b"x")
    monkeypatch.setattr(pdf_reader, "read_pages", lambda p: PAGES + ["x" * 300])
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"""
model: m
effort: medium
max_tokens: 10
api_key_env: ANTHROPIC_API_KEY
chunk_pages: 12
max_chars_per_chunk: 60000
min_text_chars: 10
banks: []
validation: {{ratio_rel_tol: 0.1, ratio_abs_tol: 0.1, pair_margin_tol_pp: 2.0}}
paths: {{pdf_dir: "{pdf_dir}", outputs: "{tmp_path / 'out'}", cache: "{tmp_path / 'cache'}"}}
""")
    code = main_mod.main(["--config", str(cfg)], client=FakeClient([answer()], stop_reason="refusal"))
    assert code == 2
    assert (tmp_path / "out" / "results.csv").exists()
    assert "NOTHING WAS EXTRACTED" in capsys.readouterr().out


def test_error_text_never_contains_an_api_key():
    secret = "sk-ant-api03-ABCDEF1234567890abcdef"
    text = extractor.describe_error(RuntimeError(f"bad header x-api-key: {secret}"))
    assert secret not in text and "[redacted]" in text


def test_schema_is_small_and_strict():
    """Guards against the 'compiled grammar is too large' error: keep the schema compact."""
    import json
    from src.schema import CHUNK_SCHEMA
    assert len(json.dumps(CHUNK_SCHEMA)) < 3000
    assert CHUNK_SCHEMA["additionalProperties"] is False
    assert set(CHUNK_SCHEMA["required"]) == set(CHUNK_SCHEMA["properties"])


def test_same_value_in_different_units_is_not_a_conflict():
    a = answer(figures=[raw_fig("net_income", 11.482, unit="billions", currency="BRL", page=2, quote="x")])
    b = answer(figures=[raw_fig("net_income", 11482, unit="millions", currency="BRL", page=4, quote="y")])
    result, flags = extractor.extract_document(FakeClient([a, b]), PAGES, CFG, "acme_3q26_brl")
    assert flags == []
    assert result["metrics"]["net_income"]["value"] == 11.482     # first one kept


def test_real_difference_is_still_a_conflict():
    a = answer(figures=[raw_fig("net_income", 11.482, unit="billions", currency="BRL", page=2)])
    b = answer(figures=[raw_fig("net_income", 9000, unit="millions", currency="BRL", page=4)])
    _, flags = extractor.extract_document(FakeClient([a, b]), PAGES, CFG, "acme_3q26_brl")
    assert [f.flag_type for f in flags] == ["CONFLICTING_VALUES_ACROSS_CHUNKS"]


def test_portuguese_period_in_pdf_matches_filename():
    from src import validator
    from src.filename_parser import parse_filename
    from tests.helpers import result as make_result
    r = make_result()
    r["pdf_period"] = "3º trimestre de 2026"
    assert validator.check_filename_vs_pdf(parse_filename("acme_3q26_usd.pdf"), r) == []


# ---------- choosing between several candidates ----------

def run_two_chunks(first, second, file="acme_3q26_brl"):
    return extractor.extract_document(
        FakeClient([answer(figures=first), answer(figures=second)]), PAGES, CFG, file)


def test_petrobras_case_reported_consolidated_beats_adjusted_headline():
    """Chunk 1 (page 6) shows adjusted net income 47,877; chunk 2 (page 21) the reported 39,783."""
    adjusted = raw_fig("net_income", 47877, currency="BRL", page=6, definition_type="adjusted",
                       definition="net income excluding one-off items")
    reported = raw_fig("net_income", 39783, currency="BRL", page=21, definition_type="reported",
                       definition="consolidated net income, income statement")
    result, flags = run_two_chunks([adjusted], [reported])
    assert result["metrics"]["net_income"]["value"] == 39783
    assert result["net_income_definition"] == "reported"
    assert flags == []          # different definition types: not a conflict


def test_both_candidates_in_one_chunk_are_ranked_too():
    adjusted = raw_fig("net_income", 47877, currency="BRL", definition_type="adjusted")
    reported = raw_fig("net_income", 39783, currency="BRL", definition_type="reported")
    result, flags = run_two_chunks([adjusted, reported], [])
    assert result["metrics"]["net_income"]["value"] == 39783 and flags == []


def test_btg_case_real_cet1_beats_total_capital_ratio():
    """Chunk 1 files the Basel (total capital) ratio 16.3% as CET1; chunk 2 has CET1 11.6%."""
    basel = raw_fig("cet1", 16.3, unit="percent", currency="none", basis="point_in_time", page=3,
                    label="Índice de Basileia", definition="total capital ratio")
    cet1 = raw_fig("cet1", 11.6, unit="percent", currency="none", basis="point_in_time", page=25,
                   label="Nível 1: CET1 & AT1 (CP - Capital Principal)", definition="CET1")
    result, flags = run_two_chunks([basel], [cet1], "btg-pactual_3q26_brl")
    assert result["metrics"]["cet1"]["value"] == 11.6
    assert flags == []          # the Basel ratio is disqualified, not a conflicting CET1


def test_only_total_capital_ratio_is_dropped_and_flagged_by_validator():
    from src import validator
    from tests.helpers import fig, result as make_result
    f = fig(16.3, "percent", None, "point_in_time")
    f["label"] = "Índice de Basileia"
    r = make_result(cet1=f)
    flags = validator.check_cet1(r)
    assert [x.flag_type for x in flags] == ["CET1_IS_TOTAL_CAPITAL_RATIO"]
    assert r["metrics"]["cet1"] is None
    ok = fig(11.6, "percent", None, "point_in_time")
    ok["label"] = "CET1 (Capital Principal)"
    assert validator.check_cet1(make_result(cet1=ok)) == []


def test_quarter_beats_year_to_date_and_consolidated_beats_segment():
    ytd = raw_fig("net_revenue", 462569, currency="BRL", basis="year_to_date", page=2)
    quarter = raw_fig("net_revenue", 169353, currency="BRL", basis="quarter", page=3)
    segment = raw_fig("net_revenue", 90000, currency="BRL", basis="quarter", scope="segment", page=1)
    result, flags = run_two_chunks([segment, ytd], [quarter])
    assert result["metrics"]["net_revenue"]["value"] == 169353
    assert flags == []          # ytd/segment differ in basis/scope: not conflicts


def test_conflict_only_when_basis_scope_and_definition_match():
    a = raw_fig("net_debt", 100, basis="point_in_time", definition="net debt excl. leases")
    b = raw_fig("net_debt", 130, basis="point_in_time", definition="net debt incl. leases")
    _, flags = run_two_chunks([a], [b])                       # same type/scope/basis, values differ
    assert [f.flag_type for f in flags] == ["CONFLICTING_VALUES_ACROSS_CHUNKS"]
    assert "incl. leases" in flags[0].message and "excl. leases" in flags[0].message
    c = raw_fig("net_debt", 130, basis="point_in_time", definition_type="adjusted")
    _, flags = run_two_chunks([a], [c])                       # different definition type
    assert flags == []


def test_prompt_and_schema_ask_for_basis_scope_and_definition():
    from src.schema import CHUNK_SCHEMA
    item = CHUNK_SCHEMA["properties"]["figures"]["items"]
    for field in ("basis", "scope", "definition_type", "definition"):
        assert field in item["required"]
    assert "CURRENT-QUARTER" in extractor.SYSTEM_PROMPT and "NOT CET1" in extractor.SYSTEM_PROMPT


def test_cet1_standardized_vs_advanced_is_not_a_conflict_and_standardized_is_kept():
    from tests.helpers import fig
    both = "Standardized ratio of 13.9% and Advanced ratio of 14.1%"   # the real quote names both approaches
    std = fig(13.9, unit="percent", currency="none", basis="point_in_time", label="CET1 ratio",
              definition="Basel III CET1, Standardized approach", quote=both)
    adv = fig(14.1, unit="percent", currency="none", basis="point_in_time", label="CET1 ratio",
              definition="Basel III CET1, Advanced approach", quote=both)
    cands = {m: [] for m in extractor.METRICS}
    cands["cet1"] = [adv, std]                      # Advanced appears first in the document
    cleaned = [{"company": "JPM", "period": "3Q26", "currency": "USD", "is_bank": True,
                "candidates": cands, "guidance": None, "tone": None}]
    result, flags = extractor.merge_chunks("jpmorgan_3q26_usd", cleaned)
    assert result["metrics"]["cet1"]["value"] == 13.9
    assert flags == []


def _cleaned(metric, figs):
    cands = {m: [] for m in extractor.METRICS}
    cands[metric] = figs
    return [{"company": "X", "period": "3Q26", "currency": "BRL", "is_bank": False,
             "candidates": cands, "guidance": None, "tone": None}]


def test_brl_file_prefers_the_brl_figure_over_a_usd_table():
    from tests.helpers import fig
    usd = fig(57847, currency="USD", basis="point_in_time", definition="net debt in US$ millions")
    brl = fig(299696, currency="BRL", basis="point_in_time", definition="net debt in R$ millions")
    result, flags = extractor.merge_chunks("petrobras_3q26_brl", _cleaned("net_debt", [usd, brl]))
    assert result["metrics"]["net_debt"]["currency"] == "BRL" and flags == []


def test_net_income_attributable_is_kept_and_total_is_not_a_conflict():
    from tests.helpers import fig
    total = fig(1275, definition="Net income including non-controlling interests")
    attr = fig(1255, definition="Net income attributable to the company's shareholders")
    result, flags = extractor.merge_chunks("vale_3q26_usd", _cleaned("net_income", [total, attr]))
    assert result["metrics"]["net_income"]["value"] == 1255 and flags == []


def test_real_conflict_with_same_definition_is_still_flagged():
    from tests.helpers import fig
    a = fig(100, definition="Net income attributable to shareholders")
    b = fig(130, definition="Net income attributable to shareholders")
    _, flags = extractor.merge_chunks("x_3q26_usd", _cleaned("net_income", [a, b]))
    assert len(flags) == 1


def test_gross_debt_ratio_is_not_a_rival_of_the_net_debt_ratio():
    from tests.helpers import fig
    net = fig(0.8, unit="times", currency="none", basis="last_12_months", definition="Net debt / adjusted LTM EBITDA")
    gross = fig(1.2, unit="times", currency="none", basis="last_12_months",
                definition="Total debt and leases / LTM Adjusted EBITDA (gross debt basis)")
    result, flags = extractor.merge_chunks("vale_3q26_usd", _cleaned("net_debt_ebitda", [gross, net]))
    assert result["metrics"]["net_debt_ebitda"]["value"] == 0.8 and flags == []
