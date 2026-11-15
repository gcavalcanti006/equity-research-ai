"""Build the output tables (CSV, Excel), the terminal summary and the flag list."""
from pathlib import Path

import pandas as pd

from src.schema import METRICS
from src.validator import ebitda_margin, net_margin

COMMON_METRICS = ["net_income", "dividends", "roe"]
NON_BANK_METRICS = ["net_revenue", "ebitda", "net_income", "net_debt", "net_debt_ebitda",
                    "capex", "dividends", "roe"]
BANK_METRICS = ["net_income", "roe", "loan_portfolio", "efficiency_ratio", "cet1", "dividends"]


def row_columns():
    """All columns of results.csv, in order (used even when nothing was extracted)."""
    cols = ["file", "company", "period_in_pdf", "currency", "kind", "net_income_definition"]
    for m in METRICS:
        cols += [m, f"{m}_unit", f"{m}_currency", f"{m}_basis", f"{m}_scope", f"{m}_note",
                 f"{m}_label", f"{m}_page", f"{m}_source"]
    return cols + ["ebitda_margin_pct", "net_margin_pct", "guidance", "guidance_page",
                   "guidance_source", "tone", "tone_page", "tone_source"]


SOURCE_COLUMNS = ["file", "metric", "value", "unit", "currency", "basis", "scope", "definition_type",
                  "definition", "label", "page", "quote"]


def result_to_row(r):
    """One wide row per PDF. Each metric has value, unit, currency, basis, page and source columns."""
    row = {"file": r["file"], "company": r["company"], "period_in_pdf": r["pdf_period"],
           "currency": r["currency"], "kind": r["kind"],
           "net_income_definition": r["net_income_definition"]}
    for m in METRICS:
        fig = r["metrics"].get(m)
        row[m] = fig["value"] if fig else None
        row[f"{m}_unit"] = fig["unit"] if fig else None
        row[f"{m}_currency"] = fig["currency"] if fig else None
        row[f"{m}_basis"] = fig["basis"] if fig else None
        row[f"{m}_scope"] = fig["scope"] if fig else None
        row[f"{m}_note"] = fig["definition"] if fig else None
        row[f"{m}_label"] = fig["label"] if fig else None
        row[f"{m}_page"] = fig["page"] if fig else None
        row[f"{m}_source"] = fig["quote"] if fig else None
    # Ratios computed by the code (percent of revenue), so currencies never mix.
    row["ebitda_margin_pct"] = ebitda_margin(r)
    row["net_margin_pct"] = net_margin(r)
    g, t = r.get("guidance"), r.get("tone")
    row["guidance"], row["guidance_page"] = (g["text"], g["page"]) if g else (None, None)
    row["guidance_source"] = g["quote"] if g else None
    row["tone"], row["tone_page"] = (t["text"], t["page"]) if t else (None, None)
    row["tone_source"] = t["quote"] if t else None
    return row


def _columns(metrics, extra):
    cols = ["file", "company", "period_in_pdf", "currency", "net_income_definition"]
    for m in metrics:
        cols += [m, f"{m}_unit", f"{m}_currency", f"{m}_basis", f"{m}_note",
                 f"{m}_page", f"{m}_source"]
    return cols + extra + ["guidance", "guidance_page", "tone", "tone_page", "tone_source"]


def sources_table(results):
    """Long format: one line per number with its source. Useful for checking against the PDF."""
    rows = []
    for r in results:
        for m in METRICS:
            fig = r["metrics"].get(m)
            if fig:
                rows.append({"file": r["file"], "metric": m, "value": fig["value"], "unit": fig["unit"],
                             "currency": fig["currency"], "basis": fig["basis"], "scope": fig["scope"],
                             "definition_type": fig["definition_type"],
                             "definition": fig["definition"], "label": fig["label"],
                             "page": fig["page"], "quote": fig["quote"]})
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


def write_outputs(results, flags, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([result_to_row(r) for r in results], columns=row_columns())
    flags_df = pd.DataFrame([f.as_dict() for f in flags],
                            columns=["company", "field", "flag_type", "message", "page", "quote"])
    flags_df = flags_df.rename(columns={"company": "file"})
    df.to_csv(out / "results.csv", index=False)
    flags_df.to_csv(out / "flags.csv", index=False)
    non_bank = df[df["kind"] == "non_financial"]
    banks = df[df["kind"] == "bank"]
    with pd.ExcelWriter(out / "results.xlsx") as xl:
        non_bank[_columns(NON_BANK_METRICS, ["ebitda_margin_pct", "net_margin_pct"])].to_excel(
            xl, sheet_name="Non-financial", index=False)
        banks[_columns(BANK_METRICS, [])].to_excel(xl, sheet_name="Banks", index=False)
        sources_table(results).to_excel(xl, sheet_name="Sources", index=False)
        flags_df.to_excel(xl, sheet_name="Flags", index=False)
    return df


def _fmt(v, suffix=""):
    return "n/a" if v is None or pd.isna(v) else f"{v:,.1f}{suffix}"


def print_summary(df):
    """Compare companies on ratios only. We never put absolute values of different currencies side by side."""
    print("\n=== COMPARISON (ratios only; no currency conversion) ===")
    if df.empty:
        print("NOTHING WAS EXTRACTED: no document produced results. See the flags below for the reason.")
        return
    nb = df[df["kind"] == "non_financial"].sort_values(["company", "currency"])
    if len(nb):
        t = pd.DataFrame({
            "file": nb["file"], "ccy": nb["currency"],
            "EBITDA margin": [_fmt(v, "%") for v in nb["ebitda_margin_pct"]],
            "net margin": [_fmt(v, "%") for v in nb["net_margin_pct"]],
            "ND/EBITDA": [_fmt(v, "x") for v in nb["net_debt_ebitda"]],
            "ROE": [_fmt(v, "%") for v in nb["roe"]],
            "NI definition": nb["net_income_definition"], "tone": nb["tone"]})
        print("\nNon-financial companies:\n" + t.to_string(index=False))
    bk = df[df["kind"] == "bank"].sort_values("company")
    if len(bk):
        t = pd.DataFrame({
            "file": bk["file"], "ccy": bk["currency"],
            "ROE": [_fmt(v, "%") for v in bk["roe"]],
            "efficiency": [_fmt(v, "%") for v in bk["efficiency_ratio"]],
            "CET1": [_fmt(v, "%") for v in bk["cet1"]],
            "NI definition": bk["net_income_definition"], "tone": bk["tone"]})
        print("\nBanks:\n" + t.to_string(index=False))


def print_flags(flags):
    """Print every flag so you can check the numbers against the PDF yourself."""
    print(f"\n=== FLAGS ({len(flags)}) ===")
    if not flags:
        print("No flags raised")
        return
    for f in flags:
        page = f" | p.{f.page}" if f.page else ""
        quote = f' "{f.quote[:120]}"' if f.quote else ""
        print(f"- {f.company} | {f.field} | {f.flag_type} | {f.message}{page}{quote}")
