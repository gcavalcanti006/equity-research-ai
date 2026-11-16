"""Score the extractor against the golden set.

Usage:  python -m src.evaluate            (after python -m src.main)

golden/golden.csv holds values read straight from the PDFs, each with the exact quote and page. It was
built by reading the PDF text (not by the extractor) and every quote is checked to be on its page.
It is a small hand-built set (49 values in 10 releases): good for spotting regressions, NOT a guarantee
of overall accuracy. Add rows whenever you check a new value.

A result counts as:
  correct   - same value as the golden one (1% tolerance, units normalised)
  accepted  - matches a documented alternative in `also_accept` (e.g. reported vs adjusted net income)
  wrong     - a value that matches neither
  missing   - the extractor returned nothing, but the release does state it
  null_ok   - golden says the release has no such number and the extractor also returned nothing
  invented  - golden says the release has no such number, but the extractor returned one
"""
import sys
from pathlib import Path

import pandas as pd

SCALE = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}


def to_plain(value, unit):
    return value * SCALE.get(unit, 1.0)


def close(a, b, tol=0.01):
    return abs(a - b) <= tol * max(abs(a), abs(b), 1e-12)


def parse_alts(text):
    """'5122 millions;14.1 percent' -> [(5122.0,'millions'), (14.1,'percent')]"""
    out = []
    for part in str(text).split(";"):
        bits = part.split()
        if len(bits) == 2:
            out.append((float(bits[0]), bits[1]))
    return out


def judge(row, got_value, got_unit):
    if pd.isna(row["value"]):
        return "null_ok" if pd.isna(got_value) else "invented"
    if pd.isna(got_value):
        return "missing"
    got = to_plain(got_value, got_unit)
    if close(got, to_plain(row["value"], row["unit"])):
        return "correct"
    if any(close(got, to_plain(v, u)) for v, u in parse_alts(row.get("also_accept", "") if not pd.isna(row.get("also_accept")) else "")):
        return "accepted"
    return "wrong"


def evaluate(golden_csv="golden/golden.csv", results_csv="outputs/results.csv"):
    gold = pd.read_csv(golden_csv)
    res = pd.read_csv(results_csv).set_index("file")
    rows = []
    for _, r in gold.iterrows():
        if r["file"] not in res.index:
            rows.append({**r, "got": None, "got_unit": None, "verdict": "file_not_run"})
            continue
        x = res.loc[r["file"]]
        v, u = x.get(r["metric"]), x.get(f"{r['metric']}_unit")
        rows.append({**r, "got": v, "got_unit": u, "verdict": judge(r, v, u if isinstance(u, str) else "")})
    return pd.DataFrame(rows)


def main():
    df = evaluate()
    ran = df[df["verdict"] != "file_not_run"]
    counts = ran["verdict"].value_counts()
    good = counts.get("correct", 0) + counts.get("accepted", 0) + counts.get("null_ok", 0)
    print(f"Golden set: {len(ran)} values checked in {ran['file'].nunique()} files")
    print(f"  right (correct + accepted alternative + correctly empty): {good} of {len(ran)} = {good / len(ran):.0%}")
    print(f"  exactly the golden value: {counts.get('correct', 0)}   accepted alternative: {counts.get('accepted', 0)}   "
          f"correctly empty: {counts.get('null_ok', 0)}")
    print(f"  wrong: {counts.get('wrong', 0)}   missing: {counts.get('missing', 0)}   invented: {counts.get('invented', 0)}")
    print("\nPer file (right / checked):")
    for f, g in ran.groupby("file"):
        ok = g["verdict"].isin(["correct", "accepted", "null_ok"]).sum()
        print(f"  {f:28s} {ok}/{len(g)}")
    bad = ran[ran["verdict"].isin(["wrong", "missing", "invented"])]
    if len(bad):
        print("\nTo look at:")
        for _, r in bad.iterrows():
            print(f"  [{r['verdict']}] {r['file']} {r['metric']}: golden {r['value']} {r['unit']} (p.{r['page']}) "
                  f"vs extractor {r['got']} {r['got_unit']}")
    alt = ran[ran["verdict"] == "accepted"]
    if len(alt):
        print("\nAccepted as an alternative definition (check this is the one you want):")
        for _, r in alt.iterrows():
            print(f"  {r['file']} {r['metric']}: extractor {r['got']} {r['got_unit']}  | golden {r['value']}  | {r['note']}")


if __name__ == "__main__":
    sys.exit(main())
