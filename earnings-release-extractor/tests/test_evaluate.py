import pandas as pd

from src import evaluate as ev


def row(value, unit="millions", also=""):
    return {"value": value, "unit": unit, "also_accept": also}


def test_correct_after_unit_normalisation():
    assert ev.judge(row(1.5, "billions"), 1500, "millions") == "correct"


def test_accepted_alternative_and_wrong():
    assert ev.judge(row(5294, also="5122 millions"), 5122, "millions") == "accepted"
    assert ev.judge(row(5294), 5000, "millions") == "wrong"


def test_missing_null_ok_and_invented():
    assert ev.judge(row(100), float("nan"), "") == "missing"
    assert ev.judge(row(float("nan")), float("nan"), "") == "null_ok"
    assert ev.judge(row(float("nan")), 7, "millions") == "invented"


def test_golden_file_is_well_formed():
    g = pd.read_csv("golden/golden.csv")
    assert {"file", "metric", "value", "unit", "page", "quote"} <= set(g.columns)
    assert g["page"].gt(0).all()
