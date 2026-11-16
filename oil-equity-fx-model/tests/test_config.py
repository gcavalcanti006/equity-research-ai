"""Checks the real config file without internet: every name used must exist."""
from collections import Counter
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[1] / "config_multifactor.yaml"
CFG = yaml.safe_load(open(PATH))          # raw file (load_config would add the stocks to 'yahoo')


def _known_factors():
    names = set(CFG["yahoo"]) | set(CFG["yahoo_candidates"]) | set(CFG["fred"]) | set(CFG["bcb"])
    names |= {"us_slope", "br_slope", "br2y", "br10y"}
    return names


def test_every_stock_has_a_valid_country_and_sector():
    for name, s in CFG["stocks"].items():
        assert s["country"] in CFG["market"], name
        assert s["sector"] in CFG["sectors"], name


def test_five_stocks_per_sector_and_country():
    counts = Counter((s["country"], s["sector"]) for s in CFG["stocks"].values())
    assert set(counts.values()) == {5} and len(counts) == 6


def test_no_stock_name_clashes_with_a_factor_name():
    assert not (set(CFG["stocks"]) & _known_factors())
    assert len({s["ticker"] for s in CFG["stocks"].values()}) == len(CFG["stocks"])      # no duplicate tickers


def test_factors_and_scenarios_use_known_names():
    known = _known_factors()
    used = set(CFG["market"].values())
    for lst in list(CFG["common_factors"].values()) + list(CFG["sectors"].values()) + [CFG["lagged_factors"]]:
        used |= set(lst)
    assert used <= known, used - known
    for sc in CFG["scenarios"].values():
        assert set(sc["shocks"]) <= known
    assert set(CFG["factor_fallbacks"]) <= known and set(CFG["factor_fallbacks"].values()) <= known


def test_episodes_have_start_before_end():
    for e in CFG["episodes"].values():
        assert e["start"] < e["end"]
