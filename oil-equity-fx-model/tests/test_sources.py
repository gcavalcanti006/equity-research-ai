import pandas as pd
import pytest

from src import sources


class FakeResp:
    def __init__(self, text="", js=None):
        self.text, self._js = text, js

    def raise_for_status(self):
        pass

    def json(self):
        return self._js


def test_fred_parses_csv_and_drops_missing_days(monkeypatch):
    csv = "observation_date,DGS10\n2024-01-02,4.00\n2024-01-03,.\n2024-01-04,4.10\n"
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResp(text=csv))
    s = sources.fetch_fred("DGS10", "2024-01-01", "2024-01-31")
    assert list(s.values) == [4.00, 4.10] and s.index[0] == pd.Timestamp("2024-01-02")


def test_bcb_parses_brazilian_dates_and_ignores_future_rows(monkeypatch):
    rows = [{"data": "02/01/2024", "valor": "11.75"}, {"data": "03/01/2024", "valor": "11.75"},
            {"data": "01/01/2099", "valor": "1.00"}]
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResp(js=rows))
    s = sources.fetch_bcb(432, "2024-01-01", "2099-12-31")
    assert len(s) == 2 and s.index[-1] == pd.Timestamp("2024-01-03")


def test_load_all_uses_snapshot_and_survives_a_failed_series(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_yahoo(ticker, start, end):
        calls["n"] += 1
        if ticker == "BAD":
            raise RuntimeError("no data")
        return pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    monkeypatch.setattr(sources, "fetch_yahoo", fake_yahoo)
    cfg = {"snapshot_dir": str(tmp_path / "snap"), "yahoo": {"good": "OK", "bad": "BAD"}}
    levels, problems = sources.load_all(cfg, "2024-01-01", "2024-02-01")
    assert list(levels.columns) == ["good"] and len(problems) == 1 and "bad" in problems[0]
    n_first = calls["n"]
    levels2, _ = sources.load_all(cfg, "2024-01-01", "2024-02-01")          # second run: snapshot
    assert (levels2["good"] == levels["good"]).all()
    assert calls["n"] == n_first + 1                                         # only 'bad' is retried


def test_manual_br_yields_csv(tmp_path):
    p = tmp_path / "br.csv"
    p.write_text("date,br2y,br10y\n2024-01-02,10.5,11.0\n2024-01-03,10.6,11.1\n")
    out = sources.load_manual_csv(p, "2024-01-01", "2024-12-31")
    assert set(out) == {"br2y", "br10y"} and out["br10y"].iloc[-1] == 11.1
