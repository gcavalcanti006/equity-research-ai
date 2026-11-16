"""Download every raw series (Yahoo prices, FRED yields, Brazilian central bank) and keep a
snapshot on disk, so a run is reproducible: the same files give the same numbers.

Rules:
- A series that fails to download is skipped with a clear message (we never invent data).
- Snapshots live in data/snapshots/<name>.csv. Delete a file (or run with --refresh) to re-download.
- Everything returned is a LEVEL series (a price or a yield), indexed by calendar date.
"""
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
BCB_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"


def _clean_index(series):
    """Plain calendar dates, no time zone, sorted, no duplicates, no future dates."""
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = series.index.normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    series = series[series.index <= pd.Timestamp(date.today())]
    return series.dropna()


def fetch_yahoo(ticker, start, end):
    import yfinance as yf     # imported here so tests do not need yfinance
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return _clean_index(close)


def fetch_fred(series_id, start, end):
    """FRED publishes every series as a CSV with no key needed. Missing days are '.'."""
    resp = requests.get(FRED_URL, params={"id": series_id}, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), na_values=["."])
    col = df.columns[1]
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0]))
    s = _clean_index(s)
    return s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]


def fetch_bcb(code, start, end):
    """Banco Central do Brasil, SGS system. Returns JSON like [{"data": "07/12/2026", "valor": "13.75"}]."""
    params = {"formato": "json",
              "dataInicial": pd.Timestamp(start).strftime("%d/%m/%Y"),
              "dataFinal": pd.Timestamp(end).strftime("%d/%m/%Y")}
    resp = requests.get(BCB_URL.format(code=code), params=params, timeout=60)
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise RuntimeError(f"BCB series {code} returned no rows")
    s = pd.Series([float(r["valor"]) for r in rows],
                  index=pd.to_datetime([r["data"] for r in rows], format="%d/%m/%Y"))
    return _clean_index(s)


def load_manual_csv(path, start, end):
    """Optional file with Brazilian yields you downloaded yourself.
    Needs a 'date' column (YYYY-MM-DD) and one column per series (e.g. br2y, br10y), in % a.a."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    out = {}
    for col in df.columns:
        s = _clean_index(pd.to_numeric(df[col], errors="coerce"))
        out[col] = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    return out


def _snapshot_path(snapshot_dir, name):
    return Path(snapshot_dir) / f"{name}.csv"


def _read_snapshot(path):
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.name = Path(path).stem
    return s


def load_all(cfg, start, end, refresh=False):
    """Return (levels DataFrame, list of problems). One column per series name in the config."""
    snap = Path(cfg["snapshot_dir"])
    snap.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for name, ticker in cfg.get("yahoo", {}).items():
        jobs[name] = lambda t=ticker: fetch_yahoo(t, start, end)
    for name, sid in cfg.get("fred", {}).items():
        jobs[name] = lambda s=sid: fetch_fred(s, start, end)
    for name, code in cfg.get("bcb", {}).items():
        jobs[name] = lambda c=code: fetch_bcb(c, start, end)

    series, problems = {}, []
    for name, job in jobs.items():
        path = _snapshot_path(snap, name)
        try:
            if path.exists() and not refresh:
                s = _read_snapshot(path)
                source = "snapshot"
            else:
                s = job()
                s.rename(name).to_frame().to_csv(path)
                source = "downloaded"
            series[name] = s
            print(f"  {name:10s} {source:10s} {len(s):5d} rows  {s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as exc:                     # one bad series must not stop the run
            problems.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  {name:10s} FAILED      {type(exc).__name__}: {exc}")

    manual = cfg.get("br_yields_csv")
    if manual:
        try:
            for name, s in load_manual_csv(manual, start, end).items():
                series[name] = s
                print(f"  {name:10s} manual csv  {len(s):5d} rows")
        except Exception as exc:
            problems.append(f"br_yields_csv: {type(exc).__name__}: {exc}")
    return pd.DataFrame(series).sort_index(), problems
