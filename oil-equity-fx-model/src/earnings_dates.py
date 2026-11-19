"""Earnings-release dates per stock: from Yahoo when it has them, plus an optional file you fill in by hand.

- Yahoo (yfinance `get_earnings_dates`) usually has US stocks and often has nothing, or only a few rows, for Brazilian ones.
- Whatever Yahoo returns is saved in data/earnings_dates/<stock>.csv (a snapshot: the next runs reuse it).
- `earnings_manual_csv` (config) is a file YOU can write: columns `stock,date` (YYYY-MM-DD, one row per release;
  use the date of the release, the day the company published, even if it was after the close). It is MERGED with Yahoo's dates.
- Future dates and dates without a reported result are dropped: only past releases are events.
Columns of the result: date (calendar date, no time), and for US stocks when Yahoo gives them: eps_estimate, eps_reported, surprise_pct.
"""
from pathlib import Path

import pandas as pd


def fetch_yahoo_dates(ticker, pages=3, page_size=100):
    import yfinance as yf          # imported here so tests do not need yfinance
    frames = []
    t = yf.Ticker(ticker)
    for k in range(pages):
        df = t.get_earnings_dates(limit=page_size, offset=k * page_size)
        if df is None or df.empty:
            break
        frames.append(df)
        if len(df) < page_size:
            break
    if not frames:
        raise RuntimeError(f"Yahoo returned no earnings dates for {ticker}")
    return normalize_yahoo(pd.concat(frames))


def normalize_yahoo(raw, today=None):
    """Yahoo table (index = timestamp of the release) -> tidy table, past releases that were actually reported."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    idx = pd.DatetimeIndex(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)                     # keep the local calendar date (what the exchange calls the day)
    out = pd.DataFrame({"date": idx.normalize()})
    for src, dst in (("EPS Estimate", "eps_estimate"), ("Reported EPS", "eps_reported"), ("Surprise(%)", "surprise_pct")):
        out[dst] = pd.to_numeric(raw[src].values, errors="coerce") if src in raw else float("nan")
    out = out[out["date"] < today]
    if out["eps_reported"].notna().any():               # a past row with no reported EPS is a postponed/estimated date
        out = out[out["eps_reported"].notna()]
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_manual(path, stock):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(p, parse_dates=["date"])
    df = df[df["stock"].str.lower() == stock.lower()]
    return pd.DataFrame({"date": df["date"].dt.normalize()})


def load_dates(stock, ticker, cfg, refresh=False, fetch=fetch_yahoo_dates):
    """Return (table, source note). Never raises: a stock without dates gets an empty table and a note."""
    folder = Path(cfg.get("earnings_dates_dir", "data/earnings_dates"))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{stock}.csv"
    notes, yahoo = [], pd.DataFrame(columns=["date"])
    if path.exists() and not refresh:
        yahoo = pd.read_csv(path, parse_dates=["date"])
        notes.append("snapshot")
    else:
        try:
            yahoo = fetch(ticker)
            yahoo.to_csv(path, index=False)
            notes.append("yahoo")
        except Exception as exc:
            notes.append(f"yahoo failed ({type(exc).__name__})")
    manual = load_manual(cfg.get("earnings_manual_csv", "data/earnings_manual.csv"), stock)
    if len(manual):
        notes.append(f"manual +{len(manual)}")
    table = pd.concat([f for f in (yahoo, manual) if len(f)], ignore_index=True) if (len(yahoo) or len(manual)) else pd.DataFrame()
    if table.empty:
        return pd.DataFrame(columns=["date", "eps_estimate", "eps_reported", "surprise_pct"]), "; ".join(notes)
    for c in ("eps_estimate", "eps_reported", "surprise_pct"):
        if c not in table:
            table[c] = float("nan")
    table = table.sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    return table[["date", "eps_estimate", "eps_reported", "surprise_pct"]], "; ".join(notes)
