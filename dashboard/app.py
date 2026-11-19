"""Equity research dashboard: puts the two projects side by side.

  Tab 1  Factor model   which stocks the model really explains, and how betas drift
  Tab 2  Scenarios      "what if Brent falls 20%?" + how the model did in past crises
  Tab 3  Months horizon  do the factors explain / predict 1-3 month moves?
  Tab 4  Earnings & macro  inflation and election tests, and the earnings-release event study
  Tab 5  Price cones   simulated price paths: how wide could the move be (not a direction forecast)
  Tab 6  Fundamentals   earnings-release extractor: ratios, flags and accuracy vs a hand-checked golden set

Run:  streamlit run dashboard/app.py      (from the equity-research-ai folder)
It only READS the output files of the two projects; the live scenario box also refits the model locally
from the saved data snapshots (no API key, no cost).
"""
import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
OIL = ROOT / "oil-equity-fx-model"
EXT = ROOT / "earnings-release-extractor"
MF = OIL / "outputs" / "multifactor"
SC = OIL / "outputs" / "scenarios"

st.set_page_config(page_title="Equity research dashboard", layout="wide")


# ---------------------------------------------------------------- data loading
@st.cache_data
def read_csv(path, **kw):
    p = Path(path)
    return pd.read_csv(p, **kw) if p.exists() else None


@st.cache_data
def stock_info():
    cfg = yaml.safe_load((OIL / "config_multifactor.yaml").read_text())
    return {k: v for k, v in cfg["stocks"].items()}, cfg


@st.cache_resource(show_spinner="Loading saved market data (first time only)...")
def oil_world():
    """Imports the oil model code and loads the saved snapshots. Both projects have a package called `src`,
    so this one is loaded only here, with the oil folder as working directory."""
    os.chdir(OIL)
    if str(OIL) not in sys.path:
        sys.path.insert(0, str(OIL))
    from src import factors, scenarios, sources
    from src.run_factors import date_range, load_config
    cfg = load_config("config_multifactor.yaml")
    start, end = date_range(cfg)
    levels, problems = sources.load_all(cfg, start, end)
    return cfg, levels, factors, scenarios, problems


@st.cache_resource
def cones_module():
    """cones.py has no imports from `src`, so it can be loaded straight from its file."""
    spec = importlib.util.spec_from_file_location("oil_cones", OIL / "src" / "cones.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_data
def simulate_cone(stock, horizon, method, n_paths, drift_total, drift_days, seed=7):
    cn = cones_module()
    prices = pd.read_csv(OIL / raw_cfg.get("snapshot_dir", "data/snapshots") / f"{stock}.csv", index_col=0, parse_dates=True).iloc[:, 0].dropna()
    r = cn.log_returns(prices).iloc[-int(raw_cfg.get("cones_lookback_days", 1260)):]
    sims = cn.simulate_returns(r, horizon, n_paths, int(raw_cfg.get("cones_block_days", 5)), method,
                               float(raw_cfg.get("cones_halflife_days", 30)), seed)
    drift = cn.scenario_drift(horizon, drift_total, drift_days) if drift_total else None
    paths = cn.price_paths(float(prices.iloc[-1]), sims, drift)
    table = cn.cone(paths)
    table.loc[0] = float(prices.iloc[-1])
    return table.sort_index(), cn.end_probabilities(paths, float(prices.iloc[-1])), float(prices.iloc[-1]), prices.index[-1]


def reliable_flag(row, p_max):
    return bool(row["p_factors_beats_capm"] < p_max and row["oos_r2_factors"] > row["oos_r2_capm"])


stocks, raw_cfg = stock_info()
p_max = raw_cfg.get("reliable_p_max", 0.01)
summary = read_csv(MF / "summary.csv")

st.title("Equity research dashboard")
st.caption("Commodity-linked stocks (oil, agro, mining/steel; Brazil and US) + earnings-release extractor. "
           "Built to show what the models can and cannot do, not to give trading advice.")

tab1, tab2, tab5, tab6, tab7, tab3, tab4 = st.tabs(["Factor model", "Scenarios", "Months horizon", "Earnings & macro",
                                                    "Price cones", "Fundamentals", "About / limits"])

# ================================================================= TAB 1
with tab1:
    if summary is None:
        st.warning("No model results yet. Run `python -m src.run_factors` in oil-equity-fx-model.")
    else:
        s = summary[summary["error"].isna()].copy()
        s["sector"] = s["stock"].map(lambda k: stocks.get(k, {}).get("sector"))
        s["country"] = s["stock"].map(lambda k: stocks.get(k, {}).get("country"))
        s["reliable"] = s.apply(lambda r: reliable_flag(r, p_max), axis=1)

        st.subheader("Which stocks does the model explain?")
        st.write(f"**Reliable** = the full factor model beats a market-only model out of sample with p < {p_max} "
                 f"(strict on purpose: with {len(s)} stocks tested, a few would pass at 0.05 by luck).")
        c1, c2 = st.columns(2)
        sectors = c1.multiselect("Sector", sorted(s["sector"].dropna().unique()), default=sorted(s["sector"].dropna().unique()))
        countries = c2.multiselect("Country", sorted(s["country"].dropna().unique()), default=sorted(s["country"].dropna().unique()))
        view = s[s["sector"].isin(sectors) & s["country"].isin(countries)].sort_values("oos_r2_factors", ascending=False)
        table = view[["stock", "country", "sector", "oos_r2_capm", "oos_r2_factors", "p_factors_beats_capm",
                      "coverage_80_factors", "reliable"]].rename(columns={
            "oos_r2_capm": "R² market only", "oos_r2_factors": "R² full model",
            "p_factors_beats_capm": "p-value vs market only", "coverage_80_factors": "80% range hit rate"})
        st.dataframe(table.round(3), width="stretch", hide_index=True)
        st.bar_chart(view.set_index("stock")[["oos_r2_capm", "oos_r2_factors"]]
                     .rename(columns={"oos_r2_capm": "market only", "oos_r2_factors": "full model"}))
        st.caption("R² = share of the stock's daily moves explained OUT OF SAMPLE (walk-forward, never using the future). "
                   "It explains same-day relationships; it is not a forecast of the factors themselves.")

        st.subheader("One stock in detail")
        pick = st.selectbox("Stock", list(view["stock"]) or list(s["stock"]), key="t1_stock")
        ols = read_csv(MF / pick / "ols_full_sample.csv", index_col=0)
        roll = read_csv(MF / pick / "rolling_coefficients.csv", index_col=0, parse_dates=True)
        left, right = st.columns(2)
        if ols is not None:
            o = ols.drop(index="const", errors="ignore")
            left.markdown("**Sensitivity to each factor (full sample)**")
            left.bar_chart(o["coef"])
            left.caption("Prices: effect on the stock's log return per 1 unit of the factor's log return. "
                         "Yields: per 1 percentage point. Factors are cleaned of the market move first.")
        if roll is not None:
            factor = right.selectbox("Rolling coefficient of", [c for c in roll.columns if c != "const"], key="t1_factor")
            right.markdown("**How that sensitivity drifted over time**")
            right.line_chart(roll[factor])

# ================================================================= TAB 2
with tab2:
    scen = read_csv(SC / "scenario_results.csv")
    eps = read_csv(SC / "backtest_episodes.csv")
    if scen is None:
        st.warning("No scenario results yet. Run `python -m src.run_scenarios` in oil-equity-fx-model.")
    else:
        names = sorted(scen["stock"].unique())
        pick2 = st.selectbox("Stock", names, key="t2_stock")
        rel = scen.loc[scen["stock"] == pick2, "reliable"].iloc[0]
        if rel is True or str(rel) == "True":
            st.success("The model is **reliable** for this stock (beats market-only out of sample).")
        else:
            st.warning("The model is **not shown to be reliable** for this stock. Treat results as illustrative.")

        st.subheader("Preset scenarios")
        reading = st.radio("Reading", ["with_correlations", "only_shocked"], horizontal=True,
                           help="with_correlations: the other factors move too, the way they usually do when the named "
                                "ones move (mixes in things that happen together, not only cause and effect). "
                                "only_shocked: only the factors you named move.")
        d = scen[(scen["stock"] == pick2) & (scen["reading"] == reading)]
        show = d[["scenario", "shocks_applied", "horizon_days", "impact_pct", "low_80_pct", "high_80_pct"]].round(1)
        st.dataframe(show, width="stretch", hide_index=True)
        st.bar_chart(d.set_index("scenario")["impact_pct"])
        st.caption("Impact = expected stock move (%) over the horizon, minus what the model predicts with no shock. "
                   "The 80% range is widened for crises, but in past crashes US oil stocks were still covered only ~58% "
                   "of the time: read the range as a minimum.")

        st.subheader("Your own scenario (live)")
        try:
            cfg, levels, factors, scenarios, problems = oil_world()
            design = factors.make_design(levels, pick2, cfg)
            X = design["X"]
            base = scenarios.base_columns(X)
            rate_cols = factors.rate_names(cfg, X.columns)
            st.write("Move any factor, leave the others at 0. Prices in %, yields in percentage points.")
            cols = st.columns(3)
            shocks = {}
            for i, f in enumerate(base):
                if f in rate_cols:
                    v = cols[i % 3].slider(f"{f} (pp)", -3.0, 3.0, 0.0, 0.25, key=f"sh_{pick2}_{f}")
                else:
                    v = cols[i % 3].slider(f"{f} (%)", -50, 50, 0, 1, key=f"sh_{pick2}_{f}") / 100
                if v:
                    shocks[f] = v
            horizon = st.slider("Horizon (trading days)", 5, 66, int(cfg["scenario_horizon_days"]), key="hz")
            if shocks:
                spec = {"custom": {"shocks": shocks, "horizon_days": horizon}}
                rows = scenarios.run_scenarios(X, design["y"], design["market_cols"], design["other_cols"],
                                               cfg, spec, rate_cols, design["substituted"])
                out = pd.DataFrame(rows)[["reading", "impact_pct", "low_80_pct", "high_80_pct", "beyond_history"]]
                st.dataframe(out.round(1), width="stretch", hide_index=True)
                if out["beyond_history"].astype(str).str.len().gt(0).any():
                    st.warning("Part of this shock is bigger than anything in the sample: the linear model is extrapolating.")
            else:
                st.info("Move at least one slider to run a scenario.")
        except Exception as exc:       # the live part needs the saved snapshots; the rest of the page still works
            st.info(f"Live scenarios unavailable here ({type(exc).__name__}: {exc}).")

        st.subheader("Would it have worked? Past episodes")
        if eps is not None:
            e = eps[(eps["stock"] == pick2) & eps["skipped"].isna()].copy()
            if len(e):
                e["label"] = e["episode"] + " (" + e["start"].astype(str) + ")"
                st.dataframe(e[["episode", "start", "end", "days", "actual_pct", "model_pct", "capm_pct", "inside_80"]]
                             .round(1), width="stretch", hide_index=True)
                st.bar_chart(e.set_index("label")[["actual_pct", "model_pct", "capm_pct"]])
                st.caption("The model is fitted ONLY on days before each episode, then fed the factor moves that really "
                           "happened. It tests the translation factor -> stock, not whether anyone could forecast the factors.")
            else:
                st.info("No episodes with enough earlier history for this stock.")


# ================================================================= TAB 5 (months horizon)
with tab5:
    LH = OIL / "outputs" / "long_horizon"
    lh = read_csv(LH / "summary.csv")
    st.subheader("Do the factors EXPLAIN the price over months? Do they PREDICT it?")
    st.write("Two different questions. **Explain**: over the same 1- or 3-month window, how much of the stock's move do the "
             "factors' moves account for? (You would need to know the factors in advance.) **Predict**: do the factors' "
             "moves over the LAST window help forecast the stock's NEXT window? That is the only one a trader could use.")
    if lh is None:
        st.warning("No results yet. Run `python -m src.run_long_horizon` in oil-equity-fx-model.")
    else:
        ok = lh[lh["error"].isna()] if "error" in lh else lh
        h = st.radio("Horizon (trading days)", sorted(ok["horizon_days"].unique()), horizontal=True,
                     format_func=lambda d: f"{d} days (~{round(d / 21)} month{'s' if d > 25 else ''})")
        v = ok[ok["horizon_days"] == h].sort_values("explain_r2_full", ascending=False)
        n_pred = int(((v["predict_r2_full"] > 0) & (v["predict_p_full_vs_mean"] < p_max)).sum())
        a, b, c = st.columns(3)
        a.metric("Stocks tested", len(v))
        b.metric("Explained better than market-only", int((v["verdict"].str.startswith("explains")).sum()))
        c.metric(f"Predicted better than a running average (p < {p_max})", n_pred)
        st.dataframe(v[["stock", "country", "sector", "n_windows", "explain_r2_capm", "explain_r2_full",
                        "predict_r2_full", "predict_p_full_vs_mean", "verdict"]].round(2),
                     width="stretch", hide_index=True)
        st.bar_chart(v.set_index("stock")[["explain_r2_full", "predict_r2_full"]]
                     .rename(columns={"explain_r2_full": "explain (same window)", "predict_r2_full": "predict (next window)"}))
        st.caption("R² is out of sample (walk-forward, with a gap so no window overlaps the test). A negative R² means the model "
                   "did WORSE than just predicting the running average return. Windows overlap, so p-values use "
                   "HAC errors. With about 30 stocks, a few p-values below 0.05 appear by chance.")
        pick5 = st.selectbox("Coefficients of", list(v["stock"]), key="t5_stock")
        co = read_csv(LH / f"{pick5}_h{h}_coefficients.csv", index_col=0)
        if co is not None:
            st.markdown(f"**Sensitivity over {h}-day windows, {pick5}** (read it, do not trade on it)")
            st.bar_chart(co.drop(index="const", errors="ignore")["coef"])

# ================================================================= TAB 6 (earnings & macro)
with tab6:
    MA, EA = OIL / "outputs" / "macro", OIL / "outputs" / "earnings"
    st.subheader("1. Do inflation or election years add anything?")
    st.write("On top of the base factor model, tested out of sample at 1 and 3 months, explaining and predicting. "
             "Many tests are run, so a few look good by luck; the **Holm** column is the honest one.")
    mt = read_csv(MA / "tests.csv")
    if mt is None:
        st.warning("No results yet. Run `python -m src.run_macro` in oil-equity-fx-model.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Tests run", len(mt))
        b.metric("Look helpful (p < 0.05, better R²)", int(mt["helps_p05"].sum()),
                 help=f"Luck alone would give about {0.025 * len(mt):.0f}.")
        c.metric(f"Helpful at p < {p_max}", int(mt["helps_alone"].sum()))
        d.metric("Survive the Holm correction", int(mt["helps_after_correction"].sum()))
        g = mt.groupby(["mode", "variant"]).agg(tests=("p", "size"), helpful_p05=("helps_p05", "sum"),
                                                survive_holm=("helps_after_correction", "sum"),
                                                share_with_positive_gain=("gain", lambda x: (x > 0).mean()))
        g["avg_R2_change_vs_base"] = (mt["r2_variant"] - mt["r2_base"]).groupby([mt["mode"], mt["variant"]]).mean()
        st.dataframe(g.round(3), width="stretch")
        st.caption("variant: infl = + change in year-on-year inflation (US CPI, Brazil IPCA, known only ~20 days after the month ends); "
                   "elec = + presidential-election-year flag (5 per country in 20 years: too few to prove anything); both = the two. "
                   "A negative R² change means the extra variable made the out-of-sample fit WORSE.")
        top = mt[mt["helps_p05"]].sort_values("p")
        if len(top):
            st.markdown("**The few that look helpful (check the Holm column before believing any):**")
            st.dataframe(top[["stock", "horizon_days", "mode", "variant", "r2_base", "r2_variant", "p", "p_holm"]].round(4),
                         width="stretch", hide_index=True)

    st.subheader("2. What happens around earnings releases?")
    st.write("For each past release, the stock's move is compared with what the factor model expected from the real moves of the market, "
             "FX, yields and commodities (model fitted only on days before). Reaction = the 2 days from the release day; "
             "drift = the 10 trading days after that.")
    pooled = read_csv(EA / "pooled.csv")
    if pooled is None:
        st.warning("No results yet. Run `python -m src.run_earnings` in oil-equity-fx-model.")
    else:
        allr = pooled[pooled["group"] == "all"].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Events", int(allr["n_events"]))
        b.metric("Reaction vs a normal 2-day move", f"{allr['move_ratio']:.1f}x", help="1.0x = nothing special")
        c.metric("Drift after the reaction (p-value)", f"{allr['drift_p']:.2f}", help="Small p = the stock tends to continue or reverse")
        if "surprise_p" in allr and pd.notna(allr.get("surprise_p")):
            d.metric("EPS surprise vs reaction (p-value)", f"{allr['surprise_p']:.4f}")
        names = {"group": "group", "n_events": "events", "mean_car_react": "avg abnormal reaction", "p_mean_car_react": "avg reaction p",
                 "move_ratio": "reaction size (x normal)", "drift_slope": "drift slope", "drift_p": "drift p",
                 "pre_slope": "run-up slope", "pre_p": "run-up p", "change_corr_market": "change in corr. with market (after - before)",
                 "change_corr_market_p": "corr. change p", "surprise_slope": "reaction per 1% EPS surprise", "surprise_p": "surprise p"}
        cols = [c for c in names if c in pooled.columns]
        st.dataframe(pooled[cols].rename(columns=names).round(4), width="stretch", hide_index=True)
        st.caption("Drift slope > 0: the stock keeps going the way it reacted; < 0: it partly reverses. Run-up slope: does it already move "
                   "before the release in the direction of the reaction? Standard errors are clustered by calendar quarter because all "
                   "companies report in the same weeks.")
        es = read_csv(EA / "summary.csv")
        if es is not None:
            es = es[es["n_events"] > 0].copy()
            es["thin"] = es["n_events"] < 20
            st.markdown("**By stock** (stocks with fewer than 20 events are marked `thin`: do not read much into them)")
            keep = [c for c in ["stock", "country", "sector", "n_events", "thin", "move_ratio", "drift_slope", "drift_p",
                                "surprise_spearman", "surprise_p"] if c in es.columns]
            st.dataframe(es.sort_values("move_ratio", ascending=False)[keep].round(3), width="stretch", hide_index=True)
            st.bar_chart(es.set_index("stock")["move_ratio"].sort_values(ascending=False))
            st.caption("Bars: how many times bigger the 2-day reaction is than a normal 2-day move. "
                       f"With {int(es['drift_p'].notna().sum())} stocks tested, about {0.05 * es['drift_p'].notna().sum():.0f} "
                       f"drift p-values below 0.05 are expected by luck alone (found: {int((es['drift_p'] < 0.05).sum())}).")
        ev = read_csv(EA / "events.csv", parse_dates=["date"])
        if ev is not None:
            pick6 = st.selectbox("One stock", sorted(ev["stock"].unique()), key="t6_stock")
            e = ev[ev["stock"] == pick6]
            l, r = st.columns(2)
            l.markdown("**Average abnormal move around the release**")
            l.bar_chart(pd.Series({"10 days before": e["car_pre"].mean(), "reaction (2 days)": e["car_react"].mean(),
                                   "10 days after": e["car_post"].mean()}) * 100)
            l.caption("Percent, averaged over events; the average hides big moves in both directions (see reaction size above).")
            sp = e[["surprise_pct", "car_react"]].dropna()
            sp = sp[sp["surprise_pct"].abs() <= 100]
            if len(sp) >= 8:
                r.markdown("**EPS surprise vs abnormal reaction**")
                r.scatter_chart(sp.rename(columns={"surprise_pct": "EPS surprise (%)", "car_react": "abnormal reaction"}),
                                x="EPS surprise (%)", y="abnormal reaction")
            else:
                r.info("Not enough EPS-surprise data for this stock.")

# ================================================================= TAB 7 (price cones)
with tab7:
    st.subheader("Where could the price be? A cone of simulated paths")
    st.write("Thousands of possible paths built by resampling the stock's own past daily returns (last ~5 years, in blocks of 5 days). "
             "The centre is **neutral on purpose**: the factor model cannot predict direction, so this shows how WIDE plausible moves are, "
             "not where the price will go.")
    cone_stocks = sorted(k for k in stocks if (OIL / raw_cfg.get("snapshot_dir", "data/snapshots") / f"{k}.csv").exists())
    c1, c2, c3 = st.columns(3)
    pick7 = c1.selectbox("Stock", cone_stocks, key="t7_stock")
    hz7 = c2.slider("Horizon (trading days)", 5, 126, 21, key="t7_h", help="21 days is about one month, 63 about three months.")
    method7 = c3.radio("Volatility", ["fhs", "bootstrap"], horizontal=True, key="t7_m",
                       format_func=lambda m: "today's volatility (fhs)" if m == "fhs" else "5-year average",
                       help="fhs: the cone is wider after turbulent weeks and narrower after calm ones.")
    scen7 = read_csv(SC / "scenario_results.csv")
    drift_total, drift_days, label = 0.0, 22, "neutral centre"
    if scen7 is not None:
        d7 = scen7[(scen7["stock"] == pick7) & (scen7["reading"] == "only_shocked")]
        opts = ["none"] + sorted(d7["scenario"].unique())
        sc_pick = st.selectbox("Optional: shift the centre by a scenario from the model (only the named factors move)", opts, key="t7_sc")
        if sc_pick != "none":
            row = d7[d7["scenario"] == sc_pick].iloc[0]
            drift_total, drift_days = float(row["impact_pct"]) / 100.0, int(row["horizon_days"])
            label = f"centre shifted by {sc_pick}: {row['impact_pct']:.1f}% over {drift_days} days"
    try:
        table, probs, last_price, last_date = simulate_cone(pick7, hz7, method7, 3000, drift_total, drift_days)
        a, b, c, d = st.columns(4)
        a.metric(f"Last price ({last_date.date()})", f"{last_price:,.2f}")
        b.metric(f"80% range after {hz7} days", f"{table['p10'].iloc[-1]:,.2f} to {table['p90'].iloc[-1]:,.2f}")
        c.metric("Chance of ending 10% lower", f"{probs['below_-10%']:.0%}")
        d.metric("Chance of ending 10% higher", f"{probs['above_+10%']:.0%}")
        st.line_chart(table[["p5", "p25", "p50", "p75", "p95"]])
        st.caption(f"Lines: 5th, 25th, 50th, 75th and 95th percentile of simulated prices ({label}). "
                   "Half of the paths end between the 25th and 75th lines, nine in ten between the 5th and 95th.")
    except Exception as exc:
        st.info(f"Cone unavailable ({type(exc).__name__}: {exc}). Needs the saved price snapshots (run run_factors).")
    cal = read_csv(OIL / "outputs" / "cones" / "calibration.csv")
    st.subheader("Is the width right? Past hit rates")
    if cal is None:
        st.warning("Run `python -m src.run_cones` in oil-equity-fx-model to check the cones against the past.")
    else:
        h_cal = sorted(cal["horizon_days"].unique())
        h_near = min(h_cal, key=lambda x: abs(x - hz7))
        one = cal[(cal["stock"] == pick7) & (cal["horizon_days"] == h_near)]
        st.write(f"At every past date (about monthly), the cone was rebuilt using only earlier data and compared with what really happened "
                 f"over the next {h_near} days. A perfect 80% cone would contain the real outcome 80% of the time.")
        if len(one):
            t = one.set_index("method")[["n_dates", "coverage_50", "coverage_80", "coverage_90", "coverage_80_turbulent"]]
            st.dataframe(t.rename(columns={"coverage_50": "50% cone hit rate", "coverage_80": "80% cone hit rate",
                                           "coverage_90": "90% cone hit rate", "coverage_80_turbulent": "80% cone, turbulent times"}).round(2),
                         width="stretch")
        avg = cal[cal["horizon_days"] == h_near].groupby("method")[["coverage_50", "coverage_80", "coverage_90", "coverage_80_turbulent"]].mean()
        st.markdown(f"**All stocks, {h_near}-day horizon (average)**")
        st.dataframe(avg.round(2), width="stretch")
        st.caption("Turbulent times = the 20% of dates with the highest volatility before the forecast. Cones built with today's volatility (fhs) "
                   "fit turbulent times better; the 5-year average is a bit too wide in calm times and too narrow in crises. "
                   "Dates overlap, so these hit rates are only approximate. A cone can be well calibrated and still say nothing about direction.")

# ================================================================= TAB 3
with tab3:
    res = read_csv(EXT / "outputs" / "results.csv")
    flags = read_csv(EXT / "outputs" / "flags.csv")
    if res is None:
        st.warning("No extractor results yet. Run `python -m src.main` in earnings-release-extractor.")
    else:
        st.subheader("Ratios only (no currency conversion)")
        nb = res[res["kind"] == "non_financial"]
        bk = res[res["kind"] == "bank"]
        if len(nb):
            st.markdown("**Non-financial companies**")
            st.dataframe(nb[["file", "currency", "ebitda_margin_pct", "net_margin_pct", "net_debt_ebitda", "net_income_definition"]]
                         .round(1), width="stretch", hide_index=True)
        if len(bk):
            st.markdown("**Banks**")
            st.dataframe(bk[["file", "currency", "roe", "efficiency_ratio", "cet1", "net_income_definition"]]
                         .round(1), width="stretch", hide_index=True)
        st.caption("`n/a` means the release does not state it (e.g. ExxonMobil's release has no revenue line) "
                   "or the extractor refused to guess. Net income definitions differ across companies.")

        st.subheader("Flags to check by hand")
        if flags is not None and len(flags):
            st.dataframe(flags[["file", "field", "flag_type", "message", "page"]], width="stretch", hide_index=True)
        else:
            st.success("No flags.")

        st.subheader("Accuracy against the hand-checked golden set")
        golden = EXT / "golden" / "golden.csv"
        if golden.exists():
            spec = importlib.util.spec_from_file_location("ext_evaluate", EXT / "src" / "evaluate.py")
            ev = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ev)
            df = ev.evaluate(str(golden), str(EXT / "outputs" / "results.csv"))
            ran = df[df["verdict"] != "file_not_run"]
            good = ran["verdict"].isin(["correct", "accepted", "null_ok"]).sum()
            a, b, c = st.columns(3)
            a.metric("Values checked", len(ran))
            b.metric("Right", f"{good} ({good / max(len(ran), 1):.0%})")
            c.metric("Wrong / missing / invented", int(len(ran) - good))
            alt = ran[ran["verdict"] == "accepted"]
            if len(alt):
                st.markdown("**Different definition than the company's headline number** (check this is what you want):")
                st.dataframe(alt[["file", "metric", "got", "value", "note"]].rename(columns={"got": "extractor", "value": "golden"}),
                             width="stretch", hide_index=True)
            st.caption(f"The golden set has {len(df)} values read from the PDF text, each with the page and exact quote. "
                       "It is small and was built after seeing the extractor's output: good for catching regressions, "
                       "not a guarantee of overall accuracy.")

# ================================================================= TAB 4
with tab4:
    st.markdown("""
**What this is**: a research prototype that links commodity prices, rates, FX and market indices to stock moves,
and extracts comparable ratios from earnings releases.

**What it does not do**
- Over months the factors EXPLAIN the price but do not PREDICT the next window (see the Months horizon tab).
- It does not forecast commodity prices, FX or rates. Scenarios are "if X moves, the stock usually moves like this".
- Coefficients are linear and fitted on the past; large shocks are extrapolations (flagged when beyond history).
- Ranges assume the size of past errors; real crises can be worse, especially for US oil stocks.
- "With correlations" readings mix in factors that tend to move together; they are not causal.
- Brazilian 2y/10y yields were tested and added almost nothing (kept off by default).
- The extractor's accuracy figure comes from a small hand-built set; releases of other companies or quarters may behave differently.
- Inflation and election years were tested and added nothing that survives correction for multiple testing; the earnings study shows reaction size and EPS-surprise link, but no usable drift after the release.
- Earnings dates come from Yahoo (some Brazilian ones may be off by a day or missing).
- Price cones have a neutral centre: they show how wide moves can be, not where the price goes. Their width was checked against the past (hit rates in the tab), and they are too narrow in the worst crises.
- Not investment advice.
""")
