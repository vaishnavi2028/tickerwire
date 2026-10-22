"""Tickerwire dashboard. Run with:  streamlit run src/app.py"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from qa import answer_question

DATA = Path(__file__).resolve().parent.parent / "analysis" / "data"

st.set_page_config(page_title="Tickerwire", page_icon="📈", layout="wide")


# ---------- data loading ----------
@st.cache_data(show_spinner=False)
def load_csv(name: str, mtime: float, **kwargs) -> pd.DataFrame:
    path = DATA / name
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def get(name: str, **kwargs) -> pd.DataFrame:
    path = DATA / name
    return load_csv(name, path.stat().st_mtime if path.exists() else 0.0, **kwargs)


summary = get("sentiment_summary.csv")
signals = get("signals.csv")
news = get("news_headlines.csv")
prices = get("combined_price_data.csv", parse_dates=["Date"])
vol = get("stock_volatility_data.csv")
keywords = get("news_keyword_frequency.csv")
forecasts = get("forecast_results.csv", parse_dates=["ds"])
epochs = get("epoch_analysis.csv")
trades = get("backtest_trades.csv")
_bt = DATA / "backtest_summary.json"
bt_summary = json.loads(_bt.read_text()) if _bt.exists() else {}

# ---------- sidebar ----------
st.sidebar.title("Tickerwire")
dark = st.sidebar.toggle("Dark mode", value=True)
BG, FG, GRID = ("#0e1117", "#fafafa", "#2b2f3a") if dark else ("#ffffff", "#1a1a1a", "#e3e3e3")
st.markdown(
    f"<style>.stApp{{background-color:{BG};color:{FG}}} "
    f"[data-testid='stSidebar']{{background-color:{'#161a23' if dark else '#f4f5f7'}}} "
    f".stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp label,.stApp span,.stApp li{{color:{FG}}}</style>",
    unsafe_allow_html=True,
)


def style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(background=BG)
        .configure_axis(labelColor=FG, titleColor=FG, gridColor=GRID, domainColor=GRID)
        .configure_title(color=FG)
        .configure_legend(labelColor=FG, titleColor=FG)
        .configure_view(strokeWidth=0)
    )


def show(chart: alt.Chart) -> None:
    st.altair_chart(style(chart), width="stretch", theme=None)


st.title("📈 Tickerwire: news-driven trading signals")
st.caption("Learning project. Nothing here is financial advice.")

if summary.empty:
    st.info("No data found yet. Run `python src/tickerwire.py` first, then refresh this page.")
    st.stop()

labels = sorted(summary["sentiment_label"].unique())
sel_labels = st.sidebar.multiselect("Sentiment", labels, default=labels)
max_m = int(summary["mentions"].max())
min_m = st.sidebar.slider("Minimum mentions", 1, max(max_m, 2), 1)
lo, hi = st.sidebar.slider("Sentiment score range", 0.0, 1.0, (0.0, 1.0), 0.01)

view = summary[
    summary["sentiment_label"].isin(sel_labels)
    & (summary["mentions"] >= min_m)
    & summary["sentiment_score"].between(lo, hi)
]
view_tickers = set(view["ticker"])

tab_sent, tab_news, tab_px, tab_fc, tab_bt, tab_qa = st.tabs(
    ["Sentiment overview", "News headlines", "Historical prices", "Forecast", "Backtest", "Ask the data"]
)

# ---------- tab 1 ----------
with tab_sent:
    n_buy = int((signals.get("signal", pd.Series(dtype=str)) == "BUY").sum())
    n_sell = int((signals.get("signal", pd.Series(dtype=str)) == "SELL").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers in the news", len(summary))
    c2.metric("Headlines analysed", len(news))
    c3.metric("BUY signals", n_buy)
    c4.metric("SELL signals", n_sell)

    if not signals.empty:
        st.subheader("Latest signals")
        cols = [c for c in ["ticker", "company", "signal", "reason", "sentiment_score", "mentions",
                            "expected_move_pct", "position_size_pct", "stop_loss", "take_profit",
                            "forecast_source"] if c in signals.columns]
        st.dataframe(signals[signals["ticker"].isin(view_tickers)][cols], width="stretch", hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment score distribution")
        show(alt.Chart(view).mark_bar(color="#4c9be8").encode(
            alt.X("sentiment_score:Q", bin=alt.Bin(maxbins=20), title="Sentiment score (0 = negative, 1 = positive)"),
            alt.Y("count():Q", title="Tickers"),
        ).properties(height=300))
    with right:
        st.subheader("Most mentioned tickers")
        top = view.sort_values("mentions", ascending=False).head(15)
        show(alt.Chart(top).mark_bar().encode(
            alt.X("mentions:Q", title="Headlines"),
            alt.Y("ticker:N", sort="-x", title=None),
            alt.Color("sentiment_score:Q", scale=alt.Scale(scheme="redyellowgreen", domain=[0, 1]), title="Sentiment"),
            tooltip=["ticker", "mentions", alt.Tooltip("sentiment_score:Q", format=".2f")],
        ).properties(height=300))

    left, right = st.columns(2)
    with left:
        st.subheader("Volatility vs return")
        v = vol[vol["ticker"].isin(view_tickers)] if not vol.empty else vol
        if v.empty:
            st.caption("No price data for the filtered tickers.")
        else:
            show(alt.Chart(v).mark_circle(size=140).encode(
                alt.X("volatility_ann:Q", title="Annualised volatility", axis=alt.Axis(format="%")),
                alt.Y("return_ann:Q", title="Annualised average return", axis=alt.Axis(format="%")),
                tooltip=["ticker", alt.Tooltip("volatility_ann:Q", format=".1%"), alt.Tooltip("return_ann:Q", format=".1%")],
            ).properties(height=300))
    with right:
        st.subheader("Top headline keywords")
        if keywords.empty:
            st.caption("No keyword data.")
        else:
            show(alt.Chart(keywords.head(20)).mark_bar(color="#e8a24c").encode(
                alt.X("count:Q", title="Occurrences"), alt.Y("keyword:N", sort="-x", title=None),
            ).properties(height=300))

# ---------- tab 2 ----------
with tab_news:
    if news.empty:
        st.caption("No headlines saved.")
    else:
        news = news.copy()
        news["published_at"] = pd.to_datetime(news["published_at"], errors="coerce", utc=True)
        c1, c2 = st.columns(2)
        dmin, dmax = news["published_at"].min().date(), news["published_at"].max().date()
        dates = c1.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
        pick = c2.multiselect("Ticker", sorted(view_tickers))
        out = news
        if isinstance(dates, tuple) and len(dates) == 2:
            out = out[out["published_at"].dt.date.between(dates[0], dates[1])]
        if pick:
            out = out[out["tickers"].fillna("").apply(lambda s: any(t in s.split(",") for t in pick))]
        cols = [c for c in ["published_at", "source", "title", "tickers", "score", "label", "url"] if c in out.columns]
        st.dataframe(
            out[cols].sort_values("published_at", ascending=False),
            width="stretch", hide_index=True,
            column_config={"url": st.column_config.LinkColumn("link"), "score": st.column_config.NumberColumn(format="%.2f")},
        )

# ---------- tab 3 ----------
with tab_px:
    if prices.empty:
        st.caption("No price data saved.")
    else:
        avail = sorted(set(prices["Ticker"]) & view_tickers) or sorted(set(prices["Ticker"]))
        chosen = st.multiselect("Tickers", avail, default=avail[:3])
        rebase = st.checkbox("Rebase to 100 so tickers are comparable", value=True)
        d = prices[prices["Ticker"].isin(chosen)].sort_values("Date").copy()
        if rebase and not d.empty:
            d["Close"] = d.groupby("Ticker")["Close"].transform(lambda s: s / s.iloc[0] * 100)
        if d.empty:
            st.caption("Pick at least one ticker.")
        else:
            show(alt.Chart(d).mark_line().encode(
                alt.X("Date:T"), alt.Y("Close:Q", title="Index (start = 100)" if rebase else "Close ($)", scale=alt.Scale(zero=False)),
                alt.Color("Ticker:N"), tooltip=["Date:T", "Ticker", alt.Tooltip("Close:Q", format=".2f")],
            ).properties(height=420))

# ---------- tab 4 ----------
with tab_fc:
    if forecasts.empty:
        st.caption("No forecasts yet. They only run for tickers with a strong sentiment score.")
    else:
        t = st.selectbox("Ticker", sorted(forecasts["ticker"].unique()))
        f = forecasts[forecasts["ticker"] == t].sort_values("ds")
        hist = prices[prices["Ticker"] == t].sort_values("Date").tail(60).rename(columns={"Date": "ds", "Close": "y"})
        st.caption(f"Source: {f['source'].iloc[0]}. Shaded bands are the 80% and 95% confidence intervals.")
        base = alt.Chart(f).encode(alt.X("ds:T", title=None))
        band95 = base.mark_area(opacity=0.15, color="#4c9be8").encode(alt.Y("lo_95:Q", scale=alt.Scale(zero=False), title="Price ($)"), alt.Y2("hi_95:Q"))
        band80 = base.mark_area(opacity=0.30, color="#4c9be8").encode(alt.Y("lo_80:Q"), alt.Y2("hi_80:Q"))
        line = base.mark_line(color="#4c9be8", strokeDash=[5, 3]).encode(alt.Y("y_hat:Q"))
        past = alt.Chart(hist).mark_line(color="#e8e8e8" if dark else "#333333").encode(alt.X("ds:T"), alt.Y("y:Q"))
        show((band95 + band80 + line + past).properties(height=420))
        sig = signals[signals["ticker"] == t]
        if not sig.empty:
            r = sig.iloc[0]
            st.write(f"**Signal: {r['signal']}**. {r['reason']}.")

    if not epochs.empty:
        st.subheader("Superposed epoch analysis")
        st.caption("Average price path around unusually large daily moves. Day 0 is the spike.")
        pick = st.selectbox("Ticker for epoch view", sorted(epochs["ticker"].unique()), key="epoch_t")
        e = epochs[epochs["ticker"] == pick]
        show(alt.Chart(e).mark_line(point=True).encode(
            alt.X("rel_day:Q", title="Days from spike"), alt.Y("mean_pct:Q", title="Average move vs day 0 (%)"),
            alt.Color("direction:N", scale=alt.Scale(domain=["up", "down"], range=["#3fb950", "#f85149"])),
            tooltip=["direction", "rel_day", alt.Tooltip("mean_pct:Q", format=".2f"), "n_events"],
        ).properties(height=300))

# ---------- backtest tab ----------
with tab_bt:
    if not bt_summary or bt_summary.get("n_trades", 0) == 0:
        st.info("No scored trades yet. Run `python src/backtest.py --backfill 25` (or keep running the pipeline daily, then `python src/backtest.py`).")
    else:
        n = bt_summary["n_trades"]
        st.caption(
            f"{n} trades from {bt_summary['first_signal']} to {bt_summary['last_signal']}. "
            f"Entered at the next day's open, held {bt_summary['hold_days']} trading days, "
            f"{bt_summary['cost_bps']:g} bps cost each way, benchmark {bt_summary['benchmark']}."
        )
        if n < 30:
            st.warning(f"Only {n} trades. That is too few to conclude anything. Treat this as a smoke test.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Hit rate", f"{bt_summary['hit_rate_pct']:.0f}%")
        m2.metric("Avg net return", f"{bt_summary['avg_net_return_pct']:+.2f}%")
        m3.metric("Avg vs benchmark", f"{bt_summary['avg_excess_vs_benchmark_pct']:+.2f}%")
        m4.metric("Same stocks, no signal", f"{bt_summary['avg_buy_and_hold_same_stocks_pct']:+.2f}%")
        lo, hi = bt_summary["excess_ci95_pct"]
        st.write(f"95% confidence interval for the average return vs benchmark: **{lo:+.2f}% to {hi:+.2f}%**.")
        if lo <= 0 <= hi:
            st.write("That range includes zero, so the result can't be told apart from luck.")
        if not trades.empty:
            t = trades.copy()
            t["net_return_pct"] = t["net_return"] * 100
            t["label"] = t["ticker"] + " " + t["signal_date"].astype(str)
            show(alt.Chart(t).mark_bar().encode(
                alt.X("net_return_pct:Q", title="Net return per trade (%)"),
                alt.Y("label:N", sort="-x", title=None),
                alt.Color("signal:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["#3fb950", "#f85149"])),
                tooltip=["ticker", "signal", "entry_date", "exit_date", alt.Tooltip("net_return_pct:Q", format=".2f")],
            ).properties(height=max(200, 22 * len(t))))
            st.dataframe(t.drop(columns=["label", "net_return_pct"]), width="stretch", hide_index=True)

# ---------- tab 5 ----------
with tab_qa:
    st.write("Ask a question about the latest run. This is keyword matching, not an LLM.")
    q = st.text_input("Question", placeholder="highest sentiment? / most volatile stock? / return of AAPL")
    if q:
        st.success(answer_question(q, summary, signals, vol))
    st.caption("Try: 'which stocks to buy?', 'signal for TSLA', 'most mentioned?'")
