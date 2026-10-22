"""
Score Tickerwire's signals against what prices actually did next.

Two ways to get signals to score:
  * Forward tracking: every `python src/tickerwire.py` run appends to
    analysis/data/signals_history.csv. Run it daily and the sample grows.
  * Backfill: `--backfill 25` rebuilds signals for the last ~25 days of news
    (NewsAPI's free plan only goes back about a month) and scores those now.

Rules that keep the test honest:
  * A signal made on date D is entered at the OPEN of the next trading day,
    never on D's close (no look-ahead).
  * It is held for `--hold-days` trading days and exits at that day's close.
  * Round-trip trading costs are subtracted (`--cost-bps` each way).
  * Every trade is compared with SPY over the same window, and with simply
    buying the same stock on the same day (does the sentiment direction add
    anything beyond just owning stocks that were in the news?).
  * Backfilled signals use the trend-extension forecast only, because calling
    TimeGPT for dozens of past days would be slow and costly. So a backfill
    tests "news sentiment + trend", not TimeGPT.

Usage:
    python src/backtest.py                       # score signals_history.csv
    python src/backtest.py --backfill 25         # rebuild last 25 days, then score
    python src/backtest.py --hold-days 3 --cost-bps 5 --long-only
This is a learning tool. A small sample cannot prove a strategy works.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import tickerwire as T

log = logging.getLogger("tickerwire.backtest")

TRADES_PATH = T.DATA_DIR / "backtest_trades.csv"
SUMMARY_PATH = T.DATA_DIR / "backtest_summary.json"
HISTORY_PATH = T.DATA_DIR / "signals_history.csv"


# --------------------------------------------------------------------------
# Core engine (pure functions, no network)
# --------------------------------------------------------------------------
def _bench_return(bench: pd.DataFrame | None, entry_date, exit_date) -> float:
    """Benchmark open(entry_date) -> close(exit_date). NaN if dates are missing."""
    if bench is None or bench.empty:
        return float("nan")
    try:
        return float(bench.loc[exit_date, "Close"] / bench.loc[entry_date, "Open"] - 1.0)
    except KeyError:
        return float("nan")


def evaluate_signals(
    history: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    bench: pd.DataFrame | None = None,
    hold_days: int = 5,
    cost_bps: float = 10.0,
    long_only: bool = False,
):
    """Turn logged BUY/SELL signals into simulated trades.
    Returns (trades_df, n_still_open). Trades whose exit date hasn't happened yet are
    skipped and counted in n_still_open."""
    wanted = ["BUY"] if long_only else ["BUY", "SELL"]
    rows, still_open = [], 0
    for r in history[history["signal"].isin(wanted)].itertuples(index=False):
        px = prices.get(r.ticker)
        if px is None or px.empty:
            continue
        idx = px.index
        entry_i = int(idx.searchsorted(pd.Timestamp(r.run_date), side="right"))  # first day AFTER signal
        exit_i = entry_i + hold_days - 1
        if exit_i >= len(px):
            still_open += 1
            continue
        entry_px = float(px["Open"].iloc[entry_i])
        exit_px = float(px["Close"].iloc[exit_i])
        gross = exit_px / entry_px - 1.0
        side = 1 if r.signal == "BUY" else -1
        net = side * gross - 2.0 * cost_bps / 1e4
        bench_ret = _bench_return(bench, idx[entry_i], idx[exit_i])
        rows.append({
            "signal_date": r.run_date,
            "ticker": r.ticker,
            "signal": r.signal,
            "sentiment_score": getattr(r, "sentiment_score", np.nan),
            "entry_date": idx[entry_i].strftime("%Y-%m-%d"),
            "exit_date": idx[exit_i].strftime("%Y-%m-%d"),
            "entry_open": entry_px,
            "exit_close": exit_px,
            "stock_return": gross,
            "net_return": net,
            "bench_return": bench_ret,
            "excess_return": net - side * bench_ret if np.isfinite(bench_ret) else float("nan"),
        })
    return pd.DataFrame(rows), still_open


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """95% confidence interval for the mean, by resampling the trades with replacement."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(trades: pd.DataFrame, still_open: int = 0) -> dict:
    """Headline numbers, all as percentages."""
    if trades.empty:
        return {"n_trades": 0, "still_open": still_open}
    pct = lambda s: float(s.mean() * 100)  # noqa: E731
    net_lo, net_hi = bootstrap_ci(trades["net_return"])
    ex_lo, ex_hi = bootstrap_ci(trades["excess_return"])
    buys, sells = trades[trades["signal"] == "BUY"], trades[trades["signal"] == "SELL"]
    return {
        "n_trades": int(len(trades)),
        "n_buy": int(len(buys)),
        "n_sell": int(len(sells)),
        "still_open": int(still_open),
        "hit_rate_pct": float((trades["net_return"] > 0).mean() * 100),
        "avg_net_return_pct": pct(trades["net_return"]),
        "median_net_return_pct": float(trades["net_return"].median() * 100),
        "net_return_ci95_pct": [net_lo * 100, net_hi * 100],
        "avg_excess_vs_benchmark_pct": float(trades["excess_return"].mean() * 100),
        "excess_ci95_pct": [ex_lo * 100, ex_hi * 100],
        # "What if I'd just bought every one of these stocks on those days?"
        "avg_buy_and_hold_same_stocks_pct": pct(trades["stock_return"]),
        "avg_buy_signal_return_pct": pct(buys["net_return"]) if len(buys) else None,
        "avg_sell_signal_return_pct": pct(sells["net_return"]) if len(sells) else None,
        "best_trade_pct": float(trades["net_return"].max() * 100),
        "worst_trade_pct": float(trades["net_return"].min() * 100),
        "first_signal": str(trades["signal_date"].min()),
        "last_signal": str(trades["signal_date"].max()),
    }


def format_report(s: dict, hold_days: int, cost_bps: float) -> str:
    if s.get("n_trades", 0) == 0:
        extra = f" ({s['still_open']} signals haven't reached their exit date yet)" if s.get("still_open") else ""
        return f"No completed trades to score yet{extra}. Keep running the pipeline, or use --backfill."
    lines = [
        f"Trades scored: {s['n_trades']}  ({s['n_buy']} BUY, {s['n_sell']} SELL), "
        f"held {hold_days} trading days, {cost_bps:g} bps cost each way",
        f"Signals from {s['first_signal']} to {s['last_signal']}"
        + (f"  ({s['still_open']} still open, not counted)" if s["still_open"] else ""),
        f"Hit rate:                 {s['hit_rate_pct']:.0f}%",
        f"Average net return:       {s['avg_net_return_pct']:+.2f}%  (95% CI {s['net_return_ci95_pct'][0]:+.2f}% to {s['net_return_ci95_pct'][1]:+.2f}%)",
        f"Average vs SPY:           {s['avg_excess_vs_benchmark_pct']:+.2f}%  (95% CI {s['excess_ci95_pct'][0]:+.2f}% to {s['excess_ci95_pct'][1]:+.2f}%)",
        f"Just buying the same stocks on the same days: {s['avg_buy_and_hold_same_stocks_pct']:+.2f}%",
    ]
    if s["avg_buy_signal_return_pct"] is not None:
        lines.append(f"  BUY signals averaged:   {s['avg_buy_signal_return_pct']:+.2f}%")
    if s["avg_sell_signal_return_pct"] is not None:
        lines.append(f"  SELL signals averaged:  {s['avg_sell_signal_return_pct']:+.2f}%")
    lines.append(f"Best / worst trade:       {s['best_trade_pct']:+.2f}% / {s['worst_trade_pct']:+.2f}%")
    lines.append("")
    if s["n_trades"] < 30:
        lines.append(f"CAUTION: only {s['n_trades']} trades. That is far too few to conclude anything. "
                     "Treat these numbers as a smoke test, not evidence.")
    lo, hi = s["excess_ci95_pct"]
    if np.isfinite(lo) and lo <= 0 <= hi:
        lines.append("The confidence interval for 'vs SPY' includes zero, so this result can't be told apart from luck.")
    lines.append("Trades on the same day are correlated (the whole market moves together), so the real uncertainty is larger than shown.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Glue that needs the network / models
# --------------------------------------------------------------------------
def backfill(cfg: T.Config, news_key: str, days: int) -> pd.DataFrame:
    """Rebuild one day of signals per past day, using only information available that day."""
    today = datetime.now(timezone.utc).date()
    # NewsAPI's free plan delays articles by ~24h, so stop two days back.
    day_list = [today - timedelta(days=k) for k in range(days + 1, 1, -1)]

    log.info("Backfill 1/4: fetching %d days of headlines (one request per day)...", len(day_list))
    frames = []
    for d in day_list:
        try:
            df = T.fetch_news(news_key, 1, start=d.isoformat(), end=(d + timedelta(days=1)).isoformat())
        except RuntimeError as exc:
            log.warning("  %s skipped: %s", d, exc)
            continue
        df["day"] = pd.to_datetime(df["published_at"], utc=True).dt.date
        frames.append(df[df["day"] == d])
    if not frames:
        raise SystemExit("No headlines could be fetched for the backfill window.")
    news = pd.concat(frames, ignore_index=True).drop_duplicates(subset="title").reset_index(drop=True)

    log.info("Backfill 2/4: finding companies and scoring sentiment on %d headlines...", len(news))
    sp500 = T.load_sp500()
    matcher = T.TickerMatcher(sp500)
    news = T.extract_companies(news, matcher)
    hits = news[news["tickers"].map(len) > 0]
    if hits.empty:
        raise SystemExit("No S&P 500 companies found in the backfill headlines.")
    scored = hits.join(T.score_headlines(hits["title"]))

    per_day = {}
    for d, grp in scored.groupby("day"):
        cand = T.select_candidates(T.aggregate_sentiment(grp), cfg)
        if not cand.empty:
            per_day[d] = cand

    log.info("Backfill 3/4: downloading prices for %d tickers...", len({t for c in per_day.values() for t in c["ticker"]}))
    prices = {}
    for t in sorted({t for c in per_day.values() for t in c["ticker"]}):
        px = T.get_daily_prices(t)
        if not px.empty:
            prices[t] = px

    log.info("Backfill 4/4: generating signals as of each day (trend forecast only)...")
    out = []
    for d, cand in sorted(per_day.items()):
        asof = {t: prices[t].loc[: pd.Timestamp(d)] for t in cand["ticker"] if t in prices}
        sig, _, _ = T.build_signals(cand, asof, cfg, nixtla_key=None, company_by_ticker=matcher.company_by_ticker)
        if not sig.empty:
            sig["run_date"] = d.isoformat()
            sig["forecast_source"] = sig.get("forecast_source", pd.Series(dtype=str)).fillna("backfill-drift")
            out.append(sig)
    if not out:
        raise SystemExit("Backfill produced no signals. Try --days with a larger window or lower --min-mentions.")
    return T.append_history(pd.concat(out, ignore_index=True), HISTORY_PATH)


def run_backtest(hold_days: int, cost_bps: float, long_only: bool, benchmark: str):
    if not HISTORY_PATH.exists():
        raise SystemExit("No signals_history.csv yet. Run `python src/tickerwire.py` first, or use --backfill 25.")
    history = pd.read_csv(HISTORY_PATH)
    tickers = sorted(history.loc[history["signal"].isin(["BUY", "SELL"]), "ticker"].unique())
    if not tickers:
        raise SystemExit("History has no BUY/SELL signals to score yet.")
    log.info("Downloading prices for %d tickers + %s benchmark...", len(tickers), benchmark)
    prices = {t: T.get_daily_prices(t) for t in tickers}
    bench = T.get_daily_prices(benchmark)

    trades, still_open = evaluate_signals(history, prices, bench, hold_days, cost_bps, long_only)
    summary = summarize(trades, still_open)
    summary.update(hold_days=hold_days, cost_bps=cost_bps, long_only=long_only, benchmark=benchmark,
                   generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
    trades.to_csv(TRADES_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + format_report(summary, hold_days, cost_bps))
    return summary


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(T.ROOT / ".env")
    except ImportError:
        pass
    d = T.Config()
    p = argparse.ArgumentParser(description="Score Tickerwire signals against realised prices")
    p.add_argument("--hold-days", type=int, default=5, help="trading days to hold each trade")
    p.add_argument("--cost-bps", type=float, default=10.0, help="trading cost each way, in basis points")
    p.add_argument("--long-only", action="store_true", help="ignore SELL signals (no shorting)")
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--backfill", type=int, default=0, metavar="N", help="rebuild signals for the last N days first")
    p.add_argument("--buy-threshold", type=float, default=d.buy_threshold)
    p.add_argument("--sell-threshold", type=float, default=d.sell_threshold)
    p.add_argument("--min-mentions", type=int, default=d.min_mentions)
    a = p.parse_args(argv)

    if a.backfill:
        cfg = T.Config(buy_threshold=a.buy_threshold, sell_threshold=a.sell_threshold, min_mentions=a.min_mentions)
        backfill(cfg, os.getenv("NEWSAPI_KEY", ""), a.backfill)
    run_backtest(a.hold_days, a.cost_bps, a.long_only, a.benchmark)


if __name__ == "__main__":
    sys.exit(main())
