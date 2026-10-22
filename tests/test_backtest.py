"""Tests for the backtest engine. All synthetic data, no network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtest as B  # noqa: E402
import tickerwire as T  # noqa: E402


def make_px(opens, closes, start="2024-03-04"):
    idx = pd.bdate_range(start, periods=len(opens))
    return pd.DataFrame({"Open": opens, "High": np.maximum(opens, closes), "Low": np.minimum(opens, closes),
                         "Close": closes, "Volume": 1}, index=idx)


def hist(rows):
    return pd.DataFrame(rows, columns=["run_date", "ticker", "signal", "sentiment_score"])


def test_entry_is_next_day_open_not_signal_day_close():
    # Signal on Mon 2024-03-04. The close that day is a huge 200; entry must be Tue's OPEN (100).
    px = make_px(opens=[100, 100, 100, 100, 100, 100, 100], closes=[200, 110, 110, 110, 110, 110, 110])
    trades, _ = B.evaluate_signals(hist([("2024-03-04", "XYZ", "BUY", 0.9)]), {"XYZ": px}, None,
                                   hold_days=3, cost_bps=0)
    t = trades.iloc[0]
    assert t["entry_date"] == "2024-03-05"
    assert t["entry_open"] == 100 and t["exit_close"] == 110
    assert t["stock_return"] == pytest.approx(0.10)       # NOT 200 -> 110 = -45%


def test_exit_date_is_hold_days_after_entry():
    px = make_px(opens=list(range(100, 110)), closes=list(range(101, 111)))
    trades, _ = B.evaluate_signals(hist([("2024-03-04", "XYZ", "BUY", 0.9)]), {"XYZ": px}, None, hold_days=5)
    assert trades.iloc[0]["entry_date"] == "2024-03-05"
    assert trades.iloc[0]["exit_date"] == "2024-03-11"    # 5 trading days: Tue..Mon


def test_sell_profits_when_price_falls_and_costs_apply():
    px = make_px(opens=[100] * 6, closes=[100, 90, 90, 90, 90, 90])
    h = hist([("2024-03-04", "XYZ", "SELL", 0.1)])
    free, _ = B.evaluate_signals(h, {"XYZ": px}, None, hold_days=2, cost_bps=0)
    paid, _ = B.evaluate_signals(h, {"XYZ": px}, None, hold_days=2, cost_bps=10)
    assert free.iloc[0]["net_return"] == pytest.approx(0.10)          # short a 10% drop
    assert paid.iloc[0]["net_return"] == pytest.approx(0.10 - 0.002)  # 10 bps each way


def test_unfinished_trades_are_counted_not_scored():
    px = make_px(opens=[100] * 4, closes=[100] * 4)
    trades, still_open = B.evaluate_signals(hist([("2024-03-06", "XYZ", "BUY", 0.9)]), {"XYZ": px}, None, hold_days=5)
    assert trades.empty and still_open == 1


def test_holds_and_long_only_filters():
    px = make_px(opens=[100] * 8, closes=[100] * 8)
    h = hist([("2024-03-04", "XYZ", "HOLD", 0.5), ("2024-03-04", "XYZ", "SELL", 0.1), ("2024-03-04", "XYZ", "BUY", 0.9)])
    both, _ = B.evaluate_signals(h, {"XYZ": px}, None, hold_days=2)
    long, _ = B.evaluate_signals(h, {"XYZ": px}, None, hold_days=2, long_only=True)
    assert set(both["signal"]) == {"BUY", "SELL"}
    assert set(long["signal"]) == {"BUY"}


def test_benchmark_excess_for_long_and_short():
    px = make_px(opens=[100] * 6, closes=[100, 110, 110, 110, 110, 110])      # stock +10%
    bench = make_px(opens=[400] * 6, closes=[400, 408, 408, 408, 408, 408])   # SPY +2%
    h = hist([("2024-03-04", "XYZ", "BUY", 0.9), ("2024-03-04", "XYZ", "SELL", 0.1)])
    t, _ = B.evaluate_signals(h, {"XYZ": px}, bench, hold_days=2, cost_bps=0)
    t = t.set_index("signal")
    assert t.loc["BUY", "excess_return"] == pytest.approx(0.10 - 0.02)
    assert t.loc["SELL", "excess_return"] == pytest.approx(-0.10 + 0.02)   # shorting the stock vs shorting SPY


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    x = np.random.default_rng(0).normal(0.01, 0.03, 50)
    lo, hi = B.bootstrap_ci(x)
    assert (lo, hi) == B.bootstrap_ci(x)
    assert lo < x.mean() < hi
    assert all(np.isnan(B.bootstrap_ci([0.1])))


def test_summarize_and_report_flags_small_samples():
    px = make_px(opens=[100] * 8, closes=[100, 105, 105, 105, 105, 105, 105, 105])
    h = hist([("2024-03-04", "XYZ", "BUY", 0.9), ("2024-03-05", "XYZ", "BUY", 0.9)])
    trades, op = B.evaluate_signals(h, {"XYZ": px}, None, hold_days=2, cost_bps=0)
    s = B.summarize(trades, op)
    assert s["n_trades"] == 2 and s["n_buy"] == 2 and s["n_sell"] == 0
    assert s["hit_rate_pct"] == 100.0
    text = B.format_report(s, 2, 0)
    assert "CAUTION" in text and "too few" in text
    assert "No completed trades" in B.format_report(B.summarize(pd.DataFrame(), 3), 2, 0)


def test_append_history_dedupes_same_day(tmp_path):
    path = tmp_path / "h.csv"
    a = pd.DataFrame({"run_date": ["2024-03-04"], "ticker": ["AAPL"], "signal": ["BUY"]})
    b = pd.DataFrame({"run_date": ["2024-03-04", "2024-03-05"], "ticker": ["AAPL", "AAPL"], "signal": ["HOLD", "SELL"]})
    T.append_history(a, path)
    out = T.append_history(b, path)
    assert len(out) == 2
    assert out[out["run_date"] == "2024-03-04"]["signal"].iloc[0] == "HOLD"    # re-run overwrote


def test_select_candidates_ranks_by_decisiveness():
    s = pd.DataFrame({"ticker": ["A", "B", "C", "D"], "sentiment_score": [0.55, 0.95, 0.05, 0.9],
                      "mentions": [5, 3, 3, 1], "sentiment_label": ["NEUTRAL"] * 4})
    c = T.select_candidates(s, T.Config(min_mentions=2, max_tickers=2))
    assert list(c["ticker"]) in (["B", "C"], ["C", "B"])      # D dropped for low mentions, A is least decisive
