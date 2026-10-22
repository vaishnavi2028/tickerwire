"""Tests for the parts of Tickerwire that don't need the network or the big ML models."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tickerwire as S  # noqa: E402
from qa import answer_question  # noqa: E402


# ---------- helpers ----------
def fake_prices(n=300, seed=0, drift=0.0005, vol=0.01):
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000},
        index=idx,
    )


# ---------- stage 2 ----------
def test_clean_name_strips_suffixes():
    assert S.clean_name("Apple Inc.") == "apple"
    assert S.clean_name("The Coca-Cola Company") == "coca cola"
    assert S.clean_name("Alphabet Inc. (Class A)") == "alphabet"


def test_ticker_matcher():
    sp = pd.DataFrame({"ticker": ["AAPL", "MSFT", "KO"],
                       "company": ["Apple Inc.", "Microsoft Corporation", "The Coca-Cola Company"]})
    m = S.TickerMatcher(sp)
    assert m.match("Apple") == "AAPL"
    assert m.match("Microsoft") == "MSFT"
    assert m.match("Coca-Cola") == "KO"
    assert m.match("Google") == "GOOGL"          # alias
    assert m.match("Microsft") == "MSFT"         # fuzzy: one missing letter
    assert m.match("Appel") is None              # too different: strict cutoff avoids false matches
    assert m.match("Reuters") is None
    assert m.match("Fed") is None                # too short / no match


# ---------- stage 3 ----------
def test_sentiment_score_range():
    assert S.sentiment_score(1.0, 0.0) == 1.0
    assert S.sentiment_score(0.0, 1.0) == 0.0
    assert S.sentiment_score(0.0, 0.0) == 0.5
    assert S.label_from_score(0.9) == "POSITIVE"
    assert S.label_from_score(0.1) == "NEGATIVE"
    assert S.label_from_score(0.5) == "NEUTRAL"


def test_aggregate_sentiment():
    df = pd.DataFrame({
        "title": ["a", "b", "c"],
        "tickers": [["AAPL"], ["AAPL", "MSFT"], ["MSFT"]],
        "score": [0.9, 0.7, 0.2],
        "p_pos": [0.9, 0.7, 0.1],
        "p_neg": [0.0, 0.1, 0.6],
    })
    out = S.aggregate_sentiment(df).set_index("ticker")
    assert out.loc["AAPL", "mentions"] == 2
    assert out.loc["AAPL", "sentiment_score"] == pytest.approx(0.8)
    assert out.loc["MSFT", "sentiment_score"] == pytest.approx(0.45)


def test_keyword_frequency_ignores_stopwords():
    kw = S.keyword_frequency(["Apple earnings beat", "Apple earnings miss"])
    d = dict(zip(kw["keyword"], kw["count"]))
    assert d["apple"] == 2 and d["earnings"] == 2 and "the" not in d


# ---------- stage 4 ----------
def test_resample_to_daily():
    idx = pd.date_range("2024-03-04 09:30", periods=6, freq="5min").append(
        pd.date_range("2024-03-05 09:30", periods=6, freq="5min"))
    df = pd.DataFrame({"Open": range(12), "High": range(12), "Low": range(12),
                       "Close": range(12), "Volume": 10}, index=idx)
    d = S.resample_to_daily(df)
    assert len(d) == 2
    assert d["Open"].iloc[0] == 0 and d["Close"].iloc[0] == 5 and d["Volume"].iloc[0] == 60


def test_volatility_and_position_size():
    close = fake_prices(vol=0.01)["Close"]
    v = S.annualized_vol(close)
    assert 0.10 < v < 0.22                         # 1% daily is roughly 16% annual
    assert S.position_size_pct(0.25, 0.25, 0.10) == 10.0
    assert S.position_size_pct(0.50, 0.25, 0.10) == 5.0
    assert S.position_size_pct(0.10, 0.25, 0.10) == 10.0   # capped
    assert S.position_size_pct(float("nan"), 0.25, 0.10) == 0.0


# ---------- stage 5 ----------
def test_prepare_series_fills_holidays():
    px = fake_prices(100).drop(pd.Timestamp("2024-02-14"))
    s = S.prepare_series(px, 60)
    assert len(s) == 60
    assert pd.infer_freq(s["ds"]) == "B"
    assert not s["y"].isna().any()


def test_drift_forecast_shape_and_band_order():
    s = S.prepare_series(fake_prices(100), 60)
    fc = S.forecast_drift(s, 7)
    assert len(fc) == 7
    assert (fc["lo_95"] <= fc["lo_80"]).all() and (fc["lo_80"] <= fc["y_hat"]).all()
    assert (fc["y_hat"] <= fc["hi_80"]).all() and (fc["hi_80"] <= fc["hi_95"]).all()
    assert fc["ds"].iloc[0] > s["ds"].iloc[-1]
    assert (fc["hi_95"] - fc["lo_95"]).is_monotonic_increasing   # uncertainty grows


def test_forecast_direction():
    fc = pd.DataFrame({"y_hat": [101.0, 103.0]})
    assert S.forecast_direction(100.0, fc, 0.1)[0] == "UP"
    assert S.forecast_direction(105.0, fc, 0.1)[0] == "DOWN"
    assert S.forecast_direction(103.01, fc, 0.1)[0] == "FLAT"


# ---------- stage 6 ----------
def test_superposed_epoch_finds_planted_spikes():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.005, 400)
    for i in (50, 120, 200, 300):                  # plant +8% days
        rets[i] = 0.08
    close = pd.Series(100 * np.exp(np.cumsum(rets)), index=pd.bdate_range("2023-01-02", periods=400))
    ep = S.superposed_epoch(close, window=5, z_thresh=2.0)
    up = ep[ep["direction"] == "up"]
    assert up["n_events"].iloc[0] >= 4
    day0 = up[up["rel_day"] == 0]["mean_pct"].iloc[0]
    day_minus1 = up[up["rel_day"] == -1]["mean_pct"].iloc[0]
    assert day0 == pytest.approx(0.0)              # normalised to the event day
    assert day_minus1 < -5                         # price was ~8% lower the day before


def test_epoch_confirms_logic():
    ep = pd.DataFrame({"direction": ["up", "down"], "rel_day": [5, 5],
                       "mean_pct": [1.2, 0.8], "n_events": [4, 4]})
    assert S.epoch_confirms(ep, "BUY") is True
    assert S.epoch_confirms(ep, "SELL") is False   # down spikes bounced back
    assert S.epoch_confirms(ep, "HOLD") is None
    assert S.epoch_confirms(pd.DataFrame(), "BUY") is None


# ---------- stage 7 ----------
@pytest.mark.parametrize("score,direction,expected", [
    (0.90, "UP", "BUY"),
    (0.90, "DOWN", "HOLD"),
    (0.10, "DOWN", "SELL"),
    (0.10, "UP", "HOLD"),
    (0.50, "UP", "HOLD"),
    (0.90, "FLAT", "HOLD"),
])
def test_decide_signal(score, direction, expected):
    assert S.decide_signal(score, direction, 0.70, 0.30)[0] == expected


def test_risk_levels():
    fc = S.forecast_drift(S.prepare_series(fake_prices(100), 60), 7)
    stop, target = S.risk_levels("BUY", fc)
    assert stop == fc["lo_95"].min() and target == fc["hi_80"].max()
    assert S.risk_levels("HOLD", fc) == (None, None)


def test_build_signals_end_to_end_without_network():
    cfg = S.Config()
    cand = pd.DataFrame({
        "ticker": ["UPCO", "MEH", "NOPX"],
        "sentiment_score": [0.95, 0.50, 0.05],
        "sentiment_label": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
        "mentions": [5, 4, 3],
    })
    prices = {"UPCO": fake_prices(300, seed=3, drift=0.003), "MEH": fake_prices(300, seed=4)}
    signals, forecasts, epochs = S.build_signals(cand, prices, cfg, nixtla_key=None)
    s = signals.set_index("ticker")
    assert s.loc["UPCO", "signal"] == "BUY"             # strong trend up + positive news
    assert s.loc["UPCO", "forecast_source"] == "drift-fallback"
    assert s.loc["UPCO", "position_size_pct"] > 0
    assert s.loc["MEH", "signal"] == "HOLD" and s.loc["MEH", "reason"] == "sentiment not decisive"
    assert s.loc["NOPX", "signal"] == "HOLD" and "price history" in s.loc["NOPX", "reason"]
    assert set(forecasts["ticker"]) == {"UPCO"}


# ---------- Q&A ----------
def test_qa_answers():
    summary = pd.DataFrame({"ticker": ["AAPL", "TSLA"], "sentiment_score": [0.9, 0.2],
                            "sentiment_label": ["POSITIVE", "NEGATIVE"], "mentions": [5, 3]})
    signals = pd.DataFrame({"ticker": ["AAPL"], "signal": ["BUY"], "reason": ["ok"]})
    vol = pd.DataFrame({"ticker": ["AAPL", "TSLA"], "volatility_ann": [0.25, 0.60],
                        "return_ann": [0.15, 0.05], "last_close": [180.0, 240.0]})
    assert "AAPL" in answer_question("highest sentiment?", summary, signals, vol)
    assert "TSLA" in answer_question("which is the most volatile stock?", summary, signals, vol)
    assert "15.0%" in answer_question("return of AAPL", summary, signals, vol)
    assert "BUY" in answer_question("signal for AAPL", summary, signals, vol)
    assert "AAPL" in answer_question("which stocks to buy?", summary, signals, vol)
    assert "no data" in answer_question("hi", pd.DataFrame(), signals, vol).lower()
