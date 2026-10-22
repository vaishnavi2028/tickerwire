"""
Tickerwire: a news-driven trading signal pipeline.

Seven stages, run in order:
  1. Fetch financial headlines (NewsAPI)
  2. Pull company names out of them (spaCy) and map to S&P 500 tickers
  3. Score each headline's sentiment (FinBERT)
  4. Download price history (Yahoo Finance)
  5. Forecast prices (Nixtla TimeGPT, with a simple fallback)
  6. Superposed epoch analysis (what usually happens around price spikes)
  7. Combine everything into BUY / SELL / HOLD

Run it with:  python src/tickerwire.py
This is a learning project. Nothing here is financial advice.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("tickerwire")

# --------------------------------------------------------------------------
# Paths and constants
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "analysis" / "data"
PRICE_DIR = DATA_DIR / "historical_price_data"

NEWS_URL = "https://newsapi.org/v2/everything"
SP500_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/main/data/constituents.csv"
)
FINBERT_MODEL = "yiyanghkust/finbert-tone"

NEWS_DOMAINS = ",".join([
    "bloomberg.com", "wsj.com", "cnbc.com", "reuters.com", "marketwatch.com",
    "seekingalpha.com", "barrons.com", "forbes.com", "fortune.com",
    "techcrunch.com", "businessinsider.com", "finance.yahoo.com",
    "investopedia.com", "fool.com", "edition.cnn.com",
])
NEWS_QUERY = (
    '(earnings OR merger OR acquisition OR IPO OR "Federal Reserve" OR '
    'guidance OR revenue OR buyback OR upgrade OR downgrade OR "analyst rating")'
)

LEVELS = (80, 90, 95)  # confidence bands for forecasts
Z_SCORES = {80: 1.2816, 90: 1.6449, 95: 1.9600}

# Hand-written shortcuts for names spaCy returns that don't match the official
# S&P 500 company name (Google -> Alphabet, Facebook -> Meta, and so on).
ALIASES = {
    "google": "GOOGL", "alphabet": "GOOGL", "facebook": "META", "meta": "META",
    "meta platforms": "META", "amazon": "AMZN", "jpmorgan": "JPM",
    "jp morgan": "JPM", "jpmorgan chase": "JPM", "disney": "DIS",
    "walt disney": "DIS", "berkshire": "BRK-B", "berkshire hathaway": "BRK-B",
    "nvidia": "NVDA", "tesla": "TSLA", "microsoft": "MSFT", "apple": "AAPL",
    "goldman sachs": "GS", "morgan stanley": "MS", "bank of america": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "citi": "C",
}

STOPWORDS = set("""
a an and are as at be by for from has have in is it its of on or that the to
was were will with after over new says said stock stocks shares share than
this amid into more up down out off about how why what who vs us
""".split())


@dataclass
class Config:
    news_days: int = 3            # how many days of headlines to pull
    buy_threshold: float = 0.70   # average sentiment needed for BUY
    sell_threshold: float = 0.30  # average sentiment needed for SELL
    min_mentions: int = 2         # ignore tickers mentioned fewer times
    max_tickers: int = 10         # cap, keeps API calls and runtime sane
    horizon: int = 7              # forecast this many trading days ahead
    lookback_bars: int = 60       # trading days of history sent to forecaster
    epoch_window: int = 5         # days before/after a spike to examine
    spike_z: float = 2.0          # a "spike" = daily return beyond 2 std devs
    flat_move_pct: float = 0.1    # forecast moves under 0.1% count as FLAT
    require_epoch: bool = False   # downgrade to HOLD if epoch analysis disagrees
    target_vol: float = 0.25      # volatility that earns the full position size
    max_position: float = 0.10    # biggest allowed position (10% of portfolio)


# --------------------------------------------------------------------------
# Stage 1: news
# --------------------------------------------------------------------------
NEWS_COLUMNS = ["published_at", "source", "title", "description", "url"]


def fetch_news(api_key: str, days: int, page_size: int = 100,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Pull recent headlines from the finance outlets listed in NEWS_DOMAINS.
    `start`/`end` (YYYY-MM-DD) override `days` and are used by the backtest backfill."""
    if not api_key:
        raise SystemExit(
            "NEWSAPI_KEY is not set. Copy .env.example to .env and add your key "
            "(free at https://newsapi.org)."
        )
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "q": NEWS_QUERY,
        "domains": NEWS_DOMAINS,
        "from": start or since,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }
    if end:
        params["to"] = end
    resp = requests.get(NEWS_URL, params=params, timeout=30)
    payload = resp.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error: {payload.get('message', resp.text)}")

    rows = [
        {
            "published_at": a.get("publishedAt"),
            "source": (a.get("source") or {}).get("name"),
            "title": a.get("title"),
            "description": a.get("description"),
            "url": a.get("url"),
        }
        for a in payload.get("articles", [])
    ]
    df = pd.DataFrame(rows, columns=NEWS_COLUMNS)
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip().ne("[Removed]")]
    return df.drop_duplicates(subset="title").reset_index(drop=True)


# --------------------------------------------------------------------------
# Stage 2: company names -> tickers
# --------------------------------------------------------------------------
def clean_name(name: str) -> str:
    """Lowercase and strip legal suffixes so 'Apple Inc.' and 'Apple' compare equal."""
    name = name.lower()
    name = re.sub(r"\(.*?\)", " ", name)
    name = re.sub(r"\bclass [a-c]\b", " ", name)
    name = re.sub(r"[^a-z0-9& ]", " ", name)
    name = re.sub(
        r"\b(inc|corp|corporation|co|company|ltd|plc|holdings|holding|group|"
        r"incorporated|limited|the|nv|sa)\b",
        " ",
        name,
    )
    return re.sub(r"\s+", " ", name).strip()


def load_sp500(cache_path: Path | None = None) -> pd.DataFrame:
    """Return a DataFrame with columns [ticker, company]. Cached on disk."""
    cache_path = cache_path or (DATA_DIR / "sp500_constituents.csv")
    if cache_path.exists():
        raw = pd.read_csv(cache_path)
    else:
        raw = pd.read_csv(SP500_URL)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(cache_path, index=False)
    name_col = "Security" if "Security" in raw.columns else "Name"
    out = pd.DataFrame({"ticker": raw["Symbol"], "company": raw[name_col]})
    out["ticker"] = out["ticker"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (Yahoo style)
    return out


class TickerMatcher:
    """Maps a company name found in a headline to an S&P 500 ticker."""

    def __init__(self, sp500: pd.DataFrame, cutoff: int = 90):
        from rapidfuzz import fuzz, process  # imported here so tests of other code stay light

        self._fuzz, self._process = fuzz, process
        self.cutoff = cutoff
        self.names: dict[str, str] = {}
        for ticker, company in zip(sp500["ticker"], sp500["company"]):
            self.names.setdefault(clean_name(company), ticker)
        self.company_by_ticker = dict(zip(sp500["ticker"], sp500["company"]))
        self._choices = list(self.names.keys())
        self._cache: dict[str, str | None] = {}

    def match(self, org: str) -> str | None:
        key = clean_name(org)
        if len(key) < 3:
            return None
        if key in self._cache:
            return self._cache[key]
        ticker = ALIASES.get(key) or self.names.get(key)
        if ticker is None:
            hit = self._process.extractOne(
                key, self._choices, scorer=self._fuzz.ratio, score_cutoff=self.cutoff
            )
            ticker = self.names[hit[0]] if hit else None
        self._cache[key] = ticker
        return ticker


def extract_companies(news: pd.DataFrame, matcher: TickerMatcher) -> pd.DataFrame:
    """Run spaCy NER over headlines; add `orgs` and `tickers` list columns."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        raise SystemExit("Missing spaCy model. Run: python -m spacy download en_core_web_sm")

    orgs_col, tickers_col = [], []
    for doc in nlp.pipe(news["title"].tolist(), batch_size=64):
        orgs = list(dict.fromkeys(e.text for e in doc.ents if e.label_ == "ORG"))
        tickers = []
        for org in orgs:
            t = matcher.match(org)
            if t and t not in tickers:
                tickers.append(t)
        orgs_col.append(orgs)
        tickers_col.append(tickers)
    news = news.copy()
    news["orgs"] = orgs_col
    news["tickers"] = tickers_col
    return news


# --------------------------------------------------------------------------
# Stage 3: FinBERT sentiment
# --------------------------------------------------------------------------
def sentiment_score(p_pos, p_neg):
    """Collapse FinBERT's probabilities into one 0-1 number.
    0.0 = confidently negative, 0.5 = neutral/mixed, 1.0 = confidently positive."""
    return (p_pos - p_neg + 1.0) / 2.0


def label_from_score(score: float) -> str:
    if score >= 0.6:
        return "POSITIVE"
    if score <= 0.4:
        return "NEGATIVE"
    return "NEUTRAL"


def score_headlines(texts: pd.Series, batch_size: int = 16) -> pd.DataFrame:
    """Run FinBERT over a series of headlines. Returns p_neg, p_neu, p_pos, score, label."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from transformers import BertTokenizer
    from transformers import BertConfig, BertForSequenceClassification

    tok = BertTokenizer.from_pretrained(FINBERT_MODEL)

    config = BertConfig.from_pretrained(FINBERT_MODEL, num_labels=3)
    model = BertForSequenceClassification.from_pretrained(
        FINBERT_MODEL, config=config
    ).eval()
    id2label = {int(i): str(l).lower() for i, l in model.config.id2label.items()}
    if not {"negative", "neutral", "positive"} <= set(id2label.values()):
        raise RuntimeError(f"Unexpected FinBERT labels: {id2label}")

    rows = []
    for i in range(0, len(texts), batch_size):
        batch = texts.iloc[i : i + batch_size].tolist()
        enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
        for p in probs:
            rows.append({id2label[j]: float(p[j]) for j in range(len(p))})

    out = pd.DataFrame(rows, index=texts.index).rename(
        columns={"negative": "p_neg", "neutral": "p_neu", "positive": "p_pos"}
    )
    out["score"] = sentiment_score(out["p_pos"], out["p_neg"])
    out["label"] = out[["p_neg", "p_neu", "p_pos"]].idxmax(axis=1).map(
        {"p_neg": "negative", "p_neu": "neutral", "p_pos": "positive"}
    )
    return out


def aggregate_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: average sentiment across every headline that mentions it."""
    exploded = scored.explode("tickers").dropna(subset=["tickers"])
    exploded = exploded.rename(columns={"tickers": "ticker"})
    summary = (
        exploded.groupby("ticker")
        .agg(
            sentiment_score=("score", "mean"),
            mentions=("title", "nunique"),
            avg_p_pos=("p_pos", "mean"),
            avg_p_neg=("p_neg", "mean"),
        )
        .reset_index()
    )
    summary["sentiment_label"] = summary["sentiment_score"].map(label_from_score)
    return summary.sort_values(["mentions", "sentiment_score"], ascending=False).reset_index(drop=True)


def keyword_frequency(titles, top: int = 50) -> pd.DataFrame:
    counts: Counter = Counter()
    for t in titles:
        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", str(t).lower())
        counts.update(w for w in words if w not in STOPWORDS)
    return pd.DataFrame(counts.most_common(top), columns=["keyword", "count"])


# --------------------------------------------------------------------------
# Stage 4: prices
# --------------------------------------------------------------------------
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Turn intraday bars (e.g. 5-minute) into one row per day."""
    daily = df.resample("D").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    return daily.dropna(subset=["Close"])


def download_prices(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance nests the ticker in the header
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return pd.DataFrame(columns=OHLCV)
    df = df[OHLCV].dropna()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df


def get_daily_prices(ticker: str) -> pd.DataFrame:
    """5 years of daily bars; falls back to resampled 5-minute bars if that fails."""
    df = download_prices(ticker, "5y", "1d")
    if df.empty:
        intraday = download_prices(ticker, "30d", "5m")
        if not intraday.empty:
            df = resample_to_daily(intraday)
    return df


def annualized_vol(close: pd.Series) -> float:
    rets = close.pct_change().dropna()
    return float(rets.std() * np.sqrt(252)) if len(rets) > 1 else float("nan")


def annualized_return(close: pd.Series) -> float:
    rets = close.pct_change().dropna()
    return float(rets.mean() * 252) if len(rets) else float("nan")


def position_size_pct(vol: float, target_vol: float, max_position: float) -> float:
    """Inverse-volatility sizing: calmer stocks get bigger positions, capped at max_position."""
    if not np.isfinite(vol) or vol <= 0:
        return 0.0
    return round(min(max_position, max_position * target_vol / vol) * 100, 2)


# --------------------------------------------------------------------------
# Stage 5: forecasting
# --------------------------------------------------------------------------
def prepare_series(daily: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Close prices on a regular business-day calendar (holidays forward-filled)."""
    s = daily["Close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    s = s[~s.index.duplicated()].sort_index().asfreq("B").ffill().tail(lookback)
    return s.rename("y").rename_axis("ds").reset_index()


def forecast_timegpt(series: pd.DataFrame, horizon: int, api_key: str) -> pd.DataFrame:
    from nixtla import NixtlaClient

    client = NixtlaClient(api_key=api_key)
    fc = client.forecast(
        df=series, h=horizon, freq="B", time_col="ds", target_col="y", level=list(LEVELS)
    )
    out = pd.DataFrame({"ds": pd.to_datetime(fc["ds"]), "y_hat": fc["TimeGPT"].to_numpy()})
    for lvl in LEVELS:
        out[f"lo_{lvl}"] = fc[f"TimeGPT-lo-{lvl}"].to_numpy()
        out[f"hi_{lvl}"] = fc[f"TimeGPT-hi-{lvl}"].to_numpy()
    return out


def forecast_drift(series: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Fallback: extend the recent average daily trend, with bands that widen over time."""
    y = series["y"].to_numpy(dtype=float)
    rets = np.diff(np.log(y))
    mu = rets.mean() if len(rets) else 0.0
    sigma = rets.std(ddof=1) if len(rets) > 1 else 0.0
    steps = np.arange(1, horizon + 1)
    mean = y[-1] * np.exp(mu * steps)
    spread = sigma * np.sqrt(steps)
    start = pd.Timestamp(series["ds"].iloc[-1]) + pd.offsets.BDay(1)
    out = pd.DataFrame({"ds": pd.bdate_range(start, periods=horizon), "y_hat": mean})
    for lvl, z in Z_SCORES.items():
        out[f"lo_{lvl}"] = mean * np.exp(-z * spread)
        out[f"hi_{lvl}"] = mean * np.exp(z * spread)
    return out


def forecast_ticker(series: pd.DataFrame, horizon: int, nixtla_key: str | None):
    """Try TimeGPT; fall back to the drift model. Returns (forecast_df, source_name)."""
    if nixtla_key:
        try:
            return forecast_timegpt(series, horizon, nixtla_key), "TimeGPT-1"
        except Exception as exc:  # network, quota, bad key...
            log.warning("TimeGPT failed (%s). Using drift fallback.", exc)
    return forecast_drift(series, horizon), "drift-fallback"


def forecast_direction(last_close: float, fc: pd.DataFrame, flat_pct: float):
    """Compare the end of the forecast to today's close. Returns (UP|DOWN|FLAT, pct_move)."""
    move = (float(fc["y_hat"].iloc[-1]) / last_close - 1.0) * 100
    if abs(move) < flat_pct:
        return "FLAT", move
    return ("UP" if move > 0 else "DOWN"), move


# --------------------------------------------------------------------------
# Stage 6: superposed epoch analysis
# --------------------------------------------------------------------------
def superposed_epoch(close: pd.Series, window: int = 5, z_thresh: float = 2.0) -> pd.DataFrame:
    """Find big one-day moves ("events"), line them all up on day 0, and average the
    price path around them. Returns rows: direction, rel_day, mean_pct, n_events."""
    ret = close.pct_change()
    z = (ret - ret.mean()) / ret.std()
    prices = close.to_numpy(dtype=float)
    rows = []
    for direction, mask in (("up", z > z_thresh), ("down", z < -z_thresh)):
        paths = [
            prices[i - window : i + window + 1] / prices[i] - 1.0
            for i in np.flatnonzero(mask.to_numpy())
            if i - window >= 0 and i + window < len(prices)
        ]
        if not paths:
            continue
        mean_path = np.mean(paths, axis=0) * 100
        for rel_day, val in zip(range(-window, window + 1), mean_path):
            rows.append(
                {"direction": direction, "rel_day": rel_day, "mean_pct": float(val), "n_events": len(paths)}
            )
    return pd.DataFrame(rows, columns=["direction", "rel_day", "mean_pct", "n_events"])


def epoch_confirms(epoch: pd.DataFrame, signal: str, window: int = 5) -> bool | None:
    """Did past spikes tend to keep going in the signal's direction? None = not enough events."""
    if epoch.empty or signal not in ("BUY", "SELL"):
        return None
    direction = "up" if signal == "BUY" else "down"
    sub = epoch[(epoch["direction"] == direction) & (epoch["rel_day"] == window)]
    if sub.empty:
        return None
    end_move = float(sub["mean_pct"].iloc[0])
    return end_move > 0 if signal == "BUY" else end_move < 0


# --------------------------------------------------------------------------
# Stage 7: signals
# --------------------------------------------------------------------------
def is_strong(score: float, buy: float, sell: float) -> bool:
    return score >= buy or score <= sell


def decide_signal(score: float, direction: str, buy: float, sell: float):
    """The core rule. Sentiment and forecast must agree, otherwise we sit out."""
    if score >= buy and direction == "UP":
        return "BUY", "positive news and forecast points up"
    if score <= sell and direction == "DOWN":
        return "SELL", "negative news and forecast points down"
    if score >= buy or score <= sell:
        return "HOLD", f"news and forecast disagree (forecast {direction})"
    return "HOLD", "sentiment not decisive"


def risk_levels(signal: str, fc: pd.DataFrame):
    """Use the forecast's confidence bands as rough stop-loss / take-profit levels."""
    if signal == "BUY":
        return float(fc["lo_95"].min()), float(fc["hi_80"].max())
    if signal == "SELL":
        return float(fc["hi_95"].max()), float(fc["lo_80"].min())
    return None, None


def build_signals(candidates, prices, cfg: Config, nixtla_key=None, company_by_ticker=None):
    """candidates: sentiment summary rows. prices: {ticker: daily OHLCV DataFrame}.
    Returns (signals_df, forecasts_df, epoch_df)."""
    company_by_ticker = company_by_ticker or {}
    sig_rows, fc_frames, epoch_frames = [], [], []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for _, r in candidates.iterrows():
        t, score = r["ticker"], float(r["sentiment_score"])
        base = {
            "ticker": t,
            "company": company_by_ticker.get(t, ""),
            "sentiment_score": round(score, 4),
            "sentiment_label": r["sentiment_label"],
            "mentions": int(r["mentions"]),
            "generated_at": now,
        }
        px = prices.get(t)
        if px is None or len(px) < max(20, cfg.lookback_bars // 2):
            sig_rows.append({**base, "signal": "HOLD", "reason": "not enough price history"})
            continue

        close = px["Close"]
        last_close = float(close.iloc[-1])
        vol = annualized_vol(close)
        base.update(
            last_close=round(last_close, 2),
            volatility_ann=round(vol, 4),
            position_size_pct=0.0,
        )

        if not is_strong(score, cfg.buy_threshold, cfg.sell_threshold):
            sig_rows.append({**base, "signal": "HOLD", "reason": "sentiment not decisive"})
            continue

        series = prepare_series(px, cfg.lookback_bars)
        fc, source = forecast_ticker(series, cfg.horizon, nixtla_key)
        direction, move = forecast_direction(last_close, fc, cfg.flat_move_pct)
        signal, reason = decide_signal(score, direction, cfg.buy_threshold, cfg.sell_threshold)

        epoch = superposed_epoch(close, cfg.epoch_window, cfg.spike_z)
        confirms = epoch_confirms(epoch, signal, cfg.epoch_window)
        if cfg.require_epoch and signal in ("BUY", "SELL") and confirms is False:
            signal, reason = "HOLD", "epoch analysis does not support the signal"

        stop, target = risk_levels(signal, fc)
        base["position_size_pct"] = (
            position_size_pct(vol, cfg.target_vol, cfg.max_position) if signal != "HOLD" else 0.0
        )
        sig_rows.append({
            **base,
            "forecast_end": round(float(fc["y_hat"].iloc[-1]), 2),
            "expected_move_pct": round(move, 2),
            "direction": direction,
            "signal": signal,
            "reason": reason,
            "stop_loss": None if stop is None else round(stop, 2),
            "take_profit": None if target is None else round(target, 2),
            "epoch_confirms": confirms,
            "forecast_source": source,
        })
        fc_frames.append(fc.assign(ticker=t, source=source))
        if not epoch.empty:
            epoch_frames.append(epoch.assign(ticker=t))

    signals = pd.DataFrame(sig_rows)
    forecasts = pd.concat(fc_frames, ignore_index=True) if fc_frames else pd.DataFrame()
    epochs = pd.concat(epoch_frames, ignore_index=True) if epoch_frames else pd.DataFrame()
    return signals, forecasts, epochs


def select_candidates(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Tickers with enough mentions, ranked by how decisive their sentiment is."""
    cand = summary[summary["mentions"] >= cfg.min_mentions].copy()
    cand["strength"] = (cand["sentiment_score"] - 0.5).abs()
    return cand.sort_values(["strength", "mentions"], ascending=False).head(cfg.max_tickers)


def append_history(signals: pd.DataFrame, path: Path | None = None) -> pd.DataFrame:
    """Add this run's signals to signals_history.csv so they can be scored later.
    One row per (run_date, ticker); re-running on the same day overwrites that day."""
    path = path or (DATA_DIR / "signals_history.csv")
    if path.exists():
        signals = pd.concat([pd.read_csv(path), signals], ignore_index=True)
    signals = signals.drop_duplicates(subset=["run_date", "ticker"], keep="last")
    signals = signals.sort_values(["run_date", "ticker"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(path, index=False)
    return signals


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run(cfg: Config, newsapi_key: str, nixtla_key: str | None) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PRICE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("[1/7] Fetching news (last %d days)...", cfg.news_days)
    news = fetch_news(newsapi_key, cfg.news_days)
    log.info("      %d unique headlines from %d sources", len(news), news["source"].nunique())
    if news.empty:
        raise SystemExit("No headlines came back. Try --days with a larger number.")

    log.info("[2/7] Finding companies in headlines...")
    sp500 = load_sp500()
    matcher = TickerMatcher(sp500)
    news = extract_companies(news, matcher)
    with_tickers = news[news["tickers"].map(len) > 0]
    log.info("      %d headlines mention an S&P 500 company", len(with_tickers))
    if with_tickers.empty:
        raise SystemExit("No S&P 500 companies found in today's headlines.")

    log.info("[3/7] Scoring sentiment with FinBERT (first run downloads the model)...")
    scores = score_headlines(with_tickers["title"])
    news = news.join(scores)
    summary = aggregate_sentiment(news.loc[with_tickers.index])

    news_out = news.copy()
    news_out["orgs"] = news_out["orgs"].map(", ".join)
    news_out["tickers"] = news_out["tickers"].map(",".join)
    news_out.to_csv(DATA_DIR / "news_headlines.csv", index=False)
    summary.to_csv(DATA_DIR / "sentiment_summary.csv", index=False)
    keyword_frequency(news["title"]).to_csv(DATA_DIR / "news_keyword_frequency.csv", index=False)

    cand = select_candidates(summary, cfg)
    if cand.empty:
        raise SystemExit(f"No ticker has {cfg.min_mentions}+ mentions. Try --min-mentions 1 or --days 7.")

    log.info("[4/7] Downloading prices for %d tickers...", len(cand))
    prices = {}
    for t in cand["ticker"]:
        df = get_daily_prices(t)
        if df.empty:
            log.warning("      no price data for %s", t)
            continue
        prices[t] = df
        df.to_csv(PRICE_DIR / f"{t}.csv")

    if prices:
        combined = pd.concat(
            [df.reset_index().assign(Ticker=t) for t, df in prices.items()], ignore_index=True
        )
        combined.to_csv(DATA_DIR / "combined_price_data.csv", index=False)
        vol_df = pd.DataFrame([
            {
                "ticker": t,
                "volatility_ann": annualized_vol(df["Close"]),
                "return_ann": annualized_return(df["Close"]),
                "last_close": float(df["Close"].iloc[-1]),
                "days": len(df),
            }
            for t, df in prices.items()
        ])
        vol_df.to_csv(DATA_DIR / "stock_volatility_data.csv", index=False)
        vol_df.to_csv(DATA_DIR / "scatter_volatility_return.csv", index=False)

    log.info("[5/7] + [6/7] Forecasting and epoch analysis for strong-sentiment tickers...")
    if not nixtla_key:
        log.info("      No NIXTLA_API_KEY found, using the simple drift fallback instead of TimeGPT.")
    signals, forecasts, epochs = build_signals(
        cand, prices, cfg, nixtla_key, matcher.company_by_ticker
    )
    if not forecasts.empty:
        forecasts.to_csv(DATA_DIR / "forecast_results.csv", index=False)
    if not epochs.empty:
        epochs.to_csv(DATA_DIR / "epoch_analysis.csv", index=False)

    log.info("[7/7] Signals")
    signals["run_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    signals.to_csv(DATA_DIR / "signals.csv", index=False)
    append_history(signals)
    for _, s in signals.iterrows():
        log.info("      %-6s %-5s score=%.3f mentions=%d  (%s)",
                 s["ticker"], s["signal"], s["sentiment_score"], s["mentions"], s["reason"])
    log.info("Done. Dashboard: streamlit run src/app.py | Score past signals: python src/backtest.py")
    return signals


def parse_args(argv=None) -> Config:
    d = Config()
    p = argparse.ArgumentParser(description="Tickerwire news-to-signal pipeline")
    p.add_argument("--days", type=int, default=d.news_days, help="days of news to pull")
    p.add_argument("--buy-threshold", type=float, default=d.buy_threshold)
    p.add_argument("--sell-threshold", type=float, default=d.sell_threshold)
    p.add_argument("--min-mentions", type=int, default=d.min_mentions)
    p.add_argument("--max-tickers", type=int, default=d.max_tickers)
    p.add_argument("--horizon", type=int, default=d.horizon, help="forecast days ahead")
    p.add_argument("--require-epoch", action="store_true",
                   help="turn BUY/SELL into HOLD when epoch analysis disagrees")
    a = p.parse_args(argv)
    return Config(
        news_days=a.days, buy_threshold=a.buy_threshold, sell_threshold=a.sell_threshold,
        min_mentions=a.min_mentions, max_tickers=a.max_tickers, horizon=a.horizon,
        require_epoch=a.require_epoch,
    )


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    cfg = parse_args(argv)
    run(cfg, os.getenv("NEWSAPI_KEY", ""), os.getenv("NIXTLA_API_KEY") or None)


if __name__ == "__main__":
    sys.exit(main())
