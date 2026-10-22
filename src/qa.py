"""
Tiny keyword-based question answering over the pipeline's CSV output.

This is NOT a language model. It looks for phrases like "highest sentiment" or
"volatility of AAPL" and answers from the data. Swapping in a real LLM later
would be a nice extension.
"""
from __future__ import annotations

import re

import pandas as pd


def _find_tickers(question: str, known: set[str]) -> list[str]:
    # Tickers must be typed in capitals ("AAPL"), otherwise words like "on" or "all"
    # would be mistaken for the tickers ON and ALL.
    tokens = re.findall(r"\b[A-Z][A-Z\-]{0,5}\b", question)
    return [t for t in tokens if t in known]


def answer_question(
    question: str,
    summary: pd.DataFrame,
    signals: pd.DataFrame,
    vol: pd.DataFrame,
) -> str:
    q = question.lower().strip()
    if not q:
        return "Ask me something, e.g. 'highest sentiment?' or 'volatility of AAPL'."
    if summary.empty:
        return "There's no data yet. Run `python src/tickerwire.py` first."

    known = set(summary["ticker"]) | set(vol.get("ticker", [])) | set(signals.get("ticker", []))
    tickers = _find_tickers(question, known)

    if tickers:
        t = tickers[0]
        parts = []
        s = summary[summary["ticker"] == t]
        sg = signals[signals["ticker"] == t] if not signals.empty else signals
        v = vol[vol["ticker"] == t] if not vol.empty else vol
        wants_all = not any(k in q for k in ("sentiment", "return", "volatil", "signal", "price", "mention"))
        if (wants_all or "sentiment" in q or "mention" in q) and not s.empty:
            r = s.iloc[0]
            parts.append(
                f"{t} sentiment is {r['sentiment_score']:.2f} ({r['sentiment_label'].lower()}) "
                f"across {int(r['mentions'])} headline(s)."
            )
        if (wants_all or "signal" in q) and not sg.empty:
            r = sg.iloc[0]
            parts.append(f"Signal: {r['signal']} ({r['reason']}).")
        if (wants_all or "return" in q) and not v.empty:
            parts.append(f"Average annualised return: {v.iloc[0]['return_ann'] * 100:.1f}%.")
        if (wants_all or "volatil" in q) and not v.empty:
            parts.append(f"Annualised volatility: {v.iloc[0]['volatility_ann'] * 100:.1f}%.")
        if ("price" in q) and not v.empty:
            parts.append(f"Last close: ${v.iloc[0]['last_close']:.2f}.")
        return " ".join(parts) or f"I have no data for {t}."

    if any(k in q for k in ("highest sentiment", "most positive", "best sentiment")):
        r = summary.sort_values("sentiment_score", ascending=False).iloc[0]
        return f"{r['ticker']} has the highest sentiment: {r['sentiment_score']:.2f} ({int(r['mentions'])} mentions)."
    if any(k in q for k in ("lowest sentiment", "most negative", "worst sentiment")):
        r = summary.sort_values("sentiment_score").iloc[0]
        return f"{r['ticker']} has the lowest sentiment: {r['sentiment_score']:.2f} ({int(r['mentions'])} mentions)."
    if any(k in q for k in ("most mentioned", "most news", "most talked", "most headlines")):
        r = summary.sort_values("mentions", ascending=False).iloc[0]
        return f"{r['ticker']} is mentioned most: {int(r['mentions'])} headline(s)."
    if "most volatile" in q or "highest volatility" in q:
        if vol.empty:
            return "No volatility data yet."
        r = vol.sort_values("volatility_ann", ascending=False).iloc[0]
        return f"{r['ticker']} is the most volatile at {r['volatility_ann'] * 100:.1f}% annualised."
    if "least volatile" in q or "lowest volatility" in q:
        if vol.empty:
            return "No volatility data yet."
        r = vol.sort_values("volatility_ann").iloc[0]
        return f"{r['ticker']} is the least volatile at {r['volatility_ann'] * 100:.1f}% annualised."
    if "buy" in q or "sell" in q:
        want = "BUY" if "buy" in q else "SELL"
        hits = signals[signals["signal"] == want]["ticker"].tolist() if not signals.empty else []
        return f"{want} signals: {', '.join(hits)}." if hits else f"No {want} signals in the latest run."
    if "how many" in q and ("ticker" in q or "compan" in q or "stock" in q):
        return f"{len(summary)} tickers appeared in the latest headlines."

    return (
        "I can answer things like: 'highest sentiment?', 'lowest sentiment?', 'most mentioned?', "
        "'most volatile stock?', 'which stocks to buy?', 'return of AAPL', 'signal for TSLA'."
    )
