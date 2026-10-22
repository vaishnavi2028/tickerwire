# Tickerwire

A small pipeline that reads financial headlines, works out which companies they're about and whether the news is good or bad, forecasts those companies' prices, and then says BUY, SELL or HOLD only when the news and the forecast agree.

## What it does

```
NewsAPI ──> spaCy ──> FinBERT ──> yfinance ──> TimeGPT ──> epoch analysis ──> signal
headlines   find the   good or     5 years of   7-day       what happens        BUY / SELL
            companies  bad news?   prices       forecast    after past spikes   / HOLD
```

1. **Fetch news.** Pulls recent headlines from 15 finance outlets (Bloomberg, WSJ, CNBC, Reuters and so on) through NewsAPI.
2. **Find the companies.** spaCy's named entity recogniser pulls out organisation names. I fuzzy-match them against the S&P 500 list to get tickers ("Apple" becomes AAPL).
3. **Score sentiment.** FinBERT, a BERT model fine-tuned on financial text, rates each headline. I average the scores per ticker.
4. **Get prices.** Five years of daily prices from Yahoo Finance for the tickers worth looking at.
5. **Forecast.** Nixtla's TimeGPT predicts the next 7 trading days, with 80/90/95% confidence bands. If you don't have a Nixtla key, a simple trend-extension fallback is used instead and the output says so.
6. **Epoch analysis.** Finds unusually big one-day moves in a stock's history and averages the price path around them, to see whether big moves tend to keep going or reverse.
7. **Signal.** BUY needs strongly positive news *and* an upward forecast. SELL needs strongly negative news *and* a downward forecast. Anything else is HOLD. Each BUY/SELL also gets a volatility-based position size and stop-loss / take-profit levels taken from the forecast bands.

There's also a Streamlit dashboard for exploring the results.

## Running it

You need Python 3.10+ and a free [NewsAPI](https://newsapi.org) key. A [Nixtla](https://www.nixtla.io) key is optional.

```bash
git clone https://github.com/vaishnavi2028/tickerwire.git
cd tickerwire
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env      # then paste your keys into .env
python src/tickerwire.py      # run the pipeline
streamlit run src/app.py  # open the dashboard
```

Useful flags:

```bash
python src/tickerwire.py --days 7 --min-mentions 1     # more headlines, looser filter
python src/tickerwire.py --buy-threshold 0.9 --sell-threshold 0.1
python src/tickerwire.py --require-epoch               # drop signals that epoch analysis disagrees with
```

NewsAPI's free plan only returns 100 articles per request and 100 requests per day, so on a quiet news day you may not get many tickers with multiple mentions.

## The dashboard

Six tabs:

- **Sentiment overview**: the latest signals, a histogram of scores, most-mentioned tickers, volatility vs return, and common headline keywords.
- **News headlines**: searchable feed with a date and ticker filter.
- **Historical prices**: multi-ticker price charts, optionally rebased to 100 so you can compare them.
- **Forecast**: price history plus the forecast and its confidence bands, and the epoch analysis chart.
- **Backtest**: results from `src/backtest.py` (see below), with a warning banner whenever the sample is small.
- **Ask the data**: type "highest sentiment?" or "volatility of AAPL" and get an answer. This is keyword matching over the CSVs, not an LLM. I'd like to swap one in eventually.

It has a dark/light toggle and sidebar filters for sentiment, minimum mentions and score range.

## Project layout

```
src/
  tickerwire.py the pipeline (all seven stages)
  backtest.py   scores past signals against what prices actually did
  app.py        Streamlit dashboard
  qa.py         the keyword-based Q&A used by the dashboard
tests/
  test_logic.py unit tests for the pipeline logic
  test_backtest.py unit tests for the backtest engine
analysis/data/  CSV output lands here (gitignored)
```

Everything the pipeline produces is plain CSV in `analysis/data/`, which is how the dashboard reads it. `signals.csv` is the main output.

## Backtesting

`src/backtest.py` scores the signals against what prices really did afterwards.

```bash
python src/backtest.py --backfill 25      # rebuild signals for the last ~25 days of news, then score them
python src/backtest.py                    # score everything logged so far
python src/backtest.py --hold-days 3 --cost-bps 5 --long-only
```

How it works:

- Every pipeline run appends its signals to `analysis/data/signals_history.csv`, so the sample grows the longer you run it.
- A signal made on day D is entered at the **open of the next trading day**, never at D's close. Using the close would quietly peek at information the signal couldn't have had.
- Trades are held for 5 trading days by default, with 10 bps of cost charged on each side.
- Every trade is compared with SPY over the same window. I also compare against simply buying the same stocks on the same days, because "stocks that were in the news went up" isn't the same claim as "the sentiment signal worked".
- Averages come with a bootstrap 95% confidence interval, and the report says so when that interval includes zero.
- Signals whose exit date hasn't arrived yet are counted separately and left out of the results.

The catch: NewsAPI's free plan only keeps about a month of history, so a backfill gives a few dozen trades at best. That is not enough to prove anything, and the report and the dashboard both say so. Backfilled signals also use the trend-based forecast rather than TimeGPT, so they test "news sentiment plus recent trend". The honest way to get a meaningful number is to run the pipeline daily for a few months and score that.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

There are 31 tests covering company-name matching, sentiment aggregation, the forecast fallback, epoch analysis (including one that plants fake spikes and checks they're found), the signal rules, and an end-to-end run of the signal builder on synthetic prices. The backtest tests include one that makes sure entry happens at the next day's open and not the signal day's close, plus checks on costs, shorting, unfinished trades and the confidence intervals. They don't touch the network or load the ML models, so they run in about a second.

## What I changed from the original design

- **Looser default thresholds.** The original design only acts on a sentiment score of 0.98 or above (or 0.02 or below). Since I average scores across headlines, that almost never triggers. I default to 0.70 / 0.30 and made it a flag. You can set it back to 0.98 / 0.02 if you want the strict version.
- **A clearer sentiment score.** The score is `(P(positive) - P(negative) + 1) / 2`, so 0.5 means neutral. That way a neutral headline doesn't get mistaken for a negative one.
- **Nixtla's Python SDK** instead of hand-written REST calls, and a fallback forecast when there's no API key so the pipeline still runs end to end.
- **API keys in a `.env` file** instead of being pasted into the source.
- **Position sizing** is simple inverse-volatility: calmer stocks get bigger positions, capped at 10%.

## Limitations (please read these)

- **The backtest sample is small.** The backtest engine is in place, but with a month of news history it can only score a few dozen trades, which can't separate skill from luck. I haven't run it on a long history, so I'm not making any claims about performance.
- Headlines are a thin signal. Markets often move before the headline is published, and a headline that mentions a company isn't always about that company.
- Entity matching is fuzzy and imperfect. "Target" the retailer and "target" the verb look alike, and spaCy's small model misses names in Title Case headlines now and then.
- Only S&P 500 companies are recognised.
- Yahoo Finance data comes through an unofficial library and can break or be rate-limited.
- The fallback forecast just extends the recent trend, so it will almost always agree with whichever way a stock has been moving.
- The stop-loss and take-profit levels come from forecast confidence bands, which assume more about how prices behave than is really true.

**This is not financial advice.**

## What I'd do next

1. Schedule the pipeline to run daily so the signal history grows, then re-run the backtest after a few months.
2. Check how well the sentiment scores predict next-day returns on their own.
3. Test different hold periods and thresholds on data I haven't tuned on, to avoid overfitting.
4. Replace the keyword Q&A with an LLM.

## Credits

Design based on the Stocex capstone (INFO-698). Models: [FinBERT-tone](https://huggingface.co/yiyanghkust/finbert-tone) by Yang et al., [spaCy](https://spacy.io), and [TimeGPT](https://www.nixtla.io) by Nixtla.
