# Bloomberg Competition Screener

Small weekly momentum screener and exploratory backtest for a simulated Bloomberg Global Trading Challenge portfolio. First download a persistent local dataset, then run the screener or the offline three-year backtest. Yahoo Finance supplies prices; trades are entered manually in Bloomberg. The project does not use the Bloomberg API or automate execution.

## Setup (Windows PowerShell, Python 3.11+)

For everyday use after setup, open one of the scripts in **`run/`** and click **Run**: `run/01_download_data.py` updates data, `run/02_run_screener.py` screens the saved dataset, and `run/03_run_backtest.py` runs the three-year backtest. These launchers use the project's `.venv` and set the working directory automatically, even if the editor selects another Python. The screening launcher explicitly skips failed ticker updates and reports exclusions. They also forward command-line arguments. Initial `.venv` setup is still required.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
download-data
run-screener
run-backtest
python -m pytest
```

Activation is optional: `.\.venv\Scripts\download-data.exe` and `.\.venv\Scripts\run-screener.exe` work directly. Module alternatives are `python -m competition_screener.download` and `python -m competition_screener`. Do not change your system execution policy just for this project.

`requirements-lock.txt` records the versions verified in the development environment. For the same versions, install it before the editable package: `python -m pip install -r requirements-lock.txt`, then `python -m pip install -e ".[dev]"`. CI runs tests and lint; no live network downloads are required by tests.

```powershell
download-data --years 2
download-data --universe my_universe.csv
download-data --refresh
run-screener --before 2026-09-26
```

Default universe: 1000 largest companies by reported USD market capitalization from the public [Nasdaq stock screener](https://www.nasdaq.com/market-activity/stocks/screener), with its country filter set to United States (not only Nasdaq-listed companies). The list refreshes with each download command, not with offline screening, and is sorted numerically by market cap. Common stock, common/ordinary shares and REIT beneficial-interest shares are accepted; name-based filters exclude funds, ETFs, preferred securities, depositary receipts, warrants, units and debt. Different share classes are grouped using normalized issuer names; the listing with the largest reported capitalization is retained, without summing company-level capitalization across classes. This is a source-based top 1000, not a certified issuer master: country labels, capitalization coverage and name-based classification can contain errors. It is not a WLS eligibility check.

The program fails if the source response is incomplete or fewer than 1000 qualifying companies are available; it never silently falls back to S&P 500. The saved `universe.csv` includes capitalization rank, ticker, source company name, issuer key, market cap and source as-of label, allowing review of the selection. Historical `--before` dates still use the current capitalization universe, not point-in-time membership.

You can override the download list with your own CSV containing a `ticker` column; `my_universe.csv` above is user-provided, not bundled. Yahoo symbols use BRK-B instead of BRK.B.

## Persistent data layer

`download-data` bootstraps three calendar years of daily history (two with `--years 2`), or available history for a recent IPO. Subsequent runs inspect the local files, request missing sessions with a seven-calendar-day overlap, and merge by date, keeping the new version. It catches up after a longer absence. Fully current histories are skipped. Expanding the requested period re-downloads it; shortening the setting does not delete archived history. Files for companies leaving top 1000 are retained, but only current members are updated. Ticker renames and reused tickers are not reconciled with permanent IDs.

Both commands accept `--data-dir` (default `data/market`). The downloader's `--before` is exclusive, defaults to today's New York date, and cannot move backwards for an existing store; use a separate directory for an older download. Screening defaults to the saved cutoff, accepts earlier cutoffs and never downloads anything. A cutoff beyond the saved dataset is rejected.

```text
data/market/
  universe.csv             # current list, source names and market caps
  universes/               # timestamped lists saved before price acquisition
  prices/TICKER.parquet    # one persistent history per ticker
  prices/^SESSION-SPY.parquet # observed trading-session reference
  download_report.csv      # ticker status, rows, first/last date, errors
  missing_sessions.csv     # one row per missing observed trading session
  dataset.json             # start, cutoff, source and complete flag
```

Yahoo is requested with `auto_adjust=False, actions=True`: split-adjusted OHLC, dividend-adjusted `Adj Close`, volume, dividends and splits are stored. Screening converts OHLC using `Adj Close / Close`. Changed overlapping prices trigger a full re-download of that ticker's stored requested period. Historical revisions outside the overlap can go undetected; `--refresh` explicitly refreshes everything. Already-current files are not probed for corrections until a later update or refresh.

Validation checks unique dates, finite positive prices, nonnegative volume and OHLC consistency. Observed SPY sessions provide a US-session reference to detect internal gaps and stale histories; this is not an authoritative exchange calendar. Short available histories are visible in the report; they may mean recent listing or limited provider coverage. Three years are requested, not guaranteed for every ticker.

Parquet replacement uses a temporary file. A failed update retains the previous good file. The dataset is marked incomplete before updating prices; interrupted runs remain incomplete. Empty batch downloads are retried once per ticker; missing sessions are additionally requested in a shorter date range, with no synthetic filling. Rerunning retries failures. Default screening refuses an incomplete dataset; `--allow-incomplete` explicitly permits available histories after you review the report. Run one downloader at a time; concurrent writers are unsupported. Old output CSVs are preserved but are not automatically migrated into the new store.

Downloads display a progress bar with percentage, checked tickers, saved histories, cached histories and failures. It advances when a batch completes, not on every individual network request. `missing_sessions.csv` lists every unresolved observed-session gap (ticker, date, reason), without truncating to ten dates; non-date-specific failures remain in `download_report.csv`. Weekends, holidays absent from the SPY reference, and days before the first available ticker row are not labeled missing sessions.

For a small sanity check, `run-screener --skip-failed` explicitly excludes failed/pending tickers according to the current download report. This retains the original top-1000 list in the data store and records excluded tickers in output metadata. It does not delete dates, interpolate prices, or mark the original dataset complete. Unlike `--allow-incomplete`, it never uses the previous file of a ticker whose latest update failed. The flags are mutually exclusive. If no ready tickers remain, screening stops.

## Transparent rules

Screening reads adjusted closes and volume locally, using only sessions strictly before its cutoff. At least 252 valid sessions are required. Missing, stale, non-finite or invalid histories are rejected. The downloader checks observed-session completeness independently of strategy filters.

Eligibility: positive 21-session and 63-session momentum; 63-session return at or above the 70th percentile of all names with valid 252-session histories (computed before other filters, with boundary ties retained); latest adjusted close within 10% of its maximum over 252 sessions and strictly above its 20-session average; average adjusted close times reported volume over 21 sessions at least USD 10m. The latter is a liquidity proxy, not exact historical dollar turnover.

The compounded 21-session return must remain positive after excluding its best daily close-to-close return: `(1 + momentum_21d) / (1 + best_day_return) - 1`. Values within 1e-12 of zero are treated as zero. This prevents a single jump from being the sole source of positive monthly momentum. Filters report the first failed rule for each rejected stock.

Score uses percentile ranks **among eligible stocks**:

| Feature | Weight |
|---|---:|
| 21-session momentum | 50% |
| 63-session momentum | 50% |

Higher values score better; ties resolve by ticker. Volatility remains a diagnostic (sample standard deviation of daily simple returns times sqrt(252)), with no score weight. Reports include the 20-session average, monthly return without its best day, best daily return, and largest positive/negative opening gaps over 21 sessions with dates. Gap = adjusted Open / previous adjusted Close - 1; dividend adjustments mean this is not exactly the raw quoted ex-dividend gap. If a gap direction never occurs, its value is zero and date blank. Gaps do not automatically exclude a stock. This is a heuristic, with no fitted parameters or evidence of predictive outperformance. The closing maximum is a 52-week approximation, **not an all-time high** or intraday high.

Top ten receive proposed equal weights of 10% each. Fewer qualifiers leave 10% per selected name and the rest in cash; the program never forces ineligible names or rescales the remaining positions. Portfolio size is defined once in `ranking.py` for reuse by the future backtest. No hedging, shorting or leverage. Weights are targets, not share quantities or validated Bloomberg orders. Confirm actual competition notional limits before trading. There is currently no sector limit: ten names can still be concentrated in related industries.

## Weekly use

Each weekend, run `download-data`, then `run-screener --skip-failed` to explicitly exclude failed updates. Both commands retain timestamped universe/output records. If downloading on Friday after the close, set `--before` to Saturday only once that date is reached (future cutoffs are rejected); the simplest routine is Saturday or Sunday. Review the ten names and trade at the next permitted session. Compare target holdings with existing positions rather than adding another 10%. This version has no holdings or order-difference calculator. Transaction prices, whole shares, corporate actions and Yahoo/Bloomberg differences require manual checking. Weekly runs are manual; no scheduler is installed.

Each screening run creates a UTC timestamped output folder with `ranking.csv`, `portfolio.csv`, `rejected.csv`, `universe.csv`, `report.txt` and `metadata.json` referring to the local dataset. The console and text report show aligned bordered tables with company names, score out of 100, percentage returns and target weights, followed by a separate risk table. CSVs retain full numerical precision and gap dates. Prices are not duplicated in each report. Later downloads can revise the price store: retain a dataset copy if exact historical-run reproduction is required. Yahoo can throttle or omit symbols; check `download_report.csv`.

## Competition context

Bloomberg's 2026 page lists October 12–November 13, 2026, starting positions due October 16, and highest time-weighted relative return against WLS as the winning measure. Source: https://portal.bloombergforeducation.com/trading_challenges/13 . Confirm your team's current regulations, permitted stocks, cash rules, position limits and execution timetable with the faculty advisor. This program does not certify regulatory eligibility. A historical top-100 threshold of 5.5% has not been verified and is not a target guarantee.

Concentration and higher volatility may increase upper-tail outcomes, while also increasing poor outcomes. They do not establish an optimal probability of finishing in the top 100. Correlated stocks can effectively form one sector bet. Weekly momentum selection can reverse sharply; fundamental news can matter even over a month.

## Structure and development

```text
src/competition_screener/
  data.py       # universe and Yahoo adapter
  storage.py    # validation, safe Parquet storage and offline reading
  download.py   # bootstrap/incremental download command
  ranking.py    # pure features, filters, score, weights
  reporting.py  # formatted console/text tables, separate from calculations
  cli.py        # command and saved artifacts
  backtest.py   # weekly 10-slot backtest and SPY comparison
run/
  01_download_data.py  # clickable data update launcher
  02_run_screener.py  # clickable weekly screener launcher
  03_run_backtest.py  # clickable backtest launcher
tests/          # offline deterministic checks
```

Keep `main` for reviewed working versions; use short branches such as `feature/holdings-diff` for later additions. Generated data and virtual environments are ignored. No GitHub remote or publication is created automatically.

The mini backtest is an exploratory sanity check, not a research-grade historical test. Run `run/03_run_backtest.py` (or `python -m competition_screener.backtest`) after downloading enough warm-up history. It uses three calendar years by default, weekly signals, ten equal target slots, next-session open execution, 10 bps per-side transaction costs and SPY as a dividend-adjusted S&P 500 proxy. It writes `equity.csv`, `weekly_selections.csv`, `trades.csv`, `statistics.csv`, `monthly_returns.csv`, `equity_curve.png` and `monthly_returns_distribution.png` under `outputs/backtests/<UTC timestamp>`. The report includes total return, CAGR, volatility, Sharpe at zero risk-free rate, maximum drawdown, positive/negative months, average monthly return, historical monthly VaR at 5% and 1%, exposure, turnover and fees. VaR is the empirical lower quantile of monthly returns, so it is shown as a negative return. Today's top 1000 creates survivorship/current-capitalization selection bias; snapshots from now do not reconstruct past constituents. A bias-free backtest would need point-in-time membership, capitalizations and delisted securities. Current adjusted histories may reflect later corporate actions.
