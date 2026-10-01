"""Small, reproducible weekly backtest for the competition screener."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from .ranking import rank_features
from .storage import load_dataset, price_path


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _format_stats(frame: pd.DataFrame) -> str:
    percent_metrics = {"total_return", "cagr", "alpha_cagr", "annual_volatility", "max_drawdown", "average_monthly_return", "average_monthly_alpha", "monthly_var_5pct", "monthly_var_1pct", "average_exposure", "transaction_costs_pct_initial", "turnover_pct_initial"}
    integer_metrics = {"months_positive", "months_negative", "trade_count"}
    rows = []
    for _, row in frame.iterrows():
        metric = str(row["metric"])
        def value(column: str, current_row: pd.Series = row, current_metric: str = metric) -> str:
            x = current_row.get(column, np.nan)
            if pd.isna(x): return "-"
            if current_metric in percent_metrics: return f"{float(x):>13.2%}"
            if current_metric in integer_metrics: return f"{round(float(x)):>14.0f}"
            if current_metric == "transaction_costs": return f"{float(x):>14,.2f}"
            return f"{float(x):>14.3f}"
        rows.append([metric, value('strategy').strip(), value('sp500').strip()])
    headers = ["Metric", "Strategy", "SPY"]
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(3)]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [border, "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(3)) + " |", border]
    lines += ["| " + " | ".join(r[i].ljust(widths[i]) for i in range(3)) + " |" for r in rows]
    lines.append(border)
    return "\n".join(lines)


def _features(prices: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(prices); started = time.monotonic()
    for number, (ticker, frame) in enumerate(prices.items(), 1):
        f = frame.sort_index().copy()
        c, o, v = f["Adj Close"], f["Open"], f["Volume"]
        ret = c.pct_change()
        x = pd.DataFrame(index=f.index)
        x["adjusted_close"] = c
        x["momentum_21d"] = c / c.shift(21) - 1
        x["momentum_63d"] = c / c.shift(63) - 1
        x["high_proximity"] = c / c.rolling(252, min_periods=252).max()
        x["volatility_21d"] = ret.rolling(21).std(ddof=1) * np.sqrt(252)
        x["traded_value_proxy_21d"] = (c * v).rolling(21).mean()
        x["sma_20"] = c.rolling(20).mean()
        x["momentum_21d_ex_best"] = (1 + x["momentum_21d"]) / (1 + ret.rolling(21).max()) - 1
        x["best_day_return_21d"] = ret.rolling(21).max()
        x["max_gap_up_21d"] = (o / c.shift(1) - 1).rolling(21).max()
        x["max_gap_down_21d"] = (o / c.shift(1) - 1).rolling(21).min()
        for signal_date in dates:
            prior = x.loc[x.index < signal_date]
            if prior.empty:
                continue
            row = prior.iloc[-1]
            if row.isna().any():
                continue
            rows.append({"signal_date": signal_date, "ticker": ticker, **row.to_dict()})
        if number == 1 or number % 50 == 0 or number == total:
            width = 30; filled = int(width * number / total)
            elapsed = time.monotonic() - started; eta = elapsed * (total-number) / number
            print(f"\rFeatures [{('#' * filled + '-' * (width-filled))}] {number/total:6.1%} | elapsed { _duration(elapsed) } | ETA {_duration(eta)}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


def _solve_rebalance(equity: float, old: dict[str, float], targets: dict[str, float], prices: dict[str, float], fee: float) -> tuple[dict[str, float], float, float, float]:
    names = set(old) | set(targets)
    current = {t: old.get(t, 0.0) * prices.get(t, 0.0) for t in names}
    def cost(e: float) -> float:
        return fee * sum(abs(targets.get(t, 0.0) * e - current.get(t, 0.0)) for t in names)
    lo, hi = 0.0, equity
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid + cost(mid) > equity: hi = mid
        else: lo = mid
    e = lo
    new = {t: targets.get(t, 0.0) * e / prices[t] for t in targets}
    notional = sum(abs(targets.get(t, 0.0) * e - current.get(t, 0.0)) for t in names)
    fees = fee * notional
    cash = e - sum(targets[t] * e for t in targets)
    return new, cash, fees, notional


def run_backtest(prices: dict[str, pd.DataFrame], benchmark: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp, initial_capital: float = 1_000_000, cost_bps: float = 10) -> dict[str, pd.DataFrame]:
    fee = cost_bps / 10_000
    spy = benchmark.sort_index()
    sessions = spy.index[(spy.index >= pd.Timestamp(start)) & (spy.index < pd.Timestamp(end))]
    if len(sessions) < 20: raise ValueError("Backtest needs at least 20 SPY sessions")
    signal_dates = sessions.to_series().groupby(sessions.to_period("W-SUN")).first().values
    signal_dates = pd.DatetimeIndex(signal_dates)
    features = _features(prices, signal_dates)
    holdings: dict[str, float] = {}; cash = initial_capital; fees_total = 0.0; turnover = 0.0
    rows, trades, selections = [], [], []
    started = time.monotonic()
    for i, trade_date in enumerate(sessions):
        sig = signal_dates[signal_dates <= trade_date]
        if len(sig) and sig[-1] == trade_date:
            snap = features[features.signal_date == trade_date].drop(columns="signal_date")
            ranked, _ = rank_features(snap)
            chosen = ranked.head(10)
            names = chosen.ticker.tolist()
            px = {t: float(prices[t].loc[trade_date, "Open"]) for t in set(names) | set(holdings) if trade_date in prices[t].index}
            # Existing positions must be valued even when the new ranking drops them.
            names = [t for t in names if t in px]
            names = [t for t in names if t in px]
            targets = {t: 0.1 for t in names}
            marked = cash + sum(q * float(prices[t].loc[trade_date, "Open"]) for t, q in holdings.items() if trade_date in prices[t].index)
            old_value = {t: q * float(prices[t].loc[trade_date, "Open"]) for t, q in holdings.items() if trade_date in prices[t].index}
            holdings, cash, fees, notion = _solve_rebalance(marked, {t: q for t, q in holdings.items() if t in old_value}, targets, px, fee)
            fees_total += fees; turnover += notion
            for t in set(old_value) | set(targets):
                delta = targets.get(t, 0) * (marked - fees) - old_value.get(t, 0)
                if abs(delta) > 1e-6: trades.append({"date": trade_date, "ticker": t, "notional": delta, "fee": abs(delta) * fee})
            for _, r in chosen.iterrows(): selections.append({"signal_date": trade_date, "trade_date": trade_date, "ticker": r.ticker, "rank": r.rank, "score": r.score, "target_weight": 0.1})
        close_values = sum(q * float(prices[t].loc[trade_date, "Close"]) for t, q in holdings.items() if trade_date in prices[t].index)
        net = cash + close_values
        rows.append({"date": trade_date, "strategy_net": net, "strategy_gross": net + fees_total, "cash": cash, "n_positions": len(holdings), "exposure": close_values / net if net else 0, "fees_cumulative": fees_total})
        if i == 0 or (i + 1) % 50 == 0 or i + 1 == len(sessions):
            width = 30; filled = int(width * (i + 1) / len(sessions))
            elapsed = time.monotonic() - started; eta = elapsed * (len(sessions)-i-1) / (i+1)
            print(f"\rBacktest [{('#' * filled + '-' * (width-filled))}] {(i+1)/len(sessions):6.1%} | elapsed { _duration(elapsed) } | ETA {_duration(eta)}", end="", flush=True)
    print()
    equity = pd.DataFrame(rows).set_index("date")
    spy_entry = float(spy.loc[sessions[0], "Open"]); spy_units = initial_capital * (1 - fee) / spy_entry
    spy_net = spy_units * spy.loc[sessions, "Close"]
    spy_net.iloc[-1] *= 1 - fee
    equity["sp500_net"] = spy_net.values
    return {"equity": equity.reset_index(), "trades": pd.DataFrame(trades), "selections": pd.DataFrame(selections), "features": features}


def statistics(equity: pd.DataFrame, trades: pd.DataFrame, start_capital: float, cost_bps: float) -> pd.DataFrame:
    e = equity.set_index("date"); out = []
    days = max((e.index[-1] - e.index[0]).days, 1); years = days / 365.25
    for col, label in [("strategy_net", "strategy"), ("sp500_net", "sp500")]:
        r = e[col].pct_change().dropna(); total = e[col].iloc[-1] / start_capital - 1
        dd = e[col] / e[col].cummax() - 1
        out += [{"metric":"total_return",label:total},{"metric":"cagr",label:(1+total)**(1/years)-1},{"metric":"annual_volatility",label:r.std(ddof=1)*np.sqrt(252)},{"metric":"sharpe_rf0",label:r.mean()/r.std(ddof=1)*np.sqrt(252) if r.std(ddof=1) else np.nan},{"metric":"max_drawdown",label:dd.min()}]
        monthly = e[col].resample("ME").last().pct_change().dropna()
        out += [
            {"metric": "months_positive", label: int((monthly > 0).sum())},
            {"metric": "months_negative", label: int((monthly < 0).sum())},
            {"metric": "average_monthly_return", label: monthly.mean()},
            {"metric": "monthly_var_5pct", label: monthly.quantile(0.05)},
            {"metric": "monthly_var_1pct", label: monthly.quantile(0.01)},
        ]
    result = pd.DataFrame(out).groupby("metric").first().reset_index()
    result = pd.concat([result, pd.DataFrame([
        {"metric": "average_exposure", "strategy": e.exposure.mean(), "sp500": 1.0},
        {"metric": "transaction_costs", "strategy": e.fees_cumulative.iloc[-1], "sp500": np.nan},
        {"metric": "transaction_costs_pct_initial", "strategy": e.fees_cumulative.iloc[-1] / start_capital, "sp500": np.nan},
        {"metric": "trade_count", "strategy": len(trades), "sp500": np.nan},
        {"metric": "turnover_pct_initial", "strategy": trades.notional.abs().sum() / start_capital, "sp500": np.nan},
    ])], ignore_index=True)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a 3-year weekly 10-slot backtest")
    p.add_argument("--data-dir", type=Path, default=Path("data/market")); p.add_argument("--before", type=str); p.add_argument("--years", type=int, default=3); p.add_argument("--cost-bps", type=float, default=10); p.add_argument("--capital", type=float, default=1_000_000); p.add_argument("--output", type=Path, default=Path("outputs/backtests"))
    a = p.parse_args(argv); root = a.data_dir
    cutoff = pd.Timestamp(a.before) if a.before else pd.Timestamp(json.loads((root / "dataset.json").read_text())["cutoff_exclusive"])
    _, prices, _ = load_dataset(root, cutoff, skip_failed=True)
    start = cutoff - pd.DateOffset(years=a.years)
    benchmark = pd.read_parquet(price_path(root, "^SESSION-SPY"))
    benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
    benchmark = benchmark.loc[benchmark["Close"].notna()].copy()
    benchmark["Adj Close"] = benchmark["Adj Close"].astype(float)
    # Use split/dividend-adjusted prices for the total-return benchmark.
    adjustment = benchmark["Adj Close"] / benchmark["Close"]
    benchmark["Open"] = benchmark["Open"] * adjustment
    benchmark["Close"] = benchmark["Close"] * adjustment
    result = run_backtest(prices, benchmark, start, cutoff, a.capital, a.cost_bps)
    out = a.output / pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    result["equity"].to_csv(out / "equity.csv", index=False); result["trades"].to_csv(out / "trades.csv", index=False); result["selections"].to_csv(out / "weekly_selections.csv", index=False)
    stats = statistics(result["equity"], result["trades"], a.capital, a.cost_bps); stats.to_csv(out / "statistics.csv", index=False)
    e=result["equity"].set_index("date")
    monthly = e[["strategy_net", "sp500_net"]].resample("ME").last().pct_change().dropna().rename(columns={"strategy_net": "strategy_return", "sp500_net": "sp500_return"})
    monthly.to_csv(out / "monthly_returns.csv")
    strategy_cagr = float(stats.loc[stats.metric == "cagr", "strategy"].iloc[0])
    sp500_cagr = float(stats.loc[stats.metric == "cagr", "sp500"].iloc[0])
    alpha_rows = pd.DataFrame([
        {"metric": "alpha_cagr", "strategy": strategy_cagr - sp500_cagr, "sp500": np.nan},
        {"metric": "average_monthly_alpha", "strategy": (monthly.strategy_return - monthly.sp500_return).mean(), "sp500": np.nan},
    ])
    stats = pd.concat([stats, alpha_rows], ignore_index=True)
    stats.to_csv(out / "statistics.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 5)); ax.plot(e.index, e.strategy_net/a.capital, label="Strategy"); ax.plot(e.index, e.sp500_net/a.capital, label="SPY total return"); ax.legend(); ax.set_ylabel("Growth of $1"); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "equity_curve.png", dpi=140); plt.close(fig)
    # Monthly observations remain in monthly_returns.csv and are also shown as
    # rug points; 1%-wide bins keep the histogram readable with only ~36 months.
    bin_width = 0.01
    low = np.floor(min(monthly.strategy_return.min(), monthly.sp500_return.min()) / bin_width) * bin_width
    high = np.ceil(max(monthly.strategy_return.max(), monthly.sp500_return.max()) / bin_width) * bin_width + bin_width
    bins = np.arange(low, high + bin_width / 2, bin_width)
    fig, (ax, rug) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
    ax.hist(monthly.strategy_return, bins=bins, alpha=.72, color="tab:blue", label="Strategy", align="left")
    ax.hist(monthly.sp500_return, bins=bins, alpha=.58, color="tab:orange", label="SPY", align="left")
    ax.axvline(monthly.strategy_return.quantile(.05), color="tab:blue", ls="--", lw=1.4, label="Strategy VaR 5%")
    ax.axvline(monthly.strategy_return.quantile(.01), color="tab:blue", ls=":", lw=1.6, label="Strategy VaR 1%")
    strategy_mean = monthly.strategy_return.mean(); sp500_mean = monthly.sp500_return.mean()
    ax.axvline(strategy_mean, color="tab:blue", ls="-", lw=1.2, alpha=.9, label=f"Strategy mean ({strategy_mean:.2%})")
    ax.axvline(sp500_mean, color="tab:orange", ls="-", lw=1.2, alpha=.9, label=f"SPY mean ({sp500_mean:.2%})")
    ax.set_ylabel("Months"); ax.set_title("Monthly return distribution", fontsize=14, pad=10)
    ax.legend(frameon=False, ncol=4, loc="upper left"); ax.grid(axis="y", alpha=.22)
    rug.scatter(monthly.strategy_return, np.ones(len(monthly)), marker="|", s=180, lw=1.5, color="tab:blue", label="Strategy")
    rug.scatter(monthly.sp500_return, np.zeros(len(monthly)), marker="|", s=180, lw=1.5, color="tab:orange", label="SPY")
    rug.set_yticks([0, 1], ["SPY", "Strategy"]); rug.set_ylim(-.5, 1.5); rug.set_xlabel("Monthly return"); rug.xaxis.set_major_formatter(PercentFormatter(1.0)); rug.grid(axis="x", alpha=.22)
    fig.tight_layout(); fig.savefig(out / "monthly_returns_distribution.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"Saved backtest: {out}"); print(_format_stats(stats))
    print(f"\nCharts: {out / 'equity_curve.png'}")
    print(f"        {out / 'monthly_returns_distribution.png'}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
