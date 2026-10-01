"""Compare 5-20 equal-weight portfolio sizes using one shared feature table."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Make the launcher work when opened directly in VS Code.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from competition_screener.backtest import _features, run_backtest, statistics
from competition_screener.storage import load_dataset, price_path


def format_table(table: pd.DataFrame) -> str:
    headers = ["Slots", "CAGR", "Total return", "Volatility", "Sharpe", "Max drawdown", "VaR 5%", "Exposure", "Fees"]
    rows = []
    for _, row in table.iterrows():
        rows.append([str(int(row.slots)), f"{row.cagr:.2%}", f"{row.total_return:.2%}", f"{row.volatility:.2%}", f"{row.sharpe:.3f}", f"{row.max_drawdown:.2%}", f"{row.var_5:.2%}", f"{row.avg_exposure:.2%}", f"{row.fees_pct:.2%}"])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    lines = [border, "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |", border]
    lines += ["| " + " | ".join(r[i].rjust(widths[i]) for i in range(len(headers))) + " |" for r in rows]
    return "\n".join(lines + [border])


def main() -> int:
    root = ROOT
    data = root / "data/market"
    cutoff = pd.Timestamp(json.loads((data / "dataset.json").read_text())["cutoff_exclusive"])
    _, prices, _ = load_dataset(data, cutoff, skip_failed=True)
    spy = pd.read_parquet(price_path(data, "^SESSION-SPY")); spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy = spy.loc[spy.Close.notna()].copy(); adj = spy["Adj Close"] / spy["Close"]; spy["Open"] *= adj; spy["Close"] *= adj
    start = cutoff - pd.DateOffset(years=3); sessions = spy.index[(spy.index >= start) & (spy.index < cutoff)]
    signal_dates = pd.DatetimeIndex(sessions.to_series().groupby(sessions.to_period("W-SUN")).first().values)
    features = _features(prices, signal_dates)
    rows = []
    for slots in range(5, 21):
        result = run_backtest(prices, spy, start, cutoff, 1_000_000, 10, True, slots, features)
        stats = statistics(result["equity"], result["trades"], 1_000_000, 10).set_index("metric")
        rows.append({"slots": slots, "cagr": stats.loc["cagr", "strategy"], "total_return": stats.loc["total_return", "strategy"], "volatility": stats.loc["annual_volatility", "strategy"], "sharpe": stats.loc["sharpe_rf0", "strategy"], "max_drawdown": stats.loc["max_drawdown", "strategy"], "var_5": stats.loc["monthly_var_5pct", "strategy"], "avg_exposure": stats.loc["average_exposure", "strategy"], "fees_pct": stats.loc["transaction_costs_pct_initial", "strategy"]})
    table = pd.DataFrame(rows); out = root / "outputs/slot_comparison"; out.mkdir(parents=True, exist_ok=True); table.to_csv(out / "slot_comparison.csv", index=False)
    heat = table.set_index("slots").T; fig, ax = plt.subplots(figsize=(14, 7)); image = ax.imshow(heat.values, aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1); ax.set_xticks(range(len(heat.columns)), heat.columns); ax.set_yticks(range(len(heat.index)), heat.index); fig.colorbar(image, ax=ax, label="Value"); ax.set_title("Portfolio-size comparison: 5 to 20 slots"); fig.tight_layout(); fig.savefig(out / "slot_comparison_heatmap.png", dpi=160); plt.close(fig)
    print(format_table(table))
    return 0


if __name__ == "__main__": raise SystemExit(main())
