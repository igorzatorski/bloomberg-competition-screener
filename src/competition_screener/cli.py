import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .ranking import FEATURES, WEIGHTS, rank_universe, select_portfolio
from .storage import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly Yahoo momentum screener; proposed trades only"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/market"),
        help="Local dataset created by download-data",
    )
    parser.add_argument(
        "--before",
        help="Exclusive YYYY-MM-DD cutoff; default saved dataset cutoff",
    )
    coverage = parser.add_mutually_exclusive_group()
    coverage.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Explicitly permit a partially updated dataset",
    )
    coverage.add_argument(
        "--skip-failed",
        action="store_true",
        help="Exclude failed/pending tickers using download_report.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    try:
        saved_metadata = json.loads(
            (args.data_dir / "dataset.json").read_text(encoding="utf-8")
        )
        cutoff = pd.Timestamp(args.before or saved_metadata["cutoff_exclusive"])
        if cutoff.tzinfo is not None or cutoff != cutoff.normalize():
            raise ValueError("--before must be a date without time or timezone")
        if cutoff.date() > datetime.now(ZoneInfo("America/New_York")).date():
            raise ValueError("Future cutoff is not allowed")
        universe, prices, dataset_metadata = load_dataset(
            args.data_dir, cutoff, args.allow_incomplete, args.skip_failed
        )
        excluded = dataset_metadata.get("excluded_tickers", [])
        if excluded:
            print(
                f"Excluded {len(excluded)} failed/pending tickers: {', '.join(excluded)}"
            )
        tickers = universe["ticker"].tolist()
        print(
            f"Reading {len(tickers)} local histories; sessions before {cutoff.date()}..."
        )
        missing = [
            t for t in tickers if t not in prices or prices[t].dropna(how="all").empty
        ]
        if missing:
            print(
                f"WARNING: {len(missing)}/{len(tickers)} symbols have no data; ranking coverage is incomplete."
            )
        ranking, rejected = rank_universe(prices, cutoff, tickers)
        portfolio = select_portfolio(ranking)
        run = args.output / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        run.mkdir(parents=True, exist_ok=False)
        universe.to_csv(run / "universe.csv", index=False)
        for name, table in [
            ("ranking", ranking),
            ("portfolio", portfolio),
            ("rejected", rejected),
        ]:
            table.to_csv(run / f"{name}.csv", index=False)
        metadata = {
            "cutoff_exclusive": str(cutoff.date()),
            "universe_source": dataset_metadata["universe_source"],
            "universe_count": len(tickers),
            "downloaded_count": len(prices),
            "eligible_count": len(ranking),
            "score_weights": dict(zip(FEATURES, WEIGHTS)),
            "cash_weight": round(1 - len(portfolio) * 0.2, 2),
            "position_cap_interpretation": "20% of portfolio; confirm competition notional rules",
            "data_directory": str(args.data_dir.resolve()),
            "dataset_updated_at_utc": dataset_metadata["updated_at_utc"],
            "dataset_complete": dataset_metadata["complete"],
            "excluded_tickers": excluded,
        }
        (run / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        if portfolio.empty:
            print(
                "No eligible stocks. Review rejected.csv; proposed portfolio is 100% cash."
            )
        else:
            print(
                portfolio[
                    [
                        "ticker",
                        "score",
                        "momentum_21d",
                        "volatility_21d",
                        "target_weight",
                    ]
                ].to_string(index=False)
            )
        print(f"Cash: {metadata['cash_weight']:.0%}. Saved: {run.resolve()}")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"Screener failed: {error}\n")
