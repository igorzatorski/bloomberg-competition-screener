"""Bootstrap three years, then update only missing sessions with overlap."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from yfinance.exceptions import YFException

from .data import UNIVERSE_SOURCE, download_prices, load_universe_table
from .storage import PRICE_COLUMNS, price_path, save_json, save_prices, validate_prices


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def print_progress(rows: list[dict], total: int, active: str = "", started: float | None = None) -> None:
    done = len(rows)
    fraction = done / total if total else 1.0
    width = 28
    filled = int(width * fraction)
    cached = sum(row["status"] == "up_to_date" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    downloaded = done - cached - failed
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - started if started is not None else 0.0
    eta = elapsed * (1 - fraction) / fraction if fraction > 0 else 0.0
    print(
        f"[{bar}] {fraction:6.1%} | {done}/{total} | elapsed: {_duration(elapsed)} | ETA: {_duration(eta)} | saved: {downloaded} | cached: {cached} | failed: {failed} {active}",
        flush=True,
    )


def correction_changed(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    overlap = old.index.intersection(new.index)
    if overlap.empty:
        return True
    # Splits can revise raw OHLC; dividends can revise adjusted-close history.
    return not np.allclose(
        old.loc[overlap, PRICE_COLUMNS[:-1]],
        new.loc[overlap, PRICE_COLUMNS[:-1]],
        rtol=1e-6,
        atol=1e-6,
    )


def update_dataset(
    root: Path,
    cutoff: pd.Timestamp,
    years: int = 3,
    universe_path: Path | None = None,
    refresh: bool = False,
    start_date: pd.Timestamp | None = None,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    start = start_date if start_date is not None else cutoff - pd.DateOffset(years=years)
    if start.tzinfo is not None or start != start.normalize() or start >= cutoff:
        raise ValueError("History start must be a date before cutoff")
    previous_path = root / "dataset.json"
    previous = None
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        start = min(start, pd.Timestamp(previous["history_start"]))
        if cutoff < pd.Timestamp(previous["cutoff_exclusive"]):
            raise ValueError(
                "Download cutoff cannot move backwards; use another --data-dir"
            )
    universe = load_universe_table(universe_path)
    # Save universe before price acquisition, and mark interrupted runs incomplete.
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    snapshots = root / "universes"
    snapshots.mkdir(exist_ok=True)
    universe.to_csv(snapshots / f"{stamp}.csv", index=False)
    temporary = root / "universe.tmp.csv"
    universe.to_csv(temporary, index=False)
    temporary.replace(root / "universe.csv")
    metadata = {
        "complete": False,
        "updated_at_utc": timestamp,
        "cutoff_exclusive": str(cutoff.date()),
        "history_start": str(start.date()),
        "universe_source": str(universe_path or UNIVERSE_SOURCE),
        "universe_count": len(universe),
        "universe_snapshot": f"universes/{stamp}.csv",
    }
    save_json(metadata, previous_path)
    pd.DataFrame({"ticker": universe.ticker, "status": "pending", "error": ""}).to_csv(
        root / "download_report.csv", index=False
    )
    pd.DataFrame(columns=["ticker", "date", "reason"]).to_csv(
        root / "missing_sessions.csv", index=False
    )
    started = time.monotonic()
    print_progress([], len(universe), "Checking saved histories...", started)
    benchmark = validate_prices(
        download_prices(["SPY"], cutoff, start).get("SPY", pd.DataFrame())
    )
    benchmark = benchmark.loc[(benchmark.index >= start) & (benchmark.index < cutoff)]
    if benchmark.empty or (cutoff - benchmark.index[-1]).days > 7:
        raise ValueError("SPY session reference missing or stale")
    save_prices(benchmark, price_path(root, "^SESSION-SPY"))
    latest = benchmark.index[-1]
    groups, existing, rows, missing_rows = {}, {}, [], []
    for ticker in universe.ticker:
        path = price_path(root, ticker)
        try:
            old = (
                validate_prices(pd.read_parquet(path))
                if path.exists() and not refresh
                else None
            )
            if old is not None:
                existing[ticker] = old
            missing = (
                old is None
                or previous is None
                or start < pd.Timestamp(previous["history_start"])
            )
            begin = (
                start
                if refresh or missing
                else max(start, old.index[-1] - pd.Timedelta(days=7))
            )
            if (
                old is not None
                and not refresh
                and not missing
                and old.index[-1] == latest
            ):
                expected = benchmark.index[benchmark.index >= old.index[0]]
                if expected.difference(old.index).empty:
                    rows.append(
                        {
                            "ticker": ticker,
                            "status": "up_to_date",
                            "rows": len(old),
                            "first_session": str(old.index[0].date()),
                            "last_session": str(latest.date()),
                            "error": "",
                        }
                    )
                    continue
            if old is not None:
                gaps = benchmark.index[
                    (benchmark.index >= max(start, old.index[0]))
                    & (benchmark.index <= old.index[-1])
                ].difference(old.index)
                if len(gaps):
                    begin = start
            groups.setdefault(begin, []).append(ticker)
        except (ValueError, OSError) as error:
            rows.append({"ticker": ticker, "status": "failed", "error": str(error)})
    print_progress(rows, len(universe), started=started)
    for begin, tickers in groups.items():
        for offset in range(0, len(tickers), 50):
            batch = tickers[offset : offset + 50]
            print_progress(rows, len(universe), f"Fetching {batch[0]} ... {batch[-1]}", started)
            try:
                fetched = download_prices(batch, cutoff, begin)
            except (
                ValueError,
                OSError,
                YFException,
            ) as error:
                print(
                    f"Batch download failed, retrying individually: {error}", flush=True
                )
                fetched = {}
            for ticker in batch:
                try:
                    candidate = fetched.get(ticker)
                    if candidate is None or candidate.dropna(how="all").empty:
                        candidate = download_prices([ticker], cutoff, begin).get(
                            ticker, pd.DataFrame()
                        )
                    new = validate_prices(candidate)
                    old = existing.get(ticker)
                    mode = "updated" if old is not None else "downloaded"
                    full = begin == start
                    if old is not None and not full and correction_changed(old, new):
                        new = validate_prices(
                            download_prices([ticker], cutoff, start).get(
                                ticker, pd.DataFrame()
                            )
                        )
                        full, mode = True, "correction_refresh"
                    if full or old is None:
                        merged = new
                    else:
                        merged = pd.concat([old, new])
                        merged = merged.loc[~merged.index.duplicated(keep="last")]
                    merged = validate_prices(merged.loc[merged.index < cutoff])
                    expected = benchmark.index[
                        benchmark.index >= max(start, merged.index[0])
                    ]
                    gaps = expected.difference(merged.index)
                    if len(gaps):
                        repair_end = min(cutoff, gaps[-1] + pd.Timedelta(days=8))
                        repair = download_prices(
                            [ticker], repair_end, gaps[0] - pd.Timedelta(days=7)
                        ).get(ticker, pd.DataFrame())
                        if not repair.empty:
                            repair = validate_prices(repair)
                            repair = repair.loc[repair.index.isin(gaps)]
                            merged = validate_prices(pd.concat([merged, repair]))
                            gaps = expected.difference(merged.index)
                    if merged.index[-1] != latest or len(gaps):
                        missing_rows.extend(
                            {
                                "ticker": ticker,
                                "date": str(day.date()),
                                "reason": "absent_after_retry",
                            }
                            for day in gaps
                        )
                        raise ValueError(
                            f"Missing sessions: {[str(d.date()) for d in gaps[:10]]}; latest={merged.index[-1].date()}"
                        )
                    save_prices(merged, price_path(root, ticker))
                    rows.append(
                        {
                            "ticker": ticker,
                            "status": mode,
                            "rows": len(merged),
                            "first_session": str(merged.index[0].date()),
                            "last_session": str(merged.index[-1].date()),
                            "error": "",
                        }
                    )
                except (
                    ValueError,
                    OSError,
                    YFException,
                ) as error:
                    rows.append(
                        {"ticker": ticker, "status": "failed", "error": str(error)}
                    )
            print_progress(rows, len(universe), started=started)
    report = pd.DataFrame(rows)
    report.to_csv(root / "download_report.csv", index=False)
    pd.DataFrame(missing_rows, columns=["ticker", "date", "reason"]).to_csv(
        root / "missing_sessions.csv", index=False
    )
    failed = int((report.status == "failed").sum())
    metadata.update(
        complete=failed == 0,
        failed_count=failed,
        missing_session_count=len(missing_rows),
        latest_session=str(latest.date()),
        updated_at_utc=datetime.now(UTC).isoformat(),
    )
    save_json(metadata, previous_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download/update local daily histories; no screening"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/market"))
    parser.add_argument("--universe", type=Path, help="Optional custom ticker CSV")
    parser.add_argument("--years", type=int, choices=[2, 3], default=3)
    parser.add_argument("--start", help="Explicit history start, overrides --years; useful for backtest warmup")
    parser.add_argument(
        "--before",
        default=datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download full stored period"
    )
    args = parser.parse_args()
    try:
        cutoff = pd.Timestamp(args.before)
        if (
            cutoff.tzinfo is not None
            or cutoff != cutoff.normalize()
            or cutoff.date() > datetime.now(ZoneInfo("America/New_York")).date()
        ):
            raise ValueError("--before must be a date no later than today in New York")
        result = update_dataset(
            args.data_dir, cutoff, args.years, args.universe, args.refresh,
            pd.Timestamp(args.start) if args.start else None,
        )
        print(f"Saved: {args.data_dir.resolve()}; failures: {result['failed_count']}")
        if not result["complete"]:
            parser.exit(
                1,
                "Dataset incomplete; see download_report.csv and rerun download-data\n",
            )
    except (ValueError, OSError, ImportError, YFException) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
