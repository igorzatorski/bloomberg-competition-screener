"""Local Parquet histories and small, atomic metadata files."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def price_path(root: Path, ticker: str) -> Path:
    if not re.fullmatch(r"[A-Z0-9^-]+", ticker):
        raise ValueError(f"Unsafe ticker: {ticker}")
    return root / "prices" / f"{ticker}.parquet"


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.dropna(how="all").sort_index().copy()
    if frame.empty or not set(PRICE_COLUMNS).issubset(frame.columns):
        raise ValueError("Empty data or missing OHLC/adjusted close/volume")
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame.index.name = "date"
    if frame.index.has_duplicates or frame.index.isna().any():
        raise ValueError("Invalid or duplicate dates")
    values = frame[PRICE_COLUMNS].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite values")
    if (frame[PRICE_COLUMNS[:-1]] <= 0).any().any() or (frame.Volume < 0).any():
        raise ValueError("Non-positive price or negative volume")
    if (
        (frame.High < frame[["Open", "Close", "Low"]].max(axis=1))
        | (frame.Low > frame[["Open", "Close", "High"]].min(axis=1))
    ).any():
        raise ValueError("Inconsistent OHLC")
    return frame


def save_prices(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary)
    temporary.replace(path)


def save_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_dataset(
    root: Path,
    cutoff: pd.Timestamp,
    allow_incomplete: bool = False,
    skip_failed: bool = False,
):
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if not metadata["complete"] and not allow_incomplete and not skip_failed:
        raise ValueError(
            "Dataset update incomplete. Run download-data again or use --skip-failed / --allow-incomplete"
        )
    if cutoff > pd.Timestamp(metadata["cutoff_exclusive"]):
        raise ValueError(
            "Dataset is older than requested cutoff. Run download-data first"
        )
    universe = pd.read_csv(root / "universe.csv")
    if skip_failed:
        report = pd.read_csv(root / "download_report.csv")
        if report.ticker.duplicated().any() or set(report.ticker) != set(
            universe.ticker
        ):
            raise ValueError("Download report does not match saved universe")
        ready = report.loc[
            report.status.isin(
                ["up_to_date", "downloaded", "updated", "correction_refresh"]
            ),
            "ticker",
        ]
        metadata["excluded_tickers"] = universe.loc[
            ~universe.ticker.isin(ready), "ticker"
        ].tolist()
        universe = universe.loc[universe.ticker.isin(ready)].reset_index(drop=True)
        if universe.empty:
            raise ValueError(
                "No ready histories remain after excluding failed/pending tickers"
            )
    prices = {}
    for ticker in universe.ticker:
        path = price_path(root, ticker)
        if not path.exists():
            continue
        frame = validate_prices(pd.read_parquet(path))
        frame = frame.loc[frame.index < cutoff].copy()
        # Convert split-adjusted Yahoo OHLC to dividend-adjusted OHLC consistently.
        factor = frame["Adj Close"] / frame["Close"]
        for col in ["Open", "High", "Low", "Close"]:
            frame[col] *= factor
        prices[ticker] = frame
    return universe, prices, metadata
