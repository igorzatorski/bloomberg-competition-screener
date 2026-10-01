"""Universe and Yahoo daily data; no brokerage integration."""

import re
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

UNIVERSE_URL = "https://api.nasdaq.com/api/screener/stocks"
UNIVERSE_SOURCE = "Nasdaq screener / United States / top 1000 by reported market cap"


def select_largest_companies(rows: list[dict], count: int = 1000) -> pd.DataFrame:
    """One common-equity listing per issuer, using source names as issuer keys."""
    records = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        # Exclude preferred securities, warrants, units, debt and depositary receipts.
        if re.search(
            r"preferred|depositary|depository|warrant|\bunits?\b|\bnotes?\b|\bbonds?\b|\betf\b|\bfund\b",
            name,
            re.IGNORECASE,
        ):
            continue
        if not re.search(
            r"common (stock|shares)|ordinary shares|shares of beneficial interest",
            name,
            re.IGNORECASE,
        ):
            continue
        if not re.fullmatch(r"[A-Z]+(?:[.-][A-Z])?", symbol):
            continue
        try:
            market_cap = float(
                str(row.get("marketCap", "")).replace(",", "").replace("$", "")
            )
        except ValueError:
            continue
        if not pd.notna(market_cap) or market_cap <= 0 or market_cap == float("inf"):
            continue
        issuer = re.split(
            r"\b(?:class\s+[A-Z0-9-]+\b|common stock|common shares|ordinary shares|shares of beneficial interest)",
            name,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        issuer = re.sub(r"[^a-z0-9]", "", issuer.lower())
        if not issuer:
            continue
        records.append(
            {
                "ticker": symbol.replace(".", "-"),
                "company": name,
                "issuer_key": issuer,
                "market_cap_usd": market_cap,
            }
        )
    table = pd.DataFrame(
        records, columns=["ticker", "company", "issuer_key", "market_cap_usd"]
    )
    table = table.sort_values(["market_cap_usd", "ticker"], ascending=[False, True])
    table = (
        table.drop_duplicates("ticker")
        .drop_duplicates("issuer_key")
        .head(count)
        .reset_index(drop=True)
    )
    if len(table) != count:
        raise ValueError(
            f"Only {len(table)} distinct common-equity companies found; expected {count}"
        )
    table.insert(0, "market_cap_rank", range(1, count + 1))
    return table


def load_universe_table(path: Path | None = None) -> pd.DataFrame:
    if path is not None:
        table = pd.read_csv(path)
        if "ticker" not in table:
            raise ValueError("Universe CSV requires a ticker column")
        tickers = sorted(
            {
                str(v).strip().upper().replace(".", "-")
                for v in table["ticker"].dropna()
                if str(v).strip()
            }
        )
        if not tickers:
            raise ValueError("Universe is empty")
        return pd.DataFrame({"ticker": tickers})
    response = requests.get(
        UNIVERSE_URL,
        params={"tableonly": "true", "limit": 10000, "country": "United States"},
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    response.raise_for_status()
    data = response.json().get("data") or {}
    rows = (data.get("table") or {}).get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Nasdaq universe response has no rows")
    try:
        total = int(data["totalrecords"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Nasdaq universe response has no valid totalrecords"
        ) from error
    if len(rows) != total:
        raise ValueError(f"Incomplete Nasdaq universe: {len(rows)} of {total} rows")
    table = select_largest_companies(rows)
    table["source_asof"] = str(data.get("asof") or "not supplied")
    return table


def load_universe(path: Path | None = None) -> list[str]:
    return load_universe_table(path)["ticker"].tolist()


def download_prices(
    tickers: list[str], cutoff: pd.Timestamp, start: pd.Timestamp | None = None
) -> dict[str, pd.DataFrame]:
    # End is exclusive: only sessions strictly before the specified date.
    start = (
        (start if start is not None else cutoff - pd.DateOffset(years=3))
        .date()
        .isoformat()
    )
    result = {}
    for offset in range(0, len(tickers), 50):
        batch = tickers[offset : offset + 50]
        frame = yf.download(
            batch,
            start=start,
            end=cutoff.date().isoformat(),
            auto_adjust=False,
            actions=True,
            group_by="ticker",
            threads=4,
            progress=False,
            timeout=30,
            multi_level_index=True,
        )
        if frame is None or frame.empty:
            continue
        for ticker in batch:
            if isinstance(frame.columns, pd.MultiIndex):
                if ticker not in frame.columns.get_level_values(0):
                    continue
                part = frame[ticker].copy()
            else:
                part = frame.copy()
            part.index = pd.to_datetime(part.index).tz_localize(None)
            result[ticker] = part.loc[part.index < cutoff].sort_index()
    return result
