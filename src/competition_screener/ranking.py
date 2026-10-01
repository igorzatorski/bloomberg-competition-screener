"""Pure feature calculation, eligibility checks and equal-weight selection."""

import numpy as np
import pandas as pd

FEATURES = ["momentum_21d", "momentum_63d", "volatility_21d", "high_proximity"]
WEIGHTS = [0.45, 0.25, 0.20, 0.10]


def rank_universe(
    prices: dict[str, pd.DataFrame], cutoff: pd.Timestamp, tickers: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, rejected = [], []
    latest = max(
        (
            df.loc[df.index < cutoff].index.max()
            for df in prices.values()
            if not df.loc[df.index < cutoff].empty
        ),
        default=None,
    )
    for ticker in tickers:
        try:
            frame = prices.get(ticker)
            if frame is None or frame.empty:
                raise ValueError("missing download")
            frame = frame.loc[frame.index < cutoff].sort_index()
            if frame.index.has_duplicates:
                raise ValueError("duplicate dates")
            if not {"Close", "Volume"}.issubset(frame.columns):
                raise ValueError("missing Close/Volume")
            clean = frame.dropna(subset=["Close", "Volume"])
            if len(clean) < 252:
                raise ValueError("fewer than 252 valid sessions")
            clean = clean.tail(252)
            if clean.index[-1] != latest or (cutoff - clean.index[-1]).days > 7:
                raise ValueError("stale data")
            if len(frame.loc[clean.index[0] :]) != len(clean):
                raise ValueError("missing sessions within lookback")
            c, v = clean["Close"], clean["Volume"]
            if not np.isfinite(clean[["Close", "Volume"]].to_numpy()).all():
                raise ValueError("non-finite data")
            if (c <= 0).any() or (v < 0).any():
                raise ValueError("invalid prices/volume")
            m21, m63 = c.iloc[-1] / c.iloc[-22] - 1, c.iloc[-1] / c.iloc[-64] - 1
            proximity = c.iloc[-1] / c.max()
            liquidity = (c * v).tail(21).mean()
            if m21 <= 0 or m63 <= 0:
                raise ValueError("non-positive 21d/63d momentum")
            if proximity < 0.90:
                raise ValueError("more than 10% below 252-session closing high")
            if liquidity < 10_000_000:
                raise ValueError("daily traded-value proxy below USD 10m")
            rows.append(
                {
                    "ticker": ticker,
                    "session": clean.index[-1].date().isoformat(),
                    "adjusted_close": c.iloc[-1],
                    "momentum_21d": m21,
                    "momentum_63d": m63,
                    "high_proximity": proximity,
                    "volatility_21d": c.pct_change(fill_method=None)
                    .tail(21)
                    .std(ddof=1)
                    * np.sqrt(252),
                    "traded_value_proxy_21d": liquidity,
                }
            )
        except ValueError as error:
            rejected.append({"ticker": ticker, "reason": str(error)})
    ranking = pd.DataFrame(
        rows,
        columns=[
            "ticker",
            "session",
            "adjusted_close",
            *FEATURES[:3],
            "high_proximity",
            "traded_value_proxy_21d",
        ],
    )
    if not ranking.empty:
        ranking["score"] = sum(
            ranking[col].rank(pct=True) * weight
            for col, weight in zip(FEATURES, WEIGHTS)
        )
        ranking = ranking.sort_values(
            ["score", "ticker"], ascending=[False, True]
        ).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    else:
        ranking.insert(0, "rank", pd.Series(dtype=int))
        ranking["score"] = pd.Series(dtype=float)
    return ranking, pd.DataFrame(rejected, columns=["ticker", "reason"])


def select_portfolio(ranking: pd.DataFrame) -> pd.DataFrame:
    selected = ranking.head(5).copy()
    # Never increase weights above the cap when fewer than five qualify.
    selected["target_weight"] = 0.20
    return selected
