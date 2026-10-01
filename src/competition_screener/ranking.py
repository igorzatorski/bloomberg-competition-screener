"""Pure feature calculation, eligibility checks and equal-weight selection."""

import numpy as np
import pandas as pd

FEATURES = ["momentum_21d", "momentum_63d"]
WEIGHTS = [0.50, 0.50]
PORTFOLIO_SIZE = 10
REPORT_COLUMNS = [
    "ticker",
    "session",
    "adjusted_close",
    *FEATURES,
    "high_proximity",
    "volatility_21d",
    "traded_value_proxy_21d",
    "sma_20",
    "momentum_21d_ex_best",
    "best_day_return_21d",
    "max_gap_up_21d",
    "max_gap_up_date",
    "max_gap_down_21d",
    "max_gap_down_date",
]


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
            if not {"Open", "Close", "Volume"}.issubset(frame.columns):
                raise ValueError("missing Open/Close/Volume")
            clean = frame.dropna(subset=["Open", "Close", "Volume"])
            if len(clean) < 252:
                raise ValueError("fewer than 252 valid sessions")
            clean = clean.tail(252)
            if clean.index[-1] != latest or (cutoff - clean.index[-1]).days > 7:
                raise ValueError("stale data")
            if len(frame.loc[clean.index[0] :]) != len(clean):
                raise ValueError("missing sessions within lookback")
            c, v = clean["Close"], clean["Volume"]
            if not np.isfinite(clean[["Open", "Close", "Volume"]].to_numpy()).all():
                raise ValueError("non-finite data")
            if (c <= 0).any() or (clean.Open <= 0).any() or (v < 0).any():
                raise ValueError("invalid prices/volume")
            m21, m63 = c.iloc[-1] / c.iloc[-22] - 1, c.iloc[-1] / c.iloc[-64] - 1
            proximity = c.iloc[-1] / c.max()
            liquidity = (c * v).tail(21).mean()
            returns = c.pct_change(fill_method=None).tail(21)
            best = returns.max()
            gaps = (clean.Open / c.shift(1) - 1).tail(21)
            up, down = gaps[gaps > 0], gaps[gaps < 0]
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
                    "sma_20": c.tail(20).mean(),
                    "momentum_21d_ex_best": (1 + m21) / (1 + best) - 1,
                    "best_day_return_21d": best,
                    "max_gap_up_21d": up.max() if len(up) else 0.0,
                    "max_gap_up_date": str(up.idxmax().date()) if len(up) else "",
                    "max_gap_down_21d": down.min() if len(down) else 0.0,
                    "max_gap_down_date": str(down.idxmin().date()) if len(down) else "",
                }
            )
        except ValueError as error:
            rejected.append({"ticker": ticker, "reason": str(error)})
    ranking = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    ranked, filtered = rank_features(ranking)
    return ranked, pd.concat([pd.DataFrame(rejected, columns=["ticker", "reason"]), filtered], ignore_index=True)


def rank_features(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared cross-sectional filters and ranking for screener and backtest."""
    ranking = features.copy()
    rejected = []
    if not ranking.empty:
        # Compute the leadership threshold across all data-valid names, before
        # liquidity/trend filters. Boundary ties are retained.
        threshold = ranking.momentum_63d.quantile(0.70)
        ranking["momentum_63d_cutoff"] = threshold
        reasons = pd.Series("", index=ranking.index)
        rules = [
            (
                (ranking.momentum_21d <= 0) | (ranking.momentum_63d <= 0),
                "non-positive 21d/63d momentum",
            ),
            (
                ranking.traded_value_proxy_21d < 10_000_000,
                "daily traded-value proxy below USD 10m",
            ),
            (
                ranking.momentum_63d < threshold,
                "outside top 30% of data-valid universe by 63d momentum",
            ),
            (
                ranking.high_proximity < 0.90,
                "more than 10% below 252-session closing high",
            ),
            (
                ranking.adjusted_close <= ranking.sma_20,
                "close not above 20-session average",
            ),
            (
                ranking.momentum_21d_ex_best <= 1e-12,
                "21d return without best day is not positive",
            ),
        ]
        for failed, reason in rules:
            reasons.loc[failed & reasons.eq("")] = reason
        rejected.extend(
            {"ticker": ranking.loc[i, "ticker"], "reason": reason}
            for i, reason in reasons.items()
            if reason
        )
        ranking = ranking.loc[reasons.eq("")].copy()
    else:
        ranking["momentum_63d_cutoff"] = pd.Series(dtype=float)
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
    selected = ranking.head(PORTFOLIO_SIZE).copy()
    # Keep cash if fewer than ten names qualify.
    selected["target_weight"] = 1 / PORTFOLIO_SIZE
    return selected
