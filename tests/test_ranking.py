import numpy as np
import pandas as pd
import pytest

from competition_screener.data import load_universe
from competition_screener.ranking import rank_universe, select_portfolio


def prices(rate=0.002):
    return pd.DataFrame(
        {"Close": 100 * np.exp(np.arange(260) * rate), "Volume": 2_000_000},
        index=pd.bdate_range("2025-01-01", periods=260),
    )


def test_features_and_cutoff():
    frame = prices()
    cutoff = frame.index[-1] + pd.Timedelta(days=1)
    future = pd.DataFrame({"Close": [1_000_000], "Volume": [2_000_000]}, index=[cutoff])
    result, _ = rank_universe({"A": pd.concat([frame, future])}, cutoff, ["A"])
    assert result.iloc[0].momentum_21d == pytest.approx(np.exp(21 * 0.002) - 1)
    assert result.iloc[0].momentum_63d == pytest.approx(np.exp(63 * 0.002) - 1)
    assert result.iloc[0].high_proximity == 1


def test_missing_falling_short_and_stale_are_rejected():
    a = prices()
    cutoff = a.index[-1] + pd.Timedelta(days=1)
    ranking, rejected = rank_universe(
        {"FALL": prices(-0.002), "SHORT": a.tail(100), "STALE": a.iloc[:-10]},
        cutoff,
        ["FALL", "SHORT", "STALE", "MISSING"],
    )
    assert ranking.empty
    assert len(rejected) == 4


def test_portfolio_cap_and_cash():
    assert (
        select_portfolio(pd.DataFrame({"ticker": list("ABCDEFG")})).target_weight.sum()
        == 1
    )
    assert select_portfolio(pd.DataFrame({"ticker": ["A"]})).target_weight.sum() == 0.2
    assert select_portfolio(pd.DataFrame()).empty


def test_ranking_deterministic_and_momentum_preference():
    a = prices()
    cutoff = a.index[-1] + pd.Timedelta(days=1)
    ranking, _ = rank_universe(
        {"B": a, "A": a, "FAST": prices(0.004)}, cutoff, ["B", "A", "FAST"]
    )
    assert ranking.ticker.tolist() == ["FAST", "A", "B"]


def test_csv_normalization(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("ticker\nbrk.b\nAAPL\nAAPL\n", encoding="utf-8")
    assert load_universe(path) == ["AAPL", "BRK-B"]
