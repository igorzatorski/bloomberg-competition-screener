import numpy as np
import pandas as pd
import pytest

from competition_screener.data import load_universe
from competition_screener.ranking import rank_universe, select_portfolio


def prices(rate=0.002):
    return pd.DataFrame(
        {
            "Open": 100 * np.exp(np.arange(260) * rate),
            "Close": 100 * np.exp(np.arange(260) * rate),
            "Volume": 2_000_000,
        },
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
    assert select_portfolio(
        pd.DataFrame({"ticker": list("ABCDEFGHIJKL")})
    ).target_weight.sum() == pytest.approx(1)
    assert len(select_portfolio(pd.DataFrame({"ticker": list("ABCDEFGHIJKL")}))) == 10
    assert select_portfolio(pd.DataFrame({"ticker": ["A"]})).target_weight.sum() == 0.1
    assert select_portfolio(
        pd.DataFrame({"ticker": list("ABC")})
    ).target_weight.sum() == pytest.approx(0.3)
    assert select_portfolio(pd.DataFrame()).empty


def test_ranking_deterministic_and_momentum_preference():
    a = prices()
    cutoff = a.index[-1] + pd.Timedelta(days=1)
    ranking, _ = rank_universe(
        {"B": a, "A": a, "FAST": prices(0.004)}, cutoff, ["B", "A", "FAST"]
    )
    assert ranking.ticker.tolist() == ["FAST"]
    tied, _ = rank_universe({"B": a, "A": a}, cutoff, ["B", "A"])
    assert tied.ticker.tolist() == ["A", "B"]


def test_one_jump_does_not_qualify():
    frame = prices(0)
    frame.loc[frame.index[-10] :, ["Open", "Close"]] *= 1.3
    ranking, rejected = rank_universe(
        {"JUMP": frame}, frame.index[-1] + pd.Timedelta(days=1), ["JUMP"]
    )
    assert ranking.empty
    assert "without best day" in rejected.iloc[0].reason


def test_gap_diagnostics_and_compounded_return():
    frame = prices()
    frame.loc[frame.index[-4], "Open"] = frame.Close.iloc[-5] * 1.15
    frame.loc[frame.index[-3], "Open"] = frame.Close.iloc[-4] * 0.91
    ranking, _ = rank_universe(
        {"A": frame}, frame.index[-1] + pd.Timedelta(days=1), ["A"]
    )
    row = ranking.iloc[0]
    assert row.max_gap_up_21d == pytest.approx(0.15)
    assert row.max_gap_down_21d == pytest.approx(-0.09)
    assert row.max_gap_up_date == str(frame.index[-4].date())
    assert row.momentum_21d_ex_best == pytest.approx(np.exp(20 * 0.002) - 1)
    assert row.score == 1


def test_recent_decline_below_average_is_rejected():
    frame = prices(0.01)
    frame.loc[frame.index[-1], ["Open", "Close"]] *= 0.90
    ranking, rejected = rank_universe(
        {"A": frame}, frame.index[-1] + pd.Timedelta(days=1), ["A"]
    )
    assert ranking.empty
    assert "20-session" in rejected.iloc[0].reason


def test_csv_normalization(tmp_path):
    path = tmp_path / "u.csv"
    path.write_text("ticker\nbrk.b\nAAPL\nAAPL\n", encoding="utf-8")
    assert load_universe(path) == ["AAPL", "BRK-B"]
