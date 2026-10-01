import json

import numpy as np
import pandas as pd
import pytest

from competition_screener import download
from competition_screener.storage import (
    load_dataset,
    price_path,
    save_prices,
    validate_prices,
)


def history(end="2024-01-12", adjusted=1):
    dates = pd.bdate_range("2024-01-02", end)
    close = np.arange(len(dates)) + 100.0
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * adjusted,
            "Volume": 1000000,
            "Dividends": 0.0,
            "Stock Splits": 0.0,
        },
        index=dates,
    )


def provider(monkeypatch, frames, calls):
    monkeypatch.setattr(
        download, "load_universe_table", lambda _: pd.DataFrame({"ticker": ["ONE"]})
    )

    def fetch(tickers, cutoff, start):
        calls.append((tickers, start))
        return {
            t: f.loc[(f.index >= start) & (f.index < cutoff)]
            for t in tickers
            if (f := frames.get(t)) is not None
        }

    monkeypatch.setattr(download, "download_prices", fetch)


def test_bootstrap_incremental_repeat_and_local_read(tmp_path, monkeypatch):
    calls = []
    frames = {"SPY": history("2024-01-05"), "ONE": history("2024-01-05")}
    provider(monkeypatch, frames, calls)
    first = download.update_dataset(tmp_path, pd.Timestamp("2024-01-06"))
    assert first["complete"]
    saved = pd.read_parquet(price_path(tmp_path, "ONE"))
    assert len(saved) == 4
    assert calls[-1][1] == pd.Timestamp("2021-01-06")
    frames.update(SPY=history(), ONE=history())
    calls.clear()
    second = download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert second["complete"]
    assert calls[-1][1] == pd.Timestamp("2023-12-29")
    saved = pd.read_parquet(price_path(tmp_path, "ONE"))
    assert len(saved) == 9 and saved.index.is_unique
    calls.clear()
    download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert all(tickers == ["SPY"] for tickers, _ in calls)
    universe, local, metadata = load_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert universe.ticker.tolist() == ["ONE"]
    assert len(local["ONE"]) == 9 and metadata["complete"]


def test_adjustment_change_refreshes_full_history(tmp_path, monkeypatch):
    calls = []
    frames = {"SPY": history("2024-01-05"), "ONE": history("2024-01-05")}
    provider(monkeypatch, frames, calls)
    download.update_dataset(tmp_path, pd.Timestamp("2024-01-06"))
    frames.update(SPY=history(), ONE=history(adjusted=0.9))
    calls.clear()
    download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert calls[-1] == (["ONE"], pd.Timestamp("2021-01-06"))
    report = pd.read_csv(tmp_path / "download_report.csv")
    assert report.iloc[0].status == "correction_refresh"
    _, prices, _ = load_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert prices["ONE"].Close.iloc[0] == pytest.approx(90)


def test_failure_keeps_previous_file_and_blocks_screener(tmp_path, monkeypatch):
    calls = []
    frames = {"SPY": history("2024-01-05"), "ONE": history("2024-01-05")}
    provider(monkeypatch, frames, calls)
    download.update_dataset(tmp_path, pd.Timestamp("2024-01-06"))
    original = price_path(tmp_path, "ONE").read_bytes()
    frames["SPY"] = history()
    frames.pop("ONE")
    result = download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert not result["complete"] and result["failed_count"] == 1
    assert price_path(tmp_path, "ONE").read_bytes() == original
    with pytest.raises(ValueError, match="incomplete"):
        load_dataset(tmp_path, pd.Timestamp("2024-01-13"))


def test_internal_gap_repaired(tmp_path, monkeypatch):
    calls = []
    frames = {"SPY": history(), "ONE": history()}
    provider(monkeypatch, frames, calls)
    download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    save_prices(history().drop(pd.Timestamp("2024-01-05")), price_path(tmp_path, "ONE"))
    calls.clear()
    download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert calls[-1][1] == pd.Timestamp("2021-01-13")
    assert len(pd.read_parquet(price_path(tmp_path, "ONE"))) == 9


def test_reject_invalid_data_and_future_local_cutoff(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        validate_prices(pd.concat([history(), history()]))
    (tmp_path / "dataset.json").write_text(
        json.dumps({"complete": True, "cutoff_exclusive": "2024-01-13"})
    )
    with pytest.raises(ValueError, match="older"):
        load_dataset(tmp_path, pd.Timestamp("2024-01-14"))


def test_missing_session_retried_in_short_window(tmp_path, monkeypatch):
    monkeypatch.setattr(
        download, "load_universe_table", lambda _: pd.DataFrame({"ticker": ["ONE"]})
    )
    calls = []

    def fetch(tickers, cutoff, start):
        calls.append((tickers, start, cutoff))
        frame = history()
        if tickers == ["ONE"] and start.year < 2024:
            frame = frame.drop(pd.Timestamp("2024-01-08"))
        return {
            t: frame.loc[(frame.index >= start) & (frame.index < cutoff)]
            for t in tickers
        }

    monkeypatch.setattr(download, "download_prices", fetch)
    result = download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert result["complete"]
    assert calls[-1][1] == pd.Timestamp("2024-01-01")
    assert len(pd.read_parquet(price_path(tmp_path, "ONE"))) == 9
    assert pd.read_csv(tmp_path / "missing_sessions.csv").empty


def test_missing_dates_report_and_explicit_exclusion(tmp_path, monkeypatch):
    calls = []
    frames = {
        "SPY": history(),
        "ONE": history().drop(pd.Timestamp("2024-01-08")),
        "GOOD": history(),
    }
    provider(monkeypatch, frames, calls)
    monkeypatch.setattr(
        download,
        "load_universe_table",
        lambda _: pd.DataFrame({"ticker": ["ONE", "GOOD"]}),
    )
    result = download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert not result["complete"]
    missing = pd.read_csv(tmp_path / "missing_sessions.csv")
    assert missing[["ticker", "date"]].to_dict("records") == [
        {"ticker": "ONE", "date": "2024-01-08"}
    ]
    universe, local, metadata = load_dataset(
        tmp_path, pd.Timestamp("2024-01-13"), skip_failed=True
    )
    assert universe.ticker.tolist() == ["GOOD"]
    assert list(local) == ["GOOD"]
    assert metadata["excluded_tickers"] == ["ONE"]
    assert not price_path(tmp_path, "ONE").exists()


def test_short_listing_has_no_artificial_prelisting_gaps(tmp_path, monkeypatch):
    calls = []
    frames = {"SPY": history(), "ONE": history().loc["2024-01-05":]}
    provider(monkeypatch, frames, calls)
    result = download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert result["complete"]
    assert pd.read_csv(tmp_path / "missing_sessions.csv").empty


def test_progress_reports_cached_saved_and_failed(capsys):
    download.print_progress(
        [{"status": "up_to_date"}, {"status": "updated"}, {"status": "failed"}], 3
    )
    output = capsys.readouterr().out
    assert "100.0%" in output and "3/3" in output
    assert "saved: 1" in output and "cached: 1" in output and "failed: 1" in output


def test_interrupted_reference_download_remains_incomplete(tmp_path, monkeypatch):
    calls = []
    provider(monkeypatch, {}, calls)
    with pytest.raises(ValueError, match="Empty"):
        download.update_dataset(tmp_path, pd.Timestamp("2024-01-13"))
    assert not json.loads((tmp_path / "dataset.json").read_text())["complete"]
    assert pd.read_csv(tmp_path / "download_report.csv").iloc[0].status == "pending"
