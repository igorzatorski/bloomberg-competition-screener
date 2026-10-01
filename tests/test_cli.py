import json
import sys

import pandas as pd

from competition_screener import cli


def test_saved_empty_outputs_are_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-screener",
            "--before",
            "2026-01-01",
            "--data-dir",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(
        cli,
        "load_dataset",
        lambda *_: (
            pd.DataFrame({"ticker": ["MISSING"]}),
            {},
            {"universe_source": "test", "updated_at_utc": "test", "complete": True},
        ),
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset.json").write_text(
        '{"cutoff_exclusive": "2026-01-01"}', encoding="utf-8"
    )
    cli.main()
    run = next((tmp_path / "out").iterdir())
    assert pd.read_csv(run / "portfolio.csv").empty
    assert pd.read_csv(run / "ranking.csv").empty
    assert pd.read_csv(run / "rejected.csv").iloc[0].reason == "missing download"
    assert json.loads((run / "metadata.json").read_text())["cash_weight"] == 1
