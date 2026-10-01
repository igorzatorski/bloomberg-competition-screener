import pytest

from competition_screener import data


def row(symbol, name, cap):
    return {"symbol": symbol, "name": name, "marketCap": cap}


def test_numeric_caps_classes_and_security_filters():
    rows = [
        row("BIG", "Big Inc. Common Stock", "10,000"),
        row("SMALL", "Small Inc. Common Stock", "900"),
        row("BRK.A", "Berkshire Inc. Class A Common Stock", "8,000"),
        row("BRK.B", "Berkshire Inc. Class B Common Stock", "8,000"),
        row("PREF", "Huge Inc. Preferred Stock", "100,000"),
        row("ADR", "Foreign Inc. American Depositary Shares", "100,000"),
        row("FUND", "Investment Fund Common Shares", "100,000"),
        row("BAD", "Bad Inc. Common Stock", "N/A"),
        row("ZERO", "Zero Inc. Common Stock", "0"),
    ]
    table = data.select_largest_companies(rows, count=3)
    assert table.ticker.tolist() == ["BIG", "BRK-A", "SMALL"]
    assert table.market_cap_usd.tolist() == [10000, 8000, 900]
    assert table.issuer_key.is_unique


def test_requires_full_requested_count():
    with pytest.raises(ValueError, match="expected 1000"):
        data.select_largest_companies([row("A", "A Inc. Common Stock", "100")])


def test_default_loads_1000_in_cap_order(monkeypatch):
    rows = [
        row(
            f"X{chr(65 + i // 26)}{chr(65 + i % 26)}",
            f"Company {i} Common Stock",
            str(i + 1),
        )
        for i in range(1100)
    ]
    # Generate alphabetic symbols accepted by the equity-symbol filter.
    for i, item in enumerate(rows):
        item["symbol"] = (
            "X" + chr(65 + i // 676) + chr(65 + i // 26 % 26) + chr(65 + i % 26)
        )

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": {"totalrecords": 1100, "table": {"rows": rows}, "asof": "test"}
            }

    def get(url, **kwargs):
        assert url == data.UNIVERSE_URL
        assert kwargs["params"]["country"] == "United States"
        return Response()

    monkeypatch.setattr(data.requests, "get", get)
    table = data.load_universe_table()
    assert len(table) == 1000
    assert table.market_cap_usd.iloc[0] == 1100
    assert table.market_cap_usd.iloc[-1] == 101
    assert table.market_cap_rank.tolist() == list(range(1, 1001))


def test_incomplete_source_is_rejected(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": {
                    "totalrecords": 2000,
                    "table": {"rows": [row("A", "A Common Stock", "10")]},
                }
            }

    monkeypatch.setattr(data.requests, "get", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="Incomplete Nasdaq"):
        data.load_universe_table()
