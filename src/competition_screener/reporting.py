"""Readable terminal and text-file reports; no portfolio calculations."""

import re

import pandas as pd


def table(headers: list[str], rows: list[list[str]], text_columns: set[int]) -> str:
    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def line(values):
        return (
            "| "
            + " | ".join(
                value.ljust(widths[i]) if i in text_columns else value.rjust(widths[i])
                for i, value in enumerate(values)
            )
            + " |"
        )

    return "\n".join(
        [border, line(headers), border, *(line(row) for row in rows), border]
    )


def portfolio_report(
    portfolio: pd.DataFrame, universe: pd.DataFrame, eligible: int, cutoff: str
) -> str:
    cash = 1 - portfolio.target_weight.sum()
    heading = f"WEEKLY MOMENTUM SCREEN | sessions before {cutoff}\nSelected: {len(portfolio)} | Eligible: {eligible} | Cash: {cash:.1%}"
    if portfolio.empty:
        return heading + "\nNo eligible stocks. See rejected.csv."
    names = (
        universe.set_index("ticker")["company"].to_dict()
        if "company" in universe
        else {}
    )
    main, risk = [], []
    for row in portfolio.itertuples():
        name = str(names.get(row.ticker, row.ticker))
        name = re.split(
            r"\s+(?:Class\s+|Common\s+|Ordinary\s+)",
            name,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        name = " ".join(name.split())
        if len(name) > 30:
            name = name[:27] + "..."
        main.append(
            [
                str(row.rank),
                row.ticker,
                name,
                f"{row.score * 100:.1f}",
                f"{row.momentum_21d:+.1%}",
                f"{row.momentum_63d:+.1%}",
                f"{row.target_weight:.0%}",
            ]
        )
        risk.append(
            [
                row.ticker,
                f"{row.momentum_21d_ex_best:+.1%}",
                f"{row.volatility_21d:.1%}",
                f"{row.max_gap_up_21d:+.1%}",
                f"{row.max_gap_down_21d:+.1%}",
            ]
        )
    return "\n\n".join(
        [
            heading,
            table(
                [
                    "#",
                    "Ticker",
                    "Company",
                    "Score /100",
                    "21d return",
                    "63d return",
                    "Weight",
                ],
                main,
                {1, 2},
            ),
            "RISK DIAGNOSTICS | last 21 sessions",
            table(
                [
                    "Ticker",
                    "Return ex-best day",
                    "Vol. annualized",
                    "Largest gap up",
                    "Largest gap down",
                ],
                risk,
                {0},
            ),
            "Score is a ranking, not a probability. Gap dates and full precision are in portfolio.csv.",
        ]
    )
