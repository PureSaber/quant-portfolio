from __future__ import annotations

import json

import pandas as pd

from quant_portfolio.cli import main


def test_optimize_cli_writes_weights_and_capacity(tmp_path) -> None:
    expected = tmp_path / "expected.csv"
    returns = tmp_path / "returns.csv"
    liquidity = tmp_path / "liquidity.csv"
    out = tmp_path / "portfolio.json"
    pd.DataFrame({"symbol": ["A", "B"], "expected_return": [0.12, 0.08]}).to_csv(
        expected, index=False
    )
    pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "A": [0.01, -0.01, 0.02],
            "B": [0.0, 0.01, -0.01],
        }
    ).to_csv(returns, index=False)
    pd.DataFrame({"symbol": ["A", "B"], "average_daily_value": [10_000_000, 2_000_000]}).to_csv(
        liquidity, index=False
    )
    assert (
        main(
            [
                "optimize",
                "--expected-returns",
                str(expected),
                "--returns",
                str(returns),
                "--liquidity",
                str(liquidity),
                "--max-weight",
                "0.8",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-8
    assert payload["capacity"]["capacity"] > 0
