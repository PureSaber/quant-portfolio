from pathlib import Path

import pandas as pd
import pytest
import yaml

from quant_portfolio.allocator import allocate


def test_allocate_combines_weights(tmp_path: Path) -> None:
    eq_nav = tmp_path / "eq_nav.csv"
    eq_hold = tmp_path / "eq_holdings.csv"
    pd.DataFrame({"date": ["2024-01-01"], "nav": [100000.0]}).to_csv(eq_nav, index=False)
    pd.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]}).to_csv(eq_hold, index=False)

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "strategies": [
                    {"name": "equity", "nav": str(eq_nav), "weight": 0.7, "position_scale": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    snap = allocate(yaml.safe_load(cfg.read_text(encoding="utf-8")))
    assert snap.total_nav > 0
    assert snap.combined_weights["AAA"] == 0.6


def test_capital_budget_retains_cash_and_full_precision(tmp_path):
    strategies = []
    for i, scale in enumerate([1.0, 0.5, 0.0]):
        nav = tmp_path / f"{i}_nav.csv"
        pd.DataFrame({"date": ["2026-09-30"], "nav": [100000.0]}).to_csv(nav, index=False)
        pd.DataFrame({"symbol": [f"asset{i}"], "weight": [1.0]}).to_csv(
            tmp_path / f"{i}_holdings.csv", index=False
        )
        strategies.append({"name": str(i), "nav": str(nav), "weight": 1.0, "position_scale": scale})
    snapshot = allocate({"strategies": strategies})
    assert snapshot.total_nav == 100000.0
    assert sum(snapshot.combined_weights.values()) == pytest.approx(0.5, abs=1e-6)
    assert sum(book["cash_weight"] for book in snapshot.books) == pytest.approx(0.5, abs=1e-6)
