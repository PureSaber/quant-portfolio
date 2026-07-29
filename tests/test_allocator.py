from pathlib import Path

import pandas as pd
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
