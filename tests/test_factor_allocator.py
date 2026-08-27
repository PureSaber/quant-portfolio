from pathlib import Path

import pandas as pd
import yaml

from quant_portfolio.allocator import allocate, load_config


def test_factor_allocator_smoke(tmp_path: Path) -> None:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir(exist_ok=True)
    nav = fixture_dir / "eq_nav.csv"
    hold = fixture_dir / "eq_holdings.csv"
    pd.DataFrame({"date": ["2024-06-01"], "nav": [100000.0]}).to_csv(nav, index=False)
    pd.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.6, 0.4]}).to_csv(hold, index=False)

    scores = fixture_dir / "factor_scores.parquet"
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "date": ["2024-06-01", "2024-06-01"],
            "momentum_20d": [0.5, -0.2],
        }
    ).to_parquet(scores, index=False)

    cfg_path = tmp_path / "factor_allocator_smoke.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "strategies": [{"name": "equity", "nav": str(nav), "weight": 1.0}],
                "factor_scores": {
                    "path": str(scores),
                    "column": "momentum_20d",
                    "weight": 0.3,
                },
            }
        ),
        encoding="utf-8",
    )
    snap = allocate(load_config(cfg_path))
    assert snap.total_nav > 0
    assert set(snap.combined_weights) == {"AAA", "BBB"}
