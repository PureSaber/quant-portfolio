import pandas as pd
import pytest
import yaml

pytest.importorskip("quant_regime")

from quant_portfolio.allocator import allocate, load_config


def test_regime_scale_flows_into_portfolio(tmp_path) -> None:
    from quant_regime.cross_asset import detect_cross_asset

    series = tmp_path / "series.csv"
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=80), "close": range(80)}).to_csv(
        series, index=False
    )
    inputs = [
        {
            "name": "eq",
            "asset_class": "equity",
            "path": str(series),
            "date_col": "date",
            "value_col": "close",
        }
    ]
    rules = {
        "vol_window": 10,
        "vol_lookback": 30,
        "vol_percentile_threshold": 0.8,
        "return_window": 10,
    }
    result = detect_cross_asset(inputs, rules)

    nav = tmp_path / "nav.csv"
    pd.DataFrame({"date": ["2024-03-01"], "nav": [100000]}).to_csv(nav, index=False)

    cfg = tmp_path / "portfolio.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "strategies": [
                    {
                        "name": "eq",
                        "nav": str(nav),
                        "weight": 1.0,
                        "position_scale": result.position_scale,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    snap = allocate(load_config(cfg))
    assert snap.total_nav > 0
