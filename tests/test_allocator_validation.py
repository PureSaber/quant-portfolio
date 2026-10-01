import json

import pandas as pd
import pytest
import yaml

from quant_portfolio.allocator import allocate
from quant_portfolio.cli import main


def inputs(tmp_path):
    nav = tmp_path / "book_nav.csv"
    pd.DataFrame({"date": ["2026-09-30"], "nav": [100000.0]}).to_csv(nav, index=False)
    holdings = tmp_path / "book_holdings.csv"
    pd.DataFrame({"symbol": ["A"], "weight": [1.0]}).to_csv(holdings, index=False)
    return {"strategies": [{"name": "test", "nav": str(nav)}]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.5, 1.5, True])
def test_invalid_scale_fails_before_cli_output_is_created(tmp_path, value):
    cfg = inputs(tmp_path)
    cfg["strategies"][0]["position_scale"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "out/status.json"
    with pytest.raises((TypeError, ValueError), match="position_scale"):
        main(["status", "--config", str(path), "--out", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0, 0.0, True])
def test_invalid_capital_budget_is_rejected(tmp_path, weight):
    cfg = inputs(tmp_path)
    cfg["strategies"][0]["weight"] = weight
    with pytest.raises((TypeError, ValueError), match="weight"):
        allocate(cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nav", float("nan")),
        ("nav", -1.0),
        ("nav", float("inf")),
        ("weight", float("nan")),
        ("weight", float("inf")),
    ],
)
def test_nonfinite_nav_and_holdings_are_rejected(tmp_path, field, value):
    cfg = inputs(tmp_path)
    if field == "nav":
        pd.DataFrame({"date": ["2026-09-30"], "nav": [value]}).to_csv(
            tmp_path / "book_nav.csv", index=False
        )
    else:
        pd.DataFrame({"symbol": ["A"], "weight": [value]}).to_csv(
            tmp_path / "book_holdings.csv", index=False
        )
    with pytest.raises(ValueError, match=field):
        allocate(cfg)


def test_zero_and_full_scale_remain_valid_strict_json(tmp_path):
    cfg = inputs(tmp_path)
    for scale in [0.0, 0.5, 1.0]:
        cfg["strategies"][0]["position_scale"] = scale
        snapshot = allocate(cfg)
        assert snapshot.total_nav == 100000.0
        assert snapshot.combined_weights["A"] == scale
        json.dumps(snapshot.__dict__, allow_nan=False)
