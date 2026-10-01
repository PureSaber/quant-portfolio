"""Public CLI regressions for portfolio budgets and security identifiers."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from quant_portfolio._inputs import read_symbol_frame
from quant_portfolio.allocator import _blend_factor_scores


def _cli(*args):
    result = subprocess.run(
        [sys.executable, "-m", "quant_portfolio.cli", *map(str, args)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def _status(tmp_path, symbols, *, scale=0.5, factor_symbols=None, tilt=0.0, parquet=False):
    nav = tmp_path / "book_nav.csv"
    pd.DataFrame({"date": ["2026-09-29"], "nav": [100000]}).to_csv(nav, index=False)
    pd.DataFrame({"symbol": symbols, "weight": [0.6, 0.4]}).to_csv(
        tmp_path / "book_holdings.csv",
        index=False,
    )
    config = {"strategies": [{"name": "test", "nav": str(nav), "position_scale": scale}]}
    if factor_symbols is not None:
        path = tmp_path / ("factors.parquet" if parquet else "factors.csv")
        scores = pd.DataFrame(
            {
                "date": "2026-09-29",
                "symbol": factor_symbols,
                "momentum_20d": range(len(factor_symbols)),
            }
        )
        if parquet:
            scores.to_parquet(path, index=False)
        else:
            scores.to_csv(path, index=False)
        config["factor_scores"] = {"path": str(path), "weight": tilt}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "status.json"
    _cli("status", "--config", config_path, "--out", out)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("factor_symbols", [["AAA", "BBB"], ["AAA"], ["OTHER"]])
def test_zero_factor_weight_preserves_scaled_portfolio(tmp_path, factor_symbols):
    result = _status(tmp_path, ["AAA", "BBB"], factor_symbols=factor_symbols)
    assert result["combined_weights"] == {"AAA": 0.3, "BBB": 0.2}
    assert result["books"][0]["budget_weight"] == 0.5


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_positive_factor_tilt_preserves_invested_budget(tmp_path, scale):
    result = _status(
        tmp_path, ["AAA", "BBB"], scale=scale, factor_symbols=["AAA", "BBB"], tilt=0.25
    )
    weights = result["combined_weights"]
    assert result["total_nav"] == 100000
    assert result["books"][0]["capital_weight"] == 1.0
    assert result["books"][0]["cash_weight"] == pytest.approx(1.0 - scale)
    assert set(weights) == {"AAA", "BBB"}
    assert sum(weights.values()) == pytest.approx(scale, abs=1e-6)
    assert all(value >= 0 for value in weights.values())
    if scale:
        assert weights["AAA"] < 0.6 * scale
        assert weights["BBB"] > 0.4 * scale


def test_missing_scores_leave_holdings_unchanged(tmp_path):
    result = _status(tmp_path, ["AAA", "BBB"], factor_symbols=["AAA", "OTHER"], tilt=0.5)
    assert result["combined_weights"] == {"AAA": 0.3, "BBB": 0.2}


def test_partial_scores_redistribute_only_the_scored_budget():
    result = _blend_factor_scores(
        {"A": 0.2, "B": 0.3, "C": 0.1},
        pd.Series({"A": -1.0, "B": 1.0}),
        0.25,
    )
    assert result["C"] == 0.1
    assert result["A"] < 0.2
    assert result["B"] > 0.3
    assert result["A"] + result["B"] == pytest.approx(0.5)


def test_factor_tilt_cannot_silently_erase_invested_capital():
    with pytest.raises(ValueError, match="removed every funded"):
        _blend_factor_scores({"A": 0.5, "B": 0.0}, pd.Series({"A": -1.0, "B": 1.0}), 2.0)


@pytest.mark.parametrize("symbol", ["", "  ", " 000001"])
def test_symbol_reader_rejects_invalid_identifiers(tmp_path, symbol):
    path = tmp_path / "holdings.csv"
    pd.DataFrame({"symbol": [symbol], "weight": [0.5]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="symbol must"):
        read_symbol_frame(path)


def test_parquet_identifiers_must_already_be_strings(tmp_path):
    path = tmp_path / "holdings.parquet"
    pd.DataFrame({"symbol": [1, 600519], "weight": [0.5, 0.5]}).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="symbol must"):
        read_symbol_frame(path)


@pytest.mark.parametrize("parquet", [False, True])
@pytest.mark.parametrize("symbols", [["000001", "600519"], ["AAA", "BBB"], ["NA", "NULL"]])
@pytest.mark.parametrize("tilt", [0.0, 0.25])
def test_status_preserves_security_identifiers(tmp_path, symbols, parquet, tilt):
    result = _status(tmp_path, symbols, factor_symbols=symbols, parquet=parquet, tilt=tilt)
    assert set(result["combined_weights"]) == set(symbols)
    assert sum(result["combined_weights"].values()) == pytest.approx(0.5, abs=1e-6)
    assert all(value > 0 for value in result["combined_weights"].values())
    if tilt == 0:
        assert result["combined_weights"] == {symbols[0]: 0.3, symbols[1]: 0.2}


@pytest.mark.parametrize("symbols", [["000001", "600519"], ["AAA", "BBB"], ["NA", "NULL"]])
def test_optimize_cli_aligns_security_identifiers_in_every_input(tmp_path, symbols):
    for filename, column, values in [
        ("expected", "expected_return", [0.12, 0.08]),
        ("current", "weight", [0.6, 0.4]),
        ("costs", "cost", [0.001, 0.002]),
        ("liquidity", "average_daily_value", [1e7, 2e6]),
    ]:
        pd.DataFrame({"symbol": symbols, column: values}).to_csv(
            tmp_path / f"{filename}.csv",
            index=False,
        )
    pd.DataFrame(
        {
            "date": ["2026-09-25", "2026-09-28", "2026-09-29"],
            symbols[0]: [0.01, -0.01, 0.02],
            symbols[1]: [0.0, 0.01, -0.01],
        }
    ).to_csv(tmp_path / "returns.csv", index=False)
    out = tmp_path / "optimized.json"
    _cli(
        "optimize",
        "--expected-returns",
        tmp_path / "expected.csv",
        "--returns",
        tmp_path / "returns.csv",
        "--current-weights",
        tmp_path / "current.csv",
        "--linear-costs",
        tmp_path / "costs.csv",
        "--liquidity",
        tmp_path / "liquidity.csv",
        "--out",
        out,
    )
    result = json.loads(out.read_text(encoding="utf-8"))
    assert set(result["weights"]) == set(symbols)
    assert sum(result["weights"].values()) == pytest.approx(1.0)
    assert result["capacity"]["capacity"] > 0
