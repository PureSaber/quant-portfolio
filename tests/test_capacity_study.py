import copy
import json
import math

import pytest

from quant_portfolio.capacity_study import calibrate_costs, evaluate_capacity, main


def scan():
    return {
        "schema": "quant-portfolio.capacity-study/v1",
        "base_currency": "USD",
        "data_nature": "synthetic",
        "as_of": "2025-01-01T00:00:00Z",
        "max_liquidity_age_days": 1,
        "holdings_weights": {"A": 0.6, "B": -0.4},
        "trade_weights": {"A": 0.1, "B": -0.2},
        "max_participation": 0.1,
        "execution_days": 2,
        "liquidation_days": 3,
        "cost_budget_return": 0.002,
        "capital": [1000000, 4000000],
        "turnover_scales": [1, 2],
        "liquidity_scales": [1, 0.5],
        "impact_coefficients": [0, 0.2],
        "volatility_scales": [1, 2],
        "liquidity": [
            {
                "asset": k,
                "adv_base": adv,
                "daily_volatility": vol,
                "linear_rate": 0.001,
                "observed_at": "2025-01-01T00:00:00Z",
                "known_at": "2025-01-01T00:00:00Z",
                "source_ref": "synthetic:test",
            }
            for k, adv, vol in [("A", 1000000, 0.02), ("B", 2000000, 0.03)]
        ],
    }


def calibration():
    orders = []
    for i in range(8):
        time = f"2025-01-{i + 1:02d}T00:00:00Z"
        notional, adv, vol = (i + 1) * 1000, 100000, 0.02
        orders.append(
            {
                "order_id": str(i),
                "accepted_at": time,
                "last_fill_at": time,
                "known_at": time,
                "reference_at": time,
                "reference_known_at": time,
                "liquidity_known_at": time,
                "source_ref": "synthetic:order-group",
                "notional_base": notional,
                "adv_base": adv,
                "daily_volatility": vol,
                "signed_adverse_rate": 0.001 + 0.2 * vol * math.sqrt(notional / adv),
            }
        )
    return {
        "schema": "quant-portfolio.cost-calibration/v1",
        "data_nature": "synthetic",
        "training_end": "2025-01-04T00:00:00Z",
        "holdout_start": "2025-01-05T00:00:00Z",
        "evaluation_as_of": "2025-01-09T00:00:00Z",
        "minimum_orders": 3,
        "max_reference_age_seconds": 60,
        "orders": orders,
    }


def test_cost_formula_capacity_and_sensitivity_independent():
    result = evaluate_capacity(scan())
    assert len(result["scenarios"]) == 32
    selected = next(
        r
        for r in result["scenarios"]
        if r["capital"] == 1000000
        and r["turnover_scale"] == r["liquidity_scale"] == r["volatility_scale"] == 1
        and r["impact_coefficient"] == 0.2
    )
    expected_impact = 100000 * 0.2 * 0.02 * math.sqrt(50000 / 1000000)
    expected_impact += 200000 * 0.2 * 0.03 * math.sqrt(100000 / 2000000)
    assert selected["linear_cost"] == 300
    assert selected["square_root_cost"] == pytest.approx(expected_impact)
    assert selected["trade_capacity"] == 2000000
    assert selected["holding_capacity"]["capacity"] == pytest.approx(500000)
    assert selected["holding_capacity_breach"]
    quadruple = next(
        r
        for r in result["scenarios"]
        if r["capital"] == 4000000
        and r["turnover_scale"] == r["liquidity_scale"] == r["volatility_scale"] == 1
        and r["impact_coefficient"] == 0.2
    )
    assert quadruple["square_root_cost"] == pytest.approx(8 * expected_impact)
    assert quadruple["linear_cost"] == 4 * selected["linear_cost"]
    assert quadruple["participation_breach"]
    spec = scan()
    spec["turnover_scales"] = [0]
    spec["holdings_weights"] = {"A": 0, "B": 0}
    no_trade = evaluate_capacity(spec)["scenarios"][0]
    assert no_trade["total_cost"] == 0 and no_trade["trade_capacity"] is None
    assert no_trade["holding_capacity"]["capacity"] is None


def test_training_coefficients_holdout_isolation_and_identifiability():
    spec = calibration()
    result = calibrate_costs(spec)
    assert result["fit"]["linear_intercept"] == pytest.approx(0.001)
    assert result["fit"]["square_root_coefficient"] == pytest.approx(0.2)
    assert result["fit"]["holdout_weighted_mae_bps"] < 1e-10
    assert result["fit"]["linear_only_holdout_mae_bps"] > 0
    assert result["real_market_validation"] is False
    spec["orders"][-1]["signed_adverse_rate"] = 0.5
    changed = calibrate_costs(spec)
    assert changed["fit"]["square_root_coefficient"] == result["fit"]["square_root_coefficient"]
    assert changed["fit"]["holdout_weighted_mae_bps"] > 100
    constant = calibration()
    for row in constant["orders"]:
        row["notional_base"] = 1000
    assert calibrate_costs(constant)["reasons"] == ["impact_feature_not_identifiable"]
    spec["orders"][0]["known_at"] = "2025-01-05T00:00:00Z"
    spec["orders"][1]["last_fill_at"] = "2025-01-05T00:00:00Z"
    spec["orders"][1]["known_at"] = "2025-01-05T00:00:00Z"
    unavailable = calibrate_costs(spec)
    assert unavailable["status"] == "unavailable" and unavailable["fit"] is None
    assert unavailable["training_orders"] == 2


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(schema="wrong"),
        lambda s: s.update(data_nature="unknown"),
        lambda s: s.update(max_participation=2),
        lambda s: s.update(execution_days=True),
        lambda s: s.update(capital=[]),
        lambda s: s.update(capital=[1, 1]),
        lambda s: s["liquidity"][0].update(adv_base=0),
        lambda s: s["liquidity"].append(s["liquidity"][0]),
        lambda s: s["trade_weights"].pop("A"),
        lambda s: s["liquidity"][0].update(known_at="2026-01-01T00:00:00Z"),
        lambda s: s["liquidity"][0].update(known_at="2024-01-01T00:00:00Z"),
        lambda s: s["liquidity"][0].update(observed_at="2024-01-01T00:00:00Z"),
    ],
)
def test_scan_rejects_bad_inputs(change):
    spec = scan()
    change(spec)
    with pytest.raises((ValueError, TypeError)):
        evaluate_capacity(spec)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(schema="wrong"),
        lambda s: s.update(holdout_start=s["training_end"]),
        lambda s: s.update(minimum_orders=2),
        lambda s: s.update(data_nature="unknown"),
        lambda s: s["orders"].append(s["orders"][0]),
        lambda s: s["orders"][0].update(last_fill_at="2024-01-01T00:00:00Z"),
        lambda s: s["orders"][0].update(reference_known_at="2026-01-01T00:00:00Z"),
        lambda s: s["orders"][0].update(reference_at="2024-01-01T00:00:00Z"),
        lambda s: s["orders"][0].update(liquidity_known_at="2026-01-01T00:00:00Z"),
    ],
)
def test_calibration_rejects_bad_inputs(change):
    spec = calibration()
    change(spec)
    with pytest.raises((ValueError, TypeError)):
        calibrate_costs(spec)


def test_gap_unavailable_holdout_and_cli(tmp_path):
    spec = calibration()
    spec["holdout_start"] = "2025-01-07T00:00:00Z"
    assert "insufficient_holdout_orders" in calibrate_costs(spec)["reasons"]
    spec = calibration()
    spec["data_nature"] = "retrospective"
    assert calibrate_costs(spec)["real_market_validation"] == "requires_source_verification"
    for mode, data in [("scan", scan()), ("calibrate", calibration())]:
        source, target = tmp_path / (mode + ".json"), tmp_path / (mode + "-result.json")
        source.write_text(json.dumps(data), encoding="utf-8")
        main([mode, "--input", str(source), "--out", str(target)])
        assert json.loads(target.read_text())["input_sha256"]
    too_many = copy.deepcopy(scan())
    too_many["capital"] = list(range(1, 10001))
    with pytest.raises(ValueError, match="100000"):
        evaluate_capacity(too_many)
