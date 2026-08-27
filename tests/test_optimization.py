import numpy as np
import pandas as pd
import pytest

from quant_portfolio.optimization import (
    OptimizationConstraints,
    estimate_capacity,
    estimate_covariance,
    optimize_mean_variance,
    square_root_impact_cost,
)


def test_covariance_is_positive_semidefinite() -> None:
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.01, 0.03], "B": [0.0, 0.01, -0.01, 0.02]})
    covariance = estimate_covariance(returns, shrinkage=0.3)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_optimizer_respects_asset_group_and_turnover_constraints() -> None:
    assets = ["A", "B", "C", "D"]
    expected = pd.Series([0.20, 0.15, 0.08, 0.05], index=assets)
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)
    current = pd.Series(0.25, index=assets)
    constraints = OptimizationConstraints(
        max_weight=0.4,
        max_turnover=0.4,
        group_by_asset={"A": "tech", "B": "tech", "C": "other", "D": "other"},
        group_caps={"tech": 0.55},
    )
    result = optimize_mean_variance(
        expected,
        covariance,
        current_weights=current,
        risk_aversion=3.0,
        constraints=constraints,
    )
    assert result.weights.sum() == pytest.approx(1.0)
    assert result.weights.max() <= 0.4 + 1e-8
    assert result.group_weights["tech"] <= 0.55 + 1e-8
    assert result.turnover <= 0.4 + 1e-8


def test_capacity_and_market_impact_bind_to_liquidity() -> None:
    weights = pd.Series({"A": 0.6, "B": 0.4})
    adv = pd.Series({"A": 10_000_000.0, "B": 2_000_000.0})
    capacity = estimate_capacity(weights, adv, max_participation=0.1)
    assert capacity["binding_asset"] == "B"
    assert capacity["capacity"] == pytest.approx(500_000.0)

    impact = square_root_impact_cost(
        pd.Series({"A": 1_000_000.0}),
        pd.Series({"A": 10_000_000.0}),
        pd.Series({"A": 0.02}),
    )
    assert impact.loc["A", "impact_cost"] > 0
