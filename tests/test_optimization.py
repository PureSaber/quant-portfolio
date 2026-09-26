import numpy as np
import pandas as pd
import pytest

from quant_portfolio.optimization import (
    OptimizationConstraints,
    estimate_capacity,
    estimate_covariance,
    optimize_mean_variance,
    research_allocation_weights,
    square_root_impact_cost,
    validate_research_allocation,
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


def test_research_allocation_modes_share_closed_constraints() -> None:
    assets = ["A", "B", "C"]
    scores = pd.Series({"A": 0.2, "B": 0.1, "C": -0.1})
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.02, 0.01],
            "B": [0.005, 0.004, -0.003, 0.002, -0.004],
            "C": [0.02, -0.03, 0.01, 0.02, -0.01],
        }
    )
    base = {"lookback": 5, "min_observations": 5}
    for mode in ("equal", "inverse_vol", "cost_aware"):
        weights = research_allocation_weights(
            scores,
            returns,
            pd.Series({"A": 0.3, "B": 0.2, "C": 0.1}),
            pd.Series(0.001, index=assets),
            invested_limit=0.6,
            max_weight=0.3,
            config={"mode": mode, **base},
        )
        assert weights.sum() == pytest.approx(0.6)
        assert weights.max() <= 0.3 + 1e-8
        assert list(weights.index) == assets
    inverse = research_allocation_weights(
        scores,
        returns,
        None,
        None,
        invested_limit=0.6,
        max_weight=0.4,
        config={"mode": "inverse_vol", **base},
    )
    assert inverse["B"] > inverse["A"]


def test_research_allocation_rejects_open_or_unsupported_inputs() -> None:
    normalized = validate_research_allocation({"mode": "equal"})
    assert normalized["lookback"] == 20
    with pytest.raises(ValueError, match="unknown fields"):
        validate_research_allocation({"mode": "equal", "fallback": True})
    with pytest.raises(ValueError, match="insufficient"):
        research_allocation_weights(
            pd.Series({"A": 1.0, "B": 0.0}),
            pd.DataFrame({"A": [0.1], "B": [0.1]}),
            None,
            None,
            invested_limit=1.0,
            max_weight=0.6,
            config={"mode": "inverse_vol"},
        )
