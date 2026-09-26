import numpy as np
import pandas as pd
import pytest

from quant_portfolio import optimization
from quant_portfolio.optimization import (
    OptimizationConstraints,
    OptimizationResult,
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


@pytest.mark.parametrize("mode", ["equal", "inverse_vol", "cost_aware"])
def test_research_allocation_turnover_includes_dropped_holdings(mode: str) -> None:
    scores = pd.Series({"A": 0.2, "B": 0.1})
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.02, 0.01],
            "B": [0.005, 0.004, -0.003, 0.002, -0.004],
        }
    )
    with pytest.raises(ValueError, match="max_turnover is infeasible"):
        research_allocation_weights(
            scores,
            returns,
            pd.Series({"A": 0.3, "B": 0.3, "DROPPED": 0.4}),
            pd.Series(0.001, index=scores.index),
            invested_limit=0.6,
            max_weight=0.4,
            config={
                "mode": mode,
                "lookback": 5,
                "min_observations": 5,
                "max_turnover": 0.3,
            },
        )


@pytest.mark.parametrize("mode", ["equal", "inverse_vol", "cost_aware"])
def test_research_allocation_zero_portfolio_does_not_fabricate_current_weights(
    mode: str,
) -> None:
    scores = pd.Series({"A": 0.2, "B": 0.1})
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.01, 0.02, -0.02, 0.01],
            "B": [0.005, 0.004, -0.003, 0.002, -0.004],
        }
    )
    with pytest.raises(ValueError, match="max_turnover is infeasible"):
        research_allocation_weights(
            scores,
            returns,
            pd.Series(dtype=float),
            pd.Series(0.001, index=scores.index),
            invested_limit=0.6,
            max_weight=0.4,
            config={
                "mode": mode,
                "lookback": 5,
                "min_observations": 5,
                "max_turnover": 0.0,
            },
        )


def test_equal_allocation_moves_toward_target_within_turnover_limit() -> None:
    weights = research_allocation_weights(
        pd.Series({"A": 0.2, "B": 0.1}),
        None,
        pd.Series({"A": 0.5, "B": 0.1}),
        None,
        invested_limit=0.6,
        max_weight=0.5,
        config={"mode": "equal", "max_turnover": 0.2},
    )
    assert weights.to_dict() == pytest.approx({"A": 0.4, "B": 0.2})
    assert (weights - pd.Series({"A": 0.5, "B": 0.1})).abs().sum() <= 0.2 + 1e-12


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


def test_optimizer_rejects_group_turnover_conflict_instead_of_false_convergence() -> None:
    assets = ["A", "B", "C", "D"]
    expected = pd.Series([0.20, 0.15, 0.08, 0.05], index=assets)
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)
    current = pd.Series([0.4, 0.4, 0.1, 0.1], index=assets)
    constraints = OptimizationConstraints(
        max_weight=0.6,
        max_turnover=0.1,
        group_by_asset={"A": "tech", "B": "tech", "C": "other", "D": "other"},
        group_caps={"tech": 0.5},
    )

    with pytest.raises(
        ValueError,
        match="minimum required turnover 0.6 exceeds max_turnover 0.1",
    ):
        optimize_mean_variance(
            expected,
            covariance,
            current_weights=current,
            constraints=constraints,
        )


@pytest.mark.parametrize(
    "current",
    [
        pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        pd.Series({"A": 0.10, "B": 0.20, "C": 0.30, "D": 0.40}),
    ],
)
def test_optimizer_independently_verifies_every_constraint_from_different_starts(
    current: pd.Series,
) -> None:
    assets = ["A", "B", "C", "D"]
    expected = pd.Series([0.20, 0.15, 0.08, 0.05], index=assets)
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)
    constraints = OptimizationConstraints(
        min_weight=0.05,
        max_weight=0.45,
        max_turnover=0.6,
        group_by_asset={"A": "tech", "B": "tech", "C": "other", "D": "other"},
        group_caps={"tech": 0.55},
    )
    exposures = pd.DataFrame({"beta": [1.2, 1.0, 0.8, 0.6]}, index=assets)
    result = optimize_mean_variance(
        expected,
        covariance,
        current_weights=current,
        constraints=constraints,
        factor_exposures=exposures,
        factor_bounds={"beta": (0.85, 0.90)},
    )

    assert result.converged
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (result.weights >= constraints.min_weight - 1e-9).all()
    assert (result.weights <= constraints.max_weight + 1e-9).all()
    assert result.weights[["A", "B"]].sum() <= 0.55 + 1e-9
    assert (result.weights - current).abs().sum() <= 0.6 + 1e-9
    beta = float(exposures["beta"] @ result.weights)
    assert 0.85 - 1e-9 <= beta <= 0.90 + 1e-9


def test_research_factor_bounds_use_absolute_target_weight_semantics() -> None:
    assets = ["A", "B"]
    exposures = pd.DataFrame({"beta": [1.2, 0.4]}, index=assets)
    weights = research_allocation_weights(
        pd.Series({"A": 0.2, "B": 0.1}),
        None,
        pd.Series({"A": 0.3, "B": 0.3}),
        None,
        invested_limit=0.6,
        max_weight=0.5,
        config={"mode": "cost_aware", "max_turnover": 1.0},
        factor_exposures=exposures,
        factor_bounds={"beta": (0.45, 0.50)},
        covariance_override=pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets),
    )

    absolute_beta = float(exposures["beta"] @ weights)
    assert weights.sum() == pytest.approx(0.6)
    assert 0.45 - 1e-9 <= absolute_beta <= 0.50 + 1e-9


def test_optimizer_rejects_incomplete_or_nonfinite_model_inputs() -> None:
    expected = pd.Series({"A": 0.1, "B": 0.2})
    covariance = pd.DataFrame(np.eye(2), index=expected.index, columns=expected.index)

    with pytest.raises(ValueError, match="expected_returns must be finite"):
        optimize_mean_variance(pd.Series({"A": 0.1, "B": np.nan}), covariance)
    invalid_covariance = covariance.copy()
    invalid_covariance.loc["A", "B"] = np.inf
    with pytest.raises(ValueError, match="covariance must contain finite"):
        optimize_mean_variance(expected, invalid_covariance)
    with pytest.raises(ValueError, match="linear_costs is missing assets"):
        optimize_mean_variance(expected, covariance, linear_costs=pd.Series({"A": 0.001}))
    with pytest.raises(ValueError, match="linear_costs must be finite"):
        optimize_mean_variance(
            expected,
            covariance,
            linear_costs=pd.Series({"A": 0.001, "B": np.inf}),
        )


def test_covariance_estimation_rejects_missing_pair_and_nonfinite_returns() -> None:
    no_pair = pd.DataFrame({"A": [0.1, 0.2, np.nan], "B": [np.nan, 0.1, 0.2]})
    with pytest.raises(ValueError, match="cannot estimate covariance for every asset pair"):
        estimate_covariance(no_pair)
    with pytest.raises(ValueError, match="finite observations"):
        estimate_covariance(pd.DataFrame({"A": [0.1, np.inf], "B": [0.2, 0.3]}))


def test_factor_constraint_validation_and_infeasibility_are_explicit() -> None:
    assets = ["A", "B"]
    expected = pd.Series([0.1, 0.2], index=assets)
    covariance = pd.DataFrame(np.eye(2), index=assets, columns=assets)
    exposures = pd.DataFrame({"beta": [1.0, 1.0]}, index=assets)

    with pytest.raises(ValueError, match="provided together"):
        optimize_mean_variance(expected, covariance, factor_exposures=exposures)
    with pytest.raises(ValueError, match="factor_bounds.*cannot be satisfied"):
        optimize_mean_variance(
            expected,
            covariance,
            factor_exposures=exposures,
            factor_bounds={"beta": (0.0, 0.5)},
        )


def test_research_cost_aware_rejects_optimizer_non_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def not_converged(*args: object, **kwargs: object) -> OptimizationResult:
        expected = args[0]
        assert isinstance(expected, pd.Series)
        return OptimizationResult(
            weights=pd.Series(0.5, index=expected.index),
            expected_return=0.0,
            volatility=0.0,
            turnover=0.0,
            objective=0.0,
            group_weights={"__ungrouped__": 1.0},
            converged=False,
            iterations=2000,
        )

    monkeypatch.setattr(optimization, "optimize_mean_variance", not_converged)
    assets = ["A", "B"]
    with pytest.raises(RuntimeError, match="did not converge after 2000 iterations"):
        research_allocation_weights(
            pd.Series({"A": 0.2, "B": 0.1}),
            None,
            pd.Series({"A": 0.5, "B": 0.5}),
            None,
            invested_limit=1.0,
            max_weight=0.8,
            config={"mode": "cost_aware"},
            covariance_override=pd.DataFrame(np.eye(2), index=assets, columns=assets),
        )
