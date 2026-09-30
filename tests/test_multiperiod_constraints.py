import numpy as np
import pandas as pd
import pytest

from quant_portfolio.optimization import OptimizationConstraints, optimize_multiperiod


def test_multiperiod_enforces_group_caps_on_every_date():
    assets = ["A", "B", "C"]
    current = pd.Series([0.2, 0.2, 0.6], index=assets)
    expected = pd.DataFrame([[0.3, 0.2, 0.0], [0.0, 0.3, 0.0]], columns=assets)
    covariance = pd.DataFrame(np.eye(3) * 0.01, index=assets, columns=assets)
    settings = OptimizationConstraints(
        max_weight=0.7,
        max_turnover=0.2,
        group_by_asset={"A": "tech", "B": "tech", "C": "other"},
        group_caps={"tech": 0.4},
    )
    result = optimize_multiperiod(
        expected, covariance, current_weights=current, constraints=settings, risk_aversion=1.0
    )
    assert result.converged
    previous = current
    for _, weights in result.weights.iterrows():
        assert weights.sum() == pytest.approx(1.0)
        assert weights.min() >= -1e-8
        assert weights.max() <= 0.7 + 1e-8
        assert weights[["A", "B"]].sum() <= 0.4 + 1e-8
        assert (weights - previous).abs().sum() <= 0.2 + 1e-8
        previous = weights
    assert result.weights.iloc[0]["A"] > current["A"]


def test_multiperiod_rejects_group_cap_and_turnover_conflict():
    assets = ["A", "B", "C"]
    with pytest.raises(ValueError, match="infeasible"):
        optimize_multiperiod(
            pd.DataFrame([[0.3, 0.3, 0.0]], columns=assets),
            pd.DataFrame(np.eye(3) * 0.01, index=assets, columns=assets),
            current_weights=pd.Series([0.4, 0.4, 0.2], index=assets),
            constraints=OptimizationConstraints(
                max_weight=1.0,
                max_turnover=0.2,
                group_by_asset={"A": "tech", "B": "tech"},
                group_caps={"tech": 0.4},
            ),
        )


@pytest.mark.parametrize("max_turnover", [0.0, 0.3, 0.79])
def test_multiperiod_rejects_insufficient_initial_turnover(max_turnover):
    assets = ["A", "B"]
    with pytest.raises(ValueError, match="infeasible"):
        optimize_multiperiod(
            pd.DataFrame([[0.1, 0.0]], columns=assets),
            pd.DataFrame(np.eye(2) * 0.01, index=assets, columns=assets),
            current_weights=pd.Series([1.0, 0.0], index=assets),
            constraints=OptimizationConstraints(max_weight=0.6, max_turnover=max_turnover),
        )


@pytest.mark.parametrize("current", [[1.0, 0.0], [0.0, 0.0], [0.3, 0.2]])
def test_multiperiod_charges_all_trades_from_real_current_weights(current):
    assets = ["A", "B"]
    expected = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]], columns=assets)
    covariance = pd.DataFrame(np.eye(2) * 0.01, index=assets, columns=assets)
    previous = pd.Series(current, index=assets)
    costs = pd.Series([0.01, 0.02], index=assets)
    result = optimize_multiperiod(
        expected,
        covariance,
        current_weights=previous,
        linear_costs=costs,
        risk_aversion=1.0,
        constraints=OptimizationConstraints(max_weight=0.6, max_turnover=1.0),
    )
    objective = 0.0
    for date, weights in result.weights.iterrows():
        trades = (weights - previous).abs()
        assert trades.sum() <= 1.0 + 1e-8
        objective += expected.loc[date] @ weights - 0.5 * (weights @ covariance @ weights)
        objective -= costs @ trades
        previous = weights
    assert result.objective == pytest.approx(objective, abs=1e-10)


def test_multiperiod_accepts_exactly_sufficient_turnover():
    assets = ["A", "B"]
    result = optimize_multiperiod(
        pd.DataFrame([[0.1, 0.0]], columns=assets),
        pd.DataFrame(np.eye(2) * 0.01, index=assets, columns=assets),
        current_weights=pd.Series([1.0, 0.0], index=assets),
        linear_costs=pd.Series(0.01, index=assets),
        risk_aversion=1.0,
        constraints=OptimizationConstraints(max_weight=0.6, max_turnover=0.8),
    )
    assert result.weights.iloc[0].tolist() == pytest.approx([0.6, 0.4])
    assert result.objective == pytest.approx(0.0494)


@pytest.mark.parametrize(
    "current",
    [
        pd.Series({"A": float("inf"), "B": 0.0}),
        pd.Series({"A": 0.6, "B": 0.6}),
        pd.Series({"A": 0.5}),
        pd.Series({"A": 0.5, "B": 0.4, "OUTSIDE": 0.1}),
    ],
)
def test_multiperiod_rejects_unaccountable_initial_positions(current):
    with pytest.raises(ValueError, match="current_weights"):
        optimize_multiperiod(
            pd.DataFrame([[0.1, 0.0]], columns=["A", "B"]),
            pd.DataFrame(np.eye(2) * 0.01, index=["A", "B"], columns=["A", "B"]),
            current_weights=current,
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"tolerance": float("inf")}, "tolerance"),
        ({"linear_costs": pd.Series({"A": float("inf"), "B": 0.0})}, "linear_costs"),
    ],
)
def test_multiperiod_rejects_invalid_solver_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        optimize_multiperiod(
            pd.DataFrame([[0.1, 0.0]], columns=["A", "B"]),
            pd.DataFrame(np.eye(2) * 0.01, index=["A", "B"], columns=["A", "B"]),
            **kwargs,
        )


def test_multiperiod_independently_checks_returned_plan(monkeypatch):
    # A regressed solver must not publish an infeasible plan as converged.
    monkeypatch.setattr(
        "quant_portfolio.optimization._project_joint_feasible_set",
        lambda *args, **kwargs: np.array([0.8, 0.2]),
    )
    with pytest.raises(RuntimeError, match="violates constraints"):
        optimize_multiperiod(
            pd.DataFrame([[0.1, 0.0]], columns=["A", "B"]),
            pd.DataFrame(np.eye(2) * 0.01, index=["A", "B"], columns=["A", "B"]),
            constraints=OptimizationConstraints(max_weight=0.6),
        )
