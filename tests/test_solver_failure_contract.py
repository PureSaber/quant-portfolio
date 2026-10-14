from decimal import Decimal

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from quant_portfolio import OptimizationConstraints, RiskSnapshot
from quant_portfolio.executable import repair_integer_target
from quant_portfolio.solver import OptimizationError, RiskMatrix, solve_portfolio


def problem(**kwargs):
    return solve_portfolio(
        [0.05, 0.02],
        RiskMatrix(covariance=np.diag([0.04, 0.09])),
        [0.5, 0.5],
        OptimizationConstraints(max_weight=0.8, max_turnover=0.2),
        **kwargs,
    )


@pytest.mark.parametrize("failure", ["exception", "inaccurate"])
def test_osqp_failure_retries_same_problem_without_publishing_inaccurate(monkeypatch, failure):
    reference = problem()
    original = cp.Problem.solve
    calls = []

    def solve(self, *args, **kwargs):
        calls.append((id(self), kwargs["solver"]))
        if kwargs["solver"] == "OSQP":
            if failure == "exception":
                raise cp.error.SolverError("injected numerical failure")
            original(self, *args, **kwargs)
            self._status = cp.OPTIMAL_INACCURATE
            return self.value
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", solve)
    actual = problem()
    assert [name for _, name in calls] == ["OSQP", "CLARABEL"]
    assert calls[0][0] == calls[1][0]
    np.testing.assert_allclose(actual.weights, reference.weights, atol=1e-7)
    assert actual.diagnostics["solver"] == "CLARABEL"


def test_two_inaccurate_results_are_not_publishable(monkeypatch):
    original = cp.Problem.solve

    def solve(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self._status = cp.OPTIMAL_INACCURATE
        return self.value

    monkeypatch.setattr(cp.Problem, "solve", solve)
    with pytest.raises(OptimizationError) as caught:
        problem()
    assert caught.value.status == "numerical_failure"
    assert len(caught.value.diagnostics["attempts"]) == 2


def test_zero_risk_positive_return_has_no_finite_sharpe():
    with pytest.raises(OptimizationError) as caught:
        solve_portfolio(
            [1, 2],
            RiskMatrix(covariance=np.zeros((2, 2))),
            [0.5, 0.5],
            OptimizationConstraints(max_weight=1),
            kind="max_sharpe",
        )
    assert caught.value.status == "undefined_ratio"


def test_factor_snapshot_cannot_change_after_validation():
    snapshot = RiskSnapshot(
        estimated_at="2026-01-01T00:00Z",
        available_at="2026-01-01T00:00Z",
        horizon_sessions=1,
        annualization=252,
        model_version="fixed",
        input_digest="fixture",
        covariance=pd.DataFrame(np.eye(2), index=list("AB"), columns=list("AB")),
    )
    snapshot.covariance.iloc[0, 0] = 2
    with pytest.raises(ValueError, match="mutated"):
        snapshot.matrix(list("AB"), "2026-01-02T00:00Z")


def test_known_unsellable_holdings_and_fractional_quantity_rejection():
    args = {
        "weights": pd.Series({"A": 0.2, "B": 0.8}),
        "current": {"A": 500},
        "prices": {"A": 10, "B": 10},
        "lots": {"A": 100, "B": 100},
        "cash": Decimal(5000),
        "nav": Decimal(10000),
        "quote_trade": lambda a, q: (-Decimal(q) * 10, Decimal(0)),
        "constraint_residuals": lambda w: {},
        "max_invested": 1,
        "max_sell": {"A": 0},
        "max_buy": {"A": 0, "B": 1000},
    }
    result = repair_integer_target(**args)
    assert result.quantities == {"A": 500, "B": 500}
    assert result.cash == 0
    with pytest.raises(ValueError, match="integers"):
        repair_integer_target(**{**args, "current": {"A": 500.5}})
