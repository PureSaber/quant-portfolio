import numpy as np
import pytest

from quant_portfolio.feasibility import diagnose_feasibility, problem_payload
from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.solver import RiskMatrix


def context(turnover=2, te=0.1):
    return {
        "assets": ["a", "b"],
        "risk": RiskMatrix(covariance=np.eye(2)),
        "settings": OptimizationConstraints(
            max_weight=0.5, max_turnover=turnover, max_tracking_error=te
        ),
        "benchmark": np.array([1.0, 0]),
        "current": np.array([0.5, 0]),
        "lower": np.zeros(2),
        "upper": np.full(2, 0.5),
        "budget": 0.8,
        "factor_exposures": None,
        "factor_bounds": None,
    }


def test_structural_floor_is_diagnostic_and_never_a_trade_target():
    result = diagnose_feasibility(context())
    assert result["status"] == "complete"
    assert result["publishable"] is False
    assert result["bounds"]["te_without_turnover"]["value"] == pytest.approx(
        np.sqrt(0.34), abs=1e-7
    )
    assert result["bounds"]["turnover_at_te_limit"]["status"] == "infeasible"
    assert "weights" not in result


def test_turnover_floor_distinguishes_budget_conflict_and_serializes_inputs():
    result = diagnose_feasibility(context(turnover=0.2, te=0.7))
    assert result["bounds"]["te_with_turnover"]["status"] == "infeasible"
    assert result["bounds"]["turnover_at_te_limit"]["value"] == pytest.approx(0.3, abs=1e-7)
    payload = problem_payload(context())
    assert payload["risk"]["covariance"] == [[1, 0], [0, 1]]
    assert payload["settings"]["max_weight"] == 0.5
