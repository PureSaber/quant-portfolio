import json

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.allocation_v2 import allocate_research
from quant_portfolio.feasibility import diagnose_feasibility, problem_payload
from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.solver import OptimizationError, RiskMatrix


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


def test_nonfinite_failure_evidence_remains_valid_json_without_replacing_numbers_by_zero():
    error = OptimizationError(
        "numerical_failure", "test", {"residuals": np.array([np.inf, np.nan, 0.1])}
    )
    evidence = error.serializable_diagnostics()
    assert evidence["residuals"] == [
        {"non_finite_number": "inf"},
        {"non_finite_number": "nan"},
        0.1,
    ]
    json.dumps(evidence, allow_nan=False)
    assert np.isnan(error.diagnostics["residuals"][1])


def test_factor_risk_failure_does_not_attach_unused_nan_return_scenarios():
    assets = ["a", "b"]
    with pytest.raises(OptimizationError) as caught:
        allocate_research(
            pd.Series([1.0, 0.0], index=assets),
            pd.DataFrame(np.nan, index=range(252), columns=assets),
            pd.Series([0.0, 0.0], index=assets),
            None,
            invested_limit=0.8,
            max_weight=0.5,
            config={"mode": "cost_aware", "max_turnover": 0.2},
            covariance_override=pd.DataFrame(np.eye(2), index=assets, columns=assets),
        )
    assert caught.value.status == "infeasible"
    assert caught.value.diagnostics["problem"]["objective"]["scenarios"] is None
    json.dumps(caught.value.diagnostics, allow_nan=False)
