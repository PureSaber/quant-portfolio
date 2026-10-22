from dataclasses import asdict

from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.retained_holdings import audit_retained_holdings, publishing_summary


def problem(weight=0.4):
    return {
        "assets": ["A"],
        "current": [weight],
        "settings": asdict(OptimizationConstraints(max_weight=0.5)),
        "budget": 0.8,
        "lower": [0],
        "upper": [0.5],
        "risk": None,
        "factor_exposures": None,
        "factor_bounds": None,
        "benchmark": None,
    }


def test_yesterday_feasible_today_drift_is_not_a_successful_hold():
    assert audit_retained_holdings(problem(), {"A": 0.4}, cash=600, tradability_known=True)[
        "hold_permitted"
    ]
    failed = audit_retained_holdings(problem(0.6), {"A": 0.6}, cash=400, tradability_known=True)
    assert failed["status"] == "noncompliant"
    assert not failed["published"]
    assert not audit_retained_holdings(problem(), {"A": 0.4}, cash=600, tradability_known=False)[
        "hold_permitted"
    ]
    assert not audit_retained_holdings(None, {}, cash=600, tradability_known=True)["hold_permitted"]
    assert not audit_retained_holdings(problem(), {"A": 0.3}, cash=600, tradability_known=True)[
        "hold_permitted"
    ]


def test_full_account_factor_and_tracking_error_constraints():
    p = problem()
    p.update(risk={"covariance": [[0.04]]}, benchmark=[1])
    p["settings"]["max_tracking_error"] = 0.03
    assert (
        audit_retained_holdings(p, {"A": 0.4}, cash=600, tradability_known=True)["status"]
        == "noncompliant"
    )
    p["settings"]["max_tracking_error"] = None
    p.update(factor_exposures=[[1]], factor_bounds={"factor": [0.1, 0.3]})
    assert not audit_retained_holdings(p, {"A": 0.4}, cash=600, tradability_known=True)[
        "hold_permitted"
    ]


def test_summary_preserves_gaps_and_causes():
    result = publishing_summary(
        [
            {"targets": {"A": 1}},
            {"published": False, "status": "timeout"},
            {"published": False, "status": "infeasible"},
            {"targets": {"A": 2}},
        ]
    )
    assert result["publish_rate"] == 0.5
    assert result["longest_without_target"] == 2
    assert result["causes"] == {"timeout": 1, "infeasible": 1}
