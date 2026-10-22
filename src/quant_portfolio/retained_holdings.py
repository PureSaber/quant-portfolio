"""Audit today's held portfolio after a failed search without running a solver."""

import numpy as np

from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.solver import RiskMatrix, constraint_residuals


def audit_retained_holdings(problem, weights, *, cash, tradability_known, tolerance=1e-7):
    result = {
        "status": "unknown",
        "hold_permitted": False,
        "published": False,
        "residuals": {},
        "reason": "missing_current_constraint_evidence",
    }
    if not problem or not np.isfinite(cash) or cash < 0 or tradability_known is not True:
        return result
    try:
        assets = problem["assets"]
        if set(weights) - set(assets):
            return result
        vector = np.array([weights.get(a, 0) for a in assets], dtype=float)
        original = np.array(problem["current"], dtype=float)
        if not np.isfinite(vector).all() or not np.allclose(vector, original, atol=1e-10, rtol=0):
            return {**result, "reason": "current_holdings_differ_from_failed_problem"}
        risk = problem.get("risk")
        risk = (
            None
            if risk is None
            else RiskMatrix(
                **{k: None if v is None else np.asarray(v, dtype=float) for k, v in risk.items()}
            )
        )
        residuals = constraint_residuals(
            vector,
            vector,
            OptimizationConstraints(**problem["settings"]),
            assets=assets,
            budget=problem["budget"],
            budget_equality=False,
            lower=np.asarray(problem["lower"]),
            upper=np.asarray(problem["upper"]),
            factor_exposures=None
            if problem.get("factor_exposures") is None
            else np.asarray(problem["factor_exposures"]),
            factor_bounds=problem.get("factor_bounds"),
            risk=risk,
            benchmark=None
            if problem.get("benchmark") is None
            else np.asarray(problem["benchmark"]),
            detailed_asset_bounds=True,
        )
        complete = bool(residuals) and all(np.isfinite(v) for v in residuals.values())
        passed = complete and max(residuals.values()) <= tolerance
        return {
            **result,
            "status": "compliant" if passed else "noncompliant" if complete else "unknown",
            "hold_permitted": passed,
            "residuals": residuals,
            "reason": "today_constraints_pass" if passed else "today_constraints_fail",
            "tolerance": tolerance,
        }
    except (KeyError, ValueError, TypeError, AttributeError):
        return result


def publishing_summary(decisions):
    """Failed searches stay in the denominator; never call them successful holds."""
    total = len(decisions)
    published = 0
    streak = longest = 0
    causes = {}
    for row in decisions:
        ok = row.get("published", bool(row.get("targets"))) is True
        published += int(ok)
        streak = 0 if ok else streak + 1
        longest = max(longest, streak)
        if not ok:
            cause = row.get("status", "no_target")
            causes[cause] = causes.get(cause, 0) + 1
    return {
        "decision_count": total,
        "published_count": published,
        "publish_rate": published / total if total else None,
        "longest_without_target": longest,
        "causes": causes,
    }
