"""Constraint diagnostics and repair seeds; neither changes the strategy objective."""

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np

from quant_portfolio.solver import OptimizationError, _run, constraint_residuals, solve_portfolio


def problem_payload(context):
    """JSON-safe complete numerical input, including the factor representation."""
    risk = context.get("risk")
    result = {}
    for name, value in context.items():
        if name == "risk":
            result[name] = (
                None
                if risk is None
                else {
                    key: None if getattr(risk, key) is None else getattr(risk, key).tolist()
                    for key in ("covariance", "exposures", "factor_covariance", "specific_variance")
                }
            )
        elif name == "settings":
            result[name] = asdict(value)
        elif isinstance(value, np.ndarray):
            result[name] = value.tolist()
        else:
            result[name] = value
    return result


def minimum_risk_seed(context, timeout_seconds):
    """Auxiliary point inside the original continuous constraint set."""
    assets = context["assets"]
    result = solve_portfolio(
        np.zeros(len(assets)),
        context["risk"],
        context["current"],
        context["settings"],
        assets=assets,
        budget=context["budget"],
        lower=context["lower"],
        upper=context["upper"],
        factor_exposures=context["factor_exposures"],
        factor_bounds=context["factor_bounds"],
        benchmark=context["benchmark"],
        active_risk=context["settings"].max_tracking_error is not None,
        timeout_seconds=timeout_seconds,
    )
    return result.weights, result.diagnostics


def diagnose_feasibility(context, *, timeout_seconds=2.0):
    """Optimistic lower bounds, independently solved; never executable outputs.

    The no-turnover problem still retains budget, asset/group/factor bounds.
    The turnover floor retains the TE constraint but removes only turnover.
    Incomplete diagnostics cannot change the original solver classification.
    """
    import cvxpy as cp

    settings, risk = context["settings"], context["risk"]
    report = {"diagnostic_only": True, "publishable": False, "bounds": {}}
    if risk is None or settings.max_tracking_error is None:
        report["status"] = "not_applicable"
        return report
    started = perf_counter()
    deadline = started + timeout_seconds
    assets = context["assets"]
    w = cp.Variable(len(assets))
    base = [cp.sum(w) == context["budget"], w >= context["lower"], w <= context["upper"]]
    for group, cap in settings.group_caps.items():
        mask = [i for i, a in enumerate(assets) if settings.group_by_asset.get(a) == group]
        base.append(cp.sum(w[mask]) <= cap)
    for i, (_, (left, right)) in enumerate((context["factor_bounds"] or {}).items()):
        exposure = context["factor_exposures"][:, i] @ w
        base.extend([exposure >= left, exposure <= right])
    root = risk.root_expression(cp, w - context["benchmark"])
    turnover = cp.norm1(w - context["current"])
    problems = [
        ("te_without_turnover", cp.Minimize(cp.sum_squares(root)), base, True),
        (
            "te_with_turnover",
            cp.Minimize(cp.sum_squares(root)),
            base + [turnover <= settings.max_turnover],
            True,
        ),
        (
            "turnover_at_te_limit",
            cp.Minimize(turnover),
            base + [cp.norm(root, 2) <= settings.max_tracking_error],
            False,
        ),
    ]
    for name, objective, constraints, square_root in problems:
        item = {"attempts": []}
        problem = cp.Problem(objective, constraints)
        try:
            solved = _run(
                cp, problem, conic=True, deadline=deadline, diagnostics=item, max_iterations=2000
            )
            item["value"] = (
                float(np.sqrt(max(problem.value, 0)) if square_root else problem.value)
                if solved
                else None
            )
            if solved:
                audit_settings = replace(
                    settings,
                    max_turnover=(settings.max_turnover if name == "te_with_turnover" else 2.0),
                    max_tracking_error=(
                        settings.max_tracking_error if name == "turnover_at_te_limit" else None
                    ),
                )
                item["residuals"] = constraint_residuals(
                    np.asarray(w.value),
                    context["current"],
                    audit_settings,
                    risk=risk,
                    assets=assets,
                    budget=context["budget"],
                    lower=context["lower"],
                    upper=context["upper"],
                    factor_exposures=context["factor_exposures"],
                    factor_bounds=context["factor_bounds"],
                    benchmark=context["benchmark"],
                )
                item["max_residual"] = max(item["residuals"].values(), default=0)
                if item["max_residual"] > 1e-7:
                    item.update(status="numerical_failure", value=None)
        except OptimizationError as exc:
            item.update(status=exc.status, value=None)
        report["bounds"][name] = item
        if perf_counter() >= deadline:
            break
    report["seconds"] = perf_counter() - started
    report["status"] = (
        "complete"
        if (
            len(report["bounds"]) == 3
            and all(
                item.get("status") in {"optimal", "infeasible"}
                for item in report["bounds"].values()
            )
        )
        else "incomplete"
    )
    return report
