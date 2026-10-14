"""Auditable convex portfolio problems; no strategy-changing solver fallback."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


class OptimizationError(ValueError, RuntimeError):
    """A classified failure. A numerical failure is not proof of infeasibility."""

    def __init__(self, status: str, message: str, diagnostics=None):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True)
class RiskMatrix:
    """Covariance or X F X' + diag(D), without forming the asset matrix."""

    covariance: np.ndarray | None = None
    exposures: np.ndarray | None = None
    factor_covariance: np.ndarray | None = None
    specific_variance: np.ndarray | None = None

    def __post_init__(self):
        dense = self.covariance is not None
        factor = any(
            x is not None for x in (self.exposures, self.factor_covariance, self.specific_variance)
        )
        if dense == factor:
            raise ValueError("provide exactly one dense or factor risk representation")
        names = (
            ("covariance",) if dense else ("exposures", "factor_covariance", "specific_variance")
        )
        for name in names:
            value = getattr(self, name)
            if value is None:
                raise ValueError("factor risk requires X, F and D")
            value = np.array(value, dtype=float, copy=True)
            if not np.isfinite(value).all():
                raise ValueError("risk matrices must be finite")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        matrix = self.covariance if dense else self.factor_covariance
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
            raise ValueError("covariance must be square")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        if np.linalg.eigvalsh(matrix).min() < -1e-12:
            raise ValueError("covariance must be positive semidefinite")
        if not dense:
            x, d = self.exposures, self.specific_variance
            if x.ndim != 2 or x.shape[1] != len(matrix) or d.shape != (len(x),):
                raise ValueError("factor risk dimensions differ")
            if (d < 0).any():
                raise ValueError("specific variance must be non-negative")

    def variance(self, weights):
        w = np.asarray(weights)
        if self.covariance is not None:
            return float(w @ self.covariance @ w)
        exposure = self.exposures.T @ w
        return float(exposure @ self.factor_covariance @ exposure + self.specific_variance @ w**2)

    def diagonal(self):
        if self.covariance is not None:
            return np.diag(self.covariance).copy()
        return (
            np.einsum("ij,ij->i", self.exposures @ self.factor_covariance, self.exposures)
            + self.specific_variance
        )

    def root_expression(self, cp, weights):
        matrix = self.covariance if self.covariance is not None else self.factor_covariance
        eigenvalues, vectors = np.linalg.eigh(matrix)
        root = np.sqrt(np.maximum(eigenvalues, 0))[:, None] * vectors.T
        if self.covariance is not None:
            return root @ weights
        return cp.hstack(
            (
                root @ (self.exposures.T @ weights),
                cp.multiply(np.sqrt(self.specific_variance), weights),
            )
        )


@dataclass
class SolveResult:
    weights: np.ndarray
    objective: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def constraint_residuals(
    weights,
    current,
    settings,
    *,
    assets,
    budget=1.0,
    lower=None,
    upper=None,
    factor_exposures=None,
    factor_bounds=None,
    risk=None,
    benchmark=None,
    turnover_offset=0.0,
    budget_equality=True,
):
    """Independent NumPy acceptance check; positive values denote violations."""
    w = np.asarray(weights, float)
    if not np.isfinite(w).all():
        return {"finite": float("inf")}
    lo = settings.min_weight if lower is None else np.asarray(lower)
    hi = settings.max_weight if upper is None else np.asarray(upper)
    result = {
        "budget": abs(float(w.sum()) - budget) if budget_equality else float(w.sum()) - budget,
        "asset_lower": float(np.max(lo - w)),
        "asset_upper": float(np.max(w - hi)),
        "turnover": float(np.abs(w - current).sum()) + turnover_offset - settings.max_turnover,
    }
    for group, cap in settings.group_caps.items():
        result["group:" + group] = (
            sum(w[i] for i, a in enumerate(assets) if settings.group_by_asset.get(a) == group) - cap
        )
    if factor_bounds:
        exposure = np.asarray(factor_exposures).T @ w
        for i, (name, (left, right)) in enumerate(factor_bounds.items()):
            result["factor_lower:" + name] = float(left - exposure[i])
            result["factor_upper:" + name] = float(exposure[i] - right)
    if settings.max_tracking_error is not None:
        result["tracking_error"] = (
            np.sqrt(max(risk.variance(w - benchmark), 0)) - settings.max_tracking_error
        )
    return result


def _run(cp, problem, *, conic, deadline, diagnostics, max_iterations):
    attempts = diagnostics.setdefault("attempts", [])
    solvers = ("CLARABEL", "CLARABEL") if conic else ("OSQP", "CLARABEL")
    for attempt, solver in enumerate(solvers):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise OptimizationError("timeout", "rebalance solve deadline exceeded", diagnostics)
        started = perf_counter()
        try:
            if solver == "OSQP":
                options = {
                    "eps_abs": 1e-10,
                    "eps_rel": 1e-10,
                    "max_iter": max_iterations,
                    "time_limit": remaining,
                    "polishing": True,
                }
            else:
                options = {
                    "tol_gap_abs": 1e-9 if attempt == 0 else 1e-8,
                    "tol_gap_rel": 1e-9 if attempt == 0 else 1e-8,
                    "tol_feas": 1e-9,
                    "max_iter": max_iterations,
                    "time_limit": remaining,
                }
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
                problem.solve(solver=solver, warm_start=attempt == 0, **options)
        except cp.error.SolverError as exc:
            attempts.append({"solver": solver, "status": "numerical_failure", "error": str(exc)})
            continue
        elapsed = perf_counter() - started
        status = problem.status
        residual = (
            max((float(np.max(np.abs(c.violation()))) for c in problem.constraints), default=0.0)
            if status == cp.OPTIMAL
            else None
        )
        stats = problem.solver_stats
        attempts.append(
            {
                "solver": solver,
                "status": status,
                "seconds": elapsed,
                "compilation_seconds": problem.compilation_time,
                "solver_seconds": stats.solve_time,
                "iterations": stats.num_iters,
                "max_constraint_residual": residual,
                "options": options,
            }
        )
        if perf_counter() > deadline:
            raise OptimizationError("timeout", "rebalance solve deadline exceeded", diagnostics)
        if status == cp.OPTIMAL and residual is not None and residual <= 1e-7:
            diagnostics.update(
                status="optimal", solver=solver, residual=residual, iterations=stats.num_iters or 0
            )
            return True
        if status == cp.INFEASIBLE:
            diagnostics.update(status="infeasible", solver=solver)
            return False
        if status == cp.UNBOUNDED:
            raise OptimizationError("unbounded", "objective is unbounded", diagnostics)
    status = (
        "timeout"
        if perf_counter() >= deadline
        else (
            "not_converged"
            if any(a["status"] == cp.USER_LIMIT for a in attempts)
            else "numerical_failure"
        )
    )
    raise OptimizationError(
        status, "no accurate solver result; constraints were not relaxed", diagnostics
    )


def solve_portfolio(
    expected,
    risk,
    current,
    settings,
    *,
    assets=None,
    budget=1.0,
    lower=None,
    upper=None,
    factor_exposures=None,
    factor_bounds=None,
    benchmark=None,
    active_risk=False,
    linear_costs=None,
    buy_costs=None,
    sell_costs=None,
    turnover_penalty=0.0,
    turnover_offset=0.0,
    risk_aversion=5.0,
    risk_scale=1.0,
    impact=None,
    kind="mean_variance",
    target=None,
    scenarios=None,
    beta=0.95,
    max_drawdown=None,
    risk_free_rate=0.0,
    timeout_seconds=30.0,
    max_iterations=2000,
):
    """Solve with all constraints present, including real pre-trade holdings."""
    import cvxpy as cp

    started = perf_counter()
    if (
        isinstance(timeout_seconds, bool)
        or not np.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be positive and finite")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    deadline = started + timeout_seconds
    mu, current = np.asarray(expected, float), np.asarray(current, float)
    n = len(mu)
    if n == 0 or mu.shape != (n,) or current.shape != (n,):
        raise ValueError("expected returns and current weights must be aligned vectors")
    if not np.isfinite(mu).all() or not np.isfinite(current).all() or (current < 0).any():
        raise ValueError(
            "expected returns and current weights must be finite; holdings non-negative"
        )
    if not np.isfinite(
        [budget, risk_scale, risk_aversion, turnover_offset, turnover_penalty]
    ).all():
        raise ValueError("problem parameters must be finite")
    if (
        not 0 <= budget <= 1
        or risk_scale <= 0
        or risk_aversion <= 0
        or turnover_offset < 0
        or turnover_penalty < 0
    ):
        raise ValueError("invalid budget, risk scale, aversion or turnover offset")
    assets = list(range(n)) if assets is None else list(assets)
    if len(assets) != n or len(set(assets)) != n:
        raise ValueError("assets must be unique and aligned")
    if not np.isfinite(settings.max_turnover) or settings.max_turnover < 0:
        raise ValueError("max_turnover must be finite and non-negative")
    if settings.max_tracking_error is not None and (
        not np.isfinite(settings.max_tracking_error) or settings.max_tracking_error < 0
    ):
        raise ValueError("max_tracking_error must be finite and non-negative")
    if factor_bounds and factor_exposures is None:
        raise ValueError("factor bounds require factor exposures")
    lo = np.broadcast_to(settings.min_weight if lower is None else lower, (n,)).copy()
    hi = np.broadcast_to(settings.max_weight if upper is None else upper, (n,)).copy()
    if (
        not np.isfinite(lo).all()
        or not np.isfinite(hi).all()
        or (lo < 0).any()
        or (hi > 1).any()
        or (lo > hi).any()
    ):
        raise ValueError("invalid per-asset bounds")
    if lo.sum() > budget + 1e-10 or hi.sum() < budget - 1e-10:
        raise OptimizationError("infeasible", "weight bounds cannot satisfy the budget constraint")
    wb = np.zeros(n) if benchmark is None else np.asarray(benchmark, float)
    if wb.shape != (n,) or not np.isfinite(wb).all() or (wb < 0).any() or wb.sum() > 1 + 1e-8:
        raise ValueError("benchmark must be a complete non-negative weight vector")
    if (active_risk or settings.max_tracking_error is not None) and benchmark is None:
        raise ValueError("active risk and max_tracking_error require benchmark_weights")
    w = cp.Variable(n, name="weights")
    trade = w - current
    constraints = [
        cp.sum(w) == budget,
        w >= lo,
        w <= hi,
        cp.norm1(trade) + turnover_offset <= settings.max_turnover,
    ]
    labels = ["budget", "asset_lower", "asset_upper", "turnover"]
    for group, cap in settings.group_caps.items():
        if not np.isfinite(cap) or not 0 <= cap <= 1:
            raise ValueError("group caps must be in [0, 1]")
        mask = np.array([settings.group_by_asset.get(a, "__ungrouped__") == group for a in assets])
        constraints.append(cp.sum(w[mask]) <= cap)
        labels.append("group:" + group)
    x = None if factor_exposures is None else np.asarray(factor_exposures, float)
    if x is not None:
        bounds = list((factor_bounds or {}).values())
        if x.shape != (n, len(bounds)) or not np.isfinite(x).all():
            raise ValueError("factor exposures and bounds must align")
        for index, (name, (left, right)) in enumerate(factor_bounds.items()):
            if left > right or not np.isfinite([left, right]).all():
                raise ValueError("factor bounds must be finite ordered pairs")
            constraints.extend([x[:, index] @ w >= left, x[:, index] @ w <= right])
            labels.extend(["factor_lower:" + name, "factor_upper:" + name])
    if risk is None:
        if settings.max_tracking_error is not None or kind in {
            "mean_variance",
            "max_sharpe",
            "max_diversification",
        }:
            raise ValueError("this objective requires risk inputs")
        risk_root = None
    else:
        risk_root = risk.root_expression(cp, w)
    conic = settings.max_tracking_error is not None or impact is not None
    if settings.max_tracking_error is not None:
        constraints.append(
            cp.norm(risk.root_expression(cp, w - wb), 2) <= settings.max_tracking_error
        )
        labels.append("tracking_error")
    cost = np.zeros(n) if linear_costs is None else np.asarray(linear_costs, float)
    buy = cost if buy_costs is None else np.asarray(buy_costs, float)
    sell = cost if sell_costs is None else np.asarray(sell_costs, float)
    if any(c.shape != (n,) or not np.isfinite(c).all() or (c < 0).any() for c in (buy, sell)):
        raise ValueError("cost rates must be complete, finite and non-negative")
    penalty = (
        buy @ cp.pos(trade)
        + sell @ cp.pos(-trade)
        + turnover_penalty * (cp.norm1(trade) + turnover_offset)
    )
    if impact is not None:
        adv, vol, coefficient, nav = impact
        adv, vol = np.asarray(adv, float), np.asarray(vol, float)
        if (
            adv.shape != (n,)
            or vol.shape != (n,)
            or not np.isfinite(adv).all()
            or not np.isfinite(vol).all()
            or (adv <= 0).any()
            or (vol < 0).any()
            or not np.isfinite([coefficient, nav]).all()
            or coefficient < 0
            or nav <= 0
        ):
            raise ValueError(
                "impact requires positive ADV/NAV and non-negative volatility/coefficient"
            )
        coefficients = coefficient * np.asarray(vol) * np.sqrt(nav / np.asarray(adv))
        penalty += coefficients @ cp.power(cp.abs(trade), 1.5)
    diagnostics = {
        "backend": "cvxpy",
        "objective_kind": kind,
        "constraints": labels,
        "risk_representation": "none"
        if risk is None
        else ("dense" if risk.covariance is not None else "factor"),
        "attempts": [],
    }

    if kind in {"max_sharpe", "max_diversification"}:
        numerator = (
            mu @ w - risk_free_rate
            if kind == "max_sharpe"
            else np.sqrt(risk.diagonal() * risk_scale) @ w
        )
        scale = (
            max(float(np.max(np.abs(mu))), abs(risk_free_rate), 1e-4)
            if kind == "max_sharpe"
            else max(float(np.max(np.sqrt(risk.diagonal() * risk_scale))), 1e-4)
        )
        positive = cp.Problem(cp.Maximize(numerator / scale), constraints)
        if not _run(
            cp,
            positive,
            conic=True,
            deadline=deadline,
            diagnostics=diagnostics,
            max_iterations=max_iterations,
        ):
            raise OptimizationError("infeasible", "joint constraints are infeasible", diagnostics)
        if float(numerator.value) <= 1e-12:
            raise OptimizationError(
                "no_positive_excess_return", "ratio has no positive numerator", diagnostics
            )
        if risk.variance(np.asarray(w.value)) <= 1e-18:
            raise OptimizationError(
                "undefined_ratio", "positive numerator with zero risk", diagnostics
            )
        ratio = cp.Parameter(nonneg=True)
        # Maximize the feasibility margin. This avoids an ill-conditioned
        # zero-objective feasibility problem exactly at the ratio boundary.
        problem = cp.Problem(
            cp.Maximize((numerator - ratio * np.sqrt(risk_scale) * cp.norm(risk_root, 2)) / scale),
            constraints,
        )

        def feasible_ratio(value):
            ratio.value = value
            if not _run(
                cp,
                problem,
                conic=True,
                deadline=deadline,
                diagnostics=diagnostics,
                max_iterations=max_iterations,
            ):
                raise OptimizationError("numerical_failure", "ratio domain changed", diagnostics)
            return float(problem.value) >= 0.0

        low, high = 0.0, 1.0
        best = np.asarray(w.value).copy()
        for _ in range(40):
            feasible = feasible_ratio(high)
            if not feasible:
                break
            best = np.asarray(w.value).copy()
            low, high = high, 2 * high
        else:
            raise OptimizationError(
                "undefined_ratio", "zero risk or unbounded positive ratio", diagnostics
            )
        for _ in range(60):
            if high - low <= 1e-9 * max(1.0, low):
                break
            if feasible_ratio((low + high) / 2):
                low, best = ratio.value, np.asarray(w.value).copy()
            else:
                high = ratio.value
        w.value = best
        variance = risk.variance(best)
        if variance <= 1e-18:
            raise OptimizationError("undefined_ratio", "zero-volatility portfolio", diagnostics)
        objective_value = float(numerator.value) / np.sqrt(variance * risk_scale)
        diagnostics["ratio_bracket"] = [low, high]
    else:
        if kind == "mean_variance":
            centered = w - wb if active_risk else w
            objective = (
                mu @ w
                - risk_aversion
                * 0.5
                * risk_scale
                * cp.sum_squares(risk.root_expression(cp, centered))
                - penalty
            )
        elif kind == "target":
            target = np.asarray(target, float)
            if target.shape != (n,) or not np.isfinite(target).all():
                raise ValueError("target must be finite and aligned")
            objective = -cp.sum_squares(w - target) - penalty
        elif kind in {"cvar", "evar", "cdar"}:
            rows = np.asarray(scenarios, float)
            if rows.ndim != 2 or rows.shape[1] != n or len(rows) < 2 or not np.isfinite(rows).all():
                raise ValueError("scenarios must be finite and complete with at least two rows")
            if not 0 < beta < 1:
                raise ValueError("beta must be in (0, 1)")
            losses = -(rows @ w)
            if kind == "cdar" or max_drawdown is not None:
                cumulative = np.cumsum(rows, axis=0) @ w
                peaks = cp.Variable(len(rows))
                constraints.extend([peaks >= 0, peaks >= cumulative, peaks[1:] >= peaks[:-1]])
                drawdowns = peaks - cumulative
                if max_drawdown is not None:
                    if not np.isfinite(max_drawdown) or max_drawdown < 0:
                        raise ValueError("max_drawdown must be finite and non-negative")
                    constraints.append(drawdowns <= max_drawdown)
                if kind == "cdar":
                    losses = drawdowns
            threshold = cp.Variable()
            if kind == "evar":
                temperature = cp.Variable(nonneg=True)
                aux = cp.Variable(len(rows))
                constraints.extend(
                    [
                        cp.ExpCone(
                            losses - threshold, cp.multiply(np.ones(len(rows)), temperature), aux
                        ),
                        cp.sum(aux) <= temperature * len(rows) * (1 - beta),
                    ]
                )
                tail_risk = threshold
            else:
                excess = cp.Variable(len(rows), nonneg=True)
                constraints.append(excess >= losses - threshold)
                tail_risk = threshold + cp.sum(excess) / (len(rows) * (1 - beta))
            objective = mu @ w - risk_aversion * tail_risk - penalty
            conic = True
        else:
            raise ValueError(f"unsupported objective {kind}")
        problem = cp.Problem(cp.Maximize(objective), constraints)
        if not _run(
            cp,
            problem,
            conic=conic,
            deadline=deadline,
            diagnostics=diagnostics,
            max_iterations=max_iterations,
        ):
            raise OptimizationError("infeasible", "joint constraints are infeasible", diagnostics)
        objective_value = float(objective.value)
    weights = np.asarray(w.value, float)
    checks = constraint_residuals(
        weights,
        current,
        settings,
        assets=assets,
        budget=budget,
        lower=lo,
        upper=hi,
        factor_exposures=x,
        factor_bounds=factor_bounds,
        risk=risk,
        benchmark=wb,
        turnover_offset=turnover_offset,
    )
    residual = max(0.0, *checks.values())
    if kind in {"cvar", "evar", "cdar"} and max_drawdown is not None:
        path = np.r_[0.0, np.cumsum(np.asarray(scenarios) @ weights)]
        drawdown = float(np.max(np.maximum.accumulate(path) - path))
        residual = max(residual, drawdown - max_drawdown)
    if not np.isfinite(weights).all() or residual > 1e-7:
        raise OptimizationError(
            "numerical_failure", "independent constraint recheck failed", diagnostics
        )
    diagnostics.update(
        status="optimal",
        max_constraint_residual=residual,
        constraint_residuals=checks,
        total_seconds=perf_counter() - started,
        iterations=sum(a.get("iterations") or 0 for a in diagnostics["attempts"]),
    )
    delta = weights - current
    estimated_buy = float(buy @ np.maximum(delta, 0))
    estimated_sell = float(sell @ np.maximum(-delta, 0))
    estimated_impact = 0.0 if impact is None else float(coefficients @ np.abs(delta) ** 1.5)
    diagnostics["estimated_costs"] = {
        "buy": estimated_buy,
        "sell": estimated_sell,
        "impact": estimated_impact,
        "total": estimated_buy + estimated_sell + estimated_impact,
        "units": "fraction_of_pre_trade_nav",
    }
    diagnostics["costs_in_objective"] = kind not in {"max_sharpe", "max_diversification"}
    diagnostics["constraint_values"] = {
        "budget": {"value": float(weights.sum()), "target": budget},
        "turnover": {
            "value": float(np.abs(delta).sum()) + turnover_offset,
            "upper": settings.max_turnover,
        },
        "asset_weights": dict(zip(assets, weights.tolist())),
        "asset_lower": dict(zip(assets, lo.tolist())),
        "asset_upper": dict(zip(assets, hi.tolist())),
        "factor_exposures": {} if x is None else dict(zip(factor_bounds, (x.T @ weights).tolist())),
        "factor_bounds": factor_bounds or {},
    }
    return SolveResult(weights.copy(), objective_value, diagnostics)
