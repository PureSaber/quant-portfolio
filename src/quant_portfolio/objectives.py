"""Independent portfolio objectives that share the long-only feasible set.

Mean-variance stays in ``optimization.py``. The solvers here are projected
gradient methods on that same set: maximum Sharpe, maximum diversification,
entropic value at risk, and conditional drawdown at risk. Integer share lots
are a separate exact local search, not a continuous relaxation that is rounded
only when an order is created.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pandas as pd


def maximum_drawdown(portfolio_returns: np.ndarray) -> float:
    """Peak-to-trough drop of cumulative simple returns, measured from zero."""

    return float(np.max(_drawdown_series(portfolio_returns)))


def conditional_drawdown_at_risk(portfolio_returns: np.ndarray, beta: float) -> float:
    """Mean drawdown at and above the empirical ``beta`` quantile."""

    _validate_beta(beta)
    drawdowns = _drawdown_series(portfolio_returns)
    level = float(np.quantile(drawdowns, float(beta)))
    tail = drawdowns[drawdowns >= level - 1e-15]
    return float(tail.mean()) if len(tail) else float(drawdowns.max())


def entropic_value_at_risk(losses: np.ndarray, beta: float) -> float:
    """Entropic value at risk of a loss sample at level ``beta``."""

    value, _temperature = _evar_temperature(np.asarray(losses, dtype=float), float(beta))
    return value


def enforce_drawdown_limit(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    limit: float,
    anchor: np.ndarray,
) -> np.ndarray:
    """Pull ``weights`` toward a feasible anchor until drawdown is inside ``limit``.

    Uncompounded drawdown is convex in weights, so the segment from an
    infeasible point to a feasible anchor enters the set.
    """

    if maximum_drawdown(asset_returns @ weights) <= limit + 1e-10:
        return weights
    if maximum_drawdown(asset_returns @ anchor) > limit + 1e-8:
        raise ValueError("max_drawdown is infeasible for these scenarios and constraints")
    low = 0.0
    high = 1.0
    best = anchor.copy()
    for _ in range(50):
        mix = 0.5 * (low + high)
        candidate = (1.0 - mix) * weights + mix * anchor
        if maximum_drawdown(asset_returns @ candidate) <= limit + 1e-10:
            best = candidate
            high = mix
        else:
            low = mix
    return best


def minimum_drawdown_weights(
    asset_returns: np.ndarray,
    project: Callable[[np.ndarray], np.ndarray],
    *,
    max_iterations: int = 400,
) -> np.ndarray:
    """Long-only weights that reduce the worst cumulative-return drawdown."""

    weights = project(np.full(asset_returns.shape[1], 1.0 / asset_returns.shape[1]))
    for _ in range(max_iterations):
        gradient = _worst_drawdown_gradient(asset_returns @ weights, asset_returns)
        scale = max(float(np.max(np.abs(gradient))), 1e-8)
        updated = project(weights - (0.05 / scale) * gradient)
        if float(np.max(np.abs(updated - weights))) <= 1e-12:
            return updated
        weights = updated
    return weights


def allocate_integer_lots(
    target: np.ndarray,
    steps: np.ndarray,
    objective: Callable[[np.ndarray], float],
    is_feasible: Callable[[np.ndarray], bool],
) -> np.ndarray:
    """Locally optimal non-negative integer lots near a continuous target.

    Cash may remain when a lot does not divide the budget. The search starts
    at the floored target, sheds lots until the book is feasible, then accepts
    one-lot buys and transfers that raise ``objective``.
    """

    grid = np.asarray(steps, dtype=float)
    point = np.asarray(target, dtype=float)
    if grid.shape != point.shape or grid.ndim != 1 or point.ndim != 1:
        raise ValueError("lot steps must match the weight vector")
    if not np.isfinite(grid).all() or not np.isfinite(point).all() or np.any(grid <= 0):
        raise ValueError("lot steps must be positive and finite")
    counts = np.maximum(np.floor(point / grid + 1e-10).astype(int), 0)
    counts = _shed_infeasible_lots(counts, grid, objective, is_feasible)
    if not is_feasible(counts * grid):
        raise ValueError("no feasible integer-lot portfolio exists under the current constraints")
    return _improve_lots(counts, grid, objective, is_feasible)


def optimize_max_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    risk_free_rate: float = 0.0,
    current_weights: pd.Series | None = None,
    constraints: object | None = None,
    benchmark_weights: pd.Series | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
):
    """Maximize excess return over volatility on the long-only feasible set."""

    problem = _LongOnlyInputs.prepare(
        expected_returns,
        covariance,
        current_weights=current_weights,
        constraints=constraints,
        benchmark_weights=benchmark_weights,
    )
    risk_free = _finite("risk_free_rate", risk_free_rate, positive=False)

    def ratio(weights: np.ndarray) -> float:
        volatility = math.sqrt(max(float(weights @ problem.covariance @ weights), 0.0))
        if volatility <= 1e-18:
            raise ValueError("Sharpe ratio is undefined for a zero-volatility portfolio")
        return (float(problem.expected @ weights) - risk_free) / volatility

    def gradient(weights: np.ndarray) -> np.ndarray:
        excess = float(problem.expected @ weights) - risk_free
        variance = max(float(weights @ problem.covariance @ weights), 1e-18)
        volatility = math.sqrt(variance)
        risk_gradient = problem.covariance @ weights
        return problem.expected / volatility - excess * risk_gradient / variance**1.5

    start = _positive_projection(problem.covariance, problem.expected - risk_free)
    return _maximize_ratio(
        problem,
        ratio,
        gradient,
        start,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def optimize_max_diversification(
    covariance: pd.DataFrame,
    *,
    current_weights: pd.Series | None = None,
    constraints: object | None = None,
    benchmark_weights: pd.Series | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-10,
):
    """Maximize Choueifaty's diversification ratio ``(w'σ) / sqrt(w'Σw)``."""

    assets = list(covariance.index)
    flat = pd.Series(0.0, index=assets)
    problem = _LongOnlyInputs.prepare(
        flat,
        covariance,
        current_weights=current_weights,
        constraints=constraints,
        benchmark_weights=benchmark_weights,
    )
    volatility = np.sqrt(np.diag(problem.covariance))
    if np.any(volatility <= 0):
        raise ValueError("maximum diversification requires positive asset variance")

    def ratio(weights: np.ndarray) -> float:
        denominator = math.sqrt(max(float(weights @ problem.covariance @ weights), 0.0))
        if denominator <= 1e-18:
            raise ValueError("diversification ratio is undefined for a zero-volatility portfolio")
        return float(volatility @ weights) / denominator

    def gradient(weights: np.ndarray) -> np.ndarray:
        numerator = float(volatility @ weights)
        variance = max(float(weights @ problem.covariance @ weights), 1e-18)
        denominator = math.sqrt(variance)
        return volatility / denominator - numerator * (problem.covariance @ weights) / variance**1.5

    start = _positive_projection(problem.covariance, volatility)
    return _maximize_ratio(
        problem,
        ratio,
        gradient,
        start,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def optimize_evar(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series,
    *,
    beta: float = 0.95,
    risk_aversion: float = 1.0,
    current_weights: pd.Series | None = None,
    constraints: object | None = None,
    max_drawdown: float | None = None,
    max_iterations: int = 800,
    tolerance: float = 1e-8,
):
    """Maximize expected return minus entropic value at risk."""

    scenarios, expected, problem = _path_problem(
        scenario_returns, expected_returns, current_weights, constraints
    )
    aversion = _finite("risk_aversion", risk_aversion, positive=True)
    _validate_beta(beta)
    anchor = _drawdown_anchor(scenarios, problem.project, max_drawdown)

    def objective_at(weights: np.ndarray) -> float:
        losses = -(scenarios @ weights)
        return float(expected @ weights) - aversion * entropic_value_at_risk(losses, beta)

    def gradient(weights: np.ndarray) -> np.ndarray:
        losses = -(scenarios @ weights)
        _value, temperature = _evar_temperature(losses, beta)
        shifted = losses / temperature
        shifted -= float(np.max(shifted))
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        return expected + aversion * (scenarios.T @ probabilities)

    return _maximize_path(
        problem,
        scenarios,
        objective_at,
        gradient,
        anchor,
        max_drawdown,
        risk_name="evar",
        risk_value=lambda weights: entropic_value_at_risk(-(scenarios @ weights), beta),
        aversion=aversion,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


def optimize_cdar(
    scenario_returns: pd.DataFrame,
    expected_returns: pd.Series,
    *,
    beta: float = 0.95,
    risk_aversion: float = 1.0,
    current_weights: pd.Series | None = None,
    constraints: object | None = None,
    max_drawdown: float | None = None,
    max_iterations: int = 800,
    tolerance: float = 1e-8,
):
    """Maximize expected return minus conditional drawdown at risk.

    Drawdown is the drop in cumulative simple return from its running peak.
    Scenario order is the path order.
    """

    scenarios, expected, problem = _path_problem(
        scenario_returns, expected_returns, current_weights, constraints
    )
    aversion = _finite("risk_aversion", risk_aversion, positive=True)
    _validate_beta(beta)
    anchor = _drawdown_anchor(scenarios, problem.project, max_drawdown)

    def objective_at(weights: np.ndarray) -> float:
        return float(expected @ weights) - aversion * conditional_drawdown_at_risk(
            scenarios @ weights, beta
        )

    def gradient(weights: np.ndarray) -> np.ndarray:
        path = scenarios @ weights
        drawdowns = _drawdown_series(path)
        level = float(np.quantile(drawdowns, float(beta)))
        tail = drawdowns >= level - 1e-15
        if not tail.any():
            tail = np.zeros(len(drawdowns), dtype=bool)
            tail[int(np.argmax(drawdowns))] = True
        risk_gradient = _tail_drawdown_gradient(path, scenarios, tail)
        return expected - aversion * risk_gradient

    return _maximize_path(
        problem,
        scenarios,
        objective_at,
        gradient,
        anchor,
        max_drawdown,
        risk_name="cdar",
        risk_value=lambda weights: conditional_drawdown_at_risk(scenarios @ weights, beta),
        aversion=aversion,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )


class _LongOnlyInputs:
    def __init__(
        self,
        assets: list[str],
        expected: np.ndarray,
        covariance: np.ndarray,
        current: np.ndarray,
        benchmark: np.ndarray | None,
        settings: object,
        project: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.assets = assets
        self.expected = expected
        self.covariance = covariance
        self.current = current
        self.benchmark = benchmark
        self.settings = settings
        self.project = project

    @classmethod
    def prepare(
        cls,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        *,
        current_weights: pd.Series | None,
        constraints: object | None,
        benchmark_weights: pd.Series | None,
    ) -> _LongOnlyInputs:
        (
            constraints_type,
            result_type,
            project_joint,
            validate_constraints,
        ) = _solver_api()
        del result_type
        if not isinstance(expected_returns, pd.Series) or not expected_returns.index.is_unique:
            raise ValueError("expected_returns must be a Series with unique assets")
        if expected_returns.empty:
            raise ValueError("expected_returns is empty")
        assets = [str(asset) for asset in expected_returns.index]
        expected = pd.to_numeric(expected_returns, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(expected).all():
            raise ValueError("expected_returns must be finite")
        if not isinstance(covariance, pd.DataFrame):
            raise TypeError("covariance must be a pandas DataFrame")
        aligned = covariance.reindex(index=assets, columns=assets)
        values = aligned.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.allclose(
            values, values.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("covariance must be finite and symmetric for every asset")
        if float(np.linalg.eigvalsh(values).min()) < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        if constraints is not None and not isinstance(constraints, constraints_type):
            raise TypeError("constraints must be OptimizationConstraints")
        settings = validate_constraints(constraints, assets)
        current = _aligned_weights(current_weights, assets, "current_weights", total=None)
        if current is None:
            current = np.repeat(1.0 / len(assets), len(assets))
        benchmark = _aligned_weights(benchmark_weights, assets, "benchmark_weights", total=1.0)
        if settings.max_tracking_error is not None and benchmark is None:
            raise ValueError("max_tracking_error requires benchmark_weights")
        active_benchmark = np.zeros(len(assets)) if benchmark is None else benchmark

        def project(point: np.ndarray) -> np.ndarray:
            return project_joint(
                point,
                assets,
                current,
                0.0,
                settings,
                None,
                {},
                tolerance=1e-10,
                covariance=None if settings.max_tracking_error is None else values,
                benchmark=None if settings.max_tracking_error is None else active_benchmark,
                max_tracking_error=settings.max_tracking_error,
            )

        return cls(assets, expected, values, current, benchmark, settings, project)


def _solver_api():
    from quant_portfolio.optimization import (
        OptimizationConstraints,
        OptimizationResult,
        _project_joint_feasible_set,
        _validated_constraints,
    )

    return (
        OptimizationConstraints,
        OptimizationResult,
        _project_joint_feasible_set,
        _validated_constraints,
    )


def _aligned_weights(
    weights: pd.Series | None,
    assets: list[str],
    name: str,
    *,
    total: float | None,
) -> np.ndarray | None:
    if weights is None:
        return None
    if not isinstance(weights, pd.Series) or not weights.index.is_unique:
        raise ValueError(f"{name} must be a Series with unique assets")
    numeric = pd.to_numeric(weights.reindex(assets), errors="coerce").to_numpy(dtype=float)
    if set(weights.index) != set(assets) or not np.isfinite(numeric).all() or np.any(numeric < 0):
        raise ValueError(f"{name} must be finite, non-negative and cover every asset")
    if total is not None and abs(float(numeric.sum()) - total) > 1e-8:
        raise ValueError(f"{name} must sum to {total:g}")
    return numeric


def _positive_projection(covariance: np.ndarray, direction: np.ndarray) -> np.ndarray:
    stabilized = covariance + np.eye(covariance.shape[0]) * 1e-10
    raw = np.linalg.solve(stabilized, np.maximum(direction, 0.0))
    if not np.isfinite(raw).all() or float(raw.sum()) <= 0:
        raw = np.maximum(direction, 0.0)
    if float(raw.sum()) <= 0:
        raw = np.ones(covariance.shape[0])
    return raw / raw.sum()


def _maximize_ratio(problem, ratio, gradient, start, *, max_iterations: int, tolerance: float):
    _result_type = _solver_api()[1]
    _iteration_limit(max_iterations)
    tolerance = _finite("tolerance", tolerance, positive=True)
    candidates = (
        problem.project(np.full(len(problem.assets), 1.0 / len(problem.assets))),
        problem.project(start),
    )
    best = candidates[0]
    best_value = ratio(best)
    for candidate in candidates[1:]:
        value = ratio(candidate)
        if value > best_value:
            best = candidate
            best_value = value
    weights = best.copy()
    converged = False
    iteration = max_iterations
    for iteration in range(1, max_iterations + 1):
        step = 0.25
        current_value = ratio(weights)
        slope = gradient(weights)
        candidate = weights
        for _ in range(20):
            proposal = problem.project(weights + step * slope)
            proposal_value = ratio(proposal)
            if proposal_value >= current_value - 1e-15:
                candidate = proposal
                break
            step *= 0.5
        else:
            converged = True
            break
        if proposal_value > best_value + tolerance:
            best = candidate.copy()
            best_value = proposal_value
        if float(np.max(np.abs(candidate - weights))) <= tolerance:
            weights = candidate
            converged = True
            break
        weights = candidate
    weights = best
    if not converged and iteration >= max_iterations:
        raise RuntimeError(f"ratio optimization did not converge after {max_iterations} iterations")
    variance = max(float(weights @ problem.covariance @ weights), 0.0)
    return _result_type(
        weights=pd.Series(weights, index=problem.assets, name="weight"),
        expected_return=float(problem.expected @ weights),
        volatility=float(math.sqrt(variance)),
        turnover=float(np.abs(weights - problem.current).sum()),
        objective=float(ratio(weights)),
        group_weights=_group_weights(problem.assets, weights, problem.settings),
        converged=True,
        iterations=iteration,
    )


def _path_problem(scenario_returns, expected_returns, current_weights, constraints):
    if getattr(constraints, "max_tracking_error", None) is not None:
        raise ValueError(
            "max_tracking_error is supported by mean-variance, maximum Sharpe "
            "and maximum diversification"
        )
    if not isinstance(scenario_returns, pd.DataFrame):
        raise TypeError("scenario_returns must be a pandas DataFrame")
    if not isinstance(expected_returns, pd.Series) or not expected_returns.index.is_unique:
        raise ValueError("expected_returns must be a Series with unique assets")
    assets = [str(asset) for asset in expected_returns.index]
    scenarios = scenario_returns.reindex(columns=assets).apply(pd.to_numeric, errors="coerce")
    values = scenarios.to_numpy(dtype=float)
    if scenarios.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("scenarios must be finite, complete and contain at least two rows")
    expected = pd.to_numeric(expected_returns, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(expected).all():
        raise ValueError("expected_returns must be finite")
    identity = pd.DataFrame(np.eye(len(assets)), index=assets, columns=assets)
    problem = _LongOnlyInputs.prepare(
        expected_returns,
        identity,
        current_weights=current_weights,
        constraints=constraints,
        benchmark_weights=None,
    )
    return values, expected, problem


def _drawdown_anchor(scenarios, project, limit: float | None):
    if limit is None:
        return None
    if (
        isinstance(limit, bool)
        or not isinstance(limit, (int, float))
        or not np.isfinite(limit)
        or limit < 0
    ):
        raise ValueError("max_drawdown must be finite and non-negative")
    anchor = minimum_drawdown_weights(scenarios, project)
    if maximum_drawdown(scenarios @ anchor) > float(limit) + 1e-8:
        raise ValueError("max_drawdown is infeasible for these scenarios and constraints")
    return anchor


def _maximize_path(
    problem,
    scenarios,
    objective_at,
    gradient,
    anchor,
    max_drawdown_limit,
    *,
    risk_name: str,
    risk_value,
    aversion: float,
    max_iterations: int,
    tolerance: float,
):
    del risk_name
    result_type = _solver_api()[1]
    _iteration_limit(max_iterations)
    tolerance = _finite("tolerance", tolerance, positive=True)
    weights = problem.project(problem.current)
    best = weights.copy()
    best_value = -np.inf
    converged = False
    iteration = max_iterations
    quiet = 0
    for iteration in range(1, max_iterations + 1):
        slope = gradient(weights)
        scale = max(float(np.max(np.abs(slope))), 1e-8)
        step = 0.25 / scale / (1.0 + 0.02 * iteration)
        candidate = problem.project(weights + step * slope)
        if anchor is not None:
            candidate = enforce_drawdown_limit(
                candidate, scenarios, float(max_drawdown_limit), anchor
            )
        value = objective_at(candidate)
        if value > best_value + tolerance:
            best = candidate.copy()
            best_value = value
            quiet = 0
        else:
            quiet += 1
        if float(np.max(np.abs(candidate - weights))) <= tolerance or quiet >= 40:
            weights = best
            converged = True
            break
        weights = candidate
    if not converged:
        raise RuntimeError(f"path optimization did not converge after {max_iterations} iterations")
    weights = best
    if (
        anchor is not None
        and maximum_drawdown(scenarios @ weights) > float(max_drawdown_limit) + 1e-8
    ):
        raise RuntimeError("optimizer produced a portfolio outside the drawdown limit")
    risk = float(risk_value(weights))
    expected = float(problem.expected @ weights)
    return result_type(
        weights=pd.Series(weights, index=problem.assets, name="weight"),
        expected_return=expected,
        volatility=float(np.std(scenarios @ weights, ddof=1)),
        turnover=float(np.abs(weights - problem.current).sum()),
        objective=expected - aversion * risk,
        group_weights=_group_weights(problem.assets, weights, problem.settings),
        converged=True,
        iterations=iteration,
    )


def _shed_infeasible_lots(counts, steps, objective, is_feasible):
    guard = int(counts.sum()) + 1
    while not is_feasible(counts * steps) and counts.sum() > 0 and guard > 0:
        guard -= 1
        eligible = np.flatnonzero(counts > 0)
        feasible_choice = None
        feasible_objective = -np.inf
        for index in eligible:
            trial = counts.copy()
            trial[index] -= 1
            if is_feasible(trial * steps):
                value = objective(trial * steps)
                if value > feasible_objective:
                    feasible_choice = index
                    feasible_objective = value
        if feasible_choice is None:
            counts[eligible[-1]] -= 1
        else:
            counts[feasible_choice] -= 1
    return counts


def _improve_lots(counts, steps, objective, is_feasible):
    best = counts.copy()
    best_value = objective(best * steps)
    for _ in range(10_000):
        improved = False
        for index in range(len(best)):
            grown = best.copy()
            grown[index] += 1
            if _accept_lot(grown, steps, objective, is_feasible, best_value):
                best = grown
                best_value = objective(best * steps)
                improved = True
                break
        if improved:
            continue
        for source in range(len(best)):
            if best[source] == 0:
                continue
            for destination in range(len(best)):
                if source == destination:
                    continue
                moved = best.copy()
                moved[source] -= 1
                moved[destination] += 1
                if _accept_lot(moved, steps, objective, is_feasible, best_value):
                    best = moved
                    best_value = objective(best * steps)
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best * steps


def _accept_lot(counts, steps, objective, is_feasible, incumbent: float) -> bool:
    weights = counts * steps
    return bool(is_feasible(weights) and objective(weights) > incumbent + 1e-12)


def _group_weights(assets: list[str], weights: np.ndarray, settings: object) -> dict[str, float]:
    grouped: dict[str, float] = {}
    mapping = getattr(settings, "group_by_asset", {})
    for asset, weight in zip(assets, weights, strict=True):
        group = mapping.get(asset, "__ungrouped__")
        grouped[group] = grouped.get(group, 0.0) + float(weight)
    return grouped


def _drawdown_series(portfolio_returns: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(np.asarray(portfolio_returns, dtype=float))
    wealth = np.concatenate([[0.0], cumulative])
    peaks = np.maximum.accumulate(wealth)
    return peaks[1:] - wealth[1:]


def _peak_indexes(portfolio_returns: np.ndarray) -> np.ndarray:
    wealth = np.concatenate([[0.0], np.cumsum(portfolio_returns)])
    peaks = np.empty(len(portfolio_returns), dtype=int)
    best = 0
    for time in range(1, len(wealth)):
        if wealth[time] >= wealth[best]:
            best = time
        peaks[time - 1] = best
    return peaks


def _exposure(asset_returns: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, asset_returns.shape[1])), np.cumsum(asset_returns, axis=0)])


def _worst_drawdown_gradient(
    portfolio_returns: np.ndarray, asset_returns: np.ndarray
) -> np.ndarray:
    drawdowns = _drawdown_series(portfolio_returns)
    worst = float(np.max(drawdowns))
    tail = drawdowns >= worst - 1e-15
    return _tail_drawdown_gradient(portfolio_returns, asset_returns, tail)


def _tail_drawdown_gradient(
    portfolio_returns: np.ndarray,
    asset_returns: np.ndarray,
    tail: np.ndarray,
) -> np.ndarray:
    peaks = _peak_indexes(portfolio_returns)
    exposure = _exposure(asset_returns)
    gradients = [exposure[int(peaks[time])] - exposure[time + 1] for time in np.flatnonzero(tail)]
    if not gradients:
        return np.zeros(asset_returns.shape[1])
    return np.mean(gradients, axis=0)


def _evar_temperature(losses: np.ndarray, beta: float) -> tuple[float, float]:
    _validate_beta(beta)
    if losses.ndim != 1 or losses.shape[0] < 2 or not np.isfinite(losses).all():
        raise ValueError("EVaR requires at least two finite loss scenarios")
    scale = max(float(np.std(losses)), float(np.max(np.abs(losses))), 1e-6)

    def objective(temperature: float) -> float:
        return temperature * (_log_mean_exp(losses / temperature) - math.log(1.0 - float(beta)))

    grid = np.geomspace(scale * 1e-4, scale * 1e3, 40)
    values = np.array([objective(float(point)) for point in grid])
    best = int(np.argmin(values))
    left = float(grid[max(best - 1, 0)])
    right = float(grid[min(best + 1, len(grid) - 1)])
    phi = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(40):
        if right - left <= 1e-8 * scale:
            break
        inner_left = right - phi * (right - left)
        inner_right = left + phi * (right - left)
        if objective(inner_left) < objective(inner_right):
            right = inner_right
        else:
            left = inner_left
    temperature = 0.5 * (left + right)
    return objective(temperature), temperature


def _log_mean_exp(values: np.ndarray) -> float:
    peak = float(np.max(values))
    return peak + math.log(float(np.mean(np.exp(values - peak))))


def _validate_beta(beta: float) -> None:
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not np.isfinite(beta):
        raise ValueError("beta must be in (0, 1)")
    if not 0 < float(beta) < 1:
        raise ValueError("beta must be in (0, 1)")


def _finite(name: str, value: float, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _iteration_limit(max_iterations: int) -> None:
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
