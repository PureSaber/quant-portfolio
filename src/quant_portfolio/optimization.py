"""Deterministic portfolio optimization with explicit constraints and costs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OptimizationConstraints:
    min_weight: float = 0.0
    max_weight: float = 0.2
    max_turnover: float = 1.0
    group_by_asset: dict[str, str] = field(default_factory=dict)
    group_caps: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    turnover: float
    objective: float
    group_weights: dict[str, float]
    converged: bool
    iterations: int


_RESEARCH_ALLOCATION_FIELDS = {
    "mode",
    "lookback",
    "min_observations",
    "covariance_shrinkage",
    "risk_aversion",
    "turnover_penalty",
    "max_turnover",
}


def validate_research_allocation(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize the closed research-allocation recipe fragment."""

    if not isinstance(value, Mapping):
        raise TypeError("allocation must be a mapping")
    unknown = set(value) - _RESEARCH_ALLOCATION_FIELDS
    if unknown:
        raise ValueError(f"allocation contains unknown fields: {sorted(unknown)}")
    mode = value.get("mode")
    if mode not in {"equal", "inverse_vol", "cost_aware"}:
        raise ValueError("allocation.mode must be equal, inverse_vol or cost_aware")

    def integer(name: str, default: int) -> int:
        result = value.get(name, default)
        if isinstance(result, bool) or not isinstance(result, int):
            raise TypeError(f"allocation.{name} must be an integer")
        return result

    def number(name: str, default: float) -> float:
        result = value.get(name, default)
        if isinstance(result, bool) or not isinstance(result, (int, float)):
            raise TypeError(f"allocation.{name} must be numeric")
        result = float(result)
        if not np.isfinite(result):
            raise ValueError(f"allocation.{name} must be finite")
        return result

    lookback = integer("lookback", 20)
    min_observations = integer("min_observations", 10)
    shrinkage = number("covariance_shrinkage", 0.2)
    risk_aversion = number("risk_aversion", 5.0)
    turnover_penalty = number("turnover_penalty", 0.0)
    max_turnover = number("max_turnover", 1.0)
    if lookback < 2:
        raise ValueError("allocation.lookback must be at least 2")
    if not 2 <= min_observations <= lookback:
        raise ValueError("allocation.min_observations must be between 2 and lookback")
    if not 0 <= shrinkage <= 1:
        raise ValueError("allocation.covariance_shrinkage must be in [0, 1]")
    if risk_aversion <= 0:
        raise ValueError("allocation.risk_aversion must be positive")
    if turnover_penalty < 0:
        raise ValueError("allocation.turnover_penalty must be non-negative")
    if not 0 <= max_turnover <= 2:
        raise ValueError("allocation.max_turnover must be in [0, 2]")
    return {
        "mode": mode,
        "lookback": lookback,
        "min_observations": min_observations,
        "covariance_shrinkage": shrinkage,
        "risk_aversion": risk_aversion,
        "turnover_penalty": turnover_penalty,
        "max_turnover": max_turnover,
    }


def research_allocation_weights(
    scores: pd.Series,
    trailing_returns: pd.DataFrame | None,
    current_weights: pd.Series | None,
    linear_costs: pd.Series | None,
    *,
    invested_limit: float,
    max_weight: float,
    config: Mapping[str, object],
    factor_exposures: pd.DataFrame | None = None,
    factor_bounds: Mapping[str, tuple[float, float]] | None = None,
    covariance_override: pd.DataFrame | None = None,
) -> pd.Series:
    """Build long-only research weights with one explicit, closed allocation policy.

    ``cost_aware`` delegates its optimization to :func:`optimize_mean_variance`.
    Every mode applies turnover to the real current portfolio, including the
    liquidation of current assets absent from ``scores``.
    """

    settings = validate_research_allocation(config)
    if not isinstance(scores, pd.Series) or scores.empty or not scores.index.is_unique:
        raise ValueError("scores must be a non-empty Series with unique assets")
    numeric_scores = pd.to_numeric(scores, errors="coerce").astype(float)
    if not np.isfinite(numeric_scores).all():
        raise ValueError("scores must be finite for every asset")
    if (
        isinstance(invested_limit, bool)
        or not isinstance(invested_limit, (int, float))
        or not np.isfinite(invested_limit)
        or not 0 < invested_limit <= 1
    ):
        raise ValueError("invested_limit must be in (0, 1]")
    if (
        isinstance(max_weight, bool)
        or not isinstance(max_weight, (int, float))
        or not np.isfinite(max_weight)
        or not 0 < max_weight <= 1
    ):
        raise ValueError("max_weight must be in (0, 1]")
    assets = list(numeric_scores.index)
    if len(assets) * float(max_weight) + 1e-12 < float(invested_limit):
        raise ValueError("asset count and max_weight cannot satisfy invested_limit")

    total = float(invested_limit)
    cap = float(max_weight)
    if current_weights is None:
        current_all = pd.Series(dtype=float)
    elif not isinstance(current_weights, pd.Series) or not current_weights.index.is_unique:
        raise ValueError("current_weights must be a Series with unique assets")
    else:
        current_all = pd.to_numeric(current_weights, errors="coerce").astype(float)
    if (
        not np.isfinite(current_all.to_numpy(dtype=float)).all()
        or (current_all < 0).any()
        or float(current_all.sum()) > 1.0 + 1e-10
    ):
        raise ValueError("current_weights must be finite, non-negative and sum to at most one")
    current = current_all.reindex(assets).fillna(0.0)
    outside_turnover = float(current_all.loc[~current_all.index.isin(assets)].sum())
    turnover_limit = float(settings["max_turnover"])
    mode = str(settings["mode"])
    if mode != "cost_aware" and any(
        value is not None for value in (factor_exposures, factor_bounds, covariance_override)
    ):
        raise ValueError(
            "factor constraints and covariance_override are only supported for cost_aware"
        )
    if mode == "equal":
        desired = np.repeat(total / len(assets), len(assets))
        weights = _constrain_turnover(
            desired,
            current.to_numpy(dtype=float),
            lower=0.0,
            upper=cap,
            total=total,
            max_turnover=turnover_limit,
            turnover_offset=outside_turnover,
        )
        return pd.Series(weights, index=assets, name="weight")

    window: pd.DataFrame | None = None
    if mode == "inverse_vol" or covariance_override is None:
        if not isinstance(trailing_returns, pd.DataFrame):
            raise TypeError(f"{mode} allocation requires trailing_returns")
        window = (
            trailing_returns.reindex(columns=assets)
            .tail(int(settings["lookback"]))
            .apply(pd.to_numeric, errors="coerce")
        )
        counts = window.notna().sum()
        missing = sorted(counts[counts < int(settings["min_observations"])].index.astype(str))
        if missing:
            raise ValueError(f"allocation return history is insufficient: {missing}")
        finite_values = window.to_numpy(dtype=float)
        if not np.isfinite(finite_values[~window.isna().to_numpy()]).all():
            raise ValueError("allocation return history must be finite")

    if mode == "inverse_vol":
        assert window is not None
        volatility = window.std(ddof=1)
        if not np.isfinite(volatility).all() or (volatility <= 0).any():
            raise ValueError("inverse_vol requires positive finite volatility for every asset")
        raw = (1.0 / volatility).to_numpy(dtype=float)
        raw = raw / raw.sum() * total
        desired = _project_box_simplex(raw, 0.0, cap, total=total)
        weights = _constrain_turnover(
            desired,
            current.to_numpy(dtype=float),
            lower=0.0,
            upper=cap,
            total=total,
            max_turnover=turnover_limit,
            turnover_offset=outside_turnover,
        )
        return pd.Series(weights, index=assets, name="weight")

    if covariance_override is not None and not isinstance(covariance_override, pd.DataFrame):
        raise TypeError("covariance_override must be a pandas DataFrame")
    covariance = (
        covariance_override.copy(deep=True)
        if covariance_override is not None
        else estimate_covariance(
            window,
            shrinkage=float(settings["covariance_shrinkage"]),
        )
    )
    if linear_costs is None:
        costs = pd.Series(0.0, index=assets)
    else:
        if not isinstance(linear_costs, pd.Series) or not linear_costs.index.is_unique:
            raise ValueError("linear_costs must be a Series with unique assets")
        missing_costs = sorted(set(assets) - set(linear_costs.index), key=str)
        if missing_costs:
            raise ValueError(f"linear_costs is missing assets: {missing_costs}")
        costs = pd.to_numeric(linear_costs.copy(deep=True).reindex(assets), errors="coerce")
    if not np.isfinite(costs.to_numpy(dtype=float)).all() or (costs < 0).any():
        raise ValueError("linear_costs must be finite and non-negative")
    sleeve_current = current / total
    validated_factor_exposures, portfolio_factor_bounds = _validated_factor_constraints(
        assets, factor_exposures, factor_bounds
    )
    sleeve_factor_bounds = (
        None
        if not portfolio_factor_bounds
        else {
            factor: (float(bounds[0]) / total, float(bounds[1]) / total)
            for factor, bounds in portfolio_factor_bounds.items()
        }
    )
    try:
        result = optimize_mean_variance(
            numeric_scores,
            covariance,
            current_weights=sleeve_current,
            linear_costs=costs,
            risk_aversion=float(settings["risk_aversion"]),
            turnover_penalty=float(settings["turnover_penalty"]),
            turnover_offset=outside_turnover / total,
            constraints=OptimizationConstraints(
                max_weight=cap / total,
                max_turnover=turnover_limit / total,
            ),
            factor_exposures=validated_factor_exposures,
            factor_bounds=sleeve_factor_bounds,
        )
    except ValueError as exc:
        if "joint constraints are infeasible" in str(exc) and validated_factor_exposures is None:
            raise ValueError(
                "allocation.max_turnover is infeasible for the current portfolio and target sleeve"
            ) from exc
        raise
    if not result.converged:
        raise RuntimeError(
            f"cost_aware optimization did not converge after {result.iterations} iterations"
        )
    weights = result.weights * total
    weights.name = "weight"
    return weights


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.2,
    annualization: int = 252,
) -> pd.DataFrame:
    """Diagonal-target covariance shrinkage for unstable small samples."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset")
    if not returns.columns.is_unique:
        raise ValueError("returns columns must be unique")
    if (
        isinstance(shrinkage, bool)
        or not isinstance(shrinkage, (int, float))
        or not np.isfinite(shrinkage)
        or not 0 <= shrinkage <= 1
    ):
        raise ValueError("shrinkage must be in [0, 1]")
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, (int, float))
        or not np.isfinite(annualization)
        or annualization <= 0
    ):
        raise ValueError("annualization must be positive and finite")
    clean = returns.apply(pd.to_numeric, errors="coerce")
    introduced_missing = returns.notna() & clean.isna()
    if introduced_missing.any().any():
        raise ValueError("returns must contain only numeric values or missing observations")
    observed = clean.to_numpy(dtype=float)
    if not np.isfinite(observed[~clean.isna().to_numpy()]).all():
        raise ValueError("returns must contain only finite observations")
    clean = clean.dropna(how="all")
    sample = clean.cov(min_periods=2)
    if sample.isna().any().any():
        raise ValueError("return history cannot estimate covariance for every asset pair")
    sample *= float(annualization)
    diagonal = pd.DataFrame(
        np.diag(np.diag(sample.to_numpy())), index=sample.index, columns=sample.columns
    )
    covariance = (1 - shrinkage) * sample + shrinkage * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.to_numpy())
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-12)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    return pd.DataFrame(repaired, index=sample.index, columns=sample.columns)


def _project_box_simplex(
    values: np.ndarray,
    lower: float,
    upper: float,
    total: float = 1.0,
) -> np.ndarray:
    count = len(values)
    if lower * count > total + 1e-12 or upper * count < total - 1e-12:
        raise ValueError("weight bounds cannot satisfy the budget constraint")
    if abs(lower * count - total) <= 1e-14:
        return np.repeat(lower, count)
    if abs(upper * count - total) <= 1e-14:
        return np.repeat(upper, count)
    projected = np.zeros(count, dtype=float)
    free = np.ones(count, dtype=bool)
    for _ in range(count + 1):
        remaining = total - float(projected[~free].sum())
        shift = (float(values[free].sum()) - remaining) / int(free.sum())
        candidates = values[free] - shift
        below = candidates < lower
        above = candidates > upper
        if not below.any() and not above.any():
            projected[free] = candidates
            return projected
        free_indices = np.flatnonzero(free)
        projected[free_indices[below]] = lower
        projected[free_indices[above]] = upper
        free[free_indices[below | above]] = False
        if not free.any():
            if abs(float(projected.sum()) - total) <= 1e-12:
                return projected
            break
    low = float(np.min(values - upper))
    high = float(np.max(values - lower))
    for _ in range(80):
        midpoint = (low + high) / 2
        fallback = np.clip(values - midpoint, lower, upper)
        if fallback.sum() > total:
            low = midpoint
        else:
            high = midpoint
    return np.clip(values - (low + high) / 2, lower, upper)


def _constrain_turnover(
    desired: np.ndarray,
    current: np.ndarray,
    *,
    lower: float,
    upper: float,
    total: float,
    max_turnover: float,
    turnover_offset: float = 0.0,
) -> np.ndarray:
    """Move toward a desired feasible target without hiding unavoidable trades."""

    if max_turnover < 0 or turnover_offset < 0:
        raise ValueError("turnover limits and offsets must be non-negative")
    anchor = _project_box_simplex(current, lower, upper, total=total)

    def turnover(target: np.ndarray) -> float:
        return float(np.abs(target - current).sum()) + turnover_offset

    minimum = turnover(anchor)
    if minimum > max_turnover + 1e-10:
        raise ValueError(
            "allocation.max_turnover is infeasible for the current portfolio and target sleeve"
        )
    if turnover(desired) <= max_turnover + 1e-12:
        return desired.copy()
    low = 0.0
    high = 1.0
    for _ in range(100):
        midpoint = (low + high) / 2
        candidate = anchor + midpoint * (desired - anchor)
        if turnover(candidate) <= max_turnover:
            low = midpoint
        else:
            high = midpoint
    return anchor + low * (desired - anchor)


def _project_l1_ball(values: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Euclidean projection onto an L1 ball around ``center``."""

    delta = values - center
    absolute = np.abs(delta)
    if float(absolute.sum()) <= radius:
        return values.copy()
    if radius <= 0:
        return center.copy()
    ordered = np.sort(absolute)[::-1]
    thresholds = (np.cumsum(ordered) - radius) / np.arange(1, len(ordered) + 1)
    active = np.flatnonzero(ordered > thresholds)
    theta = float(thresholds[active[-1]])
    return center + np.sign(delta) * np.maximum(absolute - theta, 0.0)


def _validated_constraints(
    constraints: OptimizationConstraints | None,
    assets: list[str],
) -> OptimizationConstraints:
    raw = constraints or OptimizationConstraints(max_weight=1.0)
    if not isinstance(raw, OptimizationConstraints):
        raise TypeError("constraints must be OptimizationConstraints")

    def finite_number(name: str, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"constraints.{name} must be numeric")
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"constraints.{name} must be finite")
        return result

    lower = finite_number("min_weight", raw.min_weight)
    upper = finite_number("max_weight", raw.max_weight)
    max_turnover = finite_number("max_turnover", raw.max_turnover)
    if lower < 0 or upper > 1 or lower > upper:
        raise ValueError("constraint weights must satisfy 0 <= min_weight <= max_weight <= 1")
    if lower * len(assets) > 1 + 1e-12 or upper * len(assets) < 1 - 1e-12:
        raise ValueError("weight bounds cannot satisfy the budget constraint")
    if max_turnover < 0:
        raise ValueError("constraints.max_turnover must be non-negative")
    if not isinstance(raw.group_by_asset, Mapping) or not isinstance(raw.group_caps, Mapping):
        raise TypeError("group_by_asset and group_caps must be mappings")
    group_by_asset = dict(raw.group_by_asset)
    if not all(isinstance(group, str) for group in group_by_asset.values()):
        raise TypeError("group_by_asset values must be strings")
    unknown_assets = sorted(set(group_by_asset) - set(assets), key=str)
    if unknown_assets:
        raise ValueError(f"group_by_asset contains unknown assets: {unknown_assets}")
    group_caps: dict[str, float] = {}
    represented_groups = set(group_by_asset.values())
    for group, raw_cap in dict(raw.group_caps).items():
        if not isinstance(group, str):
            raise TypeError("group_caps keys must be strings")
        cap = finite_number(f"group_caps[{group!r}]", raw_cap)
        if not 0 <= cap <= 1:
            raise ValueError(f"group cap for {group!r} must be in [0, 1]")
        if group not in represented_groups:
            raise ValueError(f"group cap for {group!r} has no mapped assets")
        member_count = sum(mapped_group == group for mapped_group in group_by_asset.values())
        if member_count * lower > cap + 1e-12:
            raise ValueError(f"group cap for {group!r} is infeasible with constraints.min_weight")
        group_caps[group] = cap
    grouped_capacity = sum(
        min(group_caps.get(group, 1.0), count * upper)
        for group, count in {
            group: list(group_by_asset.values()).count(group) for group in represented_groups
        }.items()
    )
    ungrouped_count = len(assets) - len(group_by_asset)
    if grouped_capacity + ungrouped_count * upper < 1.0 - 1e-12:
        raise ValueError("group caps and asset bounds cannot satisfy the budget constraint")
    return OptimizationConstraints(lower, upper, max_turnover, group_by_asset, group_caps)


def _validated_factor_constraints(
    assets: list[str],
    factor_exposures: pd.DataFrame | None,
    factor_bounds: Mapping[str, tuple[float, float]] | None,
) -> tuple[pd.DataFrame | None, dict[str, tuple[float, float]]]:
    if (factor_exposures is None) != (factor_bounds is None):
        raise ValueError("factor_exposures and factor_bounds must be provided together")
    if factor_exposures is None:
        return None, {}
    if not isinstance(factor_exposures, pd.DataFrame):
        raise TypeError("factor_exposures must be a pandas DataFrame")
    if not isinstance(factor_bounds, Mapping):
        raise TypeError("factor_bounds must be a mapping")
    exposures = factor_exposures.copy(deep=True)
    if not exposures.index.is_unique or not exposures.columns.is_unique:
        raise ValueError("factor_exposures asset and factor labels must be unique")
    if exposures.shape[1] == 0:
        raise ValueError("factor_exposures must contain at least one factor")
    missing_assets = sorted(set(assets) - set(exposures.index), key=str)
    extra_assets = sorted(set(exposures.index) - set(assets), key=str)
    if missing_assets or extra_assets:
        raise ValueError(
            "factor_exposures assets must exactly match expected_returns; "
            f"missing={missing_assets}, extra={extra_assets}"
        )
    if not all(isinstance(factor, str) for factor in exposures.columns):
        raise TypeError("factor_exposures column labels must be strings")
    bounds_copy = dict(factor_bounds)
    missing_bounds = sorted(set(exposures.columns) - set(bounds_copy), key=str)
    extra_bounds = sorted(set(bounds_copy) - set(exposures.columns), key=str)
    if missing_bounds or extra_bounds:
        raise ValueError(
            "factor_bounds must exactly match factor_exposures columns; "
            f"missing={missing_bounds}, extra={extra_bounds}"
        )
    numeric = exposures.reindex(index=assets).apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("factor_exposures must contain finite values for every asset and factor")
    validated_bounds: dict[str, tuple[float, float]] = {}
    for factor in numeric.columns:
        bounds = bounds_copy[factor]
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise TypeError(f"factor_bounds[{factor!r}] must be a (lower, upper) tuple")
        lower, upper = bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
        ):
            raise TypeError(f"factor_bounds[{factor!r}] must contain numeric values")
        lower_value = float(lower)
        upper_value = float(upper)
        if not np.isfinite(lower_value) or not np.isfinite(upper_value):
            raise ValueError(f"factor_bounds[{factor!r}] must be finite")
        if lower_value > upper_value:
            raise ValueError(f"factor_bounds[{factor!r}] lower bound exceeds upper bound")
        validated_bounds[factor] = (lower_value, upper_value)
    return numeric, validated_bounds


def _constraint_violations(
    weights: np.ndarray,
    assets: list[str],
    current: np.ndarray,
    turnover_offset: float,
    constraints: OptimizationConstraints,
    factor_exposures: pd.DataFrame | None,
    factor_bounds: Mapping[str, tuple[float, float]],
    *,
    tolerance: float,
) -> dict[str, float]:
    violations: dict[str, float] = {}

    def record(name: str, value: float) -> None:
        if value > tolerance:
            violations[name] = value

    record("budget", abs(float(weights.sum()) - 1.0))
    record("min_weight", float(np.max(constraints.min_weight - weights)))
    record("max_weight", float(np.max(weights - constraints.max_weight)))
    turnover = float(np.abs(weights - current).sum()) + turnover_offset
    record("turnover", turnover - constraints.max_turnover)
    groups = np.array([constraints.group_by_asset.get(asset, "__ungrouped__") for asset in assets])
    for group, cap in constraints.group_caps.items():
        record(f"group_cap[{group}]", float(weights[groups == group].sum()) - cap)
    if factor_exposures is not None:
        exposure_values = factor_exposures.to_numpy(dtype=float).T @ weights
        for factor, value in zip(factor_exposures.columns, exposure_values, strict=True):
            lower, upper = factor_bounds[factor]
            record(f"factor_lower[{factor}]", lower - float(value))
            record(f"factor_upper[{factor}]", float(value) - upper)
    return violations


def _project_joint_feasible_set(
    values: np.ndarray,
    assets: list[str],
    current: np.ndarray,
    turnover_offset: float,
    constraints: OptimizationConstraints,
    factor_exposures: pd.DataFrame | None,
    factor_bounds: Mapping[str, tuple[float, float]],
    *,
    tolerance: float,
    trade_penalties: np.ndarray | None = None,
    penalty_scale: float = 0.0,
    max_iterations: int = 20_000,
) -> np.ndarray:
    """Apply the joint constraint/trading-cost prox using Dykstra's algorithm."""

    turnover_radius = constraints.max_turnover - turnover_offset
    if turnover_radius < -tolerance:
        raise ValueError("joint constraints are infeasible: turnover_offset exceeds max_turnover")
    turnover_radius = max(turnover_radius, 0.0)
    required_reductions = float(np.maximum(current - constraints.max_weight, 0.0).sum())
    required_increases = float(np.maximum(constraints.min_weight - current, 0.0).sum())
    minimum_turnover = max(
        abs(1.0 - float(current.sum())),
        required_reductions + required_increases,
    )
    groups_for_bound = np.array(
        [constraints.group_by_asset.get(asset, "__ungrouped__") for asset in assets]
    )
    for group, cap in constraints.group_caps.items():
        mask = groups_for_bound == group
        current_group = float(current[mask].sum())
        current_outside = float(current[~mask].sum())
        required_group_sales = max(current_group - cap, 0.0)
        required_outside_buys = max(1.0 - cap - current_outside, 0.0)
        minimum_turnover = max(
            minimum_turnover,
            required_group_sales + required_outside_buys,
        )
    if minimum_turnover > turnover_radius + tolerance:
        raise ValueError(
            "joint constraints are infeasible: minimum required turnover "
            f"{minimum_turnover + turnover_offset:.6g} exceeds max_turnover "
            f"{constraints.max_turnover:.6g}"
        )
    if factor_exposures is not None:
        available = 1.0 - len(assets) * constraints.min_weight
        room_per_asset = constraints.max_weight - constraints.min_weight
        for factor in factor_exposures.columns:
            coefficients = factor_exposures[factor].to_numpy(dtype=float)
            base_exposure = constraints.min_weight * float(coefficients.sum())
            minimum_exposure = base_exposure
            remaining = available
            for coefficient in sorted(coefficients):
                allocation = min(room_per_asset, remaining)
                minimum_exposure += allocation * float(coefficient)
                remaining -= allocation
                if remaining <= tolerance:
                    break
            maximum_exposure = base_exposure
            remaining = available
            for coefficient in sorted(coefficients, reverse=True):
                allocation = min(room_per_asset, remaining)
                maximum_exposure += allocation * float(coefficient)
                remaining -= allocation
                if remaining <= tolerance:
                    break
            lower, upper = factor_bounds[factor]
            if lower > maximum_exposure + tolerance or upper < minimum_exposure - tolerance:
                raise ValueError(
                    f"joint constraints are infeasible: factor_bounds[{factor!r}] cannot be "
                    "satisfied with budget and asset bounds"
                )
    projectors: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = []
    if trade_penalties is not None and np.any(trade_penalties > 0):
        thresholds = penalty_scale * trade_penalties

        def proximal_trade_cost(point: np.ndarray) -> np.ndarray:
            trade = point - current
            return current + np.sign(trade) * np.maximum(np.abs(trade) - thresholds, 0.0)

        projectors.append(("trade_cost", proximal_trade_cost))
    projectors.append(
        (
            "budget_and_bounds",
            lambda point: _project_box_simplex(
                point,
                constraints.min_weight,
                constraints.max_weight,
                total=1.0,
            ),
        )
    )

    groups = groups_for_bound
    halfspaces: list[tuple[str, np.ndarray, float]] = []
    for group, cap in constraints.group_caps.items():
        halfspaces.append((f"group_cap[{group}]", (groups == group).astype(float), cap))
    if factor_exposures is not None:
        for factor in factor_exposures.columns:
            exposure = factor_exposures[factor].to_numpy(dtype=float)
            lower, upper = factor_bounds[factor]
            halfspaces.append((f"factor_upper[{factor}]", exposure, upper))
            halfspaces.append((f"factor_lower[{factor}]", -exposure, -lower))
    for name, normal, upper in halfspaces:
        squared_norm = float(normal @ normal)

        def project_halfspace(
            point: np.ndarray,
            *,
            normal: np.ndarray = normal,
            upper: float = upper,
            squared_norm: float = squared_norm,
        ) -> np.ndarray:
            excess = float(normal @ point) - upper
            if excess <= 0 or squared_norm == 0:
                return point.copy()
            return point - (excess / squared_norm) * normal

        projectors.append((name, project_halfspace))
    projectors.append(
        (
            "turnover",
            lambda point: _project_l1_ball(point, current, turnover_radius),
        )
    )

    projected = values.astype(float, copy=True)
    corrections = [np.zeros_like(projected) for _ in projectors]
    for _ in range(max_iterations):
        previous = projected.copy()
        for index, (_, projector) in enumerate(projectors):
            corrected = projected + corrections[index]
            next_projected = projector(corrected)
            corrections[index] = corrected - next_projected
            projected = next_projected
        violations = _constraint_violations(
            projected,
            assets,
            current,
            turnover_offset,
            constraints,
            factor_exposures,
            factor_bounds,
            tolerance=tolerance,
        )
        if float(np.max(np.abs(projected - previous))) <= tolerance and not violations:
            return projected
    violations = _constraint_violations(
        projected,
        assets,
        current,
        turnover_offset,
        constraints,
        factor_exposures,
        factor_bounds,
        tolerance=tolerance,
    )
    detail = ", ".join(f"{name}={value:.6g}" for name, value in violations.items())
    if not detail:
        detail = "projection did not converge"
    raise ValueError(f"joint constraints are infeasible: {detail}")


def optimize_mean_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    current_weights: pd.Series | None = None,
    linear_costs: pd.Series | None = None,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.0,
    turnover_offset: float = 0.0,
    constraints: OptimizationConstraints | None = None,
    factor_exposures: pd.DataFrame | None = None,
    factor_bounds: Mapping[str, tuple[float, float]] | None = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Proximal-gradient long-only optimizer.

    Maximizes expected return minus quadratic risk, linear trading costs and an
    additional turnover penalty. The non-smooth trading objective and all
    constraints share one Dykstra proximal step on every iteration.
    """
    if not isinstance(expected_returns, pd.Series):
        raise TypeError("expected_returns must be a pandas Series")
    if not expected_returns.index.is_unique:
        raise ValueError("expected_returns index must be unique")
    assets = list(expected_returns.index)
    if not assets:
        raise ValueError("expected_returns is empty")
    numeric_expected = pd.to_numeric(expected_returns.copy(deep=True), errors="coerce")
    if (
        numeric_expected.isna().any()
        or not np.isfinite(numeric_expected.to_numpy(dtype=float)).all()
    ):
        raise ValueError("expected_returns must be finite for every asset")
    if (
        isinstance(risk_aversion, bool)
        or not isinstance(risk_aversion, (int, float))
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0
    ):
        raise ValueError("risk_aversion must be positive and finite")
    if (
        isinstance(turnover_penalty, bool)
        or not isinstance(turnover_penalty, (int, float))
        or not np.isfinite(turnover_penalty)
        or turnover_penalty < 0
    ):
        raise ValueError("turnover_penalty must be non-negative and finite")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations <= 0
    ):
        raise ValueError("max_iterations must be a positive integer")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError("tolerance must be positive and finite")
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("covariance asset labels must be unique")
    missing_rows = sorted(set(assets) - set(covariance.index), key=str)
    missing_columns = sorted(set(assets) - set(covariance.columns), key=str)
    if missing_rows or missing_columns:
        raise ValueError(
            f"covariance is missing assets: rows={missing_rows}, columns={missing_columns}"
        )
    numeric_covariance = (
        covariance.copy(deep=True)
        .reindex(index=assets, columns=assets)
        .apply(pd.to_numeric, errors="coerce")
    )
    if (
        numeric_covariance.isna().any().any()
        or not np.isfinite(numeric_covariance.to_numpy(dtype=float)).all()
    ):
        raise ValueError("covariance must contain finite values for every asset pair")
    cov = numeric_covariance.to_numpy(dtype=float)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    covariance_eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalue_tolerance = max(float(np.max(np.abs(cov))) * 1e-10, 1e-12)
    if float(covariance_eigenvalues.min()) < -eigenvalue_tolerance:
        raise ValueError("covariance must be positive semidefinite")

    settings = _validated_constraints(constraints, assets)
    validated_exposures, validated_factor_bounds = _validated_factor_constraints(
        assets, factor_exposures, factor_bounds
    )
    if current_weights is None:
        current = np.repeat(1 / len(assets), len(assets))
    else:
        if not isinstance(current_weights, pd.Series) or not current_weights.index.is_unique:
            raise ValueError("current_weights must be a Series with unique assets")
        numeric_current = pd.to_numeric(current_weights.copy(deep=True), errors="coerce")
        if (
            numeric_current.isna().any()
            or not np.isfinite(numeric_current.to_numpy(dtype=float)).all()
        ):
            raise ValueError("current_weights must be finite and non-negative")
        unexpected_current = sorted(set(numeric_current.index) - set(assets), key=str)
        if unexpected_current:
            raise ValueError(
                "current_weights contains assets outside expected_returns; pass their required "
                f"liquidation through turnover_offset: {unexpected_current}"
            )
        current = numeric_current.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    if not np.isfinite(current).all() or (current < 0).any():
        raise ValueError("current_weights must be finite and non-negative")
    if (
        isinstance(turnover_offset, bool)
        or not isinstance(turnover_offset, (int, float))
        or not np.isfinite(turnover_offset)
        or turnover_offset < 0
    ):
        raise ValueError("turnover_offset must be finite and non-negative")
    if linear_costs is None:
        costs = np.zeros(len(assets))
    else:
        if not isinstance(linear_costs, pd.Series) or not linear_costs.index.is_unique:
            raise ValueError("linear_costs must be a Series with unique assets")
        missing_costs = sorted(set(assets) - set(linear_costs.index), key=str)
        if missing_costs:
            raise ValueError(f"linear_costs is missing assets: {missing_costs}")
        numeric_costs = pd.to_numeric(linear_costs.copy(deep=True).reindex(assets), errors="coerce")
        costs = numeric_costs.to_numpy(dtype=float)
        if not np.isfinite(costs).all() or (costs < 0).any():
            raise ValueError("linear_costs must be finite and non-negative for every asset")
    mu = numeric_expected.to_numpy(dtype=float)
    largest_eigenvalue = max(float(covariance_eigenvalues.max()), 1e-12)
    step = 0.5 / (risk_aversion * largest_eigenvalue + 1.0)
    projection_tolerance = max(min(float(tolerance), 1e-10), 1e-12)
    weights = _project_joint_feasible_set(
        current,
        assets,
        current,
        float(turnover_offset),
        settings,
        validated_exposures,
        validated_factor_bounds,
        tolerance=projection_tolerance,
    )
    converged = False

    for iteration in range(1, max_iterations + 1):
        gradient = mu - risk_aversion * (cov @ weights)
        candidate = _project_joint_feasible_set(
            weights + step * gradient,
            assets,
            current,
            float(turnover_offset),
            settings,
            validated_exposures,
            validated_factor_bounds,
            tolerance=projection_tolerance,
            trade_penalties=costs + turnover_penalty,
            penalty_scale=step,
        )
        if float(np.max(np.abs(candidate - weights))) <= tolerance:
            weights = candidate
            converged = True
            break
        weights = candidate
    else:
        iteration = max_iterations

    violations = _constraint_violations(
        weights,
        assets,
        current,
        float(turnover_offset),
        settings,
        validated_exposures,
        validated_factor_bounds,
        tolerance=max(float(tolerance) * 10, 1e-9),
    )
    if violations:
        detail = ", ".join(f"{name}={value:.6g}" for name, value in violations.items())
        raise RuntimeError(f"optimizer produced an infeasible portfolio: {detail}")

    trade = weights - current
    expected = float(mu @ weights)
    variance = max(float(weights @ cov @ weights), 0.0)
    turnover = float(np.abs(trade).sum()) + turnover_offset
    objective = (
        expected
        - 0.5 * risk_aversion * variance
        - float(costs @ np.abs(trade))
        - turnover_penalty * turnover
    )
    group_weights: dict[str, float] = {}
    for asset, weight in zip(assets, weights, strict=True):
        group = settings.group_by_asset.get(asset, "__ungrouped__")
        group_weights[group] = group_weights.get(group, 0.0) + float(weight)
    return OptimizationResult(
        weights=pd.Series(weights, index=assets, name="weight"),
        expected_return=expected,
        volatility=float(np.sqrt(variance)),
        turnover=turnover,
        objective=objective,
        group_weights=group_weights,
        converged=converged,
        iterations=iteration,
    )


def estimate_capacity(
    target_weights: pd.Series,
    average_daily_value: pd.Series,
    *,
    max_participation: float = 0.1,
    liquidation_days: int = 1,
) -> dict[str, float | str]:
    """Maximum portfolio capital implied by the tightest liquidity position."""
    if not 0 < max_participation <= 1 or liquidation_days <= 0:
        raise ValueError("invalid participation or liquidation horizon")
    weights = target_weights.abs()
    adv = average_daily_value.reindex(weights.index)
    if adv.isna().any() or (adv <= 0).any():
        raise ValueError("average_daily_value must be positive for every asset")
    active = weights[weights > 0]
    if active.empty:
        raise ValueError("target portfolio has no active weights")
    capacity_by_asset = adv[active.index] * max_participation * liquidation_days / active
    binding = str(capacity_by_asset.idxmin())
    return {
        "capacity": float(capacity_by_asset.min()),
        "binding_asset": binding,
        "max_participation": max_participation,
        "liquidation_days": liquidation_days,
    }


def square_root_impact_cost(
    order_notional: pd.Series,
    average_daily_value: pd.Series,
    daily_volatility: pd.Series,
    *,
    impact_coefficient: float = 0.1,
) -> pd.DataFrame:
    """Square-root market-impact estimate by asset."""
    assets = order_notional.index
    adv = average_daily_value.reindex(assets)
    vol = daily_volatility.reindex(assets)
    if adv.isna().any() or vol.isna().any() or (adv <= 0).any() or (vol < 0).any():
        raise ValueError("ADV and volatility must be valid for every order")
    participation = order_notional.abs() / adv
    impact_rate = impact_coefficient * vol * np.sqrt(participation)
    return pd.DataFrame(
        {
            "order_notional": order_notional,
            "participation": participation,
            "impact_rate": impact_rate,
            "impact_cost": order_notional.abs() * impact_rate,
        }
    )
