"""Deterministic portfolio optimization with explicit constraints and costs."""

from __future__ import annotations

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


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    shrinkage: float = 0.2,
    annualization: int = 252,
) -> pd.DataFrame:
    """Diagonal-target covariance shrinkage for unstable small samples."""
    if returns.empty or returns.shape[1] == 0:
        raise ValueError("returns must contain at least one asset")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    clean = returns.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    sample = clean.cov(min_periods=2).fillna(0.0) * annualization
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
    low = float(np.min(values - upper))
    high = float(np.max(values - lower))
    for _ in range(100):
        midpoint = (low + high) / 2
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > total:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - high, lower, upper)
    projected += (total - projected.sum()) / count
    return np.clip(projected, lower, upper)


def _enforce_group_caps(
    weights: np.ndarray,
    assets: list[str],
    constraints: OptimizationConstraints,
) -> np.ndarray:
    if not constraints.group_by_asset or not constraints.group_caps:
        return weights
    result = weights.copy()
    groups = np.array([constraints.group_by_asset.get(asset, "__ungrouped__") for asset in assets])
    for _ in range(20):
        changed = False
        for group, cap in constraints.group_caps.items():
            mask = groups == group
            group_weight = float(result[mask].sum())
            if group_weight <= cap + 1e-12:
                continue
            excess = group_weight - cap
            result[mask] *= cap / group_weight
            eligible = ~mask & (result < constraints.max_weight - 1e-12)
            room = np.maximum(constraints.max_weight - result[eligible], 0.0)
            if room.sum() + 1e-12 < excess:
                raise ValueError(f"group cap for {group} makes constraints infeasible")
            result[eligible] += excess * room / room.sum()
            changed = True
        if not changed:
            break
    return _project_box_simplex(result, constraints.min_weight, constraints.max_weight, total=1.0)


def optimize_mean_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    current_weights: pd.Series | None = None,
    linear_costs: pd.Series | None = None,
    risk_aversion: float = 5.0,
    turnover_penalty: float = 0.0,
    constraints: OptimizationConstraints | None = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-10,
) -> OptimizationResult:
    """Projected-gradient long-only optimizer.

    Maximizes expected return minus quadratic risk, linear trading costs and an
    additional turnover penalty. Constraints are enforced on every iteration.
    """
    assets = list(expected_returns.index)
    if not assets:
        raise ValueError("expected_returns is empty")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    covariance = covariance.reindex(index=assets, columns=assets)
    if covariance.isna().any().any():
        raise ValueError("covariance is missing assets")
    settings = constraints or OptimizationConstraints(max_weight=1.0)
    current = (
        current_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        if current_weights is not None
        else np.repeat(1 / len(assets), len(assets))
    )
    current = _project_box_simplex(current, settings.min_weight, settings.max_weight, total=1.0)
    costs = (
        linear_costs.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        if linear_costs is not None
        else np.zeros(len(assets))
    )
    mu = expected_returns.to_numpy(dtype=float)
    cov = covariance.to_numpy(dtype=float)
    largest_eigenvalue = max(float(np.linalg.eigvalsh(cov).max()), 1e-12)
    step = 0.5 / (risk_aversion * largest_eigenvalue + 1.0)
    weights = current.copy()
    converged = False

    for iteration in range(1, max_iterations + 1):
        trade = weights - current
        gradient = mu - risk_aversion * (cov @ weights)
        gradient -= (costs + turnover_penalty) * np.sign(trade)
        candidate = _project_box_simplex(
            weights + step * gradient,
            settings.min_weight,
            settings.max_weight,
            total=1.0,
        )
        candidate = _enforce_group_caps(candidate, assets, settings)
        turnover = float(np.abs(candidate - current).sum())
        if turnover > settings.max_turnover:
            blend = settings.max_turnover / turnover
            candidate = current + blend * (candidate - current)
        if float(np.max(np.abs(candidate - weights))) <= tolerance:
            weights = candidate
            converged = True
            break
        weights = candidate
    else:
        iteration = max_iterations

    trade = weights - current
    expected = float(mu @ weights)
    variance = max(float(weights @ cov @ weights), 0.0)
    turnover = float(np.abs(trade).sum())
    objective = (
        expected
        - 0.5 * risk_aversion * variance
        - float((costs + turnover_penalty) @ np.abs(trade))
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
