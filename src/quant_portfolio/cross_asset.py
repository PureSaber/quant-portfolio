"""Causal, deterministic cross-asset target construction.

This module deliberately has no broker, ledger, network, or credential code.  It
consumes immutable QExec/QDK contracts and can only emit QExec ``OrderIntent``
suggestions.  All boundary monetary values remain ``FixedPoint``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from hashlib import sha256
from math import isfinite, sqrt
from types import MappingProxyType

import numpy as np
import pandas as pd
from quant_data_kit import FixedPoint, InstrumentSpec, ensure_utc_datetime
from quant_data_kit.exceptions import ValidationError
from quant_execution import (
    OrderIntent,
    OrderType,
    PortfolioRiskSnapshot,
    Side,
    TimeInForce,
)

_EPSILON = 1e-10


def _decimal(value: FixedPoint) -> Decimal:
    if not isinstance(value, FixedPoint):
        raise ValidationError("cross-asset public values must be FixedPoint")
    return value.to_decimal()


def _fixed(value: Decimal, scale: int) -> FixedPoint:
    return FixedPoint.from_decimal(value, scale, rounding=ROUND_DOWN)


def _finite_non_negative(value: float, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
    ):
        raise ValidationError(f"{field_name} must be a finite non-negative number")


def _finite_positive(value: float, field_name: str) -> None:
    _finite_non_negative(value, field_name)
    if value <= 0:
        raise ValidationError(f"{field_name} must be positive")


def _finite(value: float, field_name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
    ):
        raise ValidationError(f"{field_name} must be finite")


def _as_mapping(values: Mapping[str, float], field_name: str) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ValidationError(f"{field_name} must be a mapping")
    checked: dict[str, float] = {}
    normalized_keys: set[str] = set()
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValidationError(f"{field_name} keys must be non-empty strings")
        normalized_key = key.casefold()
        if normalized_key in normalized_keys:
            raise ValidationError(f"{field_name} keys must be unique ignoring case")
        _finite_non_negative(value, f"{field_name}[{key!r}]")
        normalized_keys.add(normalized_key)
        checked[key] = float(value)
    return MappingProxyType(checked)


@dataclass(frozen=True, slots=True)
class PITFixedPoint:
    """A fixed-point observation with the time at which it became usable."""

    value: FixedPoint
    available_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.value, FixedPoint):
            raise ValidationError("PIT value must be FixedPoint")
        object.__setattr__(
            self, "available_at", ensure_utc_datetime(self.available_at, field="available_at")
        )


@dataclass(frozen=True, slots=True)
class PITMarketSnapshot:
    """Explicit causal price, FX and liquidity inputs for one instrument."""

    reference_price: PITFixedPoint
    fx_to_base: PITFixedPoint
    average_daily_value_base: PITFixedPoint

    def __post_init__(self) -> None:
        for field_name in ("reference_price", "fx_to_base", "average_daily_value_base"):
            value = getattr(self, field_name)
            if not isinstance(value, PITFixedPoint) or value.value.units <= 0:
                raise ValidationError(f"{field_name} must be a positive PITFixedPoint")

    def validate_at(self, decision_time: datetime) -> None:
        for field_name in ("reference_price", "fx_to_base", "average_daily_value_base"):
            if getattr(self, field_name).available_at > decision_time:
                raise ValidationError(f"{field_name} is not available at decision_time")


@dataclass(frozen=True, slots=True)
class CrossAssetInput:
    """Static instrument metadata plus causal market inputs for target construction."""

    instrument: InstrumentSpec
    market: PITMarketSnapshot
    strategy_id: str
    expected_return: float
    daily_volatility: float
    linear_cost_bps: float = 0.0
    impact_coefficient: float = 0.1
    initial_margin_rate: float = 0.0
    maintenance_margin_rate: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.instrument, InstrumentSpec):
            raise ValidationError("instrument must be an InstrumentSpec")
        if not isinstance(self.market, PITMarketSnapshot):
            raise ValidationError("market must be a PITMarketSnapshot")
        if not isinstance(self.strategy_id, str) or not self.strategy_id.strip():
            raise ValidationError("strategy_id must be non-empty")
        _finite(self.expected_return, "expected_return")
        for field_name in (
            "daily_volatility",
            "linear_cost_bps",
            "impact_coefficient",
            "initial_margin_rate",
            "maintenance_margin_rate",
        ):
            _finite_non_negative(getattr(self, field_name), field_name)
        if self.maintenance_margin_rate > self.initial_margin_rate + _EPSILON:
            raise ValidationError("maintenance_margin_rate cannot exceed initial_margin_rate")
        object.__setattr__(self, "strategy_id", self.strategy_id.strip())


@dataclass(frozen=True, slots=True)
class CrossAssetConstraints:
    """Explicit cross-asset limits, all expressed in base-currency NAV ratios."""

    max_gross_leverage: float
    min_net_leverage: float
    max_net_leverage: float
    max_single_instrument: float
    max_turnover: float
    max_adv_participation: float
    max_days_to_liquidate: float
    max_initial_margin_utilization: float = 1.0
    max_maintenance_margin_utilization: float = 1.0
    min_cash_base: FixedPoint = field(default_factory=lambda: FixedPoint(0, 2))
    asset_class_caps: Mapping[str, float] = field(default_factory=dict)
    currency_caps: Mapping[str, float] = field(default_factory=dict)
    venue_caps: Mapping[str, float] = field(default_factory=dict)
    strategy_caps: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "max_gross_leverage",
            "max_single_instrument",
            "max_turnover",
            "max_adv_participation",
            "max_days_to_liquidate",
            "max_initial_margin_utilization",
            "max_maintenance_margin_utilization",
        ):
            _finite_positive(getattr(self, field_name), field_name)
        for field_name in ("min_net_leverage", "max_net_leverage"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not isfinite(value)
            ):
                raise ValidationError(f"{field_name} must be finite")
        if self.min_net_leverage > self.max_net_leverage:
            raise ValidationError("min_net_leverage must not exceed max_net_leverage")
        if self.max_single_instrument > self.max_gross_leverage + _EPSILON:
            raise ValidationError("max_single_instrument cannot exceed max_gross_leverage")
        if not isinstance(self.min_cash_base, FixedPoint):
            raise ValidationError("min_cash_base must be FixedPoint")
        for field_name in ("asset_class_caps", "currency_caps", "venue_caps", "strategy_caps"):
            object.__setattr__(self, field_name, _as_mapping(getattr(self, field_name), field_name))


@dataclass(frozen=True, slots=True)
class ConstraintBinding:
    code: str
    observed: float
    limit: float
    scope: str


@dataclass(frozen=True, slots=True)
class OptimizationFailure:
    code: str
    message: str
    bindings: tuple[ConstraintBinding, ...]


@dataclass(frozen=True, slots=True)
class TargetPortfolio:
    decision_time: datetime
    account_id: str
    base_currency: str
    quantities: Mapping[str, FixedPoint]
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "decision_time", ensure_utc_datetime(self.decision_time, field="decision_time")
        )
        if not isinstance(self.account_id, str) or not self.account_id:
            raise ValidationError("account_id must be non-empty")
        if not isinstance(self.base_currency, str) or not self.base_currency:
            raise ValidationError("base_currency must be non-empty")
        quantities = dict(self.quantities)
        if tuple(quantities) != tuple(sorted(quantities)):
            raise ValidationError("target quantities must be sorted by instrument_id")
        if any(not isinstance(value, FixedPoint) for value in quantities.values()):
            raise ValidationError("target quantities must be FixedPoint")
        object.__setattr__(self, "quantities", MappingProxyType(quantities))
        weights = dict(self.weights)
        if tuple(weights) != tuple(sorted(weights)) or set(weights) != set(quantities):
            raise ValidationError("target weights must be sorted and match quantities")
        if any(not isfinite(value) for value in weights.values()):
            raise ValidationError("target weights must be finite")
        object.__setattr__(self, "weights", MappingProxyType(weights))


@dataclass(frozen=True, slots=True)
class CrossAssetReport:
    gross_leverage: float
    net_leverage: float
    turnover: float
    margin_utilization: float
    maintenance_margin_utilization: float
    expected_linear_cost: FixedPoint
    expected_impact_cost: FixedPoint
    expected_total_cost: FixedPoint
    initial_margin: FixedPoint
    maintenance_margin: FixedPoint
    cash_residual: FixedPoint
    max_adv_participation: float
    max_days_to_liquidate: float
    binding_constraints: tuple[ConstraintBinding, ...]


@dataclass(frozen=True, slots=True)
class CrossAssetOptimizationResult:
    feasible: bool
    target: TargetPortfolio | None
    report: CrossAssetReport | None
    failure: OptimizationFailure | None
    iterations: int


def _validate_inputs(
    snapshot: PortfolioRiskSnapshot,
    decision_time: datetime,
    inputs: Sequence[CrossAssetInput],
) -> tuple[CrossAssetInput, ...]:
    if not isinstance(snapshot, PortfolioRiskSnapshot):
        raise ValidationError("portfolio_snapshot must be a QExec PortfolioRiskSnapshot")
    decision_time = ensure_utc_datetime(decision_time, field="decision_time")
    if snapshot.event_time != decision_time:
        raise ValidationError("portfolio snapshot event_time must equal decision_time")
    if not inputs:
        raise ValidationError("at least one cross-asset input is required")
    ordered = tuple(sorted(inputs, key=lambda item: item.instrument.instrument_id))
    ids = [item.instrument.instrument_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValidationError("duplicate instrument input")
    known_positions = {position.instrument_id for position in snapshot.positions}
    if not known_positions.issubset(set(ids)):
        raise ValidationError("every QExec position must have a cross-asset input")
    for item in ordered:
        item.market.validate_at(decision_time)
        if item.instrument.available_at > decision_time:
            raise ValidationError("instrument spec is not available at decision_time")
        if item.instrument.effective_from > decision_time or (
            item.instrument.effective_to is not None
            and item.instrument.effective_to <= decision_time
        ):
            raise ValidationError("instrument spec is not effective at decision_time")
        if item.instrument.inverse:
            raise ValidationError("inverse instruments are outside the v2.0 linear contract")
    return ordered


def _current_weights(
    snapshot: PortfolioRiskSnapshot, inputs: Sequence[CrossAssetInput]
) -> np.ndarray:
    nav = float(_decimal(snapshot.nav))
    if nav <= 0:
        raise ValidationError("portfolio NAV must be positive")
    by_id = {position.instrument_id: position.base_notional for position in snapshot.positions}
    return np.array(
        [
            float(
                _decimal(
                    by_id.get(item.instrument.instrument_id, FixedPoint(0, snapshot.nav.scale))
                )
                / _decimal(snapshot.nav)
            )
            for item in inputs
        ],
        dtype=float,
    )


def _scale_bucket(weights: np.ndarray, indexes: list[int], cap: float) -> None:
    observed = float(np.abs(weights[indexes]).sum())
    if observed > cap + _EPSILON:
        weights[indexes] *= cap / observed


def _bucket_value(item: CrossAssetInput, attribute: str) -> str:
    if attribute == "asset_class":
        return item.instrument.asset_class.value
    if attribute == "strategy_id":
        return item.strategy_id
    return str(getattr(item.instrument, attribute))


def _unit_base(item: CrossAssetInput) -> Decimal:
    rounded_price = _quantize_quantity(
        _decimal(item.market.reference_price.value), item.instrument.price_tick
    )
    if rounded_price.units <= 0:
        raise ValidationError("reference price rounds to zero at price_tick")
    return (
        _decimal(rounded_price)
        * _decimal(item.instrument.contract_multiplier)
        * _decimal(item.market.fx_to_base.value)
    )


def _project_weights(
    values: np.ndarray,
    current: np.ndarray,
    inputs: Sequence[CrossAssetInput],
    constraints: CrossAssetConstraints,
) -> np.ndarray:
    result = np.clip(values, -constraints.max_single_instrument, constraints.max_single_instrument)
    gross = float(np.abs(result).sum())
    if gross > constraints.max_gross_leverage:
        result *= constraints.max_gross_leverage / gross
    for attribute, caps in (
        ("asset_class", constraints.asset_class_caps),
        ("settlement_currency", constraints.currency_caps),
        ("venue", constraints.venue_caps),
        ("strategy_id", constraints.strategy_caps),
    ):
        for bucket, cap in caps.items():
            indexes = [
                index
                for index, item in enumerate(inputs)
                if _bucket_value(item, attribute).casefold() == bucket.casefold()
            ]
            if indexes:
                _scale_bucket(result, indexes, cap)
    net = float(result.sum())
    if net > constraints.max_net_leverage:
        positives = result > 0
        positive_sum = float(result[positives].sum())
        negative_sum = float(result[~positives].sum())
        if positive_sum <= 0:
            return result
        result[positives] *= max(0.0, (constraints.max_net_leverage - negative_sum) / positive_sum)
    elif net < constraints.min_net_leverage:
        negatives = result < 0
        negative_sum = float(result[negatives].sum())
        positive_sum = float(result[~negatives].sum())
        if negative_sum < 0:
            result[negatives] *= max(
                0.0, (constraints.min_net_leverage - positive_sum) / negative_sum
            )
    turnover = float(np.abs(result - current).sum())
    if turnover > constraints.max_turnover:
        result = current + (result - current) * constraints.max_turnover / turnover
    return result


def _quantize_quantity(value: Decimal, step: FixedPoint) -> FixedPoint:
    direction = Decimal(1) if value >= 0 else Decimal(-1)
    magnitude = abs(value)
    step_decimal = _decimal(step)
    rounded = (magnitude / step_decimal).to_integral_value(rounding=ROUND_DOWN) * step_decimal
    return _fixed(direction * rounded, step.scale)


def _target_quantities(
    weights: np.ndarray,
    snapshot: PortfolioRiskSnapshot,
    inputs: Sequence[CrossAssetInput],
) -> Mapping[str, FixedPoint]:
    nav = _decimal(snapshot.nav)
    quantities: dict[str, FixedPoint] = {}
    for weight, item in zip(weights, inputs, strict=True):
        unit_base = _unit_base(item)
        quantities[item.instrument.instrument_id] = _quantize_quantity(
            Decimal(str(float(weight))) * nav / unit_base,
            item.instrument.quantity_step,
        )
    return MappingProxyType(dict(sorted(quantities.items())))


def _realized_weights(
    target: Mapping[str, FixedPoint],
    snapshot: PortfolioRiskSnapshot,
    inputs: Sequence[CrossAssetInput],
) -> Mapping[str, float]:
    nav = _decimal(snapshot.nav)
    return MappingProxyType(
        {
            item.instrument.instrument_id: float(
                _decimal(target[item.instrument.instrument_id]) * _unit_base(item) / nav
            )
            for item in inputs
        }
    )


def _evaluate(
    target: Mapping[str, FixedPoint],
    snapshot: PortfolioRiskSnapshot,
    inputs: Sequence[CrossAssetInput],
    constraints: CrossAssetConstraints,
) -> tuple[CrossAssetReport, tuple[ConstraintBinding, ...]]:
    nav_decimal = _decimal(snapshot.nav)
    current = {position.instrument_id: position.quantity for position in snapshot.positions}
    weights: dict[str, float] = {}
    total_linear = Decimal(0)
    total_impact = Decimal(0)
    initial_margin = Decimal(0)
    maintenance_margin = Decimal(0)
    cash_delta = Decimal(0)
    turnover_base = Decimal(0)
    max_participation = 0.0
    max_days = 0.0
    bindings: list[ConstraintBinding] = []
    buckets: dict[str, dict[str, float]] = {
        "asset": {},
        "currency": {},
        "venue": {},
        "strategy": {},
    }
    for item in inputs:
        instrument_id = item.instrument.instrument_id
        target_quantity = _decimal(target[instrument_id])
        current_quantity = _decimal(
            current.get(instrument_id, FixedPoint(0, item.instrument.quantity_step.scale))
        )
        unit_base = _unit_base(item)
        target_notional = target_quantity * unit_base
        trade = target_quantity - current_quantity
        trade_base = abs(trade * unit_base)
        weights[instrument_id] = float(target_notional / nav_decimal)
        turnover_base += trade_base
        linear = trade_base * Decimal(str(item.linear_cost_bps)) / Decimal(10_000)
        participation = float(trade_base / _decimal(item.market.average_daily_value_base.value))
        impact = trade_base * Decimal(
            str(item.impact_coefficient * item.daily_volatility * sqrt(participation))
        )
        total_linear += linear
        total_impact += impact
        max_participation = max(max_participation, participation)
        max_days = max(max_days, participation / constraints.max_adv_participation)
        initial_margin += abs(target_notional) * Decimal(str(item.initial_margin_rate))
        maintenance_margin += abs(target_notional) * Decimal(str(item.maintenance_margin_rate))
        if item.instrument.margin_mode.value in {"none", "cash"}:
            cash_delta -= trade * unit_base
        for group, key in (
            ("asset", item.instrument.asset_class.value),
            ("currency", item.instrument.settlement_currency),
            ("venue", item.instrument.venue),
            ("strategy", item.strategy_id),
        ):
            normalized_key = key.casefold()
            buckets[group][normalized_key] = buckets[group].get(normalized_key, 0.0) + abs(
                float(target_notional / nav_decimal)
            )
    gross = float(sum(abs(weight) for weight in weights.values()))
    net = float(sum(weights.values()))
    turnover = float(turnover_base / nav_decimal)
    cash_residual = _decimal(snapshot.cash_value) + cash_delta - total_linear - total_impact
    total_cost = total_linear + total_impact
    initial_utilization = float(initial_margin / nav_decimal)
    maintenance_utilization = float(maintenance_margin / nav_decimal)

    def bind(code: str, observed: float, limit: float, scope: str) -> None:
        if observed > limit + _EPSILON:
            bindings.append(ConstraintBinding(code, observed, limit, scope))

    bind("GROSS_LEVERAGE", gross, constraints.max_gross_leverage, "portfolio")
    if net < constraints.min_net_leverage - _EPSILON:
        bindings.append(
            ConstraintBinding("NET_LEVERAGE_MIN", net, constraints.min_net_leverage, "portfolio")
        )
    if net > constraints.max_net_leverage + _EPSILON:
        bindings.append(
            ConstraintBinding("NET_LEVERAGE_MAX", net, constraints.max_net_leverage, "portfolio")
        )
    bind("TURNOVER", turnover, constraints.max_turnover, "portfolio")
    bind("ADV_PARTICIPATION", max_participation, constraints.max_adv_participation, "portfolio")
    bind("DAYS_TO_LIQUIDATE", max_days, constraints.max_days_to_liquidate, "portfolio")
    bind(
        "INITIAL_MARGIN",
        initial_utilization,
        constraints.max_initial_margin_utilization,
        "portfolio",
    )
    bind(
        "MAINTENANCE_MARGIN",
        maintenance_utilization,
        constraints.max_maintenance_margin_utilization,
        "portfolio",
    )
    if cash_residual < _decimal(constraints.min_cash_base):
        bindings.append(
            ConstraintBinding(
                "CASH",
                float(cash_residual),
                float(_decimal(constraints.min_cash_base)),
                "portfolio",
            )
        )
    for code, cap_map, bucket in (
        ("ASSET_CLASS", constraints.asset_class_caps, buckets["asset"]),
        ("CURRENCY", constraints.currency_caps, buckets["currency"]),
        ("VENUE", constraints.venue_caps, buckets["venue"]),
        ("STRATEGY", constraints.strategy_caps, buckets["strategy"]),
    ):
        for key, cap in cap_map.items():
            bind(code, bucket.get(key.casefold(), 0.0), cap, key)
    scale = snapshot.nav.scale
    report = CrossAssetReport(
        gross_leverage=gross,
        net_leverage=net,
        turnover=turnover,
        margin_utilization=initial_utilization,
        maintenance_margin_utilization=maintenance_utilization,
        expected_linear_cost=_fixed(total_linear, scale),
        expected_impact_cost=_fixed(total_impact, scale),
        expected_total_cost=_fixed(total_cost, scale),
        initial_margin=_fixed(initial_margin, scale),
        maintenance_margin=_fixed(maintenance_margin, scale),
        cash_residual=_fixed(cash_residual, scale),
        max_adv_participation=max_participation,
        max_days_to_liquidate=max_days,
        binding_constraints=tuple(bindings),
    )
    return report, tuple(bindings)


def optimize_cross_asset(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    *,
    portfolio_snapshot: PortfolioRiskSnapshot,
    decision_time: datetime,
    inputs: Sequence[CrossAssetInput],
    constraints: CrossAssetConstraints,
    risk_aversion: float = 5.0,
    max_iterations: int = 800,
    tolerance: float = 1e-10,
) -> CrossAssetOptimizationResult:
    """Return a constrained target or a structured, fail-closed infeasibility result."""
    _finite_positive(risk_aversion, "risk_aversion")
    _finite_positive(tolerance, "tolerance")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations <= 0:
        raise ValidationError("max_iterations must be a positive integer")
    if not isinstance(expected_returns, pd.Series):
        raise ValidationError("expected_returns must be a pandas Series")
    if not isinstance(covariance, pd.DataFrame):
        raise ValidationError("covariance must be a pandas DataFrame")
    ordered = _validate_inputs(portfolio_snapshot, decision_time, inputs)
    assets = [item.instrument.instrument_id for item in ordered]
    if list(expected_returns.index) != assets:
        raise ValidationError(
            "expected_returns index must exactly match sorted input instrument_ids"
        )
    covariance = covariance.reindex(index=assets, columns=assets)
    if covariance.isna().any().any() or not np.isfinite(covariance.to_numpy(dtype=float)).all():
        raise ValidationError("covariance must contain finite values for every input")
    mu = expected_returns.to_numpy(dtype=float)
    if not np.isfinite(mu).all():
        raise ValidationError("expected_returns must be finite")
    current = _current_weights(portfolio_snapshot, ordered)
    current_target = _target_quantities(current, portfolio_snapshot, ordered)
    _, current_bindings = _evaluate(current_target, portfolio_snapshot, ordered, constraints)
    if current_bindings:
        return CrossAssetOptimizationResult(
            feasible=False,
            target=None,
            report=None,
            failure=OptimizationFailure(
                "CURRENT_PORTFOLIO_CONSTRAINT_BREACH",
                "current portfolio violates enabled constraints",
                current_bindings,
            ),
            iterations=0,
        )
    cov = covariance.to_numpy(dtype=float)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValidationError("covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(cov)
    if float(eigenvalues.min()) < -1e-12:
        raise ValidationError("covariance must be positive semidefinite")
    largest = max(float(eigenvalues.max()), 1e-12)
    step = 0.5 / (risk_aversion * largest + 1.0)
    weights = current.copy()
    linear = np.array([item.linear_cost_bps / 10_000 for item in ordered], dtype=float)
    for iteration in range(1, max_iterations + 1):
        trade = weights - current
        gradient = mu - risk_aversion * (cov @ weights) - linear * np.sign(trade)
        candidate = _project_weights(weights + step * gradient, current, ordered, constraints)
        if float(np.max(np.abs(candidate - weights))) <= tolerance:
            weights = candidate
            break
        weights = candidate
    else:
        iteration = max_iterations
    quantities = _target_quantities(weights, portfolio_snapshot, ordered)
    report, bindings = _evaluate(quantities, portfolio_snapshot, ordered, constraints)
    if bindings:
        return CrossAssetOptimizationResult(
            feasible=False,
            target=None,
            report=report,
            failure=OptimizationFailure(
                "TARGET_PORTFOLIO_INFEASIBLE",
                "rounded target violates enabled constraints",
                bindings,
            ),
            iterations=iteration,
        )
    target = TargetPortfolio(
        decision_time=decision_time,
        account_id=portfolio_snapshot.account_id,
        base_currency=portfolio_snapshot.base_currency,
        quantities=quantities,
        weights=_realized_weights(quantities, portfolio_snapshot, ordered),
    )
    return CrossAssetOptimizationResult(True, target, report, None, iteration)


def target_to_order_intents(
    target: TargetPortfolio,
    *,
    portfolio_snapshot: PortfolioRiskSnapshot,
    inputs: Sequence[CrossAssetInput],
    time_in_force: TimeInForce = TimeInForce.DAY,
) -> tuple[OrderIntent, ...]:
    """Convert a target to deterministic market-order suggestions without state mutation."""
    ordered = _validate_inputs(portfolio_snapshot, target.decision_time, inputs)
    if (
        portfolio_snapshot.account_id != target.account_id
        or portfolio_snapshot.base_currency != target.base_currency
    ):
        raise ValidationError("target and portfolio snapshot must describe the same account")
    if not isinstance(time_in_force, TimeInForce):
        raise ValidationError("time_in_force must be a TimeInForce")
    expected_ids = {item.instrument.instrument_id for item in ordered}
    if set(target.quantities) != expected_ids:
        raise ValidationError("target instruments must exactly match cross-asset inputs")
    current = {
        position.instrument_id: _decimal(position.quantity)
        for position in portfolio_snapshot.positions
    }
    intents: list[OrderIntent] = []
    for item in ordered:
        instrument_id = item.instrument.instrument_id
        target_quantity = _decimal(target.quantities[instrument_id])
        current_quantity = current.get(instrument_id, Decimal(0))
        legs: list[tuple[Decimal, bool]]
        if current_quantity and target_quantity and current_quantity * target_quantity < 0:
            legs = [(-current_quantity, True), (target_quantity, False)]
        else:
            delta = target_quantity - current_quantity
            reducing = (
                current_quantity
                and abs(target_quantity) < abs(current_quantity)
                and current_quantity * target_quantity >= 0
            )
            legs = [(delta, bool(reducing))]
        for sequence, (delta, reduce_only) in enumerate(legs):
            quantity = _quantize_quantity(abs(delta), item.instrument.quantity_step)
            if quantity.units == 0:
                continue
            side = Side.BUY if delta > 0 else Side.SELL
            seed = "|".join(
                (
                    target.account_id,
                    item.strategy_id,
                    instrument_id,
                    target.decision_time.isoformat(),
                    str(sequence),
                    side.value,
                    str(quantity.units),
                    str(quantity.scale),
                    str(reduce_only),
                )
            )
            intents.append(
                OrderIntent(
                    idempotency_key=sha256(seed.encode("utf-8")).hexdigest(),
                    account_id=target.account_id,
                    strategy_id=item.strategy_id,
                    instrument_id=instrument_id,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    time_in_force=time_in_force,
                    created_at=target.decision_time,
                    reduce_only=reduce_only,
                )
            )
    return tuple(intents)
