from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest
from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec, MarginMode
from quant_data_kit.exceptions import ValidationError
from quant_execution import PortfolioRiskSnapshot, PositionRiskSnapshot, Side

from quant_portfolio.cross_asset import (
    CrossAssetConstraints,
    CrossAssetInput,
    PITFixedPoint,
    PITMarketSnapshot,
    TargetPortfolio,
    _decimal,
    _project_weights,
    _quantize_quantity,
    optimize_cross_asset,
    target_to_order_intents,
)

DECISION_TIME = datetime(2026, 8, 29, 1, tzinfo=timezone.utc)


def fp(value: str | int, scale: int = 2) -> FixedPoint:
    return FixedPoint.from_decimal(value, scale)


def instrument(
    instrument_id: str,
    asset_class: AssetClass,
    venue: str,
    *,
    multiplier: str = "1",
    margin_mode: MarginMode = MarginMode.CASH,
) -> InstrumentSpec:
    return InstrumentSpec(
        instrument_id=instrument_id,
        asset_class=asset_class,
        product_type="linear",
        venue=venue,
        native_symbol=instrument_id,
        settlement_currency="USD",
        price_tick=fp("0.01"),
        quantity_step=fp("1", 0),
        contract_multiplier=fp(multiplier, 0),
        calendar_id="UTC_24X7",
        effective_from=DECISION_TIME - timedelta(days=365),
        available_at=DECISION_TIME - timedelta(days=1),
        margin_mode=margin_mode,
    )


def pit(
    price: str, adv: str = "1000000", *, available_at: datetime | None = None
) -> PITMarketSnapshot:
    observed_at = available_at or DECISION_TIME - timedelta(minutes=1)
    return PITMarketSnapshot(
        reference_price=PITFixedPoint(fp(price), observed_at),
        fx_to_base=PITFixedPoint(fp("1", 6), observed_at),
        average_daily_value_base=PITFixedPoint(fp(adv), observed_at),
    )


def inputs(*, crypto_adv: str = "1000000") -> tuple[CrossAssetInput, ...]:
    return (
        CrossAssetInput(
            instrument("ASHARE:600000", AssetClass.EQUITY, "SSE"),
            pit("10"),
            "alpha-cn",
            0.15,
            0.02,
            linear_cost_bps=5,
        ),
        CrossAssetInput(
            instrument(
                "CRYPTO:BTC-USDT-PERP", AssetClass.CRYPTO, "BINANCE", margin_mode=MarginMode.CROSS
            ),
            pit("10000", crypto_adv),
            "basis-crypto",
            -0.10,
            0.05,
            linear_cost_bps=2,
            impact_coefficient=0.15,
            initial_margin_rate=0.10,
            maintenance_margin_rate=0.05,
        ),
        CrossAssetInput(
            instrument(
                "FUTURE:IF2609",
                AssetClass.FUTURE,
                "CFFEX",
                multiplier="10",
                margin_mode=MarginMode.CROSS,
            ),
            pit("2000"),
            "carry-future",
            0.08,
            0.03,
            linear_cost_bps=3,
            initial_margin_rate=0.10,
            maintenance_margin_rate=0.06,
        ),
    )


def snapshot() -> PortfolioRiskSnapshot:
    positions = (
        PositionRiskSnapshot(
            instrument_id="ASHARE:600000",
            asset_class=AssetClass.EQUITY,
            venue="SSE",
            settlement_currency="USD",
            quantity=fp("1000", 0),
            mark_price=fp("10"),
            base_notional=fp("10000"),
            initial_margin=fp("0"),
            maintenance_margin=fp("0"),
        ),
        PositionRiskSnapshot(
            instrument_id="CRYPTO:BTC-USDT-PERP",
            asset_class=AssetClass.CRYPTO,
            venue="BINANCE",
            settlement_currency="USD",
            quantity=fp("-1", 0),
            mark_price=fp("10000"),
            base_notional=fp("-10000"),
            initial_margin=fp("1000"),
            maintenance_margin=fp("500"),
        ),
        PositionRiskSnapshot(
            instrument_id="FUTURE:IF2609",
            asset_class=AssetClass.FUTURE,
            venue="CFFEX",
            settlement_currency="USD",
            quantity=fp("1", 0),
            mark_price=fp("2000"),
            base_notional=fp("20000"),
            initial_margin=fp("2000"),
            maintenance_margin=fp("1200"),
        ),
    )
    return PortfolioRiskSnapshot(
        account_id="paper-m5",
        event_time=DECISION_TIME,
        base_currency="USD",
        nav=fp("100000"),
        cash_value=fp("80000"),
        gross_exposure=fp("40000"),
        net_exposure=fp("20000"),
        initial_margin=fp("3000"),
        maintenance_margin=fp("1700"),
        positions=positions,
    )


def constraints(**changes: object) -> CrossAssetConstraints:
    values: dict[str, object] = {
        "max_gross_leverage": 1.5,
        "min_net_leverage": -0.6,
        "max_net_leverage": 0.8,
        "max_single_instrument": 0.7,
        "max_turnover": 1.4,
        "max_adv_participation": 0.2,
        "max_days_to_liquidate": 5.0,
        "max_initial_margin_utilization": 0.5,
        "max_maintenance_margin_utilization": 0.3,
        "asset_class_caps": {"equity": 0.7, "future": 0.7, "crypto": 0.7},
        "currency_caps": {"USD": 1.5},
        "venue_caps": {"SSE": 0.7, "CFFEX": 0.7, "BINANCE": 0.7},
        "strategy_caps": {"alpha-cn": 0.7, "carry-future": 0.7, "basis-crypto": 0.7},
    }
    values.update(changes)
    return CrossAssetConstraints(**values)  # type: ignore[arg-type]


def expected() -> tuple[pd.Series, pd.DataFrame]:
    assets = [item.instrument.instrument_id for item in inputs()]
    returns = pd.Series([0.15, -0.10, 0.08], index=assets)
    covariance = pd.DataFrame(np.diag([0.04, 0.09, 0.05]), index=assets, columns=assets)
    return returns, covariance


def optimize(**changes: object):
    returns, covariance = expected()
    decision_time = changes.pop("decision_time", DECISION_TIME)
    return optimize_cross_asset(
        returns,
        covariance,
        portfolio_snapshot=snapshot(),
        decision_time=decision_time,
        inputs=changes.pop("inputs", inputs()),
        constraints=changes.pop("constraints", constraints()),
        **changes,
    )


def test_cross_asset_golden_a_share_futures_crypto_is_cash_aware() -> None:
    result = optimize()
    assert result.feasible and result.target and result.report and result.failure is None
    assert tuple(result.target.quantities) == (
        "ASHARE:600000",
        "CRYPTO:BTC-USDT-PERP",
        "FUTURE:IF2609",
    )
    assert result.report.gross_leverage <= 1.5
    assert -0.6 <= result.report.net_leverage <= 0.8
    assert result.report.margin_utilization <= 0.5
    assert result.report.maintenance_margin_utilization <= 0.3
    assert result.report.cash_residual.to_decimal() >= 0
    assert result.report.expected_total_cost == fp(
        result.report.expected_linear_cost.to_decimal()
        + result.report.expected_impact_cost.to_decimal()
    )


def test_crypto_spot_is_supported_by_the_same_cash_aware_contract() -> None:
    spot = CrossAssetInput(
        instrument("CRYPTO:ETH-USDT-SPOT", AssetClass.CRYPTO, "BINANCE"),
        pit("2000"),
        "spot-alpha",
        0.05,
        0.04,
        linear_cost_bps=4,
    )
    spot_snapshot = PortfolioRiskSnapshot(
        account_id="paper-spot",
        event_time=DECISION_TIME,
        base_currency="USD",
        nav=fp("100000"),
        cash_value=fp("98000"),
        gross_exposure=fp("2000"),
        net_exposure=fp("2000"),
        initial_margin=fp("0"),
        maintenance_margin=fp("0"),
        positions=(
            PositionRiskSnapshot(
                instrument_id=spot.instrument.instrument_id,
                asset_class=AssetClass.CRYPTO,
                venue="BINANCE",
                settlement_currency="USD",
                quantity=fp("1", 0),
                mark_price=fp("2000"),
                base_notional=fp("2000"),
                initial_margin=fp("0"),
                maintenance_margin=fp("0"),
            ),
        ),
    )
    result = optimize_cross_asset(
        pd.Series([0.05], index=[spot.instrument.instrument_id]),
        pd.DataFrame(
            [[0.04]],
            index=[spot.instrument.instrument_id],
            columns=[spot.instrument.instrument_id],
        ),
        portfolio_snapshot=spot_snapshot,
        decision_time=DECISION_TIME,
        inputs=(spot,),
        constraints=constraints(
            asset_class_caps={"crypto": 0.7},
            currency_caps={"usd": 1.5},
            venue_caps={"binance": 0.7},
            strategy_caps={"spot-alpha": 0.7},
        ),
    )
    assert result.feasible and result.target and result.report
    assert result.report.cash_residual.to_decimal() >= 0
    assert all(
        intent.instrument_id == "CRYPTO:ETH-USDT-SPOT"
        for intent in target_to_order_intents(
            result.target, portfolio_snapshot=spot_snapshot, inputs=(spot,)
        )
    )


def test_cross_asset_target_emits_only_deterministic_order_intents() -> None:
    result = optimize()
    assert result.target is not None
    runs = [
        target_to_order_intents(result.target, portfolio_snapshot=snapshot(), inputs=inputs())
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]
    digest = [sha256(repr(run).encode()).hexdigest() for run in runs]
    assert digest[0] == digest[1] == digest[2]
    assert all(intent.quantity.units > 0 for intent in runs[0])
    assert {intent.instrument_id for intent in runs[0]} <= set(result.target.quantities)


def test_reduce_only_close_then_open_on_crossing_target() -> None:
    target = TargetPortfolio(
        decision_time=DECISION_TIME,
        account_id="paper-m5",
        base_currency="USD",
        quantities={
            "ASHARE:600000": fp("1000", 0),
            "CRYPTO:BTC-USDT-PERP": fp("2", 0),
            "FUTURE:IF2609": fp("1", 0),
        },
        weights={"ASHARE:600000": 0.1, "CRYPTO:BTC-USDT-PERP": 0.2, "FUTURE:IF2609": 0.2},
    )
    intents = target_to_order_intents(target, portfolio_snapshot=snapshot(), inputs=inputs())
    crypto = [intent for intent in intents if intent.instrument_id == "CRYPTO:BTC-USDT-PERP"]
    assert [(intent.side, intent.quantity, intent.reduce_only) for intent in crypto] == [
        (Side.BUY, fp("1", 0), True),
        (Side.BUY, fp("2", 0), False),
    ]
    assert all(intent.instrument_id != "ASHARE:600000" for intent in intents)


def test_current_constraint_breach_is_structured_and_never_relaxed() -> None:
    result = optimize(constraints=constraints(max_gross_leverage=0.1, max_single_instrument=0.1))
    assert not result.feasible
    assert result.target is None and result.report is None and result.failure is not None
    assert result.failure.code == "CURRENT_PORTFOLIO_CONSTRAINT_BREACH"
    assert {binding.code for binding in result.failure.bindings} >= {"GROSS_LEVERAGE"}


def test_adv_and_days_to_liquidate_infeasibility_is_structured() -> None:
    result = optimize(
        inputs=inputs(crypto_adv="100"), constraints=constraints(max_adv_participation=0.001)
    )
    assert not result.feasible and result.report and result.failure
    assert result.failure.code == "TARGET_PORTFOLIO_INFEASIBLE"
    assert {binding.code for binding in result.failure.bindings} & {
        "ADV_PARTICIPATION",
        "DAYS_TO_LIQUIDATE",
    }


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda: tuple(
                list(inputs()[:1])
                + [
                    CrossAssetInput(
                        inputs()[0].instrument,
                        pit("10"),
                        "duplicate",
                        0.1,
                        0.02,
                    )
                ]
                + list(inputs()[1:])
            ),
            "duplicate instrument input",
        ),
        (
            lambda: tuple(
                CrossAssetInput(
                    item.instrument,
                    PITMarketSnapshot(
                        item.market.reference_price,
                        PITFixedPoint(fp("1", 6), DECISION_TIME + timedelta(seconds=1)),
                        item.market.average_daily_value_base,
                    ),
                    item.strategy_id,
                    item.expected_return,
                    item.daily_volatility,
                    item.linear_cost_bps,
                    item.impact_coefficient,
                    item.initial_margin_rate,
                    item.maintenance_margin_rate,
                )
                for item in inputs()
            ),
            "fx_to_base is not available",
        ),
    ],
)
def test_missing_or_future_pit_inputs_fail_closed(mutator, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        optimize(inputs=mutator())


def test_nonfinite_signal_and_missing_qexec_position_input_fail_closed() -> None:
    returns, covariance = expected()
    returns.iloc[0] = np.nan
    with pytest.raises(ValidationError, match="expected_returns must be finite"):
        optimize_cross_asset(
            returns,
            covariance,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    with pytest.raises(ValidationError, match="every QExec position"):
        optimize(inputs=inputs()[:-1])


def test_contract_boundaries_and_invalid_constraints_fail_closed() -> None:
    with pytest.raises(ValidationError, match="event_time must equal"):
        optimize(decision_time=DECISION_TIME + timedelta(seconds=1))
    with pytest.raises(ValidationError, match="max_single_instrument"):
        constraints(max_single_instrument=2.0, max_gross_leverage=1.0)
    with pytest.raises(ValidationError, match="target quantities must be sorted"):
        TargetPortfolio(
            decision_time=DECISION_TIME,
            account_id="paper-m5",
            base_currency="USD",
            quantities={"Z": fp("1", 0), "A": fp("1", 0)},
            weights={"A": 0.0, "Z": 0.0},
        )


def test_cross_asset_public_type_and_mapping_boundaries_fail_closed() -> None:
    with pytest.raises(ValidationError, match="public values must be FixedPoint"):
        _decimal("10")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="instrument must be an InstrumentSpec"):
        CrossAssetInput("not-an-instrument", pit("10"), "s", 0.1, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="market must be a PITMarketSnapshot"):
        CrossAssetInput(instrument("X", AssetClass.EQUITY, "SSE"), "not-a-market", "s", 0.1, 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="keys must be non-empty strings"):
        constraints(asset_class_caps={"": 0.5})
    with pytest.raises(ValidationError, match="min_cash_base must be FixedPoint"):
        constraints(min_cash_base=100)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="min_net_leverage must be finite"):
        constraints(min_net_leverage=float("nan"))


def test_cross_asset_input_lifecycle_and_inverse_contracts_fail_closed() -> None:
    future_effective = replace(
        inputs()[0].instrument, effective_from=DECISION_TIME + timedelta(seconds=1)
    )
    with pytest.raises(ValidationError, match="not effective"):
        optimize(inputs=(replace(inputs()[0], instrument=future_effective), *inputs()[1:]))

    inverse = replace(inputs()[0].instrument, inverse=True)
    with pytest.raises(ValidationError, match="inverse instruments"):
        optimize(inputs=(replace(inputs()[0], instrument=inverse), *inputs()[1:]))


def test_cross_asset_optimizer_input_and_snapshot_types_fail_closed() -> None:
    returns, covariance = expected()
    with pytest.raises(ValidationError, match="QExec PortfolioRiskSnapshot"):
        optimize_cross_asset(
            returns,
            covariance,
            portfolio_snapshot=object(),  # type: ignore[arg-type]
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    with pytest.raises(ValidationError, match="at least one"):
        optimize_cross_asset(
            pd.Series(dtype=float),
            pd.DataFrame(),
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=(),
            constraints=constraints(),
        )
    with pytest.raises(ValidationError, match="portfolio NAV must be positive"):
        optimize_cross_asset(
            returns,
            covariance,
            portfolio_snapshot=replace(snapshot(), nav=fp("0")),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    with pytest.raises(ValidationError, match="expected_returns must be a pandas Series"):
        optimize_cross_asset(
            returns.tolist(),  # type: ignore[arg-type]
            covariance,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    with pytest.raises(ValidationError, match="covariance must be a pandas DataFrame"):
        optimize_cross_asset(
            returns,
            covariance.to_numpy().tolist(),  # type: ignore[arg-type]
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    invalid_covariance = covariance.copy()
    invalid_covariance.iloc[0, 0] = np.nan
    with pytest.raises(ValidationError, match="covariance must contain finite"):
        optimize_cross_asset(
            returns,
            invalid_covariance,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )


def test_projection_covers_empty_sign_buckets_unknown_mapping_and_turnover() -> None:
    upper_only = _project_weights(
        np.array([-0.5, -0.5, -0.5]),
        np.zeros(3),
        inputs(),
        constraints(
            max_gross_leverage=6.0,
            max_single_instrument=2.0,
            min_net_leverage=-3.0,
            max_net_leverage=-2.0,
            max_turnover=10.0,
        ),
    )
    assert np.allclose(upper_only, [-0.5, -0.5, -0.5])

    lower_only = _project_weights(
        np.array([0.5, 0.5, 0.5]),
        np.zeros(3),
        inputs(),
        constraints(
            max_gross_leverage=6.0,
            max_single_instrument=2.0,
            min_net_leverage=2.0,
            max_net_leverage=3.0,
            max_turnover=10.0,
        ),
    )
    assert np.allclose(lower_only, [0.5, 0.5, 0.5])

    lower_with_negative = _project_weights(
        np.array([-0.5, -0.5, -0.5]),
        np.zeros(3),
        inputs(),
        constraints(
            max_gross_leverage=6.0,
            max_single_instrument=2.0,
            min_net_leverage=-1.0,
            max_net_leverage=1.0,
            max_turnover=10.0,
        ),
    )
    assert np.sum(lower_with_negative) >= -1.0 - 1e-10

    turnover_limited = _project_weights(
        np.array([0.5, -0.5, 0.5]),
        np.zeros(3),
        inputs(),
        constraints(max_turnover=0.1),
    )
    assert np.abs(turnover_limited).sum() <= 0.1 + 1e-10
    unknown_bucket = _project_weights(
        np.array([0.1, 0.1, 0.1]),
        np.zeros(3),
        inputs(),
        constraints(asset_class_caps={"unknown": 0.1}),
    )
    assert np.allclose(unknown_bucket, [0.1, 0.1, 0.1])


@pytest.mark.parametrize(
    ("change", "binding"),
    [
        ({"min_net_leverage": 0.5}, "NET_LEVERAGE_MIN"),
        ({"max_net_leverage": 0.1}, "NET_LEVERAGE_MAX"),
        ({"min_cash_base": fp("90000")}, "CASH"),
    ],
)
def test_current_portfolio_reports_net_and_cash_constraint_bindings(
    change: dict[str, object], binding: str
) -> None:
    result = optimize(constraints=constraints(**change))
    assert not result.feasible and result.failure is not None
    assert binding in {item.code for item in result.failure.bindings}


def test_optimizer_iteration_limit_and_order_time_in_force_fail_closed() -> None:
    result = optimize(max_iterations=1, tolerance=1e-30)
    assert result.iterations == 1
    assert result.target is not None
    with pytest.raises(ValidationError, match="time_in_force must be a TimeInForce"):
        target_to_order_intents(
            result.target,
            portfolio_snapshot=snapshot(),
            inputs=inputs(),
            time_in_force="DAY",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: PITFixedPoint("not-fixed", DECISION_TIME), "PIT value"),
        (
            lambda: PITMarketSnapshot(
                PITFixedPoint(fp("0"), DECISION_TIME),
                pit("10").fx_to_base,
                pit("10").average_daily_value_base,
            ),
            "reference_price",
        ),
        (
            lambda: CrossAssetInput(
                instrument("X", AssetClass.EQUITY, "SSE"), pit("10"), "", 0.1, 0.1
            ),
            "strategy_id",
        ),
        (
            lambda: CrossAssetInput(
                instrument("X", AssetClass.EQUITY, "SSE"), pit("10"), "s", float("nan"), 0.1
            ),
            "expected_return",
        ),
        (
            lambda: CrossAssetInput(
                instrument("X", AssetClass.EQUITY, "SSE"),
                pit("10"),
                "s",
                0.1,
                0.1,
                initial_margin_rate=0.1,
                maintenance_margin_rate=0.2,
            ),
            "maintenance_margin_rate",
        ),
        (lambda: CrossAssetConstraints(1, 1, 0, 1, 1, 1, 1), "min_net_leverage"),
        (
            lambda: CrossAssetConstraints(1, -1, 1, 1, 1, 1, 1, asset_class_caps=[]),
            "asset_class_caps",
        ),
    ],
)
def test_contract_constructors_reject_invalid_values(factory, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory()


@pytest.mark.parametrize(
    "field_name",
    (
        "daily_volatility",
        "linear_cost_bps",
        "impact_coefficient",
        "initial_margin_rate",
        "maintenance_margin_rate",
    ),
)
def test_negative_cost_risk_and_margin_inputs_fail_closed(field_name: str) -> None:
    with pytest.raises(ValidationError, match=field_name):
        replace(inputs()[0], **{field_name: -0.01})


def test_case_insensitive_caps_have_one_consistent_meaning() -> None:
    result = optimize(constraints=constraints(currency_caps={"usd": 0.1}))
    assert not result.feasible and result.failure
    assert any(binding.code == "CURRENCY" for binding in result.failure.bindings)
    with pytest.raises(ValidationError, match="unique ignoring case"):
        constraints(currency_caps={"USD": 0.5, "usd": 0.4})


def test_target_portfolio_contract_rejects_bad_public_values() -> None:
    base = {
        "decision_time": DECISION_TIME,
        "account_id": "paper-m5",
        "base_currency": "USD",
        "quantities": {"A": fp("1", 0)},
        "weights": {"A": 0.0},
    }
    with pytest.raises(ValidationError, match="account_id"):
        TargetPortfolio(**(base | {"account_id": ""}))
    with pytest.raises(ValidationError, match="base_currency"):
        TargetPortfolio(**(base | {"base_currency": ""}))
    with pytest.raises(ValidationError, match="FixedPoint"):
        TargetPortfolio(**(base | {"quantities": {"A": 1}}))
    with pytest.raises(ValidationError, match="match quantities"):
        TargetPortfolio(**(base | {"weights": {"B": 0.0}}))
    with pytest.raises(ValidationError, match="finite"):
        TargetPortfolio(**(base | {"weights": {"A": float("nan")}}))


def test_projection_enforces_group_gross_net_and_turnover_without_relaxing_limits() -> None:
    projected = _project_weights(
        np.array([2.0, -2.0, 2.0]),
        np.array([0.1, -0.1, 0.2]),
        inputs(),
        constraints(
            max_gross_leverage=0.8,
            min_net_leverage=-0.1,
            max_net_leverage=0.1,
            max_single_instrument=0.7,
            max_turnover=0.3,
            asset_class_caps={"equity": 0.2, "crypto": 0.2, "future": 0.2},
        ),
    )
    assert np.abs(projected).sum() <= 0.8 + 1e-10
    assert np.abs(projected - np.array([0.1, -0.1, 0.2])).sum() <= 0.3 + 1e-10


def test_unavailable_spec_and_price_that_rounds_to_zero_fail_closed() -> None:
    future_spec = replace(inputs()[0].instrument, available_at=DECISION_TIME + timedelta(seconds=1))
    invalid_spec = replace(inputs()[0], instrument=future_spec)
    with pytest.raises(ValidationError, match="instrument spec is not available"):
        optimize(inputs=(invalid_spec, *inputs()[1:]))
    coarse_spec = replace(inputs()[0].instrument, price_tick=fp("100"))
    coarse_input = replace(inputs()[0], instrument=coarse_spec)
    with pytest.raises(ValidationError, match="reference price rounds to zero"):
        optimize(inputs=(coarse_input, *inputs()[1:]))


def test_optimizer_and_order_conversion_reject_invalid_boundary_contracts() -> None:
    returns, covariance = expected()
    with pytest.raises(ValidationError, match="risk_aversion"):
        optimize_cross_asset(
            returns,
            covariance,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
            risk_aversion=0,
        )
    with pytest.raises(ValidationError, match="max_iterations"):
        optimize(max_iterations=0)
    with pytest.raises(ValidationError, match="tolerance"):
        optimize(tolerance=0)
    with pytest.raises(ValidationError, match="expected_returns index"):
        optimize_cross_asset(
            returns.sort_index(ascending=False),
            covariance,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    asymmetric = covariance.copy()
    asymmetric.iloc[0, 1] = 0.1
    with pytest.raises(ValidationError, match="symmetric"):
        optimize_cross_asset(
            returns,
            asymmetric,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    indefinite = covariance.copy()
    indefinite.iloc[0, 0] = -0.1
    with pytest.raises(ValidationError, match="positive semidefinite"):
        optimize_cross_asset(
            returns,
            indefinite,
            portfolio_snapshot=snapshot(),
            decision_time=DECISION_TIME,
            inputs=inputs(),
            constraints=constraints(),
        )
    target = TargetPortfolio(
        decision_time=DECISION_TIME,
        account_id="wrong",
        base_currency="USD",
        quantities={
            "ASHARE:600000": fp("1000", 0),
            "CRYPTO:BTC-USDT-PERP": fp("-1", 0),
            "FUTURE:IF2609": fp("1", 0),
        },
        weights={"ASHARE:600000": 0.1, "CRYPTO:BTC-USDT-PERP": -0.1, "FUTURE:IF2609": 0.2},
    )
    with pytest.raises(ValidationError, match="same account"):
        target_to_order_intents(target, portfolio_snapshot=snapshot(), inputs=inputs())
    incomplete_target = TargetPortfolio(
        decision_time=DECISION_TIME,
        account_id="paper-m5",
        base_currency="USD",
        quantities={"ASHARE:600000": fp("1000", 0)},
        weights={"ASHARE:600000": 0.1},
    )
    with pytest.raises(ValidationError, match="exactly match"):
        target_to_order_intents(incomplete_target, portfolio_snapshot=snapshot(), inputs=inputs())


def test_quantization_is_toward_zero_and_zero_orders_are_not_emitted() -> None:
    assert _quantize_quantity(Decimal("-1.9"), fp("1", 0)) == fp("-1", 0)
    target = TargetPortfolio(
        decision_time=DECISION_TIME,
        account_id="paper-m5",
        base_currency="USD",
        quantities={
            "ASHARE:600000": fp("1000", 0),
            "CRYPTO:BTC-USDT-PERP": fp("-1", 0),
            "FUTURE:IF2609": fp("1", 0),
        },
        weights={"ASHARE:600000": 0.1, "CRYPTO:BTC-USDT-PERP": -0.1, "FUTURE:IF2609": 0.2},
    )
    assert target_to_order_intents(target, portfolio_snapshot=snapshot(), inputs=inputs()) == ()
