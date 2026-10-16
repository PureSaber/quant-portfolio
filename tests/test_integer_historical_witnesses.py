"""Frozen 200-company examples with independently reconciled exact trade costs."""

import json
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.executable import repair_integer_target
from quant_portfolio.feasibility import minimum_risk_seed
from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.solver import RiskMatrix, constraint_residuals

EXAMPLES = json.loads(
    (Path(__file__).parent / "fixtures/barra_200_integer_witnesses.json").read_text()
)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda x: x["model"])
def test_original_constraints_admit_executable_historical_cost_aware_target(example):
    context = dict(example["context"])
    context["settings"] = OptimizationConstraints(**context["settings"])
    context["risk"] = RiskMatrix(**context["risk"])
    assets = context["assets"]
    for key in ("current", "lower", "upper", "benchmark"):
        context[key] = np.asarray(context[key])
    marks = {a: Decimal(p) for a, p in example["marks"].items()}

    def quote(asset, quantity):
        direction = 1 if quantity > 0 else -1
        tick = Decimal(".01")
        price = marks[asset] * (1 + direction * Decimal(".0005"))
        price = (price / tick).to_integral_value(
            rounding=ROUND_CEILING if quantity > 0 else ROUND_FLOOR
        ) * tick
        notional = abs(quantity) * price
        fees = max(Decimal(5), notional * Decimal(".0003"))
        if quantity < 0:
            fees += notional * Decimal(".0005")
        fees = fees.quantize(Decimal(".00000001"), rounding=ROUND_HALF_UP)
        change = -direction * notional - fees
        return change, -change - quantity * marks[asset]

    def check(weights):
        return constraint_residuals(
            weights.reindex(assets).to_numpy(),
            context["current"],
            context["settings"],
            assets=assets,
            budget=0.8,
            budget_equality=False,
            risk=context["risk"],
            benchmark=context["benchmark"],
            detailed_asset_bounds=True,
        )

    def seed(remaining):
        point, diagnostics = minimum_risk_seed(context, min(5, remaining / 2))
        return pd.Series(point, index=assets), diagnostics

    result = repair_integer_target(
        pd.Series(example["weights"]),
        current={},
        prices=marks,
        lots=dict.fromkeys(assets, 100),
        cash=1000000,
        nav=1000000,
        quote_trade=quote,
        constraint_residuals=check,
        max_invested=0.8,
        feasibility_seed=seed,
    )
    assert max(check(result.weights).values()) <= 1e-7
    assert all(q % 100 == 0 for q in result.quantities.values())
    assert result.cash >= 0
    assert result.cash + sum(result.quantities[a] * marks[a] for a in result.quantities) == (
        Decimal(1000000) - result.costs
    )
    assert result.diagnostics["integer_global_optimum"] is False


def test_odd_remainder_can_exit_but_cannot_be_bought_or_partly_disposed():
    kwargs = {
        "current": {"A": 155},
        "prices": {"A": Decimal(10)},
        "lots": {"A": 100},
        "cash": 0,
        "nav": 1550,
        "quote_trade": lambda a, q: (-Decimal(q) * 10, Decimal(0)),
        "constraint_residuals": lambda w: {},
        "max_invested": 1,
    }
    result = repair_integer_target(pd.Series({"A": 0.0}), odd_lot_sell_assets={"A"}, **kwargs)
    assert result.quantities == {}
    assert result.cash == 1550
    # The default remains conservative for instruments without an odd-lot rule.
    result = repair_integer_target(pd.Series({"A": 0.0}), **kwargs)
    assert result.quantities == {"A": 55}
    with pytest.raises(ValueError, match="fixed target"):
        repair_integer_target(
            pd.Series({"A": 0.0}), odd_lot_sell_assets={"A"}, fixed_quantities={"A": 25}, **kwargs
        )
