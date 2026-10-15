from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.executable import repair_integer_target


def test_200_asset_repair_has_bounded_exact_work_and_preserves_account_rules():
    assets = [f"A{i:03d}" for i in range(200)]
    weights = pd.Series(0.95 / 200, index=assets)
    kwargs = {
        "current": {},
        "prices": dict.fromkeys(assets, Decimal(10)),
        "lots": dict.fromkeys(assets, 100),
        "cash": Decimal(1000000),
        "nav": Decimal(1000000),
        "quote_trade": lambda a, q: (-Decimal(q) * 10 - Decimal(5), Decimal(5)),
        "constraint_residuals": lambda w: {"cap": w.max() - 0.05, "budget": w.sum() - 0.95},
        "max_invested": 0.95,
    }
    result = repair_integer_target(weights, **kwargs)
    assert result.diagnostics["exact_candidate_evaluations"] < 3000
    assert result.cash + sum(result.quantities.values()) * 10 == Decimal(1000000) - result.costs
    assert all(q % 100 == 0 for q in result.quantities.values())
    assert result.weights.sum() <= 0.95 + 1e-7
    assert result.diagnostics["budget_shortfall"] < 0.002
    repeated = repair_integer_target(weights.iloc[::-1], **kwargs)
    assert repeated.quantities == result.quantities
    assert repeated.cash == result.cash


def test_exchange_can_repair_when_no_single_move_reduces_worst_constraint():
    # Fixed budget and A minimum jointly require an exchange from B to A.
    result = repair_integer_target(
        pd.Series({"A": 0.5, "B": 0.5}),
        current={"A": 0, "B": 1000},
        prices={"A": Decimal(10), "B": Decimal(10)},
        lots={"A": 100, "B": 100},
        cash=Decimal(0),
        nav=Decimal(10000),
        quote_trade=lambda a, q: (-Decimal(q) * 10, Decimal(0)),
        constraint_residuals=lambda w: {"budget": abs(w.sum() - 1), "A": 0.6 - w.A},
        max_invested=1,
    )
    assert result.quantities == {"A": 600, "B": 400}
    assert any(len(move) == 2 for move in result.diagnostics["repair_actions"])
    assert result.cash == 0


@pytest.mark.parametrize("value", [0, -1, True, 1.5, np.inf])
def test_exchange_budget_must_be_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_exchange_candidates"):
        repair_integer_target(
            pd.Series({"A": 1.0}),
            current={},
            prices={},
            lots={},
            cash=1,
            nav=1,
            quote_trade=None,
            constraint_residuals=None,
            max_invested=1,
            max_exchange_candidates=value,
        )
