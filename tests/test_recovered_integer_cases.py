import gzip
import json
import time
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.executable import repair_integer_target
from quant_portfolio.feasibility import minimum_risk_seed
from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.solver import OptimizationError, RiskMatrix, constraint_residuals


def exact_quote(quantity, mark, day):
    if not quantity:
        return Decimal(0), Decimal(0)
    direction = 1 if quantity > 0 else -1
    shifted = mark * (1 + Decimal(direction) * Decimal(".0005"))
    price = (shifted / Decimal(".01")).to_integral_value(
        rounding=ROUND_CEILING if direction > 0 else ROUND_FLOOR
    ) * Decimal(".01")
    notional = abs(quantity) * price
    fee = max(Decimal(5), notional * Decimal(".0003"))
    if quantity < 0:
        fee += notional * (
            Decimal(".001") if day < pd.Timestamp("2023-08-28") else Decimal(".0005")
        )
    fee = fee.quantize(Decimal(".00000001"), rounding=ROUND_HALF_UP)
    cash_change = -direction * notional - fee
    return cash_change, -cash_change - quantity * mark


def replay(case):
    d = case["new"]["diagnostics"]
    a = d["account_input"]
    ctx = dict(d["problem"])
    ctx["settings"] = OptimizationConstraints(**ctx["settings"])
    ctx["risk"] = RiskMatrix(**ctx["risk"])
    assets = ctx["assets"]
    for key in ("current", "lower", "upper", "benchmark", "factor_exposures"):
        if ctx.get(key) is not None:
            ctx[key] = np.asarray(ctx[key])
    marks = {k: Decimal(v) for k, v in a["marks"].items()}

    def check(w):
        return constraint_residuals(
            w.reindex(assets, fill_value=0).to_numpy(),
            ctx["current"],
            ctx["settings"],
            assets=assets,
            budget=ctx["budget"],
            budget_equality=False,
            risk=ctx["risk"],
            benchmark=ctx["benchmark"],
            factor_exposures=ctx["factor_exposures"],
            factor_bounds=ctx["factor_bounds"],
            detailed_asset_bounds=True,
        )

    def seed(remaining):
        point, info = minimum_risk_seed(ctx, min(5, remaining / 2))
        return pd.Series(point, index=assets).reindex(sorted(marks)), info

    started = time.perf_counter()
    row = {k: case[k] for k in ("scenario", "date")}
    continuous_seconds = d["continuous_solver"].get(
        "total_seconds", d["continuous_solver"].get("seconds", 0)
    )
    remaining = 30 - continuous_seconds
    try:
        result = repair_integer_target(
            pd.Series(d["continuous_weights"]).reindex(sorted(marks), fill_value=0),
            current=a["current_quantities"],
            prices=marks,
            lots=a["lots"],
            cash=a["cash"],
            nav=a["nav"],
            non_cash_assets=a.get("non_cash_assets", "0"),
            fixed_quantities=a["fixed_targets"],
            max_buy=a["max_buy"],
            max_sell=a["max_sell"],
            odd_lot_sell_assets=a["odd_lot_sell_assets"],
            quote_trade=lambda asset, q: exact_quote(q, marks[asset], pd.Timestamp(case["date"])),
            constraint_residuals=check,
            max_invested=ctx["budget"],
            feasibility_seed=seed,
            timeout_seconds=remaining,
        )
        row.update(
            status="published",
            targets=result.quantities,
            diagnostics=result.diagnostics,
            residual=max(check(result.weights).values()),
        )
    except OptimizationError as exc:
        row.update(status=exc.status, diagnostics=exc.serializable_diagnostics())
    row["seconds"] = time.perf_counter() - started
    row["combined_recorded_seconds"] = continuous_seconds + row["seconds"]
    return row


with gzip.open(
    Path(__file__).parent / "fixtures/recovered_input_integer_failures.json.gz",
    "rt",
    encoding="utf8",
) as stream:
    CASES = json.load(stream)
KNOWN = {
    (9, "2024-06-11"),
    (14, "2024-06-11"),
    (19, "2023-09-06"),
    (26, "2023-08-16"),
    (50, "2023-08-23"),
}


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if (c["scenario"], c["date"]) in KNOWN],
    ids=lambda c: str(c["scenario"]) + "-" + c["date"],
)
def test_frozen_witness_cases_repaired_from_continuous_target(case):
    result = replay(case)
    assert result["status"] == "published"
    assert result["residual"] <= 1e-7
    assert result["seconds"] <= 30
    assert result["combined_recorded_seconds"] <= 30
    assert result["diagnostics"]["integer_global_optimum"] is False
