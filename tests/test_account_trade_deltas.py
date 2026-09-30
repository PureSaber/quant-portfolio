"""Synthetic statement CLI regressions for fee-aware, executable trade deltas."""

import csv
import json
import subprocess
import sys
from decimal import Decimal as D

import pytest

from quant_portfolio.account_import import propose


def card(weight="0.25"):
    return {
        "schema_version": "quant.decision/v1",
        "status": "paper_ready",
        "run_id": "synthetic-regression",
        "as_of": "2026-09-29",
        "valid_until": "2099-01-01T00:00:00Z",
        "targets": [{"symbol": "000001", "weight": weight}],
    }


def config():
    return {"as_of": "2026-09-29T15:00:00+08:00", "managed_symbols": ["000001"]}


def statement_cli(tmp_path, quantity, weight="0.25"):
    cash = 100000 - quantity * 10
    schemas = {
        "opening_positions": (
            ["symbol", "quantity", "average_cost", "acquired_on"],
            [["000001", quantity, "10", "2026-09-25"]],
        ),
        "trades": (["trade_id", "symbol", "side", "quantity", "price", "fee", "event_time"], []),
        "cash_flows": (["transfer_id", "amount", "event_time"], []),
        "closing_positions": (
            ["symbol", "quantity", "available_quantity"],
            [["000001", quantity, quantity]],
        ),
        "prices": (
            ["symbol", "venue", "price", "as_of"],
            [["000001", "SZSE", "10", "2026-09-29T15:00:00+08:00"]],
        ),
    }
    cfg = {
        **config(),
        "opening_at": "2026-09-28T15:00:00+08:00",
        "opening_cash": str(cash),
        "closing_cash": str(cash),
    }
    for field, (headers, data) in schemas.items():
        cfg[field] = f"{field}.csv"
        with (tmp_path / cfg[field]).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(data)
    path, decision, out = (
        tmp_path / "account.json",
        tmp_path / "decision.json",
        tmp_path / "review.json",
    )
    path.write_text(json.dumps(cfg))
    decision.write_text(json.dumps(card(weight)))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_portfolio.account_import",
            "--config",
            str(path),
            "--decision",
            str(decision),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out.read_text())


def test_unchanged_target_does_not_trade_or_charge_fees(tmp_path):
    result = statement_cli(tmp_path, 2500)
    assert result["status"] == "review_ready"
    assert result["proposed_trades"] == []


def test_odd_opening_position_buys_whole_increment_lots(tmp_path):
    result = statement_cli(tmp_path, 2350)
    assert result["status"] == "review_ready"
    trade = result["proposed_trades"][0]
    assert trade["side"] == "buy" and D(trade["quantity"]) == 100
    assert D(trade["estimated_fee"]) == 5


@pytest.mark.parametrize("quantity", [50, 2350])
def test_odd_lot_liquidation_requires_manual_review(tmp_path, quantity):
    result = statement_cli(tmp_path, quantity, "0")
    assert result["status"] == "observe" and not result["proposed_trades"]
    assert "odd_lot_liquidation_requires_manual_review:000001" in result["reasons"]


def test_sell_costs_apply_only_to_actual_delta():
    trades, reasons = propose(
        card(),
        config(),
        {"000001": D(4000)},
        {"000001": D(4000)},
        {"000001": D(10)},
        D(60000),
        D(100000),
    )
    assert reasons == []
    assert trades == [
        {
            "symbol": "000001",
            "side": "sell",
            "quantity": "1500",
            "estimated_price": "9.99",
            "estimated_fee": "12.492500",
        }
    ]


def test_zero_cost_buy_and_small_gap():
    cfg = config()
    cfg["costs"] = {key: "0" for key in ("commission", "min_commission", "stamp_tax", "slippage")}
    trades, reasons = propose(card(), cfg, {}, {}, {"000001": D(10)}, D(100000), D(100000))
    assert not reasons and D(trades[0]["quantity"]) == 2500
    trades, reasons = propose(
        card(), cfg, {"000001": D(2450)}, {}, {"000001": D(10)}, D(75500), D(100000)
    )
    assert trades == [] and reasons == []


def test_projected_concentration_checks_cost_reduced_nav():
    cfg = config()
    cfg["managed_symbols"] = ["000001", "000002"]
    decision = card("0.30")
    decision["targets"].append({"symbol": "000002", "weight": "0.1"})
    trades, reasons = propose(
        decision,
        cfg,
        {"000001": D(3000)},
        {},
        {"000001": D(10), "000002": D(10)},
        D(70000),
        D(100000),
    )
    assert trades == []
    assert "projected_concentration_limit:000001" in reasons
