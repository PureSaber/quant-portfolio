import csv
import json
from datetime import datetime, timezone

import pytest
import yaml

from quant_portfolio.account_import import propose, reconcile


@pytest.fixture
def statement(tmp_path):
    data = {
        "opening_positions": (
            ["symbol", "quantity", "average_cost", "acquired_on"],
            [["000001", "100", "10", "2026-09-16"]],
        ),
        "trades": (
            ["trade_id", "symbol", "side", "quantity", "price", "fee", "event_time"],
            [["t1", "000001", "sell", "100", "11", "5", "2026-09-18T10:00:00+08:00"]],
        ),
        "cash_flows": (
            ["transfer_id", "amount", "event_time"],
            [["d1", "500", "2026-09-18T09:00:00+08:00"]],
        ),
        "closing_positions": (["symbol", "quantity", "available_quantity"], []),
        "prices": (
            ["symbol", "venue", "price", "as_of"],
            [["000001", "SZSE", "11", "2026-09-18T15:00:00+08:00"]],
        ),
    }
    cfg = {
        "opening_at": "2026-09-17T15:00:00+08:00",
        "as_of": "2026-09-18T15:00:00+08:00",
        "opening_cash": "1000",
        "closing_cash": "2595",
        "managed_symbols": ["000001"],
    }
    for name, (header, values) in data.items():
        path = tmp_path / f"{name}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(values)
        cfg[name] = path.name
    config = tmp_path / "account.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return config, cfg


def test_statement_cash_positions_and_flow_reconcile_exactly(statement):
    path, _ = statement
    result = reconcile(path)
    assert result["status"] == "reconciled"
    assert result["cash"] == "2595.00000000"
    assert result["positions"] == [] and result["performance"] is None
    assert result["external_cash_flow"] == "500"
    assert any("external:" in row["reference_id"] for row in result["ledger"])
    json.dumps(result)


def test_mismatch_and_duplicates_cannot_generate_proposals(statement):
    path, cfg = statement
    cfg["closing_cash"] = "2600"
    path.write_text(yaml.safe_dump(cfg))
    result = reconcile(path)
    assert result["status"] == "blocked" and not result["proposed_trades"]
    trades = path.parent / cfg["trades"]
    content = trades.read_text()
    trades.write_text(content + content.splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        reconcile(path)


def test_malformed_number_replaces_old_action_with_blocked_card(statement, monkeypatch):
    from quant_portfolio.account_import import main

    path, cfg = statement
    cfg["opening_cash"] = "corrupt"
    path.write_text(yaml.safe_dump(cfg))
    output = path.parent / "account.json"
    output.write_text('{"status":"review_ready","proposed_trades":[{}]}')
    monkeypatch.setattr("sys.argv", ["account", "--config", str(path), "--output", str(output)])
    assert main() == 2
    card = json.loads(output.read_text())
    assert card["status"] == "blocked" and card["proposed_trades"] == []


def test_proposals_use_real_nav_available_shares_and_settled_cash():
    from decimal import Decimal as D

    card = {
        "schema_version": "quant.decision/v1",
        "status": "paper_ready",
        "as_of": "2026-09-18",
        "valid_until": "2026-09-21T09:30:00+08:00",
        "targets": [{"symbol": "000001", "weight": "0.25"}],
    }
    cfg = {"as_of": "2026-09-18T15:00:00+08:00", "managed_symbols": ["000001"]}
    now = datetime(2026, 9, 19, tzinfo=timezone.utc)
    trades, reasons = propose(card, cfg, {}, {}, {"000001": D(10)}, D(100000), D(100000), now=now)
    assert not reasons and trades[0]["quantity"] == "2400"
    assert propose(card, cfg, {}, {}, {"000001": D(10)}, D(100), D(100000), now=now)[1]
    assert propose(
        card,
        cfg,
        {"000001": D(4000)},
        {"000001": D(100)},
        {"000001": D(10)},
        D(60000),
        D(100000),
        now=now,
    )[1]
    assert (
        propose(
            card,
            cfg,
            {},
            {},
            {"000001": D(10)},
            D(100000),
            D(100000),
            now=datetime(2026, 9, 22, tzinfo=timezone.utc),
        )[0]
        == []
    )
