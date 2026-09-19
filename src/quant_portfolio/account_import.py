"""Local statement reconciliation and account-specific, review-only trade deltas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from pathlib import Path

import yaml
from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec, MarkPriceEvent
from quant_data_kit.exceptions import ValidationError
from quant_execution import ExactAccountLedger, Fee, Fill, Side, execution_payload


def number(value) -> Decimal:
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid numeric statement value") from exc
    if not result.is_finite():
        raise ValueError("Non-finite statement value")
    return result


def fp(value, scale=8):
    scaled = number(value) * 10**scale
    if scaled != scaled.to_integral_value():
        raise ValueError("Statement value exceeds precision")
    return FixedPoint(int(scaled), scale)


def instant(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Statement timestamps require a timezone")
    return result.astimezone(timezone.utc)


def rows(path: Path, required: set[str], identity: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path.name}: {sorted(required)}")
        result = list(reader)
    if any(not r[identity] for r in result) or len({r[identity] for r in result}) != len(result):
        raise ValueError(f"Duplicate/empty {identity} in {path.name}")
    return result


def reconcile(config_path: Path, decision_path: Path | None = None, *, now=None) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent
    opening, closing = instant(cfg["opening_at"]), instant(cfg["as_of"])
    if opening > closing or closing > (now or datetime.now(timezone.utc)):
        raise ValueError("Invalid statement observation period")
    files = {
        name: (root / cfg[name]).resolve()
        for name in ("opening_positions", "trades", "cash_flows", "closing_positions", "prices")
    }
    positions = rows(
        files["opening_positions"], {"symbol", "quantity", "average_cost", "acquired_on"}, "symbol"
    )
    trades = rows(
        files["trades"],
        {"trade_id", "symbol", "side", "quantity", "price", "fee", "event_time"},
        "trade_id",
    )
    flows = rows(files["cash_flows"], {"transfer_id", "amount", "event_time"}, "transfer_id")
    statement = rows(
        files["closing_positions"], {"symbol", "quantity", "available_quantity"}, "symbol"
    )
    prices = rows(files["prices"], {"symbol", "venue", "price", "as_of"}, "symbol")
    instruments, marks = {}, {}
    for row in prices:
        symbol = row["symbol"]
        if len(symbol) != 6 or not symbol.isdigit() or row["venue"] not in {"SSE", "SZSE"}:
            raise ValueError("Explicit six-digit A-share symbol and venue required")
        if instant(row["as_of"]) != closing or number(row["price"]) <= 0:
            raise ValueError("Prices must be positive and at the statement cutoff")
        instruments[symbol] = InstrumentSpec(
            instrument_id=symbol,
            asset_class=AssetClass.EQUITY,
            product_type="stock",
            venue=row["venue"],
            native_symbol=symbol,
            settlement_currency="CNY",
            price_tick=fp("0.01", 2),
            quantity_step=fp(1, 0),
            contract_multiplier=fp(1, 0),
            calendar_id="CN-A-SHARE",
            effective_from=opening,
            available_at=opening,
            metadata={"lot_size": "100", "catalog_scope": "user-supplied statement"},
        )
        marks[symbol] = number(row["price"])
    if number(cfg["opening_cash"]) < 0:
        raise ValueError("Opening cash cannot be negative")
    ledger = ExactAccountLedger(
        account_id="imported-account",
        base_currency="CNY",
        instruments=instruments,
        initial_cash={"CNY": fp(cfg["opening_cash"])},
        opened_at=opening,
    )
    for row in positions:
        ledger.book_opening_position(
            instrument_id=row["symbol"],
            quantity=fp(row["quantity"], 0),
            average_cost=fp(row["average_cost"]),
            acquired_on=datetime.fromisoformat(row["acquired_on"]).date(),
        )
    events = [(instant(r["event_time"]), "flow", r) for r in flows] + [
        (instant(r["event_time"]), "trade", r) for r in trades
    ]
    for at, kind, row in sorted(
        events, key=lambda x: (x[0], x[1], x[2].get("transfer_id", x[2].get("trade_id")))
    ):
        if not opening < at <= closing:
            raise ValueError("Cash/trade event outside statement interval")
        if kind == "flow":
            ledger.book_external_cash(
                transfer_id=row["transfer_id"],
                amount=fp(row["amount"]),
                currency="CNY",
                event_time=at,
            )
        else:
            if number(row["fee"]) < 0:
                raise ValueError("Trade fee cannot be negative")
            symbol = row["symbol"]
            mark = MarkPriceEvent(
                event_id="trade-mark:" + row["trade_id"],
                instrument_id=symbol,
                event_time=at,
                available_at=at,
                received_at=at,
                source="user_statement",
                trading_day=at.date(),
                session_id="statement",
                sequence=0,
                price=fp(row["price"]),
            )
            ledger.mark(mark)
            ledger.apply(
                Fill(
                    fill_id=row["trade_id"],
                    order_id=row["trade_id"],
                    account_id="imported-account",
                    strategy_id="confirmed-statement",
                    instrument_id=symbol,
                    side=Side(row["side"]),
                    quantity=fp(row["quantity"], 0),
                    price=fp(row["price"]),
                    event_time=at,
                )
            )
            ledger.apply(
                Fee(
                    fee_id="fee:" + row["trade_id"],
                    fill_id=row["trade_id"],
                    account_id="imported-account",
                    amount=fp(row["fee"]),
                    currency="CNY",
                    event_time=at,
                    fee_type="confirmed_total",
                )
            )
    for symbol, price in marks.items():
        ledger.mark(
            MarkPriceEvent(
                event_id="closing-mark:" + symbol,
                instrument_id=symbol,
                event_time=closing,
                available_at=closing,
                received_at=closing,
                source="user_statement",
                trading_day=closing.date(),
                session_id="statement",
                sequence=0,
                price=fp(price),
            )
        )
    snapshot = ledger.snapshot(closing)
    quantities = {symbol: q.to_decimal() for symbol, q in snapshot.positions.items() if q.units}
    reported = {
        row["symbol"]: number(row["quantity"]) for row in statement if number(row["quantity"])
    }
    available = {row["symbol"]: number(row["available_quantity"]) for row in statement}
    if any(
        number(row["quantity"]) < 0
        or available[row["symbol"]] < 0
        or available[row["symbol"]] > number(row["quantity"])
        or number(row["quantity"]) != number(row["quantity"]).to_integral_value()
        or available[row["symbol"]] != available[row["symbol"]].to_integral_value()
        for row in statement
    ):
        raise ValueError("Invalid statement position/available quantity")
    cash = snapshot.cash_balances["CNY"].to_decimal()
    differences = []
    if quantities != reported:
        differences.append("positions_do_not_reconcile")
    if cash != number(cfg["closing_cash"]):
        differences.append("cash_does_not_reconcile")
    result = {
        "schema_version": "quant.account-review/v1",
        "as_of": cfg["as_of"],
        "scope": "user_statement_review_only",
        "status": "blocked" if differences else "reconciled",
        "reasons": differences,
        "cash": str(cash),
        "nav": str(snapshot.nav.to_decimal()),
        "positions": [
            {
                "symbol": s,
                "quantity": str(q),
                "available_quantity": str(available.get(s, 0)),
                "price": str(marks[s]),
            }
            for s, q in sorted(quantities.items())
        ],
        "external_cash_flow": str(sum((number(row["amount"]) for row in flows), Decimal(0))),
        "proposed_trades": [],
        "performance": None,
        "evidence": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()
        },
        "ledger": [execution_payload(t) for t in ledger.transactions],
    }
    if decision_path and not differences:
        card = json.loads(decision_path.read_text(encoding="utf-8"))
        result["decision_run_id"] = card["run_id"]
        result["decision_sha256"] = hashlib.sha256(decision_path.read_bytes()).hexdigest()
        proposals, reasons = propose(
            card, cfg, quantities, available, marks, cash, snapshot.nav.to_decimal(), now=now
        )
        result.update(
            proposed_trades=proposals,
            reasons=reasons,
            status="observe" if reasons else "review_ready",
        )
    return result


def propose(card, cfg, quantities, available, prices, cash, nav, *, now=None):
    now = now or datetime.now(timezone.utc)
    if (
        card.get("schema_version") != "quant.decision/v1"
        or card.get("status") != "paper_ready"
        or not card.get("valid_until")
        or instant(card["valid_until"]) <= now
        or card.get("as_of") != str(cfg["as_of"])[:10]
    ):
        return [], ["decision_unavailable_expired_or_different_cutoff"]
    managed = set(cfg["managed_symbols"])
    target_weights = {r["symbol"]: number(r["weight"]) for r in card["targets"]}
    if len(target_weights) != len(card["targets"]) or not set(target_weights).issubset(managed):
        raise ValueError("Decision targets outside explicitly managed universe")
    if (
        nav <= 0
        or any(w < 0 or w > 1 for w in target_weights.values())
        or sum(target_weights.values()) > 1
    ):
        raise ValueError("Invalid target weights or account NAV")
    limits = cfg.get("limits", {})
    max_weight = number(limits.get("max_single_weight", "0.30"))
    reserve = number(limits.get("cash_buffer", "0.05"))
    if not 0 < max_weight <= 1 or not 0 <= reserve < 1:
        raise ValueError("Invalid account risk limits")
    fees = cfg.get("costs", {})
    commission, minimum, tax, slip = [
        number(fees.get(k, d))
        for k, d in (
            ("commission", "0.0003"),
            ("min_commission", "5"),
            ("stamp_tax", "0.0005"),
            ("slippage", "0.001"),
        )
    ]
    if min(commission, minimum, tax, slip) < 0 or slip >= 1:
        raise ValueError("Invalid cost parameters")
    trades, reasons = [], []
    for symbol, quantity in quantities.items():
        if symbol not in managed and quantity * prices[symbol] > nav * max_weight:
            reasons.append("unmanaged_concentration_limit:" + symbol)
    buy_cash = Decimal(0)
    for symbol in sorted(managed):
        if symbol not in prices:
            raise ValueError("Missing current price for managed instrument")
        price = prices[symbol]
        weight = target_weights.get(symbol, Decimal(0))
        if weight > max_weight:
            reasons.append("target_concentration_limit:" + symbol)
        target = (
            int((nav * weight - minimum) / (price * (1 + slip) * (1 + commission)) // 100) * 100
            if weight
            else 0
        )
        target = max(0, target)
        delta = Decimal(target) - quantities.get(symbol, 0)
        if not delta:
            continue
        if delta < 0 and -delta > available.get(symbol, 0):
            reasons.append("insufficient_available_shares:" + symbol)
        side = "buy" if delta > 0 else "sell"
        execution_price = (
            price * (1 + (slip if delta > 0 else -slip)) / Decimal("0.01")
        ).to_integral_value(rounding=ROUND_CEILING if delta > 0 else ROUND_FLOOR) * Decimal("0.01")
        notional = abs(delta) * execution_price
        cost = max(minimum, notional * commission) + (notional * tax if delta < 0 else 0)
        if delta > 0:
            buy_cash += notional + cost
        trades.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": str(abs(delta)),
                "estimated_price": str(execution_price),
                "estimated_fee": str(cost),
            }
        )
    # Do not spend proceeds from sells that have not actually been confirmed.
    if buy_cash > max(Decimal(0), cash - nav * reserve):
        reasons.append("insufficient_settled_cash_reconcile_sells_first")
    return ([] if reasons else trades), reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = reconcile(args.config, args.decision)
    except (ValueError, KeyError, OSError, TypeError, ValidationError) as exc:
        result = {
            "schema_version": "quant.account-review/v1",
            "status": "blocked",
            "reasons": [str(exc)],
            "proposed_trades": [],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(args.output)
    return 2 if result["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
