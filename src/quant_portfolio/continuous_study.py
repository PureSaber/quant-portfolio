"""Frozen continuous capital and old/new revision studies; offline research only."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
from copy import deepcopy
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, localcontext
from pathlib import Path

from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec, MarginMode
from quant_data_kit.artifacts import ArtifactRef, ArtifactStore, canonical_json
from quant_data_kit.revision_impact import RevisionRegistry, json_changes, sha256
from quant_execution.continuous import SharedCapitalAccount, number, utc


def code_identity():
    """Hash actual installed source bytes and runtime versions, including dirty edits."""
    import hashlib

    identity = {}
    for name in ("quant_data_kit", "quant_execution", "quant_portfolio"):
        root = Path(importlib.import_module(name).__file__).parent
        files = {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*.py"))
        }
        if not files:
            raise ValueError("source identity unavailable")
        identity[name] = sha256(files)
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "pandas",
            "pyarrow",
            "jsonschema",
            "quant-data-kit",
            "quant-execution",
            "quant-portfolio",
        )
    }
    identity["runtime"] = sha256({"python": platform.python_version(), "packages": versions})
    return identity


def freeze_plan(definition):
    definition = json.loads(canonical_json(definition))
    if definition.get("schema") != "quant.capital-plan/v1":
        raise ValueError("capital plan v1 required")
    if any(k in definition for k in ("code_identity", "frozen_at", "controls_sha256")):
        raise ValueError("already frozen; cannot silently refreeze historical controls")
    return {
        **definition,
        "code_identity": code_identity(),
        "controls_sha256": sha256(definition),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }


def _validate_plan(plan):
    definition = {
        k: v for k, v in plan.items() if k not in {"code_identity", "controls_sha256", "frozen_at"}
    }
    if plan.get("schema") != "quant.capital-plan/v1" or sha256(definition) != plan.get(
        "controls_sha256"
    ):
        raise ValueError("frozen controls changed or missing")
    if plan.get("code_identity") != code_identity():
        raise ValueError("frozen executable code/runtime changed")
    utc(plan["frozen_at"])
    if plan["evidence_kind"] not in {"synthetic", "retrospective"}:
        raise ValueError("this adapter only supports synthetic or retrospective evidence")
    if plan["sample_kind"] not in {"deterministic_fixture", "in_sample"}:
        raise ValueError("this fixed-policy adapter does not certify OOS predictions")
    if number(plan["cost_policy"]["fee_per_unit"]) < 0:
        raise ValueError("nonnegative explicit fee required")
    if plan["cost_policy"].get("model") != "explicit_fee_per_unit":
        raise ValueError("unsupported cost policy")
    if plan["valuation_policy"] != {
        "model": "last_available_mark_and_fx",
        "positive_futures_pnl_spendable": False,
    }:
        raise ValueError("unsupported valuation policy")
    return definition


def _account(plan):
    specs = []
    for raw in plan["instruments"]:
        item = deepcopy(raw)
        item["asset_class"] = AssetClass(item["asset_class"])
        item["margin_mode"] = MarginMode(item.get("margin_mode", "none"))
        for key in ("price_tick", "quantity_step", "contract_multiplier"):
            item[key] = FixedPoint(**item[key])
        for key in ("effective_from", "available_at", "effective_to", "superseded_at"):
            if item.get(key) is not None:
                item[key] = utc(item[key])
        if item.get("expiry_date"):
            from datetime import date

            item["expiry_date"] = date.fromisoformat(item["expiry_date"])
        specs.append(InstrumentSpec(**item))
    return SharedCapitalAccount(
        specs=specs,
        owners=plan["owners"],
        allocations=plan["allocations"],
        base_currency=plan["base_currency"],
        opened_at=plan["opened_at"],
        fx_rates=plan["opening_fx"],
    )


def _schedule(plan, market):
    if market.get("schema") != "quant.capital-market/v1":
        raise ValueError("capital market v1 required")
    observations = {}
    schedule = []
    for row in market["observations"]:
        name, at = row["observation_id"], utc(row["at"])
        if name in observations or utc(row["available_at"]) > at:
            raise ValueError("duplicate observation or unavailable market data")
        observations[name] = row
        sequence = 100
        for currency, rate in sorted(row["fx"].items()):
            schedule.append(
                {
                    "event_id": f"{name}:fx:{currency}",
                    "at": row["at"],
                    "available_at": row["available_at"],
                    "sequence": sequence,
                    "kind": "fx_mark",
                    "currency": currency,
                    "rate": rate,
                }
            )
            sequence += 1
        for instrument, price in sorted(row["prices"].items()):
            schedule.append(
                {
                    "event_id": f"{name}:mark:{instrument}",
                    "at": row["at"],
                    "available_at": row["available_at"],
                    "sequence": sequence,
                    "kind": "mark",
                    "instrument_id": instrument,
                    "price": price,
                }
            )
            sequence += 1
        if sequence >= 1000:
            raise ValueError("at most 899 marks per observation")
    decision_ids = set()
    for n, decision in enumerate(plan["decisions"]):
        if decision["decision_id"] in decision_ids:
            raise ValueError("duplicate decision ID")
        decision_ids.add(decision["decision_id"])
        signal = observations[decision["signal_observation_id"]]
        execution = observations[decision["execution_observation_id"]]
        if utc(signal["at"]) >= utc(execution["at"]):
            raise ValueError("signal observation must strictly precede execution")
        schedule.append(
            {
                "event_id": "decision:" + decision["decision_id"],
                "kind": "decision",
                "at": execution["at"],
                "sequence": 1000 + n,
                "decision": decision,
            }
        )
    for event in plan.get("cash_events", []):
        if not 0 <= event["sequence"] < 100:
            raise ValueError("cash event sequences must be in [0,100)")
        if event["kind"] not in {
            "transfer",
            "external_cash",
            "fx_convert",
            "settle",
            "charge",
            "variation_margin",
        }:
            raise ValueError("unsupported scheduled cash event")
        schedule.append(deepcopy(event))
    ordered = sorted(schedule, key=lambda e: (utc(e["at"]), e["sequence"]))
    keys = [(utc(e["at"]), e["sequence"]) for e in ordered]
    ids = [e["event_id"] for e in ordered]
    if len(keys) != len(set(keys)) or len(ids) != len(set(ids)):
        raise ValueError("ambiguous event schedule or duplicate ID")
    if not ordered:
        raise ValueError("nonempty observation schedule required")
    return observations, ordered


def run_study(plan, market):
    with localcontext() as context:
        context.prec = 64
        return _run_study(plan, market)


def _run_study(plan, market):
    _validate_plan(plan)
    plan, market = json.loads(canonical_json(plan)), json.loads(canonical_json(market))
    account = _account(plan)
    observations, ordered = _schedule(plan, market)
    orders, signals = [], []
    for event in ordered:
        if event["kind"] != "decision":
            account.apply(event)
            continue
        decision = event["decision"]
        instrument = decision["instrument_id"]
        spec = account.specs[instrument]
        signal = observations[decision["signal_observation_id"]]
        execution = observations[decision["execution_observation_id"]]
        price = number(signal["prices"][instrument])
        if price <= 0:
            raise ValueError("positive signal price required")
        notional = number(decision["target_notional"])
        step = spec.quantity_step.to_decimal()
        target = (
            notional / (price * spec.contract_multiplier.to_decimal()) / step
        ).to_integral_value(rounding=ROUND_DOWN) * step
        held = (
            account.ledger.snapshot(account.last_key[0])
            .positions.get(instrument, FixedPoint(0, 0))
            .to_decimal()
        )
        quantity = target - held
        signals.append(
            {
                "decision_id": decision["decision_id"],
                "at": signal["at"],
                "instrument_id": instrument,
                "target_notional": str(notional),
                "target_quantity": str(target),
                "policy": "frozen_notional_to_lots",
            }
        )
        order = {
            "event_id": event["event_id"],
            "order_id": decision["decision_id"],
            "at": event["at"],
            "sequence": event["sequence"],
            "available_at": execution["available_at"],
            "kind": "fill",
            "sleeve": account.owners[instrument],
            "instrument_id": instrument,
            "quantity": str(quantity),
            "price": execution["prices"][instrument],
            "fee": str(abs(quantity) * number(plan["cost_policy"]["fee_per_unit"])),
            "settles_at": decision["settles_at"],
        }
        orders.append({**order, "status": "no_change" if quantity == 0 else "declared_fill"})
        if quantity:
            account.apply(order)
        else:
            account.apply(
                {
                    k: v
                    for k, v in {**order, "kind": "settle"}.items()
                    if k in {"event_id", "at", "sequence", "available_at", "kind"}
                }
            )
    result = account.report()
    if code_identity() != plan["code_identity"]:
        raise ValueError("code changed during evaluation")
    # Same-instant asynchronous events retain their sequence in history. Publish
    # one observation after all events at that instant for the common measurement
    # grid; its return compounds those event returns (including flow timing).
    opening = utc(plan["opened_at"]).isoformat().replace("+00:00", "Z")
    grouped = {}
    for row in result["history"]:
        at = utc(row["at"]).isoformat().replace("+00:00", "Z")
        if utc(at) <= utc(opening):
            raise ValueError("study observations must follow the opening boundary")
        prior = grouped.setdefault(
            at, {"at": at, "growth": Decimal(1), "external_flow": Decimal(0)}
        )
        if prior["growth"] is not None:
            prior["growth"] = (
                None
                if row["net_return"] is None
                else prior["growth"] * (1 + number(row["net_return"]))
            )
        prior["external_flow"] += number(row["external_flow"])
        prior["nav"] = row["nav"]
    observations = [
        {
            "at": at,
            "nav": row["nav"],
            "external_flow": str(row["external_flow"]),
            "net_return": None if row["growth"] is None else str(row["growth"] - 1),
        }
        for at, row in grouped.items()
    ]
    times = [opening] + [row["at"] for row in observations]
    inputs = {
        "market": "sha256:" + sha256(market),
        "opening_fx": "sha256:" + sha256(plan["opening_fx"]),
        "instruments": "sha256:" + sha256(plan["instruments"]),
    }
    actual_cost_policy = {
        **plan["cost_policy"],
        "scheduled_charges": [e for e in plan.get("cash_events", []) if e["kind"] == "charge"],
    }
    basis = {
        "currency": plan["base_currency"],
        "monetary_unit": "major_currency",
        "return_unit": "decimal",
        "return_basis": "net_simple_time_weighted",
        "period_start": times[0],
        "period_end": times[-1],
        "frequency": "event",
        "calendar": "explicit_research_events",
        "timezone": "UTC",
        "annualization_periods": None,
        "annualization_policy": "not_annualized",
        "risk_free_rate": "0",
        "risk_free_compounding": "simple",
        "risk_free_unit": "annual_decimal",
        "flow_policy": "external_flows_excluded_internal_transfers_neutral",
        "cost_policy": "actual_ledger_fees_and_charges",
        "cost_policy_id": "sha256:" + sha256(actual_cost_policy),
        "valuation_policy": "event_close_mark",
        "valuation_policy_id": "sha256:" + sha256(plan["valuation_policy"]),
        "benchmark_id": "none",
        "universe_id": "sha256:" + sha256(plan["instruments"]),
        "timing_policy": "prior_observation_signal_next_observation_declared_execution",
        "denominator_policy": "sequential_event_nav_compounded_within_timestamp",
        "observation_schedule_id": "sha256:" + sha256(times),
        "capital_basis": "net_assets_including_cash_and_receivables_less_liabilities",
        "initial_capital": result["initial_nav"],
    }
    return {
        **result,
        "schema": "quant.continuous-study/v1",
        "evidence_kind": plan["evidence_kind"],
        "sample_kind": plan["sample_kind"],
        "code_identity": plan["code_identity"],
        "input_identity": inputs,
        "observations": observations,
        "observation_timestamps": times,
        "measurement_configs": {
            "cost_policy": actual_cost_policy,
            "valuation_policy": plan["valuation_policy"],
        },
        "plan_sha256": sha256(plan),
        "controls_sha256": plan["controls_sha256"],
        "signals": signals,
        "orders": orders,
        "unavailable_stages": ["features", "forecasts"],
        "measurement": {
            "schema": "quant.measurement/v1",
            "evidence_kind": plan["evidence_kind"],
            "sample_kind": plan["sample_kind"],
            "basis": basis,
            "provenance": {
                "code_identity": plan["code_identity"],
                "input_identity": inputs,
            },
            "unsupported_scopes": result["unsupported_scopes"],
        },
    }


def paired_study(store, plan, old_input, new_input, *, owner):
    """Execute the identical frozen plan twice against verified immutable bytes."""
    _validate_plan(plan)
    registry = RevisionRegistry(store)
    plan_ref = store.put_bytes(canonical_json(plan), owner=owner)
    outputs, derivations = [], []
    for ref in (old_input, new_input):
        market = json.loads(store.read_bytes(ref))
        result = run_study(plan, market)
        output = store.put_bytes(canonical_json(result), owner=owner)
        descriptor = registry.register(
            output=output,
            inputs={"market": ref, "frozen_plan": plan_ref},
            stage="continuous_study",
            code_identity=plan["code_identity"],
            controls={"frozen_plan_sha256": plan_ref.sha256},
            owner=owner,
        )
        outputs.append(result)
        derivations.append(descriptor)
    pair = registry.paired(*derivations, old_input=old_input, new_input=new_input)
    stages = ("signals", "orders", "history", "net_pnl", "cumulative_return", "final_account")
    report = {
        **pair,
        "schema": "quant.revision-study/v1",
        "frozen_plan": plan_ref.to_dict(),
        "evidence_kind": plan["evidence_kind"],
        "sample_kind": plan["sample_kind"],
        "interpretation": "retrospective sensitivity under frozen controls; no historical decision replacement",
        "affected_objects": registry.impact(old_input, new_input)["affected_objects"],
        "unavailable_stages": ["features", "forecasts"],
        "economic_changes": {s: json_changes(outputs[0][s], outputs[1][s]) for s in stages},
    }
    report_ref = store.put_bytes(canonical_json(report), owner=owner)
    registry.register(
        output=report_ref,
        inputs={"old": derivations[0], "new": derivations[1]},
        stage="revision_report",
        code_identity=plan["code_identity"],
        controls={"paired": True},
        owner=owner,
    )
    return {**report, "report_artifact": report_ref.to_dict()}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def command(args):
    if args.command == "freeze-capital-plan":
        result = freeze_plan(_read(args.plan))
    elif args.command == "continuous-study":
        result = run_study(_read(args.plan), _read(args.market))
    else:
        request = _read(args.request)
        store = ArtifactStore(Path(request["store"]))
        result = paired_study(
            store,
            _read(request["plan"]),
            ArtifactRef.from_dict(request["old_input"]),
            ArtifactRef.from_dict(request["new_input"]),
            owner=request["owner"],
        )
    # Never overwrite a prior plan, run, or revision decision.
    with Path(args.out).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return 0


def add_parsers(sub):
    for name in ("freeze-capital-plan", "continuous-study", "revision-study"):
        parser = sub.add_parser(name, help=__doc__)
        if name == "revision-study":
            parser.add_argument("--request", required=True)
        else:
            parser.add_argument("--plan", required=True)
        if name == "continuous-study":
            parser.add_argument("--market", required=True)
        parser.add_argument("--out", required=True)
        parser.set_defaults(func=command)


def main(argv=None):
    parser = argparse.ArgumentParser()
    add_parsers(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return command(args)


if __name__ == "__main__":
    raise SystemExit(main())
