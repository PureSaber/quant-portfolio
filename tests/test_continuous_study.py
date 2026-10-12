import json
from copy import deepcopy
from decimal import Decimal

import pytest
from quant_data_kit import AssetClass, FixedPoint, InstrumentSpec
from quant_data_kit.artifacts import ArtifactStore, canonical_json
from quant_data_kit.domain_v2 import dataclass_payload
from quant_execution.continuous import utc

from quant_portfolio.cli import main
from quant_portfolio.continuous_study import freeze_plan, paired_study, run_study


def definition():
    specs = [
        InstrumentSpec(
            instrument_id=i,
            asset_class=AssetClass.ETF,
            product_type="cash_etf",
            venue="SYNTHETIC",
            native_symbol=i,
            settlement_currency="USD",
            price_tick=FixedPoint(1, 2),
            quantity_step=FixedPoint(1, 0),
            contract_multiplier=FixedPoint(1, 0),
            calendar_id="explicit_fixture",
            effective_from=utc("2026-01-01T00:00:00Z"),
            available_at=utc("2026-01-01T00:00:00Z"),
        )
        for i in ("A", "B")
    ]
    return {
        "schema": "quant.capital-plan/v1",
        "evidence_kind": "synthetic",
        "sample_kind": "deterministic_fixture",
        "instruments": [dataclass_payload(s) for s in specs],
        "owners": {"A": "alpha", "B": "beta"},
        "allocations": {"alpha": {"USD": "600"}, "beta": {"USD": "400"}},
        "base_currency": "USD",
        "opened_at": "2026-01-01T00:00:00Z",
        "opening_fx": {},
        "cost_policy": {"model": "explicit_fee_per_unit", "fee_per_unit": "1"},
        "valuation_policy": {
            "model": "last_available_mark_and_fx",
            "positive_futures_pnl_spendable": False,
        },
        "decisions": [
            {
                "decision_id": "buy-a",
                "signal_observation_id": "o1",
                "execution_observation_id": "o2",
                "instrument_id": "A",
                "target_notional": "500",
                "settles_at": "2026-01-03T00:00:00Z",
            },
            {
                "decision_id": "buy-b",
                "signal_observation_id": "o1",
                "execution_observation_id": "o2",
                "instrument_id": "B",
                "target_notional": "300",
                "settles_at": "2026-01-03T00:00:00Z",
            },
        ],
        "cash_events": [],
    }


def market():
    return {
        "schema": "quant.capital-market/v1",
        "observations": [
            {
                "observation_id": f"o{i}",
                "at": f"2026-01-0{i + 1}T00:00:00Z",
                "available_at": f"2026-01-0{i + 1}T00:00:00Z",
                "prices": {"A": p, "B": "50"},
                "fx": {},
            }
            for i, p in [(1, "100"), (2, "101"), (3, "110")]
        ],
    }


def test_actual_policy_execution_and_independent_expected_result():
    result = run_study(freeze_plan(definition()), market())
    # A: floor(500/100)=5, execution 101, fee 5, final mark 110 => +40.
    # B: floor(300/50)=6, execution 50, fee 6, final mark 50 => -6.
    assert Decimal(result["net_pnl"]) == 34
    assert Decimal(result["cumulative_return"]) == pytest.approx(Decimal("0.034"))
    assert [o["quantity"] for o in result["orders"]] == ["5", "6"]
    assert result["measurement"]["schema"] == "quant.measurement/v1"
    assert result["reconciliation"]["ledger_balanced"]


def test_revision_pair_recomputes_orders_and_cash_preserves_old(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    old_data, new_data = market(), market()
    new_data["observations"][0]["prices"]["A"] = "125"
    old, new = [store.put_bytes(canonical_json(data)) for data in (old_data, new_data)]
    plan = freeze_plan(definition())
    pair = paired_study(store, plan, old, new, owner="test")
    assert pair["intervention"] == "data_revision"
    assert pair["economic_changes"]["signals"]
    assert pair["economic_changes"]["orders"]
    assert pair["economic_changes"]["net_pnl"][0]["delta"] == "-8.00000000"
    assert pair["unavailable_stages"] == ["features", "forecasts"]
    assert store.read_bytes(old) == canonical_json(old_data)
    same = paired_study(store, plan, old, old, owner="same")
    assert not same["changes"] and not same["affected_objects"]
    with pytest.raises(FileExistsError):
        request = tmp_path / "request.json"
        request.write_text("{}", encoding="utf-8")
        # Output preservation is checked through a real successful continuous command below.
        paths = [tmp_path / name for name in ("plan.json", "market.json")]
        for path, doc in zip(paths, (plan, old_data)):
            path.write_text(json.dumps(doc), encoding="utf-8")
        main(
            [
                "continuous-study",
                "--plan",
                str(paths[0]),
                "--market",
                str(paths[1]),
                "--out",
                str(request),
            ]
        )


def test_freeze_run_and_revision_cli(tmp_path):
    draft, frozen, market_path, output = [
        tmp_path / p for p in ("draft.json", "frozen.json", "market.json", "run.json")
    ]
    draft.write_text(json.dumps(definition()), encoding="utf-8")
    market_path.write_text(json.dumps(market()), encoding="utf-8")
    assert main(["freeze-capital-plan", "--plan", str(draft), "--out", str(frozen)]) == 0
    assert (
        main(
            [
                "continuous-study",
                "--plan",
                str(frozen),
                "--market",
                str(market_path),
                "--out",
                str(output),
            ]
        )
        == 0
    )
    assert Decimal(json.loads(output.read_bytes())["net_pnl"]) == 34
    store = ArtifactStore(tmp_path / "store")
    old = store.put_bytes(canonical_json(market()))
    revised = market()
    revised["observations"][-1]["prices"]["A"] = "111"
    new = store.put_bytes(canonical_json(revised))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "store": str(store.root),
                "plan": str(frozen),
                "old_input": old.to_dict(),
                "new_input": new.to_dict(),
                "owner": "cli",
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(["revision-study", "--request", str(request), "--out", str(tmp_path / "pair.json")])
        == 0
    )


def test_changed_code_controls_future_prices_and_leakage_fail(monkeypatch):
    plan = freeze_plan(definition())
    changed = deepcopy(plan)
    changed["cost_policy"]["fee_per_unit"] = "2"
    with pytest.raises(ValueError, match="controls"):
        run_study(changed, market())
    future = market()
    future["observations"][0]["available_at"] = "2027-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="unavailable"):
        run_study(plan, future)
    invalid = definition()
    invalid["decisions"][0]["signal_observation_id"] = "o2"
    with pytest.raises(ValueError, match="strictly precede"):
        run_study(freeze_plan(invalid), market())
    from quant_portfolio import continuous_study

    monkeypatch.setattr(continuous_study, "code_identity", lambda: {"producer": "b" * 64})
    with pytest.raises(ValueError, match="code/runtime"):
        run_study(plan, market())
