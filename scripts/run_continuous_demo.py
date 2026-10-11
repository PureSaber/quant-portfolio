"""Actual offline CLI acceptance with independent scalar expected values."""

import argparse
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

from quant_data_kit.artifacts import ArtifactStore, canonical_json


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    fixture = Path(__file__).resolve().parents[1] / "examples/continuous"
    commands = [
        [
            "freeze-capital-plan",
            "--plan",
            str(fixture / "plan.json"),
            "--out",
            str(out / "plan.json"),
        ],
        [
            "continuous-study",
            "--plan",
            str(out / "plan.json"),
            "--market",
            str(fixture / "market-old.json"),
            "--out",
            str(out / "old.json"),
        ],
        [
            "continuous-study",
            "--plan",
            str(out / "plan.json"),
            "--market",
            str(fixture / "market-new.json"),
            "--out",
            str(out / "new.json"),
        ],
    ]
    store = ArtifactStore(out / "store")
    refs = [
        store.put_bytes(
            canonical_json(json.loads((fixture / name).read_bytes())), owner="synthetic-demo"
        )
        for name in ("market-old.json", "market-new.json")
    ]
    request = {
        "store": str(store.root.resolve()),
        "plan": str((out / "plan.json").resolve()),
        "old_input": refs[0].to_dict(),
        "new_input": refs[1].to_dict(),
        "owner": "synthetic-demo",
    }
    (out / "request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
    for name, ref in zip(("old-ref.json", "new-ref.json"), refs):
        (out / name).write_text(json.dumps(ref.to_dict(), indent=2), encoding="utf-8")
    commands.append(
        [
            "revision-study",
            "--request",
            str(out / "request.json"),
            "--out",
            str(out / "revision.json"),
        ]
    )
    evidence = []
    for args in commands:
        command = [sys.executable, "-m", "quant_portfolio.cli", *args]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        evidence.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    old, new = [json.loads((out / name).read_bytes()) for name in ("old.json", "new.json")]
    # Independent arithmetic, deliberately not calling ledger, policy or metric helpers:
    # old A: 5*(110-101)-5=40; revised A: 4*(110-101)-4=32;
    # both B: 6*(50-50)-6=-6; opening capital=1000.
    expected_old = Decimal(5) * (Decimal(110) - Decimal(101)) - Decimal(5) - Decimal(6)
    expected_new = Decimal(4) * (Decimal(110) - Decimal(101)) - Decimal(4) - Decimal(6)
    assert Decimal(old["net_pnl"]) == expected_old == 34
    assert Decimal(new["net_pnl"]) == expected_new == 26
    assert Decimal(old["final_account"]["nav"]["units"]) / 10**8 == 1034
    assert Decimal(new["final_account"]["nav"]["units"]) / 10**8 == 1026
    pair = json.loads((out / "revision.json").read_bytes())
    assert pair["economic_changes"]["orders"]
    assert Decimal(pair["economic_changes"]["net_pnl"][0]["delta"]) == -8
    summary = {
        "evidence_kind": "synthetic",
        "independent_expected_pnl": {"old": "34", "new": "26", "delta": "-8"},
        "actual_pnl": {"old": old["net_pnl"], "new": new["net_pnl"]},
        "code_identity": old["code_identity"],
        "commands": evidence,
        "claims": [
            "CLI/API plumbing",
            "independent scalar cash/PnL",
            "frozen data revision sensitivity",
        ],
        "does_not_establish": ["market validity", "forecast skill", "live readiness"],
    }
    (out / "acceptance.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "README.md").write_text(
        "# Synthetic continuous-capital acceptance\n\n"
        "Old NAV 1034 USD / PnL +34; revised NAV 1026 USD / PnL +26; delta -8.\n\n"
        "The changed first signal price reduces A's target from 5 to 4 units. "
        "Later execution and terminal prices are held fixed. Both runs charge explicit fees. "
        "Inputs, frozen plan and both complete outputs remain pinned in the artifact store.\n\n"
        "This validates deterministic accounting and revision plumbing, not real-market performance. "
        "See acceptance.json for exact commands and independent scalar expectations, revision.json "
        "for affected objects and stage changes, and old.json/new.json for all ledger postings.\n",
        encoding="utf-8",
    )
    print(json.dumps({"old_nav": "1034", "new_nav": "1026", "delta_pnl": "-8", "out": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
