import copy
import json

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.methods import equal_risk_contribution_weights
from quant_portfolio.portfolio_study import (
    aligned_returns,
    diversification,
    evaluate_study,
    main,
    normalize_observations,
)


def study():
    dates = pd.date_range("2025-01-01", periods=9, tz="UTC")
    returns = {
        "equity": [0.01, -0.02, 0.03, 0.01, -0.01, 0.04, -0.02, 0.01],
        "bond": [0.002, 0.003, -0.002, 0.005, 0.001, -0.003, 0.004, 0.002],
    }
    spec = {
        "schema": "quant-portfolio.sleeve-study/v1",
        "data_nature": "synthetic",
        "base_currency": "USD",
        "evaluation_as_of": dates[-1].isoformat(),
        "sleeves": [
            {
                "id": name,
                "currency": "USD",
                "source_ref": "synthetic:test",
                "return_basis": "net_simple_total_return",
            }
            for name in returns
        ],
        "observations": [],
        "evaluation_periods": [],
        "allocation": {
            "mode": "risk_parity",
            "lookback": 4,
            "min_observations": 3,
            "max_turnover": 1.0,
        },
        "max_weight": 0.65,
        "invested_limit": 0.9,
        "initial_capital": 1000,
        "initial_weights": {"equity": 0.45, "bond": 0.45},
        "linear_costs": {"equity": 0.001, "bond": 0.002},
    }
    for name, values in returns.items():
        for i, value in enumerate(values):
            spec["observations"].append(
                {
                    "sleeve": name,
                    "start": dates[i].isoformat(),
                    "end": dates[i + 1].isoformat(),
                    "known_at": dates[i + 1].isoformat(),
                    "net_return": value,
                    "fx_start": 1.0,
                    "fx_end": 1.0,
                    "fx_start_known_at": dates[i].isoformat(),
                    "fx_end_known_at": dates[i + 1].isoformat(),
                }
            )
    for i in range(4, 8):
        spec["evaluation_periods"].append(
            {
                "start": dates[i].isoformat(),
                "end": dates[i + 1].isoformat(),
                "cash_return": 0.001,
            }
        )
    return spec


def test_common_constraints_cash_cost_and_drift_independent_replay():
    spec = study()
    report = evaluate_study(spec)
    frame, names = normalize_observations(spec)
    realized, _ = aligned_returns(frame, names, spec["evaluation_as_of"])
    for result in report["results"].values():
        cash, equity, bond = 100.0, 450.0, 450.0
        for row in result["periods"]:
            start, end = pd.Timestamp(row["start"]), pd.Timestamp(row["end"])
            wealth = cash + equity + bond
            w = row["weights"]
            trade = abs(w["equity"] - equity / wealth) + abs(w["bond"] - bond / wealth)
            cost = abs(w["equity"] - equity / wealth) * 0.001
            cost += abs(w["bond"] - bond / wealth) * 0.002
            assert row["turnover"] == pytest.approx(trade)
            assert trade <= 1.0 + 1e-8 and max(w.values()) <= 0.65 + 1e-8
            assert sum(w.values()) == pytest.approx(0.9)
            assert row["cash_weight"] == pytest.approx(0.1)
            remaining = wealth * (1 - cost)
            equity = remaining * w["equity"] * (1 + realized.loc[(start, end), "equity"])
            bond = remaining * w["bond"] * (1 + realized.loc[(start, end), "bond"])
            cash = remaining * 0.1 * 1.001
            assert row["closing_capital"] == pytest.approx(cash + equity + bond)
            assert pd.Timestamp(row["history_coverage"]["latest_input_available_at"]) <= start


def test_future_revision_cannot_change_earlier_decisions_and_fx_cross_term():
    spec = study()
    original = evaluate_study(spec)
    revised = copy.deepcopy(spec["observations"][0])
    revised.update(known_at="2025-01-09T00:00:00Z", net_return=0.9)
    spec["observations"].append(revised)
    changed = evaluate_study(spec)
    for method, result in changed["results"].items():
        for new, old in zip(result["periods"], original["results"][method]["periods"]):
            assert new["weights"] == old["weights"]
            assert new["closing_capital"] == old["closing_capital"]
            assert new["risk"] == old["risk"]
    spec["sleeves"][0]["currency"] = "EUR"
    spec["observations"][0]["fx_end"] = 1.1
    frame, _ = normalize_observations(spec)
    assert frame.iloc[0]["base_return"] == pytest.approx(1.01 * 1.1 - 1)
    spec["observations"][0]["fx_end_known_at"] = "2025-01-04T00:00:00Z"
    frame, names = normalize_observations(spec)
    history, coverage = aligned_returns(frame, names, "2025-01-03T00:00:00Z")
    assert len(history) == 1 and coverage["incomplete_periods"] == 1


def test_independent_sample_variance_euler_and_null_constant_correlation():
    data = pd.DataFrame({"A": [0.1, -0.1, 0.0], "B": [0.0, 0.1, -0.1]})
    out = diversification(data, pd.Series({"A": 0.4, "B": 0.6}))
    # Means are zero; variances .01, covariance -.005 using divisor n - 1.
    expected = 0.4**2 * 0.01 + 0.6**2 * 0.01 - 2 * 0.4 * 0.6 * 0.005
    assert out["variance"] == pytest.approx(expected)
    assert sum(out["variance_contributions"].values()) == pytest.approx(expected)
    assert out["diversification_ratio"] == pytest.approx(0.1 / np.sqrt(expected))
    constant = diversification(data * 0, pd.Series({"A": 0.4, "B": 0.6}))
    assert constant["diversification_ratio"] is None
    assert constant["correlation"] == [[None, None], [None, None]]


def test_negative_correlation_risk_parity_against_two_asset_closed_form():
    # For two assets the cross terms cancel: w1*sigma1 = w2*sigma2.
    cov = pd.DataFrame([[0.04, -0.012], [-0.012, 0.01]], index=["A", "B"], columns=["A", "B"])
    weights = equal_risk_contribution_weights(cov)
    assert weights.to_list() == pytest.approx([1 / 3, 2 / 3], abs=1e-10)
    assert equal_risk_contribution_weights(cov * 1e-8).to_list() == pytest.approx(weights)
    with pytest.raises(RuntimeError, match="converge"):
        equal_risk_contribution_weights(cov, max_iterations=1)
    for settings in ({"max_iterations": 0}, {"tolerance": 0}, {"tolerance": float("nan")}):
        with pytest.raises(ValueError):
            equal_risk_contribution_weights(cov, **settings)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda s: s.update(schema="unknown"), "schema"),
        (lambda s: s.update(data_nature="live"), "data_nature"),
        (lambda s: s["sleeves"].append(s["sleeves"][0]), "uniquely"),
        (lambda s: s["sleeves"][0].update(return_basis="gross"), "net simple"),
        (lambda s: s["observations"][0].update(known_at="2025-01-01T00:00:00Z"), "known_at"),
        (lambda s: s["observations"][0].update(sleeve="missing"), "undeclared"),
        (lambda s: s["observations"][0].update(fx_start=2), "base-currency"),
        (lambda s: s["observations"][0].update(fx_start_known_at="2024-01-01T00:00:00Z"), "FX"),
        (lambda s: s["observations"].append(s["observations"][0]), "vintages"),
        (lambda s: s["observations"][0].update(start="2025-01-01"), "timezone"),
        (lambda s: s["observations"][0].update(net_return=True), "numeric"),
        (lambda s: s["observations"][0].update(net_return=float("nan")), "finite"),
        (lambda s: s["observations"][0].update(fx_start=0), "positive"),
        (lambda s: s["sleeves"][0].update(source_ref=" "), "nonblank"),
        (lambda s: s["allocation"].update(mode="cost_aware"), "study supports"),
        (lambda s: s.update(max_weight=0.1), "infeasible"),
        (lambda s: s["initial_weights"].update(equity=1), "sum"),
        (lambda s: s["observations"].pop(), "missing"),
        (lambda s: s["evaluation_periods"][1].update(start="2025-01-05T00:00:00Z"), "contiguous"),
        (lambda s: s.update(evaluation_periods=[]), "nonempty"),
        (lambda s: s["allocation"].update(min_observations=4, lookback=8), "insufficient"),
    ],
)
def test_reject_invalid(change, match):
    spec = study()
    if match == "insufficient":
        spec["observations"][0]["known_at"] = "2025-02-01T00:00:00Z"
    change(spec)
    with pytest.raises((ValueError, TypeError), match=match):
        evaluate_study(spec)


def test_overlap_empty_cost_exhaustion_insolvency_and_cli(tmp_path):
    spec = study()
    spec["observations"][1]["start"] = "2025-01-01T12:00:00Z"
    with pytest.raises(ValueError, match="overlapping"):
        evaluate_study(spec)
    spec["observations"] = []
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_study(spec)
    spec = study()
    spec["linear_costs"] = {"equity": 10.0, "bond": 10.0}
    with pytest.raises(ValueError, match="costs exhaust"):
        evaluate_study(spec)
    spec = study()
    spec["allocation"]["mode"] = "equal"
    spec["invested_limit"] = 1.0
    for item in spec["observations"]:
        if item["start"] == spec["evaluation_periods"][0]["start"]:
            item["net_return"] = -1
    with pytest.raises(ValueError, match="insolvent"):
        evaluate_study(spec)
    source, target = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(json.dumps(study()), encoding="utf-8")
    main(["--input", str(source), "--out", str(target)])
    assert json.loads(target.read_text())["input_sha256"]
    with pytest.raises(FileExistsError):
        main(["--input", str(source), "--out", str(target)])
    with pytest.raises(ValueError, match="two matched"):
        diversification(pd.DataFrame({"A": [0.1]}), pd.Series({"A": 1.0}))
