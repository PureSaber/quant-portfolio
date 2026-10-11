"""Capital/turnover/liquidity sensitivity and chronological cost diagnostics."""

from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd

from quant_portfolio.optimization import estimate_capacity, square_root_impact_cost
from quant_portfolio.study_io import digest, number, publish, text, timestamp


def _axis(spec, key, *, minimum=0, positive=False):
    values = [number(v, key, minimum=minimum, positive=positive) for v in spec[key]]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{key} must contain distinct nonempty values")
    return values


def evaluate_capacity(spec):
    if spec["schema"] != "quant-portfolio.capacity-study/v1":
        raise ValueError("unsupported capacity study schema")
    as_of = timestamp(spec["as_of"])
    text(spec["base_currency"])
    if spec["data_nature"] not in {
        "synthetic",
        "retrospective",
        "historical_pit",
        "independent_forward",
    }:
        raise ValueError("unknown data_nature")
    names = [text(row["asset"]) for row in spec["liquidity"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("liquidity must cover distinct assets")
    if set(names) != set(spec["holdings_weights"]) or set(names) != set(spec["trade_weights"]):
        raise ValueError("liquidity, holdings and trade universes must match exactly")
    for row in spec["liquidity"]:
        if timestamp(row["observed_at"]) > as_of or timestamp(row["known_at"]) > as_of:
            raise ValueError("future liquidity observation")
        if timestamp(row["known_at"]) < timestamp(row["observed_at"]):
            raise ValueError("liquidity cannot be known before observation")
        text(row["source_ref"])
    max_age = number(spec["max_liquidity_age_days"], "max_liquidity_age_days", minimum=0)
    if any(
        (as_of - timestamp(r["observed_at"])).total_seconds() > max_age * 86400
        for r in spec["liquidity"]
    ):
        raise ValueError("stale liquidity observation")
    adv = pd.Series(
        {r["asset"]: number(r["adv_base"], "ADV", positive=True) for r in spec["liquidity"]}
    )
    vol = pd.Series(
        {
            r["asset"]: number(r["daily_volatility"], "volatility", minimum=0)
            for r in spec["liquidity"]
        }
    )
    linear = pd.Series(
        {r["asset"]: number(r["linear_rate"], "linear_rate", minimum=0) for r in spec["liquidity"]}
    )
    weights = pd.Series({k: number(spec["holdings_weights"][k], "holding") for k in names})
    trades = pd.Series({k: number(spec["trade_weights"][k], "trade") for k in names})
    participation = number(spec["max_participation"], "max_participation", positive=True)
    if participation > 1:
        raise ValueError("participation cannot exceed one")
    days, horizon = spec["execution_days"], spec["liquidation_days"]
    if any(type(v) is not int or v < 1 for v in (days, horizon)):
        raise ValueError("execution/liquidation days must be positive integers")
    budget = number(spec["cost_budget_return"], "cost_budget_return", minimum=0)
    axes = [
        _axis(spec, "capital", positive=True),
        _axis(spec, "turnover_scales"),
        _axis(spec, "liquidity_scales", positive=True),
        _axis(spec, "impact_coefficients"),
        _axis(spec, "volatility_scales"),
    ]
    if np.prod([len(a) for a in axes]) > 100000:
        raise ValueError("capacity grid exceeds 100000 scenarios")
    rows = []
    for capital, turnover, liquidity, coefficient, volatility in product(*axes):
        adjusted_adv = adv * liquidity
        notional = capital * trades * turnover
        impact = square_root_impact_cost(
            notional / days,
            adjusted_adv,
            vol * volatility,
            impact_coefficient=coefficient,
        )
        linear_cost = notional.abs() * linear
        impact_cost = impact["impact_cost"] * days
        total = float(linear_cost.sum() + impact_cost.sum())
        holding_capacity = (
            estimate_capacity(
                weights, adjusted_adv, max_participation=participation, liquidation_days=horizon
            )
            if weights.abs().sum() > 0
            else {"capacity": None, "binding_asset": None}
        )
        active = (trades * turnover).abs()
        capacities = adjusted_adv[active > 0] * participation * days / active[active > 0]
        peak = float(impact["participation"].max())
        rows.append(
            {
                "capital": capital,
                "turnover_scale": turnover,
                "liquidity_scale": liquidity,
                "impact_coefficient": coefficient,
                "volatility_scale": volatility,
                "two_way_turnover": float(active.sum()),
                "linear_cost": float(linear_cost.sum()),
                "square_root_cost": float(impact_cost.sum()),
                "total_cost": total,
                "cost_return": total / capital,
                "peak_daily_participation": peak,
                "trade_capacity": float(capacities.min()) if not capacities.empty else None,
                "trade_binding_asset": str(capacities.idxmin()) if not capacities.empty else None,
                "holding_capacity": holding_capacity,
                "participation_breach": peak > participation,
                "holding_capacity_breach": (
                    capital > holding_capacity["capacity"]
                    if holding_capacity["capacity"] is not None
                    else False
                ),
                "cost_budget_breach": total / capital > budget,
                "assets": [
                    {
                        "asset": k,
                        "order_notional": float(notional[k]),
                        "daily_participation": float(impact.loc[k, "participation"]),
                        "linear_cost": float(linear_cost[k]),
                        "impact_cost": float(impact_cost[k]),
                    }
                    for k in names
                ],
            }
        )
    return {
        "schema": "quant-portfolio.capacity-study-result/v1",
        "data_nature": spec["data_nature"],
        "base_currency": spec["base_currency"],
        "spec_digest": digest(spec),
        "scenarios": rows,
        "assumptions": [
            "ADV and volatility are frozen explicit scenario inputs, already in base currency.",
            "Orders are split equally across execution days; no carryover or intraday schedule.",
            "Impact uses existing coefficient*sigma*sqrt(daily notional/ADV), charged on notional.",
            "Linear rates are separate costs; coefficient and turnover grids are assumptions.",
            "Holding capacity and trade capacity are different constraints.",
            "This is sensitivity, not realized execution calibration or a profitability forecast.",
        ],
    }


def calibrate_costs(spec):
    """Fit a signed order-level predictive rate on training only; no causal impact claim."""
    if spec["schema"] != "quant-portfolio.cost-calibration/v1":
        raise ValueError("unsupported cost calibration schema")
    train_end, holdout_start = timestamp(spec["training_end"]), timestamp(spec["holdout_start"])
    as_of = timestamp(spec["evaluation_as_of"])
    if not train_end < holdout_start <= as_of:
        raise ValueError("training must precede holdout and evaluation")
    minimum = spec["minimum_orders"]
    if type(minimum) is not int or minimum < 3:
        raise ValueError("minimum_orders must be an integer >= 3")
    nature = spec["data_nature"]
    if nature not in {"synthetic", "retrospective", "historical_pit", "independent_forward"}:
        raise ValueError("unknown data_nature")
    rows, identities = [], set()
    for raw in spec["orders"]:
        identity = text(raw["order_id"])
        if identity in identities:
            raise ValueError("orders must already be aggregated, with unique order_id")
        identities.add(identity)
        start, end, known = (timestamp(raw[k]) for k in ("accepted_at", "last_fill_at", "known_at"))
        if not start <= end <= known:
            raise ValueError("order timing is inconsistent")
        reference = timestamp(raw["reference_at"])
        if reference > start or timestamp(raw["reference_known_at"]) > start:
            raise ValueError("execution reference must be available when order is accepted")
        age = number(spec["max_reference_age_seconds"], "reference age", minimum=0)
        if (start - reference).total_seconds() > age:
            raise ValueError("stale execution reference")
        text(raw["source_ref"])
        quantity = number(raw["notional_base"], "notional", positive=True)
        adv = number(raw["adv_base"], "ADV", positive=True)
        vol = number(raw["daily_volatility"], "volatility", minimum=0)
        if timestamp(raw["liquidity_known_at"]) > start:
            raise ValueError("liquidity feature unavailable at order acceptance")
        observed = number(raw["signed_adverse_rate"], "signed_adverse_rate")
        if start <= train_end:
            partition = "training" if end <= train_end and known <= train_end else "excluded"
        else:
            partition = "holdout" if start >= holdout_start and known <= as_of else "excluded"
        rows.append(
            {
                "order_id": identity,
                "feature": vol * np.sqrt(quantity / adv),
                "observed_rate": observed,
                "weight": quantity,
                "partition": partition,
            }
        )
    train = [r for r in rows if r["partition"] == "training"]
    holdout = [r for r in rows if r["partition"] == "holdout"]
    reasons = []
    if len(train) < minimum:
        reasons.append("insufficient_training_orders")
    if len(holdout) < minimum:
        reasons.append("insufficient_holdout_orders")
    fit = None
    if not reasons:
        x, y, w = (np.array([r[k] for r in train]) for k in ("feature", "observed_rate", "weight"))
        dispersion = float(np.std(x))
        if dispersion <= 1e-12:
            reasons.append("impact_feature_not_identifiable")
        else:
            center = float(np.mean(x))
            design = np.column_stack([np.ones(len(x)), (x - center) / dispersion])
            root_weight = np.sqrt(w / w.sum())
            coefficients, _, rank, _ = np.linalg.lstsq(
                design * root_weight[:, None], y * root_weight, rcond=None
            )
            if rank != 2:
                reasons.append("rank_deficient_training_design")
            else:
                slope = float(coefficients[1] / dispersion)
                intercept = float(coefficients[0] - slope * center)
                constant = float(np.average(y, weights=w))
                hx, hy, hw = (
                    np.array([r[k] for r in holdout])
                    for k in ("feature", "observed_rate", "weight")
                )
                prediction = intercept + slope * hx
                fit = {
                    "linear_intercept": intercept,
                    "square_root_coefficient": slope,
                    "linear_only_rate": constant,
                    "holdout_weighted_mae_bps": float(
                        np.average(abs(hy - prediction), weights=hw) * 1e4
                    ),
                    "linear_only_holdout_mae_bps": float(
                        np.average(abs(hy - constant), weights=hw) * 1e4
                    ),
                    "holdout_signed_error_bps": float(
                        np.average(hy - prediction, weights=hw) * 1e4
                    ),
                    "nonnegative_cost_model_compatible": intercept >= 0 and slope >= 0,
                }
    return {
        "schema": "quant-portfolio.cost-calibration-result/v1",
        "data_nature": nature,
        "status": "unavailable" if reasons else "predictive_diagnostic",
        "reasons": reasons,
        "fit": fit,
        "orders": rows,
        "spec_digest": digest(spec),
        "training_orders": len(train),
        "holdout_orders": len(holdout),
        "real_market_validation": False
        if nature == "synthetic"
        else "requires_source_verification",
        "limitations": [
            "Signed adverse rate includes spread, timing and market movement; not isolated impact.",
            "Order grouping and PIT ADV/reference provenance must be supplied by the producer.",
            "No tuning on holdout; coefficients are unconstrained diagnostics, not auto-deployed costs.",
            "OHLCV alone cannot identify execution costs or a square-root coefficient.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["scan", "calibrate"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    publish(args.input, args.out, evaluate_capacity if args.mode == "scan" else calibrate_costs)


if __name__ == "__main__":
    main()
