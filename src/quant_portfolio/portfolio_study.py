"""PIT multi-sleeve research, reusing the existing allocation solvers.

Return observations are exact, non-overlapping periods. No NAV interpolation,
forward fill, or presumed publication timestamp is permitted.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from quant_portfolio.optimization import (
    research_allocation_weights,
    validate_research_allocation,
)
from quant_portfolio.study_io import digest, number, publish, text, timestamp

NATURES = {"synthetic", "retrospective", "historical_pit", "independent_forward"}


def normalize_observations(spec):
    """Return one validated table retaining all publication vintages."""
    if spec.get("schema") != "quant-portfolio.sleeve-study/v1":
        raise ValueError("unsupported sleeve study schema")
    base = text(spec["base_currency"])
    if spec["data_nature"] not in NATURES:
        raise ValueError("unknown data_nature")
    definitions = spec["sleeves"]
    names = [text(item["id"]) for item in definitions]
    if not names or len(set(names)) != len(names):
        raise ValueError("sleeves must be nonempty and uniquely named")
    currencies = {item["id"]: text(item["currency"]) for item in definitions}
    for item in definitions:
        text(item["source_ref"])
        if item["return_basis"] != "net_simple_total_return":
            raise ValueError("supply explicit net simple total returns, including distributions")
    rows = []
    for observation in spec["observations"]:
        name = observation["sleeve"]
        if name not in currencies:
            raise ValueError("observation names an undeclared sleeve")
        start, end = timestamp(observation["start"]), timestamp(observation["end"])
        known = timestamp(observation["known_at"])
        if start >= end or known < end:
            raise ValueError("returns require start < end <= known_at")
        local = number(observation["net_return"], "net_return", minimum=-1)
        fx_start = number(observation["fx_start"], "fx_start", positive=True)
        fx_end = number(observation["fx_end"], "fx_end", positive=True)
        fx_start_known = timestamp(observation["fx_start_known_at"])
        fx_end_known = timestamp(observation["fx_end_known_at"])
        if fx_start_known < start or fx_end_known < end:
            raise ValueError("FX endpoints cannot be known before their valuation timestamps")
        if currencies[name] == base and (fx_start != 1 or fx_end != 1):
            raise ValueError("base-currency FX must equal one")
        fx_return = fx_end / fx_start - 1
        rows.append(
            {
                "sleeve": name,
                "start": start,
                "end": end,
                "vintage": known,
                "available": max(known, fx_start_known, fx_end_known),
                "local_return": local,
                "fx_return": fx_return,
                "base_return": (1 + local) * (1 + fx_return) - 1,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["sleeve", "start", "end", "vintage"]).any():
        raise ValueError("return observations must be nonempty with unique publication vintages")
    for _, group in frame.groupby("sleeve"):
        periods = group[["start", "end"]].drop_duplicates().sort_values("start")
        if any(
            periods["start"].iloc[i] < periods["end"].iloc[i - 1] for i in range(1, len(periods))
        ):
            raise ValueError("overlapping periods within a sleeve cannot be used")
    return frame, names


def aligned_returns(frame, names, as_of, *, before=None):
    """Select latest *available* vintage, then intersect exact period boundaries."""
    visible = frame[frame["available"] <= timestamp(as_of)]
    if before is not None:
        visible = visible[visible["end"] <= timestamp(before)]
    visible = visible.sort_values("vintage").drop_duplicates(
        ["sleeve", "start", "end"], keep="last"
    )
    wide = visible.pivot(index=["start", "end"], columns="sleeve", values="base_return")
    wide = wide.reindex(columns=names).sort_index()
    complete = wide.dropna()
    return complete, {
        "visible_observations": len(visible),
        "complete_periods": len(complete),
        "incomplete_periods": len(wide) - len(complete),
        "excluded_unavailable_observations": int((frame["available"] > timestamp(as_of)).sum()),
        "latest_input_available_at": (
            visible["available"].max().isoformat() if not visible.empty else None
        ),
    }


def diversification(returns, weights):
    """Sample covariance and Euler variance attribution, without annualization guesses."""
    if len(returns) < 2:
        raise ValueError("diversification requires two matched observations")
    covariance = returns.cov().to_numpy()
    values = weights.reindex(returns.columns).to_numpy(dtype=float)
    component = values * (covariance @ values)
    variance = float(component.sum())
    volatility = float(np.sqrt(max(variance, 0)))
    standalone = np.sqrt(np.maximum(np.diag(covariance), 0))
    numerator = float(np.abs(values) @ standalone)
    correlation = returns.corr()
    return {
        "covariance": covariance.tolist(),
        "assets": list(returns.columns),
        "correlation": [
            [float(x) if np.isfinite(x) else None for x in row] for row in correlation.to_numpy()
        ],
        "variance": variance,
        "volatility": volatility,
        "variance_contributions": dict(zip(returns.columns, component.tolist())),
        "diversification_ratio": numerator / volatility if volatility > 0 else None,
        "volatility_reduction": numerator - volatility,
        "observations": len(returns),
        "frequency": "exact_input_periods",
    }


def evaluate_study(spec):
    frame, names = normalize_observations(spec)
    as_of = timestamp(spec["evaluation_as_of"])
    config = validate_research_allocation(spec["allocation"])
    # These modes need only historical returns. Forecast scores/views require a
    # separate PIT forecast contract and are deliberately not inferred from NAV.
    if config["mode"] not in {"risk_parity", "hrp", "inverse_vol", "equal"}:
        raise ValueError("study supports equal, inverse_vol, risk_parity and hrp")
    invested = number(spec["invested_limit"], "invested_limit", positive=True)
    cap = number(spec["max_weight"], "max_weight", positive=True)
    if invested > 1 or cap > 1 or cap * len(names) < invested:
        raise ValueError("infeasible investment or position cap")
    costs = pd.Series({k: number(spec["linear_costs"][k], "linear_cost", minimum=0) for k in names})
    held0 = pd.Series({k: number(spec["initial_weights"][k], "weight", minimum=0) for k in names})
    if set(spec["initial_weights"]) != set(names) or held0.sum() > 1 + 1e-12:
        raise ValueError("initial weights must cover exactly the sleeves and sum <= 1")
    capital = number(spec["initial_capital"], "initial_capital", positive=True)
    realized, coverage = aligned_returns(frame, names, as_of)
    periods = []
    for raw in spec["evaluation_periods"]:
        start, end = timestamp(raw["start"]), timestamp(raw["end"])
        cash = number(raw["cash_return"], "cash_return", minimum=-1)
        if start >= end or end > as_of or (periods and start != periods[-1][1]):
            raise ValueError("evaluation periods must be contiguous, ordered and mature")
        if (start, end) not in realized.index:
            raise ValueError("evaluation has a missing or not-yet-available sleeve period")
        periods.append((start, end, cash))
    if not periods:
        raise ValueError("evaluation_periods must be nonempty")
    methods = {
        "candidate": config["mode"],
        "equal_benchmark": "equal",
        "inverse_vol_benchmark": "inverse_vol",
    }
    results = {}
    for label, mode in methods.items():
        held, wealth, rows = held0.copy(), capital, []
        for start, end, cash in periods:
            history, history_coverage = aligned_returns(frame, names, start, before=start)
            history = history.tail(config["lookback"])
            if len(history) < config["min_observations"]:
                raise ValueError("insufficient common PIT history for same-window comparison")
            settings = dict(config, mode=mode)
            if mode in {"equal", "inverse_vol"}:
                settings["covariance_estimator"] = "diagonal"
                settings.pop("ewma_decay", None)
            target = research_allocation_weights(
                pd.Series(0.0, index=names),
                history,
                held,
                costs,
                invested_limit=invested,
                max_weight=cap,
                config=settings,
            )
            turnover = float((target - held).abs().sum())
            cost = float((target - held).abs() @ costs)
            if cost >= 1:
                raise ValueError("costs exhaust portfolio")
            actual = realized.loc[(start, end)]
            gross = float(target @ actual + (1 - target.sum()) * cash)
            if gross <= -1:
                raise ValueError("portfolio is insolvent")
            net = (1 - cost) * (1 + gross) - 1
            opening = wealth
            wealth *= 1 + net
            held = target * (1 + actual) / (1 + gross)
            rows.append(
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "weights": target.to_dict(),
                    "cash_weight": float(1 - target.sum()),
                    "turnover": turnover,
                    "cost_return": cost,
                    "gross_return": gross,
                    "net_return": net,
                    "opening_capital": opening,
                    "closing_capital": wealth,
                    "history_periods": [[a.isoformat(), b.isoformat()] for a, b in history.index],
                    "history_coverage": history_coverage,
                    "risk": diversification(history, target),
                }
            )
        path = np.array([capital] + [row["closing_capital"] for row in rows])
        results[label] = {
            "method": mode,
            "periods": rows,
            "total_return": wealth / capital - 1,
            "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
            "total_turnover": sum(row["turnover"] for row in rows),
        }
    return {
        "schema": "quant-portfolio.sleeve-study-result/v1",
        "evaluation_status": "evaluated",
        "data_nature": spec["data_nature"],
        "base_currency": spec["base_currency"],
        "source_refs": {item["id"]: item["source_ref"] for item in spec["sleeves"]},
        "spec_digest": digest(spec),
        "coverage": coverage,
        "results": results,
        "constraints": {
            "invested_limit": invested,
            "max_weight": cap,
            "max_turnover": config["max_turnover"],
        },
        "assumptions": [
            "Sleeve net returns include native trading/financing costs.",
            "Overlay linear costs are additional reallocation costs before returns.",
            "Base/local FX includes the multiplicative cross term; no FX hedging.",
            "Cash returns are explicit base-currency realized inputs.",
            "No interpolated NAV, assumed execution, or causal alpha attribution.",
            "Publication provenance is supplied, not independently authenticated.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    publish(args.input, args.out, evaluate_study)


if __name__ == "__main__":
    main()
