from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from quant_portfolio.allocator import allocate, load_config
from quant_portfolio.optimization import (
    OptimizationConstraints,
    estimate_capacity,
    estimate_covariance,
    optimize_mean_variance,
)


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    snap = allocate(cfg)
    payload = {
        "as_of": snap.as_of,
        "total_nav": snap.total_nav,
        "books": snap.books,
        "combined_weights": snap.combined_weights,
    }
    out = Path(args.out) if args.out else None
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


def _indexed_series(path: str, value_column: str) -> pd.Series | None:
    if not path:
        return None
    frame = pd.read_csv(path)
    required = {"symbol", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}缺少字段: {missing}")
    return frame.set_index("symbol")[value_column].astype(float)


def cmd_optimize(args: argparse.Namespace) -> int:
    expected = _indexed_series(args.expected_returns, "expected_return")
    if expected is None:
        raise ValueError("expected returns are required")
    history = pd.read_csv(args.returns)
    if "date" in history:
        history = history.drop(columns="date")
    history = history.reindex(columns=expected.index)
    covariance = estimate_covariance(
        history, shrinkage=args.shrinkage, annualization=args.annualization
    )
    current = _indexed_series(args.current_weights, "weight")
    costs = _indexed_series(args.linear_costs, "cost")
    constraints = OptimizationConstraints(
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        max_turnover=args.max_turnover,
    )
    result = optimize_mean_variance(
        expected,
        covariance,
        current_weights=current,
        linear_costs=costs,
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
        constraints=constraints,
    )
    payload: dict[str, object] = {
        "weights": {str(key): float(value) for key, value in result.weights.items()},
        "expected_return": result.expected_return,
        "volatility": result.volatility,
        "turnover": result.turnover,
        "objective": result.objective,
        "converged": result.converged,
        "iterations": result.iterations,
    }
    liquidity = _indexed_series(args.liquidity, "average_daily_value")
    if liquidity is not None:
        payload["capacity"] = estimate_capacity(
            result.weights,
            liquidity,
            max_participation=args.max_participation,
            liquidation_days=args.liquidation_days,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant-portfolio")
    sub = p.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="Combine strategy books into portfolio view")
    status.add_argument("--config", required=True)
    status.add_argument("--out", default="")
    status.set_defaults(func=cmd_status)
    optimize = sub.add_parser("optimize", help="Optimize asset weights with explicit costs")
    optimize.add_argument("--expected-returns", required=True)
    optimize.add_argument("--returns", required=True, help="Wide historical return CSV")
    optimize.add_argument("--current-weights", default="")
    optimize.add_argument("--linear-costs", default="")
    optimize.add_argument("--liquidity", default="")
    optimize.add_argument("--out", required=True)
    optimize.add_argument("--risk-aversion", type=float, default=5.0)
    optimize.add_argument("--turnover-penalty", type=float, default=0.0)
    optimize.add_argument("--shrinkage", type=float, default=0.2)
    optimize.add_argument("--annualization", type=int, default=252)
    optimize.add_argument("--min-weight", type=float, default=0.0)
    optimize.add_argument("--max-weight", type=float, default=1.0)
    optimize.add_argument("--max-turnover", type=float, default=1.0)
    optimize.add_argument("--max-participation", type=float, default=0.1)
    optimize.add_argument("--liquidation-days", type=int, default=1)
    optimize.set_defaults(func=cmd_optimize)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
