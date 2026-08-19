#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from quant_portfolio.synthetic_spread import (
    SyntheticSpreadConfig,
    backtest_synthetic_spread,
    generate_synthetic_pair,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic spread research demo")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthetic_spread_demo.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/synthetic_demo"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    config = SyntheticSpreadConfig(**raw)
    frame = generate_synthetic_pair(config)
    result, metrics = backtest_synthetic_spread(
        frame,
        lookback=config.lookback,
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        cost_bps=config.cost_bps,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "synthetic_series.csv", index=False)
    payload = {
        "disclaimer": "SYNTHETIC DEMONSTRATION — NOT INVESTMENT PERFORMANCE",
        "config": raw,
        "metrics": metrics.to_dict(),
    }
    (args.out / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
