"""Offline evidence check of public Coinbase candles; no execution calibration.

Input directory contains BTC-USD.json, ETH-USD.json and provenance.json.
Run with --input-dir and a new --out-dir. Retrieval stays outside library code.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

from quant_portfolio.capacity_study import evaluate_capacity


def validate(source, target):
    source, target = Path(source), Path(target)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8-sig"))
    start, end = pd.Timestamp("2025-01-01T00:00:00Z"), pd.Timestamp("2025-03-01T00:00:00Z")
    expected = pd.date_range(start, end, inclusive="left", freq="D")
    rows, quality = [], []
    for asset in ("BTC-USD", "ETH-USD"):
        raw = (source / (asset + ".json")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == provenance["files"][asset + ".json"]
        data = json.loads(raw)
        assert all(len(row) == 6 for row in data)
        table = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        table["time"] = pd.to_datetime(table["time"], unit="s", utc=True)
        duplicates = int(table.time.duplicated().sum())
        assert duplicates == 0
        trimmed = table[(table.time >= start) & (table.time < end)].sort_values("time")
        assert list(trimmed.time) == list(expected), "missing daily buckets; no fill permitted"
        assert np.isfinite(trimmed.drop(columns="time").to_numpy()).all()
        assert (trimmed.volume >= 0).all() and (trimmed.low > 0).all()
        assert (trimmed.low <= trimmed[["open", "close"]].min(axis=1)).all()
        assert (trimmed.high >= trimmed[["open", "close"]].max(axis=1)).all()
        returns = trimmed.close.pct_change().dropna()
        adv = float((trimmed.close * trimmed.volume).mean())
        volatility = float(returns.std(ddof=1))
        # Independent Python scalar implementation, not the pandas computation above.
        plain = list(trimmed.itertuples(index=False))
        scalar_adv = sum(r.close * r.volume for r in plain) / len(plain)
        scalar_returns = [plain[i].close / plain[i - 1].close - 1 for i in range(1, len(plain))]
        assert math.isclose(scalar_adv, adv, rel_tol=1e-14)
        assert math.isclose(statistics.stdev(scalar_returns), volatility, rel_tol=1e-14)
        quality.append(
            {
                "asset": asset,
                "downloaded_rows": len(table),
                "used_rows": len(trimmed),
                "excluded_outside_requested_interval": len(table) - len(trimmed),
                "duplicate_keys": duplicates,
                "missing_days": 0,
                "invalid_ohlcv_rows": 0,
                "return_observations": len(returns),
                "adv_close_volume_proxy": adv,
                "adv_low_volume_lower_bound": float((trimmed.low * trimmed.volume).mean()),
                "adv_high_volume_upper_bound": float((trimmed.high * trimmed.volume).mean()),
                "daily_close_return_sample_volatility": volatility,
            }
        )
        rows.append(
            {
                "asset": asset,
                "adv_base": adv,
                "daily_volatility": volatility,
                "linear_rate": 0.001,
                "observed_at": end.isoformat(),
                "known_at": provenance["retrieved_at"],
                "source_ref": provenance["urls"][asset],
            }
        )
    spec = {
        "schema": "quant-portfolio.capacity-study/v1",
        "base_currency": "USD",
        "data_nature": "retrospective",
        "as_of": provenance["retrieved_at"],
        "max_liquidity_age_days": 1000,
        "holdings_weights": {"BTC-USD": 0.5, "ETH-USD": 0.5},
        "trade_weights": {"BTC-USD": 0.1, "ETH-USD": -0.1},
        "max_participation": 0.01,
        "execution_days": 1,
        "liquidation_days": 3,
        "cost_budget_return": 0.002,
        "capital": [1000000, 10000000, 100000000],
        "turnover_scales": [1, 2],
        "liquidity_scales": [0.5, 1],
        "impact_coefficients": [0, 0.1, 0.3],
        "volatility_scales": [1, 2],
        "liquidity": rows,
    }
    report = evaluate_capacity(spec)
    findings = {
        "source": provenance,
        "grain": "product / UTC daily candle",
        "quality": quality,
        "data_nature": "retrospective",
        "input_moment_independent_checks": "passed",
        "covered_period": "2025-01-01 through 2025-02-28 UTC, 59 days each",
        "permitted_claim": "capacity sensitivity conditional on observed 2025 venue liquidity proxies",
        "not_verified": [
            "original 2025 publication vintages",
            "cross-venue total market ADV",
            "execution reference quotes and parent-order footprints",
            "real cost coefficients",
            "current liquidity capacity",
            "profitable or executable portfolio strategy",
        ],
        "findings": [
            {
                "severity": "high",
                "issue": "No parent-order/reference-price execution facts",
                "impact": "real impact calibration is unavailable; coefficients remain explicit assumptions",
            },
            {
                "severity": "medium",
                "issue": "close*volume is a notional proxy",
                "impact": "low*volume and high*volume bounds are retained; no exact dollar-ADV claim",
            },
            {
                "severity": "high",
                "issue": "Historical candles retrieved in 2026, no publication archive",
                "impact": "known_at is retrieval time; no historical-PIT claim or earlier allocation decision",
            },
        ],
    }
    target.mkdir(parents=True, exist_ok=False)
    for filename, value in [
        ("quality.json", findings),
        ("capacity-input.json", spec),
        ("capacity-result.json", report),
    ]:
        (target / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    validate(args.input_dir, args.out_dir)
