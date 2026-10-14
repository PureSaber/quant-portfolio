"""Reproducible factor-only scalability probe (not a live trading latency claim)."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from quant_portfolio.optimization import OptimizationConstraints
from quant_portfolio.solver import RiskMatrix, solve_portfolio


def peak_memory():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        api = ctypes.windll.psapi.GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if not api(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError()
        return counters.PeakWorkingSetSize
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n, k = args.assets, 25
    rng = np.random.default_rng(20261014 + n)
    x = rng.normal(0, 0.4, (n, k))
    f = np.eye(k) * 0.003
    d = rng.uniform(0.01, 0.04, n)
    risk = RiskMatrix(exposures=x, factor_covariance=f, specific_variance=d)
    current = np.ones(n) / n
    mu = rng.normal(0.001, 0.0001, n)
    times, records = [], []
    for _ in range(args.repeats):
        start = perf_counter()
        got = solve_portfolio(
            mu,
            risk,
            current,
            OptimizationConstraints(max_weight=3 / n, max_turnover=0.4),
            risk_scale=1 / 252,
            risk_aversion=5,
            linear_costs=np.ones(n) * 0.00005,
            timeout_seconds=args.timeout,
        )
        times.append(perf_counter() - start)
        records.append(got.diagnostics)
    payload = {
        "assets": n,
        "factors": k,
        "seed": 20261014 + n,
        "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "elapsed_seconds": times,
        "p95_seconds": float(np.quantile(times, 0.95)),
        "process_peak_memory_bytes": peak_memory(),
        "asset_covariance_materialized": False,
        "diagnostics": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "diagnostics"}))


if __name__ == "__main__":
    main()
