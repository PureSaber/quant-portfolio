# Quant Research Portfolio

A small, reviewable portfolio of quantitative-research engineering patterns:

- deterministic synthetic data generation;
- causal signal construction with no look-ahead;
- transparent transaction-cost accounting;
- multi-strategy portfolio allocation;
- reproducible YAML-driven runs, tests, and machine-readable artifacts.

> **Disclosure:** every market series and demonstration result in this repository is synthetic. This repository contains no employer data, client data, real trades, internal strategy parameters, or claims of live investment performance.

## What this demonstrates

| Area | Evidence in this repository |
|---|---|
| Research discipline | lagged rolling features, explicit costs, deterministic seeds |
| Engineering | typed Python, CLI entry points, YAML configs, CI, pytest, Ruff |
| Portfolio construction | strategy-book budgeting, holdings aggregation, optional factor tilts |
| Reproducibility | fixed fixtures plus CSV/JSON outputs that can be reconciled |
| Risk awareness | drawdown, turnover, cost and disclosure checks |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
python scripts/run_synthetic_demo.py
```

The demo writes:

```text
artifacts/synthetic_demo/metrics.json
artifacts/synthetic_demo/synthetic_series.csv
```

Each report is labeled `SYNTHETIC DEMONSTRATION — NOT INVESTMENT PERFORMANCE`.

## Existing allocator CLI

```bash
quant-portfolio status \
  --config configs/factor_allocator_smoke.yaml \
  --out state/portfolio.json
```

The allocator combines versioned strategy-book NAV and holdings fixtures, then optionally applies a normalized synthetic factor tilt.

## Repository map

```text
src/quant_portfolio/
├── allocator.py           # multi-book allocation and optional factor tilt
├── cli.py                 # command-line interface
└── synthetic_spread.py    # synthetic generator + causal, cost-aware demo
configs/
├── factor_allocator_smoke.yaml
└── synthetic_spread_demo.yaml
scripts/
└── run_synthetic_demo.py
docs/
├── ARCHITECTURE.md
└── DISCLOSURE.md
tests/
└── ...
```

## Important limitations

- The synthetic process is intentionally simple and is not calibrated to a real market.
- A positive synthetic backtest does not imply investability or future returns.
- Execution, liquidity, capacity, financing, taxes, and market-impact models are incomplete.
- This code is educational and is not investment advice.

See [Disclosure and data policy](docs/DISCLOSURE.md) and [Architecture](docs/ARCHITECTURE.md).
