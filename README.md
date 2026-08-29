# Quant Research Portfolio

A small, reviewable portfolio of quantitative-research engineering patterns:

- deterministic synthetic data generation;
- causal signal construction with no look-ahead;
- transparent transaction-cost accounting;
- multi-strategy portfolio allocation;
- causal cross-asset long/short target construction with QExec order suggestions;
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
python -m pip install --no-deps -r requirements.lock
python -m pip check
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
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

## Cost-aware optimizer

```bash
quant-portfolio optimize \
  --expected-returns expected_returns.csv \
  --returns asset_returns_wide.csv \
  --current-weights current_weights.csv \
  --linear-costs linear_costs.csv \
  --liquidity liquidity.csv \
  --max-weight 0.10 \
  --max-turnover 0.40 \
  --out state/target_portfolio.json
```

The optimizer uses a shrinkage/PSD-repaired covariance matrix and enforces budget, asset bounds,
and turnover while charging linear costs. The Python API additionally supports group caps and the
square-root market-impact model.

## Cross-asset target API

`quant_portfolio.optimize_cross_asset` consumes an immutable QExec `PortfolioRiskSnapshot`,
QDK `InstrumentSpec`, and explicit point-in-time price, FX and ADV observations. It supports
long/short, cash-aware gross/net leverage, instrument/asset-class/currency/venue/strategy caps,
margin, turnover, participation, liquidation-horizon, linear-cost and square-root-impact limits.
It either returns a deterministic `TargetPortfolio` plus a fixed-point report, or a structured
failure containing the binding constraints. `target_to_order_intents` is the only output path;
it emits QExec `OrderIntent` suggestions and never alters a ledger, positions, or cash.

The module has no live-order, network, or credential capability. Missing, future, duplicate, or
non-finite PIT inputs fail closed.

## M6 governance and reproducibility

Version `0.4.1` consumes only published annotated internal tags:

- `quant-data-kit v0.6.1` (`edf1351690dc60691cc6330390adcdbf8bc79c5f`)
- `quant-execution v0.4.1` (`29eccc0e392968b5f7c31976a329605aacce369a`)

`[tool.quant-workspace]` declares the real QDK `puresaber.instrument-spec` input and QExec
`puresaber.execution.account-snapshot`/`puresaber.execution.order-intent` boundaries. The
portfolio optimizer only reads snapshots and emits order suggestions; it cannot alter the ledger.
`requirements.lock` is the sole audited Python3.10-3.12 lock for runtime, development, and
editable-build requirements. Rebuild it only from Python3.10 with:

```bash
python -m piptools compile --extra dev --build-deps-for editable --allow-unsafe --strip-extras \
  --resolver backtracking --index-url https://pypi.org/simple \
  --constraint requirements-constraints.txt --output-file requirements.lock pyproject.toml
```

Run the four locked installation commands above, Ruff check/format, the full test suite, and the
synthetic demo before proposing a release. To roll back this governance update, use `git revert`
for the governing commit so `pyproject.toml`, constraints, and `requirements.lock` move together;
never move, delete, or recreate existing tags or historical research artifacts.

## Repository map

```text
src/quant_portfolio/
├── allocator.py           # multi-book allocation and optional factor tilt
├── cross_asset.py          # causal cross-asset targets -> QExec OrderIntent suggestions
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
