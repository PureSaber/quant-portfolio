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
`total_nav` uses the normalized strategy capital budgets at full calculation precision,
independently of `position_scale`. Each book reports `capital_weight`, the invested
`budget_weight`, and the reserve `cash_weight` released by scaling; the reserve is retained
in NAV and is not inserted as a security into `combined_weights`.
Strategy weights must be finite and nonnegative with a positive finite total;
`position_scale` must be finite and within [0, 1]. Boolean values are not numeric
configuration values. NAV observations must be finite and nonnegative, and holdings
weights must be finite. Invalid inputs fail before a status file is written; status JSON
rejects NaN and infinity.
The tilt preserves the invested budget after `position_scale`; zero factor weight leaves
holdings unchanged. Symbols without a score keep their original weights, and scores for
symbols outside the holdings are ignored. Redistribution stays within the scored sleeve's
existing budget. A tilt that removes every funded position fails explicitly.
Security identifiers are strings: CSV input preserves leading zeros and identifiers such as
`NA`; Parquet symbol columns must already contain strings. Blank identifiers are rejected.

## Cost-aware optimizer

For exact-period, multi-currency strategy/asset results, the
[PIT sleeve study](docs/SLEEVE_STUDY.md) adds causal dynamic allocation, same-constraint
benchmarks, cash/FX handling, correlation and diversification diagnostics. It reuses the
existing allocators and requires explicit return/publication semantics from producers.

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

The optimizer uses a shrinkage/PSD-repaired covariance and jointly enforces budget, asset
bounds, turnover, group caps, and optional linear factor-exposure bounds while charging linear
costs. Covariance estimation is `diagonal` (fixed shrinkage, default 0.2), `ledoit_wolf`,
`ewma`, `constant_correlation`, `oas`, or `random_matrix`.
`factor_model_covariance` builds `X F X' + diag(D)`. `benchmark_weights` switches the risk
term to active variance `(w - w_b)' Σ (w - w_b)`, and `factor_bound_reference="active"`
bounds `X'(w - w_b)` directly. `max_tracking_error` is a hard cap on that active volatility.
Square-root impact enters that same objective when ADV, volatility, an impact coefficient and
portfolio NAV are supplied; `square_root_impact_cost` remains available as a standalone
estimate. `weight_steps` then chooses integer lots around the continuous solution and may
leave residual cash. Infeasible intersections fail explicitly, and every returned portfolio
is rechecked against all configured constraints.

`optimize_max_sharpe` and `optimize_max_diversification` are separate ratio objectives on the
same long-only feasible set. `optimize_cvar` maximizes expected return minus historical CVaR.
`optimize_evar` uses entropic value at risk and `optimize_cdar` uses conditional drawdown at
risk. Those three path solvers accept `max_drawdown`, a cap on the drop in cumulative simple
return. `optimize_multiperiod` plans a finite sequence of long-only rebalances and charges
linear costs between dates. Each date jointly enforces asset bounds, group caps and turnover,
and every returned row is checked again. The first trade and its cost are measured from the
real `current_weights`, including any required correction of an initially out-of-bounds
holding. An insufficient turnover budget fails explicitly. The cross-asset solver is separate:
it projects leverage, margin,
venue and participation limits, then searches quantity steps instead of only rounding toward
zero after the fact.

Research recipes can call `validate_research_allocation` and
`research_allocation_weights` with `equal`, `inverse_vol`, `risk_parity`, `hrp`, `cvar`,
`evar`, `cdar`, `max_sharpe`, `max_diversification`, or `cost_aware`. `inverse_vol` ignores
correlation. `risk_parity` equalizes `w_i (Σw)_i`. `hrp` is hierarchical risk parity.
`cvar`, `evar` and `cdar` use the scenarios in the lookback window. `cost_aware` delegates
to the mean-variance optimizer. Its `expected_return_model` is `score` (the raw score is μ),
`ic_vol` (Grinold `IC * volatility * z-score`), or `black_litterman` (those scaled scores are
views around `risk_aversion * Σ w_market`). `covariance_estimator` on the covariance modes is
`diagonal`, `ledoit_wolf`, `factor`, `ewma`, `constant_correlation`, `oas`, or `random_matrix`.
All modes return a long-only sleeve whose sum equals the explicit invested limit and whose
names obey the same absolute position cap. `max_turnover` is measured from the real current
portfolio and includes selling holdings omitted from the new score set; a budget too small
for the required sleeve change fails explicitly. The `cost_aware` mode accepts the same
factor inputs plus a validated `covariance_override`. Factor bounds stay absolute unless
`factor_bound_reference` is `active`. Benchmark-relative inputs and square-root impact
require an invested limit of 1, because the sleeve solver is fully invested.
The `equal` mode also accepts absolute factor bounds. It projects its 1/N target
onto the joint budget, position, factor and turnover feasible set, including
liquidation outside the selected universe. Bounds remain fractions of total NAV,
so a cash sleeve does not rescale the declared economic exposure limits.
An optional covariance override is checked for finite values, symmetry and PSD
but does not change the equal-weight objective. Tracking-error checks remain the
caller's responsibility; the equal mode does not claim covariance optimization.
Missing model inputs, non-finite values, incomplete covariance estimates,
and solver non-convergence also fail closed.

## Cross-asset target API

`simulate_market` rebalances from a policy that sees only past returns, then charges linear
and optional square-root costs and lets weights drift. Its decision index must be unique,
strictly increasing and contain no missing values: use a `DatetimeIndex`, `PeriodIndex`, or
finite numeric step index. Convert date strings explicitly before calling; unordered input
is rejected before any policy call. `cpcv_splits`,
`deflated_sharpe_ratio` and `probability_of_backtest_overfitting` are the purged-split,
deflated-Sharpe and CSCV overfit diagnostics for those simulated returns.

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

Version `0.4.2` consumes only published annotated internal tags:

- `quant-data-kit v0.8.1` (`b5621379e1a15562371be31c03f354a6acf512e4`)
- `quant-execution v0.5.1` (`99ab9b1445d72164fa4e8c6d1ebde859b80dba1f`)

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
├── objectives.py           # Sharpe, diversification, EVaR, CDaR and lot search
├── evaluation.py           # causal simulator, CPCV, deflated Sharpe and PBO
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

Portfolio `status` selects factor observations at or before its reported `as_of`.
NAV rows are ordered by parsed timestamps. Date-only labels mean the end of that UTC day;
explicit timestamps are compared in UTC (naive timestamps are treated as UTC).
Factors may supply `available_at` to exclude late publications and revisions. Without that
column, `date` is assumed to be the availability date. Undated files must declare
`factor_scores.as_of` in the config. Ambiguous duplicate observations and invalid dates fail
validation; holdings without eligible scores retain their original allocation.

- The synthetic process is intentionally simple and is not calibrated to a real market.
- A positive synthetic backtest does not imply investability or future returns.
- Execution, liquidity, capacity, financing, taxes, and market-impact models are incomplete.
- This code is educational and is not investment advice.

See [Disclosure and data policy](docs/DISCLOSURE.md) and [Architecture](docs/ARCHITECTURE.md).
