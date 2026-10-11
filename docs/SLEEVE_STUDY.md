# Point-in-time multi-sleeve study

Run from an installed checkout:

```bash
python -m quant_portfolio.portfolio_study --input examples/sleeve_study.json --out artifacts/sleeve-study.json
```

Output must not exist. The JSON includes the exact input SHA-256, source references,
per-decision training periods and availability audit, allocations, costs, drifted capital,
sample correlations, Euler variance contributions and diversification ratios. No account
is created or changed. `evaluate_study(dict)` is the equivalent Python API.

## Input boundary

`quant-portfolio.sleeve-study/v1` accepts asset or strategy sleeves with `id`, `currency`,
`source_ref`, and `return_basis=net_simple_total_return`. Native producers supply returns
after their native trading/financing costs and including distributions. NAV indexes,
gross returns, price-only returns and money-weighted returns must not be relabeled as
this contract. Convert them in the producer using its actual accounting facts first.

Each observation supplies `sleeve`, timezone-aware `start`, `end`, `known_at`, `net_return`,
`fx_start`, `fx_end`, `fx_start_known_at`, and `fx_end_known_at`. FX is base currency per
unit local currency at the exact return boundaries. For base-currency sleeves it is one.
Effective availability is the latest of return and FX publication times. Separate return
vintages can revise a period; the latest visible vintage is used at each decision.
The consumer validates timestamps and declared provenance, but does not authenticate
historical publication claims. Data nature is explicitly `synthetic`, `retrospective`,
`historical_pit` or `independent_forward`; it is never upgraded by a successful run.

Training intersects exact start/end pairs across all sleeves, with no zero filling,
interpolation, frequency conversion or pairwise covariance deletion. A weekly fund and
daily strategy can be compared only after the daily strategy producer compounds into
the same weekly boundaries. Within-sleeve overlapping intervals are rejected. Evaluation
requires every sleeve in every explicitly requested contiguous period to be mature at
`evaluation_as_of`. Missing evaluation data fails the entire comparison.

Local and base returns obey `1+r_base=(1+r_local)*FX_end/FX_start`; the local/FX cross term
is retained. Forecast history must end and be available by the decision's period start.
The realized return vintage is selected at `evaluation_as_of`, independently of training.

## Dynamic comparison and financial semantics

Candidate methods are `risk_parity`, `hrp`, `inverse_vol`, and `equal`. They call the existing
`research_allocation_weights` implementation. Equal and inverse-volatility benchmarks
share the candidate's exact evaluation periods, common training window, initial holdings,
invested budget, per-sleeve cap, turnover limit, overlay cost rates and cash returns.
Each evolves its own actual holdings; infeasible rebalances fail instead of relaxing
constraints. This research wrapper does not claim common venue, margin or nonlinear
execution constraints; those remain the cross-asset optimizer's boundary.

Each evaluation period declares realized base-currency `cash_return`. With target weights
w, drifted current weights h and linear overlay rates c:

* turnover is `sum(abs(w-h))` (two-way sleeve notional, excluding the cash leg);
* cost is `k=sum(c*abs(w-h))`, additionally to costs already inside native sleeve returns;
* gross return is `g=w*r_base+(1-sum(w))*r_cash`;
* capital multiplies by `(1-k)*(1+g)` and next held weights are `w*(1+r_base)/(1+g)`.

This is the existing simulator's proportional pre-return overlay-cost convention, not
a claim of a self-financing fill ledger. Exact cash/security execution belongs to native
accounts. No reallocation impact or redemption delay is inferred from sleeve NAV.

Risk diagnostics use unannualized sample covariance on the actual matched training
periods: `v=w'Σw`, component variance `w_i(Σw)_i`, and diversification ratio
`sum(abs(w_i)*sqrt(Σ_ii))/sqrt(v)`. A zero variance/correlation denominator is `null`,
not a passing diversification value. Under binding caps, risk-parity targets are
constrained by the existing solver; equal risk contribution need not survive those caps.

The equal-risk-contribution routine now uses coordinate descent on the convex objective
`x'Σx/2 - sum(log(x))/n`, then normalizes x. This fixes failures when a valid negative
correlation creates a negative intermediate marginal contribution. References:
[Spinu (2013)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2297383) and
[Griveau-Billion, Richard and Roncalli (2013)](https://arxiv.org/abs/1311.4057).
Tests independently check the two-asset solution `w1*sigma1=w2*sigma2`, a sample
covariance hand calculation, and a dollar-by-dollar cash/cost/drift replay.

The bundled case is synthetic software validation. It is not independent forward
performance, proof of historical PIT data, or a transaction-supported causal attribution.
