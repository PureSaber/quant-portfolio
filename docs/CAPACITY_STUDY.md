# Capital and cost sensitivity

```bash
python -m quant_portfolio.capacity_study scan --input examples/capacity_study.json --out artifacts/capacity.json
python -m quant_portfolio.capacity_study calibrate --input examples/cost_calibration.json --out artifacts/calibration.json
```

`evaluate_capacity` and `calibrate_costs` are the equivalent pure Python APIs. Output
files must be new. Both retain exact input hashes and explicit data nature. They add no
dependencies and do not change optimization, execution, market data or release locks.

## Scan contract and formula

`quant-portfolio.capacity-study/v1` requires base currency, as-of time, `liquidity` rows
with asset identity, positive `adv_base`, daily volatility, linear cost rate, observation
time, known-at time and source reference. `max_liquidity_age_days` is explicitly enforced.
The holdings and trade dictionaries must cover exactly the liquidity universe. Long and
short notionals use absolute values for costs and participation. No missing ADV is set to zero.

The grid spans `capital`, `turnover_scales`, `liquidity_scales`, `impact_coefficients`, and
`volatility_scales`. Every row uses the existing `square_root_impact_cost` and
`estimate_capacity` functions. For absolute order size Q split equally over D days:

`linear cost = Q*c`; `impact cost = Q*Y*sigma*sqrt(Q/(D*ADV))`.

Daily participation is `Q/(D*ADV)`. Thus quadrupling capital multiplies linear cost by four
and square-root cost by eight, holding everything else fixed. Tests check these scalings
and direct scalar calculations independently of the called cost helper.

Holding capacity is `min(ADV_i*max_participation*liquidation_days/abs(holding_weight_i))`.
Trade capacity replaces liquidation days with execution days and holding weights with
scaled trade weights. These are distinct: a low-turnover rebalance may fit market liquidity
while the full book cannot liquidate within its promised horizon. Zero trades/holdings
report no finite capacity (`null`) and zero costs, not fictitious infinite numeric values.
The scan reports cost-budget, trade participation and holding-capacity breaches separately.
It does not optimize a feasible target or estimate future profitability.

## Calibration and identifiability

`quant-portfolio.cost-calibration/v1` consumes already aggregated **parent order groups**,
not isolated fills. Each row supplies order identity, accepted/last-fill/publication
times, reference and reference-availability times, known-at liquidity, notional, ADV,
volatility, signed adverse execution rate and source reference. The rate is a signed
execution-price diagnostic; positive means adverse and negative means improvement.
Fees should not be silently merged into this rate when interpreting its intercept.

Training is restricted to orders whose entire execution and resulting observation are
known by `training_end`. Orders crossing the cutoff, in the gap, or unavailable at
evaluation are excluded explicitly. Holdout starts after training; it never changes the
fit. The two-parameter weighted least-squares model is
`rate = intercept + coefficient * sigma*sqrt(order_notional/ADV)`, with notional weights.
It is compared with a training-only weighted constant-rate baseline on chronological
holdout using signed error and absolute error in basis points. At least three train and
three holdout order groups and varying training features are required. Insufficient or
rank-deficient samples yield `unavailable`, never a calibration pass.

Coefficients remain unconstrained predictive diagnostics. Negative coefficients are
reported and flagged as incompatible with a nonnegative cost model rather than clipped
into a misleading calibrated curve. Parent groups still need not be statistically
independent. Spread, price drift, trade selection and timing confound the square-root
term, so predictive fit alone does not identify causal impact. No model is auto-applied
to an allocator or account.

## Public evidence workflow

`scripts/validate_public_capacity.py` consumes raw public Coinbase candle JSON and a
`provenance.json` with hashes, retrieval timestamp and exact URLs; it has no network calls.
The two series are BTC-USD and ETH-USD at one-day granularity, requested for
2025-01-01 through 2025-03-01. It filters the documented possibly-extra boundary buckets,
requires all 59 daily buckets through February 28, checks duplicate keys and OHLCV
validity, and independently recomputes ADV proxies and sample volatility with scalar
Python arithmetic. Reproduce with:

```bash
python scripts/validate_public_capacity.py --input-dir /path/to/raw --out-dir /path/to/new-evidence
```

[`Get product candles`](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles)
defines bucket start timestamps and warns that historical buckets can be incomplete.
The download timestamp is the actual `known_at`. A retrospective historical sample is
not a historical publication archive, and must not be used as available at 2025 decision
times. `close*volume` is a venue notional proxy; `low*volume` and `high*volume` provide
bounds rather than falsely identifying exact dollar ADV. The historical scenario uses
an explicitly generous age limit and makes no claim about current market capacity.

The sample supports observed-liquidity/volatility sensitivity. It has no parent-order
execution/reference-price facts, so **real execution-cost calibration remains unavailable**.
The bundled cost-calibration input is synthetic and only checks software and formula
recovery. Neither sample validates exchange liquidation rules or live account behavior.
