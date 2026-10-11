# Continuous shared capital and paired revisions

Install this repository with its pinned QDK/QExec dependencies. The new commands
are exposed by the existing `quant-portfolio` CLI. The deterministic example is
entirely synthetic; it establishes accounting and revision plumbing only.

```bash
python scripts/run_continuous_demo.py --out /new/empty/path/capital-demo
```

This executes the real freeze, continuous-study and revision-study CLIs, pins both
market versions and result derivations in QDK's `ArtifactStore`, and writes
`acceptance.json`, `README.md`, both account reports, and a revision report. The
independent scalar expectation is +34 USD old PnL, +26 new PnL, and -8 difference.
No data, accounts, models, credentials or real orders are acquired or modified.

Run individual steps with your own explicitly declared research inputs:

```bash
quant-portfolio freeze-capital-plan --plan examples/continuous/plan.json --out plan-frozen.json
quant-portfolio continuous-study --plan plan-frozen.json --market examples/continuous/market-old.json --out old.json
quant-portfolio revision-study --request request.json --out revision.json
```

The draft `quant.capital-plan/v1` carries full QDK `InstrumentSpec` payloads,
instrument-to-sleeve ownership, opening currency allocations and FX, a fee-per-unit
cost policy, valuation policy, decisions, and optional explicit cash events.
Each decision fixes a notional target, signal observation, later execution
observation and settlement instant. It rounds the prior-observation notional to
quantity steps, then submits the target delta through QExec at the explicit later
price. It does not adapt or optimize the policy to evaluation outcomes. Missing
prices, unavailable observations, ambiguous order, insufficient funds, unsupported
assets or silently changed controls fail the run. Executions are declared research
fills, with explicit fees; this adapter does not claim order-book matching.

Market input `quant.capital-market/v1` has observations with unique IDs, UTC `at`,
`available_at <= at`, price mapping and FX mapping. Cash-event sequences occupy
0–99, observation marks 100–999, decisions 1000 upward. The QExec account retains
all ordered events, settlements, sleeves, journal postings and cash/margin state
continuously. See QExec `docs/SHARED_CAPITAL.md` for the exact event contract and
limitations. B's forecast replay may call that additive API with its own fills.

Freezing copies the controls and binds actual installed package Python source
hashes and runtime versions. Editing the plan, source or runtime requires a new
explicit freeze and cannot be reported as a data-only comparison. Freezing is
performed now: historical fixture dates do not confer historical/OOS provenance.
The adapter supports `synthetic/deterministic_fixture` or explicitly declared
retrospective studies, and exposes features/forecasts as unavailable.

For revision requests, publish old/new market JSON using `ArtifactStore.put_bytes`
and put their complete `ArtifactRef.to_dict()` objects in `old_input` and
`new_input`. `request.json` also contains absolute `store`, frozen `plan` path and
durable `owner`. `paired_study` checks the same frozen executable and controls,
recalculates both outputs, and registers exact dependencies. The report includes
transitive affected objects, signals/orders/history/NAV/return changes, unchanged
stage results, original references and explicit `intervention=data_revision`.
It never replaces earlier decisions. Dependency reachability is not causal proof.

Reports emit the common `quant.measurement/v1`. Return observations group all
same-instant events after respecting their explicit sequence; their return is the
product of each flow-adjusted event return. The timestamp grid includes opening
and final boundaries, is UTC and strictly increasing. Event returns are never
annualized. Cash is in base-currency major units, returns are decimal fractions,
and exact source, market, instrument, FX and economic-policy hashes are retained.
Old/new revisions intentionally have different input identities and therefore
must fail ordinary same-input economic comparison. The revision report is the
explicit intervention analysis.

Every CLI output uses exclusive creation. Keep frozen plans, raw references and
artifact-store pins with the reports. Re-running into a different new directory
is safe; overwriting past research is rejected. Roll back through Git revert while
preserving output history. No deployment or default-branch merge is performed.
