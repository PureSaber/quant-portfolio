# Constraint-directed integer repair

Continuous optimization, investment budget, turnover, tracking error and exact
execution fees are unchanged. This revision changes only the local integer
search. It does not prove integer optimality or infeasibility.

The repair prioritizes actual post-fee per-asset bound violations, tries joint
cap reductions and unequal-lot compensating purchases, and compares a bounded
batch of candidates while infeasible. Each remaining start receives reserved
time. Once feasible, refinement minimizes distance to the original strategy
target without leaving the feasible set. Exact cash conservation and final
residual acceptance at 1e-7 remain mandatory.

`repair_attempts` includes each seed's elapsed time, final residuals, termination
and actions. Budget shortfall is still disclosed; feasibility is not a promise
that every lot or budget deviation is globally minimal. Unsuccessful search
remains `repair_failed`, never an integer-infeasibility certificate.

The compressed fixture contains ten public historical numerical problems from
the recovered 200-company study. It retains continuous targets and actual
account constraints, but no old integer search endpoint is supplied as a seed.
Five independently demonstrated feasible counterexamples are mandatory CI
regressions. Other unresolved cases remain in the full offline replay receipt.

The frozen-account experiment recovered 19 of 41 prior failures, including
8 of the latest ten and all five proven-feasible counterexamples. Independent
exact fee/cash/quantity and full-risk checks accepted all 19; the maximum
normalized residual was 7.205636508994528e-8. The remaining 22 are bounded
search failures, not proofs of infeasibility. Successful targets retain
0.06–1.45 percentage points of budget shortfall; this is not certified minimal.

The fixture helper deducts the archived continuous solve's `total_seconds`
from the original 30-second rebalance budget before searching. Reconciliation
of all 41 archived continuous solves and measured new repairs found a maximum
combined recorded time of 29.4451967 seconds. This is a stage-time check for
the frozen cases, not a claim about unrecorded full wall clocks of every
historical failure or a guarantee of the same search outcome on every host.

Full own-account paths are evaluated separately from these fixed-account
counterfactuals, with method, model, date, execution stop and independent ledger
receipts in the related quant-risk-monitor robustness research. A published
repair changes later holdings, so 19 recovered fixed cases must not be added
to the number of published targets in an entire replay.

Rollback: use the previous package commit in a new frozen environment and a new
run directory. Do not overwrite either version's account or research artifacts.
Merging this change does not deploy a simulator or alter a frozen strategy.

## Completed account-path acceptance (2026-10-22)

The completed 2026-10-19 campaign is now exported in
[the acceptance receipt](lot-repair-acceptance-20261022/acceptance.json),
[all 41 fixed cases](lot-repair-acceptance-20261022/all-41-fixed-cases.csv),
[all 62 scenarios](lot-repair-acceptance-20261022/scenarios.csv), and
[the complete decision-date comparison](lot-repair-acceptance-20261022/decisions.csv).
Thirteen frozen input files were rehashed against the original receipts on
2026-10-22. The tested producer's runtime source is identical to PR head fc7e94c.

Search-only published targets increased from 1519 to 1607 over the common
decision dates. The full paths retain 25 integer-search failures; three
previously published decisions become failures after holdings change. The
41 fixed-case counterfactuals remain a separate population (19 recovered,
22 unresolved). Six account endpoints differ; full-horizon accounts decrease
from 32 to 27 because changed holdings reach execution-input protection earlier.
Do not compare unequal-horizon returns as strategy improvement.

The independent audit covers all 72 actual search/held replay attempts and
3234 published targets, with balanced exact journals and maximum residual
9.665701646011104e-8, below the unchanged 1e-7 threshold. The receipt explicitly
labels that combined audit scope. Search-only published rebalance time peaks
at 16.4967 seconds (zero over 30 seconds); this is not a claim about unrecorded
historical failure wall clocks. The 26 certified unchanged scenarios are
retained in the scenario table, not presented as new replay attempts.

The held-exposure variant's additional 20 targets and risk-model conclusions
are excluded from this software acceptance. Risk Monitor #37 remains deferred.
