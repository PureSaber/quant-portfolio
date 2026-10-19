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

Rollback: use the previous package commit in a new frozen environment and a new
run directory. Do not overwrite either version's account or research artifacts.
Merging this change does not deploy a simulator or alter a frozen strategy.
