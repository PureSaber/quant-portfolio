# Failure diagnostics and bounded integer repair

`AllocationInputError` identifies known unavailable inputs. `OptimizationError`
keeps infeasible, numerical failure, timeout and repair failure separate. A
generic programming `ValueError` is not an unavailable-data result.

The complete allocation result and failures retain the numerical problem (dense
or factor risk), objective, current weights, actual constraints and continuous
candidate. The execution adapter adds exact account/fee/tradability inputs.
Feasibility diagnostics independently solve minimum TE with/without turnover and
minimum turnover at the TE cap. They are optimistic explanations, never order
targets. Numerical residuals accompany each successful diagnostic bound.

Integer repair uses the original continuous target, a constrained minimum-risk
seed and current holdings. A seed is not a substitute strategy or proof that the
cost-aware objective is optimal. Deterministic multi-lot moves, unequal exchanges
and bounded backtracking extend the local search. Every accepted output is
rechecked with exact quoted fees, available cash/shares, post-fee weights and
constraint residuals at most 1e-7. Odd-lot sales require the execution adapter's
explicit venue policy. Initial account preparation, continuous solve and repair
share the requested deadline (30 seconds by default).

Search exhaustion remains `repair_failed`. A feasible integer result is not a
global integer optimum. A timeout is not proof of infeasibility. The fixtures for
scenarios 41/46 on 2024-01-05 retain the original benchmark and investment limits.

Default exposure models, strategy investment budgets and frozen runs are not
changed. Restore the prior package lock for old-run reproduction; write reruns
under new identities.
