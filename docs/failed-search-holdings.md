# Failed searches and retained holdings

`audit_retained_holdings` evaluates the current held weights against the failed
problem's original budget, bounds, factor/group exposures, covariance and tracking
error constraints. Cash must be nonnegative and tradability known. The current
weights must match the serialized problem; absent, stale or malformed evidence
cannot certify a hold. No second optimizer or changed objective is substituted.

`hold_permitted` means that retaining the existing portfolio passes those checks.
It is never a published target or a successful optimization. The equity replay also
checks its current account-level limits, cancels outstanding orders after a search
failure and latches an unassessable/noncompliant hold. That latch cannot become an
automatic liquidation on a later risk event.

`publishing_summary` includes every decision in its denominator and reports target
publication rate, cause counts and the longest consecutive gap. An explicitly
published all-cash target counts as a publication. No fixed success-rate threshold
or investment-performance conclusion is inferred.
