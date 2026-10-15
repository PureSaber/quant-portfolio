"""Deterministic local repair of continuous targets; never an integer optimality claim."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from time import perf_counter

import numpy as np
import pandas as pd

from quant_portfolio.solver import OptimizationError


@dataclass(frozen=True)
class ExecutableTarget:
    quantities: dict[str, int]
    weights: pd.Series
    cash: Decimal
    costs: Decimal
    diagnostics: dict


def repair_integer_target(
    weights,
    *,
    current,
    prices,
    lots,
    cash,
    nav,
    quote_trade,
    constraint_residuals,
    max_invested,
    non_cash_assets=Decimal(0),
    fixed_quantities=None,
    max_buy=None,
    max_sell=None,
    timeout_seconds=30,
    max_iterations=2000,
    max_exchange_candidates=256,
):
    """quote_trade(asset, delta) returns exact (cash change, total friction).

    It is supplied by the execution adapter so rounding and fee rules have one
    owner. Residuals must be independently computed in post-fee NAV weights.
    non_cash_assets is independently evidenced account value (e.g. unpaid
    dividends): it contributes to NAV, but can never fund a purchase.
    Current quantities can contain odd-lot remainders; only board-lot trades
    are proposed. No unobserved future market status is used.
    """
    started = perf_counter()
    if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("repair timeout must be positive and finite")
    if not np.isfinite(max_invested) or not 0 <= max_invested <= 1:
        raise ValueError("max_invested must be in [0, 1]")
    if type(max_iterations) is not int or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if type(max_exchange_candidates) is not int or max_exchange_candidates <= 0:
        raise ValueError("max_exchange_candidates must be a positive integer")
    assets = sorted(set(weights.index) | set(current))
    if not weights.index.is_unique or not np.isfinite(weights).all() or (weights < -1e-7).any():
        raise ValueError("invalid continuous weights")
    nav, cash = Decimal(str(nav)), Decimal(str(cash))
    non_cash_assets = Decimal(str(non_cash_assets))
    if not non_cash_assets.is_finite() or non_cash_assets < 0:
        raise ValueError("invalid non-cash account assets")
    if not nav.is_finite() or nav <= 0 or not cash.is_finite() or cash < 0:
        raise ValueError("invalid account cash or NAV")
    n = len(assets)
    if n == 0:
        raise ValueError("at least one executable asset is required")
    fixed_quantities = fixed_quantities or {}
    if set(fixed_quantities) - set(assets):
        raise ValueError("fixed quantities contain an unknown asset")
    for mapping in (current, lots, max_buy or {}, max_sell or {}, fixed_quantities):
        if any(isinstance(v, bool) or not np.isfinite(v) or int(v) != v for v in mapping.values()):
            raise ValueError("quantities and board lots must be integers")
    quantities = np.array([current.get(a, 0) for a in assets], dtype=np.int64)
    steps = np.array([lots[a] for a in assets], dtype=np.int64)
    if (quantities < 0).any() or (steps <= 0).any():
        raise ValueError("quantities and board lots must be non-negative integers")
    marks = [Decimal(str(prices[a])) for a in assets]
    if any(not p.is_finite() or p <= 0 for p in marks):
        raise ValueError("positive exact prices are required for every asset")
    target = weights.reindex(assets, fill_value=0).to_numpy()
    sell = np.array(
        [
            min(quantities[i], (max_sell or {}).get(a, int(quantities[i])))
            for i, a in enumerate(assets)
        ]
    )
    buy = np.array([(max_buy or {}).get(a, int(nav / marks[i])) for i, a in enumerate(assets)])
    if (sell < 0).any() or (buy < 0).any():
        raise ValueError("invalid tradable quantity")
    minimum = quantities - (sell // steps) * steps
    maximum = quantities + (buy // steps) * steps
    for i, asset in enumerate(assets):
        if asset in fixed_quantities:
            fixed = int(fixed_quantities[asset])
            if not minimum[i] <= fixed <= maximum[i] or (fixed - quantities[i]) % steps[i]:
                raise ValueError("fixed target violates known quantity or board-lot constraints")
            minimum[i] = maximum[i] = fixed
    quote_cache = {}
    evaluations = 0

    def evaluate(q):
        nonlocal evaluations
        evaluations += 1
        balance, costs = cash, Decimal(0)
        for i, asset in enumerate(assets):
            delta = int(q[i] - quantities[i])
            if delta:
                key = (asset, delta)
                if key not in quote_cache:
                    change, friction = quote_trade(asset, delta)
                    if not isinstance(change, Decimal) or not isinstance(friction, Decimal):
                        raise TypeError("execution quote must use exact Decimal amounts")
                    if not change.is_finite() or not friction.is_finite() or friction < 0:
                        raise ValueError("invalid execution quote")
                    quote_cache[key] = change, friction
                change, friction = quote_cache[key]
                balance += change
                costs += friction
        post_nav = nav - costs
        if post_nav <= 0:
            return (float("inf"), float("inf")), None
        invested = sum((Decimal(int(q[i])) * marks[i] for i in range(n)), Decimal(0))
        # Ledger conservation is exact at the supplied marks and quote precision.
        if balance + invested + non_cash_assets != post_nav:
            raise ValueError("execution quote and account marks do not reconcile")
        w = pd.Series(
            [float(Decimal(int(q[i])) * marks[i] / post_nav) for i in range(n)], index=assets
        )
        violations = constraint_residuals(w)
        if not np.isfinite(list(violations.values())).all():
            raise ValueError("constraint residuals must be finite")
        bound_error = max(
            float(np.max(minimum - q, initial=0)), float(np.max(q - maximum, initial=0))
        )
        residual = max(
            [
                0.0,
                max(1e-6, -float(balance / nav)) if balance < 0 else 0.0,
                float(invested / nav) - max_invested,
                bound_error,
                *(max(float(v), 0) for v in violations.values()),
            ]
        )
        objective = float(np.sum((w.to_numpy() - target) ** 2))
        return (residual if residual > 1e-7 else 0.0, objective), (w, balance, costs, violations)

    desired = np.array([int(nav * Decimal(str(max(target[i], 0))) / marks[i]) for i in range(n)])
    # Preserve the account's odd-lot remainder and enforce known quantity limits.
    floor = quantities + np.floor_divide(desired - quantities, steps) * steps
    floor = np.minimum(np.maximum(floor, minimum), maximum)
    starts = [floor, np.minimum(np.maximum(quantities, minimum), maximum)]
    best = None
    attempts = []
    mark_weights = np.array([float(p / nav) for p in marks])
    for seed in starts:
        actions = []
        q = seed.copy()
        score, details = evaluate(q)
        seen = set()
        for iteration in range(max_iterations):
            if perf_counter() - started > timeout_seconds:
                break
            key = tuple(q)
            if key in seen:
                break
            seen.add(key)
            winner, winner_score, winner_details, move = q, score, details, None

            # Rank moves by pre-fee distance, then verify the complete account
            # and risk constraints. Ranking is a heuristic, never a feasibility
            # test. First improvement avoids an exhaustive O(n^2) exact ledger
            # evaluation on every iteration. Ties use sorted asset identifiers.
            def candidates(q):
                distance = q * mark_weights - target
                changes = []
                for i in range(n):
                    for direction in (-1, 1):
                        delta = direction * steps[i]
                        if minimum[i] <= q[i] + delta <= maximum[i]:
                            dw = delta * mark_weights[i]
                            changes.append((2 * distance[i] * dw + dw * dw, i, direction))
                ranked = sorted(changes)
                for _, i, direction in ranked:
                    candidate = q.copy()
                    candidate[i] += direction * steps[i]
                    yield candidate, [(assets[i], direction * int(steps[i]))]
                exchanges = []
                # Pair scores are additive. The best K valid pairs are among
                # the best K+1 legs on either side (one same-asset exclusion).
                # Bound temporary memory independently of the asset universe.
                reductions = [(v, i) for v, i, d in ranked if d == -1][
                    : max_exchange_candidates + 1
                ]
                additions = [(v, i) for v, i, d in ranked if d == 1][: max_exchange_candidates + 1]
                for reduction, i in reductions:
                    for addition, j in additions:
                        if i == j:
                            continue
                        exchanges.append((reduction + addition, i, j))
                for _, i, j in sorted(exchanges)[:max_exchange_candidates]:
                    candidate = q.copy()
                    candidate[i] -= steps[i]
                    candidate[j] += steps[j]
                    yield candidate, [(assets[i], -int(steps[i])), (assets[j], int(steps[j]))]

            for candidate, changes in candidates(q):
                if perf_counter() - started > timeout_seconds:
                    break
                candidate_score, candidate_details = evaluate(candidate)
                if candidate_score < winner_score:
                    winner, winner_score, winner_details, move = (
                        candidate,
                        candidate_score,
                        candidate_details,
                        changes,
                    )
                    break
            if move is None:
                break
            q, score, details = winner, winner_score, winner_details
            actions.append(move)
        if score[0] == 0 and (best is None or score < best[0]):
            best = (score, q.copy(), details, list(actions))
        attempts.append({"seed": seed.tolist(), "actions": actions, "residual": score[0]})
        # The current-position seed is a fallback for failed repair, not a
        # second optimization from an unrelated allocation after success.
        if best is not None:
            break
    if perf_counter() - started > timeout_seconds:
        raise OptimizationError(
            "timeout", "integer repair deadline exceeded", {"repair_attempts": attempts}
        )
    if best is None:
        raise OptimizationError(
            "repair_failed",
            "local repair found no verified executable target",
            {"repair_attempts": attempts, "seconds": perf_counter() - started},
        )
    score, q, details, actions = best
    final_score, details = evaluate(q)
    w, balance, costs, residuals = details
    if final_score[0] != 0 or balance < 0:
        raise OptimizationError("repair_failed", "final account/risk verification failed")
    return ExecutableTarget(
        {asset: int(q[i]) for i, asset in enumerate(assets) if q[i]},
        w,
        balance,
        costs,
        {
            "status": "repaired",
            "integer_global_optimum": False,
            "search_method": "ranked_first_improvement",
            "max_exchange_candidates": max_exchange_candidates,
            "exact_candidate_evaluations": evaluations,
            "repair_actions": actions,
            "repair_attempts": attempts,
            "constraint_residuals": residuals,
            "seconds": perf_counter() - started,
            "cash": str(balance),
            "non_cash_assets": str(non_cash_assets),
            "costs": str(costs),
            "post_fee_nav": str(nav - costs),
            "budget_shortfall": float(max_invested)
            - float(sum(Decimal(int(q[i])) * marks[i] for i in range(n)) / nav),
            "shortfall_reasons": ["fees", "board_lot_granularity", "known_quantity_constraints"],
            "integer_weights": w.to_dict(),
        },
    )
