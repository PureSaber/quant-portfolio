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
    feasibility_seed=None,
    odd_lot_sell_assets=(),
    max_backtracks=8,
):
    """quote_trade(asset, delta) returns exact (cash change, total friction).

    It is supplied by the execution adapter so rounding and fee rules have one
    owner. Residuals must be independently computed in post-fee NAV weights.
    non_cash_assets is independently evidenced account value (e.g. unpaid
    dividends): it contributes to NAV, but can never fund a purchase.
    Purchases use board lots; explicitly opted-in assets may sell their entire
    odd remainder in one order. No unobserved future market status is used.
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
    odd_lot_sell_assets = set(odd_lot_sell_assets)
    if odd_lot_sell_assets - set(assets):
        raise ValueError("odd-lot policy contains an unknown asset")
    if type(max_backtracks) is not int or max_backtracks < 0:
        raise ValueError("max_backtracks must be a non-negative integer")
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
        if asset in odd_lot_sell_assets:
            # A sell may dispose of the entire odd remainder once, never buy it.
            remainder = quantities[i] % steps[i]
            if sell[i] >= remainder:
                odd_sale = remainder + ((sell[i] - remainder) // steps[i]) * steps[i]
                minimum[i] = min(minimum[i], quantities[i] - odd_sale)

    def legal(i, value):
        delta = int(value - quantities[i])
        if not minimum[i] <= value <= maximum[i]:
            return False
        if delta % steps[i] == 0:
            return True
        return (
            assets[i] in odd_lot_sell_assets
            and delta < 0
            and (-delta) % steps[i] == quantities[i] % steps[i]
        )

    for i, asset in enumerate(assets):
        if asset in fixed_quantities:
            fixed = int(fixed_quantities[asset])
            if not legal(i, fixed):
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
            return (float("inf"), float("inf"), float("inf")), None
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
        if any(not legal(i, q[i]) for i in range(n)):
            bound_error = max(bound_error, 1.0)
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
        # A squared violation merit allows fixing several equally binding asset
        # caps one by one. Max-residual-only descent stalls on those plateaus.
        energy = sum(max(float(v), 0) ** 2 for v in violations.values()) + (
            max(-float(balance / nav), 0) ** 2
            + max(float(invested / nav) - max_invested, 0) ** 2
            + bound_error**2
        )
        score = (
            (max(energy, 1e-30), residual, objective) if residual > 1e-7 else (0.0, 0.0, objective)
        )
        return score, (w, balance, costs, violations)

    def rounded(point):
        desired = np.array([int(nav * Decimal(str(max(point[i], 0))) / marks[i]) for i in range(n)])
        floor = quantities + np.floor_divide(desired - quantities, steps) * steps
        floor = np.minimum(np.maximum(floor, minimum), maximum)
        for i in range(n):
            if assets[i] in odd_lot_sell_assets and desired[i] == 0 and legal(i, 0):
                floor[i] = 0
        return floor

    floor = rounded(target)
    best = None
    attempts = []

    def starts():
        yield "rounded_strategy", floor, {}
        if feasibility_seed is not None:
            remaining = timeout_seconds - (perf_counter() - started)
            if remaining > 0:
                try:
                    point, metadata = feasibility_seed(remaining)
                    if not isinstance(point, pd.Series) or set(point.index) != set(assets):
                        raise ValueError("feasibility seed must cover executable assets")
                    point = point.reindex(assets).to_numpy(dtype=float)
                    if not np.isfinite(point).all() or (point < -1e-7).any():
                        raise ValueError("invalid feasibility seed")
                    yield "minimum_risk_feasibility", rounded(point), metadata
                except OptimizationError as exc:
                    attempts.append(
                        {
                            "seed_kind": "minimum_risk_feasibility",
                            "status": exc.status,
                            "diagnostics": exc.diagnostics,
                        }
                    )
        yield "current_account", np.minimum(np.maximum(quantities, minimum), maximum), {}

    mark_weights = np.array([float(p / nav) for p in marks])
    seed_count = 3 if feasibility_seed is not None else 2
    for seed_index, (seed_name, seed, seed_metadata) in enumerate(starts()):
        seed_started = perf_counter()
        actions = []
        q = seed.copy()
        score, details = evaluate(q)
        deadline = started + timeout_seconds * 0.98
        # Reserve a fair share for every remaining start, including the actual
        # account. A risk seed must not starve that fallback of all search time.
        seed_deadline = min(
            deadline,
            perf_counter() + max(0, deadline - perf_counter()) / (seed_count - seed_index),
        )
        seen = set()
        frontier = []
        backtracks = 0
        termination = "iteration_limit"
        for iteration in range(max_iterations):
            if perf_counter() > seed_deadline:
                termination = "seed_time_budget"
                break
            key = tuple(q)
            if key in seen:
                termination = "repeated_state"
                break
            seen.add(key)
            winner, winner_score, winner_details, move = q, score, details, None

            # Rank moves by pre-fee distance, then verify the complete account
            # and risk constraints. Ranking is a heuristic, never a feasibility
            # test. First improvement avoids an exhaustive O(n^2) exact ledger
            # evaluation on every iteration. Ties use sorted asset identifiers.
            def candidates(q, score=score, backtracks=backtracks, details=details):
                # Deal with the actual post-fee violations before ranking by
                # distance to the strategy target. Per-asset diagnostics carry
                # arbitrary caps; no 5% constant is embedded in the repair.
                if score[0] > 0 and details is not None:
                    _, _, costs_now, violations = details
                    unit = np.array([float(p / (nav - costs_now)) for p in marks])
                    simultaneous = q.copy()
                    moves = []
                    for i, asset in enumerate(assets):
                        excess = violations.get("asset_upper:" + asset, 0.0)
                        shortage = violations.get("asset_lower:" + asset, 0.0)
                        if excess <= 1e-7 and shortage <= 1e-7:
                            continue
                        direction = -1 if excess > 1e-7 else 1
                        amount = max(excess, shortage)
                        count = max(1, int(np.ceil(amount / (unit[i] * steps[i]))))
                        for extra in (0, 1):
                            delta = direction * (count + extra) * steps[i]
                            if legal(i, q[i] + delta):
                                candidate = q.copy()
                                candidate[i] += delta
                                yield candidate, [(asset, int(delta))]
                        delta = direction * count * steps[i]
                        if legal(i, q[i] + delta):
                            simultaneous[i] += delta
                            moves.append((asset, int(delta)))
                    if moves:
                        yield simultaneous, moves
                        # A binding TE/turnover limit can make an isolated cap
                        # reduction worse. Try a compensating purchase together
                        # with it, including unequal lot sizes/notionals.
                        for i, asset in enumerate(assets):
                            if simultaneous[i] >= q[i]:
                                continue
                            released = (q[i] - simultaneous[i]) * unit[i]
                            for j in range(n):
                                if i == j or simultaneous[j] != q[j]:
                                    continue
                                count = max(1, int(released / (unit[j] * steps[j])))
                                for multiple in sorted({1, count, count + 1}):
                                    delta = multiple * steps[j]
                                    if legal(j, simultaneous[j] + delta):
                                        candidate = simultaneous.copy()
                                        candidate[j] += delta
                                        yield candidate, moves + [(assets[j], int(delta))]
                distance = q * mark_weights - target
                changes = []
                for i in range(n):
                    for direction in (-1, 1):
                        delta = direction * steps[i]
                        if legal(i, q[i] + delta):
                            dw = delta * mark_weights[i]
                            changes.append((2 * distance[i] * dw + dw * dw, i, direction))
                ranked = sorted(changes)
                for _, i, direction in ranked:
                    candidate = q.copy()
                    candidate[i] += direction * steps[i]
                    yield candidate, [(assets[i], direction * int(steps[i]))]
                # Sell the remaining odd shares as one order, including a full
                # exit. Ordinary purchases remain whole board lots.
                for i in range(n):
                    for value in (q[i] - q[i] % steps[i], 0):
                        if value != q[i] and legal(i, value):
                            candidate = q.copy()
                            candidate[i] = value
                            yield candidate, [(assets[i], int(value - q[i]))]
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
                # Larger legs and unequal exchanges cross local single-lot
                # barriers. Exact account evaluation still decides acceptance.
                if score[0] > 0:
                    for _, i, direction in ranked[:64]:
                        for multiple in (2, 4, 8):
                            delta = direction * steps[i] * multiple
                            if legal(i, q[i] + delta):
                                candidate = q.copy()
                                candidate[i] += delta
                                yield candidate, [(assets[i], int(delta))]
                    width = min(max_exchange_candidates * (1 + backtracks), 2048)
                    for _, i, j in sorted(exchanges)[:width]:
                        for left, right in ((1, 2), (2, 1), (2, 2), (1, 4), (4, 1)):
                            if legal(i, q[i] - left * steps[i]) and legal(
                                j, q[j] + right * steps[j]
                            ):
                                candidate = q.copy()
                                candidate[i] -= left * steps[i]
                                candidate[j] += right * steps[j]
                                yield (
                                    candidate,
                                    [
                                        (assets[i], -int(left * steps[i])),
                                        (assets[j], int(right * steps[j])),
                                    ],
                                )

            candidate_count = 0
            for candidate, changes in candidates(q):
                if perf_counter() > seed_deadline:
                    break
                if tuple(candidate) in seen:
                    continue
                candidate_score, candidate_details = evaluate(candidate)
                candidate_count += 1
                if candidate_score < winner_score:
                    winner, winner_score, winner_details, move = (
                        candidate,
                        candidate_score,
                        candidate_details,
                        changes,
                    )
                    # During infeasible search, compare a bounded batch instead
                    # of chasing the first tiny fee/merit improvement. Once
                    # feasible, retain it immediately and refine only within
                    # the feasible set toward the original continuous target.
                    if candidate_score[0] == 0 or score[0] == 0:
                        break
                if score[0] > 0 and backtracks < max_backtracks and tuple(candidate) not in seen:
                    frontier.append(
                        (candidate_score, tuple(candidate), candidate_details, actions + [changes])
                    )
                    frontier.sort(key=lambda item: (item[0], item[1]))
                    del frontier[max_backtracks:]
                if move is not None and candidate_count >= max(64, 2 * n):
                    break
            if move is None:
                while frontier and tuple(frontier[0][1]) in seen:
                    frontier.pop(0)
                if score[0] > 0 and frontier and backtracks < max_backtracks:
                    score, point, details, actions = frontier.pop(0)
                    q = np.asarray(point, dtype=np.int64)
                    backtracks += 1
                    continue
                termination = "local_search_exhausted" if score[0] > 0 else "feasible_local_optimum"
                break
            q, score, details = winner, winner_score, winner_details
            actions.append(move)
        if score[0] == 0 and (best is None or score < best[0]):
            best = (score, q.copy(), details, list(actions))
        attempts.append(
            {
                "seed_kind": seed_name,
                "seed": seed.tolist(),
                "actions": actions,
                "residual": score[1],
                "violation_energy": score[0],
                "constraint_residuals": None if details is None else details[3],
                "final_quantities": q.tolist(),
                "termination": termination,
                "backtracks": backtracks,
                "seed_diagnostics": seed_metadata,
                "seconds": perf_counter() - seed_started,
            }
        )
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
            {
                "stage": "integer_repair",
                "asset_order": assets,
                "repair_attempts": attempts,
                "exact_candidate_evaluations": evaluations,
                "seconds": perf_counter() - started,
                "integer_infeasibility_proven": False,
            },
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
            "search_method": "constraint_directed_batched_repair_v2",
            "max_exchange_candidates": max_exchange_candidates,
            "exact_candidate_evaluations": evaluations,
            "repair_actions": actions,
            "repair_attempts": attempts,
            "asset_order": assets,
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
