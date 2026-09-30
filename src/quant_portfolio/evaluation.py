"""Causal market simulation and backtest-overfitting diagnostics.

The simulator shows a policy only the rows strictly before the decision date,
then charges linear and optional square-root costs and lets weights drift with
the realized return. CPCV, the deflated Sharpe ratio, and the probability of
backtest overfitting are the López de Prado diagnostics that sit next to that
simulation. They do not change the optimizer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from quant_portfolio.methods import square_root_impact_penalty

Policy = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class SimulationResult:
    wealth: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    net_returns: pd.Series


@dataclass(frozen=True)
class DeflatedSharpe:
    sharpe: float
    benchmark_sharpe: float
    probability: float
    n_trials: int
    observations: int


@dataclass(frozen=True)
class OverfitResult:
    probability: float
    logits: tuple[float, ...]
    n_splits: int


def simulate_market(
    asset_returns: pd.DataFrame,
    policy: Policy,
    *,
    initial_weights: pd.Series | None = None,
    linear_costs: pd.Series | float = 0.0,
    initial_wealth: float = 1.0,
    impact_adv: pd.Series | None = None,
    impact_volatility: pd.Series | None = None,
    impact_coefficient: float = 0.0,
) -> SimulationResult:
    """Rebalance on a causal clock and apply costs before the period return."""

    returns = _return_frame(asset_returns)
    assets = list(returns.columns)
    if initial_weights is None:
        held = np.repeat(1.0 / len(assets), len(assets))
    else:
        held = _policy_weights(initial_weights, assets, name="initial_weights")
    costs = _cost_vector(linear_costs, assets)
    impact = _impact_inputs(assets, impact_adv, impact_volatility, impact_coefficient)
    if (
        isinstance(initial_wealth, bool)
        or not isinstance(initial_wealth, (int, float))
        or not np.isfinite(initial_wealth)
        or initial_wealth <= 0
    ):
        raise ValueError("initial_wealth must be positive and finite")
    wealth = float(initial_wealth)
    wealth_path = []
    weight_rows = []
    turnover_path = []
    cost_path = []
    net_path = []
    for time in range(len(returns)):
        history = returns.iloc[:time]
        target = _policy_weights(policy(history), assets, name="policy weights")
        trade = target - held
        cost = float(costs @ np.abs(trade))
        if impact is not None:
            adv, volatility, coefficient = impact
            impact_cost, _gradient = square_root_impact_penalty(
                trade,
                adv,
                volatility,
                impact_coefficient=coefficient,
                portfolio_nav=wealth,
            )
            cost += impact_cost
        if cost >= 1:
            raise ValueError("transaction costs consumed the portfolio")
        gross = float(target @ returns.iloc[time].to_numpy(dtype=float))
        if gross <= -1:
            raise ValueError("portfolio return wiped out the book")
        wealth *= 1.0 - cost
        wealth *= 1.0 + gross
        held = target * (1.0 + returns.iloc[time].to_numpy(dtype=float)) / (1.0 + gross)
        wealth_path.append(wealth)
        weight_rows.append(target)
        turnover_path.append(float(np.abs(trade).sum()))
        cost_path.append(cost)
        net_path.append((1.0 - cost) * (1.0 + gross) - 1.0)
    index = returns.index
    return SimulationResult(
        wealth=pd.Series(wealth_path, index=index, name="wealth"),
        weights=pd.DataFrame(weight_rows, index=index, columns=assets),
        turnover=pd.Series(turnover_path, index=index, name="turnover"),
        costs=pd.Series(cost_path, index=index, name="cost"),
        net_returns=pd.Series(net_path, index=index, name="net_return"),
    )


def cpcv_splits(
    n_observations: int,
    n_groups: int,
    n_test_groups: int,
    *,
    purge: int = 0,
    embargo: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Combinatorial purged splits over contiguous groups.

    Training rows that fall in the ``purge`` observations before a test block,
    or the ``embargo`` observations after it, are removed.
    """

    groups = _contiguous_groups(n_observations, n_groups, minimum_length=1)
    _nonnegative_int("purge", purge)
    _nonnegative_int("embargo", embargo)
    if isinstance(n_test_groups, bool) or not isinstance(n_test_groups, int):
        raise TypeError("n_test_groups must be an integer")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be between 1 and n_groups - 1")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for chosen in combinations(range(n_groups), n_test_groups):
        test = np.concatenate([groups[index] for index in chosen])
        test.sort()
        blocked = {int(index) for index in test}
        for index in chosen:
            start = int(groups[index][0])
            stop = int(groups[index][-1]) + 1
            blocked.update(range(max(start - purge, 0), start))
            blocked.update(range(stop, min(stop + embargo, n_observations)))
        train = np.array(
            [index for index in range(n_observations) if index not in blocked], dtype=int
        )
        if len(train) == 0:
            raise ValueError("purge and embargo removed every training observation")
        splits.append((train, test))
    return tuple(splits)


def deflated_sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    *,
    n_trials: int | None = None,
    selected: str | None = None,
) -> DeflatedSharpe:
    """Bailey and López de Prado deflated Sharpe probability.

    A DataFrame is a set of trials, one column per strategy. The selected
    column is the highest full-sample Sharpe, with the earliest column winning
    ties. A Series uses ``n_trials`` and, when that exceeds one, the analytical
    Sharpe variance as the trial dispersion.
    """

    if isinstance(returns, pd.Series):
        frame = returns.to_frame(name=returns.name or "strategy")
    elif isinstance(returns, pd.DataFrame):
        frame = returns
    else:
        raise TypeError("returns must be a Series or DataFrame")
    if frame.empty or frame.shape[1] == 0 or not frame.columns.is_unique:
        raise ValueError("returns must contain at least one uniquely named strategy")
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("deflated Sharpe requires at least three finite observations")
    sharpes = np.array([_sharpe(values[:, column]) for column in range(values.shape[1])])
    if not np.isfinite(sharpes).all():
        raise ValueError("deflated Sharpe requires finite trial Sharpes")
    if selected is None:
        chosen = int(np.argmax(sharpes))
    else:
        if selected not in frame.columns:
            raise ValueError("selected strategy is not in the return columns")
        chosen = int(frame.columns.get_loc(selected))
    trials = frame.shape[1] if n_trials is None else n_trials
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if trials < frame.shape[1]:
        raise ValueError("n_trials cannot be smaller than the number of return columns")
    observed = float(sharpes[chosen])
    moments = _sharpe_moments(values[:, chosen])
    sharpe_variance = moments[0]
    if frame.shape[1] > 1:
        sharpe_variance = float(np.var(sharpes, ddof=1))
    benchmark = _expected_maximum_sharpe(sharpe_variance, trials)
    probability = _sharpe_probability(observed, benchmark, moments[1], values.shape[0])
    return DeflatedSharpe(observed, benchmark, probability, trials, int(values.shape[0]))


def probability_of_backtest_overfitting(
    strategy_returns: pd.DataFrame,
    *,
    n_groups: int = 8,
) -> OverfitResult:
    """CSCV probability that the in-sample winner is below the out-of-sample median.

    Groups are contiguous and of equal length as far as the sample allows.
    ``n_groups`` must be even and at least 2. Each combination of half the
    groups is one in-sample set. Ties in Sharpe keep the earliest column.
    """

    if not isinstance(strategy_returns, pd.DataFrame):
        raise TypeError("strategy_returns must be a DataFrame")
    if strategy_returns.shape[1] < 2 or not strategy_returns.columns.is_unique:
        raise ValueError("overfit probability requires at least two uniquely named strategies")
    values = strategy_returns.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("strategy returns must be finite")
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 2 or n_groups % 2:
        raise ValueError("n_groups must be an even integer of at least 2")
    groups = _contiguous_groups(values.shape[0], n_groups, minimum_length=2)
    half = n_groups // 2
    logits: list[float] = []
    below_median = 0
    splits = 0
    for chosen in combinations(range(n_groups), half):
        in_sample = np.concatenate([groups[index] for index in chosen])
        complement = [index for index in range(n_groups) if index not in chosen]
        out_sample = np.concatenate([groups[index] for index in complement])
        in_scores = np.array(
            [_sharpe(values[in_sample, column]) for column in range(values.shape[1])]
        )
        out_scores = np.array(
            [_sharpe(values[out_sample, column]) for column in range(values.shape[1])]
        )
        if not np.isfinite(in_scores).all() or not np.isfinite(out_scores).all():
            raise ValueError("a CSCV split produced a non-finite Sharpe")
        winner = int(np.argmax(in_scores))
        rank = _relative_rank(out_scores, winner)
        clipped = min(max(rank, 1e-6), 1.0 - 1e-6)
        logits.append(math.log(clipped / (1.0 - clipped)))
        if rank < 0.5:
            below_median += 1
        splits += 1
    return OverfitResult(below_median / splits, tuple(logits), splits)


def _return_frame(asset_returns: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(asset_returns, pd.DataFrame):
        raise TypeError("asset_returns must be a DataFrame")
    if asset_returns.empty or asset_returns.shape[1] == 0 or not asset_returns.columns.is_unique:
        raise ValueError("asset_returns must contain at least one uniquely named asset")
    if not asset_returns.index.is_unique:
        raise ValueError("asset_returns index must be unique")
    index = asset_returns.index
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)) and not (
        pd.api.types.is_numeric_dtype(index.dtype) and not pd.api.types.is_bool_dtype(index.dtype)
    ):
        raise ValueError(
            "asset_returns index must contain datetime, period or numeric decision times"
        )
    if index.hasnans or (
        pd.api.types.is_numeric_dtype(index.dtype) and not np.isfinite(index.to_numpy()).all()
    ):
        raise ValueError("asset_returns index must contain finite, non-missing decision times")
    if not index.is_monotonic_increasing:
        raise ValueError("asset_returns index must be strictly increasing")
    values = asset_returns.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("asset_returns must be finite")
    return values


def _policy_weights(weights: pd.Series, assets: Sequence[str], *, name: str) -> np.ndarray:
    if not isinstance(weights, pd.Series) or not weights.index.is_unique:
        raise ValueError(f"{name} must be a Series with unique assets")
    numeric = pd.to_numeric(weights.reindex(assets), errors="coerce").to_numpy(dtype=float)
    if set(map(str, weights.index)) != set(map(str, assets)) or not np.isfinite(numeric).all():
        raise ValueError(f"{name} must be finite and cover every asset")
    if np.any(numeric < 0) or abs(float(numeric.sum()) - 1.0) > 1e-8:
        raise ValueError(f"{name} must be non-negative and sum to one")
    return numeric


def _cost_vector(linear_costs: pd.Series | float, assets: Sequence[str]) -> np.ndarray:
    if isinstance(linear_costs, pd.Series):
        if not linear_costs.index.is_unique:
            raise ValueError("linear_costs index must be unique")
        costs = pd.to_numeric(linear_costs.reindex(assets), errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(costs).all():
            raise ValueError("linear_costs must be finite for every asset")
    else:
        if (
            isinstance(linear_costs, bool)
            or not isinstance(linear_costs, (int, float))
            or not np.isfinite(linear_costs)
        ):
            raise ValueError("linear_costs must be finite and non-negative")
        costs = np.repeat(float(linear_costs), len(assets))
    if np.any(costs < 0):
        raise ValueError("linear_costs must be finite and non-negative")
    return costs


def _impact_inputs(
    assets: Sequence[str],
    impact_adv: pd.Series | None,
    impact_volatility: pd.Series | None,
    impact_coefficient: float,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    supplied = impact_adv is not None or impact_volatility is not None or impact_coefficient != 0
    if not supplied:
        return None
    if (
        isinstance(impact_coefficient, bool)
        or not isinstance(impact_coefficient, (int, float))
        or not np.isfinite(impact_coefficient)
        or impact_coefficient <= 0
        or impact_adv is None
        or impact_volatility is None
    ):
        raise ValueError("square-root impact requires a positive coefficient, ADV and volatility")
    adv = pd.to_numeric(impact_adv.reindex(assets), errors="coerce").to_numpy(dtype=float)
    volatility = pd.to_numeric(impact_volatility.reindex(assets), errors="coerce").to_numpy(
        dtype=float
    )
    if not np.isfinite(adv).all() or not np.isfinite(volatility).all() or np.any(adv <= 0):
        raise ValueError("ADV must be positive and volatility must be finite")
    if np.any(volatility < 0):
        raise ValueError("volatility must be non-negative")
    return adv, volatility, float(impact_coefficient)


def _contiguous_groups(
    n_observations: int,
    n_groups: int,
    *,
    minimum_length: int,
) -> tuple[np.ndarray, ...]:
    if (
        isinstance(n_observations, bool)
        or not isinstance(n_observations, int)
        or n_observations < 1
    ):
        raise ValueError("n_observations must be a positive integer")
    if isinstance(n_groups, bool) or not isinstance(n_groups, int) or n_groups < 1:
        raise ValueError("n_groups must be a positive integer")
    if n_observations < n_groups * minimum_length:
        raise ValueError("each group needs enough observations for the requested split")
    edges = np.linspace(0, n_observations, n_groups + 1)
    bounds = np.round(edges).astype(int)
    bounds[0] = 0
    bounds[-1] = n_observations
    groups = tuple(
        np.arange(int(bounds[index]), int(bounds[index + 1])) for index in range(n_groups)
    )
    if any(len(group) < minimum_length for group in groups):
        raise ValueError("group lengths are uneven; use more observations or fewer groups")
    return groups


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        raise ValueError("Sharpe ratio requires at least two observations")
    deviation = float(np.std(returns, ddof=1))
    mean = float(np.mean(returns))
    if deviation == 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")
    return mean / deviation


def _sharpe_moments(returns: np.ndarray) -> tuple[float, float]:
    sharpe = _sharpe(returns)
    centered = returns - float(np.mean(returns))
    second = float(np.mean(centered**2))
    if second <= 0:
        raise ValueError("Sharpe ratio is undefined for zero volatility")
    skew = float(np.mean(centered**3)) / second**1.5
    kurtosis = float(np.mean(centered**4)) / second**2
    variance = (1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2) / (len(returns) - 1)
    if variance <= 0:
        raise ValueError("Sharpe variance estimate is not positive")
    return variance, variance


def _expected_maximum_sharpe(variance: float, n_trials: int) -> float:
    if n_trials == 1 or variance <= 0:
        return 0.0
    euler = 0.5772156649015329
    independent = 1.0 - 1.0 / n_trials
    extreme = 1.0 - 1.0 / (n_trials * math.e)
    benchmark = (1.0 - euler) * _norm_ppf(independent) + euler * _norm_ppf(extreme)
    return math.sqrt(variance) * benchmark


def _sharpe_probability(
    observed: float, benchmark: float, variance: float, observations: int
) -> float:
    del observations
    scale = math.sqrt(variance)
    if scale == 0:
        return 1.0 if observed > benchmark else 0.0
    return _norm_cdf((observed - benchmark) / scale)


def _relative_rank(scores: np.ndarray, chosen: int) -> float:
    selected = float(scores[chosen])
    strictly_worse = float(np.sum(scores < selected))
    tied = float(np.sum(scores == selected))
    return (strictly_worse + 0.5 * tied) / len(scores)


def _nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _norm_ppf(probability: float) -> float:
    """Acklam's rational approximation of the standard normal quantile."""

    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    if probability < plow:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if probability > 1.0 - plow:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )
