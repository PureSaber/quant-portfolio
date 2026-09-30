import math

import numpy as np
import pandas as pd
import pytest

from quant_portfolio.evaluation import (
    _norm_ppf,
    cpcv_splits,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    simulate_market,
)


def test_normal_quantile_matches_known_values() -> None:
    assert _norm_ppf(0.5) == pytest.approx(0.0, abs=1e-8)
    assert _norm_ppf(0.975) == pytest.approx(1.95996398454, abs=1e-6)
    assert _norm_ppf(0.025) == pytest.approx(-1.95996398454, abs=1e-6)


def test_simulator_is_causal_and_charges_costs_before_drift() -> None:
    returns = pd.DataFrame(
        {"A": [1.0, 0.0], "B": [0.0, 0.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    seen: list[int] = []

    def policy(history: pd.DataFrame) -> pd.Series:
        seen.append(len(history))
        return pd.Series(0.5, index=["A", "B"])

    result = simulate_market(returns, policy, initial_weights=pd.Series(0.5, index=["A", "B"]))
    assert seen == [0, 1]
    assert result.turnover.iloc[0] == pytest.approx(0.0)
    assert result.turnover.iloc[1] == pytest.approx(1 / 3)
    assert result.wealth.iloc[0] == pytest.approx(1.5)
    costly = simulate_market(
        returns.iloc[:1],
        lambda _history: pd.Series({"A": 1.0, "B": 0.0}),
        initial_weights=pd.Series(0.5, index=["A", "B"]),
        linear_costs=0.01,
    )
    free = simulate_market(
        returns.iloc[:1],
        lambda _history: pd.Series({"A": 1.0, "B": 0.0}),
        initial_weights=pd.Series(0.5, index=["A", "B"]),
    )
    assert costly.wealth.iloc[0] < free.wealth.iloc[0]
    with pytest.raises(ValueError, match="sum to one"):
        simulate_market(returns, lambda _history: pd.Series({"A": 0.2, "B": 0.2}))


def test_cpcv_purges_and_embargoes_around_each_test_block() -> None:
    splits = cpcv_splits(12, 4, 1, purge=1, embargo=1)
    assert len(splits) == 4
    for train, test in splits:
        assert set(train).isdisjoint(test)
        start = int(test.min())
        stop = int(test.max()) + 1
        blocked = set(range(max(start - 1, 0), start)) | set(range(stop, min(stop + 1, 12)))
        assert set(train).isdisjoint(blocked)
    with pytest.raises(ValueError, match="training observation"):
        cpcv_splits(8, 2, 1, purge=8, embargo=8)


@pytest.mark.parametrize(
    "index",
    [
        pd.to_datetime(["2026-09-29", "2026-09-28", "2026-09-27"]),
        pd.to_datetime(["2026-09-27", "2026-09-29", "2026-09-28"]),
        pd.to_datetime(["2026-09-27", "2026-09-27", "2026-09-28"]),
        pd.to_datetime(["2026-09-27", None, "2026-09-29"]),
        pd.Index([2, 1, 0]),
        pd.Index([0.0, float("inf"), 2.0]),
        pd.Index(["not-a-date", "still-not-a-date", "zzz"]),
    ],
)
def test_simulator_rejects_invalid_clock_before_calling_policy(index):
    def policy(_history):
        pytest.fail("invalid decision times reached the policy")

    with pytest.raises(ValueError, match="index"):
        simulate_market(pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=index), policy)


@pytest.mark.parametrize(
    "index",
    [
        pd.date_range("2026-09-27", periods=3),
        pd.date_range("2026-09-27", periods=3, tz="Asia/Shanghai"),
        pd.period_range("2026-09-27", periods=3, freq="D"),
        pd.RangeIndex(3),
    ],
)
def test_every_policy_call_sees_only_earlier_decision_times(index):
    calls = []

    def policy(history):
        decision_time = index[len(history)]
        assert all(time < decision_time for time in history.index)
        calls.append(decision_time)
        return pd.Series({"A": 1.0})

    result = simulate_market(pd.DataFrame({"A": [0.01, 0.02, 0.03]}, index=index), policy)
    assert calls == list(index)
    assert result.wealth.index.equals(index)


def test_deflated_sharpe_falls_as_the_trial_count_grows() -> None:
    rng = np.random.default_rng(5)
    noise = rng.normal(scale=0.02, size=60)
    returns = pd.Series(noise)
    single = deflated_sharpe_ratio(returns, n_trials=1)
    many = deflated_sharpe_ratio(returns, n_trials=10_000)
    assert many.benchmark_sharpe > single.benchmark_sharpe
    assert many.probability < single.probability
    trials = pd.DataFrame({"noisy": noise, "sharper": noise + 0.002})
    selected = deflated_sharpe_ratio(trials, selected="sharper")
    assert selected.n_trials == 2
    assert selected.sharpe > deflated_sharpe_ratio(trials, selected="noisy").sharpe
    with pytest.raises(ValueError, match="three finite"):
        deflated_sharpe_ratio(pd.Series([0.01, 0.02]), n_trials=1)


def test_probability_of_backtest_overfitting_detects_a_mirror_strategy() -> None:
    block = np.array([0.02, -0.01, 0.02, -0.01])
    opposite = -block
    strategies = pd.DataFrame(
        {
            "early": np.concatenate([block, block, opposite, opposite]),
            "late": np.concatenate([opposite, opposite, block, block]),
        }
    )
    result = probability_of_backtest_overfitting(strategies, n_groups=4)
    assert result.n_splits == math.comb(4, 2)
    assert result.probability == pytest.approx(1 / 3)
    dominant = pd.DataFrame({"good": np.tile([0.02, -0.005], 8), "bad": np.tile([-0.02, 0.005], 8)})
    clean = probability_of_backtest_overfitting(dominant, n_groups=4)
    assert clean.probability == pytest.approx(0.0)
    assert _norm_ppf(1e-4) < -3
    assert _norm_ppf(1 - 1e-4) > 3
    priced = pd.DataFrame({"A": [0.01, -0.02], "B": [0.0, 0.01]})
    impacted = simulate_market(
        priced,
        lambda _history: pd.Series({"A": 1.0, "B": 0.0}),
        initial_weights=pd.Series({"A": 0.0, "B": 1.0}),
        impact_adv=pd.Series({"A": 1_000_000.0, "B": 1_000_000.0}),
        impact_volatility=pd.Series({"A": 0.02, "B": 0.02}),
        impact_coefficient=0.1,
        initial_wealth=1_000_000.0,
    )
    assert impacted.costs.iloc[0] > 0
    with pytest.raises(ValueError, match="initial_wealth"):
        simulate_market(priced, lambda _history: pd.Series(0.5, index=["A", "B"]), initial_wealth=0)
    with pytest.raises(ValueError, match="wiped out"):
        simulate_market(
            pd.DataFrame({"A": [-1.0], "B": [0.0]}),
            lambda _history: pd.Series({"A": 1.0, "B": 0.0}),
        )
    with pytest.raises(ValueError, match="consumed"):
        simulate_market(
            priced.iloc[:1],
            lambda _history: pd.Series({"A": 1.0, "B": 0.0}),
            initial_weights=pd.Series(0.5, index=["A", "B"]),
            linear_costs=1.0,
        )
    with pytest.raises(ValueError, match="even integer"):
        probability_of_backtest_overfitting(strategies, n_groups=3)
    with pytest.raises(TypeError, match="DataFrame"):
        probability_of_backtest_overfitting([0.01, 0.02])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square-root impact"):
        simulate_market(
            priced,
            lambda _history: pd.Series(0.5, index=["A", "B"]),
            impact_coefficient=0.1,
        )
