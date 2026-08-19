from __future__ import annotations

import pandas as pd
import pytest

from quant_portfolio.synthetic_spread import (
    SyntheticSpreadConfig,
    add_causal_zscore,
    backtest_synthetic_spread,
    generate_synthetic_pair,
)


def test_generation_is_deterministic() -> None:
    config = SyntheticSpreadConfig(periods=120, lookback=20)
    first = generate_synthetic_pair(config)
    second = generate_synthetic_pair(config)
    pd.testing.assert_frame_equal(first, second)


def test_zscore_uses_only_prior_observations() -> None:
    frame = pd.DataFrame({"spread": [0.0, 1.0, 2.0, 100.0]})
    output = add_causal_zscore(frame, lookback=3)
    expected = (100.0 - 1.0) / ((2.0 / 3.0) ** 0.5)
    assert float(output.loc[3, "zscore"]) == pytest.approx(expected)


def test_costs_reduce_return_and_metrics_are_finite() -> None:
    config = SyntheticSpreadConfig(periods=240, lookback=20)
    frame = generate_synthetic_pair(config)
    _, no_cost = backtest_synthetic_spread(
        frame,
        lookback=config.lookback,
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        cost_bps=0.0,
    )
    result, with_cost = backtest_synthetic_spread(
        frame,
        lookback=config.lookback,
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        cost_bps=5.0,
    )
    assert with_cost.total_return <= no_cost.total_return
    assert with_cost.trades > 0
    assert with_cost.observations == len(frame)
    assert result["equity"].notna().all()
