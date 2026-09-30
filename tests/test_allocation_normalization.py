import numpy as np
import pandas as pd
import pytest

from quant_portfolio.optimization import research_allocation_weights, validate_research_allocation


@pytest.mark.parametrize(
    "mode",
    [
        "equal",
        "inverse_vol",
        "cost_aware",
        "risk_parity",
        "hrp",
        "cvar",
        "max_sharpe",
        "max_diversification",
        "evar",
        "cdar",
    ],
)
def test_normalized_allocation_can_be_serialized_and_revalidated(mode):
    import json

    normalized = validate_research_allocation({"mode": mode})
    assert validate_research_allocation(json.loads(json.dumps(normalized))) == normalized


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "cost_aware", "covariance_estimator": "ewma", "ewma_decay": 0.9},
        {
            "mode": "cost_aware",
            "expected_return_model": "black_litterman",
            "black_litterman_tau": 0.2,
        },
        {"mode": "max_sharpe", "risk_free_rate": 0.0001},
        {"mode": "cost_aware", "max_tracking_error": 0.1},
        {"mode": "cvar", "cvar_beta": 0.9},
        {"mode": "evar", "evar_beta": 0.9},
        {"mode": "cdar", "cdar_beta": 0.9},
    ],
)
def test_mode_specific_parameters_survive_revalidation(config):
    normalized = validate_research_allocation(config)
    assert validate_research_allocation(normalized) == normalized
    assert all(normalized[key] == value for key, value in config.items())


@pytest.mark.parametrize(
    "field,value",
    [
        ("ewma_decay", 0.94),
        ("risk_free_rate", 0.0),
        ("max_tracking_error", None),
        ("information_coefficient", 0.05),
        ("black_litterman_tau", 0.05),
        ("cvar_beta", 0.95),
        ("evar_beta", 0.95),
        ("cdar_beta", 0.95),
    ],
)
def test_irrelevant_explicit_parameters_still_fail_closed(field, value):
    with pytest.raises(ValueError, match="only"):
        validate_research_allocation({"mode": "equal", field: value})


@pytest.mark.parametrize("mode", ["equal", "inverse_vol", "cost_aware", "hrp"])
def test_execution_accepts_prevalidated_recipe_fragment(mode):
    scores = pd.Series({"A": 0.02, "B": 0.01})
    returns = pd.DataFrame(np.random.default_rng(12).normal(0, 0.01, (30, 2)), columns=["A", "B"])
    config = {"mode": mode}

    def allocate(fragment):
        return research_allocation_weights(
            scores,
            returns,
            None,
            None,
            invested_limit=0.5,
            max_weight=0.4,
            config=fragment,
        )

    pd.testing.assert_series_equal(allocate(config), allocate(validate_research_allocation(config)))
