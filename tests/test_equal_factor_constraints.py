import numpy as np
import pandas as pd
import pytest

from quant_portfolio.optimization import research_allocation_weights


def allocate(*, current=None, turnover=2.0, bounds=None, exposures=None, covariance=None):
    return research_allocation_weights(
        pd.Series({"A": 10.0, "B": -10.0}),
        None,
        current,
        None,
        invested_limit=0.6,
        max_weight=0.5,
        config={"mode": "equal", "max_turnover": turnover},
        factor_exposures=exposures
        if exposures is not None
        else pd.DataFrame({"beta": [1.2, 0.4]}, index=["A", "B"]),
        factor_bounds=bounds if bounds is not None else {"beta": (0.52, 0.56)},
        covariance_override=covariance,
    )


def test_equal_target_has_hand_solved_absolute_factor_projection():
    # w_A+w_B=.6 and 1.2*w_A+.4*w_B>=.52 imply w_A>=.35.
    # The closest feasible point to (.3,.3) is therefore (.35,.25).
    weights = allocate()
    assert weights.to_dict() == pytest.approx({"A": 0.35, "B": 0.25}, abs=1e-9)
    assert weights.sum() == pytest.approx(0.6, abs=1e-9)
    assert 1.2 * weights.A + 0.4 * weights.B >= 0.52 - 1e-9


def test_factor_projection_enforces_turnover_and_dropped_positions_together():
    current = pd.Series({"A": 0.5, "B": 0.0, "DROPPED": 0.1})
    # Selling the dropped .1 position leaves .2 of the .3 turnover budget.
    # |a-.5| + |.6-a| <= .2 requires a>=.45.
    weights = allocate(current=current, turnover=0.3, bounds={"beta": (0.0, 0.7)})
    assert weights.to_dict() == pytest.approx({"A": 0.45, "B": 0.15}, abs=1e-9)
    assert float((weights - current.reindex(weights.index)).abs().sum()) + 0.1 <= 0.3 + 1e-9


def test_factor_projection_accepts_partial_cash_with_cap_above_sleeve():
    weights = research_allocation_weights(
        pd.Series({"A": 1.0}),
        None,
        None,
        None,
        invested_limit=0.2,
        max_weight=0.4,
        config={"mode": "equal"},
        factor_exposures=pd.DataFrame({"beta": [1.0]}, index=["A"]),
        factor_bounds={"beta": (0.0, 0.3)},
    )
    assert weights.to_dict() == pytest.approx({"A": 0.2})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bounds": {"beta": (0.8, 0.9)}},
        {"turnover": 0.0},
        {"current": pd.Series({"A": 0.3, "B": 0.3, "DROPPED": 0.4}), "turnover": 0.3},
        {"current": pd.Series({"A": 0.3, "B": 0.3}), "turnover": 0.01},
    ],
)
def test_infeasible_equal_factor_constraints_are_never_relaxed(kwargs):
    with pytest.raises(ValueError, match="joint constraints are infeasible"):
        allocate(**kwargs)


def test_equal_projection_is_label_aligned_and_does_not_change_inputs():
    exposures = pd.DataFrame({"beta": [0.4, 1.2]}, index=["B", "A"])
    before = exposures.copy(deep=True)
    covariance = pd.DataFrame([[0.2, 0.03], [0.03, 0.1]], index=["B", "A"], columns=["B", "A"])
    weights = allocate(exposures=exposures, covariance=covariance)
    pd.testing.assert_series_equal(weights, allocate())
    pd.testing.assert_frame_equal(exposures, before)


@pytest.mark.parametrize(
    "covariance, error",
    [
        ([[1, 0], [0, 1]], "DataFrame"),
        (pd.DataFrame([[1]], index=["A"], columns=["A"]), "missing assets"),
        (pd.DataFrame(np.eye(2), index=["A", "A"], columns=["A", "B"]), "unique"),
        (pd.DataFrame([[1, np.nan], [0, 1]], index=["A", "B"], columns=["A", "B"]), "finite"),
        (pd.DataFrame([[1, np.inf], [0, 1]], index=["A", "B"], columns=["A", "B"]), "finite"),
        (pd.DataFrame([[1, 0.1], [0, 1]], index=["A", "B"], columns=["A", "B"]), "symmetric"),
        (pd.DataFrame([[1, 2], [2, 1]], index=["A", "B"], columns=["A", "B"]), "semidefinite"),
    ],
)
def test_equal_mode_validates_supplied_risk_covariance(covariance, error):
    with pytest.raises((TypeError, ValueError), match=error):
        allocate(covariance=covariance)


def test_equal_mode_rejects_unpaired_factor_inputs():
    with pytest.raises(ValueError, match="provided together"):
        research_allocation_weights(
            pd.Series({"A": 1.0}),
            None,
            None,
            None,
            invested_limit=0.6,
            max_weight=1.0,
            config={"mode": "equal"},
            factor_bounds={"beta": (0.0, 1.0)},
        )
