import numpy as np
import pandas as pd
import pytest

from quant_portfolio.methods import (
    black_litterman_expected_returns,
    equal_risk_contribution_weights,
    factor_model_covariance,
    hierarchical_risk_parity_weights,
    ic_vol_expected_returns,
)
from quant_portfolio.optimization import (
    OptimizationConstraints,
    estimate_covariance,
    optimize_cvar,
    optimize_mean_variance,
    optimize_multiperiod,
    research_allocation_weights,
    validate_research_allocation,
)


def _correlated_covariance() -> pd.DataFrame:
    values = np.array(
        [
            [0.04, 0.036, 0.0],
            [0.036, 0.04, 0.0],
            [0.0, 0.0, 0.04],
        ]
    )
    assets = ["A", "B", "C"]
    return pd.DataFrame(values, index=assets, columns=assets)


def test_ledoit_wolf_is_positive_definite_and_not_fixed_diagonal_shrinkage() -> None:
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(scale=0.01, size=(48, 5)), columns=list("ABCDE"))
    ledoit = estimate_covariance(returns, method="ledoit_wolf")
    diagonal = estimate_covariance(returns, method="diagonal", shrinkage=0.2)
    assert np.linalg.eigvalsh(ledoit.to_numpy()).min() > 0
    assert not np.allclose(ledoit.to_numpy(), diagonal.to_numpy())
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete observations"):
        estimate_covariance(broken, method="ledoit_wolf")


def test_factor_model_covariance_matches_xfx_plus_specific() -> None:
    loadings = pd.DataFrame({"mkt": [1.0, 0.5]}, index=["A", "B"])
    factor_cov = pd.DataFrame([[0.04]], index=["mkt"], columns=["mkt"])
    specific = pd.Series({"A": 0.01, "B": 0.02})
    covariance = factor_model_covariance(loadings, factor_cov, specific)
    assert covariance.loc["A", "A"] == pytest.approx(0.05)
    assert covariance.loc["B", "B"] == pytest.approx(0.03)
    assert covariance.loc["A", "B"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="positive semidefinite"):
        factor_model_covariance(
            loadings, pd.DataFrame([[-0.01]], index=["mkt"], columns=["mkt"]), specific
        )


def test_ic_vol_scales_scores_by_volatility() -> None:
    scores = pd.Series({"A": 1.0, "B": -1.0})
    volatility = pd.Series({"A": 0.10, "B": 0.40})
    expected = ic_vol_expected_returns(scores, volatility, 0.05)
    assert expected["A"] == pytest.approx(0.005)
    assert expected["B"] == pytest.approx(-0.02)
    with pytest.raises(ValueError, match="dispersion"):
        ic_vol_expected_returns(pd.Series({"A": 1.0, "B": 1.0}), volatility, 0.05)


def test_black_litterman_pulls_views_toward_equilibrium() -> None:
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets)
    market = pd.Series(0.5, index=assets)
    views = pd.Series({"A": 0.20, "B": -0.10})
    posterior = black_litterman_expected_returns(
        covariance, market, views, risk_aversion=2.5, tau=0.05
    )
    equilibrium = 2.5 * covariance.to_numpy() @ market.to_numpy()
    assert abs(posterior["A"] - equilibrium[0]) < abs(views["A"] - equilibrium[0])
    assert posterior["A"] > posterior["B"]


def test_equal_risk_contribution_uses_correlation() -> None:
    diagonal = pd.DataFrame(np.diag([0.04, 0.16]), index=["A", "B"], columns=["A", "B"])
    inverse_vol = equal_risk_contribution_weights(diagonal)
    assert inverse_vol["A"] == pytest.approx(2 / 3, abs=1e-4)
    covariance = _correlated_covariance()
    weights = equal_risk_contribution_weights(covariance)
    contribution = weights.to_numpy() * (covariance.to_numpy() @ weights.to_numpy())
    assert np.allclose(contribution, contribution.mean(), atol=1e-6)
    assert weights["C"] > weights["A"]
    assert weights["C"] > weights["B"]


def test_hierarchical_risk_parity_gives_the_uncorrelated_asset_more_weight() -> None:
    weights = hierarchical_risk_parity_weights(_correlated_covariance())
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()
    assert weights["C"] > weights["A"]
    assert weights["C"] > weights["B"]


def test_cvar_avoids_the_crash_asset_when_means_match() -> None:
    calm = [0.01] * 19
    scenarios = pd.DataFrame({"A": calm + [-0.40], "B": [0.01] * 20})
    expected = pd.Series(0.01, index=["A", "B"])
    result = optimize_cvar(scenarios, expected, beta=0.90, risk_aversion=8.0)
    assert result.converged
    assert result.weights["B"] > result.weights["A"]
    assert result.objective == pytest.approx(
        result.expected_return - 8.0 * _cvar(scenarios.to_numpy() @ result.weights.to_numpy(), 0.90)
    )


def _cvar(portfolio_returns: np.ndarray, beta: float) -> float:
    losses = -portfolio_returns
    tail = losses[losses >= np.quantile(losses, beta) - 1e-15]
    return float(tail.mean())


def test_active_risk_stays_on_the_benchmark_when_expected_returns_are_flat() -> None:
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets)
    expected = pd.Series(0.0, index=assets)
    benchmark = pd.Series({"A": 0.8, "B": 0.2})
    active = optimize_mean_variance(expected, covariance, benchmark_weights=benchmark)
    total = optimize_mean_variance(expected, covariance)
    assert active.weights["A"] == pytest.approx(0.8, abs=1e-4)
    assert total.weights["A"] == pytest.approx(0.5, abs=1e-4)


def test_active_factor_bounds_limit_exposure_versus_the_benchmark() -> None:
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets)
    benchmark = pd.Series(0.5, index=assets)
    exposures = pd.DataFrame({"beta": [1.5, 0.5]}, index=assets)
    result = optimize_mean_variance(
        pd.Series({"A": 0.30, "B": 0.0}),
        covariance,
        benchmark_weights=benchmark,
        factor_exposures=exposures,
        factor_bounds={"beta": (-0.05, 0.05)},
        factor_bound_reference="active",
        risk_aversion=1.0,
    )
    active_beta = float(exposures["beta"] @ (result.weights - benchmark))
    assert -0.05 - 1e-8 <= active_beta <= 0.05 + 1e-8


def test_square_root_impact_reduces_the_illiquid_overweight() -> None:
    assets = ["A", "B"]
    covariance = pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets)
    expected = pd.Series({"A": 0.20, "B": 0.05})
    current = pd.Series(0.5, index=assets)
    plain = optimize_mean_variance(expected, covariance, current_weights=current, risk_aversion=2.0)
    impacted = optimize_mean_variance(
        expected,
        covariance,
        current_weights=current,
        risk_aversion=2.0,
        impact_adv=pd.Series({"A": 1.0e5, "B": 1.0e8}),
        impact_volatility=pd.Series(0.02, index=assets),
        impact_coefficient=0.8,
        portfolio_nav=1.0e7,
    )
    assert impacted.weights["A"] < plain.weights["A"]
    assert impacted.objective < plain.objective


def test_multiperiod_plan_follows_each_date_expected_return() -> None:
    expected = pd.DataFrame([[0.30, 0.0], [0.0, 0.30]], index=[0, 1], columns=["A", "B"])
    covariance = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"])
    result = optimize_multiperiod(
        expected,
        covariance,
        current_weights=pd.Series(0.5, index=["A", "B"]),
        risk_aversion=2.0,
        constraints=OptimizationConstraints(max_weight=1.0, max_turnover=2.0),
    )
    assert result.converged
    assert result.weights.loc[0, "A"] > result.weights.loc[1, "A"]
    assert result.weights.loc[1, "B"] > result.weights.loc[0, "B"]
    assert result.weights.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])


def test_research_modes_cover_risk_parity_hrp_cvar_and_expected_return_models() -> None:
    covariance = _correlated_covariance()
    scores = pd.Series({"A": 0.4, "B": 0.1, "C": -0.2})
    parity = research_allocation_weights(
        scores,
        None,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={"mode": "risk_parity"},
        covariance_override=covariance,
    )
    clustered = research_allocation_weights(
        scores,
        None,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={"mode": "hrp"},
        covariance_override=covariance,
    )
    assert parity.sum() == pytest.approx(1.0)
    assert clustered["C"] > clustered["A"]
    scenarios = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.30, 0.01],
            "B": [0.01, 0.0, 0.02, 0.01],
            "C": [0.0, 0.01, 0.01, 0.0],
        }
    )
    downside = research_allocation_weights(
        scores,
        scenarios,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={
            "mode": "cvar",
            "lookback": 4,
            "min_observations": 4,
            "cvar_beta": 0.75,
            "risk_aversion": 6.0,
        },
    )
    assert downside.sum() == pytest.approx(1.0)
    sleeve = pd.DataFrame(np.diag([0.04, 0.25]), index=["A", "B"], columns=["A", "B"])
    pair_scores = pd.Series({"A": 1.0, "B": -1.0})
    base = {"mode": "cost_aware", "risk_aversion": 8.0, "max_turnover": 2.0}
    raw = research_allocation_weights(
        pair_scores,
        None,
        pd.Series(0.5, index=["A", "B"]),
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={**base, "expected_return_model": "score"},
        covariance_override=sleeve,
    )
    scaled = research_allocation_weights(
        pair_scores,
        None,
        pd.Series(0.5, index=["A", "B"]),
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={**base, "expected_return_model": "ic_vol", "information_coefficient": 0.05},
        covariance_override=sleeve,
    )
    blended = research_allocation_weights(
        pair_scores,
        None,
        pd.Series(0.5, index=["A", "B"]),
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={
            **base,
            "expected_return_model": "black_litterman",
            "information_coefficient": 0.05,
            "black_litterman_tau": 0.05,
        },
        covariance_override=sleeve,
        market_weights=pd.Series(0.5, index=["A", "B"]),
    )
    assert not np.isclose(raw["A"], scaled["A"])
    assert abs(blended["A"] - 0.5) < abs(raw["A"] - 0.5)


def test_research_factor_covariance_and_closed_recipe_rejections() -> None:
    loadings = pd.DataFrame({"mkt": [1.0, 0.2, 0.8]}, index=["A", "B", "C"])
    factor_cov = pd.DataFrame([[0.04]], index=["mkt"], columns=["mkt"])
    specific = pd.Series({"A": 0.01, "B": 0.03, "C": 0.02})
    weights = research_allocation_weights(
        pd.Series({"A": 0.2, "B": 0.0, "C": -0.1}),
        None,
        None,
        None,
        invested_limit=1.0,
        max_weight=0.8,
        config={"mode": "risk_parity", "covariance_estimator": "factor"},
        factor_loadings=loadings,
        factor_covariance=factor_cov,
        specific_variance=specific,
    )
    assert weights.sum() == pytest.approx(1.0)
    normalized = validate_research_allocation(
        {"mode": "hrp", "covariance_estimator": "ledoit_wolf"}
    )
    assert normalized["covariance_estimator"] == "ledoit_wolf"
    with pytest.raises(ValueError, match="only supported for cost_aware"):
        validate_research_allocation({"mode": "equal", "expected_return_model": "ic_vol"})
    with pytest.raises(ValueError, match="hrp or cvar"):
        validate_research_allocation({"mode": "not-a-mode"})
