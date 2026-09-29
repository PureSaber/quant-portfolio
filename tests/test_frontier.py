import numpy as np
import pandas as pd
import pytest

from quant_portfolio.methods import (
    constant_correlation_covariance,
    ewma_covariance,
    oas_covariance,
    random_matrix_covariance,
)
from quant_portfolio.objectives import (
    conditional_drawdown_at_risk,
    entropic_value_at_risk,
    maximum_drawdown,
)
from quant_portfolio.optimization import (
    OptimizationConstraints,
    estimate_covariance,
    optimize_cdar,
    optimize_cvar,
    optimize_evar,
    optimize_max_diversification,
    optimize_max_sharpe,
    optimize_mean_variance,
    optimize_multiperiod,
    research_allocation_weights,
)


def _diagonal_book() -> tuple[pd.Series, pd.DataFrame]:
    expected = pd.Series({"A": 0.10, "B": 0.02})
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    return expected, covariance


def test_covariance_estimators_are_positive_definite_and_distinct() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(scale=0.01, size=(36, 4)), columns=list("ABCD"))
    estimators = {
        "ewma": estimate_covariance(returns, method="ewma", annualization=1),
        "constant_correlation": estimate_covariance(
            returns, method="constant_correlation", annualization=1
        ),
        "oas": estimate_covariance(returns, method="oas", annualization=1),
        "random_matrix": estimate_covariance(returns, method="random_matrix", annualization=1),
    }
    for covariance in estimators.values():
        assert np.linalg.eigvalsh(covariance.to_numpy()).min() > 0
    assert not np.allclose(estimators["ewma"], estimators["oas"])
    broken = returns.copy()
    broken.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="complete observations"):
        estimate_covariance(broken, method="oas")


def test_ewma_puts_more_weight_on_a_recent_shock() -> None:
    calm = np.full((30, 1), 0.0)
    early = calm.copy()
    early[0, 0] = 0.2
    late = calm.copy()
    late[-1, 0] = 0.2
    early_variance = ewma_covariance(pd.DataFrame(early), decay=0.8, annualization=1).iloc[0, 0]
    late_variance = ewma_covariance(pd.DataFrame(late), decay=0.8, annualization=1).iloc[0, 0]
    assert late_variance > early_variance
    with pytest.raises(ValueError, match="decay"):
        ewma_covariance(pd.DataFrame(calm), decay=1.0, annualization=1)


def test_constant_correlation_pulls_pairwise_correlations_together() -> None:
    rng = np.random.default_rng(2)
    returns = pd.DataFrame(rng.normal(scale=0.02, size=(25, 6)), columns=list("ABCDEF"))
    shrunk = constant_correlation_covariance(returns, annualization=1).to_numpy()
    centered = returns.to_numpy() - returns.to_numpy().mean(axis=0)
    sample = centered.T @ centered / len(returns)

    def dispersion(covariance: np.ndarray) -> float:
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        mask = ~np.eye(len(scale), dtype=bool)
        return float(np.std(correlation[mask]))

    assert dispersion(shrunk) < dispersion(sample)
    assert np.linalg.eigvalsh(shrunk).min() > 0


def test_oas_shrinks_small_samples_toward_a_scaled_identity() -> None:
    rng = np.random.default_rng(3)
    returns = pd.DataFrame(rng.normal(size=(12, 5)), columns=list("ABCDE"))
    shrunk = oas_covariance(returns, annualization=1).to_numpy()
    centered = returns.to_numpy() - returns.to_numpy().mean(axis=0)
    sample = centered.T @ centered / len(returns)

    def off_diagonal(covariance: np.ndarray) -> float:
        return float(np.abs(covariance - np.diag(np.diag(covariance))).sum())

    assert off_diagonal(shrunk) < off_diagonal(sample)


def test_random_matrix_replaces_the_noise_bulk() -> None:
    rng = np.random.default_rng(4)
    factor = rng.normal(size=(40, 1))
    loadings = rng.normal(size=(1, 8))
    noise = rng.normal(scale=0.4, size=(40, 8))
    returns = pd.DataFrame(factor @ loadings + noise, columns=list("ABCDEFGH"))
    covariance = random_matrix_covariance(returns, annualization=1)
    scale = np.sqrt(np.diag(covariance.to_numpy()))
    correlation = covariance.to_numpy() / np.outer(scale, scale)
    eigenvalues = np.linalg.eigvalsh(correlation)
    assert eigenvalues.min() > 0
    assert np.std(eigenvalues[:4]) < np.std(eigenvalues[4:])


def test_max_sharpe_matches_the_diagonal_closed_form() -> None:
    expected, covariance = _diagonal_book()
    result = optimize_max_sharpe(
        expected, covariance, constraints=OptimizationConstraints(max_weight=1)
    )
    assert result.converged
    assert result.weights["A"] == pytest.approx(5 / 9, abs=1e-3)
    assert result.weights["B"] == pytest.approx(4 / 9, abs=1e-3)
    assert result.objective == pytest.approx(float(expected @ result.weights) / result.volatility)


def test_max_diversification_is_inverse_volatility_when_assets_are_uncorrelated() -> None:
    _expected, covariance = _diagonal_book()
    result = optimize_max_diversification(
        covariance, constraints=OptimizationConstraints(max_weight=1)
    )
    assert result.weights["A"] == pytest.approx(1 / 3, abs=1e-3)
    assert result.weights["B"] == pytest.approx(2 / 3, abs=1e-3)
    assert result.weights.sum() == pytest.approx(1.0)


def test_tracking_error_is_a_hard_cap_around_the_benchmark() -> None:
    expected = pd.Series({"A": 0.30, "B": 0.01})
    covariance = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"])
    benchmark = pd.Series(0.5, index=["A", "B"])
    result = optimize_mean_variance(
        expected,
        covariance,
        benchmark_weights=benchmark,
        risk_aversion=0.1,
        constraints=OptimizationConstraints(max_weight=1.0, max_tracking_error=0.05),
    )
    active = result.weights.to_numpy() - benchmark.to_numpy()
    tracking = float(np.sqrt(active @ covariance.to_numpy() @ active))
    assert tracking <= 0.05 + 1e-6
    assert result.weights["A"] > 0.6
    with pytest.raises(ValueError, match="benchmark_weights"):
        optimize_mean_variance(
            expected,
            covariance,
            constraints=OptimizationConstraints(max_weight=1.0, max_tracking_error=0.05),
        )


def test_integer_lots_beat_truncation_and_respect_the_step() -> None:
    expected = pd.Series({"A": 0.40, "B": 0.05, "C": 0.0})
    covariance = pd.DataFrame(np.eye(3) * 0.01, index=list("ABC"), columns=list("ABC"))
    steps = pd.Series(0.25, index=list("ABC"))
    result = optimize_mean_variance(
        expected,
        covariance,
        current_weights=pd.Series(0.0, index=list("ABC")),
        risk_aversion=0.01,
        constraints=OptimizationConstraints(max_weight=1.0, max_turnover=2.0),
        weight_steps=steps,
    )
    remainder = np.mod(result.weights.to_numpy(), 0.25)
    assert np.allclose(remainder, 0.0, atol=1e-8)
    assert result.weights.sum() <= 1.0 + 1e-8
    assert result.weights["A"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="positive"):
        optimize_mean_variance(
            expected,
            covariance,
            constraints=OptimizationConstraints(max_weight=1.0),
            weight_steps=pd.Series({"A": 0.2, "B": 0.0, "C": 0.2}),
        )


def test_evar_bounds_cvar_and_avoids_the_crash() -> None:
    losses = np.array([0.01, 0.0, -0.02, 0.4, 0.02, 0.0, 0.01, 0.03])
    beta = 0.75
    evar = entropic_value_at_risk(losses, beta)
    quantile = float(np.quantile(losses, beta))
    tail = losses[losses >= quantile - 1e-15]
    assert evar + 1e-8 >= float(tail.mean())
    assert entropic_value_at_risk(np.zeros(4), 0.95) == pytest.approx(0.0, abs=1e-4)
    calm = [0.01] * 19
    scenarios = pd.DataFrame({"A": calm + [-0.40], "B": [0.01] * 20})
    expected = pd.Series(0.01, index=["A", "B"])
    result = optimize_evar(
        scenarios,
        expected,
        beta=0.9,
        risk_aversion=8.0,
        constraints=OptimizationConstraints(max_weight=1.0),
    )
    assert result.converged
    assert result.weights["B"] > result.weights["A"]


def test_drawdown_constraint_drops_the_path_that_gives_back_its_gains() -> None:
    calm = np.full(12, 0.01)
    boom_bust = np.concatenate([np.full(8, 0.03), np.full(4, -0.08)])
    scenarios = pd.DataFrame({"A": calm, "B": boom_bust})
    expected = pd.Series({"A": 0.01, "B": 0.01})
    assert maximum_drawdown(boom_bust) > maximum_drawdown(calm)
    constrained = optimize_cvar(
        scenarios,
        expected,
        beta=0.75,
        risk_aversion=1.0,
        max_drawdown=0.05,
        constraints=OptimizationConstraints(max_weight=1.0),
    )
    assert maximum_drawdown(scenarios.to_numpy() @ constrained.weights.to_numpy()) <= 0.05 + 1e-8
    assert constrained.weights["A"] > constrained.weights["B"]
    cdar = optimize_cdar(
        scenarios,
        expected,
        beta=0.75,
        risk_aversion=6.0,
        constraints=OptimizationConstraints(max_weight=1.0),
    )
    assert cdar.weights["A"] > cdar.weights["B"]
    assert conditional_drawdown_at_risk(scenarios.to_numpy() @ cdar.weights.to_numpy(), 0.75) >= 0
    crash = np.concatenate([np.full(8, 0.04), np.full(4, -0.06)])
    both = pd.DataFrame({"A": crash, "B": crash})
    with pytest.raises(ValueError, match="infeasible"):
        optimize_cdar(
            both,
            expected,
            max_drawdown=0.01,
            constraints=OptimizationConstraints(max_weight=1.0),
        )


def test_path_solvers_reject_a_tracking_error_constraint() -> None:
    scenarios = pd.DataFrame({"A": [0.01, 0.02, -0.01], "B": [0.0, 0.01, 0.02]})
    expected = pd.Series({"A": 0.02, "B": 0.01})
    constraints = OptimizationConstraints(max_weight=1.0, max_tracking_error=0.1)
    with pytest.raises(ValueError, match="maximum diversification"):
        optimize_cvar(scenarios, expected, constraints=constraints)
    with pytest.raises(ValueError, match="maximum diversification"):
        optimize_multiperiod(
            pd.DataFrame([expected, expected]),
            pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "B"]),
            constraints=constraints,
        )


def test_single_asset_and_flat_series_estimators_fail_closed() -> None:
    single = pd.DataFrame({"A": [0.01, -0.02, 0.015]})
    for estimator in (
        constant_correlation_covariance,
        oas_covariance,
        random_matrix_covariance,
    ):
        covariance = estimator(single, annualization=1)
        assert covariance.shape == (1, 1)
        assert covariance.iloc[0, 0] > 0
    flat = pd.DataFrame({"A": [0.01, 0.01], "B": [0.02, 0.02]})
    with pytest.raises(ValueError, match="positive asset variance"):
        constant_correlation_covariance(flat, annualization=1)
    with pytest.raises(ValueError, match="annualization"):
        ewma_covariance(single, annualization=0)


def test_lots_stay_inside_a_binding_weight_cap() -> None:
    expected = pd.Series({"A": 0.5, "B": 0.0})
    covariance = pd.DataFrame(np.eye(2) * 0.01, index=["A", "B"], columns=["A", "B"])
    result = optimize_mean_variance(
        expected,
        covariance,
        current_weights=pd.Series(0.0, index=["A", "B"]),
        risk_aversion=0.01,
        constraints=OptimizationConstraints(max_weight=0.5, max_turnover=2.0),
        weight_steps=pd.Series(0.4, index=["A", "B"]),
    )
    assert result.weights["A"] == pytest.approx(0.4)
    assert result.weights["B"] == pytest.approx(0.4)
    assert result.weights.sum() <= 0.5 * 2 + 1e-8


def test_research_recipes_expose_the_new_modes() -> None:
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.01]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    scores = pd.Series({"A": 0.10, "B": 0.02})
    sharpe = research_allocation_weights(
        scores,
        None,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={"mode": "max_sharpe"},
        covariance_override=covariance,
    )
    diversified = research_allocation_weights(
        scores,
        None,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={"mode": "max_diversification"},
        covariance_override=covariance,
    )
    assert sharpe["A"] == pytest.approx(5 / 9, abs=1e-3)
    assert diversified["B"] > diversified["A"]
    history = pd.DataFrame({"A": [0.01] * 6, "B": [0.03, 0.03, 0.03, 0.03, -0.08, -0.08]})
    downside = research_allocation_weights(
        pd.Series({"A": 0.01, "B": 0.01}),
        history,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={
            "mode": "evar",
            "lookback": 6,
            "min_observations": 6,
            "evar_beta": 0.7,
            "risk_aversion": 8.0,
        },
    )
    assert downside.sum() == pytest.approx(1.0)
    assert downside["A"] > downside["B"]
    drawdown = research_allocation_weights(
        pd.Series({"A": 0.01, "B": 0.01}),
        history,
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={
            "mode": "cdar",
            "lookback": 6,
            "min_observations": 6,
            "cdar_beta": 0.7,
            "risk_aversion": 8.0,
        },
    )
    assert drawdown.sum() == pytest.approx(1.0)
    assert drawdown["A"] > drawdown["B"]
    ewma = research_allocation_weights(
        scores,
        pd.DataFrame(np.random.default_rng(6).normal(scale=0.01, size=(12, 2)), columns=["A", "B"]),
        None,
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={
            "mode": "max_diversification",
            "covariance_estimator": "ewma",
            "ewma_decay": 0.9,
            "lookback": 12,
            "min_observations": 12,
        },
    )
    assert ewma.sum() == pytest.approx(1.0)
