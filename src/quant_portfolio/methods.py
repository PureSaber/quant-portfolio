"""Estimators that close the gaps around the single-period mean-variance book.

The optimizers stay in ``optimization.py``. This module only builds the inputs
they were missing: a data-driven covariance, a factor covariance, expected
returns that are not raw scores, equal risk contributions, and hierarchical
risk parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def repair_covariance(covariance: np.ndarray) -> np.ndarray:
    """Lift non-positive eigenvalues and return a symmetric matrix."""

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * 1e-10, 1e-12)
    repaired = eigenvectors @ np.diag(np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    return 0.5 * (repaired + repaired.T)


def ledoit_wolf_covariance(returns: pd.DataFrame, *, annualization: float) -> pd.DataFrame:
    """Ledoit-Wolf (2004) covariance, shrunk toward a scaled identity.

    The shrinkage intensity is the analytical formula in Ledoit and Wolf,
    Journal of Multivariate Analysis (2004). The sample second moment uses the
    ``1 / T`` divisor from that paper, then the matrix is multiplied by
    ``annualization``.
    """

    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    values = returns.to_numpy(dtype=float)
    observations, assets = values.shape
    if observations < 2 or assets < 1:
        raise ValueError("ledoit_wolf requires at least two observations and one asset")
    if not np.isfinite(values).all():
        raise ValueError("ledoit_wolf requires complete finite observations")
    centered = values - values.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / observations
    if assets == 1:
        shrunk = sample
    else:
        shrinkage = _ledoit_wolf_shrinkage(centered)
        target = float(np.trace(sample) / assets)
        shrunk = (1.0 - shrinkage) * sample
        shrunk.flat[:: assets + 1] += shrinkage * target
    shrunk = repair_covariance(shrunk * float(annualization))
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def _ledoit_wolf_shrinkage(centered: np.ndarray) -> float:
    observations, assets = centered.shape
    squared = centered**2
    variance = np.sum(squared, axis=0) / observations
    mu = float(np.sum(variance) / assets)
    beta_total = float(np.sum(squared.T @ squared))
    second_moment = float(np.sum((centered.T @ centered) ** 2) / observations**2)
    beta = (beta_total / observations - second_moment) / (assets * observations)
    delta = second_moment - 2.0 * mu * float(variance.sum()) + assets * mu**2
    delta /= assets
    if delta <= 0:
        return 0.0
    beta = min(beta, delta)
    if beta <= 0:
        return 0.0
    return float(min(beta / delta, 1.0))


def factor_model_covariance(
    loadings: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_variance: pd.Series,
) -> pd.DataFrame:
    """Barra-style covariance ``X F X' + diag(D)``."""

    if not isinstance(loadings, pd.DataFrame):
        raise TypeError("factor loadings must be a pandas DataFrame")
    if not isinstance(factor_covariance, pd.DataFrame):
        raise TypeError("factor covariance must be a pandas DataFrame")
    if not isinstance(specific_variance, pd.Series):
        raise TypeError("specific variance must be a pandas Series")
    if loadings.empty or loadings.shape[1] == 0:
        raise ValueError("factor loadings must contain at least one asset and one factor")
    if not loadings.index.is_unique or not loadings.columns.is_unique:
        raise ValueError("factor loading labels must be unique")
    numeric_loadings = loadings.apply(pd.to_numeric, errors="coerce")
    if numeric_loadings.isna().any().any() or not np.isfinite(numeric_loadings.to_numpy()).all():
        raise ValueError("factor loadings must be finite")
    factors = list(numeric_loadings.columns)
    aligned = factor_covariance.reindex(index=factors, columns=factors)
    numeric_factor = aligned.apply(pd.to_numeric, errors="coerce")
    if numeric_factor.isna().any().any() or not np.isfinite(numeric_factor.to_numpy()).all():
        raise ValueError("factor covariance must be finite for every loading factor")
    factor_values = numeric_factor.to_numpy(dtype=float)
    if not np.allclose(factor_values, factor_values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("factor covariance must be symmetric")
    if float(np.linalg.eigvalsh(factor_values).min()) < -1e-10:
        raise ValueError("factor covariance must be positive semidefinite")
    assets = list(numeric_loadings.index)
    if not specific_variance.index.is_unique:
        raise ValueError("specific variance index must be unique")
    specific = pd.to_numeric(specific_variance.reindex(assets), errors="coerce")
    if specific.isna().any() or not np.isfinite(specific.to_numpy()).all() or (specific < 0).any():
        raise ValueError("specific variance must be finite and non-negative for every asset")
    exposure = numeric_loadings.to_numpy(dtype=float)
    covariance = exposure @ factor_values @ exposure.T
    covariance[np.diag_indices_from(covariance)] += specific.to_numpy(dtype=float)
    covariance = repair_covariance(covariance)
    return pd.DataFrame(covariance, index=assets, columns=assets)


def ic_vol_expected_returns(
    scores: pd.Series,
    volatility: pd.Series,
    information_coefficient: float,
) -> pd.Series:
    """Grinold mapping ``IC * volatility * z-score``.

    ``scores`` are cross-sectionally standardized with a population standard
    deviation. ``volatility`` must be the horizon volatility that matches the
    covariance used downstream.
    """

    if not isinstance(scores, pd.Series) or not scores.index.is_unique or scores.empty:
        raise ValueError("scores must be a non-empty Series with unique assets")
    if (
        isinstance(information_coefficient, bool)
        or not isinstance(information_coefficient, (int, float))
        or not np.isfinite(information_coefficient)
        or not 0 < float(information_coefficient) <= 1
    ):
        raise ValueError("information_coefficient must be in (0, 1]")
    numeric_scores = pd.to_numeric(scores, errors="coerce").astype(float)
    if not np.isfinite(numeric_scores).all():
        raise ValueError("scores must be finite")
    dispersion = float(numeric_scores.std(ddof=0))
    if dispersion <= 0:
        raise ValueError("ic_vol requires scores with positive cross-sectional dispersion")
    standardized = (numeric_scores - float(numeric_scores.mean())) / dispersion
    if not isinstance(volatility, pd.Series) or not volatility.index.is_unique:
        raise ValueError("volatility must be a Series with unique assets")
    aligned = pd.to_numeric(volatility.reindex(numeric_scores.index), errors="coerce")
    if aligned.isna().any() or not np.isfinite(aligned.to_numpy()).all() or (aligned <= 0).any():
        raise ValueError("volatility must be positive and finite for every score")
    expected = float(information_coefficient) * aligned * standardized
    expected.name = "expected_return"
    return expected


def black_litterman_expected_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    views: pd.Series,
    *,
    risk_aversion: float,
    tau: float = 0.05,
    view_matrix: pd.DataFrame | None = None,
) -> pd.Series:
    """Black-Litterman (1992) posterior expected returns.

    Equilibrium returns are ``risk_aversion * Σ w_market``. Absolute views use
    an identity pick matrix. ``view_matrix`` is an optional views-by-asset pick
    matrix. View uncertainty is the diagonal of ``P (tau Σ) P'``.
    """

    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if not isinstance(market_weights, pd.Series) or not market_weights.index.is_unique:
        raise ValueError("market_weights must be a Series with unique assets")
    if not isinstance(views, pd.Series) or not views.index.is_unique or views.empty:
        raise ValueError("views must be a non-empty Series with unique names")
    if (
        isinstance(risk_aversion, bool)
        or not isinstance(risk_aversion, (int, float))
        or not np.isfinite(risk_aversion)
        or risk_aversion <= 0
    ):
        raise ValueError("risk_aversion must be positive and finite")
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or not np.isfinite(tau)
        or not 0 < float(tau) <= 1
    ):
        raise ValueError("tau must be in (0, 1]")
    assets = list(covariance.index)
    if list(covariance.columns) != assets or not covariance.index.is_unique:
        raise ValueError("covariance labels must be unique and identical on both axes")
    cov = covariance.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(cov).all() or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if float(np.linalg.eigvalsh(cov).min()) <= 0:
        raise ValueError("Black-Litterman requires a positive definite covariance")
    weights = pd.to_numeric(market_weights.reindex(assets), errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(weights).all()
        or (weights < 0).any()
        or abs(float(weights.sum()) - 1) > 1e-8
    ):
        raise ValueError("market_weights must be finite, non-negative and sum to one")
    if view_matrix is None:
        if list(views.index) != assets and set(views.index) != set(assets):
            raise ValueError("absolute views must name exactly the covariance assets")
        pick = np.eye(len(assets))
        view_values = pd.to_numeric(views.reindex(assets), errors="coerce").to_numpy(dtype=float)
    else:
        if not isinstance(view_matrix, pd.DataFrame):
            raise TypeError("view_matrix must be a pandas DataFrame")
        if list(view_matrix.index) != list(views.index) or list(view_matrix.columns) != assets:
            raise ValueError("view_matrix rows must match views and columns must match assets")
        pick = view_matrix.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        view_values = pd.to_numeric(views, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(pick).all() or not np.isfinite(view_values).all():
        raise ValueError("views and the pick matrix must be finite")
    tau_cov = float(tau) * cov
    uncertainty = np.diag(pick @ tau_cov @ pick.T)
    if not np.isfinite(uncertainty).all() or (uncertainty <= 0).any():
        raise ValueError("view uncertainty must be positive for every view")
    inverse_uncertainty = np.diag(1.0 / uncertainty)
    equilibrium = float(risk_aversion) * (cov @ weights)
    left = np.linalg.solve(tau_cov, np.eye(len(assets))) + pick.T @ inverse_uncertainty @ pick
    right = np.linalg.solve(tau_cov, equilibrium) + pick.T @ inverse_uncertainty @ view_values
    posterior = np.linalg.solve(left, right)
    if not np.isfinite(posterior).all():
        raise ValueError("Black-Litterman posterior is not finite")
    return pd.Series(posterior, index=assets, name="expected_return")


def equal_risk_contribution_weights(
    covariance: pd.DataFrame,
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-12,
) -> pd.Series:
    """Long-only equal risk contribution (Maillard, Roncalli, Teiletche).

    Correlated assets share one risk budget. The fixed point is
    ``w_i (Σw)_i`` equal across assets and ``sum(w) = 1``.
    """

    cov, assets = _validated_covariance(covariance)
    if (np.diag(cov) <= 0).any():
        raise ValueError("equal risk contribution requires positive asset variance")
    weights = np.full(len(assets), 1.0 / len(assets))
    converged = False
    for _ in range(max_iterations):
        contribution = weights * (cov @ weights)
        if not np.isfinite(contribution).all() or (contribution <= 0).any():
            raise ValueError("equal risk contribution produced a non-positive risk contribution")
        target = weights * (float(contribution.mean()) / contribution)
        target /= target.sum()
        updated = 0.5 * weights + 0.5 * target
        if float(np.max(np.abs(updated - weights))) <= tolerance:
            weights = updated
            converged = True
            break
        weights = updated
    if not converged:
        raise RuntimeError(
            f"equal risk contribution did not converge after {max_iterations} iterations"
        )
    contribution = weights * (cov @ weights)
    gap = float(np.max(np.abs(contribution - contribution.mean())))
    if gap > max(1e-6, 1e-4 * float(np.abs(contribution.mean()))):
        raise RuntimeError("equal risk contribution did not equalize risk contributions")
    return pd.Series(weights, index=assets, name="weight")


def hierarchical_risk_parity_weights(covariance: pd.DataFrame) -> pd.Series:
    """Lopez de Prado hierarchical risk parity from a covariance matrix.

    Correlation distance, single-linkage order, then recursive inverse-variance
    bisection. Weights are positive and sum to one.
    """

    cov, assets = _validated_covariance(covariance)
    variance = np.diag(cov).astype(float)
    if (variance <= 0).any() or not np.isfinite(variance).all():
        raise ValueError("hierarchical risk parity requires positive asset variance")
    scale = np.sqrt(variance)
    correlation = cov / np.outer(scale, scale)
    correlation = np.clip(correlation, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    distance = np.sqrt(np.maximum(0.5 * (1.0 - correlation), 0.0))
    root, children, members = _linkage_tree(distance)
    weights = np.ones(len(assets))
    _hrp_bisect(weights, cov, root, children, members)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("hierarchical risk parity produced a non-positive weight")
    weights /= weights.sum()
    return pd.Series(weights, index=assets, name="weight")


def square_root_impact_penalty(
    trade: np.ndarray,
    average_daily_value: np.ndarray,
    daily_volatility: np.ndarray,
    *,
    impact_coefficient: float,
    portfolio_nav: float,
    smoothing: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Square-root impact as a return drag, plus a smoothed weight gradient.

    Currency cost matches ``coefficient * volatility * notional * sqrt(notional / ADV)``.
    Dividing by NAV puts that cost next to expected return. The gradient uses
    ``sqrt(trade^2 + smoothing)`` so a zero trade has a finite derivative.
    """

    if (
        isinstance(impact_coefficient, bool)
        or not isinstance(impact_coefficient, (int, float))
        or not np.isfinite(impact_coefficient)
        or impact_coefficient < 0
    ):
        raise ValueError("impact_coefficient must be finite and non-negative")
    if (
        isinstance(portfolio_nav, bool)
        or not isinstance(portfolio_nav, (int, float))
        or not np.isfinite(portfolio_nav)
        or portfolio_nav <= 0
    ):
        raise ValueError("portfolio_nav must be positive and finite")
    if (
        isinstance(smoothing, bool)
        or not isinstance(smoothing, (int, float))
        or not np.isfinite(smoothing)
        or smoothing <= 0
    ):
        raise ValueError("smoothing must be positive and finite")
    adv = np.asarray(average_daily_value, dtype=float)
    vol = np.asarray(daily_volatility, dtype=float)
    delta = np.asarray(trade, dtype=float)
    if adv.shape != delta.shape or vol.shape != delta.shape:
        raise ValueError("impact inputs must match the trade vector")
    if (
        not np.isfinite(adv).all()
        or not np.isfinite(vol).all()
        or (adv <= 0).any()
        or (vol < 0).any()
    ):
        raise ValueError("ADV must be positive and volatility must be non-negative")
    notional = np.abs(delta) * float(portfolio_nav)
    exact = impact_coefficient * vol * notional * np.sqrt(notional / adv) / float(portfolio_nav)
    scale = impact_coefficient * vol * np.sqrt(float(portfolio_nav) / adv)
    smooth = delta * delta + float(smoothing)
    gradient = scale * 1.5 * delta * smooth**-0.25
    return float(np.sum(exact)), gradient


def _complete_return_matrix(returns: pd.DataFrame) -> np.ndarray:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] == 0 or not returns.columns.is_unique:
        raise ValueError("returns must contain at least one uniquely labeled asset")
    if returns.shape[0] < 2:
        raise ValueError("covariance estimators require at least two observations")
    values = returns.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("covariance estimators require complete finite observations")
    return values


def _annualized(covariance: np.ndarray, annualization: float, index: pd.Index) -> pd.DataFrame:
    if (
        isinstance(annualization, bool)
        or not isinstance(annualization, (int, float))
        or not np.isfinite(annualization)
        or annualization <= 0
    ):
        raise ValueError("annualization must be positive and finite")
    repaired = repair_covariance(covariance * float(annualization))
    return pd.DataFrame(repaired, index=index, columns=index)


def ewma_covariance(
    returns: pd.DataFrame,
    *,
    decay: float = 0.94,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """RiskMetrics EWMA covariance with normalized zero-mean outer products.

    The latest observation has weight proportional to ``1 - decay``. Weights are
    rescaled to sum to one. Returns are not demeaned.
    """

    if (
        isinstance(decay, bool)
        or not isinstance(decay, (int, float))
        or not np.isfinite(decay)
        or not 0 < float(decay) < 1
    ):
        raise ValueError("decay must be in (0, 1)")
    values = _complete_return_matrix(returns)
    observations = values.shape[0]
    ages = np.arange(observations - 1, -1, -1, dtype=float)
    weights = (1.0 - float(decay)) * float(decay) ** ages
    weights /= weights.sum()
    scaled = values * np.sqrt(weights)[:, None]
    return _annualized(scaled.T @ scaled, annualization, returns.columns)


def constant_correlation_covariance(
    returns: pd.DataFrame,
    *,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage toward the constant-correlation target.

    The target keeps sample variances and replaces off-diagonal correlations
    with their sample average. The intensity is the Ledoit-Wolf (2004)
    estimator ``max(0, min(1, kappa / T))``.
    """

    values = _complete_return_matrix(returns)
    observations, assets = values.shape
    centered = values - values.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / observations
    if assets == 1:
        return _annualized(sample, annualization, returns.columns)
    variance = np.diag(sample).reshape(-1, 1)
    if np.any(variance <= 0):
        raise ValueError("constant correlation requires positive asset variance")
    scale = np.sqrt(variance)
    unit = scale @ scale.T
    average_correlation = ((sample / unit).sum() - assets) / (assets * (assets - 1))
    prior = average_correlation * unit
    np.fill_diagonal(prior, variance.ravel())
    squared = centered**2
    phi_matrix = (squared.T @ squared) / observations - sample**2
    gamma = float(np.sum((sample - prior) ** 2))
    if gamma <= 0:
        shrinkage = 0.0
    else:
        theta = ((centered**3).T @ centered) / observations - variance * sample
        np.fill_diagonal(theta, 0.0)
        variance_ratio = scale.T / scale
        rho = float(
            np.diag(phi_matrix).sum() + average_correlation * np.sum(theta * variance_ratio)
        )
        kappa = (float(phi_matrix.sum()) - rho) / gamma
        shrinkage = float(min(max(kappa / observations, 0.0), 1.0))
    shrunk = shrinkage * prior + (1.0 - shrinkage) * sample
    return _annualized(shrunk, annualization, returns.columns)


def oas_covariance(returns: pd.DataFrame, *, annualization: float = 252.0) -> pd.DataFrame:
    """Oracle Approximating Shrinkage toward a scaled identity.

    The intensity is the Chen, Wiesel, Eldar and Hero (2010) formula, clipped
    to ``[0, 1]``. The sample second moment uses the ``1 / T`` divisor.
    """

    values = _complete_return_matrix(returns)
    observations, assets = values.shape
    centered = values - values.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / observations
    if assets == 1:
        return _annualized(sample, annualization, returns.columns)
    trace = float(np.trace(sample))
    trace_square = float(np.sum(sample * sample))
    numerator = (1.0 - 2.0 / assets) * trace_square + trace**2
    denominator = (observations + 1.0 - 2.0 / assets) * (trace_square - trace**2 / assets)
    if denominator <= 0:
        shrinkage = 1.0
    else:
        shrinkage = float(min(max(numerator / denominator, 0.0), 1.0))
    target = trace / assets
    shrunk = (1.0 - shrinkage) * sample
    shrunk.flat[:: assets + 1] += shrinkage * target
    return _annualized(shrunk, annualization, returns.columns)


def random_matrix_covariance(
    returns: pd.DataFrame,
    *,
    annualization: float = 252.0,
) -> pd.DataFrame:
    """Marchenko-Pastur denoising of the correlation matrix.

    Eigenvalues at or below ``(1 + sqrt(N / T)) ** 2`` are replaced by their
    average. The result is rescaled to a correlation matrix and then back to
    the sample variances. Noise variance is fixed at 1, the correlation scale.
    """

    values = _complete_return_matrix(returns)
    observations, assets = values.shape
    centered = values - values.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / observations
    if assets == 1:
        return _annualized(sample, annualization, returns.columns)
    std = np.sqrt(np.diag(sample))
    if np.any(std <= 0):
        raise ValueError("random matrix denoising requires positive asset variance")
    correlation = sample / np.outer(std, std)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    upper = (1.0 + np.sqrt(assets / observations)) ** 2
    noise = eigenvalues <= upper
    if noise.any() and not noise.all():
        cleaned = eigenvalues.copy()
        cleaned[noise] = float(eigenvalues[noise].mean())
        denoised = eigenvectors @ np.diag(cleaned) @ eigenvectors.T
        rescale = np.sqrt(np.clip(np.diag(denoised), 1e-18, None))
        denoised = denoised / np.outer(rescale, rescale)
        np.fill_diagonal(denoised, 1.0)
    else:
        denoised = correlation
    covariance = denoised * np.outer(std, std)
    return _annualized(covariance, annualization, returns.columns)


def _validated_covariance(covariance: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if not isinstance(covariance, pd.DataFrame):
        raise TypeError("covariance must be a pandas DataFrame")
    if covariance.empty or not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("covariance must be non-empty with unique labels")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError("covariance labels must match on both axes")
    values = covariance.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.allclose(values, values.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be finite and symmetric")
    if float(np.linalg.eigvalsh(values).min()) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return values, [str(asset) for asset in covariance.index]


def _linkage_tree(
    distance: np.ndarray,
) -> tuple[int, dict[int, tuple[int, int]], dict[int, list[int]]]:
    size = distance.shape[0]
    members: dict[int, list[int]] = {index: [index] for index in range(size)}
    children: dict[int, tuple[int, int]] = {}
    active = set(range(size))
    next_id = size
    while len(active) > 1:
        left, right = min(
            ((first, second) for first in active for second in active if first < second),
            key=lambda item: (
                min(distance[i, j] for i in members[item[0]] for j in members[item[1]]),
                item[0],
                item[1],
            ),
        )
        members[next_id] = members[left] + members[right]
        children[next_id] = (left, right)
        active.remove(left)
        active.remove(right)
        active.add(next_id)
        next_id += 1
    return next(iter(active)), children, members


def _hrp_bisect(
    weights: np.ndarray,
    covariance: np.ndarray,
    node: int,
    children: dict[int, tuple[int, int]],
    members: dict[int, list[int]],
) -> None:
    if node not in children:
        return
    left, right = children[node]
    left_members = members[left]
    right_members = members[right]
    left_variance = _cluster_variance(covariance, left_members)
    right_variance = _cluster_variance(covariance, right_members)
    total = left_variance + right_variance
    if total <= 0:
        raise ValueError("hierarchical risk parity cluster variance must be positive")
    left_allocation = 1.0 - left_variance / total
    weights[left_members] *= left_allocation
    weights[right_members] *= 1.0 - left_allocation
    _hrp_bisect(weights, covariance, left, children, members)
    _hrp_bisect(weights, covariance, right, children, members)


def _cluster_variance(covariance: np.ndarray, members: list[int]) -> float:
    block = covariance[np.ix_(members, members)]
    inverse_variance = 1.0 / np.diag(block)
    allocation = inverse_variance / inverse_variance.sum()
    return float(allocation @ block @ allocation)
