"""Unified research allocation in whole-account weights, including cash."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from quant_data_kit.allocation_protocol import METHODS, validate_v2

from quant_portfolio.risk_snapshot import RiskSnapshot, mature_ic
from quant_portfolio.solver import RiskMatrix, solve_portfolio


@dataclass(frozen=True)
class AllocationResult:
    weights: pd.Series
    diagnostics: dict = field(default_factory=dict)
    validation_context: dict = field(default_factory=dict, repr=False)

    def to_dict(self):
        return {"continuous_weights": self.weights.to_dict(), "diagnostics": self.diagnostics}


def allocate_research(
    scores,
    trailing_returns,
    current_weights,
    linear_costs,
    *,
    invested_limit,
    max_weight,
    config,
    factor_exposures=None,
    factor_bounds=None,
    covariance_override=None,
    benchmark_weights=None,
    factor_loadings=None,
    factor_covariance=None,
    specific_variance=None,
    risk_snapshot=None,
    decision_at=None,
    expected_returns=None,
    ic_labels=None,
    lower_bounds=None,
    upper_bounds=None,
    buy_costs=None,
    sell_costs=None,
    impact_adv=None,
    impact_volatility=None,
    impact_coefficient=0.0,
    portfolio_nav=None,
    market_weights=None,
    signal_cross_section=None,
    max_tracking_error_override=None,
    factor_bounds_absolute=False,
):
    from quant_portfolio.optimization import (
        OptimizationConstraints,
        _research_covariance,
        ic_vol_expected_returns,
        validate_research_allocation,
    )

    v2 = config.get("protocol_version") == 2
    cfg = validate_v2(config) if v2 else validate_research_allocation(config)
    if not isinstance(scores, pd.Series) or scores.empty or not scores.index.is_unique:
        raise ValueError("scores must be a non-empty Series with unique assets")
    scores = scores.astype(float)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    candidates = list(scores.index)
    current = pd.Series(dtype=float) if current_weights is None else current_weights.astype(float)
    if (
        not current.index.is_unique
        or not np.isfinite(current).all()
        or (current < 0).any()
        or current.sum() > 1 + 1e-10
    ):
        raise ValueError("current weights must be unique finite nonnegative account weights")
    if not 0 < invested_limit <= 1 or not 0 < max_weight <= 1:
        raise ValueError("invalid invested_limit or max_weight")
    if risk_snapshot is not None:
        if not isinstance(risk_snapshot, RiskSnapshot):
            raise TypeError("risk_snapshot must be RiskSnapshot")
        if decision_at is None:
            raise ValueError("risk snapshot requires explicit decision_at")
        if benchmark_weights is not None:
            raise ValueError("benchmark must be provided only once in risk_snapshot")
        benchmark_weights = risk_snapshot.benchmark
    benchmark = None if benchmark_weights is None else benchmark_weights.astype(float)
    if benchmark is not None and not benchmark.index.is_unique:
        raise ValueError("benchmark labels must be unique")
    assets = sorted(
        set(candidates)
        | set(current.index)
        | (set() if benchmark is None else set(benchmark.index))
    )
    current = current.reindex(assets, fill_value=0.0)
    wb = None if benchmark is None else benchmark.reindex(assets, fill_value=0.0).to_numpy()
    lo = np.zeros(len(assets))
    hi = np.array([max_weight if a in candidates else 0.0 for a in assets])
    for bounds, destination in ((lower_bounds, lo), (upper_bounds, hi)):
        if bounds is not None:
            if not bounds.index.is_unique or set(bounds.index) - set(assets):
                raise ValueError("invalid asset bound labels")
            for i, asset in enumerate(assets):
                if asset in bounds.index:
                    destination[i] = float(bounds.loc[asset])
    mode = cfg["mode"]
    risk = None
    history = None
    if trailing_returns is not None:
        history = trailing_returns.reindex(columns=assets).tail(cfg["lookback"])
        if v2:
            from quant_portfolio.risk_snapshot import timestamp

            if decision_at is None or not isinstance(history.index, pd.DatetimeIndex):
                raise ValueError("v2 history requires timestamped returns and decision_at")
            # Index must denote availability, not a label's start date.
            if any(timestamp(t) > timestamp(decision_at) for t in history.index):
                raise ValueError("return history contains unavailable observations")
    if risk_snapshot is not None:
        risk = risk_snapshot.matrix(assets, decision_at)
    elif covariance_override is not None:
        if set(assets) - set(covariance_override.index):
            raise ValueError("covariance does not cover the complete asset union")
        risk = RiskMatrix(covariance=covariance_override.loc[assets, assets].to_numpy())
    elif factor_loadings is not None:
        risk = RiskMatrix(
            exposures=factor_loadings.loc[assets].to_numpy(),
            factor_covariance=factor_covariance.loc[
                factor_loadings.columns, factor_loadings.columns
            ].to_numpy(),
            specific_variance=specific_variance.loc[assets].to_numpy(),
        )
    elif mode != "equal" or cfg.get("max_tracking_error") is not None:
        if (
            history is None
            or len(history) < cfg["min_observations"]
            or not np.isfinite(history).all().all()
        ):
            raise ValueError("complete return history for the asset union is insufficient")
        if v2:
            from quant_portfolio.optimization import estimate_covariance

            covariance = estimate_covariance(history, shrinkage=cfg["covariance_shrinkage"])
        else:
            covariance = _research_covariance(cfg, history, None, assets, None, None, None)
        risk = RiskMatrix(covariance=covariance.to_numpy())
    if (
        v2
        and risk_snapshot is not None
        and risk_snapshot.horizon_sessions != cfg["forecast_horizon_sessions"]
    ):
        raise ValueError("risk prediction horizon differs from the forecast horizon")
    horizon_scale = (
        (
            cfg["forecast_horizon_sessions"]
            / (252 if risk_snapshot is None else risk_snapshot.annualization)
        )
        if v2
        else 1.0
    )
    diag = {
        "protocol_version": 2 if v2 else 1,
        "compatibility_mode": not v2,
        "risk_objective": cfg.get("risk_objective", "absolute"),
        "risk_usage": METHODS[mode]["risk_usage"],
        "asset_union": assets,
        "invested_budget": invested_limit,
        "cash_target": 1 - invested_limit,
        "risk_horizon_scale": horizon_scale,
    }
    if risk_snapshot is not None:
        diag["risk_snapshot"] = risk_snapshot.metadata()
    mu = np.zeros(len(assets))
    if mode in {"cost_aware", "max_sharpe", "cvar", "evar", "cdar"}:
        if v2:
            if cfg["expected_return_model"] == "provided":
                if expected_returns is None or not expected_returns.index.is_unique:
                    raise ValueError("provided return forecasts are required")
                forecasts = expected_returns.reindex(candidates).to_numpy(dtype=float)
            else:
                if risk is None:
                    raise ValueError("ic_vol requires risk inputs")
                ic, evidence = mature_ic(
                    ic_labels,
                    decision_at=decision_at,
                    horizon_sessions=cfg["forecast_horizon_sessions"],
                    lookback=cfg["ic_lookback"],
                    min_periods=cfg["ic_min_periods"],
                )
                diag.update(evidence, information_coefficient=ic)
                cross_section = (
                    scores if signal_cross_section is None else signal_cross_section.astype(float)
                )
                if (
                    not cross_section.index.is_unique
                    or not np.isfinite(cross_section).all()
                    or set(candidates) - set(cross_section.index)
                ):
                    raise ValueError(
                        "standardization universe must cover candidates with finite signals"
                    )
                raw = scores.to_numpy()
                scale = float(cross_section.std(ddof=0))
                z = (
                    np.zeros(len(scores))
                    if scale == 0
                    else (raw - float(cross_section.mean())) / scale
                )
                diag["signal_standardization_assets"] = len(cross_section)
                diag["constant_signal_cross_section"] = scale == 0
                vol = pd.Series(np.sqrt(risk.diagonal() * horizon_scale), index=assets)
                forecasts = ic * vol.loc[candidates].to_numpy() * z
            if not np.isfinite(forecasts).all():
                raise ValueError("return forecasts must cover all candidates")
            mu = pd.Series(forecasts, index=candidates).reindex(assets, fill_value=0).to_numpy()
        elif mode == "cost_aware" and cfg["expected_return_model"] != "score":
            if cfg["expected_return_model"] == "black_litterman":
                raise ValueError(
                    "Black-Litterman requires an explicit views API and full market weights"
                )
            mu = ic_vol_expected_returns(
                scores.reindex(assets, fill_value=0),
                pd.Series(np.sqrt(risk.diagonal()), index=assets),
                float(cfg["information_coefficient"]),
            ).to_numpy()
        else:
            mu = scores.reindex(assets, fill_value=0).to_numpy()
    x, bounds = None, dict(factor_bounds or {})
    if bounds:
        if factor_exposures is None:
            raise ValueError("factor bounds require exposures")
        x = factor_exposures.reindex(index=assets, columns=list(bounds)).to_numpy()
        if not np.isfinite(x).all():
            raise ValueError("constraint exposures must cover the asset union")
        if cfg.get("factor_bound_reference") == "active" and not factor_bounds_absolute:
            if wb is None:
                raise ValueError("active factor bounds require full benchmark")
            baseline = x.T @ wb
            bounds = {
                key: (pair[0] + baseline[i], pair[1] + baseline[i])
                for i, (key, pair) in enumerate(bounds.items())
            }
    target = None
    if METHODS[mode]["objective"] == "target":
        if mode == "equal":
            raw = pd.Series(1 / len(candidates), index=candidates)
        elif mode == "inverse_vol":
            vol = pd.Series(np.sqrt(risk.diagonal()), index=assets).loc[candidates]
            if (vol <= 0).any():
                raise ValueError("inverse_vol requires positive variance")
            raw = 1 / vol
            raw /= raw.sum()
        else:
            from quant_portfolio.methods import (
                equal_risk_contribution_weights,
                hierarchical_risk_parity_weights,
            )

            if risk.covariance is None:
                raise ValueError(
                    f"{mode} target construction requires explicit dense candidate covariance"
                )
            cov = pd.DataFrame(risk.covariance, index=assets, columns=assets).loc[
                candidates, candidates
            ]
            raw = (
                equal_risk_contribution_weights(cov)
                if mode == "risk_parity"
                else hierarchical_risk_parity_weights(cov)
            )
        target = raw.reindex(assets, fill_value=0).to_numpy() * invested_limit

    def cost_vector(value):
        if value is None:
            return None
        vector = value.reindex(assets)
        # Liquidations need a cost estimate too; never silently assign zero.
        needed = (current > 0) | pd.Series([a in candidates for a in assets], index=assets)
        if vector[needed].isna().any():
            raise ValueError("cost rates must cover candidates and all held assets")
        return vector.fillna(0).to_numpy(dtype=float)

    impact = None
    if impact_coefficient or impact_adv is not None or impact_volatility is not None:
        from quant_portfolio.optimization import _validated_impact

        impact = _validated_impact(
            assets, impact_adv, impact_volatility, impact_coefficient, portfolio_nav
        )
    tracking_limits = [
        value
        for value in (cfg.get("max_tracking_error"), max_tracking_error_override)
        if value is not None
    ]
    settings = OptimizationConstraints(
        max_weight=max_weight,
        max_turnover=cfg["max_turnover"],
        max_tracking_error=min(tracking_limits) if tracking_limits else None,
    )
    scenarios = None if history is None else history.to_numpy()
    if v2 and mode in {"cvar", "evar", "cdar"}:
        horizon = cfg["forecast_horizon_sessions"]
        blocks = len(history) // horizon
        if blocks < cfg["min_observations"]:
            raise ValueError("insufficient non-overlapping horizon scenarios")
        scenarios = (
            np.prod(
                1 + scenarios[-blocks * horizon :].reshape(blocks, horizon, len(assets)), axis=1
            )
            - 1
        )
    solved = solve_portfolio(
        mu,
        risk,
        current.to_numpy(),
        settings,
        assets=assets,
        budget=invested_limit,
        lower=lo,
        upper=hi,
        factor_exposures=x,
        factor_bounds=bounds,
        benchmark=wb,
        active_risk=diag["risk_objective"] == "active",
        kind=METHODS[mode]["objective"],
        target=target,
        linear_costs=cost_vector(linear_costs),
        buy_costs=cost_vector(buy_costs),
        sell_costs=cost_vector(sell_costs),
        risk_aversion=cfg["risk_aversion"],
        risk_scale=horizon_scale,
        turnover_penalty=cfg["turnover_penalty"],
        impact=impact,
        scenarios=scenarios,
        beta=cfg.get(mode + "_beta", 0.95),
        risk_free_rate=cfg.get("risk_free_rate", 0),
        timeout_seconds=cfg.get("timeout_seconds", 30),
    )
    w = solved.weights
    diag.update(
        solved.diagnostics, expected_return=float(mu @ w), turnover=float(np.abs(w - current).sum())
    )
    diag["versions"] = {
        name: importlib.metadata.version(name) for name in ("cvxpy", "osqp", "clarabel", "numpy")
    }
    if risk is not None:
        diag["annualized_volatility"] = float(np.sqrt(max(risk.variance(w), 0)))
        if risk_snapshot is not None and risk_snapshot.exposures is not None:
            names = list(risk_snapshot.exposures.columns)
            diag["factor_exposures"] = dict(zip(names, (risk.exposures.T @ w).tolist()))
            if wb is not None:
                diag["active_exposures"] = dict(zip(names, (risk.exposures.T @ (w - wb)).tolist()))
        if wb is not None:
            diag["annualized_tracking_error"] = float(np.sqrt(max(risk.variance(w - wb), 0)))
        if target is not None:

            def contributions(point):
                marginal = (
                    risk.covariance @ point
                    if risk.covariance is not None
                    else risk.exposures @ (risk.factor_covariance @ (risk.exposures.T @ point))
                    + risk.specific_variance * point
                )
                return (point * marginal).tolist()

            diag["target_risk_contributions"] = contributions(target)
            diag["repaired_risk_contributions"] = contributions(w)
    result = pd.Series(w, index=assets, name="weight")
    result.attrs["allocation_diagnostics"] = diag
    return AllocationResult(
        result,
        diag,
        {
            "assets": assets,
            "risk": risk,
            "benchmark": wb,
            "settings": settings,
            "factor_exposures": x,
            "factor_bounds": bounds,
            "current": current.to_numpy(),
        },
    )
