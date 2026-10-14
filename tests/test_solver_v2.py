from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from quant_portfolio.allocation_v2 import allocate_research
from quant_portfolio.executable import repair_integer_target
from quant_portfolio.optimization import (
    OptimizationConstraints,
    _project_box_simplex,
    research_allocation_weights,
)
from quant_portfolio.risk_snapshot import RiskSnapshot, mature_ic
from quant_portfolio.solver import OptimizationError, RiskMatrix, solve_portfolio


def test_projection_kkt_regression():
    y = np.array([-0.470931, 0.461829, -0.810164, 1.860959, -1.46164, -0.256543])
    x = _project_box_simplex(y, 0, 0.35)
    assert x == pytest.approx([0.042806, 0.35, 0, 0.35, 0, 0.257194], abs=1e-6)
    assert x.sum() == pytest.approx(1, abs=1e-12)
    free = (x > 1e-12) & (x < 0.35 - 1e-12)
    theta = (y - x)[free][0]
    assert x == pytest.approx(np.clip(y - theta, 0, 0.35), abs=1e-12)


@pytest.mark.parametrize("case", [0, 6, 7])
def test_legacy_reference_retains_fixed_six_asset_cases(case):
    from quant_portfolio.optimization import optimize_mean_variance

    rng = np.random.default_rng(20261014)
    for _ in range(case + 1):
        a = rng.normal(size=(6, 6))
        cov = 0.02 * np.eye(6) + a @ a.T * 0.005 / 6
        mu, cost = rng.normal(0.05, 0.06, 6), rng.uniform(0.001, 0.005, 6)
    assets = list("ABCDEF")
    common = {
        "current_weights": pd.Series(1 / 6, index=assets),
        "linear_costs": pd.Series(cost, index=assets),
        "constraints": OptimizationConstraints(
            max_weight=0.35,
            max_turnover=0.6,
            group_by_asset={a: "G" if i < 3 else "H" for i, a in enumerate(assets)},
            group_caps={"G": 0.55},
        ),
        "factor_exposures": pd.DataFrame({"style": np.linspace(-1, 1, 6)}, index=assets),
        "factor_bounds": {"style": (-0.15, 0.15)},
    }
    expected, covariance = (
        pd.Series(mu, index=assets),
        pd.DataFrame(cov, index=assets, columns=assets),
    )
    reference = optimize_mean_variance(expected, covariance, solver_backend="legacy", **common)
    production = optimize_mean_variance(expected, covariance, **common)
    assert reference.converged
    assert reference.objective == pytest.approx(production.objective, abs=1e-8)


@pytest.mark.parametrize(
    "mode",
    [
        "equal",
        "inverse_vol",
        "risk_parity",
        "hrp",
        "cost_aware",
        "max_sharpe",
        "max_diversification",
        "cvar",
        "evar",
        "cdar",
    ],
)
def test_v2_all_methods_solve_real_current_joint_constraints(mode):
    rng = np.random.default_rng(9)
    assets = list("ABCD")
    history = pd.DataFrame(
        rng.normal(0.001, 0.01, (80, 4)),
        index=pd.date_range("2025-01-01", periods=80, tz="UTC"),
        columns=assets,
    )
    uses = mode in {"cost_aware", "max_sharpe", "cvar", "evar", "cdar"}
    result = allocate_research(
        pd.Series([1.0, 2.0, 3.0, 4.0], index=assets),
        history,
        pd.Series(0.2, index=assets),
        None,
        invested_limit=0.8,
        max_weight=0.35,
        config={
            "protocol_version": 2,
            "mode": mode,
            "risk_objective": "absolute",
            "forecast_horizon_sessions": 1,
            "expected_return_model": "provided" if uses else "none",
            "max_turnover": 0.2,
        },
        expected_returns=pd.Series(0.001, index=assets),
        decision_at=history.index[-1],
    )
    assert result.weights.sum() == pytest.approx(0.8)
    assert abs(result.weights - 0.2).sum() <= 0.2 + 1e-7
    assert result.weights.max() <= 0.35 + 1e-7
    assert result.diagnostics["compatibility_mode"] is False
    if not uses and mode != "max_diversification":
        assert "repaired_risk_contributions" in result.diagnostics or mode == "equal"


@pytest.mark.parametrize("case", range(200))
def test_200_small_problems_against_independent_slsqp(case):
    # Cases 0, 6 and 7 include the original false-infeasibility regressions.
    rng = np.random.default_rng(20261014)
    for _ in range(case + 1):
        matrix = rng.normal(size=(6, 6))
        cov = 0.02 * np.eye(6) + matrix @ matrix.T * 0.005 / 6
        mu = rng.normal(0.05, 0.06, 6)
        cost = rng.uniform(0.001, 0.005, 6)
    current, factor = np.ones(6) / 6, np.linspace(-1, 1, 6)
    settings = OptimizationConstraints(
        max_weight=0.35,
        max_turnover=0.6,
        group_by_asset={i: "G" if i < 3 else "H" for i in range(6)},
        group_caps={"G": 0.55},
    )
    got = solve_portfolio(
        mu,
        RiskMatrix(covariance=cov),
        current,
        settings,
        linear_costs=cost,
        factor_exposures=factor[:, None],
        factor_bounds={"style": (-0.15, 0.15)},
    )

    def loss(z):
        w, t = z[:6], z[6:]
        return -mu @ w + 2.5 * w @ cov @ w + cost @ t

    def inequalities(z):
        w, t = z[:6], z[6:]
        return np.r_[
            t - w + current,
            t + w - current,
            0.6 - t.sum(),
            0.55 - w[:3].sum(),
            0.15 - factor @ w,
            0.15 + factor @ w,
        ]

    oracle = minimize(
        loss,
        np.r_[current, np.zeros(6)],
        method="SLSQP",
        jac=lambda z: np.r_[-mu + 5 * cov @ z[:6], cost],
        bounds=[(0, 0.35)] * 6 + [(0, None)] * 6,
        constraints=[
            {"type": "eq", "fun": lambda z: z[:6].sum() - 1},
            {"type": "ineq", "fun": inequalities},
        ],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    assert oracle.success, oracle.message
    assert inequalities(oracle.x).min() >= -1e-7
    assert abs(got.objective + oracle.fun) <= 1e-6 * max(abs(oracle.fun), 1e-6)
    assert got.diagnostics["max_constraint_residual"] <= 1e-7


def test_turnover_limited_sharpe_matches_grid_reference():
    assets = list("ABC")
    mu = pd.Series([0.12, 0.08, 0.04], index=assets)
    cov = pd.DataFrame(
        [[0.04, 0.008, 0.002], [0.008, 0.01, 0.001], [0.002, 0.001, 0.09]],
        index=assets,
        columns=assets,
    )
    w = research_allocation_weights(
        mu,
        None,
        pd.Series([0.0, 0.0, 1.0], index=assets),
        None,
        invested_limit=1.0,
        max_weight=1.0,
        config={"mode": "max_sharpe", "max_turnover": 0.4},
        covariance_override=cov,
    )
    assert w.to_numpy() == pytest.approx([0.2, 0, 0.8], abs=2e-6)
    grid = np.array([[a, 0.2 - a, 0.8] for a in np.linspace(0, 0.2, 2001)])
    ratios = (grid @ mu.to_numpy()) / np.sqrt(np.einsum("ij,ij->i", grid @ cov.to_numpy(), grid))
    actual = float(mu @ w) / np.sqrt(float(w @ cov @ w))
    assert actual >= ratios.max() - 1e-8


def risk_snapshot(assets, *, benchmark=None, exposures=None, available="2026-01-02T07:00Z"):
    x = (
        pd.DataFrame({"market": np.ones(len(assets))}, index=assets)
        if exposures is None
        else exposures
    )
    return RiskSnapshot(
        estimated_at="2026-01-02T07:00Z",
        available_at=available,
        horizon_sessions=1,
        annualization=252,
        model_version="test-v1",
        input_digest="test-input",
        exposures=x,
        factor_covariance=pd.DataFrame(
            np.eye(x.shape[1]) * 0.01, index=x.columns, columns=x.columns
        ),
        specific_variance=pd.Series(0.04, index=assets),
        benchmark=benchmark,
    )


def test_factor_dense_risk_and_gradient_agree_without_materialization():
    rng = np.random.default_rng(4)
    x, a = rng.normal(size=(10, 3)), rng.normal(size=(3, 3))
    f, d = a @ a.T, rng.uniform(0.001, 0.1, 10)
    factor = RiskMatrix(exposures=x, factor_covariance=f, specific_variance=d)
    dense = RiskMatrix(covariance=x @ f @ x.T + np.diag(d))
    w = rng.uniform(0, 1, 10)
    assert factor.variance(w) == pytest.approx(dense.variance(w), rel=1e-12)
    assert factor.diagonal() == pytest.approx(dense.diagonal())
    eps = 1e-5
    gradient = np.array(
        [
            (factor.variance(w + np.eye(10)[i] * eps) - factor.variance(w - np.eye(10)[i] * eps))
            / (2 * eps)
            for i in range(10)
        ]
    )
    assert gradient == pytest.approx(2 * dense.covariance @ w, rel=1e-8)


@pytest.mark.parametrize("return_model", ["score", "ic_vol"])
def test_legacy_benchmark_does_not_silently_change_absolute_risk_objective(return_model):
    assets = list("AB")
    common = {
        "scores": pd.Series([0.001, 0.002], index=assets),
        "trailing_returns": None,
        "current_weights": pd.Series([0.5, 0.5], index=assets),
        "linear_costs": None,
        "invested_limit": 1.0,
        "max_weight": 1.0,
        "decision_at": "2026-01-02T07:00Z",
        "config": {"mode": "cost_aware", "expected_return_model": return_model},
    }
    without = allocate_research(**common, risk_snapshot=risk_snapshot(assets))
    with_benchmark = allocate_research(
        **common, risk_snapshot=risk_snapshot(assets, benchmark=pd.Series([0.9, 0.1], index=assets))
    )
    assert with_benchmark.diagnostics["risk_objective"] == "absolute"
    assert with_benchmark.diagnostics["compatibility_mode"] is True
    assert with_benchmark.weights.to_numpy() == pytest.approx(without.weights.to_numpy(), abs=1e-7)


@pytest.mark.parametrize("budget", [1.0, 0.8])
def test_partial_cash_and_pool_external_benchmark_survive(budget):
    assets = list("ABCD")
    benchmark = pd.Series([0.25, 0.25, 0.25, 0.25], index=assets)
    snapshot = risk_snapshot(assets, benchmark=benchmark)
    result = allocate_research(
        pd.Series({"A": 1.0, "B": 2.0, "C": 3.0}),
        None,
        pd.Series({"D": 0.1}),
        None,
        invested_limit=budget,
        max_weight=0.7,
        risk_snapshot=snapshot,
        decision_at="2026-01-02T07:00Z",
        expected_returns=pd.Series({"A": 0.001, "B": 0.002, "C": 0.003}),
        config={
            "protocol_version": 2,
            "mode": "cost_aware",
            "risk_objective": "active",
            "forecast_horizon_sessions": 1,
            "expected_return_model": "provided",
            "max_turnover": 2.0,
            "max_tracking_error": 0.09,
        },
    )
    w = result.weights
    assert list(w.index) == assets
    assert w.D == pytest.approx(0, abs=1e-8)
    assert w.sum() == pytest.approx(budget)
    risk = snapshot.matrix(assets, "2026-01-02T07:00Z")
    assert result.diagnostics["annualized_tracking_error"] == pytest.approx(
        np.sqrt(risk.variance(w - benchmark))
    )
    assert result.diagnostics["annualized_tracking_error"] <= 0.09 + 1e-7
    assert result.diagnostics["risk_snapshot"]["model_version"] == "test-v1"


def test_future_risk_rejected_and_no_positive_sharpe_explicit():
    snapshot = risk_snapshot(list("AB"), available="2026-01-03T07:00Z")
    with pytest.raises(ValueError, match="unavailable"):
        snapshot.matrix(list("AB"), "2026-01-02T07:00Z")
    with pytest.raises(OptimizationError) as error:
        solve_portfolio(
            [-1.0, -2.0],
            RiskMatrix(covariance=np.eye(2)),
            [0.5, 0.5],
            OptimizationConstraints(max_weight=1),
            kind="max_sharpe",
        )
    assert error.value.status == "no_positive_excess_return"


def test_mature_ic_ignores_future_labels_and_revisions():
    rows = []
    dates = pd.date_range("2026-01-01", periods=8, tz="UTC")
    for i in range(6):
        for j in range(3):
            rows.append(
                {
                    "signal_at": dates[i],
                    "label_end": dates[i + 2],
                    "available_at": dates[i + 2],
                    "asset": str(j),
                    "score": j,
                    "return_": j * 0.01,
                    "horizon_sessions": 2,
                }
            )
    labels = pd.DataFrame(rows).rename(columns={"return_": "return"})
    before = mature_ic(labels, decision_at=dates[6], horizon_sessions=2, lookback=10, min_periods=3)
    labels.loc[labels.available_at > dates[6], "return"] *= -100
    revisions = labels.loc[labels.signal_at == dates[0]].copy()
    revisions["available_at"], revisions["return"] = dates[7], -999
    after = mature_ic(
        pd.concat([labels, revisions]),
        decision_at=dates[6],
        horizon_sessions=2,
        lookback=10,
        min_periods=3,
    )
    assert before == after
    assert before[0] == pytest.approx(1)
    assert before[1]["ic_periods"] == 3


def test_integer_repair_accounts_for_floor_fee_and_rechecks_risk():
    def quote(asset, delta):
        fee = Decimal(5)
        return -Decimal(delta) * 10 - fee, fee

    def residuals(w):
        return {"A_cap": w.A - 0.51, "B_floor": 0.45 - w.B}

    got = repair_integer_target(
        pd.Series({"A": 0.5, "B": 0.5}),
        current={},
        prices={"A": Decimal(10), "B": Decimal(10)},
        lots={"A": 100, "B": 100},
        cash=Decimal(10000),
        nav=Decimal(10000),
        quote_trade=quote,
        constraint_residuals=residuals,
        max_invested=1.0,
    )
    assert got.cash >= 0
    assert got.quantities == {"A": 400, "B": 500}
    assert got.costs == Decimal(10)
    assert got.cash == Decimal(990)
    assert max(residuals(got.weights).values()) <= 1e-7
    assert got.diagnostics["integer_global_optimum"] is False


def test_unsuccessful_integer_search_does_not_claim_infeasibility():
    with pytest.raises(OptimizationError) as exc:
        repair_integer_target(
            pd.Series({"A": 1.0}),
            current={},
            prices={"A": Decimal(10)},
            lots={"A": 100},
            cash=Decimal(100),
            nav=Decimal(100),
            quote_trade=lambda a, q: (-Decimal(q) * 10, Decimal(0)),
            constraint_residuals=lambda w: {"lower": 0.9 - w.A},
            max_invested=1.0,
        )
    assert exc.value.status == "repair_failed"


def test_unpaid_dividend_counts_in_nav_but_cannot_fund_purchases():
    kwargs = {
        "current": {},
        "prices": {"A": Decimal(10)},
        "lots": {"A": 100},
        "cash": Decimal(900),
        "nav": Decimal(1100),
        "non_cash_assets": Decimal(200),
        "quote_trade": lambda a, q: (-Decimal(q) * 10, Decimal(0)),
        "constraint_residuals": lambda w: {},
        "max_invested": 1.0,
    }
    result = repair_integer_target(pd.Series({"A": 1.0}), **kwargs)
    assert result.quantities == {}
    assert result.cash == Decimal(900)
    assert result.diagnostics["non_cash_assets"] == "200"
    kwargs["non_cash_assets"] = Decimal(199)
    with pytest.raises(ValueError, match="do not reconcile"):
        repair_integer_target(pd.Series({"A": 1.0}), **kwargs)


def test_fixed_exit_target_cannot_be_undone_by_integer_distance_repair():
    result = repair_integer_target(
        pd.Series({"A": 0.0, "B": 0.5}),
        current={"A": 100},
        prices={"A": Decimal(10), "B": Decimal(100)},
        lots={"A": 100, "B": 100},
        cash=Decimal(1000),
        nav=Decimal(2000),
        quote_trade=lambda a, q: (-Decimal(q) * (10 if a == "A" else 100), Decimal(0)),
        constraint_residuals=lambda w: {},
        max_invested=0.5,
        fixed_quantities={"A": 0},
    )
    assert result.quantities == {}
    assert result.cash == Decimal(2000)
