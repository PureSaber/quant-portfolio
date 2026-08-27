"""Sanitized quantitative research and portfolio engineering demos."""

from quant_portfolio.allocator import PortfolioSnapshot, allocate
from quant_portfolio.synthetic_spread import (
    BacktestMetrics,
    SyntheticSpreadConfig,
    backtest_synthetic_spread,
    generate_synthetic_pair,
)

from quant_portfolio.optimization import (
    OptimizationConstraints,
    OptimizationResult,
    estimate_capacity,
    estimate_covariance,
    optimize_mean_variance,
    square_root_impact_cost,
)

__all__ = [
    "BacktestMetrics",
    "OptimizationConstraints",
    "OptimizationResult",
    "PortfolioSnapshot",
    "SyntheticSpreadConfig",
    "allocate",
    "backtest_synthetic_spread",
    "estimate_capacity",
    "estimate_covariance",
    "generate_synthetic_pair",
    "optimize_mean_variance",
    "square_root_impact_cost",
]
