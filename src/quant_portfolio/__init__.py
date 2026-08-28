"""Sanitized quantitative research and portfolio engineering demos."""

from quant_portfolio.allocator import PortfolioSnapshot, allocate
from quant_portfolio.cross_asset import (
    CrossAssetConstraints,
    CrossAssetInput,
    CrossAssetOptimizationResult,
    OptimizationFailure,
    PITFixedPoint,
    PITMarketSnapshot,
    TargetPortfolio,
    optimize_cross_asset,
    target_to_order_intents,
)
from quant_portfolio.optimization import (
    OptimizationConstraints,
    OptimizationResult,
    estimate_capacity,
    estimate_covariance,
    optimize_mean_variance,
    square_root_impact_cost,
)
from quant_portfolio.synthetic_spread import (
    BacktestMetrics,
    SyntheticSpreadConfig,
    backtest_synthetic_spread,
    generate_synthetic_pair,
)

__all__ = [
    "BacktestMetrics",
    "CrossAssetConstraints",
    "CrossAssetInput",
    "CrossAssetOptimizationResult",
    "OptimizationConstraints",
    "OptimizationFailure",
    "OptimizationResult",
    "PITFixedPoint",
    "PITMarketSnapshot",
    "PortfolioSnapshot",
    "SyntheticSpreadConfig",
    "TargetPortfolio",
    "allocate",
    "backtest_synthetic_spread",
    "estimate_capacity",
    "estimate_covariance",
    "generate_synthetic_pair",
    "optimize_cross_asset",
    "optimize_mean_variance",
    "square_root_impact_cost",
    "target_to_order_intents",
]
