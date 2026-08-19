"""Sanitized quantitative research and portfolio engineering demos."""

from quant_portfolio.allocator import PortfolioSnapshot, allocate
from quant_portfolio.synthetic_spread import (
    BacktestMetrics,
    SyntheticSpreadConfig,
    backtest_synthetic_spread,
    generate_synthetic_pair,
)

__all__ = [
    "BacktestMetrics",
    "PortfolioSnapshot",
    "SyntheticSpreadConfig",
    "allocate",
    "backtest_synthetic_spread",
    "generate_synthetic_pair",
]
