from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from random import Random

import pandas as pd


@dataclass(frozen=True)
class SyntheticSpreadConfig:
    """Configuration for a deterministic synthetic mean-reverting spread."""

    periods: int = 756
    seed: int = 20260819
    start_price: float = 100.0
    spread_mean: float = 0.0
    mean_reversion: float = 0.08
    spread_volatility: float = 0.65
    market_volatility: float = 0.45
    lookback: int = 40
    entry_z: float = 1.8
    exit_z: float = 0.45
    cost_bps: float = 2.0

    def validate(self) -> None:
        if self.periods <= self.lookback + 2:
            raise ValueError("periods must exceed lookback + 2")
        if not 0 < self.mean_reversion < 1:
            raise ValueError("mean_reversion must be in (0, 1)")
        if self.spread_volatility <= 0 or self.market_volatility <= 0:
            raise ValueError("volatility inputs must be positive")
        if self.lookback < 2:
            raise ValueError("lookback must be at least 2")
        if self.entry_z <= self.exit_z or self.exit_z < 0:
            raise ValueError("entry_z must exceed a non-negative exit_z")
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    turnover: float
    trades: int
    observations: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def generate_synthetic_pair(config: SyntheticSpreadConfig) -> pd.DataFrame:
    """Generate two synthetic price legs with a stationary latent spread.

    The series is for demonstration only. It does not model any real asset,
    exchange, employer dataset, or investable opportunity.
    """

    config.validate()
    rng = Random(config.seed)
    market = 0.0
    spread = config.spread_mean
    rows: list[dict[str, float | int]] = []

    for step in range(config.periods):
        market += rng.gauss(0.0, config.market_volatility)
        innovation = rng.gauss(0.0, config.spread_volatility)
        spread += config.mean_reversion * (config.spread_mean - spread) + innovation
        leg_a = config.start_price + market + spread / 2.0
        leg_b = config.start_price + market - spread / 2.0
        rows.append(
            {
                "step": step,
                "leg_a": leg_a,
                "leg_b": leg_b,
                "spread": leg_a - leg_b,
            }
        )

    return pd.DataFrame(rows)


def add_causal_zscore(frame: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Add a one-period-lagged rolling z-score to avoid look-ahead bias."""

    if "spread" not in frame.columns:
        raise ValueError("frame must contain a spread column")
    if lookback < 2:
        raise ValueError("lookback must be at least 2")

    out = frame.copy()
    prior = out["spread"].shift(1)
    rolling = prior.rolling(lookback, min_periods=lookback)
    mean = rolling.mean()
    std = rolling.std(ddof=0).replace(0.0, pd.NA)
    out["zscore"] = ((out["spread"] - mean) / std).astype("Float64")
    return out


def _positions_from_zscore(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    positions: list[float] = []
    current = 0.0
    for raw_value in zscore:
        if pd.isna(raw_value):
            current = 0.0
        else:
            value = float(raw_value)
            if current == 0.0:
                if value >= entry_z:
                    current = -1.0
                elif value <= -entry_z:
                    current = 1.0
            elif abs(value) <= exit_z:
                current = 0.0
        positions.append(current)
    return pd.Series(positions, index=zscore.index, dtype=float)


def backtest_synthetic_spread(
    frame: pd.DataFrame,
    *,
    lookback: int,
    entry_z: float,
    exit_z: float,
    cost_bps: float,
) -> tuple[pd.DataFrame, BacktestMetrics]:
    """Run a transparent, cost-aware mean-reversion demonstration."""

    if entry_z <= exit_z or exit_z < 0:
        raise ValueError("entry_z must exceed a non-negative exit_z")
    if cost_bps < 0:
        raise ValueError("cost_bps cannot be negative")

    out = add_causal_zscore(frame, lookback)
    signal_position = _positions_from_zscore(out["zscore"], entry_z, exit_z)
    out["position"] = signal_position.shift(1).fillna(0.0)
    out["spread_change"] = out["spread"].diff().fillna(0.0)

    scale = max(float(out["spread"].std(ddof=0)), 1e-12)
    gross_return = out["position"] * out["spread_change"] / scale / 100.0
    position_change = out["position"].diff().abs().fillna(out["position"].abs())
    cost = position_change * cost_bps / 10_000.0
    out["gross_return"] = gross_return
    out["transaction_cost"] = cost
    out["net_return"] = gross_return - cost
    out["equity"] = (1.0 + out["net_return"]).cumprod()
    out["drawdown"] = out["equity"] / out["equity"].cummax() - 1.0

    net = out["net_return"]
    annualized_volatility = float(net.std(ddof=0) * sqrt(252))
    sharpe = float(net.mean() / net.std(ddof=0) * sqrt(252)) if net.std(ddof=0) else 0.0
    metrics = BacktestMetrics(
        total_return=float(out["equity"].iloc[-1] - 1.0),
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=float(out["drawdown"].min()),
        turnover=float(position_change.sum()),
        trades=int((position_change > 0).sum()),
        observations=len(out),
    )
    return out, metrics
