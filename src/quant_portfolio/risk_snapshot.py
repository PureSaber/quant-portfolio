"""Point-in-time, labeled, unit-bearing optimizer risk inputs."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quant_portfolio.solver import AllocationInputError, RiskMatrix


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("risk timestamps must be valid and timezone-aware")
    return result.tz_convert("UTC")


@dataclass(frozen=True, kw_only=True)
class RiskSnapshot:
    estimated_at: object
    available_at: object
    horizon_sessions: int
    annualization: int
    model_version: str
    input_digest: str
    covariance: pd.DataFrame | None = None
    exposures: pd.DataFrame | None = None
    factor_covariance: pd.DataFrame | None = None
    specific_variance: pd.Series | None = None
    benchmark: pd.Series | None = None
    coverage: dict = field(default_factory=dict)
    _fingerprint: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "coverage", deepcopy(self.coverage))
        # JSON-safe metadata is included in the immutable input fingerprint.
        json.dumps(self.coverage, sort_keys=True, allow_nan=False)
        for key in ("estimated_at", "available_at"):
            object.__setattr__(self, key, timestamp(getattr(self, key)))
        if self.estimated_at > self.available_at:
            raise ValueError("risk cannot be available before estimation")
        for key in ("horizon_sessions", "annualization"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"invalid risk {key}")
        if not self.model_version or not self.input_digest:
            raise ValueError("risk model version and input digest are required")
        for key in (
            "covariance",
            "exposures",
            "factor_covariance",
            "specific_variance",
            "benchmark",
        ):
            value = getattr(self, key)
            if value is not None:
                if not value.index.is_unique or (
                    isinstance(value, pd.DataFrame) and not value.columns.is_unique
                ):
                    raise ValueError("risk labels must be unique")
                object.__setattr__(self, key, value.copy(deep=True))
        self.matrix(self.assets, self.available_at)
        object.__setattr__(self, "_fingerprint", self._digest())

    @property
    def assets(self):
        source = self.covariance if self.covariance is not None else self.exposures
        if source is None:
            raise ValueError("risk representation is missing")
        return list(source.index)

    def matrix(self, assets, decision_at):
        self._check_unchanged()
        if self.available_at > timestamp(decision_at):
            raise AllocationInputError(
                "risk snapshot was unavailable at decision time", reason="risk_not_available"
            )
        missing = set(assets) - set(self.assets)
        if missing:
            raise AllocationInputError(
                "risk snapshot does not cover candidates, holdings and benchmark union",
                reason="risk_union_incomplete",
                assets=missing,
                details={a: self.coverage.get(a, {"status": "unavailable"}) for a in missing},
            )
        if self.benchmark is not None:
            b = self.benchmark
            if (
                not np.isfinite(b).all()
                or (b < 0).any()
                or b.sum() > 1 + 1e-10
                or set(b.index) - set(self.assets)
            ):
                raise ValueError("benchmark must be complete and covered by risk snapshot")
        if self.covariance is not None:
            if any(
                x is not None
                for x in (self.exposures, self.factor_covariance, self.specific_variance)
            ):
                raise ValueError("provide exactly one risk representation")
            if list(self.covariance.index) != list(self.covariance.columns):
                raise ValueError("covariance labels must match in order")
            return RiskMatrix(covariance=self.covariance.loc[assets, assets].to_numpy())
        if self.factor_covariance is None or self.specific_variance is None:
            raise ValueError(
                "factor risk requires exposures, factor covariance and specific variance"
            )
        factors = list(self.exposures.columns)
        if (
            set(self.factor_covariance.index) != set(factors)
            or set(self.factor_covariance.columns) != set(factors)
            or set(self.specific_variance.index) != set(self.assets)
        ):
            raise ValueError("factor risk labels must match exactly")
        return RiskMatrix(
            exposures=self.exposures.loc[assets].to_numpy(),
            factor_covariance=self.factor_covariance.loc[factors, factors].to_numpy(),
            specific_variance=self.specific_variance.loc[assets].to_numpy(),
        )

    def _check_unchanged(self):
        if self._fingerprint is not None and self._digest() != self._fingerprint:
            raise ValueError("risk snapshot was mutated after validation")

    def _digest(self):
        digest = hashlib.sha256()
        if self.coverage:
            digest.update(json.dumps(self.coverage, sort_keys=True, allow_nan=False).encode())
        for value in (
            self.covariance,
            self.exposures,
            self.factor_covariance,
            self.specific_variance,
            self.benchmark,
        ):
            if value is not None:
                digest.update(pd.util.hash_pandas_object(value, index=True).to_numpy().tobytes())
                if isinstance(value, pd.DataFrame):
                    digest.update(repr(tuple(value.columns)).encode())
        return digest.hexdigest()

    def metadata(self):
        self._check_unchanged()
        return {
            "estimated_at": self.estimated_at.isoformat(),
            "available_at": self.available_at.isoformat(),
            "horizon_sessions": self.horizon_sessions,
            "annualization": self.annualization,
            "units": "annualized_variance",
            "model_version": self.model_version,
            "input_digest": self.input_digest,
            "snapshot_digest": self._fingerprint,
            "coverage": deepcopy(self.coverage),
        }


def mature_ic(labels, *, decision_at, horizon_sessions, lookback, min_periods):
    """Cross-sectional IC from available, non-overlapping, completed labels only."""
    required = {
        "signal_at",
        "label_end",
        "available_at",
        "asset",
        "score",
        "return",
        "horizon_sessions",
    }
    if not isinstance(labels, pd.DataFrame) or not required <= set(labels.columns):
        raise ValueError("ic_vol requires timestamped mature label history")
    data = labels.copy()
    for key in ("signal_at", "label_end", "available_at"):
        data[key] = data[key].map(timestamp)
    cutoff = timestamp(decision_at)
    data = data.loc[
        (data.available_at <= cutoff)
        & (data.label_end <= cutoff)
        & (data.signal_at < cutoff)
        & (data.horizon_sessions == horizon_sessions)
    ]
    if ((data.label_end <= data.signal_at) | (data.available_at < data.label_end)).any():
        raise ValueError("invalid label availability")
    # Keep the last revision available at the decision, never a future revision.
    data = data.sort_values("available_at").drop_duplicates(["signal_at", "asset"], keep="last")
    periods, next_start = [], cutoff
    for start, rows in reversed(list(data.groupby("signal_at", sort=True))):
        if rows.label_end.nunique() != 1 or rows.label_end.max() > next_start:
            continue
        values = rows[["score", "return"]].to_numpy(dtype=float)
        if len(rows) < 3 or not np.isfinite(values).all() or (np.std(values, axis=0) == 0).any():
            continue
        periods.append((start, float(np.corrcoef(values.T)[0, 1])))
        next_start = start
        if len(periods) == lookback:
            break
    if len(periods) < min_periods:
        raise AllocationInputError(
            "insufficient mature non-overlapping IC periods",
            reason="insufficient_mature_ic",
            details={"observed": len(periods), "required": min_periods},
        )
    return float(np.mean([x[1] for x in periods])), {
        "ic_periods": len(periods),
        "ic_first_signal": periods[-1][0].isoformat(),
        "ic_last_signal": periods[0][0].isoformat(),
        "ic_horizon_sessions": horizon_sessions,
    }
