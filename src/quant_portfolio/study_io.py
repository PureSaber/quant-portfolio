"""Small strict input/output boundary for offline portfolio research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def number(value, name, *, minimum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def timestamp(value):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError("timestamps must be explicit and timezone-aware")
    return result.tz_convert("UTC")


def text(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError("identifiers must be nonblank strings without surrounding whitespace")
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def publish(input_path, output, evaluator):
    """Evaluate completely before exclusive creation; never overwrite a prior report."""
    source = Path(input_path)
    spec = json.loads(source.read_text(encoding="utf-8-sig"))
    result = evaluator(spec)
    result["input_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return result
