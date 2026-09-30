"""Shared input boundary for opaque security identifiers."""

from pathlib import Path

import pandas as pd


def read_symbol_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        # Both leading zeros and symbols such as NA/NULL are meaningful identifiers.
        frame = pd.read_csv(path, dtype={"symbol": "string"}, keep_default_na=False)
    if "symbol" not in frame:
        raise ValueError(f"symbol column is required: {path}")
    if (
        not frame["symbol"]
        .map(
            lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()
        )
        .all()
    ):
        raise ValueError(
            f"symbol must contain non-empty strings without surrounding whitespace: {path}"
        )
    return frame
