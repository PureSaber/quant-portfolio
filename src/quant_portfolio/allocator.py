from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass
class StrategyBook:
    name: str
    nav_path: Path
    weight: float = 1.0
    position_scale: float = 1.0


@dataclass
class PortfolioSnapshot:
    as_of: str
    total_nav: float
    books: list[dict] = field(default_factory=list)
    combined_weights: dict[str, float] = field(default_factory=dict)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _read_nav(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty nav file: {path}")
    date_col = "date" if "date" in df.columns else df.columns[0]
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"no nav column in {path}")
    out = df[[date_col, value_cols[0]]].rename(columns={date_col: "date", value_cols[0]: "nav"})
    out["date"] = out["date"].astype(str)
    return out


def _read_holdings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "symbol" not in df.columns or "weight" not in df.columns:
        raise ValueError(f"holdings need symbol,weight columns: {path}")
    return df[["symbol", "weight"]].copy()


def _read_factor_scores(path: Path, column: str) -> pd.Series:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"factor column {column} missing in {path}")
    if "symbol" not in df.columns:
        raise ValueError(f"factor scores need symbol column: {path}")
    latest = (
        df.sort_values("date").groupby("symbol", as_index=False).tail(1)
        if "date" in df.columns
        else df
    )
    scores = latest.set_index("symbol")[column].astype(float)
    return scores


def _blend_factor_scores(
    combined: dict[str, float],
    factor_scores: pd.Series,
    weight: float,
) -> dict[str, float]:
    if factor_scores.empty:
        return combined
    aligned = {sym: combined.get(sym, 0.0) for sym in factor_scores.index}
    if not aligned:
        return combined
    score = factor_scores.reindex(list(aligned.keys())).fillna(0.0)
    score_norm = (score - score.mean()) / (score.std(ddof=0) or 1.0)
    blended: dict[str, float] = {}
    for sym, base_w in aligned.items():
        tilt = 1.0 + weight * float(score_norm.get(sym, 0.0))
        blended[sym] = max(base_w * tilt, 0.0)
    total = sum(blended.values()) or 1.0
    return {k: round(v / total, 6) for k, v in blended.items()}


def allocate(config: dict) -> PortfolioSnapshot:
    strategies_cfg = config.get("strategies") or []
    books: list[StrategyBook] = []
    for entry in strategies_cfg:
        books.append(
            StrategyBook(
                name=str(entry["name"]),
                nav_path=Path(entry["nav"]),
                weight=float(entry.get("weight", 1.0)),
                position_scale=float(entry.get("position_scale", 1.0)),
            )
        )

    if not books:
        raise ValueError("no strategies configured")

    total_weight = sum(b.weight for b in books)
    nav_rows: list[dict] = []
    combined: dict[str, float] = {}

    for book in books:
        nav_df = _read_nav(book.nav_path)
        latest = nav_df.iloc[-1]
        scaled_weight = book.weight / total_weight * book.position_scale
        nav_rows.append(
            {
                "name": book.name,
                "as_of": str(latest["date"]),
                "nav": float(latest["nav"]),
                "budget_weight": round(scaled_weight, 6),
            }
        )
        holdings_path = Path(str(book.nav_path).replace("nav.csv", "holdings.csv"))
        if holdings_path.is_file():
            h = _read_holdings(holdings_path)
            for _, row in h.iterrows():
                sym = str(row["symbol"])
                combined[sym] = combined.get(sym, 0.0) + float(row["weight"]) * scaled_weight

    total_nav = sum(r["nav"] * r["budget_weight"] for r in nav_rows)
    as_of = max(r["as_of"] for r in nav_rows)

    factor_cfg = config.get("factor_scores") or {}
    if factor_cfg.get("path"):
        scores = _read_factor_scores(
            Path(factor_cfg["path"]),
            str(factor_cfg.get("column", "momentum_20d")),
        )
        combined = _blend_factor_scores(
            combined,
            scores,
            float(factor_cfg.get("weight", 0.25)),
        )

    return PortfolioSnapshot(
        as_of=as_of,
        total_nav=round(total_nav, 4),
        books=nav_rows,
        combined_weights={k: round(v, 6) for k, v in sorted(combined.items())},
    )
