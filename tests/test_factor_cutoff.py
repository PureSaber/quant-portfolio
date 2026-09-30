"""A historical snapshot must be invariant to later factor observations."""

import json
import subprocess
import sys

import pandas as pd
import pytest

from quant_portfolio.allocator import allocate


def setup_config(tmp_path, rows):
    nav = tmp_path / "book_nav.csv"
    pd.DataFrame({"date": ["2026-09-29"], "nav": [100000]}).to_csv(nav, index=False)
    pd.DataFrame({"symbol": ["AAA", "BBB"], "weight": [0.5, 0.5]}).to_csv(
        tmp_path / "book_holdings.csv", index=False
    )
    factors = tmp_path / "factors.csv"
    pd.DataFrame(rows).to_csv(factors, index=False)
    return {
        "strategies": [{"name": "test", "nav": str(nav)}],
        "factor_scores": {"path": str(factors), "column": "score", "weight": 0.25},
    }


def rows(date="2026-09-29", scores=(0, 1), **extra):
    return [
        {"date": date, "symbol": symbol, "score": score, **extra}
        for symbol, score in zip(("AAA", "BBB"), scores)
    ]


def test_cli_future_observations_cannot_change_past_snapshot(tmp_path):
    config = setup_config(tmp_path, rows())
    path, out = tmp_path / "config.json", tmp_path / "out.json"
    path.write_text(json.dumps(config))
    results = []
    for data in (rows(), rows() + rows("2026-09-30", (1, 0))):
        pd.DataFrame(data).to_csv(config["factor_scores"]["path"], index=False)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "quant_portfolio.cli",
                "status",
                "--config",
                str(path),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        results.append(json.loads(out.read_text()))
    assert results[0] == results[1]
    assert results[0]["combined_weights"] == {"AAA": 0.375, "BBB": 0.625}


def test_late_revision_uses_availability_cutoff(tmp_path):
    config = setup_config(
        tmp_path, rows(available_at="2026-09-29") + rows(scores=(1, 0), available_at="2026-09-30")
    )
    assert allocate(config).combined_weights == {"AAA": 0.375, "BBB": 0.625}


def test_all_future_scores_leave_holdings_unchanged(tmp_path):
    assert allocate(setup_config(tmp_path, rows("2026-09-30"))).combined_weights == {
        "AAA": 0.5,
        "BBB": 0.5,
    }


@pytest.mark.parametrize("date", ["invalid", "", "2026-02-30"])
def test_invalid_factor_dates_fail(tmp_path, date):
    with pytest.raises(ValueError, match="date|time"):
        allocate(setup_config(tmp_path, rows(date)))


def test_ambiguous_duplicate_observations_fail(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        allocate(setup_config(tmp_path, rows() + rows()))


def test_undated_scores_require_explicit_as_of(tmp_path):
    data = [{k: v for k, v in row.items() if k != "date"} for row in rows()]
    config = setup_config(tmp_path, data)
    with pytest.raises(ValueError, match="as_of"):
        allocate(config)
    config["factor_scores"]["as_of"] = "2026-09-29"
    assert allocate(config).combined_weights == {"AAA": 0.375, "BBB": 0.625}
    config["factor_scores"]["as_of"] = "2026-09-30"
    assert allocate(config).combined_weights == {"AAA": 0.5, "BBB": 0.5}


def test_nav_is_chronological_and_intraday_cutoff_uses_utc(tmp_path):
    config = setup_config(
        tmp_path, rows("2026-09-29T06:00:00Z") + rows("2026-09-29T08:00:00Z", (1, 0))
    )
    pd.DataFrame(
        {"date": ["2026-09-29T15:00:00+08:00", "2026-09-28"], "nav": [100000, 90000]}
    ).to_csv(config["strategies"][0]["nav"], index=False)
    snapshot = allocate(config)
    assert snapshot.total_nav == 100000
    assert snapshot.combined_weights == {"AAA": 0.375, "BBB": 0.625}


def test_daily_factor_is_unavailable_before_day_end(tmp_path):
    config = setup_config(tmp_path, rows())
    pd.DataFrame({"date": ["2026-09-29T15:00:00Z"], "nav": [100000]}).to_csv(
        config["strategies"][0]["nav"], index=False
    )
    assert allocate(config).combined_weights == {"AAA": 0.5, "BBB": 0.5}
