import pandas as pd
import pytest

from quant_portfolio.allocator import allocate


def book(tmp_path, name, date, holdings=None):
    nav = tmp_path / f"{name}_nav.csv"
    pd.DataFrame({"date": [date], "nav": [100]}).to_csv(nav, index=False)
    if holdings is not None:
        pd.DataFrame(holdings).to_csv(tmp_path / f"{name}_holdings.csv", index=False)
    return {"name": name, "nav": str(nav)}


def test_mixed_nav_dates_fail_without_explicit_staleness_policy(tmp_path):
    config = {"strategies": [book(tmp_path, "A", "2026-09-01"), book(tmp_path, "B", "2026-10-12")]}
    with pytest.raises(ValueError, match="mismatched NAV"):
        allocate(config)
    result = allocate(config | {"max_nav_age_days": 42})
    assert result.books[0]["nav_age_days"] == 41
    assert result.warnings


@pytest.mark.parametrize(
    "holdings,match",
    [
        ({"symbol": ["A"], "weight": [1], "date": ["2026-10-13"]}, "holdings date"),
        ({"symbol": ["A", "A"], "weight": [0.5, 0.5]}, "one row"),
        ({"symbol": ["A", "B"], "weight": [0.6, 0.6]}, "total weight"),
        ({"symbol": ["A"], "weight": [1], "available_at": ["2026-10-13"]}, "not available"),
    ],
)
def test_bad_holdings_are_rejected(tmp_path, holdings, match):
    with pytest.raises(ValueError, match=match):
        allocate({"strategies": [book(tmp_path, "A", "2026-10-12", holdings)]})


def test_historical_cutoff_never_uses_future_nav_or_undated_holdings(tmp_path):
    item = book(tmp_path, "A", "2026-10-12", {"symbol": ["A"], "weight": [1]})
    pd.DataFrame({"date": ["2026-10-12", "2026-10-13"], "nav": [100, 999]}).to_csv(
        item["nav"], index=False
    )
    config = {"strategies": [item], "as_of": "2026-10-12"}
    with pytest.raises(ValueError, match="historical holdings"):
        allocate(config)
    item["holdings_as_of"] = "2026-10-12"
    result = allocate(config)
    assert result.total_nav == 100 and not result.warnings
