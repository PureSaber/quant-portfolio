from decimal import Decimal

import pandas as pd

from quant_portfolio.lookthrough import fund_exposures


def test_disclosure_overlap_is_aggregated_and_missing_weight_retained():
    frame = pd.DataFrame(
        [
            {
                "disclosure_id": fund,
                "fund_id": fund,
                "holding_date": "2024-01-01",
                "available_at": "2024-01-02T00:00:00Z",
                "instrument_id": "SAME-STOCK",
                "asset_type": "security",
                "currency": "CNY",
                "weight": ".8",
                "source": "synthetic",
                "evidence_id": fund,
            }
            for fund in ("F1", "F2")
        ]
    )
    result = fund_exposures({"F1": ".5", "F2": ".5"}, frame, "2024-02-01T00:00:00Z")
    assert result["exposures"][0]["weight"] == Decimal(".8")
    assert result["unknown_weight"] == Decimal(".2")
