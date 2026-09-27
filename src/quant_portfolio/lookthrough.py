"""Read-only portfolio candidate diagnostics using shared disclosure truth."""

from quant_data_kit.financial.holdings import exposure_summary, look_through


def fund_exposures(weights, disclosures, at, **coverage):
    leaves = look_through(disclosures, weights, at, **coverage)
    return {
        **exposure_summary(leaves),
        "leaves": leaves.to_dict("records"),
        "scope": "disclosed_holdings_not_realtime_positions",
    }
