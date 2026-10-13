"""
Unit tests for config/fees.py.
Verifies Deribit fee capping, spread calculations, and Indian tax/TDS friction.
"""

from config.fees import (
    calculate_deribit_fee,
    calculate_spread_cost,
    calculate_india_friction,
    calculate_total_friction,
    FrictionParameters
)


def test_deribit_fee_cap():
    spot = 65000.0
    # 0.03% of 65000 = 19.5 USD
    # If option premium is very cheap, e.g. 50 USD, 12.5% cap is 6.25 USD
    cheap_premium = 50.0
    fee_cheap = calculate_deribit_fee(spot, cheap_premium)
    assert fee_cheap == 6.25

    # If option premium is expensive, e.g. 1000 USD, 12.5% cap is 125 USD -> fee should be 19.5 USD
    expensive_premium = 1000.0
    fee_exp = calculate_deribit_fee(spot, expensive_premium)
    assert fee_exp == 19.5


def test_spread_cost():
    bid = 710.0
    ask = 730.0
    cost = calculate_spread_cost(bid, ask)
    assert cost == 20.0


def test_india_friction():
    gross_profit = 200.0
    exit_value = 1000.0

    # Advisory / Offshore mode (no domestic VDA tax on personal account)
    offshore = calculate_india_friction(gross_profit, exit_value, is_domestic_taxable=False)
    # 1% TDS on 1000 = 10 USD
    assert offshore == 10.0

    # Domestic taxable mode: 10 USD TDS + (31.2% of 200 = 62.4 USD) = 72.4 USD
    domestic = calculate_india_friction(gross_profit, exit_value, is_domestic_taxable=True)
    assert round(domestic, 2) == 72.4


def test_calculate_total_friction():
    spot = 65000.0
    entry_ask = 730.0
    exit_bid = 900.0

    res = calculate_total_friction(spot, entry_ask, exit_bid, is_domestic_taxable=False)
    assert "total_friction" in res
    assert res["entry_fee"] == 19.5
    assert res["exit_fee"] == 19.5
    assert res["india_tax_drag"] == 9.0  # 1% of 900
    assert res["total_friction"] == (19.5 + 19.5 + 9.0)
