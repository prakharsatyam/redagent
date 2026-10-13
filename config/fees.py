"""
Exchange Fee Schedules and Indian Localization Friction Model.
Encapsulates Deribit options maker/taker fees, bid-ask spread crossing,
and Indian Section 115BBH (31.2% tax) + Section 194S (1% TDS) drag.
"""

from pydantic import BaseModel, Field


class FrictionParameters(BaseModel):
    """Parameters governing exchange fee calculations and tax drag."""
    deribit_maker_fee_rate: float = Field(default=0.0003, description="0.03% of underlying asset")
    deribit_taker_fee_rate: float = Field(default=0.0003, description="0.03% of underlying asset")
    deribit_fee_cap_ratio: float = Field(default=0.125, description="Max 12.5% of option value")
    india_tds_rate: float = Field(default=0.01, description="Section 194S 1% TDS on transaction value")
    india_vda_tax_rate: float = Field(default=0.312, description="Section 115BBH 30% tax + 4% cess = 31.2%")
    usdt_inr_premium_rate: float = Field(default=0.018, description="P2P banking INR/USDT premium ~1.8%")


def calculate_deribit_fee(spot_price: float, option_premium: float, is_maker: bool = False, params: FrictionParameters = FrictionParameters()) -> float:
    """
    Calculates Deribit options fee in USD:
    Fee = min(0.0003 * spot_price, 0.125 * option_premium)
    """
    fee_rate = params.deribit_maker_fee_rate if is_maker else params.deribit_taker_fee_rate
    raw_fee = fee_rate * spot_price
    fee_cap = params.deribit_fee_cap_ratio * option_premium
    return min(raw_fee, fee_cap)


def calculate_spread_cost(bid_price: float, ask_price: float) -> float:
    """
    Crossing the bid-ask spread costs half the spread on entry and half on exit.
    Round-trip crossing cost = ask_price - bid_price.
    """
    return max(0.0, ask_price - bid_price)


def calculate_india_friction(gross_profit: float, exit_value: float, is_domestic_taxable: bool = False, params: FrictionParameters = FrictionParameters()) -> float:
    """
    Calculates Indian tax & TDS friction:
    - 1% TDS on total exit value (Section 194S)
    - If is_domestic_taxable=True, applies 31.2% tax on gross profit with zero loss offset.
    """
    tds_drag = params.india_tds_rate * exit_value
    tax_drag = (params.india_vda_tax_rate * gross_profit) if (is_domestic_taxable and gross_profit > 0) else 0.0
    return tds_drag + tax_drag


def calculate_total_friction(
    spot_price: float,
    entry_ask: float,
    exit_bid: float,
    is_domestic_taxable: bool = False,
    is_maker_exit: bool = True,
    params: FrictionParameters = FrictionParameters()
) -> dict:
    """
    Aggregates all frictional drag components for a round-trip options trade:
    1. Entry Taker Fee
    2. Exit Fee (Maker or Taker)
    3. Bid-Ask Spread Crossing
    4. Indian TDS / Tax drag (if domestic)
    """
    entry_fee = calculate_deribit_fee(spot_price, entry_ask, is_maker=False, params=params)
    exit_fee = calculate_deribit_fee(spot_price, exit_bid, is_maker=is_maker_exit, params=params)
    spread_cost = calculate_spread_cost(exit_bid, entry_ask) if entry_ask > exit_bid else 0.0
    gross_profit = max(0.0, exit_bid - entry_ask)
    india_tax = calculate_india_friction(gross_profit, exit_bid, is_domestic_taxable=is_domestic_taxable, params=params)

    total_cost = entry_fee + exit_fee + spread_cost + india_tax

    return {
        "entry_fee": entry_fee,
        "exit_fee": exit_fee,
        "spread_cost": spread_cost,
        "india_tax_drag": india_tax,
        "total_friction": total_cost,
    }
