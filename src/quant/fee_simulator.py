"""
Friction Simulator and Net Profit Hurdle Gatekeeper.
Enforces the mandatory >= 2-3% net profit constraint after deducting
exchange fees, bid-ask spread crossing, daily theta decay drift, and Indian TDS.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field
from config.fees import calculate_total_friction, FrictionParameters


class TradePayoffEvaluation(BaseModel):
    """Detailed payoff and friction evaluation for an options trade candidate."""
    contract_symbol: str
    entry_ask_usd: float
    exit_target_usd: float
    spot_price_usd: float
    holding_hours: float
    spread_ratio: float
    total_friction_usd: float
    net_profit_usd: float
    net_return_pct: float
    breakeven_spot_move_pct: float
    passes_hurdle: bool
    rejection_reason: Optional[str] = None


def evaluate_option_candidate(
    contract_symbol: str,
    spot_price: float,
    bid_price: float,
    ask_price: float,
    delta: float,
    theta_daily: float,
    target_spot_move_pct: float = 0.025,  # 2.5% spot move
    holding_hours: float = 24.0,
    min_net_hurdle_pct: float = 0.03,     # 3% net profit hurdle
    max_spread_ratio: float = 0.05,       # 5% max spread/mid
    is_domestic_taxable: bool = False,
    params: FrictionParameters = FrictionParameters()
) -> TradePayoffEvaluation:
    """
    Evaluates whether a candidate Call option satisfies the strict >= 2-3% net hurdle.
    
    1. Checks liquidity: spread / mid <= 5%
    2. Projects target option price based on Delta sensitivity to underlying spot move.
    3. Deducts round-trip fees (Taker in, Maker out) and spread.
    4. Deducts holding period time decay (Theta * holding_days).
    5. Computes net return on capital.
    """
    mid_price = (bid_price + ask_price) / 2.0
    spread_ratio = (ask_price - bid_price) / mid_price if mid_price > 0 else 1.0

    # Liquidity check
    if spread_ratio > max_spread_ratio:
        return TradePayoffEvaluation(
            contract_symbol=contract_symbol,
            entry_ask_usd=ask_price,
            exit_target_usd=0.0,
            spot_price_usd=spot_price,
            holding_hours=holding_hours,
            spread_ratio=spread_ratio,
            total_friction_usd=0.0,
            net_profit_usd=0.0,
            net_return_pct=0.0,
            breakeven_spot_move_pct=0.0,
            passes_hurdle=False,
            rejection_reason=f"Illiquid: Bid-Ask spread ratio ({spread_ratio:.1%}) exceeds maximum ({max_spread_ratio:.1%})"
        )

    # 1. Delta-based price expansion at target spot move
    spot_move_usd = spot_price * target_spot_move_pct
    gross_option_gain_usd = delta * spot_move_usd
    target_exit_bid = ask_price + gross_option_gain_usd

    # 2. Holding period theta decay
    holding_days = holding_hours / 24.0
    theta_drag_usd = abs(theta_daily) * holding_days

    # 3. Exchange friction & spread
    friction_breakdown = calculate_total_friction(
        spot_price=spot_price,
        entry_ask=ask_price,
        exit_bid=target_exit_bid,
        is_domestic_taxable=is_domestic_taxable,
        is_maker_exit=True,
        params=params
    )
    exchange_and_tax_friction = friction_breakdown["total_friction"]
    total_friction_usd = exchange_and_tax_friction + theta_drag_usd

    # 4. Net profit and return %
    gross_profit = target_exit_bid - ask_price
    net_profit_usd = gross_profit - total_friction_usd
    net_return_pct = (net_profit_usd / ask_price) if ask_price > 0 else 0.0

    # 5. Breakeven spot move % needed to cover all friction
    # spot_move * delta = total_friction -> spot_move = total_friction / delta
    breakeven_spot_move_usd = (total_friction_usd / delta) if delta > 0 else spot_price
    breakeven_spot_move_pct = (breakeven_spot_move_usd / spot_price) * 100.0

    # Hurdle check: Net Return >= 3.0%
    passes_hurdle = net_return_pct >= min_net_hurdle_pct

    rejection_reason = None
    if not passes_hurdle:
        rejection_reason = (
            f"Insufficient Net Profit: Projected net return ({net_return_pct:.2%}) "
            f"below minimum hurdle ({min_net_hurdle_pct:.2%}) after {total_friction_usd:.2f} USD friction"
        )

    return TradePayoffEvaluation(
        contract_symbol=contract_symbol,
        entry_ask_usd=ask_price,
        exit_target_usd=target_exit_bid,
        spot_price_usd=spot_price,
        holding_hours=holding_hours,
        spread_ratio=spread_ratio,
        total_friction_usd=total_friction_usd,
        net_profit_usd=net_profit_usd,
        net_return_pct=net_return_pct,
        breakeven_spot_move_pct=breakeven_spot_move_pct,
        passes_hurdle=passes_hurdle,
        rejection_reason=rejection_reason
    )
