"""
Multi-Modal Context Fusion Serializer and System 1 Contract Generator.
Fuses quantitative Greeks, Merton Jump pricing, Volatility Surface regimes,
Stage-1 News Sentiment, and Indian friction payoffs into structured System 1 state text.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from src.quant.greeks import GreeksResult
from src.quant.merton_jump import MertonResult
from src.quant.volatility_surface import VolatilityMetrics
from src.quant.fee_simulator import TradePayoffEvaluation
from src.quant.technical_indicators import TechnicalIndicatorSummary


# Standard System 1 Typed Decision Question Schemas for Laya Stage-2
OPTIONS_DECISION_QUESTIONS: Dict[str, Any] = {
    "should_buy_call": {
        "type": "noul",
        "instructions": (
            "Based on the market state, Greeks, and fee simulation, should a trader enter "
            "this Call option to achieve >= 3% net profit after all fees, slippage, and theta decay?"
        )
    },
    "expected_net_return": {
        "type": "choice",
        "instructions": "What is the expected net return bracket of this Call option trade after all friction?",
        "options": [
            "loss_or_negative",
            "breakeven_to_2pct",
            "target_3pct_to_10pct",
            "explosive_above_10pct"
        ]
    },
    "risk_regime": {
        "type": "choice",
        "instructions": "What is the primary risk regime governing this contract?",
        "options": [
            "high_theta_decay",
            "iv_crush_risk",
            "favorable_low_vol_expansion",
            "trending_spot_momentum"
        ]
    }
}


class SerializedMarketState(BaseModel):
    """Container holding the fused state string and the decision questions."""
    contract_symbol: str
    timestamp_iso: str
    state_text: str
    questions: Dict[str, Any] = Field(default_factory=lambda: OPTIONS_DECISION_QUESTIONS)


def serialize_options_state(
    contract_symbol: str,
    spot_price: float,
    strike_price: float,
    dte_days: float,
    greeks: GreeksResult,
    payoff: TradePayoffEvaluation,
    vol_metrics: Optional[VolatilityMetrics] = None,
    merton_res: Optional[MertonResult] = None,
    news_regime: Optional[str] = None,
    news_confidence: Optional[float] = None,
    funding_rate_8h: Optional[float] = None,
    technical_summary: Optional[TechnicalIndicatorSummary] = None,
    timestamp: Optional[datetime] = None
) -> SerializedMarketState:
    """
    Serializes multi-source parameters into a high-density structured state document.
    """
    ts = timestamp or datetime.now(timezone.utc)
    ts_str = ts.isoformat()

    moneyness = "ITM" if spot_price > strike_price else ("ATM" if abs(spot_price - strike_price) / spot_price < 0.01 else "OTM")

    # Quant block
    quant_lines = [
        f"[TIMESTAMP: {ts_str} | ASSET: {contract_symbol}]",
        "UNDERLYING_METRICS:",
        f"- Spot Price: ${spot_price:,.2f} | Strike: ${strike_price:,.2f} | DTE: {dte_days:.2f} days | Moneyness: {moneyness}",
        f"- Delta: {greeks.delta:.3f} | Gamma: {greeks.gamma:.6f} | Theta: ${greeks.theta_daily:.2f}/day | Vega: ${greeks.vega_1pct:.2f}/1%",
        f"- Implied Vol: {greeks.iv:.1%}"
    ]

    if vol_metrics:
        quant_lines.append(
            f"- 90d IV Rank: {vol_metrics.iv_rank:.1f}% | 30d RV: {vol_metrics.realized_vol_30d:.1%} | VRP: {vol_metrics.vrp:.1%} | Regime: {vol_metrics.regime}"
        )

    if merton_res:
        quant_lines.append(
            f"- Merton Jump Price: ${merton_res.merton_price:.2f} vs BSM: ${merton_res.bsm_baseline_price:.2f} (Jump Prem: ${merton_res.jump_premium_usd:.2f})"
        )

    if funding_rate_8h is not None:
        quant_lines.append(f"- 8h Funding Rate: {funding_rate_8h:+.4%}")

    # Technical Indicators block
    tech_lines = []
    if technical_summary:
        tech_lines = [technical_summary.to_serialized_text()]

    # Stage-1 News block
    if news_regime:
        conf_str = f"{news_confidence:.0%}" if news_confidence is not None else "N/A"
        news_block = [
            "STAGE1_NEWS_SENTIMENT:",
            f"- Impact Regime: {news_regime.upper()} (Conviction: {conf_str})"
        ]
    else:
        news_block = [
            "STAGE1_NEWS_SENTIMENT:",
            "- Impact Regime: NEUTRAL_NO_CATALYST (Standard quant mode)"
        ]

    # Friction & Hurdle block
    friction_block = [
        "INDIA_FRICTION_AND_PAYOFF_SIMULATION:",
        f"- Entry Ask: ${payoff.entry_ask_usd:.2f} | Spread Ratio: {payoff.spread_ratio:.2%}",
        f"- Projected Round-Trip Friction: ${payoff.total_friction_usd:.2f}",
        f"- Breakeven Spot Move: +{payoff.breakeven_spot_move_pct:.2f}%",
        f"- Projected Net Return (at target): {payoff.net_return_pct:+.2%} (Hurdle >= 3%: {'PASSED' if payoff.passes_hurdle else 'FAILED'})"
    ]

    full_state = "\n".join(quant_lines + tech_lines + news_block + friction_block)

    return SerializedMarketState(
        contract_symbol=contract_symbol,
        timestamp_iso=ts_str,
        state_text=full_state,
        questions=OPTIONS_DECISION_QUESTIONS
    )
