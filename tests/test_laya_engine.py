"""
Unit tests for src/model/laya_engine.py.
Verifies Stage-2 options evaluation, conviction threshold gating, and signal triggers.
"""

from src.quant.greeks import calculate_greeks
from src.quant.fee_simulator import evaluate_option_candidate
from src.quant.volatility_surface import analyze_volatility_regime
from src.model.state_serializer import serialize_options_state
from src.model.laya_engine import LayaOptionsEngine, OptionDecisionResult


def test_laya_options_engine_high_conviction_trigger():
    engine = LayaOptionsEngine()

    spot = 65000.0
    strike = 67500.0
    dte = 3.5
    T = dte / 365.0
    r = 0.04
    sigma = 0.45

    greeks = calculate_greeks(spot, strike, T, r, sigma)
    payoff = evaluate_option_candidate(
        contract_symbol="BTC-24OCT26-67500-C",
        spot_price=spot,
        bid_price=715.0,
        ask_price=735.0,
        delta=greeks.delta,
        theta_daily=greeks.theta_daily
    )
    vol_metrics = analyze_volatility_regime(
        current_iv=0.40,
        realized_vol_30d=0.39,
        historical_iv_series=[0.35 + i * 0.005 for i in range(100)]
    )

    state = serialize_options_state(
        contract_symbol="BTC-24OCT26-67500-C",
        spot_price=spot,
        strike_price=strike,
        dte_days=dte,
        greeks=greeks,
        payoff=payoff,
        vol_metrics=vol_metrics,
        news_regime="bullish_vol_expansion",
        news_confidence=0.85
    )

    result = engine.evaluate_state(state)
    assert isinstance(result, OptionDecisionResult)
    assert result.should_buy_call is True
    assert result.confidence >= 0.75
    assert result.is_signal_triggered is True
    assert result.rejection_reason is None


def test_laya_options_engine_bearish_news_rejection():
    engine = LayaOptionsEngine()

    spot = 65000.0
    strike = 68000.0
    dte = 3.0
    T = dte / 365.0
    r = 0.04
    sigma = 0.50

    greeks = calculate_greeks(spot, strike, T, r, sigma)
    payoff = evaluate_option_candidate(
        contract_symbol="BTC-24OCT26-68000-C",
        spot_price=spot,
        bid_price=715.0,
        ask_price=735.0,
        delta=greeks.delta,
        theta_daily=greeks.theta_daily
    )

    # State injected with bearish news dump
    state = serialize_options_state(
        contract_symbol="BTC-24OCT26-68000-C",
        spot_price=spot,
        strike_price=strike,
        dte_days=dte,
        greeks=greeks,
        payoff=payoff,
        news_regime="bearish_dump",
        news_confidence=0.85
    )

    result = engine.evaluate_state(state)
    assert result.is_signal_triggered is False
    assert result.should_buy_call is False
    assert "Bearish macro news" in result.rejection_reason
