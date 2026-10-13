"""
Unit tests for src/model/state_serializer.py.
Verifies multi-modal context fusion, token formatting, and question schema integrity.
"""

from datetime import datetime, timezone
from src.quant.greeks import calculate_greeks
from src.quant.fee_simulator import evaluate_option_candidate
from src.quant.volatility_surface import analyze_volatility_regime
from src.quant.merton_jump import evaluate_merton_regime
from src.model.state_serializer import serialize_options_state, OPTIONS_DECISION_QUESTIONS


def test_serialize_options_state():
    spot = 65000.0
    strike = 68000.0
    dte = 3.5
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
    vol_metrics = analyze_volatility_regime(
        current_iv=0.45,
        realized_vol_30d=0.42,
        historical_iv_series=[0.35 + i * 0.005 for i in range(100)]
    )
    merton_res = evaluate_merton_regime(spot, strike, T, r, sigma)

    serialized = serialize_options_state(
        contract_symbol="BTC-24OCT26-68000-C",
        spot_price=spot,
        strike_price=strike,
        dte_days=dte,
        greeks=greeks,
        payoff=payoff,
        vol_metrics=vol_metrics,
        merton_res=merton_res,
        news_regime="bullish_vol_expansion",
        news_confidence=0.88,
        funding_rate_8h=0.0003
    )

    assert serialized.contract_symbol == "BTC-24OCT26-68000-C"
    assert "UNDERLYING_METRICS:" in serialized.state_text
    assert "STAGE1_NEWS_SENTIMENT:" in serialized.state_text
    assert "BULLISH_VOL_EXPANSION" in serialized.state_text
    assert "INDIA_FRICTION_AND_PAYOFF_SIMULATION:" in serialized.state_text
    assert "Delta:" in serialized.state_text
    assert "Hurdle >= 3%: PASSED" in serialized.state_text

    # Verify System 1 question schema
    assert "should_buy_call" in serialized.questions
    assert serialized.questions["should_buy_call"]["type"] == "noul"
    assert "expected_net_return" in serialized.questions
    assert serialized.questions["expected_net_return"]["type"] == "choice"


def test_serialize_options_state_with_technical_indicators():
    from src.quant.technical_indicators import TechnicalIndicatorSummary

    tech = TechnicalIndicatorSummary(
        symbol="BTCUSDT",
        interval="1h",
        current_price=65000.0,
        trend_regime="BULLISH_UPTREND",
        ema_20=64500.0,
        ema_50=64100.0,
        rsi_14=63.5,
        rsi_regime="BULLISH_MOMENTUM",
        macd_line=120.0,
        macd_signal=95.0,
        macd_hist=25.0,
        macd_bias="BULLISH_EXPANSION",
        bb_upper=66200.0,
        bb_lower=63800.0,
        bb_width_pct=3.7,
        is_bb_squeeze=False,
        support_level=63800.0,
        resistance_level=66200.0,
        vwap=64700.0
    )

    spot = 65000.0
    strike = 68000.0
    dte = 3.5
    greeks = calculate_greeks(spot, strike, dte / 365.0, 0.04, 0.50)
    payoff = evaluate_option_candidate(
        contract_symbol="BTC-24OCT26-68000-C",
        spot_price=spot,
        bid_price=715.0,
        ask_price=735.0,
        delta=greeks.delta,
        theta_daily=greeks.theta_daily
    )

    serialized = serialize_options_state(
        contract_symbol="BTC-24OCT26-68000-C",
        spot_price=spot,
        strike_price=strike,
        dte_days=dte,
        greeks=greeks,
        payoff=payoff,
        technical_summary=tech
    )

    assert "CHART_TECHNICAL_INDICATORS (1H Candlesticks):" in serialized.state_text
    assert "- Trend: BULLISH_UPTREND" in serialized.state_text
    assert "RSI(14) = 63.5" in serialized.state_text
    assert "VWAP: $64,700.0" in serialized.state_text

