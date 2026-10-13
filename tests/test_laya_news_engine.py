"""
Unit tests for src/model/laya_news_engine.py.
Verifies market-moving classification, impact regime detection, and the bearish dump circuit breaker.
"""

from src.model.laya_news_engine import LayaNewsEngine, NewsSentimentResult


def test_news_engine_bearish_circuit_breaker():
    engine = LayaNewsEngine()
    bearish_headline = "[BREAKING] SEC issues Wells Notice to major crypto custodian. $120M long liquidation waterfall underway."

    result = engine.evaluate_headline(bearish_headline)
    assert isinstance(result, NewsSentimentResult)
    assert result.is_market_moving is True
    assert result.impact_regime == "bearish_dump"
    assert result.confidence >= 0.70
    # Circuit breaker must trigger to protect capital!
    assert result.circuit_breaker_triggered is True


def test_news_engine_bullish_catalyst():
    engine = LayaNewsEngine()
    bullish_headline = "[BREAKING] SEC approves first in-kind spot ETF options. $90M short liquidation surge across derivatives desks."

    result = engine.evaluate_headline(bullish_headline)
    assert isinstance(result, NewsSentimentResult)
    assert result.is_market_moving is True
    assert result.impact_regime == "bullish_vol_expansion"
    assert result.confidence >= 0.75
    # Bullish catalyst should NOT trigger circuit breaker
    assert result.circuit_breaker_triggered is False


def test_news_engine_neutral_noise():
    engine = LayaNewsEngine()
    noise_headline = "Gaming guild announces migration of token utility to testnet polygon sub-chain."

    result = engine.evaluate_headline(noise_headline)
    assert isinstance(result, NewsSentimentResult)
    assert result.impact_regime == "neutral_noise"
    assert result.circuit_breaker_triggered is False
