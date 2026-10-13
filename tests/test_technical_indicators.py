"""
Unit tests for src/quant/technical_indicators.py and Binance klines ingestion.
"""

from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.quant.technical_indicators import (
    compute_technical_indicators,
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_vwap,
    TechnicalIndicatorSummary
)
from src.data.binance_client import BinanceClient


def _make_dummy_ohlcv(n: int = 60, trend: str = "up") -> pd.DataFrame:
    np.random.seed(42)
    base = 65000.0
    drift = 100.0 if trend == "up" else (-100.0 if trend == "down" else 0.0)
    prices = [base + i * drift + np.random.normal(0, 30.0) for i in range(n)]

    rows = []
    for i, p in enumerate(prices):
        rows.append({
            "timestamp": pd.Timestamp("2026-10-10 12:00:00", tz="UTC") + pd.Timedelta(hours=i),
            "open": p - 10.0,
            "high": p + 40.0,
            "low": p - 40.0,
            "close": p,
            "volume": 120.0 + np.random.uniform(10, 50)
        })
    return pd.DataFrame(rows)


def test_calculate_rsi_bounds():
    df = _make_dummy_ohlcv(n=50, trend="up")
    rsi = calculate_rsi(df["close"], period=14)
    assert len(rsi) == len(df)
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()
    # In an uptrend, latest RSI should be elevated
    assert rsi.iloc[-1] > 55.0


def test_calculate_ema_and_macd():
    df = _make_dummy_ohlcv(n=60, trend="up")
    ema20 = calculate_ema(df["close"], 20)
    assert len(ema20) == len(df)

    macd_dict = calculate_macd(df["close"])
    assert "macd" in macd_dict
    assert "signal" in macd_dict
    assert "hist" in macd_dict
    assert len(macd_dict["hist"]) == len(df)


def test_bollinger_bands_and_squeeze():
    # Constant low-volatility price series should trigger squeeze (< 3.5% width)
    df = pd.DataFrame({
        "close": [65000.0 + np.random.normal(0, 5.0) for _ in range(30)],
        "high": [65010.0] * 30,
        "low": [64990.0] * 30,
        "volume": [10.0] * 30
    })
    bb = calculate_bollinger_bands(df["close"], period=20)
    assert bb["bandwidth"].iloc[-1] < 3.5

    summary = compute_technical_indicators(df, symbol="BTCUSDT", interval="1h")
    assert summary.is_bb_squeeze is True
    assert summary.trend_regime in ["BULLISH_UPTREND", "BEARISH_DOWNTREND", "SIDEWAYS_CONSOLIDATION"]


def test_compute_technical_indicators_uptrend_summary():
    df = _make_dummy_ohlcv(n=60, trend="up")
    summary = compute_technical_indicators(df, symbol="BTCUSDT", interval="1h")

    assert isinstance(summary, TechnicalIndicatorSummary)
    assert summary.trend_regime == "BULLISH_UPTREND"
    assert summary.rsi_regime in ["BULLISH_MOMENTUM", "OVERBOUGHT"]
    assert summary.support_level < summary.resistance_level
    assert summary.vwap is not None

    text = summary.to_serialized_text()
    assert "CHART_TECHNICAL_INDICATORS (1H Candlesticks):" in text
    assert "- Trend: BULLISH_UPTREND" in text
    assert "RSI(14)" in text
    assert "BB Width:" in text
    assert "Key Structure:" in text


def test_compute_technical_indicators_empty_fallback():
    empty_df = pd.DataFrame()
    summary = compute_technical_indicators(empty_df, symbol="ETHUSDT", interval="4h")
    assert summary.trend_regime == "SIDEWAYS_CONSOLIDATION"
    assert summary.rsi_14 == 50.0
    text = summary.to_serialized_text()
    assert "CHART_TECHNICAL_INDICATORS (4H Candlesticks):" in text


@patch.object(BinanceClient, "get_klines")
def test_binance_client_get_klines_mock(mock_klines):
    mock_df = pd.DataFrame([
        {
            "timestamp": pd.Timestamp("2026-10-10 12:00:00", tz="UTC"),
            "open": 65000.0,
            "high": 65500.0,
            "low": 64800.0,
            "close": 65200.0,
            "volume": 250.0
        }
    ])
    mock_klines.return_value = mock_df

    client = BinanceClient()
    df = client.get_klines(symbol="BTCUSDT", interval="1h", limit=1)
    assert len(df) == 1
    assert "close" in df.columns
    assert df["close"].iloc[0] == 65200.0
