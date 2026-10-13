"""
Unit tests for DataPipeline consolidator.
"""

from unittest.mock import MagicMock
from src.data.data_pipeline import DataPipeline, UnifiedMarketSnapshot
from src.data.deribit_client import DeribitOptionSummary
from src.data.binance_client import BinanceMarketContext
from src.data.liquidation_feed import LiquidationSummary


def test_data_pipeline_snapshot_mocked():
    mock_deribit = MagicMock()
    mock_binance = MagicMock()
    mock_news = MagicMock()
    mock_liq = MagicMock()

    import pandas as pd
    mock_binance.get_market_context.return_value = BinanceMarketContext(
        symbol="BTCUSDT",
        spot_price=64_500.0,
        funding_rate_8h=0.00015,
        open_interest=50_000.0
    )
    mock_binance.get_klines.return_value = pd.DataFrame({
        "timestamp": pd.date_range("2026-10-10", periods=30, freq="1h", tz="UTC"),
        "open": [64000.0 + i * 20 for i in range(30)],
        "high": [64100.0 + i * 20 for i in range(30)],
        "low": [63900.0 + i * 20 for i in range(30)],
        "close": [64050.0 + i * 20 for i in range(30)],
        "volume": [100.0] * 30
    })

    mock_call = DeribitOptionSummary(
        instrument_name="BTC-28OCT26-65000-C",
        currency="BTC",
        strike=65_000.0,
        option_type="call",
        expiration_timestamp=1790000000,
        dte_days=5.0,
        best_bid_price=1450.0,
        best_ask_price=1500.0,
        mark_price=1475.0,
        underlying_price=64_500.0,
        implied_volatility=0.55,
        delta=0.48,
        open_interest=120.0
    )
    mock_deribit.get_call_options_matrix.return_value = [mock_call]

    mock_news.get_latest_headlines.return_value = [
        "SEC Approves In-Kind Bitcoin Creation",
        "Fed Rate Cut Anticipation Mounts"
    ]

    mock_liq.get_summary.return_value = LiquidationSummary(
        window_minutes=15,
        total_long_usd=1_000_000.0,
        total_short_usd=500_000.0,
        net_bias="BALANCED",
        is_cascade_active=False,
        dominant_volume_usd=1_000_000.0,
        headline_alert="Moderate Activity"
    )
    mock_liq._events = [1]  # bypass fetch_recent

    pipeline = DataPipeline(
        deribit_client=mock_deribit,
        binance_client=mock_binance,
        news_streamer=mock_news,
        liquidation_feed=mock_liq
    )

    snapshot = pipeline.get_snapshot("BTC")
    assert isinstance(snapshot, UnifiedMarketSnapshot)
    assert snapshot.currency == "BTC"
    assert snapshot.spot_price == 64_500.0
    assert snapshot.funding_rate_8h == 0.00015
    assert len(snapshot.call_options) == 1
    assert len(snapshot.recent_headlines) == 2
    assert snapshot.liquidation_summary.net_bias == "BALANCED"
    assert snapshot.technical_summary is not None
    assert snapshot.technical_summary.trend_regime == "BULLISH_UPTREND"


def test_filter_tradeable_calls():
    pipeline = DataPipeline()

    call_good = DeribitOptionSummary(
        instrument_name="BTC-GOOD-C",
        strike=65000.0,
        option_type="call",
        expiration_timestamp=1790000000,
        dte_days=5.0,
        best_bid_price=1000.0,
        best_ask_price=1050.0,  # 5% spread
        mark_price=1025.0,
        underlying_price=64000.0,
        implied_volatility=0.50,
        delta=0.45,
        open_interest=50.0
    )

    call_wide_spread = DeribitOptionSummary(
        instrument_name="BTC-WIDE-C",
        strike=75000.0,
        option_type="call",
        expiration_timestamp=1790000000,
        dte_days=5.0,
        best_bid_price=500.0,
        best_ask_price=700.0,  # 28.5% spread > 12% max
        mark_price=600.0,
        underlying_price=64000.0,
        implied_volatility=0.60,
        delta=0.25,
        open_interest=10.0
    )

    call_low_oi = DeribitOptionSummary(
        instrument_name="BTC-LOW-OI-C",
        strike=65000.0,
        option_type="call",
        expiration_timestamp=1790000000,
        dte_days=5.0,
        best_bid_price=1000.0,
        best_ask_price=1050.0,
        mark_price=1025.0,
        underlying_price=64000.0,
        implied_volatility=0.50,
        delta=0.45,
        open_interest=0.01  # below min OI
    )

    filtered = pipeline.filter_tradeable_calls([call_good, call_wide_spread, call_low_oi])
    assert len(filtered) == 1
    assert filtered[0].instrument_name == "BTC-GOOD-C"
