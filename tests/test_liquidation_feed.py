"""
Unit tests for LiquidationFeed.
"""

from datetime import datetime, timezone, timedelta
from src.data.liquidation_feed import LiquidationFeed, LiquidationEvent


def test_liquidation_event_model():
    ev = LiquidationEvent(
        symbol="BTC",
        side="SELL",
        amount_usd=5_000_000.0,
        price=64_000.0
    )
    assert ev.symbol == "BTC"
    assert ev.side == "SELL"
    assert ev.amount_usd == 5_000_000.0


def test_liquidation_summary_low_activity():
    feed = LiquidationFeed(cascade_threshold_usd=15_000_000.0)
    summary = feed.get_summary(window_minutes=15)
    assert summary.net_bias == "LOW_ACTIVITY"
    assert summary.is_cascade_active is False
    assert summary.total_long_usd == 0.0
    assert summary.total_short_usd == 0.0


def test_liquidation_heavy_long_cascade():
    feed = LiquidationFeed(cascade_threshold_usd=10_000_000.0)
    now = datetime.now(timezone.utc)

    # Add $12M in forced sell liquidations (longs wiped out)
    for _ in range(4):
        feed.record_event(
            LiquidationEvent(
                timestamp=now - timedelta(minutes=2),
                symbol="BTC",
                side="SELL",
                amount_usd=3_000_000.0,
                price=63_500.0
            )
        )

    # Add $1M in short liquidations
    feed.record_event(
        LiquidationEvent(
            timestamp=now - timedelta(minutes=5),
            symbol="BTC",
            side="BUY",
            amount_usd=1_000_000.0,
            price=63_600.0
        )
    )

    summary = feed.get_summary(window_minutes=15)
    assert summary.is_cascade_active is True
    assert summary.net_bias == "HEAVY_LONG_CASCADE"
    assert summary.total_long_usd == 12_000_000.0
    assert summary.total_short_usd == 1_000_000.0
    assert "Waterfall" in summary.headline_alert


def test_liquidation_short_squeeze_cascade():
    feed = LiquidationFeed(cascade_threshold_usd=10_000_000.0)
    now = datetime.now(timezone.utc)

    # Add $15M in short liquidations (forced buys)
    feed.record_event(
        LiquidationEvent(
            timestamp=now - timedelta(minutes=1),
            symbol="BTC",
            side="BUY",
            amount_usd=15_000_000.0,
            price=65_000.0
        )
    )

    summary = feed.get_summary(window_minutes=15)
    assert summary.is_cascade_active is True
    assert summary.net_bias == "HEAVY_SHORT_SQUEEZE"
    assert "Squeeze" in summary.headline_alert


def test_liquidation_pruning():
    feed = LiquidationFeed(cascade_threshold_usd=10_000_000.0, max_history_hours=1)
    old_time = datetime.now(timezone.utc) - timedelta(hours=3)

    feed.record_event(
        LiquidationEvent(
            timestamp=old_time,
            symbol="BTC",
            side="SELL",
            amount_usd=20_000_000.0,
            price=60_000.0
        )
    )

    # An event 3 hours old should be pruned when a new event is recorded or during evaluation
    feed.record_event(
        LiquidationEvent(
            timestamp=datetime.now(timezone.utc),
            symbol="BTC",
            side="SELL",
            amount_usd=100_000.0,
            price=64_000.0
        )
    )

    summary = feed.get_summary(window_minutes=60)
    assert summary.total_long_usd == 100_000.0
