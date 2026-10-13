"""
Unit tests for SignalEngine master scanner loop and circuit-breaker integration.
"""

from unittest.mock import MagicMock
from src.signals.signal_engine import SignalEngine
from src.data.data_pipeline import UnifiedMarketSnapshot
from src.data.deribit_client import DeribitOptionSummary
from src.data.liquidation_feed import LiquidationSummary
from src.signals.alert_bot import ValidatedTradeSignal


def create_mock_snapshot(headlines=None, is_cascade=False, call_options=None):
    return UnifiedMarketSnapshot(
        currency="BTC",
        spot_price=65_000.0,
        funding_rate_8h=0.0001,
        dvol=50.0,
        call_options=call_options or [],
        recent_headlines=headlines or ["Bitcoin consolidates at $65,000"],
        liquidation_summary=LiquidationSummary(
            window_minutes=15,
            total_long_usd=20_000_000.0 if is_cascade else 100_000.0,
            total_short_usd=50_000.0,
            net_bias="HEAVY_LONG_CASCADE" if is_cascade else "LOW_ACTIVITY",
            is_cascade_active=is_cascade,
            dominant_volume_usd=20_000_000.0 if is_cascade else 100_000.0,
            headline_alert="Over $20M in Bitcoin Long Positions Liquidated in Sudden Downside Waterfall" if is_cascade else "Normal conditions"
        )
    )


def test_signal_engine_circuit_breaker_bearish_news():
    mock_pipeline = MagicMock()
    # Provide a bearish headline that triggers Stage-1 circuit breaker
    snapshot = create_mock_snapshot(headlines=["Major Crypto Exchange Reports $400M Exploit and Halts Withdrawals"])
    mock_pipeline.get_snapshot.return_value = snapshot

    engine = SignalEngine(pipeline=mock_pipeline)
    signals = engine.scan_market(currency="BTC")

    # Circuit breaker must halt call options scanner immediately
    assert len(signals) == 0


def test_signal_engine_circuit_breaker_liquidation_cascade():
    mock_pipeline = MagicMock()
    # Provide normal headline, but liquidation cascade active
    snapshot = create_mock_snapshot(headlines=["Bitcoin trading steady"], is_cascade=True)
    mock_pipeline.get_snapshot.return_value = snapshot

    engine = SignalEngine(pipeline=mock_pipeline)
    signals = engine.scan_market(currency="BTC")

    # Massive long liquidation cascade must halt call buying
    assert len(signals) == 0


def test_signal_engine_end_to_end_high_conviction(tmp_path):
    mock_pipeline = MagicMock()
    mock_alert_bot = MagicMock()

    candidate = DeribitOptionSummary(
        instrument_name="BTC-28OCT26-66000-C",
        currency="BTC",
        strike=66_000.0,
        option_type="call",
        expiration_timestamp=1790000000,
        dte_days=3.0,
        best_bid_price=980.0,
        best_ask_price=1000.0,  # 2% tight spread
        mark_price=990.0,
        underlying_price=65_000.0,
        implied_volatility=0.52,
        delta=0.48,
        open_interest=85.0
    )

    snapshot = create_mock_snapshot(
        headlines=["SEC Approves First Spot Bitcoin ETF In-Kind Redemptions"],
        is_cascade=False,
        call_options=[candidate]
    )
    mock_pipeline.get_snapshot.return_value = snapshot
    mock_pipeline.filter_tradeable_calls.return_value = [candidate]

    # Initialize paper trader with temporary test ledger
    from src.signals.paper_trader import PaperTrader
    test_ledger = tmp_path / "test_paper_ledger.json"
    paper_trader = PaperTrader(ledger_file=test_ledger)

    engine = SignalEngine(
        pipeline=mock_pipeline,
        alert_bot=mock_alert_bot,
        paper_trader=paper_trader
    )

    signals = engine.scan_market(
        currency="BTC",
        broadcast_alerts=True,
        record_paper_trades=True
    )

    assert len(signals) == 1
    sig = signals[0]
    assert isinstance(sig, ValidatedTradeSignal)
    assert sig.contract_symbol == "BTC-28OCT26-66000-C"
    assert sig.laya_confidence >= 0.75
    assert sig.projected_net_return_pct >= 0.03

    # Verify broadcast was called
    mock_alert_bot.broadcast.assert_called_once_with(sig)

    # Verify paper trader recorded the position
    assert len(paper_trader.positions) == 1
    opened_pos = list(paper_trader.positions.values())[0]
    assert opened_pos.contract_symbol == "BTC-28OCT26-66000-C"
    assert opened_pos.entry_ask_usd == 1000.0


def test_signal_engine_single_loop_iteration():
    mock_pipeline = MagicMock()
    snapshot = create_mock_snapshot(headlines=["Routine crypto news"])
    mock_pipeline.get_snapshot.return_value = snapshot
    mock_pipeline.filter_tradeable_calls.return_value = []

    engine = SignalEngine(pipeline=mock_pipeline)
    # Run 1 iteration of scanner loop
    engine.run_scanner_loop(currency="BTC", interval_seconds=0, iterations=1)
    mock_pipeline.get_snapshot.assert_called_once()
