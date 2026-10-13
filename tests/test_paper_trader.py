"""
Unit tests for src/signals/paper_trader.py.
Verifies virtual position lifecycle, target take-profit exits, and stop-loss execution.
"""

from pathlib import Path
from src.signals.alert_bot import ValidatedTradeSignal
from src.signals.paper_trader import PaperTrader


def test_paper_trader_lifecycle(tmp_path: Path):
    ledger = tmp_path / "test_ledger.json"
    trader = PaperTrader(ledger_file=ledger)

    sig = ValidatedTradeSignal(
        contract_symbol="BTC-24OCT26-68000-C",
        underlying_spot=65000.0,
        strike_price=68000.0,
        dte_days=3.0,
        mark_premium_usd=720.0,
        best_ask_usd=730.0,
        delta=0.35,
        gamma=0.00004,
        theta_daily=-130.0,
        iv_annual=0.48,
        laya_confidence=0.82,
        projected_net_return_pct=0.18,
        breakeven_spot_move_pct=1.8,
        target_exit_usd=900.0,
        stop_loss_usd=550.0
    )

    # 1. Open Position
    pos = trader.open_position_from_signal(sig, contracts=1.0)
    assert pos.status == "OPEN"
    assert pos.entry_ask_usd == 730.0
    assert pos.entry_fee_usd > 0.0

    # 2. Update price within range (still OPEN)
    updated = trader.update_position_price(pos.position_id, current_bid=800.0, current_spot=66000.0)
    assert updated.status == "OPEN"

    # 3. Price crosses target (910 USD >= 900 USD target) -> Auto Exit
    closed = trader.update_position_price(pos.position_id, current_bid=910.0, current_spot=67500.0)
    assert closed.status == "CLOSED_TARGET"
    assert closed.realized_pnl_usd is not None
    assert closed.realized_pnl_usd > 0.0
    assert closed.realized_return_pct > 0.15

    # 4. Check Portfolio Summary
    summary = trader.get_summary()
    assert summary.total_trades == 1
    assert summary.open_positions_count == 0
    assert summary.closed_positions_count == 1
    assert summary.win_count == 1
    assert summary.win_rate == 1.0
    assert summary.total_realized_pnl_usd > 100.0


def test_paper_trader_stop_loss(tmp_path: Path):
    ledger = tmp_path / "test_ledger_stop.json"
    trader = PaperTrader(ledger_file=ledger)

    sig = ValidatedTradeSignal(
        contract_symbol="BTC-24OCT26-70000-C",
        underlying_spot=65000.0,
        strike_price=70000.0,
        dte_days=2.0,
        mark_premium_usd=500.0,
        best_ask_usd=520.0,
        delta=0.25,
        gamma=0.00003,
        theta_daily=-100.0,
        iv_annual=0.45,
        laya_confidence=0.76,
        projected_net_return_pct=0.15,
        breakeven_spot_move_pct=1.9,
        target_exit_usd=700.0,
        stop_loss_usd=400.0
    )

    pos = trader.open_position_from_signal(sig, contracts=1.0)
    # Price falls below stop loss (380 USD <= 400 USD stop) -> Auto Exit
    closed = trader.update_position_price(pos.position_id, current_bid=380.0, current_spot=63500.0)
    assert closed.status == "CLOSED_STOP"
    assert closed.realized_pnl_usd < 0.0

    summary = trader.get_summary()
    assert summary.loss_count == 1
    assert summary.win_rate == 0.0
