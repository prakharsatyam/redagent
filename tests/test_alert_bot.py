"""
Unit tests for src/signals/alert_bot.py.
Verifies alert message formatting and mocked Telegram API delivery.
"""

from unittest.mock import patch, MagicMock
from src.signals.alert_bot import AlertBot, ValidatedTradeSignal


def test_format_alert_content():
    sig = ValidatedTradeSignal(
        contract_symbol="BTC-24OCT26-68000-C",
        underlying_spot=65850.0,
        strike_price=68000.0,
        dte_days=3.8,
        mark_premium_usd=725.0,
        best_ask_usd=735.0,
        delta=0.34,
        gamma=0.000045,
        theta_daily=-138.0,
        iv_annual=0.485,
        iv_rank=22.5,
        laya_confidence=0.845,
        projected_net_return_pct=0.192,
        breakeven_spot_move_pct=1.78,
        target_exit_usd=930.0,
        stop_loss_usd=550.0
    )

    bot = AlertBot(bot_token="test_token", chat_id="12345")
    msg = bot.format_alert(sig)

    assert "REDAGENT CALL OPTION SIGNAL" in msg
    assert "BTC-24OCT26-68000-C" in msg
    assert "84.5%" in msg
    assert "$930.00" in msg
    assert "$550.00" in msg
    assert "+19.2%" in msg


@patch("requests.post")
def test_send_telegram_success(mock_post):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_post.return_value = mock_resp

    bot = AlertBot(bot_token="valid_token", chat_id="valid_chat")
    success = bot.send_telegram("Test message")
    assert success is True
    mock_post.assert_called_once()
