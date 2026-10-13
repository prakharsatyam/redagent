"""
Signal Alert Broadcaster for Telegram and Discord.
Formats high-conviction trade setups into structured, actionable messages.
"""

from typing import Dict, Any, Optional
import requests
from pydantic import BaseModel, Field

from config.settings import settings


class ValidatedTradeSignal(BaseModel):
    """Complete signal payload ready for broadcast and execution."""
    contract_symbol: str
    underlying_spot: float
    strike_price: float
    dte_days: float
    mark_premium_usd: float
    best_ask_usd: float
    delta: float
    gamma: float
    theta_daily: float
    iv_annual: float
    iv_rank: Optional[float] = None
    laya_confidence: float
    projected_net_return_pct: float
    breakeven_spot_move_pct: float
    target_exit_usd: float
    stop_loss_usd: float


class AlertBot:
    """Broadcaster for Telegram and Discord signal channels."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_seconds: float = 8.0
    ):
        self.bot_token = bot_token or settings.api.telegram_bot_token
        self.chat_id = chat_id or settings.api.telegram_chat_id
        self.timeout = timeout_seconds

    def format_alert(self, sig: ValidatedTradeSignal) -> str:
        """Formats signal payload into professional Telegram markdown."""
        target_gain_pct = ((sig.target_exit_usd - sig.best_ask_usd) / sig.best_ask_usd) * 100.0 if sig.best_ask_usd > 0 else 25.0
        stop_loss_pct = ((sig.stop_loss_usd - sig.best_ask_usd) / sig.best_ask_usd) * 100.0 if sig.best_ask_usd > 0 else -25.0
        iv_rank_str = f"{sig.iv_rank:.1f}%" if sig.iv_rank is not None else "N/A"

        msg = (
            f"🚨 *REDAGENT CALL OPTION SIGNAL* 🚨\n\n"
            f"🎯 *Asset:* `{sig.contract_symbol}`\n"
            f"⚡ *Conviction:* `{sig.laya_confidence:.1%}` (Laya System 1 Decision)\n\n"
            f"📊 *Market Context:*\n"
            f"• Spot: `${sig.underlying_spot:,.2f}` | Strike: `${sig.strike_price:,.2f}`\n"
            f"• DTE: `{sig.dte_days:.1f} days` | Mark Premium: `${sig.mark_premium_usd:.2f}`\n"
            f"• Greeks: Delta `{sig.delta:.3f}`, Theta `-${abs(sig.theta_daily):.2f}/day`\n"
            f"• IV: `{sig.iv_annual:.1%}` | IV Rank: `{iv_rank_str}`\n\n"
            f"💰 *Payoff & Friction Simulation:*\n"
            f"• Breakeven Spot Move: `+{sig.breakeven_spot_move_pct:.2f}%`\n"
            f"• Est. Net Return: `+{sig.projected_net_return_pct:.1%}` (After all fees & theta)\n\n"
            f"📋 *Actionable Execution Plan:*\n"
            f"• Entry Ask: `${sig.best_ask_usd:.2f}` (Limit Maker/Taker)\n"
            f"• Target Exit: `${sig.target_exit_usd:.2f}` (`+{target_gain_pct:.1f}%`)\n"
            f"• Stop Loss: `${sig.stop_loss_usd:.2f}` (`{stop_loss_pct:.1f}%`)\n"
        )
        return msg

    def send_telegram(self, message: str) -> bool:
        """Sends message via Telegram Bot API."""
        if not self.bot_token or not self.chat_id:
            return False

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "Markdown"
        }
        try:
            res = requests.post(url, json=payload, timeout=self.timeout)
            return res.status_code == 200
        except Exception:
            return False

    def broadcast(self, signal: ValidatedTradeSignal) -> str:
        """Formats and broadcasts signal, returning formatted string."""
        formatted = self.format_alert(signal)
        self.send_telegram(formatted)
        return formatted
