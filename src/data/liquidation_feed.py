"""
Real-time Liquidation Cascade Radar.
Tracks long and short liquidation waterfalls across Binance and major derivatives exchanges
to detect momentum cascades and feed protective alerts into Stage-1 Laya News Radar.
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Deque
from collections import deque
import logging
import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class LiquidationEvent(BaseModel):
    """Normalized liquidation event container."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str  # e.g. "BTC" or "BTCUSDT"
    side: str    # "SELL" (Long liquidated) or "BUY" (Short liquidated)
    amount_usd: float
    price: float


class LiquidationSummary(BaseModel):
    """Aggregated liquidation cascade metrics over a rolling window."""
    window_minutes: int
    total_long_usd: float
    total_short_usd: float
    net_bias: str  # "HEAVY_LONG_CASCADE", "HEAVY_SHORT_SQUEEZE", "BALANCED", "LOW_ACTIVITY"
    is_cascade_active: bool
    dominant_volume_usd: float
    headline_alert: str


class LiquidationFeed:
    """Rolling window tracker for crypto liquidations."""

    def __init__(self, cascade_threshold_usd: float = 15_000_000.0, max_history_hours: int = 2):
        self.cascade_threshold_usd = cascade_threshold_usd
        self.max_history_hours = max_history_hours
        self._events: Deque[LiquidationEvent] = deque()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RedAgent-LiquidationFeed/1.0"})

    def record_event(self, event: LiquidationEvent) -> None:
        """Records a new liquidation tick and evicts old ticks outside window."""
        self._events.append(event)
        self._prune_old_events()

    def record_batch(self, events: List[LiquidationEvent]) -> None:
        """Batch records liquidation events."""
        for ev in events:
            self._events.append(ev)
        self._prune_old_events()

    def _prune_old_events(self) -> None:
        """Evicts events older than max_history_hours."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.max_history_hours)
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def get_summary(self, window_minutes: int = 15) -> LiquidationSummary:
        """Computes aggregate liquidation metrics over the specified rolling minutes."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_minutes)

        long_vol = 0.0
        short_vol = 0.0

        for ev in self._events:
            if ev.timestamp >= cutoff:
                if ev.side.upper() in ["SELL", "LONG"]:  # Forced sell = long position liquidated
                    long_vol += ev.amount_usd
                else:  # Forced buy = short position liquidated
                    short_vol += ev.amount_usd

        total_vol = long_vol + short_vol

        # Classify cascade regime
        if total_vol < 1_000_000.0:
            net_bias = "LOW_ACTIVITY"
            headline = f"Normal crypto derivatives liquidity (15m liqs: ${total_vol / 1e6:.1f}M)"
        elif long_vol > short_vol * 2.0 and long_vol >= self.cascade_threshold_usd:
            net_bias = "HEAVY_LONG_CASCADE"
            headline = f"Over ${long_vol / 1e6:.1f}M in Bitcoin Long Positions Liquidated in Sudden Downside Waterfall"
        elif short_vol > long_vol * 2.0 and short_vol >= self.cascade_threshold_usd:
            net_bias = "HEAVY_SHORT_SQUEEZE"
            headline = f"Over ${short_vol / 1e6:.1f}M in Short Positions Liquidated in Aggressive Squeeze"
        elif total_vol >= self.cascade_threshold_usd:
            net_bias = "BALANCED_HIGH_VOLATILITY"
            headline = f"High Dual-Sided Liquidation Churn: ${total_vol / 1e6:.1f}M Total Liquidated in {window_minutes}m"
        else:
            net_bias = "BALANCED"
            headline = f"Moderate Derivatives Liquidations: ${total_vol / 1e6:.1f}M in {window_minutes}m"

        is_cascade = total_vol >= self.cascade_threshold_usd

        return LiquidationSummary(
            window_minutes=window_minutes,
            total_long_usd=long_vol,
            total_short_usd=short_vol,
            net_bias=net_bias,
            is_cascade_active=is_cascade,
            dominant_volume_usd=max(long_vol, short_vol),
            headline_alert=headline
        )

    def fetch_recent_binance_liquidations(self, symbol: str = "BTCUSDT") -> List[LiquidationEvent]:
        """
        Queries Binance public forceOrders REST endpoint if available.
        Falls back to graceful empty list on network failure or rate limits.
        """
        url = "https://fapi.binance.com/fapi/v1/forceOrders"
        params = {"symbol": symbol, "limit": 50}
        try:
            resp = self.session.get(url, params=params, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                events = []
                for item in data:
                    ts = datetime.fromtimestamp(item.get("time", 0) / 1000, tz=timezone.utc)
                    side = item.get("side", "SELL")
                    qty = float(item.get("executedQty", 0.0))
                    avg_price = float(item.get("avgPrice", 0.0))
                    amount_usd = qty * avg_price
                    events.append(
                        LiquidationEvent(
                            timestamp=ts,
                            symbol=symbol,
                            side=side,
                            amount_usd=amount_usd,
                            price=avg_price
                        )
                    )
                self.record_batch(events)
                return events
        except Exception as e:
            logger.debug(f"Binance liquidation query failed or timed out: {e}")

        return []
