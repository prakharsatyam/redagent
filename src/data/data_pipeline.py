"""
Central Data Pipeline and Multi-Source Consolidator.
Synchronizes Deribit options chains, Binance spot/funding context,
CryptoPanic/news feeds, and liquidation cascade alerts into a UnifiedMarketSnapshot.
"""

from datetime import datetime, timezone
from typing import List, Optional
import logging
from pydantic import BaseModel, Field

from src.data.deribit_client import DeribitClient, DeribitOptionSummary
from src.data.binance_client import BinanceClient, BinanceMarketContext
from src.data.news_streamer import NewsStreamer
from src.data.liquidation_feed import LiquidationFeed, LiquidationSummary
from src.quant.technical_indicators import TechnicalIndicatorSummary, compute_technical_indicators

logger = logging.getLogger(__name__)


class UnifiedMarketSnapshot(BaseModel):
    """Consolidated multi-source market state at a specific snapshot timestamp."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    currency: str = "BTC"
    spot_price: float
    funding_rate_8h: float
    dvol: float = 52.0
    call_options: List[DeribitOptionSummary] = Field(default_factory=list)
    recent_headlines: List[str] = Field(default_factory=list)
    liquidation_summary: LiquidationSummary
    technical_summary: Optional[TechnicalIndicatorSummary] = None


class DataPipeline:
    """Central orchestrator for all real-time and alternative data feeds."""

    def __init__(
        self,
        deribit_client: Optional[DeribitClient] = None,
        binance_client: Optional[BinanceClient] = None,
        news_streamer: Optional[NewsStreamer] = None,
        liquidation_feed: Optional[LiquidationFeed] = None,
    ):
        self.deribit = deribit_client or DeribitClient()
        self.binance = binance_client or BinanceClient()
        self.news = news_streamer or NewsStreamer()
        self.liquidation = liquidation_feed or LiquidationFeed()

    def get_snapshot(
        self,
        currency: str = "BTC",
        max_dte: float = 14.0,
        min_dte: float = 0.5
    ) -> UnifiedMarketSnapshot:
        """
        Gathers live spot momentum, options matrix, breaking headlines,
        and liquidation cascades across all sources.
        """
        symbol_pair = f"{currency}USDT"

        # 1. Spot and funding rate from Binance
        try:
            binance_ctx: BinanceMarketContext = self.binance.get_market_context(symbol_pair)
            spot_price = binance_ctx.spot_price
            funding_rate = binance_ctx.funding_rate_8h
        except Exception as e:
            logger.warning(f"Error fetching Binance context: {e}; falling back to Deribit index.")
            try:
                spot_price = self.deribit.get_index_price(f"{currency.lower()}_usd")
            except Exception:
                curr_upper = currency.upper()
                if curr_upper == "ETH":
                    spot_price = 2600.0
                elif curr_upper == "SOL":
                    spot_price = 150.0
                else:
                    spot_price = 65_000.0
            funding_rate = 0.0001

        # 2. Deribit Call options matrix
        try:
            calls = self.deribit.get_call_options_matrix(
                currency=currency,
                max_dte=max_dte,
                min_dte=min_dte
            )
        except Exception as e:
            logger.warning(f"Error fetching Deribit options chain: {e}")
            calls = []

        # 3. Breaking headlines from NewsStreamer
        try:
            headlines = self.news.get_latest_headlines(currency=currency, limit=5)
        except Exception as e:
            logger.warning(f"Error fetching news headlines: {e}")
            headlines = []

        # 4. Liquidation cascade summary
        try:
            # Attempt to pull recent liquidations if feed is empty
            if not self.liquidation._events:
                self.liquidation.fetch_recent_binance_liquidations(symbol=symbol_pair)
            liq_summary = self.liquidation.get_summary(window_minutes=15)
        except Exception as e:
            logger.warning(f"Error computing liquidation summary: {e}")
            liq_summary = LiquidationSummary(
                window_minutes=15,
                total_long_usd=0.0,
                total_short_usd=0.0,
                net_bias="BALANCED",
                is_cascade_active=False,
                dominant_volume_usd=0.0,
                headline_alert="Normal derivatives conditions"
            )

        # 5. DVOL estimation
        dvol = 52.0
        if calls:
            valid_ivs = [c.implied_volatility * 100.0 for c in calls if c.implied_volatility > 0]
            if valid_ivs:
                dvol = sum(valid_ivs) / len(valid_ivs)

        # 6. Technical indicators from 1h OHLCV candlesticks
        try:
            klines_df = self.binance.get_klines(symbol=symbol_pair, interval="1h", limit=100)
            tech_summary = compute_technical_indicators(klines_df, symbol=symbol_pair, interval="1h")
        except Exception as e:
            logger.warning(f"Error computing technical indicators: {e}")
            tech_summary = None

        return UnifiedMarketSnapshot(
            currency=currency,
            spot_price=spot_price,
            funding_rate_8h=funding_rate,
            dvol=dvol,
            call_options=calls,
            recent_headlines=headlines,
            liquidation_summary=liq_summary,
            technical_summary=tech_summary
        )

    def filter_tradeable_calls(
        self,
        calls: List[DeribitOptionSummary],
        min_delta: float = 0.15,
        max_delta: float = 0.85,
        max_spread_pct: float = 0.12,
        min_open_interest: float = 0.1
    ) -> List[DeribitOptionSummary]:
        """
        Filters candidates strictly for liquid, tradeable Call options.
        Rejects wide bid-ask spreads, near-zero open interest, and deep wings.
        """
        tradeable = []
        for c in calls:
            # Spread check: (ask - bid) / ask
            if c.best_ask_price <= 0:
                continue
            spread_pct = (c.best_ask_price - c.best_bid_price) / c.best_ask_price
            if spread_pct > max_spread_pct:
                continue

            # Open interest check
            if c.open_interest < min_open_interest:
                continue

            # Delta bounds check (if delta available)
            if c.delta is not None and not (min_delta <= c.delta <= max_delta):
                continue

            tradeable.append(c)

        return tradeable
