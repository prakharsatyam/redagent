"""
Live Breaking News and Social Streamer.
Fetches and aggregates real-time news headlines from CryptoPanic, crypto RSS feeds,
and market-moving alerts to feed Stage-1 Laya News Protective Radar.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import logging
import requests
from pydantic import BaseModel, Field

from config.settings import settings

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    """Normalized breaking news headline container."""
    title: str
    source: str
    published_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    url: str = ""
    currencies: List[str] = Field(default_factory=lambda: ["BTC"])
    domain: str = ""
    is_hot: bool = False


# High-impact keywords for crypto market volatility
KEY_VOLATILITY_TERMS = [
    "sec", "etf", "fed", "powell", "fomc", "rate cut", "hike", "inflation", "cpi",
    "hack", "exploit", "outflow", "inflow", "liquidation", "insolvent", "binance",
    "coinbase", "tether", "usdt", "trump", "treasury", "whale", "dump", "pump"
]

# Curated fallback simulation headlines for testing and offline environments
MOCK_HEADLINES = [
    {
        "title": "US SEC Approves First In-Kind Creation for Spot Bitcoin ETFs",
        "source": "CryptoPanic",
        "domain": "coindesk.com",
        "is_hot": True
    },
    {
        "title": "Federal Reserve Hints at Accelerated 50bps Rate Cut in Upcoming Meeting",
        "source": "MacroFeed",
        "domain": "bloomberg.com",
        "is_hot": True
    },
    {
        "title": "Over $180M in Bitcoin Short Positions Liquidated in 1-Hour Spike",
        "source": "CoinglassAlert",
        "domain": "coinglass.com",
        "is_hot": True
    },
    {
        "title": "Institutional Treasury Allocates $500M into Spot BTC Reserve",
        "source": "CryptoPanic",
        "domain": "theblock.co",
        "is_hot": False
    }
]


class NewsStreamer:
    """Streamer & aggregator for crypto breaking headlines."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or getattr(settings, "cryptopanic_api_key", "")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RedAgent-NewsStreamer/1.0"})

    def fetch_cryptopanic(self, currency: str = "BTC", filter_kind: str = "rising") -> List[NewsArticle]:
        """
        Fetches breaking headlines from CryptoPanic developer API if token available.
        Falls back to curated recent headlines if token is empty or endpoint fails.
        """
        if not self.api_key or self.api_key == "YOUR_CRYPTOPANIC_API_KEY":
            logger.debug("CryptoPanic API key not set; using simulated market feed.")
            return self._get_fallback_articles(currency)

        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": self.api_key,
            "currencies": currency,
            "filter": filter_kind,
            "public": "true"
        }

        try:
            resp = self.session.get(url, params=params, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                articles = []
                for item in results:
                    articles.append(
                        NewsArticle(
                            title=item.get("title", ""),
                            source=item.get("source", {}).get("title", "CryptoPanic"),
                            url=item.get("url", ""),
                            domain=item.get("source", {}).get("domain", ""),
                            currencies=[c.get("code", "BTC") for c in item.get("currencies", [])] or [currency],
                            is_hot=item.get("votes", {}).get("positive", 0) > 10
                        )
                    )
                return articles
            else:
                logger.warning(f"CryptoPanic API returned status {resp.status_code}. Using fallback.")
                return self._get_fallback_articles(currency)
        except Exception as e:
            logger.warning(f"Failed to fetch CryptoPanic API: {e}. Using fallback.")
            return self._get_fallback_articles(currency)

    def _get_fallback_articles(self, currency: str = "BTC") -> List[NewsArticle]:
        """Generates realistic structured news articles for testing / offline mode."""
        articles = []
        now = datetime.now(timezone.utc)
        for h in MOCK_HEADLINES:
            articles.append(
                NewsArticle(
                    title=h["title"],
                    source=h["source"],
                    published_at=now,
                    domain=h["domain"],
                    currencies=[currency],
                    is_hot=h["is_hot"]
                )
            )
        return articles

    def get_latest_headlines(self, currency: str = "BTC", limit: int = 10) -> List[str]:
        """Returns list of raw headline strings for Stage-1 Laya News Classifier."""
        articles = self.fetch_cryptopanic(currency=currency)
        # Deduplicate titles
        seen = set()
        headlines = []
        for a in articles:
            clean_title = a.title.strip()
            if clean_title and clean_title not in seen:
                seen.add(clean_title)
                headlines.append(clean_title)
            if len(headlines) >= limit:
                break
        return headlines

    def filter_high_impact_headlines(self, headlines: List[str]) -> List[str]:
        """Filters headlines that match market-moving macro or crypto keywords."""
        high_impact = []
        for h in headlines:
            h_lower = h.lower()
            if any(term in h_lower for term in KEY_VOLATILITY_TERMS):
                high_impact.append(h)
        return high_impact
