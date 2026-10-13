"""
Binance Public REST Client for Spot Prices, Funding Rates, and Open Interest.
Provides spot momentum and perpetual funding rate context for RedAgent.
"""

from typing import Optional, Dict, Any
import requests
import pandas as pd
from pydantic import BaseModel, Field


class BinanceMarketContext(BaseModel):
    """Underlying spot price and perpetual futures sentiment indicators."""
    symbol: str = "BTCUSDT"
    spot_price: float
    funding_rate_8h: float = Field(description="Latest 8h funding rate (e.g. 0.0001 = 0.01%)")
    open_interest: float = 0.0


class BinanceClient:
    """REST Client for Binance Public Market Data."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.spot_base_url = "https://api.binance.com"
        self.fapi_base_url = "https://fapi.binance.com"
        self.timeout = timeout_seconds
        self.session = requests.Session()

    def get_spot_price(self, symbol: str = "BTCUSDT") -> float:
        """Fetches current spot price from Binance."""
        url = f"{self.spot_base_url}/api/v3/ticker/price"
        res = self.session.get(url, params={"symbol": symbol}, timeout=self.timeout)
        res.raise_for_status()
        return float(res.json().get("price", 0.0))

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """Fetches latest 8h funding rate from Binance USDT-M Futures."""
        url = f"{self.fapi_base_url}/fapi/v1/fundingRate"
        res = self.session.get(url, params={"symbol": symbol, "limit": 1}, timeout=self.timeout)
        res.raise_for_status()
        data = res.json()
        if data and isinstance(data, list):
            return float(data[0].get("fundingRate", 0.0))
        return 0.0

    def get_open_interest(self, symbol: str = "BTCUSDT") -> float:
        """Fetches open interest in coin contracts from Binance USDT-M Futures."""
        url = f"{self.fapi_base_url}/fapi/v1/openInterest"
        res = self.session.get(url, params={"symbol": symbol}, timeout=self.timeout)
        res.raise_for_status()
        return float(res.json().get("openInterest", 0.0))

    def get_market_context(self, symbol: str = "BTCUSDT") -> BinanceMarketContext:
        """Gathers synchronized spot price, funding rate, and open interest."""
        try:
            spot = self.get_spot_price(symbol)
        except Exception:
            spot = 65000.0

        try:
            funding = self.get_funding_rate(symbol)
        except Exception:
            funding = 0.0001

        try:
            oi = self.get_open_interest(symbol)
        except Exception:
            oi = 0.0

        return BinanceMarketContext(
            symbol=symbol,
            spot_price=spot,
            funding_rate_8h=funding,
            open_interest=oi
        )

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV candlestick records from Binance Public REST API.
        Returns a DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
        """
        url = f"{self.spot_base_url}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            res = self.session.get(url, params=params, timeout=self.timeout)
            res.raise_for_status()
            data = res.json()
            rows = []
            for item in data:
                rows.append({
                    "timestamp": pd.to_datetime(item[0], unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
