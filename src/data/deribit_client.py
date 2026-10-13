"""
Deribit API v2 Client for Crypto Options Market Data.
Connects to Deribit REST endpoints to fetch options matrices, tick order books,
DVOL volatility indices, and index spot prices.
"""

import time
from typing import List, Dict, Optional, Any
import requests
from pydantic import BaseModel, Field

from config.settings import settings


class DeribitOptionSummary(BaseModel):
    """Normalized snapshot of a Deribit option contract."""
    instrument_name: str
    currency: str = "BTC"
    strike: float
    option_type: str = Field(description="'call' or 'put'")
    expiration_timestamp: int
    dte_days: float
    best_bid_price: float
    best_ask_price: float
    mark_price: float
    underlying_price: float
    implied_volatility: float
    delta: Optional[float] = None
    open_interest: float = 0.0


class DeribitClient:
    """REST Client for Deribit Options API."""

    def __init__(self, use_testnet: Optional[bool] = None, timeout_seconds: float = 10.0):
        self.use_testnet = settings.trading.deribit_use_testnet if use_testnet is None else use_testnet
        self.base_url = (
            "https://test.deribit.com/api/v2" if self.use_testnet else "https://www.deribit.com/api/v2"
        )
        self.timeout = timeout_seconds
        self.session = requests.Session()

    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = self.session.get(url, params=params or {}, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if "result" not in payload:
            raise ValueError(f"Deribit API error: {payload.get('error', 'Unknown response format')}")
        return payload["result"]

    def get_index_price(self, index_name: str = "btc_usd") -> float:
        """Fetches current underlying index spot price (e.g. BTC-USD index)."""
        res = self._get("/public/get_index_price", {"index_name": index_name})
        return float(res.get("index_price", 0.0))

    def get_active_options(self, currency: str = "BTC") -> List[Dict[str, Any]]:
        """Fetches all active (non-expired) option instruments for a currency."""
        curr = currency.upper()
        if curr == "SOL":
            # Deribit lists SOL options as linear contracts under currency=USDC
            params = {"currency": "USDC", "kind": "option", "expired": "false"}
            instruments = self._get("/public/get_instruments", params)
            return [
                inst for inst in instruments
                if inst.get("base_currency") == "SOL" or inst.get("instrument_name", "").startswith("SOL")
            ]
        params = {"currency": curr, "kind": "option", "expired": "false"}
        return self._get("/public/get_instruments", params)

    def get_order_book(self, instrument_name: str, depth: int = 5) -> Dict[str, Any]:
        """Fetches top-of-book and depth for a specific option contract."""
        return self._get("/public/get_order_book", {"instrument_name": instrument_name, "depth": depth})

    def get_call_options_matrix(
        self,
        currency: str = "BTC",
        max_dte: float = 14.0,
        min_dte: float = 0.5
    ) -> List[DeribitOptionSummary]:
        """
        Fetches and filters active Call options within the desired DTE window.
        Extracts strikes, current bids/asks, mark prices, and Greeks.
        """
        instruments = self.get_active_options(currency=currency)
        now_ms = int(time.time() * 1000)
        candidates = []

        for inst in instruments:
            # Filter for Call options only
            if inst.get("option_type") != "call" or not inst.get("is_active", True):
                continue

            exp_ms = inst.get("expiration_timestamp", 0)
            diff_ms = exp_ms - now_ms
            dte = diff_ms / (1000.0 * 86400.0)

            # DTE filter
            if not (min_dte <= dte <= max_dte):
                continue

            name = inst.get("instrument_name")
            strike = float(inst.get("strike", 0.0))

            try:
                ob = self.get_order_book(name, depth=1)
                best_bid = float(ob.get("best_bid_price", 0.0))
                best_ask = float(ob.get("best_ask_price", 0.0))
                mark_price = float(ob.get("mark_price", 0.0))
                underlying_price = float(ob.get("underlying_price", 0.0))
                iv = float(ob.get("mark_iv", 0.0)) / 100.0  # Convert percent to decimal (e.g. 52.5 -> 0.525)
                greeks = ob.get("greeks", {})
                delta = float(greeks.get("delta")) if greeks.get("delta") is not None else None
                oi = float(ob.get("open_interest", 0.0))

                # Check if contract is linear USDC quoted (e.g. SOL) vs inverse coin-margined (BTC, ETH)
                is_linear_usd = (
                    currency.upper() == "SOL"
                    or name.startswith("SOL")
                    or "_USDC" in name
                    or name.endswith("-USDC")
                )
                if is_linear_usd:
                    bid_usd = best_bid
                    ask_usd = best_ask
                    mark_usd = mark_price
                else:
                    # Inverse options are quoted in base crypto -> convert to USD by multiplying underlying_price
                    bid_usd = best_bid * underlying_price
                    ask_usd = best_ask * underlying_price
                    mark_usd = mark_price * underlying_price

                # Exclude contracts with zero bids/asks
                if bid_usd <= 0.0 or ask_usd <= 0.0:
                    continue

                candidates.append(
                    DeribitOptionSummary(
                        instrument_name=name,
                        currency=currency,
                        strike=strike,
                        option_type="call",
                        expiration_timestamp=exp_ms,
                        dte_days=dte,
                        best_bid_price=bid_usd,
                        best_ask_price=ask_usd,
                        mark_price=mark_usd,
                        underlying_price=underlying_price,
                        implied_volatility=iv,
                        delta=delta,
                        open_interest=oi,
                    )
                )
            except Exception:
                continue

        return candidates
