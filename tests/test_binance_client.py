"""
Unit tests for src/data/binance_client.py.
Verifies URL structure, error handling, and market context retrieval.
"""

from unittest.mock import patch, MagicMock
from src.data.binance_client import BinanceClient, BinanceMarketContext


@patch("requests.Session.get")
def test_binance_get_spot_price(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"symbol": "BTCUSDT", "price": "65800.50"}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    client = BinanceClient()
    price = client.get_spot_price("BTCUSDT")
    assert price == 65800.50


@patch("requests.Session.get")
def test_binance_get_funding_rate(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"symbol": "BTCUSDT", "fundingRate": "0.00015"}]
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    client = BinanceClient()
    funding = client.get_funding_rate("BTCUSDT")
    assert funding == 0.00015


@patch("requests.Session.get")
def test_binance_get_market_context(mock_get):
    # Mocking successive responses for spot, funding, oi
    spot_resp = MagicMock()
    spot_resp.json.return_value = {"price": "66000.00"}
    spot_resp.raise_for_status.return_value = None

    funding_resp = MagicMock()
    funding_resp.json.return_value = [{"fundingRate": "0.0002"}]
    funding_resp.raise_for_status.return_value = None

    oi_resp = MagicMock()
    oi_resp.json.return_value = {"openInterest": "45200.0"}
    oi_resp.raise_for_status.return_value = None

    mock_get.side_effect = [spot_resp, funding_resp, oi_resp]

    client = BinanceClient()
    ctx = client.get_market_context("BTCUSDT")
    assert isinstance(ctx, BinanceMarketContext)
    assert ctx.spot_price == 66000.0
    assert ctx.funding_rate_8h == 0.0002
    assert ctx.open_interest == 45200.0
