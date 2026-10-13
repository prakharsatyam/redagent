"""
Unit tests for src/data/deribit_client.py.
Tests URL configuration, response parsing, and mocked matrix extraction.
"""

from unittest.mock import patch, MagicMock
from src.data.deribit_client import DeribitClient, DeribitOptionSummary


def test_deribit_client_urls():
    client_testnet = DeribitClient(use_testnet=True)
    assert "test.deribit.com" in client_testnet.base_url

    client_mainnet = DeribitClient(use_testnet=False)
    assert "www.deribit.com" in client_mainnet.base_url


@patch.object(DeribitClient, "_get")
def test_get_index_price(mock_get):
    mock_get.return_value = {"index_price": 65420.50}
    client = DeribitClient(use_testnet=True)
    price = client.get_index_price("btc_usd")
    assert price == 65420.50
    mock_get.assert_called_once_with("/public/get_index_price", {"index_name": "btc_usd"})


@patch.object(DeribitClient, "get_order_book")
@patch.object(DeribitClient, "get_active_options")
def test_get_call_options_matrix(mock_active, mock_ob):
    now_ms = 1700000000000
    # Expiry 3 days in future
    exp_ms = now_ms + int(3.5 * 86400 * 1000)

    mock_active.return_value = [
        {
            "instrument_name": "BTC-30OCT26-68000-C",
            "option_type": "call",
            "is_active": True,
            "strike": 68000.0,
            "expiration_timestamp": exp_ms,
        },
        {
            # Put option should be filtered out
            "instrument_name": "BTC-30OCT26-60000-P",
            "option_type": "put",
            "is_active": True,
            "strike": 60000.0,
            "expiration_timestamp": exp_ms,
        }
    ]

    mock_ob.return_value = {
        "best_bid_price": 0.011,  # BTC quote
        "best_ask_price": 0.012,  # BTC quote
        "mark_price": 0.0115,
        "underlying_price": 65000.0,
        "mark_iv": 48.5,
        "greeks": {"delta": 0.34},
        "open_interest": 450.0,
    }

    with patch("time.time", return_value=now_ms / 1000.0):
        client = DeribitClient(use_testnet=True)
        matrix = client.get_call_options_matrix(currency="BTC", max_dte=14.0, min_dte=0.5)

    assert len(matrix) == 1
    call = matrix[0]
    assert isinstance(call, DeribitOptionSummary)
    assert call.instrument_name == "BTC-30OCT26-68000-C"
    assert call.strike == 68000.0
    # Converted to USD: 0.011 * 65000 = 715 USD
    assert call.best_bid_price == 715.0
    assert call.best_ask_price == 780.0
    assert call.delta == 0.34
    assert call.implied_volatility == 0.485


@patch.object(DeribitClient, "_get")
def test_get_active_options_sol(mock_get):
    mock_get.return_value = [
        {"instrument_name": "SOL_USDC-14OCT26-105-C", "base_currency": "SOL", "option_type": "call"},
        {"instrument_name": "BTC_USDC-14OCT26-65000-C", "base_currency": "BTC", "option_type": "call"},
    ]
    client = DeribitClient(use_testnet=True)
    sol_opts = client.get_active_options("SOL")
    assert len(sol_opts) == 1
    assert sol_opts[0]["instrument_name"] == "SOL_USDC-14OCT26-105-C"
    mock_get.assert_called_once_with("/public/get_instruments", {"currency": "USDC", "kind": "option", "expired": "false"})


@patch.object(DeribitClient, "get_order_book")
@patch.object(DeribitClient, "get_active_options")
def test_get_call_options_matrix_sol_linear(mock_active, mock_ob):
    now_ms = 1700000000000
    exp_ms = now_ms + int(2.0 * 86400 * 1000)

    mock_active.return_value = [
        {
            "instrument_name": "SOL_USDC-14OCT26-105-C",
            "option_type": "call",
            "is_active": True,
            "strike": 105.0,
            "expiration_timestamp": exp_ms,
        }
    ]

    mock_ob.return_value = {
        "best_bid_price": 4.5,  # Linear USDC quote directly in USD
        "best_ask_price": 5.0,
        "mark_price": 4.75,
        "underlying_price": 108.0,
        "mark_iv": 62.0,
        "greeks": {"delta": 0.55},
        "open_interest": 1200.0,
    }

    with patch("time.time", return_value=now_ms / 1000.0):
        client = DeribitClient(use_testnet=True)
        matrix = client.get_call_options_matrix(currency="SOL", max_dte=14.0, min_dte=0.5)

    assert len(matrix) == 1
    call = matrix[0]
    assert call.instrument_name == "SOL_USDC-14OCT26-105-C"
    # Linear: 4.5 is already USD, should NOT be 4.5 * 108
    assert call.best_bid_price == 4.5
    assert call.best_ask_price == 5.0
    assert call.delta == 0.55

