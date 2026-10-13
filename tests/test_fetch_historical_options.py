"""
Unit tests for scripts/fetch_historical_options.py.
"""

from pathlib import Path
import pandas as pd
from unittest.mock import MagicMock
from scripts.fetch_historical_options import fetch_historical_options_ticks


def test_fetch_historical_options_mock_client():
    mock_client = MagicMock()
    mock_client.get_index_price.return_value = 65_000.0
    mock_client.get_active_options.return_value = [
        {
            "instrument_name": "BTC-28OCT26-66000-C",
            "option_type": "call",
            "is_active": True,
            "strike": 66000.0,
            "expiration_timestamp": 1790000000
        }
    ]
    mock_client.get_order_book.return_value = {
        "best_bid_price": 0.015,
        "best_ask_price": 0.016,
        "mark_iv": 52.0
    }

    df = fetch_historical_options_ticks(currency="BTC", days=7, client=mock_client)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "spot_price" in df.columns
    assert "bid_price" in df.columns
    assert "ask_price" in df.columns


def test_fetch_historical_options_parquet_save(tmp_path: Path):
    mock_client = MagicMock()
    mock_client.get_index_price.return_value = 65_000.0
    mock_client.get_active_options.return_value = []

    out_parquet = tmp_path / "test_ticks.parquet"
    df = fetch_historical_options_ticks(currency="BTC", days=5, output_path=out_parquet, client=mock_client)

    assert out_parquet.exists()
    loaded_df = pd.read_parquet(out_parquet)
    assert len(loaded_df) == len(df)
    assert loaded_df["spot_price"].iloc[0] > 0


def test_fetch_historical_options_sol_linear():
    mock_client = MagicMock()
    mock_client.get_index_price.return_value = 108.0
    mock_client.get_active_options.return_value = [
        {
            "instrument_name": "SOL_USDC-14OCT26-105-C",
            "option_type": "call",
            "is_active": True,
            "strike": 105.0,
            "expiration_timestamp": 1790000000
        }
    ]
    mock_client.get_order_book.return_value = {
        "best_bid_price": 4.5,
        "best_ask_price": 5.0,
        "mark_iv": 62.0
    }

    df = fetch_historical_options_ticks(currency="SOL", days=5, client=mock_client)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    # In linear options, bid_price is 4.5 USD
    assert df["bid_price"].iloc[0] == 4.5
    assert df["ask_price"].iloc[0] == 5.0
    assert df["currency"].iloc[0] == "SOL"


def test_fetch_historical_options_eth_simulation():
    # Tests the multi-day synthetic generator for ETH when API has no ticks
    mock_client = MagicMock()
    mock_client.get_index_price.return_value = 2600.0
    mock_client.get_active_options.return_value = []

    df = fetch_historical_options_ticks(currency="ETH", days=5, client=mock_client)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert df["currency"].iloc[0] == "ETH"
    assert 2000.0 < df["spot_price"].iloc[0] < 3500.0

