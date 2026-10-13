"""
Unit tests for config/settings.py.
Verifies default parameters, directory resolution, and environment loading.
"""

from pathlib import Path
from config.settings import settings, Settings, TradingSettings


def test_settings_initialization():
    assert isinstance(settings, Settings)
    assert isinstance(settings.trading, TradingSettings)
    assert settings.trading.base_currency == "BTC"
    assert settings.trading.min_conviction_threshold == 0.75
    assert settings.trading.min_net_return_hurdle == 0.03
    assert settings.trading.max_bid_ask_spread_ratio == 0.05
    assert settings.trading.deribit_use_testnet is True


def test_settings_directories():
    assert settings.root_dir.is_dir()
    assert settings.models_dir == settings.root_dir / "models"
    assert settings.data_dir == settings.root_dir / "data"
