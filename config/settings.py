"""
Centralized Configuration and Settings for RedAgent.
Loads credentials from keys.env and defines operational trading parameters.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base Directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / "keys.env"

# Load environment variables if keys.env exists
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)


class TradingSettings(BaseModel):
    """Runtime parameters for the RedAgent decision & scanning loop."""
    base_currency: str = Field(default="BTC", description="Base asset symbol")
    deribit_use_testnet: bool = Field(default=True, description="Flag for testnet vs mainnet")
    scan_interval_seconds: int = Field(default=60, description="Scanning loop interval in seconds")
    min_conviction_threshold: float = Field(default=0.75, description="Minimum Stage-2 Laya confidence (0.0 - 1.0)")
    min_net_return_hurdle: float = Field(default=0.03, description="Minimum net profit constraint (3% = 0.03)")
    max_bid_ask_spread_ratio: float = Field(default=0.05, description="Max acceptable spread/mid ratio (5%)")
    max_dte_days: float = Field(default=14.0, description="Max days to expiration for Call options")
    min_dte_days: float = Field(default=0.5, description="Min days to expiration to avoid zero-day theta burn")


class ApiCredentials(BaseModel):
    """API Keys and Credentials loaded securely from environment."""
    deribit_client_id: str = Field(default_factory=lambda: os.getenv("DERIBIT_CLIENT_ID", ""))
    deribit_client_secret: str = Field(default_factory=lambda: os.getenv("DERIBIT_CLIENT_SECRET", ""))
    binance_api_key: str = Field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = Field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    telegram_bot_token: str = Field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = Field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    cryptopanic_api_key: str = Field(default_factory=lambda: os.getenv("CRYPTOPANIC_API_KEY", ""))


class Settings(BaseModel):
    """Master Application Settings container."""
    trading: TradingSettings = Field(default_factory=TradingSettings)
    api: ApiCredentials = Field(default_factory=ApiCredentials)
    root_dir: Path = Field(default=ROOT_DIR)
    models_dir: Path = Field(default=ROOT_DIR / "models")
    data_dir: Path = Field(default=ROOT_DIR / "data")


# Singleton settings instance
settings = Settings()
