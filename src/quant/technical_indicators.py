"""
Parametric Technical Indicators Engine from OHLCV Candlesticks.
Calculates Trend, Momentum, Squeeze, Volatility, and Key Price Levels (Support/Resistance/VWAP)
for high-density serialization into Laya System 1 decision prompts.
"""

from typing import Optional, Dict, Any, Union, List
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd


class TechnicalIndicatorSummary(BaseModel):
    """Normalized parametric representation of candlestick technical conditions."""
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    current_price: float
    trend_regime: str = Field(description="'BULLISH_UPTREND', 'BEARISH_DOWNTREND', 'SIDEWAYS_CONSOLIDATION'")
    ema_20: float
    ema_50: float
    sma_200: Optional[float] = None
    rsi_14: float
    rsi_regime: str = Field(description="'OVERSOLD', 'BULLISH_MOMENTUM', 'NEUTRAL', 'OVERBOUGHT'")
    macd_line: float
    macd_signal: float
    macd_hist: float
    macd_bias: str = Field(description="'BULLISH_EXPANSION', 'BEARISH_EXPANSION', 'NEUTRAL'")
    bb_upper: float
    bb_lower: float
    bb_width_pct: float
    is_bb_squeeze: bool
    support_level: float
    resistance_level: float
    vwap: Optional[float] = None

    def to_serialized_text(self) -> str:
        """Formats indicators into a compact, high-density block for LLM prompts."""
        sma_str = f" > 200 SMA ${self.sma_200:,.0f}" if self.sma_200 else ""
        vwap_str = ""
        if self.vwap and self.current_price > 0:
            vwap_diff = ((self.current_price - self.vwap) / self.vwap) * 100.0
            vwap_str = f" | VWAP: ${self.vwap:,.1f} ({vwap_diff:+.2f}%)"

        squeeze_str = "ACTIVE (Volatility Compression)" if self.is_bb_squeeze else "Normal Expansion"

        lines = [
            f"CHART_TECHNICAL_INDICATORS ({self.interval.upper()} Candlesticks):",
            f"- Trend: {self.trend_regime} (Spot ${self.current_price:,.1f} vs 20 EMA ${self.ema_20:,.1f}{sma_str})",
            f"- Momentum: RSI(14) = {self.rsi_14:.1f} ({self.rsi_regime}) | MACD Hist: {self.macd_hist:+,.1f} ({self.macd_bias})",
            f"- Volatility & Squeeze: BB Width: {self.bb_width_pct:.1f}% ({squeeze_str}) | Upper: ${self.bb_upper:,.1f} | Lower: ${self.bb_lower:,.1f}",
            f"- Key Structure: Support: ${self.support_level:,.1f} | Resistance: ${self.resistance_level:,.1f}{vwap_str}"
        ]
        return "\n".join(lines)


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """Calculates Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index using Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Where avg_loss is 0, if avg_gain > 0, RSI is 100.0 (maximum overbought)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi = np.where(avg_loss == 0, np.where(avg_gain > 0, 100.0, 50.0), 100.0 - (100.0 / (1.0 + rs)))

    return pd.Series(rsi, index=series.index).fillna(50.0)


def calculate_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Dict[str, pd.Series]:
    """Calculates MACD Line, Signal Line, and Histogram."""
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "hist": macd_hist
    }


def calculate_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> Dict[str, pd.Series]:
    """Calculates Bollinger Upper, Lower, and Bandwidth Percentage."""
    rolling_mean = series.rolling(window=period, min_periods=1).mean()
    rolling_std = series.rolling(window=period, min_periods=1).std().fillna(0.0)

    upper = rolling_mean + (rolling_std * num_std)
    lower = rolling_mean - (rolling_std * num_std)
    bandwidth = ((upper - lower) / rolling_mean.replace(0, np.nan)) * 100.0
    return {
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth.fillna(0.0)
    }


def calculate_vwap(df: pd.DataFrame) -> Optional[float]:
    """Calculates Volume Weighted Average Price from OHLCV data."""
    if "volume" not in df.columns or df["volume"].sum() <= 0:
        return None
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    total_vp = (typical_price * df["volume"]).sum()
    total_volume = df["volume"].sum()
    return float(total_vp / total_volume) if total_volume > 0 else None


def compute_technical_indicators(
    df: pd.DataFrame,
    symbol: str = "BTCUSDT",
    interval: str = "1h"
) -> TechnicalIndicatorSummary:
    """
    Computes comprehensive parametric technical indicators from an OHLCV DataFrame.
    DataFrame must contain columns: ['close', 'high', 'low', 'volume'].
    """
    if df.empty or "close" not in df.columns:
        # Fallback neutral summary
        return TechnicalIndicatorSummary(
            symbol=symbol,
            interval=interval,
            current_price=65000.0,
            trend_regime="SIDEWAYS_CONSOLIDATION",
            ema_20=65000.0,
            ema_50=65000.0,
            rsi_14=50.0,
            rsi_regime="NEUTRAL",
            macd_line=0.0,
            macd_signal=0.0,
            macd_hist=0.0,
            macd_bias="NEUTRAL",
            bb_upper=66000.0,
            bb_lower=64000.0,
            bb_width_pct=3.0,
            is_bb_squeeze=False,
            support_level=64000.0,
            resistance_level=66000.0
        )

    closes = df["close"].astype(float)
    highs = df["high"].astype(float) if "high" in df.columns else closes
    lows = df["low"].astype(float) if "low" in df.columns else closes

    current_price = float(closes.iloc[-1])

    # 1. Moving Averages & Trend
    ema20 = calculate_ema(closes, 20)
    ema50 = calculate_ema(closes, 50)
    last_ema20 = float(ema20.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])

    last_sma200 = None
    if len(closes) >= 200:
        last_sma200 = float(closes.rolling(200).mean().iloc[-1])

    if current_price > last_ema20 >= last_ema50:
        trend_regime = "BULLISH_UPTREND"
    elif current_price < last_ema20 <= last_ema50:
        trend_regime = "BEARISH_DOWNTREND"
    else:
        trend_regime = "SIDEWAYS_CONSOLIDATION"

    # 2. RSI & Momentum
    rsi_series = calculate_rsi(closes, period=14)
    last_rsi = float(rsi_series.iloc[-1])

    if last_rsi >= 70.0:
        rsi_regime = "OVERBOUGHT"
    elif last_rsi >= 55.0:
        rsi_regime = "BULLISH_MOMENTUM"
    elif last_rsi <= 30.0:
        rsi_regime = "OVERSOLD"
    else:
        rsi_regime = "NEUTRAL"

    # 3. MACD
    macd_res = calculate_macd(closes)
    last_macd = float(macd_res["macd"].iloc[-1])
    last_signal = float(macd_res["signal"].iloc[-1])
    last_hist = float(macd_res["hist"].iloc[-1])

    if last_hist > 0 and last_macd > last_signal:
        macd_bias = "BULLISH_EXPANSION"
    elif last_hist < 0 and last_macd < last_signal:
        macd_bias = "BEARISH_EXPANSION"
    else:
        macd_bias = "NEUTRAL"

    # 4. Bollinger Bands
    bb_res = calculate_bollinger_bands(closes, period=20, num_std=2.0)
    last_upper = float(bb_res["upper"].iloc[-1])
    last_lower = float(bb_res["lower"].iloc[-1])
    last_bandwidth = float(bb_res["bandwidth"].iloc[-1])
    # A bandwidth < 3.5% indicates severe volatility compression (squeeze)
    is_squeeze = last_bandwidth < 3.5

    # 5. Support / Resistance (Rolling window)
    window_lookback = min(len(df), 30)
    support = float(lows.iloc[-window_lookback:].min())
    resistance = float(highs.iloc[-window_lookback:].max())

    # 6. VWAP
    vwap = calculate_vwap(df)

    return TechnicalIndicatorSummary(
        symbol=symbol,
        interval=interval,
        current_price=round(current_price, 2),
        trend_regime=trend_regime,
        ema_20=round(last_ema20, 2),
        ema_50=round(last_ema50, 2),
        sma_200=round(last_sma200, 2) if last_sma200 else None,
        rsi_14=round(last_rsi, 1),
        rsi_regime=rsi_regime,
        macd_line=round(last_macd, 2),
        macd_signal=round(last_signal, 2),
        macd_hist=round(last_hist, 2),
        macd_bias=macd_bias,
        bb_upper=round(last_upper, 2),
        bb_lower=round(last_lower, 2),
        bb_width_pct=round(last_bandwidth, 2),
        is_bb_squeeze=is_squeeze,
        support_level=round(support, 2),
        resistance_level=round(resistance, 2),
        vwap=round(vwap, 2) if vwap else None
    )
