"""
Crypto Volatility Surface Analytics, Skew, and SABR Smile Engine.
Computes IV Rank, Volatility Risk Premium (VRP), Call Skew,
and calibrates SABR volatility smile parameters.
"""

import math
from typing import List, Dict, Optional
import numpy as np
from pydantic import BaseModel, Field


class VolatilityMetrics(BaseModel):
    """Volatility surface summary metrics for an underlying asset."""
    current_iv: float = Field(description="Current annualized Implied Volatility")
    realized_vol_30d: float = Field(description="30-day historical realized volatility")
    vrp: float = Field(description="Volatility Risk Premium (IV - RV)")
    iv_rank: float = Field(description="90-day IV Rank (0.0 to 100.0%)")
    iv_percentile: float = Field(description="90-day IV Percentile (0.0 to 100.0%)")
    call_skew_25d: float = Field(description="25-Delta Call Skew (IV_25d_call - IV_25d_put)")
    regime: str = Field(description="Regime label: cheap_vol_expansion, neutral, or expensive_crush")


def calculate_iv_rank(current_iv: float, historical_iv_series: List[float]) -> float:
    """
    Computes IV Rank:
    IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
    """
    if not historical_iv_series:
        return 50.0

    min_iv = min(historical_iv_series)
    max_iv = max(historical_iv_series)

    if max_iv <= min_iv:
        return 50.0

    rank = (current_iv - min_iv) / (max_iv - min_iv) * 100.0
    return max(0.0, min(100.0, rank))


def calculate_iv_percentile(current_iv: float, historical_iv_series: List[float]) -> float:
    """
    Computes IV Percentile:
    Percentage of historical days where IV was lower than current IV.
    """
    if not historical_iv_series:
        return 50.0

    count_lower = sum(1 for iv in historical_iv_series if iv < current_iv)
    percentile = (count_lower / len(historical_iv_series)) * 100.0
    return max(0.0, min(100.0, percentile))


def sabr_implied_volatility(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float = 1.0,
    rho: float = -0.20,
    nu: float = 0.40
) -> float:
    """
    Computes Hagan et al. (2002) SABR model implied volatility smile:
    F: Forward price (approx spot for short DTE)
    K: Strike price
    T: Time to expiration in years
    alpha: Initial volatility level
    beta: Elasticity (1.0 for log-normal crypto options)
    rho: Correlation between asset price and volatility movements
    nu: Volatility of volatility (vol-vol)
    """
    if T <= 0.0 or F <= 0.0 or K <= 0.0:
        return alpha

    # ATM case: K == F
    if abs(F - K) < 1e-6:
        term1 = alpha / (F ** (1.0 - beta))
        term2 = 1.0 + (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / (F ** (2.0 - 2.0 * beta)))
            + 0.25 * (rho * beta * nu * alpha / (F ** (1.0 - beta)))
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
        ) * T
        return term1 * term2

    # OTM / ITM case
    log_FK = math.log(F / K)
    FK_pow = (F * K) ** ((1.0 - beta) / 2.0)

    z = (nu / alpha) * FK_pow * log_FK
    sqrt_1_2rz_z2 = math.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = math.log((sqrt_1_2rz_z2 + z - rho) / (1.0 - rho))

    denom1 = FK_pow * (
        1.0
        + ((1.0 - beta) ** 2 / 24.0) * (log_FK ** 2)
        + ((1.0 - beta) ** 4 / 1920.0) * (log_FK ** 4)
    )

    factor1 = alpha / denom1
    factor2 = z / x_z if abs(x_z) > 1e-8 else 1.0

    factor3 = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / ((F * K) ** (1.0 - beta)))
        + 0.25 * (rho * beta * nu * alpha / FK_pow)
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
    ) * T

    return factor1 * factor2 * factor3


def analyze_volatility_regime(
    current_iv: float,
    realized_vol_30d: float,
    historical_iv_series: List[float],
    call_skew_25d: float = 0.02
) -> VolatilityMetrics:
    """
    Classifies the asset's current volatility surface into actionable trading regimes.
    """
    iv_rank = calculate_iv_rank(current_iv, historical_iv_series)
    iv_pct = calculate_iv_percentile(current_iv, historical_iv_series)
    vrp = current_iv - realized_vol_30d

    # Regime categorization:
    # Cheap vol: Low IV rank (<30%) and negative/low VRP -> Ideal for buying Call options!
    if iv_rank < 30.0 and vrp <= 0.03:
        regime = "cheap_vol_expansion"
    elif iv_rank > 70.0 or vrp > 0.15:
        regime = "expensive_crush"
    else:
        regime = "neutral"

    return VolatilityMetrics(
        current_iv=current_iv,
        realized_vol_30d=realized_vol_30d,
        vrp=vrp,
        iv_rank=iv_rank,
        iv_percentile=iv_pct,
        call_skew_25d=call_skew_25d,
        regime=regime,
    )
