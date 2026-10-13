"""
Black-Scholes-Merton Analytical Options Pricing and Greeks Engine.
Computes European Call option price, implied volatility (IV),
and analytical first- and second-order Greeks (Delta, Gamma, Theta, Vega, Rho).
"""

import math
from typing import Optional
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from pydantic import BaseModel, Field


class GreeksResult(BaseModel):
    """Complete analytical pricing and Greek sensitivities for a Call option."""
    price: float = Field(description="Theoretical option price")
    iv: float = Field(description="Implied volatility annualized (e.g. 0.50 = 50%)")
    delta: float = Field(description="Delta: sensitivity to underlying price movement")
    gamma: float = Field(description="Gamma: rate of change of delta per $1 move in underlying")
    theta_daily: float = Field(description="Theta: time decay in USD lost per calendar day")
    vega_1pct: float = Field(description="Vega: USD change in option price per 1% change in IV")
    rho_1pct: float = Field(description="Rho: USD change in option price per 1% change in interest rate")


def bsm_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Standard Black-Scholes-Merton European Call pricing formula.
    S: Spot price of underlying
    K: Strike price
    T: Time to expiration in years (e.g. 7 days = 7/365)
    r: Risk-free interest rate (e.g. 0.04 = 4%)
    sigma: Volatility (annualized standard deviation, e.g. 0.55 = 55%)
    q: Continuous dividend / convenience yield (0 for crypto spot)
    """
    if T <= 0.0:
        return max(0.0, S - K)
    if sigma <= 0.0:
        return max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return max(0.0, price)


def calculate_implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float = 0.04,
    q: float = 0.0,
    min_vol: float = 0.01,
    max_vol: float = 5.0
) -> Optional[float]:
    """
    Solves for Implied Volatility (IV) given market price using Brent's root-finding method.
    Returns annualized volatility (e.g., 0.52 for 52%) or None if intrinsic bound violated.
    """
    if T <= 0.0 or market_price <= 0.0:
        return None

    intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T))
    if market_price < intrinsic:
        return None

    def objective(sig: float) -> float:
        return bsm_call_price(S, K, T, r, sig, q) - market_price

    try:
        iv = brentq(objective, min_vol, max_vol, xtol=1e-5, maxiter=100)
        return float(iv)
    except (ValueError, RuntimeError):
        return None


def calculate_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> GreeksResult:
    """
    Calculates analytical Greeks for a European Call option.
    """
    if T <= 0.0 or sigma <= 0.0:
        intrinsic = max(0.0, S - K)
        delta = 1.0 if S > K else 0.0
        return GreeksResult(
            price=intrinsic,
            iv=sigma,
            delta=delta,
            gamma=0.0,
            theta_daily=0.0,
            vega_1pct=0.0,
            rho_1pct=0.0,
        )

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    exp_qT = math.exp(-q * T)
    exp_rT = math.exp(-r * T)

    # Theoretical Call Price
    price = S * exp_qT * cdf_d1 - K * exp_rT * cdf_d2

    # Delta: dC / dS
    delta = exp_qT * cdf_d1

    # Gamma: d2C / dS2
    gamma = (exp_qT * pdf_d1) / (S * sigma * sqrt_T)

    # Theta: dC / dt (annualized -> divide by 365 for daily USD burn)
    theta_term1 = -(S * sigma * exp_qT * pdf_d1) / (2.0 * sqrt_T)
    theta_term2 = -r * K * exp_rT * cdf_d2
    theta_term3 = q * S * exp_qT * cdf_d1
    theta_annual = theta_term1 + theta_term2 + theta_term3
    theta_daily = theta_annual / 365.0

    # Vega: dC / dSigma (per 1% = 0.01 change in vol)
    vega_annual = S * exp_qT * sqrt_T * pdf_d1
    vega_1pct = vega_annual * 0.01

    # Rho: dC / dr (per 1% = 0.01 change in interest rate)
    rho_annual = K * T * exp_rT * cdf_d2
    rho_1pct = rho_annual * 0.01

    return GreeksResult(
        price=price,
        iv=sigma,
        delta=delta,
        gamma=gamma,
        theta_daily=theta_daily,
        vega_1pct=vega_1pct,
        rho_1pct=rho_1pct,
    )
