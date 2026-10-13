"""
Merton Jump-Diffusion and Corrado-Su Skewness/Kurtosis Pricing Engine.
Accommodates crypto fat tails, discontinuous Poisson jumps, and volatility skew.
"""

import math
from typing import Optional
from scipy.stats import norm
from pydantic import BaseModel, Field
from src.quant.greeks import bsm_call_price


class MertonResult(BaseModel):
    """Merton Jump-Diffusion pricing and jump sensitivity metrics."""
    merton_price: float = Field(description="Call price under Merton Jump-Diffusion")
    bsm_baseline_price: float = Field(description="Call price under standard Black-Scholes")
    jump_premium_usd: float = Field(description="USD difference between Merton and Black-Scholes")
    jump_intensity_lambda: float = Field(description="Annualized jump arrival rate")
    jump_mean: float = Field(description="Mean jump log-return")
    jump_std: float = Field(description="Standard deviation of jump magnitude")


def merton_jump_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    jump_lambda: float = 2.5,
    jump_mean: float = 0.02,
    jump_std: float = 0.08,
    max_jumps: int = 40
) -> float:
    """
    Prices European Call option under Merton's (1976) Jump-Diffusion model.
    S: Spot price
    K: Strike price
    T: Time to expiry in years
    r: Continuous risk-free rate
    sigma: Continuous diffusion volatility
    jump_lambda: Jump arrival rate (jumps per year, e.g. 2.5)
    jump_mean: Mean of log-jump size gamma
    jump_std: Standard deviation of log-jump size delta
    max_jumps: Series truncation threshold
    """
    if T <= 0.0:
        return max(0.0, S - K)

    # Expected relative jump size: kappa = E[Y - 1] = exp(gamma + delta^2/2) - 1
    kappa = math.exp(jump_mean + 0.5 * jump_std * jump_std) - 1.0
    lambda_prime = jump_lambda * (1.0 + kappa)

    total_price = 0.0
    # Poisson probability factor: P(n) = exp(-lambda_prime * T) * (lambda_prime * T)^n / n!
    poisson_factor = math.exp(-lambda_prime * T)

    for n in range(max_jumps):
        if n > 0:
            poisson_factor *= (lambda_prime * T) / n

        sigma_n = math.sqrt(sigma * sigma + (n * jump_std * jump_std) / T)
        r_n = r - jump_lambda * kappa + (n * math.log(1.0 + kappa)) / T

        bs_price_n = bsm_call_price(S, K, T, r_n, sigma_n)
        term = poisson_factor * bs_price_n
        total_price += term

        # Early break if added value is below floating-point precision
        if term < 1e-9 and n > 5:
            break

    return total_price


def corrado_su_call_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    skewness: float = 0.35,
    kurtosis: float = 4.2
) -> float:
    """
    Prices European Call option using Corrado-Su (1996) Gram-Charlier expansion.
    Adjusts standard Black-Scholes for skewness (gamma1) and excess kurtosis (gamma2 - 3).
    """
    if T <= 0.0 or sigma <= 0.0:
        return max(0.0, S - K)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)

    bs_price = bsm_call_price(S, K, T, r, sigma)
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)

    # Skewness adjustment term Q3
    q3 = (1.0 / 6.0) * S * sigma * sqrt_T * (
        (2.0 * sigma * sqrt_T - d1) * n_d1 + sigma * sigma * T * N_d1
    )

    # Kurtosis adjustment term Q4
    d1_sq = d1 * d1
    q4 = (1.0 / 24.0) * S * sigma * sqrt_T * (
        (d1_sq - 1.0 - 3.0 * sigma * sqrt_T * d1 + 3.0 * sigma * sigma * T) * n_d1
    )

    excess_kurtosis = kurtosis - 3.0
    adjusted_price = bs_price + skewness * q3 + excess_kurtosis * q4
    return max(0.0, adjusted_price)


def evaluate_merton_regime(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    jump_lambda: float = 2.5,
    jump_mean: float = 0.02,
    jump_std: float = 0.08
) -> MertonResult:
    """
    Evaluates option price under Merton Jump-Diffusion vs BSM baseline
    to quantify the jump risk premium priced into out-of-the-money calls.
    """
    merton_price = merton_jump_call_price(
        S, K, T, r, sigma, jump_lambda=jump_lambda, jump_mean=jump_mean, jump_std=jump_std
    )
    bsm_price = bsm_call_price(S, K, T, r, sigma)

    return MertonResult(
        merton_price=merton_price,
        bsm_baseline_price=bsm_price,
        jump_premium_usd=merton_price - bsm_price,
        jump_intensity_lambda=jump_lambda,
        jump_mean=jump_mean,
        jump_std=jump_std,
    )
