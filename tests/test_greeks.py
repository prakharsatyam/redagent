"""
Unit tests for src/quant/greeks.py.
Verifies Black-Scholes call pricing, IV root-solver, and Greeks properties.
"""

import math
from src.quant.greeks import bsm_call_price, calculate_implied_volatility, calculate_greeks


def test_bsm_call_price():
    S = 65000.0
    K = 65000.0  # ATM
    T = 7.0 / 365.0  # 7 days
    r = 0.04
    sigma = 0.50

    price = bsm_call_price(S, K, T, r, sigma)
    assert price > 0.0
    # For BTC at 65k with 50% IV for 7 days, ATM call is roughly 1200-1500 USD
    assert 1000.0 < price < 2000.0


def test_implied_volatility_solver():
    S = 65000.0
    K = 68000.0
    T = 14.0 / 365.0
    r = 0.04
    true_sigma = 0.55

    # Generate synthetic theoretical price
    market_price = bsm_call_price(S, K, T, r, true_sigma)

    # Invert to recover IV
    recovered_iv = calculate_implied_volatility(market_price, S, K, T, r)
    assert recovered_iv is not None
    assert math.isclose(recovered_iv, true_sigma, rel_tol=1e-4)


def test_greeks_properties():
    S = 65000.0
    K = 68000.0  # OTM call
    T = 10.0 / 365.0
    r = 0.04
    sigma = 0.50

    greeks = calculate_greeks(S, K, T, r, sigma)

    # 1. Delta for OTM Call must be strictly between 0 and 1
    assert 0.0 < greeks.delta < 1.0
    assert greeks.delta < 0.50  # OTM call has delta < 0.50

    # 2. Gamma must be strictly positive
    assert greeks.gamma > 0.0

    # 3. Theta for long call must be negative (time decay burns premium)
    assert greeks.theta_daily < 0.0

    # 4. Vega must be strictly positive (rising volatility increases call value)
    assert greeks.vega_1pct > 0.0

    # 5. Rho must be positive for call options
    assert greeks.rho_1pct > 0.0
