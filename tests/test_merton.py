"""
Unit tests for src/quant/merton_jump.py.
Verifies Merton Jump-Diffusion series convergence and Corrado-Su fat-tail skew adjustments.
"""

from src.quant.merton_jump import merton_jump_call_price, corrado_su_call_price, evaluate_merton_regime
from src.quant.greeks import bsm_call_price


def test_merton_jump_price():
    S = 65000.0
    K = 70000.0  # Far OTM call
    T = 14.0 / 365.0
    r = 0.04
    sigma = 0.45

    # 1. Pure BSM price for far OTM call with 45% vol
    bsm_price = bsm_call_price(S, K, T, r, sigma)

    # 2. Merton Jump price with positive jump potential (jumps of +2% mean with 8% std)
    merton_price = merton_jump_call_price(S, K, T, r, sigma, jump_lambda=3.0, jump_mean=0.03, jump_std=0.08)

    # In crypto, jump potential significantly enhances OTM Call value over pure Gaussian BSM
    assert merton_price > bsm_price
    assert merton_price > 0.0


def test_corrado_su_adjustment():
    S = 65000.0
    K = 68000.0
    T = 10.0 / 365.0
    r = 0.04
    sigma = 0.50

    bs_price = bsm_call_price(S, K, T, r, sigma)
    # Crypto has positive skewness and fat tails (excess kurtosis > 0)
    adjusted_price = corrado_su_call_price(S, K, T, r, sigma, skewness=0.40, kurtosis=4.5)

    assert adjusted_price > 0.0
    # Positive skewness adds value to right tail (Call options)
    assert adjusted_price >= bs_price


def test_evaluate_merton_regime():
    S = 65000.0
    K = 67000.0
    T = 7.0 / 365.0
    r = 0.04
    sigma = 0.50

    res = evaluate_merton_regime(S, K, T, r, sigma)
    assert res.merton_price > 0.0
    assert res.bsm_baseline_price > 0.0
    assert res.jump_intensity_lambda == 2.5
    assert res.jump_premium_usd == (res.merton_price - res.bsm_baseline_price)
