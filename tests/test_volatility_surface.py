"""
Unit tests for src/quant/volatility_surface.py.
Verifies IV Rank, IV Percentile, SABR smile formula, and volatility regime classification.
"""

from src.quant.volatility_surface import (
    calculate_iv_rank,
    calculate_iv_percentile,
    sabr_implied_volatility,
    analyze_volatility_regime
)


def test_iv_rank_and_percentile():
    # 90-day history with min 0.35, max 0.85
    history = [0.35 + i * (0.50 / 99) for i in range(100)]
    current_iv = 0.45

    rank = calculate_iv_rank(current_iv, history)
    # (0.45 - 0.35) / (0.85 - 0.35) * 100 = 0.10 / 0.50 * 100 = 20%
    assert round(rank, 1) == 20.0

    pct = calculate_iv_percentile(current_iv, history)
    assert 18.0 <= pct <= 22.0


def test_sabr_volatility_smile():
    F = 65000.0
    T = 14.0 / 365.0
    alpha = 0.50  # 50% ATM vol
    beta = 1.0
    rho = -0.15  # Negative skew (OTM puts higher than OTM calls)
    nu = 0.45

    # 1. ATM strike
    atm_vol = sabr_implied_volatility(F, K=65000.0, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
    assert atm_vol > 0.0
    assert 0.45 < atm_vol < 0.55

    # 2. OTM Call strike
    otm_call_vol = sabr_implied_volatility(F, K=70000.0, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
    assert otm_call_vol > 0.0

    # 3. OTM Put strike
    otm_put_vol = sabr_implied_volatility(F, K=60000.0, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
    assert otm_put_vol > 0.0


def test_volatility_regime_classification():
    history = [0.35 + i * 0.005 for i in range(100)]  # Min 0.35, Max 0.845
    
    # Cheap vol setup: low current IV (0.40 -> rank ~10%), low RV (0.39)
    cheap_metrics = analyze_volatility_regime(
        current_iv=0.40, realized_vol_30d=0.39, historical_iv_series=history
    )
    assert cheap_metrics.regime == "cheap_vol_expansion"
    assert cheap_metrics.iv_rank < 30.0

    # Expensive vol setup: high current IV (0.80 -> rank ~90%)
    expensive_metrics = analyze_volatility_regime(
        current_iv=0.80, realized_vol_30d=0.50, historical_iv_series=history
    )
    assert expensive_metrics.regime == "expensive_crush"
