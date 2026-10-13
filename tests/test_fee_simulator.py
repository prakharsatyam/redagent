"""
Unit tests for src/quant/fee_simulator.py.
Verifies the mandatory >= 3% net profit hurdle, liquidity filters, and friction modeling.
"""

from src.quant.fee_simulator import evaluate_option_candidate


def test_liquid_profitable_call_candidate():
    spot = 65000.0
    bid = 715.0
    ask = 735.0  # Spread = 20 / 725 = 2.76% (below 5% max spread)
    delta = 0.35
    theta_daily = -135.0

    eval_res = evaluate_option_candidate(
        contract_symbol="BTC-24OCT26-68000-C",
        spot_price=spot,
        bid_price=bid,
        ask_price=ask,
        delta=delta,
        theta_daily=theta_daily,
        target_spot_move_pct=0.025,  # 2.5% spot move
        holding_hours=24.0,
        min_net_hurdle_pct=0.03
    )

    # 2.5% on 65k = 1625 USD move * 0.35 delta = 568.75 USD gross gain
    # Net gain after fees and 135 USD theta should be well above 3% (> 22 USD)
    assert eval_res.passes_hurdle is True
    assert eval_res.net_return_pct >= 0.03
    assert eval_res.net_profit_usd > 100.0
    assert eval_res.rejection_reason is None


def test_illiquid_candidate_rejection():
    spot = 65000.0
    bid = 600.0
    ask = 750.0  # Spread = 150 / 675 = 22.2% (vastly exceeds 5% max spread)
    delta = 0.30
    theta_daily = -100.0

    eval_res = evaluate_option_candidate(
        contract_symbol="BTC-24OCT26-72000-C",
        spot_price=spot,
        bid_price=bid,
        ask_price=ask,
        delta=delta,
        theta_daily=theta_daily,
        max_spread_ratio=0.05
    )

    assert eval_res.passes_hurdle is False
    assert "Illiquid" in eval_res.rejection_reason


def test_insufficient_net_profit_rejection():
    spot = 65000.0
    bid = 720.0
    ask = 730.0
    delta = 0.05  # Very low delta (far OTM call)
    theta_daily = -140.0  # High theta decay

    # Small spot move of 0.5% with 0.05 delta produces only 16.25 USD gross gain
    # But 140 USD theta and ~40 USD fees will result in a net loss
    eval_res = evaluate_option_candidate(
        contract_symbol="BTC-24OCT26-80000-C",
        spot_price=spot,
        bid_price=bid,
        ask_price=ask,
        delta=delta,
        theta_daily=theta_daily,
        target_spot_move_pct=0.005,
        holding_hours=24.0,
        min_net_hurdle_pct=0.03
    )

    assert eval_res.passes_hurdle is False
    assert eval_res.net_return_pct < 0.03
    assert "Insufficient Net Profit" in eval_res.rejection_reason
