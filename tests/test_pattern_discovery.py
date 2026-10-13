"""
Unit tests for PatternDiscoveryEngine.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from src.quant.pattern_discovery import PatternDiscoveryEngine, DiscoveredRuleSet


def create_synthetic_data(n_samples: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    iv_rank = np.random.uniform(10.0, 90.0, n_samples)
    delta = np.random.uniform(0.15, 0.85, n_samples)
    spread = np.random.uniform(0.01, 0.10, n_samples)
    funding = np.random.uniform(-0.0005, 0.0010, n_samples)
    vrp = np.random.uniform(-0.05, 0.15, n_samples)

    # Win condition: low IV rank, tight spread, positive delta
    won = (iv_rank < 40.0) & (spread < 0.04) & (delta > 0.35)
    won_trade = won.astype(int)

    return pd.DataFrame({
        "iv_rank": iv_rank,
        "delta": delta,
        "spread_ratio": spread,
        "funding_rate_8h": funding,
        "vrp": vrp,
        "won_trade": won_trade
    })


def test_fit_and_rule_discovery():
    df = create_synthetic_data(150)
    engine = PatternDiscoveryEngine(max_depth=3, min_samples_leaf=5)
    rule_set = engine.fit_from_dataframe(df, target_col="won_trade")

    assert isinstance(rule_set, DiscoveredRuleSet)
    assert len(rule_set.feature_importances) > 0
    # iv_rank or spread_ratio should be top features
    top_feature = list(rule_set.feature_importances.keys())[0]
    assert top_feature in ["iv_rank", "spread_ratio", "delta"]
    assert len(rule_set.rules) > 0
    assert rule_set.accuracy > 0.70


def test_save_and_load_rules(tmp_path: Path):
    df = create_synthetic_data(80)
    engine = PatternDiscoveryEngine()
    rule_set = engine.fit_from_dataframe(df, target_col="won_trade")

    output_file = tmp_path / "discovered_rules.json"
    engine.save_rules(rule_set, output_file)
    assert output_file.exists()

    loaded = engine.load_rules(output_file)
    assert loaded.accuracy == rule_set.accuracy
    assert loaded.feature_importances == rule_set.feature_importances


def test_contract_scoring():
    df = create_synthetic_data(80)
    engine = PatternDiscoveryEngine()
    rule_set = engine.fit_from_dataframe(df, target_col="won_trade")

    # High quality candidate: cheap IV rank, tight spread, liquid delta
    good_candidate = {
        "iv_rank": 20.0,
        "spread_ratio": 0.02,
        "delta": 0.50,
        "vrp": 0.01,
        "funding_rate_8h": 0.0002
    }
    score_good = engine.score_contract(good_candidate, rule_set)

    # Poor quality candidate: high IV rank, wide spread
    bad_candidate = {
        "iv_rank": 85.0,
        "spread_ratio": 0.08,
        "delta": 0.10,
        "vrp": 0.12,
        "funding_rate_8h": -0.0005
    }
    score_bad = engine.score_contract(bad_candidate, rule_set)

    assert score_good > score_bad
