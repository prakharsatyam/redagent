"""
AI Mathematical Formula and Feature Pattern Discovery Engine.
Extracts non-linear rule combinations and symbolic decision boundaries
from historical options order book ticks and trade payoff distributions.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

from config.settings import settings


class DiscoveredRuleSet:
    """Container holding extracted mathematical rules and feature importances."""
    def __init__(self, rules: List[str], feature_importances: Dict[str, float], accuracy: float):
        self.rules = rules
        self.feature_importances = feature_importances
        self.accuracy = accuracy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "feature_importances": self.feature_importances,
            "rules": self.rules
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveredRuleSet":
        return cls(
            rules=data.get("rules", []),
            feature_importances=data.get("feature_importances", {}),
            accuracy=data.get("accuracy", 0.0)
        )


class PatternDiscoveryEngine:
    """Discovers interpretable mathematical formulas and indicators for Call option setups."""

    def __init__(self, max_depth: int = 4, min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "won_trade"
    ) -> DiscoveredRuleSet:
        """
        Trains tree models and extracts decision rules that predict >= 3% net profit trades.
        """
        if feature_cols is None:
            # Standard quantitative candidate features
            default_cols = [
                "spot_momentum_1h", "iv_annual", "iv_rank", "vrp",
                "delta", "gamma", "theta_daily", "spread_ratio", "funding_rate_8h"
            ]
            feature_cols = [c for c in default_cols if c in df.columns]

        if not feature_cols:
            raise ValueError("No matching feature columns found in dataframe.")

        X = df[feature_cols].fillna(0.0)
        y = df[target_col].astype(int)

        # 1. Feature importances via Random Forest
        rf = RandomForestClassifier(n_estimators=50, max_depth=self.max_depth, random_state=42)
        rf.fit(X, y)
        importances = {
            col: round(float(imp), 4)
            for col, imp in zip(feature_cols, rf.feature_importances_)
        }
        sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

        # 2. Extract interpretable symbolic decision rules via shallow Decision Tree
        dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        dt.fit(X, y)
        acc = float(dt.score(X, y))

        tree_text = export_text(dt, feature_names=feature_cols)
        rules = self._parse_tree_to_rules(tree_text)

        return DiscoveredRuleSet(
            rules=rules,
            feature_importances=sorted_importances,
            accuracy=round(acc, 4)
        )

    def _parse_tree_to_rules(self, tree_text: str) -> List[str]:
        """Cleans and extracts winning decision paths from tree text."""
        lines = [line.strip() for line in tree_text.splitlines() if line.strip()]
        rules = []
        for line in lines:
            if "class: 1" in line or "<=" in line or ">" in line:
                rules.append(line)
        return rules[:10]  # Retain top 10 most salient rule branches

    def save_rules(self, rule_set: DiscoveredRuleSet, output_path: Path) -> None:
        """Saves discovered rules to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(rule_set.to_dict(), f, indent=2)

    def load_rules(self, input_path: Path) -> DiscoveredRuleSet:
        """Loads rules from JSON."""
        with open(input_path, "r") as f:
            data = json.load(f)
        return DiscoveredRuleSet.from_dict(data)

    def score_contract(self, features: Dict[str, float], rule_set: DiscoveredRuleSet) -> float:
        """
        Evaluates a contract's features against discovered high-importance features.
        Returns conformance score between 0.0 and 1.0.
        """
        if not rule_set.feature_importances:
            return 0.5

        score = 0.0
        total_weight = 0.0

        for feat, weight in rule_set.feature_importances.items():
            if feat not in features:
                continue
            val = features[feat]
            total_weight += weight

            # Heuristic bounds on top discovered features:
            if feat == "iv_rank" and val <= 35.0:
                score += weight  # Cheap IV
            elif feat == "spread_ratio" and val <= 0.03:
                score += weight  # Tight spread
            elif feat == "delta" and 0.30 <= val <= 0.65:
                score += weight  # Liquid ATM/NTM
            elif feat == "vrp" and val <= 0.04:
                score += weight  # Low Vol Risk Premium
            elif feat == "funding_rate_8h" and val >= 0.0:
                score += weight  # Spot momentum tailwind

        if total_weight <= 0.0:
            return 0.5
        return round(score / total_weight, 3)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pattern Discovery Engine")
    parser.add_argument("--input", default="data/raw/deribit_btc.parquet", help="Path to input tick data")
    parser.add_argument("--output", default="models/discovered_rules.json", help="Path to output rules JSON")
    args = parser.parse_args()

    engine = PatternDiscoveryEngine()
    print("Pattern Discovery Engine CLI initialized.")
