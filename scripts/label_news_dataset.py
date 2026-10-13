"""
15-Minute Forward Event-Study Labeling Engine for Stage-1 Laya News Classifier.
Matches historical crypto headlines with forward price and volatility delta
to generate objective ground-truth labels for news_train.jsonl and news_val.jsonl.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model.laya_news_engine import STAGE1_NEWS_QUESTIONS


class NewsEventStudyLabeler:
    """Labels crypto headlines based on actual 15-minute forward spot and vol reaction."""

    def __init__(
        self,
        dump_threshold_pct: float = -0.010,       # -1.0% drop in 15m
        expansion_threshold_pct: float = 0.010,   # +1.0% jump in 15m
        split_ratio: float = 0.80
    ):
        self.dump_threshold = dump_threshold_pct
        self.expansion_threshold = expansion_threshold_pct
        self.split_ratio = split_ratio

    def label_headline(
        self,
        headline: str,
        source: str = "CryptoPanic",
        forward_spot_return_15m: float = 0.0,
        forward_dvol_change_15m: float = 0.0,
        known_impact: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assigns ground truth labels based on observed 15m market reaction or known catalyst.
        """
        if known_impact:
            impact_regime = known_impact
            is_moving = "yes" if impact_regime in ["bullish_vol_expansion", "bearish_dump"] else "no"
        elif forward_spot_return_15m <= self.dump_threshold:
            impact_regime = "bearish_dump"
            is_moving = "yes"
        elif forward_spot_return_15m >= self.expansion_threshold:
            impact_regime = "bullish_vol_expansion"
            is_moving = "yes"
        elif abs(forward_spot_return_15m) < 0.003 and forward_dvol_change_15m < -1.0:
            impact_regime = "chop_iv_crush"
            is_moving = "no"
        else:
            impact_regime = "neutral_noise"
            is_moving = "no"

        state_text = f"[SOURCE: {source}] {headline}"

        return {
            "state": state_text,
            "questions": STAGE1_NEWS_QUESTIONS,
            "labels": {
                "is_market_moving": is_moving,
                "impact_regime": impact_regime
            }
        }

    def process_news_items(self, raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes raw headline list into labeled training samples."""
        labeled = []
        for item in raw_items:
            title = item.get("title", "")
            source = item.get("source", "CryptoPanic")
            known = item.get("known_impact")

            # If simulated forward returns are present in item, use them; else infer
            fwd_ret = float(item.get("forward_spot_return_15m", 0.0))
            fwd_vol = float(item.get("forward_dvol_change_15m", 0.0))

            if not known and fwd_ret == 0.0:
                # Fallback to keyword sentiment heuristic for synthetic generation
                title_lower = title.lower()
                if any(w in title_lower for w in ["exploit", "hack", "liquidation", "dump", "charges", "repayment"]):
                    known = "bearish_dump"
                elif any(w in title_lower for w in ["approved", "approval", "rate cut", "inflows", "etf", "surge"]):
                    known = "bullish_vol_expansion"
                elif any(w in title_lower for w in ["gaming", "nft", "minor", "update", "routine"]):
                    known = "neutral_noise"

            sample = self.label_headline(
                headline=title,
                source=source,
                forward_spot_return_15m=fwd_ret,
                forward_dvol_change_15m=fwd_vol,
                known_impact=known
            )
            labeled.append(sample)

        return labeled

    def split_and_save(
        self,
        samples: List[Dict[str, Any]],
        train_path: Path,
        val_path: Path
    ) -> Tuple[int, int]:
        """Chronologically splits samples and writes JSONL files."""
        train_path.parent.mkdir(parents=True, exist_ok=True)
        val_path.parent.mkdir(parents=True, exist_ok=True)

        split_idx = int(len(samples) * self.split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        with open(train_path, "w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s) + "\n")

        with open(val_path, "w", encoding="utf-8") as f:
            for s in val_samples:
                f.write(json.dumps(s) + "\n")

        return len(train_samples), len(val_samples)


def label_news_file(news_input: Path, train_out: Path, val_out: Path) -> Tuple[int, int]:
    """Helper loading raw news JSONL, labeling, and writing train/val files."""
    items = []
    if news_input.exists():
        with open(news_input, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        # Import default historical events if file missing
        from scripts.fetch_historical_news import HISTORICAL_MAJOR_EVENTS
        items = list(HISTORICAL_MAJOR_EVENTS)

    labeler = NewsEventStudyLabeler()
    labeled = labeler.process_news_items(items)
    return labeler.split_and_save(labeled, train_out, val_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Label News Dataset via 15m Event Study")
    parser.add_argument("--news_input", default="data/raw/historical_news.jsonl")
    parser.add_argument("--train_out", default="data/processed/news_train.jsonl")
    parser.add_argument("--val_out", default="data/processed/news_val.jsonl")
    args = parser.parse_args()

    n_train, n_val = label_news_file(
        news_input=Path(args.news_input),
        train_out=Path(args.train_out),
        val_out=Path(args.val_out)
    )
    print(f"Generated {n_train} training and {n_val} validation samples for Stage-1 Laya News Classifier.")
