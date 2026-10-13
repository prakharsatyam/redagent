"""
Unit tests for scripts/label_news_dataset.py.
"""

from pathlib import Path
import json
from scripts.label_news_dataset import NewsEventStudyLabeler, label_news_file


def test_label_headline_bearish_dump():
    labeler = NewsEventStudyLabeler()
    res = labeler.label_headline(
        headline="Major Exchange Faces Insolvency Rumors",
        source="Twitter",
        forward_spot_return_15m=-0.025  # -2.5% dump
    )

    assert res["labels"]["is_market_moving"] == "yes"
    assert res["labels"]["impact_regime"] == "bearish_dump"
    assert "Twitter" in res["state"]


def test_label_headline_bullish_expansion():
    labeler = NewsEventStudyLabeler()
    res = labeler.label_headline(
        headline="SEC Approves Spot Ethereum Options Trading",
        source="SEC",
        forward_spot_return_15m=0.018  # +1.8% expansion
    )

    assert res["labels"]["is_market_moving"] == "yes"
    assert res["labels"]["impact_regime"] == "bullish_vol_expansion"


def test_split_and_save(tmp_path: Path):
    labeler = NewsEventStudyLabeler(split_ratio=0.75)
    raw_items = [
        {"title": f"Test crypto headline {i}", "forward_spot_return_15m": 0.015 if i % 2 == 0 else -0.015}
        for i in range(20)
    ]

    labeled = labeler.process_news_items(raw_items)
    train_path = tmp_path / "news_train.jsonl"
    val_path = tmp_path / "news_val.jsonl"

    n_train, n_val = labeler.split_and_save(labeled, train_path, val_path)
    assert n_train == 15
    assert n_val == 5
    assert train_path.exists()
    assert val_path.exists()

    with open(train_path, "r", encoding="utf-8") as f:
        line = json.loads(f.readline())
        assert "state" in line
        assert "questions" in line
        assert "labels" in line
