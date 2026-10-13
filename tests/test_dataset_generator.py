"""
Unit tests for Stage-2 DatasetGenerator.
"""

from pathlib import Path
import json
from src.model.dataset_generator import DatasetGenerator


def test_dataset_generator_synthetic_processing():
    gen = DatasetGenerator(min_net_hurdle=0.03, train_split_ratio=0.80)
    df = gen.generate_synthetic_historical_df(n_samples=50)

    assert len(df) == 50
    assert "spot_price" in df.columns
    assert "forward_spot_move_pct" in df.columns

    records = gen.process_records_from_df(df)
    assert len(records) >= 50  # Hard negative mining may add bull trap duplicates

    # Validate schema of each record
    first = records[0]
    assert "state" in first
    assert "questions" in first
    assert "labels" in first
    assert first["labels"]["should_buy_call"] in ["yes", "no"]
    assert first["labels"]["expected_net_return"] in [
        "loss_or_negative", "breakeven_to_2pct", "target_3pct_to_10pct", "explosive_above_10pct"
    ]
    assert first["labels"]["risk_regime"] in [
        "high_theta_decay", "iv_crush_risk", "favorable_low_vol_expansion", "trending_spot_momentum"
    ]


def test_walk_forward_split_and_save(tmp_path: Path):
    gen = DatasetGenerator(min_net_hurdle=0.03, train_split_ratio=0.75)
    df = gen.generate_synthetic_historical_df(n_samples=40)
    records = gen.process_records_from_df(df)

    train_rec, val_rec = gen.split_walk_forward(records)
    assert len(train_rec) > 0
    assert len(val_rec) > 0
    assert len(train_rec) + len(val_rec) == len(records)

    train_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "val.jsonl"

    gen.save_jsonl(train_rec, train_file)
    gen.save_jsonl(val_rec, val_file)

    assert train_file.exists()
    assert val_file.exists()

    # Verify JSON lines can be parsed
    with open(train_file, "r") as f:
        line = f.readline()
        parsed = json.loads(line)
        assert "state" in parsed
        assert "labels" in parsed
