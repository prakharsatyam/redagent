"""
Unit tests for scripts/sync_colab.py.
"""

from pathlib import Path
import json
from scripts.sync_colab import ColabSyncBridge


def test_colab_sync_upload_and_download(tmp_path: Path):
    drive_dir = tmp_path / "mock_drive"
    local_data = tmp_path / "local_data"
    local_models = tmp_path / "local_models"

    local_data.mkdir()
    local_models.mkdir()

    # Create dummy dataset files
    sample_train = local_data / "train.jsonl"
    sample_train.write_text('{"state": "sample", "labels": {}}\n')

    bridge = ColabSyncBridge(drive_base_dir=drive_dir)
    bridge.manifest_file = tmp_path / "manifest.json"

    # 1. Test Upload
    uploaded = bridge.upload_datasets(source_dir=local_data, target_subfolder="data")
    assert "train.jsonl" in uploaded
    assert (drive_dir / "data" / "train.jsonl").exists()

    # 2. Simulate model training in Drive
    mock_model_dir = drive_dir / "models" / "laya_options_agent"
    mock_model_dir.mkdir(parents=True)
    (mock_model_dir / "config.json").write_text('{"model_type": "laya"}\n')

    # 3. Test Download
    downloaded = bridge.download_checkpoints(destination_dir=local_models, source_subfolder="models")
    assert "laya_options_agent" in downloaded
    assert (local_models / "laya_options_agent" / "config.json").exists()

    # 4. Verify manifest recording
    assert bridge.manifest_file.exists()
    with open(bridge.manifest_file, "r") as f:
        m = json.load(f)
        assert "last_upload" in m
        assert "last_download" in m


def test_compute_sha256(tmp_path: Path):
    test_file = tmp_path / "hash_test.txt"
    test_file.write_text("RedAgent Laya Engine\n")

    bridge = ColabSyncBridge(drive_base_dir=tmp_path)
    hash_val = bridge.compute_sha256(test_file)
    assert isinstance(hash_val, str)
    assert len(hash_val) == 64
