"""
Two-Way Google Drive & Google Colab Synchronization Utility.
Bridges training datasets from local Mac workspace to Google Drive for Colab GPU training,
and downloads fine-tuned model checkpoints back to local models/ directory for sub-25ms inference.
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import settings


class ColabSyncBridge:
    """Manages file transfers and manifests between local Mac and Google Drive/Colab."""

    def __init__(self, drive_base_dir: Optional[Path] = None):
        # Default Google Drive mount path on macOS or local fallback
        default_mac_drive = Path.home() / "Google Drive" / "My Drive" / "redagent"
        self.drive_dir = drive_base_dir or default_mac_drive
        self.manifest_file = settings.root_dir / "data" / "sync_manifest.json"

    def compute_sha256(self, file_path: Path) -> str:
        """Computes SHA-256 hash of a file for integrity verification."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def upload_datasets(self, source_dir: Path, target_subfolder: str = "data") -> List[str]:
        """
        Uploads processed datasets (train.jsonl, val.jsonl, etc.) to Google Drive.
        """
        target_dir = self.drive_dir / target_subfolder
        target_dir.mkdir(parents=True, exist_ok=True)
        uploaded = []

        if not source_dir.exists():
            return uploaded

        for file_path in source_dir.glob("*.jsonl"):
            dest_file = target_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            uploaded.append(file_path.name)

        self._record_manifest("upload", uploaded, str(target_dir))
        return uploaded

    def download_checkpoints(self, destination_dir: Path, source_subfolder: str = "models") -> List[str]:
        """
        Downloads trained Laya checkpoints from Google Drive to local models/ directory.
        """
        source_dir = self.drive_dir / source_subfolder
        destination_dir.mkdir(parents=True, exist_ok=True)
        downloaded = []

        if not source_dir.exists():
            return downloaded

        for item in source_dir.iterdir():
            dest_item = destination_dir / item.name
            if item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)
                downloaded.append(item.name)
            elif item.is_file():
                shutil.copy2(item, dest_item)
                downloaded.append(item.name)

        self._record_manifest("download", downloaded, str(destination_dir))
        return downloaded

    def _record_manifest(self, action: str, file_list: List[str], target_path: str) -> None:
        """Updates sync manifest JSON file."""
        self.manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_data = {}
        if self.manifest_file.exists():
            try:
                with open(self.manifest_file, "r") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = {}

        manifest_data[f"last_{action}"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "target": target_path,
            "files": file_list
        }

        with open(self.manifest_file, "w") as f:
            json.dump(manifest_data, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Colab / Google Drive Sync Bridge")
    parser.add_argument("--upload", action="store_true", help="Upload datasets to Google Drive")
    parser.add_argument("--download", action="store_true", help="Download trained models from Google Drive")
    parser.add_argument("--drive_dir", default=None, help="Custom Google Drive root path")
    parser.add_argument("--source", default="data/processed", help="Local source directory for upload")
    parser.add_argument("--destination", default="models", help="Local destination directory for download")
    args = parser.parse_args()

    custom_drive = Path(args.drive_dir) if args.drive_dir else None
    bridge = ColabSyncBridge(drive_base_dir=custom_drive)

    if args.upload:
        files = bridge.upload_datasets(Path(args.source))
        print(f"Uploaded {len(files)} files to Google Drive: {files}")
    elif args.download:
        files = bridge.download_checkpoints(Path(args.destination))
        print(f"Downloaded {len(files)} model checkpoints from Google Drive: {files}")
    else:
        print("Please specify either --upload or --download.")
