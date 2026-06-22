"""Run the pending found-item registration pipeline from the command line.

Usage:
    python scripts/run_registration.py
    python scripts/run_registration.py --skip-embedding
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from registration_service import (  # noqa: E402
    DEFAULT_RAW_INFO_PATH,
    register_pending_raw_found_items,
)


def main() -> None:
    args = _parse_args()
    records_before = _read_records(DEFAULT_RAW_INFO_PATH)
    pending_ids = {
        str(record.get("raw_id"))
        for record in records_before
        if record.get("status", "pending") == "pending"
    }

    if not pending_ids:
        print("No pending raw found-image records to register.")
        return

    print(f"Starting registration for {len(pending_ids)} pending raw image(s)...")
    items = register_pending_raw_found_items(register_embedding=not args.skip_embedding)

    records_after = _read_records(DEFAULT_RAW_INFO_PATH)
    processed = 0
    failed_records: list[dict[str, Any]] = []
    for record in records_after:
        if str(record.get("raw_id")) not in pending_ids:
            continue
        if record.get("status") == "processed":
            processed += 1
        elif record.get("status") == "failed":
            failed_records.append(record)

    print(f"Processed raw images: {processed}")
    print(f"Registered found items: {len(items)}")
    print(f"Failed raw images: {len(failed_records)}")

    for record in failed_records:
        print(f"  {record.get('raw_id', '<unknown>')}: {record.get('error', 'unknown error')}")

    if args.skip_embedding:
        print("Image embedding registration was skipped.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process pending records from data/raw_found_image_info.json."
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Register detected items without generating image embeddings.",
    )
    return parser.parse_args()


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    data = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{file_path} must contain a JSON list.")
    return [record for record in data if isinstance(record, dict)]


if __name__ == "__main__":
    main()
