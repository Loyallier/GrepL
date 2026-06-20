"""Import raw found-item images into the local JSON database.

Usage:
    python scripts/add_raw_found_images.py path/to/source_images
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.options import LOCATION_OPTIONS, option_label  # noqa: E402


RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw_found_images"
RAW_INFO_PATH = PROJECT_ROOT / "data" / "raw_found_image_info.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    args = _parse_args()
    source_dir = Path(args.source_dir).expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    images = _find_images(source_dir)
    if not images:
        raise SystemExit(f"No supported image files found in: {source_dir}")

    records = _read_records(RAW_INFO_PATH)
    RAW_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    imported = 0
    for index, source_path in enumerate(images, start=1):
        print(f"\n[{index}/{len(images)}] {source_path.name}")
        found_time = _prompt_time()
        found_location = _prompt_location()
        raw_id = _new_raw_id(records)
        target_path = RAW_IMAGE_DIR / f"{raw_id}{source_path.suffix.lower()}"

        shutil.copy2(source_path, target_path)
        records.append(
            {
                "raw_id": raw_id,
                "image_path": _project_relative(target_path),
                "found_time": found_time,
                "found_location": found_location,
                "status": "pending",
            }
        )
        imported += 1
        print(f"Imported {source_path.name} -> {target_path.name}")

    _write_records(RAW_INFO_PATH, records)
    print(f"\nImported {imported} raw image(s).")
    print(f"Updated {RAW_INFO_PATH.relative_to(PROJECT_ROOT)}.")
    print("Next step: run registration_service.register_pending_raw_found_items().")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy raw found-item images into data/raw_found_images and append JSON records."
    )
    parser.add_argument("source_dir", help="Directory containing raw images to import.")
    return parser.parse_args()


def _find_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _prompt_time() -> dict[str, int | str | None]:
    found_date = _prompt_date()
    found_hour = _prompt_hour()
    return {"date": found_date, "hour": found_hour}


def _prompt_date() -> str | None:
    while True:
        value = input("Found date (YYYY-MM-DD, empty if unknown): ").strip()
        if not value:
            return None
        try:
            date.fromisoformat(value)
        except ValueError:
            print("Invalid date. Example: 2026-06-21")
            continue
        return value


def _prompt_hour() -> int | None:
    while True:
        value = input("Found hour (0-23, empty if unknown): ").strip()
        if not value:
            return None
        try:
            hour = int(value)
        except ValueError:
            print("Invalid hour. Enter an integer from 0 to 23.")
            continue
        if 0 <= hour <= 23:
            return hour
        print("Invalid hour. Enter an integer from 0 to 23.")


def _prompt_location() -> str:
    _print_locations()
    while True:
        value = input("Found location key: ").strip()
        if value in LOCATION_OPTIONS and value != "any":
            return value
        print("Invalid location key. Use one of the keys shown above.")


def _print_locations() -> None:
    print("Available location keys:")
    for key in LOCATION_OPTIONS:
        if key == "any":
            continue
        label = option_label(key, LOCATION_OPTIONS) or key
        print(f"  {key}: {label}")


def _new_raw_id(records: list[dict[str, Any]]) -> str:
    prefix = f"raw_{date.today().strftime('%Y%m%d')}"
    used_ids = {str(record.get("raw_id")) for record in records}
    used_names = {path.stem for path in RAW_IMAGE_DIR.glob("*") if path.is_file()}
    counter = 1
    while True:
        raw_id = f"{prefix}_{counter:03d}"
        if raw_id not in used_ids and raw_id not in used_names:
            return raw_id
        counter += 1


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return [record for record in data if isinstance(record, dict)]


def _write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


if __name__ == "__main__":
    main()
