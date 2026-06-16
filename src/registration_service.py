"""Found-item registration workflow.

This module coordinates the database-facing registration pipeline:
load raw found-image records, detect individual items and crop the corresponding image, create
searchable found-item records, and register image embeddings.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from contracts import LostItem, RawFoundItem, RegisterItem, RowItem, TimePoint
from detector import detect_objects


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_INFO_PATH = PROJECT_ROOT / "data" / "raw_found_image_info.json"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "generated" / "found_items.json"
DEFAULT_RUN_LOG_PATH = PROJECT_ROOT / "data" / "generated" / "registration_runs.json"

DetectorFunction = Callable[[str], list[RowItem]]


def register_pending_raw_found_items(
    *,
    raw_info_path: str | Path = DEFAULT_RAW_INFO_PATH,
    detector_fn: DetectorFunction = detect_objects,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    run_log_path: str | Path = DEFAULT_RUN_LOG_PATH,
    register_embedding: bool = True,
) -> list[LostItem]:
    """Register every raw image record whose JSON status is ``pending``."""

    raw_info = Path(raw_info_path)
    records = _read_json_list(raw_info)
    all_items: list[LostItem] = []

    for record in records:
        if record.get("status", "pending") != "pending":
            continue

        raw_id = str(record.get("raw_id", ""))
        started_at = _utc_now()
        try:
            raw_item = _raw_item_from_record(record)
            items = register_raw_found_item(
                raw_item,
                detector_fn=detector_fn,
                registry_path=registry_path,
                register_embedding=register_embedding,
            )
            record["status"] = "processed"
            record["processed_at"] = _utc_now()
            _append_run_log(
                Path(run_log_path),
                {
                    "raw_id": raw_item.raw_id,
                    "status": "processed",
                    "item_count": len(items),
                    "started_at": started_at,
                    "finished_at": record["processed_at"],
                    "error": None,
                },
            )
            all_items.extend(items)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            record["processed_at"] = _utc_now()
            _append_run_log(
                Path(run_log_path),
                {
                    "raw_id": raw_id,
                    "status": "failed",
                    "item_count": 0,
                    "started_at": started_at,
                    "finished_at": record["processed_at"],
                    "error": str(error),
                },
            )

    _write_json(raw_info, records)
    return all_items


def register_raw_found_item(
    raw_item: RawFoundItem,
    *,
    category: str | None = None,
    detector_fn: DetectorFunction = detect_objects,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    register_embedding: bool = True,
) -> list[LostItem]:
    """Register all detected items from one RawFoundItem batch.

    The detector returns cropped item images. Each crop becomes one LostItem
    record, and each image is optionally passed to embedding_engine.py through
    register_item_image(RegisterItem).
    """

    row_items = detector_fn(raw_item.image_path)
    items: list[LostItem] = []

    for row_item in row_items:
        item = LostItem(
            item_id=_new_item_id(),
            image_path=row_item.image_path,
            found_time=raw_item.found_time,
            found_location=raw_item.found_location,
            bound_confidence=row_item.bound_confidence,
            raw_id=raw_item.raw_id,
            category=category,
        )
        embedding_registered = _register_embedding(item, enabled=register_embedding)
        _append_local_record(
            item=item,
            row_item=row_item,
            embedding_registered=embedding_registered,
            registry_path=Path(registry_path),
            raw_image_path=raw_item.image_path,
        )
        items.append(item)

    return items


def load_registered_items(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[LostItem]:
    """Load locally registered found-item records.

    This is a lightweight stand-in until a real database module owns storage.
    """

    path = Path(registry_path)
    if not path.is_file():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))
    return [_lost_item_from_record(record) for record in records]


def _register_embedding(item: LostItem, *, enabled: bool) -> bool:
    if not enabled:
        return False

    embedding_engine = _optional_module("embedding_engine")
    if embedding_engine is None or not hasattr(embedding_engine, "register_item_image"):
        return False

    # To call embedding to process the image and get vector
    return bool(
        embedding_engine.register_item_image(
            RegisterItem(item_id=item.item_id, image_path=item.image_path)
        )
    )


def _append_local_record(
    *,
    item: LostItem,
    row_item: RowItem,
    embedding_registered: bool,
    registry_path: Path,
    raw_image_path: str,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    records = _read_registry_records(registry_path)
    records.append(
        {
            **_lost_item_to_record(item),
            "raw_image_path": raw_image_path,
            "embedding_registered": embedding_registered,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    registry_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_registry_records(registry_path: Path) -> list[dict[str, object]]:
    if not registry_path.is_file():
        return []

    records = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Registry file must contain a JSON list: {registry_path}")
    return records


# Conversion and JSON helpers below keep workflow metadata in files while
# converting only core contract fields into RawFoundItem and LostItem objects.
def _lost_item_to_record(item: LostItem) -> dict[str, object]:
    record = asdict(item)
    record["found_time"] = asdict(item.found_time) if item.found_time else None
    return record


def _lost_item_from_record(record: dict[str, object]) -> LostItem:
    found_time_data = record.get("found_time")
    found_time = (
        TimePoint(**found_time_data)
        if isinstance(found_time_data, dict)
        else None
    )
    return LostItem(
        item_id=str(record["item_id"]),
        image_path=str(record["image_path"]),
        found_time=found_time,
        found_location=(
            str(record["found_location"])
            if record.get("found_location") is not None
            else None
        ),
        bound_confidence=_float_or_default(record.get("bound_confidence"), 0.0),
        raw_id=str(record["raw_id"]) if record.get("raw_id") is not None else None,
        category=str(record["category"]) if record.get("category") is not None else None,
    )


def _raw_item_from_record(record: dict[str, object]) -> RawFoundItem:
    found_time_data = record.get("found_time")
    found_time = (
        TimePoint(**found_time_data)
        if isinstance(found_time_data, dict)
        else None
    )
    return RawFoundItem(
        raw_id=str(record["raw_id"]),
        image_path=str(record["image_path"]),
        found_time=found_time,
        found_location=(
            str(record["found_location"])
            if record.get("found_location") is not None
            else None
        ),
    )


def _append_run_log(run_log_path: Path, entry: dict[str, object]) -> None:
    records = _read_json_list(run_log_path)
    records.append(entry)
    _write_json(run_log_path, records)


def _read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"JSON database file must contain a list: {path}")
    return records


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return None
        raise


def _new_item_id() -> str:
    return f"item_{uuid4().hex[:12]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
