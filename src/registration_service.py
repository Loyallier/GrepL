""" 
Manage the workflow that turns raw photos found item into searchable local 
records in the database.

This module loads pending raw image records, crops detected items, stores
searchable records of found items, and optionally registers image embeddings.
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


# Absolute project root used to resolve local data files.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# JSON file that tracks records of raw found images and processing status.
DEFAULT_RAW_INFO_PATH = PROJECT_ROOT / "data" / "raw_found_image_info.json"

# JSON registry that stores searchable records of found items generated after the raw images are cropped.
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "generated" / "found_items.json"

# Runtime contract for object detector functions used druing registration.
DetectorFunction = Callable[[str], list[RowItem]]


def register_pending_raw_found_items(
    *,
    raw_info_path: str | Path = DEFAULT_RAW_INFO_PATH,
    detector_fn: DetectorFunction = detect_objects,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    register_embedding: bool = True,
) -> list[LostItem]:
    """ Register every raw image record whose JSON status is pending. """

    raw_info = Path(raw_info_path)
    records = _read_json_list(raw_info)
    all_items: list[LostItem] = []

    for record in records:
        if record.get("status", "pending") != "pending":
            continue

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
            record["item_count"] = len(items)
            record.pop("error", None)
            all_items.extend(items)
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            record["processed_at"] = _utc_now()
            record["item_count"] = 0

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
    """
    Register all detected items from one raw found image which can contain lots
    of items .

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
        )
        items.append(item)

    return items


def load_registered_items(
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> list[LostItem]:
    """ Load all locally registered records of found items. """

    path = Path(registry_path)
    if not path.is_file():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))
    return [_lost_item_from_record(record) for record in records]


def _register_embedding(item: LostItem, *, enabled: bool) -> bool:
    """ Register an item image embedding when embedding support is enabled. """

    if not enabled:
        return False

    embedding_engine = _optional_module("embedding_engine")
    if embedding_engine is None or not hasattr(embedding_engine, "register_item_image"):
        return False

    # Delegate vector creation to embedding_engine.py.
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
) -> None:
    """ Append one registered item record to the local JSON registry. """

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    records = _read_registry_records(registry_path)
    records.append(
        {
            **_lost_item_to_record(item),
            "embedding_registered": embedding_registered,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    registry_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_registry_records(registry_path: Path) -> list[dict[str, object]]:
    """Read the found-item registry and validate its list shape."""

    if not registry_path.is_file():
        return []

    records = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Registry file must contain a JSON list: {registry_path}")
    return records


# Conversion and JSON helpers below keep workflow metadata in files while
# converting only core contract fields into RawFoundItem and LostItem objects.
def _lost_item_to_record(item: LostItem) -> dict[str, object]:
    """ Convert a LostItem dataclass into a JSON-serializable record. """

    record = asdict(item)
    record["found_time"] = asdict(item.found_time) if item.found_time else None
    return record


def _lost_item_from_record(record: dict[str, object]) -> LostItem:
    """ Convert a stored registry record into a LostItem object. """

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
    """ Convert a raw-image JSON record into a RawFoundItem object. """

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


def _read_json_list(path: Path) -> list[dict[str, object]]:
    """ Read a JSON list file and return only dictionary records. """

    if not path.is_file():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"JSON database file must contain a list: {path}")
    return records


def _write_json(path: Path, data: object) -> None:
    """ Write JSON data with UTF-8 encoding and stable formatting. """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _float_or_default(value: object, default: float) -> float:
    """ Convert a value to float, falling back when conversion fails. """

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_module(module_name: str):
    """ Import an optional top-level module without hiding nested import errors. """

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return None
        raise


def _new_item_id() -> str:
    """ Create a short unique identifier for one cropped found item. """

    return f"item_{uuid4().hex[:12]}"


def _utc_now() -> str:
    """ Return the current UTC timestamp. """

    return datetime.now(timezone.utc).isoformat()
