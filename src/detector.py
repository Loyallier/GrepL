"""Object detection and cropping pipeline for found-item photos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from contracts import RowItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP_DIR = PROJECT_ROOT / "data" / "cropped_items"


@dataclass(frozen=True)
class ObjectBox:
    """A detected object bounding box in pixel coordinates.

    Coordinates follow the Pillow crop convention: left, top, right, bottom.
    The right and bottom values are exclusive crop boundaries.
    """

    left: int
    top: int
    right: int
    bottom: int
    confidence: float


def detect_objects(row_image_path: str) -> list[RowItem]:
    """Detect objects from one raw image, crop them, and return cropped items."""

    source_path = Path(row_image_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw image does not exist: {row_image_path}")

    # Replace this call with the final model inference function if the teammate
    # uses a different name or return format for bounding-box prediction.
    object_boxes = infer_object_boxes(str(source_path))
    return crop_detected_objects(source_path, object_boxes)


def infer_object_boxes(row_image_path: str) -> list[ObjectBox]:
    """Infer object bounding boxes from a raw image.

    This is a temporary interface for the object-detection teammate to replace.
    The final implementation should return one ObjectBox per detected item.
    """

    # TODO: Replace this placeholder with the real model loading and inference
    # code. Keep the returned coordinates in left/top/right/bottom pixel order,
    # or update crop_detected_objects if the model returns another box format.
    return []

                                
def crop_detected_objects(
    row_image_path: str | Path,
    object_boxes: Iterable[ObjectBox],
) -> list[RowItem]:
    """Crop sub-images using detected object coordinates.

    Args:
        row_image_path: Raw found-item image path.
        object_boxes: Bounding boxes produced by the detection step.
        Cropped sub-images are saved under data/cropped_items/.

    Returns:
        RowItem objects containing each cropped image path and detection
        confidence.
    """

    source_path = Path(row_image_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw image does not exist: {source_path}")

    crop_dir = DEFAULT_CROP_DIR
    crop_dir.mkdir(parents=True, exist_ok=True)

    cropped_items: list[RowItem] = []
    with Image.open(source_path) as image:
        width, height = image.size
        for index, object_box in enumerate(object_boxes, start=1):
            normalized_box = _normalize_box(object_box, width, height)
            if normalized_box is None:
                continue

            crop_path = crop_dir / f"{source_path.stem}_object_{index:03d}.png"
            image.crop(normalized_box).save(crop_path)
            cropped_items.append(
                RowItem(
                    image_path=str(crop_path),
                    bound_confidence=float(object_box.confidence),
                )
            )

    return cropped_items


def _normalize_box(object_box: ObjectBox, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    """Clamp a detected box to image boundaries and reject empty crops."""

    left = max(0, min(int(object_box.left), image_width))
    top = max(0, min(int(object_box.top), image_height))
    right = max(0, min(int(object_box.right), image_width))
    bottom = max(0, min(int(object_box.bottom), image_height))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


