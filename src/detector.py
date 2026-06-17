"""Object detection and cropping pipeline for found-item photos."""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from contracts import RowItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP_DIR = PROJECT_ROOT / "data" / "cropped_items"
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_MAX_OBJECTS = 10


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

    object_boxes = infer_object_boxes(str(source_path))
    return crop_detected_objects(source_path, object_boxes)


def infer_object_boxes(row_image_path: str) -> list[ObjectBox]:
    """Infer object bounding boxes from a raw image.

    This function is the detector-only part of the pipeline. It returns only
    crop coordinates and confidence values; labels are intentionally ignored
    because the shared RowItem interface does not include object labels.
    """

    source_path = Path(row_image_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw image does not exist: {row_image_path}")

    weights_path = os.environ.get("GREPL_DETECTOR_WEIGHTS")
    config_path = os.environ.get("GREPL_DETECTOR_CONFIG")

    if weights_path:
        try:
            object_boxes = _detect_with_opencv_dnn(
                source_path,
                weights_path=Path(weights_path),
                config_path=Path(config_path) if config_path else None,
                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                max_objects=DEFAULT_MAX_OBJECTS,
            )
            if object_boxes:
                return object_boxes
        except Exception:
            pass

    return _detect_with_foreground_regions(
        source_path,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        max_objects=DEFAULT_MAX_OBJECTS,
    )


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

    from PIL import Image

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


def _detect_with_opencv_dnn(
    image_path: Path,
    *,
    weights_path: Path,
    config_path: Path | None,
    confidence_threshold: float,
    max_objects: int,
) -> list[ObjectBox]:
    import cv2

    if not weights_path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"Detector config not found: {config_path}")

    if config_path is not None and config_path.suffix in {".prototxt", ".txt"}:
        net = cv2.dnn.readNetFromCaffe(str(config_path), str(weights_path))
    elif config_path is not None:
        net = cv2.dnn.readNet(str(weights_path), str(config_path))
    else:
        net = cv2.dnn.readNet(str(weights_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.007843, size=(300, 300), mean=127.5)
    net.setInput(blob)
    raw_detections = net.forward()

    object_boxes: list[ObjectBox] = []
    for raw in raw_detections.reshape(-1, 7):
        confidence = float(raw[2])
        if confidence < confidence_threshold:
            continue

        x_min, y_min, x_max, y_max = raw[3:7]
        object_box = ObjectBox(
            left=int(x_min * width),
            top=int(y_min * height),
            right=int(x_max * width),
            bottom=int(y_max * height),
            confidence=confidence,
        )
        normalized_box = _normalize_box(object_box, width, height)
        if normalized_box is None:
            continue

        object_boxes.append(
            ObjectBox(
                left=normalized_box[0],
                top=normalized_box[1],
                right=normalized_box[2],
                bottom=normalized_box[3],
                confidence=confidence,
            )
        )

    object_boxes.sort(key=lambda object_box: object_box.confidence, reverse=True)
    return object_boxes[:max_objects]


def _detect_with_foreground_regions(
    image_path: Path,
    *,
    confidence_threshold: float,
    max_objects: int,
) -> list[ObjectBox]:
    import numpy as np
    from PIL import Image

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        original_width, original_height = image.size
        scale = min(1.0, 900 / max(original_width, original_height))
        if scale < 1.0:
            image = image.resize((int(original_width * scale), int(original_height * scale)))

    data = np.asarray(image).astype(np.int16)
    height, width = data.shape[:2]
    if height < 8 or width < 8:
        return []

    background = _estimate_border_color(data)
    distance = np.linalg.norm(data - background, axis=2)
    threshold = max(28.0, float(np.percentile(distance, 72)))
    mask = distance > threshold

    min_area = max(80, int(width * height * 0.008))
    components = _connected_components(mask, min_area=min_area)
    object_boxes: list[ObjectBox] = []
    for left, top, right, bottom, area in components:
        object_box = _scale_object_box(
            ObjectBox(
                left=left,
                top=top,
                right=right + 1,
                bottom=bottom + 1,
                confidence=0.0,
            ),
            scale_x=original_width / width,
            scale_y=original_height / height,
        )
        normalized_box = _normalize_box(object_box, original_width, original_height)
        if normalized_box is None:
            continue

        area_ratio = area / (width * height)
        contrast = float(distance[top : bottom + 1, left : right + 1].mean())
        confidence = min(0.92, 0.35 + area_ratio * 2.5 + contrast / 255 * 0.35)
        if confidence < confidence_threshold:
            continue

        object_boxes.append(
            ObjectBox(
                left=normalized_box[0],
                top=normalized_box[1],
                right=normalized_box[2],
                bottom=normalized_box[3],
                confidence=round(confidence, 3),
            )
        )

    object_boxes.sort(key=lambda object_box: _box_area(object_box), reverse=True)
    return object_boxes[:max_objects]


def _connected_components(mask: np.ndarray, *, min_area: int) -> list[tuple[int, int, int, int, int]]:
    import numpy as np

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        left = right = int(start_x)
        top = bottom = int(start_y)
        area = 0

        while queue:
            y, x = queue.popleft()
            area += 1
            left = min(left, x)
            right = max(right, x)
            top = min(top, y)
            bottom = max(bottom, y)

            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                    continue
                if visited[next_y, next_x] or not mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                queue.append((next_y, next_x))

        box_area = (right - left + 1) * (bottom - top + 1)
        if area >= min_area and box_area >= min_area:
            components.append((left, top, right, bottom, area))

    return components


def _estimate_border_color(data: np.ndarray) -> np.ndarray:
    import numpy as np

    top = data[0, :, :]
    bottom = data[-1, :, :]
    left = data[:, 0, :]
    right = data[:, -1, :]
    border_pixels = np.concatenate((top, bottom, left, right), axis=0)
    return np.median(border_pixels, axis=0)


def _scale_object_box(object_box: ObjectBox, *, scale_x: float, scale_y: float) -> ObjectBox:
    return ObjectBox(
        left=max(0, int(round(object_box.left * scale_x))),
        top=max(0, int(round(object_box.top * scale_y))),
        right=max(0, int(round(object_box.right * scale_x))),
        bottom=max(0, int(round(object_box.bottom * scale_y))),
        confidence=object_box.confidence,
    )


def _normalize_box(object_box: ObjectBox, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    """Clamp a detected box to image boundaries and reject empty crops."""

    left = max(0, min(int(object_box.left), image_width))
    top = max(0, min(int(object_box.top), image_height))
    right = max(0, min(int(object_box.right), image_width))
    bottom = max(0, min(int(object_box.bottom), image_height))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _box_area(object_box: ObjectBox) -> int:
    return max(0, object_box.right - object_box.left) * max(0, object_box.bottom - object_box.top)
