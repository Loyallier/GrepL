"""Object detection and bounding-box output for found-item photos.

The module exposes two levels of API:

- detect_object_boxes(): locate objects and return boxes in original pixels.
- detect_objects(): locate objects, crop them, save crops, and return RowItem
  objects for the registration pipeline.

If OpenCV DNN model files are configured, the detector uses that model. Without
model files, it falls back to a lightweight foreground-region detector so the
rest of the project can still be integrated and demonstrated.
"""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from contracts import BoundingBox, DetectedObject, RowItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP_DIR = PROJECT_ROOT / "data" / "crops"
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_MAX_OBJECTS = 10

_VOC_LABELS = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


def detect_object_boxes(
    raw_image_path: str,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    model_weights: str | None = None,
    model_config: str | None = None,
    labels: tuple[str, ...] = _VOC_LABELS,
    allow_fallback: bool = True,
) -> list[DetectedObject]:
    """Detect objects and return bounding boxes in original image coordinates."""

    image_path = _validate_image_path(raw_image_path)
    weights_path = model_weights or os.environ.get("GREPL_DETECTOR_WEIGHTS")
    config_path = model_config or os.environ.get("GREPL_DETECTOR_CONFIG")

    if weights_path:
        try:
            detections = _detect_with_opencv_dnn(
                image_path,
                weights_path=Path(weights_path),
                config_path=Path(config_path) if config_path else None,
                labels=labels,
                confidence_threshold=confidence_threshold,
                max_objects=max_objects,
            )
            if detections:
                return detections
        except Exception:
            if not allow_fallback:
                raise

    return _detect_with_foreground_regions(
        image_path,
        confidence_threshold=confidence_threshold,
        max_objects=max_objects,
    )


def detect_objects(
    raw_image_path: str,
    *,
    output_dir: str | Path = DEFAULT_CROP_DIR,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    padding: int = 8,
    model_weights: str | None = None,
    model_config: str | None = None,
) -> list[RowItem]:
    """Detect objects, crop each bounding box, and return cropped RowItem data."""

    detections = detect_object_boxes(
        raw_image_path,
        confidence_threshold=confidence_threshold,
        max_objects=max_objects,
        model_weights=model_weights,
        model_config=model_config,
    )
    return crop_detected_objects(raw_image_path, detections, output_dir=output_dir, padding=padding)


def crop_detected_objects(
    raw_image_path: str,
    detections: list[DetectedObject],
    *,
    output_dir: str | Path = DEFAULT_CROP_DIR,
    padding: int = 8,
) -> list[RowItem]:
    """Save one crop per detection and return RowItem objects."""

    image_path = _validate_image_path(raw_image_path)
    crop_dir = Path(output_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    source_stem = image_path.stem

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        row_items: list[RowItem] = []
        for index, detection in enumerate(detections, start=1):
            box = _pad_box(detection.bbox, width=width, height=height, padding=padding)
            crop = image.crop((box.x_min, box.y_min, box.x_max, box.y_max))
            crop_path = crop_dir / f"{source_stem}_object_{index:02d}.jpg"
            crop.save(crop_path, quality=92)
            row_items.append(
                RowItem(
                    image_path=str(crop_path),
                    bound_confidence=detection.confidence,
                    bbox=detection.bbox,
                    label=detection.label,
                )
            )
    return row_items


def _detect_with_opencv_dnn(
    image_path: Path,
    *,
    weights_path: Path,
    config_path: Path | None,
    labels: tuple[str, ...],
    confidence_threshold: float,
    max_objects: int,
) -> list[DetectedObject]:
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

    detections: list[DetectedObject] = []
    for raw in raw_detections.reshape(-1, 7):
        confidence = float(raw[2])
        if confidence < confidence_threshold:
            continue

        class_id = int(raw[1])
        label = labels[class_id] if 0 <= class_id < len(labels) else f"class_{class_id}"
        x_min, y_min, x_max, y_max = raw[3:7]
        box = _clip_box(
            BoundingBox(
                x_min=int(x_min * width),
                y_min=int(y_min * height),
                x_max=int(x_max * width),
                y_max=int(y_max * height),
            ),
            width=width,
            height=height,
        )
        if _box_area(box) > 0:
            detections.append(DetectedObject(label=label, confidence=confidence, bbox=box))

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    return detections[:max_objects]


def _detect_with_foreground_regions(
    image_path: Path,
    *,
    confidence_threshold: float,
    max_objects: int,
) -> list[DetectedObject]:
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
    detections: list[DetectedObject] = []
    for x_min, y_min, x_max, y_max, area in components:
        box = _scale_box(
            BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max + 1, y_max=y_max + 1),
            scale_x=original_width / width,
            scale_y=original_height / height,
        )
        area_ratio = area / (width * height)
        contrast = float(distance[y_min : y_max + 1, x_min : x_max + 1].mean())
        confidence = min(0.92, 0.35 + area_ratio * 2.5 + contrast / 255 * 0.35)
        if confidence >= confidence_threshold:
            detections.append(DetectedObject(label="object", confidence=round(confidence, 3), bbox=box))

    detections.sort(key=lambda detection: (_box_area(detection.bbox), detection.confidence), reverse=True)
    return detections[:max_objects]


def _connected_components(mask: np.ndarray, *, min_area: int) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int, int, int]] = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        x_min = x_max = int(start_x)
        y_min = y_max = int(start_y)
        area = 0

        while queue:
            y, x = queue.popleft()
            area += 1
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if next_y < 0 or next_y >= height or next_x < 0 or next_x >= width:
                    continue
                if visited[next_y, next_x] or not mask[next_y, next_x]:
                    continue
                visited[next_y, next_x] = True
                queue.append((next_y, next_x))

        box_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        if area >= min_area and box_area >= min_area:
            components.append((x_min, y_min, x_max, y_max, area))

    return components


def _estimate_border_color(data: np.ndarray) -> np.ndarray:
    top = data[0, :, :]
    bottom = data[-1, :, :]
    left = data[:, 0, :]
    right = data[:, -1, :]
    border_pixels = np.concatenate((top, bottom, left, right), axis=0)
    return np.median(border_pixels, axis=0)


def _validate_image_path(raw_image_path: str) -> Path:
    image_path = Path(raw_image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {raw_image_path}")
    return image_path


def _clip_box(box: BoundingBox, *, width: int, height: int) -> BoundingBox:
    return BoundingBox(
        x_min=max(0, min(width - 1, box.x_min)),
        y_min=max(0, min(height - 1, box.y_min)),
        x_max=max(0, min(width, box.x_max)),
        y_max=max(0, min(height, box.y_max)),
    )


def _pad_box(box: BoundingBox, *, width: int, height: int, padding: int) -> BoundingBox:
    return _clip_box(
        BoundingBox(
            x_min=box.x_min - padding,
            y_min=box.y_min - padding,
            x_max=box.x_max + padding,
            y_max=box.y_max + padding,
        ),
        width=width,
        height=height,
    )


def _scale_box(box: BoundingBox, *, scale_x: float, scale_y: float) -> BoundingBox:
    return BoundingBox(
        x_min=max(0, int(round(box.x_min * scale_x))),
        y_min=max(0, int(round(box.y_min * scale_y))),
        x_max=max(0, int(round(box.x_max * scale_x))),
        y_max=max(0, int(round(box.y_max * scale_y))),
    )


def _box_area(box: BoundingBox) -> int:
    return max(0, box.x_max - box.x_min) * max(0, box.y_max - box.y_min)
