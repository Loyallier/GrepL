"""Object detection and cropping pipeline for found-item photos.

The module tries a trained OpenCV DNN detector first. If the model files are
not available, or the detector returns no usable boxes, it falls back to a
simple foreground-region detector based on color and texture differences.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from contracts import RowItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CROP_DIR = PROJECT_ROOT / "data" / "cropped_item_image"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "ssd_mobilenet_v2_coco_2018_03_29"
DEFAULT_DETECTOR_WEIGHTS = DEFAULT_MODEL_DIR / "frozen_inference_graph.pb"
DEFAULT_DETECTOR_CONFIG = DEFAULT_MODEL_DIR / "graph.pbtxt"
DEFAULT_DNN_CONFIDENCE_THRESHOLD = 0.10
DEFAULT_FOREGROUND_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_DNN_INPUT_SIZE = 640
DEFAULT_MAX_OBJECTS = 30
DEFAULT_CROP_BOTTOM_RIGHT_PADDING_RATIO = 0.04
DEFAULT_CROP_BOTTOM_RIGHT_PADDING_MAX = 48
FALLBACK_FULL_IMAGE_CONFIDENCE = 0.2
LOGGER = logging.getLogger(__name__)


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
    """Detect objects from one raw image, crop them, and return cropped items.

    This is the main entry point used by the rest of the search pipeline.
    """

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

    weights_path = _model_path_from_env(
        "GREPL_DETECTOR_WEIGHTS",
        default_path=DEFAULT_DETECTOR_WEIGHTS,
    )
    config_path = _model_path_from_env(
        "GREPL_DETECTOR_CONFIG",
        default_path=DEFAULT_DETECTOR_CONFIG,
    )
    max_objects = _int_from_env("GREPL_DETECTOR_MAX_OBJECTS", DEFAULT_MAX_OBJECTS)

    if weights_path:
        try:
            # Prefer the trained model when it is present because it usually
            # produces tighter boxes than the generic foreground fallback.
            object_boxes = _detect_with_opencv_dnn(
                source_path,
                weights_path=weights_path,
                config_path=config_path,
                confidence_threshold=_float_from_env(
                    "GREPL_DNN_CONFIDENCE_THRESHOLD",
                    DEFAULT_DNN_CONFIDENCE_THRESHOLD,
                ),
                max_objects=max_objects,
            )
            if object_boxes:
                return object_boxes
        except Exception as error:
            LOGGER.warning("OpenCV detector failed for %s: %s", source_path, error)

    # The fallback keeps the application usable in local setups where the
    # large detector model has not been downloaded.
    object_boxes = _detect_with_foreground_regions(
        source_path,
        confidence_threshold=DEFAULT_FOREGROUND_CONFIDENCE_THRESHOLD,
        max_objects=max_objects,
    )
    if object_boxes:
        return object_boxes

    return _full_image_box(source_path, confidence=FALLBACK_FULL_IMAGE_CONFIDENCE)


def crop_detected_objects(
    row_image_path: str | Path,
    object_boxes: Iterable[ObjectBox],
) -> list[RowItem]:
    """Crop sub-images using detected object coordinates.

    Args:
        row_image_path: Raw found-item image path.
        object_boxes: Bounding boxes produced by the detection step.
        Cropped sub-images are saved under data/cropped_item_image/.

    Returns:
        RowItem objects containing each cropped image path and detection
        confidence.
    """

    source_path = Path(row_image_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw image does not exist: {source_path}")

    from PIL import Image, ImageOps

    crop_dir = DEFAULT_CROP_DIR
    # Crops are materialized on disk because later embedding and ranking steps
    # work with image paths rather than in-memory image objects.
    crop_dir.mkdir(parents=True, exist_ok=True)

    cropped_items: list[RowItem] = []
    with Image.open(source_path) as source:
        # Respect EXIF orientation so boxes and crops line up with the image as
        # it is visually displayed.
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        for index, object_box in enumerate(object_boxes, start=1):
            normalized_box = _normalize_box(object_box, width, height)
            if normalized_box is None:
                continue
            crop_box = _expand_crop_box_bottom_right(
                normalized_box,
                image_width=width,
                image_height=height,
            )

            crop_path = crop_dir / f"{source_path.stem}_object_{index:03d}.png"
            image.crop(crop_box).save(crop_path)
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
    """Run an OpenCV DNN model and convert raw detections to ObjectBox values."""

    import cv2

    if not weights_path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"Detector config not found: {config_path}")

    model_format = _model_format(weights_path, config_path)
    # OpenCV exposes slightly different loaders for TensorFlow, Caffe, and
    # generic DNN models, so choose the reader from the file layout.
    if model_format == "tensorflow":
        net = cv2.dnn.readNetFromTensorflow(str(weights_path), str(config_path))
    elif config_path is not None and config_path.suffix in {".prototxt", ".txt"}:
        net = cv2.dnn.readNetFromCaffe(str(config_path), str(weights_path))
    elif config_path is not None:
        net = cv2.dnn.readNet(str(weights_path), str(config_path))
    else:
        net = cv2.dnn.readNet(str(weights_path))

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    height, width = image.shape[:2]
    if model_format == "tensorflow":
        # TensorFlow object detection models usually expect a square image with
        # RGB channel order.
        input_size = _int_from_env("GREPL_DNN_INPUT_SIZE", DEFAULT_DNN_INPUT_SIZE)
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=1.0,
            size=(input_size, input_size),
            mean=0,
            swapRB=True,
            crop=False,
        )
    else:
        # MobileNet SSD Caffe-style models commonly use 300x300 input and the
        # standard 127.5 mean subtraction.
        blob = cv2.dnn.blobFromImage(
            image,
            scalefactor=0.007843,
            size=(300, 300),
            mean=127.5,
        )
    net.setInput(blob)
    raw_detections = net.forward()

    object_boxes: list[ObjectBox] = []
    try:
        detections = raw_detections.reshape(-1, 7)
    except ValueError as error:
        raise ValueError(
            f"Unsupported detector output shape: {raw_detections.shape}"
        ) from error

    # Expected detection row format: [batch_id, class_id, confidence,
    # x_min, y_min, x_max, y_max], where coordinates are normalized to [0, 1].
    for raw in detections:
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
        if _is_upper_background_box(normalized_box, image_height=height):
            # Ignore boxes entirely in the top strip. In the project images this
            # area is often background or page chrome instead of a found item.
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

    object_boxes = _non_max_suppression(object_boxes, iou_threshold=0.55)
    object_boxes.sort(key=lambda object_box: object_box.confidence, reverse=True)
    return object_boxes[:max_objects]


def _detect_with_foreground_regions(
    image_path: Path,
    *,
    confidence_threshold: float,
    max_objects: int,
) -> list[ObjectBox]:
    """Find foreground regions without a trained model.

    This heuristic looks for pixels that differ from the border background and
    for textured areas, then turns connected mask regions into crop boxes.
    """

    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        original_width, original_height = image.size
        # Downscale large photos for faster mask operations, then scale boxes
        # back to the original image size before returning.
        scale = min(1.0, 900 / max(original_width, original_height))
        if scale < 1.0:
            image = image.resize((int(original_width * scale), int(original_height * scale)))

    data = np.asarray(image).astype(np.int16)
    height, width = data.shape[:2]
    if height < 8 or width < 8:
        return []

    background = _estimate_border_color(data)
    distance = np.linalg.norm(data - background, axis=2)
    color_threshold = _foreground_color_threshold(distance)
    texture = _texture_strength(data)
    texture_threshold = _foreground_texture_threshold(texture)
    # A pixel is considered foreground if it is either visually different from
    # the border color or has enough local texture.
    mask = (distance > color_threshold) | (texture > texture_threshold)
    mask = _remove_border_noise(mask)
    # Morphological cleanup joins nearby foreground pixels and removes isolated
    # gaps before connected-component analysis.
    mask = _dilate_mask(mask, iterations=max(2, min(width, height) // 90))
    mask = _erode_mask(mask, iterations=1)

    min_area = max(60, int(width * height * 0.0015))
    components = _connected_components(mask, min_area=min_area)
    object_boxes: list[ObjectBox] = []
    for left, top, right, bottom, area in components:
        padded = _pad_box(
            ObjectBox(
                left=left,
                top=top,
                right=right + 1,
                bottom=bottom + 1,
                confidence=0.0,
            ),
            padding=max(4, int(min(width, height) * 0.025)),
        )
        object_box = _scale_object_box(
            padded,
            scale_x=original_width / width,
            scale_y=original_height / height,
        )
        normalized_box = _normalize_box(object_box, original_width, original_height)
        if normalized_box is None:
            continue

        area_ratio = area / (width * height)
        contrast = float(distance[top : bottom + 1, left : right + 1].mean())
        texture_score = float(texture[top : bottom + 1, left : right + 1].mean())
        # Confidence is a heuristic score based on region size, contrast, and
        # texture strength. It is not calibrated like a model probability.
        confidence = min(
            0.92,
            0.3 + area_ratio * 3.0 + contrast / 255 * 0.28 + texture_score / 255 * 0.22,
        )
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

    object_boxes = _merge_overlapping_boxes(
        object_boxes,
        image_width=original_width,
        image_height=original_height,
    )
    object_boxes = _split_oversized_boxes(
        object_boxes,
        image_width=original_width,
        image_height=original_height,
    )
    object_boxes.sort(key=lambda object_box: _box_area(object_box), reverse=True)
    return object_boxes[:max_objects]


def _connected_components(mask: np.ndarray, *, min_area: int) -> list[tuple[int, int, int, int, int]]:
    """Group neighboring foreground pixels into rectangular components."""

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
        # Filter both by true foreground area and by bounding rectangle size so
        # small specks and thin noise lines do not become crops.
        if area >= min_area and box_area >= min_area:
            components.append((left, top, right, bottom, area))

    return components


def _estimate_border_color(data: np.ndarray) -> np.ndarray:
    import numpy as np

    # The border is used as a rough background sample because found items are
    # usually centered away from the image edges.
    top = data[0, :, :]
    bottom = data[-1, :, :]
    left = data[:, 0, :]
    right = data[:, -1, :]
    border_pixels = np.concatenate((top, bottom, left, right), axis=0)
    return np.median(border_pixels, axis=0)


def _foreground_color_threshold(distance: np.ndarray) -> float:
    import numpy as np

    p60 = float(np.percentile(distance, 60))
    p85 = float(np.percentile(distance, 85))
    return max(12.0, min(32.0, (p60 + p85) / 2))


def _texture_strength(data: np.ndarray) -> np.ndarray:
    import numpy as np

    gray = data.astype(np.float32).mean(axis=2)
    horizontal = np.zeros_like(gray)
    vertical = np.zeros_like(gray)
    horizontal[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    vertical[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    return np.maximum(horizontal, vertical)


def _foreground_texture_threshold(texture: np.ndarray) -> float:
    import numpy as np

    p90 = float(np.percentile(texture, 90))
    p98 = float(np.percentile(texture, 98))
    return max(10.0, min(45.0, (p90 + p98) / 2))


def _remove_border_noise(mask: np.ndarray) -> np.ndarray:
    """Clear a thin image border to avoid selecting frame/background edges."""

    cleaned = mask.copy()
    height, width = cleaned.shape
    border = max(1, min(width, height) // 60)
    cleaned[:border, :] = False
    cleaned[-border:, :] = False
    cleaned[:, :border] = False
    cleaned[:, -border:] = False
    return cleaned


def _dilate_mask(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    """Expand foreground pixels to join nearby pieces of the same object."""

    expanded = mask.copy()
    for _ in range(iterations):
        padded = _pad_boolean_mask(expanded)
        expanded = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return expanded


def _erode_mask(mask: np.ndarray, *, iterations: int) -> np.ndarray:
    """Shrink the mask after dilation to reduce over-expanded boundaries."""

    eroded = mask.copy()
    for _ in range(iterations):
        padded = _pad_boolean_mask(eroded)
        eroded = (
            padded[1:-1, 1:-1]
            & padded[:-2, 1:-1]
            & padded[2:, 1:-1]
            & padded[1:-1, :-2]
            & padded[1:-1, 2:]
        )
    return eroded


def _pad_boolean_mask(mask: np.ndarray) -> np.ndarray:
    import numpy as np

    return np.pad(mask, pad_width=1, mode="constant", constant_values=False)


def _pad_box(object_box: ObjectBox, *, padding: int) -> ObjectBox:
    return ObjectBox(
        left=object_box.left - padding,
        top=object_box.top - padding,
        right=object_box.right + padding,
        bottom=object_box.bottom + padding,
        confidence=object_box.confidence,
    )


def _merge_overlapping_boxes(
    object_boxes: list[ObjectBox],
    *,
    image_width: int,
    image_height: int,
) -> list[ObjectBox]:
    """Merge boxes that likely describe the same physical item."""

    merged: list[ObjectBox] = []
    for object_box in object_boxes:
        normalized = _normalize_box(object_box, image_width, image_height)
        if normalized is None:
            continue

        candidate = ObjectBox(*normalized, confidence=object_box.confidence)
        for index, existing in enumerate(merged):
            if not _boxes_should_merge(existing, candidate):
                continue

            merged[index] = ObjectBox(
                left=min(existing.left, candidate.left),
                top=min(existing.top, candidate.top),
                right=max(existing.right, candidate.right),
                bottom=max(existing.bottom, candidate.bottom),
                confidence=max(existing.confidence, candidate.confidence),
            )
            break
        else:
            merged.append(candidate)

    return merged


def _split_oversized_boxes(
    object_boxes: list[ObjectBox],
    *,
    image_width: int,
    image_height: int,
) -> list[ObjectBox]:
    """Split very large fallback boxes into smaller search candidates."""

    expanded: list[ObjectBox] = []
    for object_box in object_boxes:
        if not _box_is_oversized(object_box, image_width, image_height):
            expanded.append(object_box)
            continue

        box_width = max(1, object_box.right - object_box.left)
        box_height = max(1, object_box.bottom - object_box.top)
        columns = 4 if box_width >= box_height * 1.15 else 3
        rows = 2 if box_width >= box_height else 3
        expanded.extend(
            _tile_box(
                object_box,
                columns=columns,
                rows=rows,
                image_width=image_width,
                image_height=image_height,
            )
        )

    return _deduplicate_boxes(expanded)


def _box_is_oversized(
    object_box: ObjectBox,
    image_width: int,
    image_height: int,
) -> bool:
    image_area = max(1, image_width * image_height)
    width_ratio = (object_box.right - object_box.left) / max(1, image_width)
    height_ratio = (object_box.bottom - object_box.top) / max(1, image_height)
    area_ratio = _box_area(object_box) / image_area
    return area_ratio >= 0.45 or (width_ratio >= 0.75 and height_ratio >= 0.55)


def _tile_box(
    object_box: ObjectBox,
    *,
    columns: int,
    rows: int,
    image_width: int,
    image_height: int,
) -> list[ObjectBox]:
    """Create overlapping tiles inside one large box."""

    tiles: list[ObjectBox] = []
    box_width = object_box.right - object_box.left
    box_height = object_box.bottom - object_box.top
    step_x = box_width / columns
    step_y = box_height / rows
    overlap = 0.18

    for row in range(rows):
        for column in range(columns):
            left = object_box.left + column * step_x - step_x * overlap / 2
            top = object_box.top + row * step_y - step_y * overlap / 2
            right = left + step_x * (1 + overlap)
            bottom = top + step_y * (1 + overlap)
            normalized = _normalize_box(
                ObjectBox(
                    left=int(round(left)),
                    top=int(round(top)),
                    right=int(round(right)),
                    bottom=int(round(bottom)),
                    confidence=max(0.25, object_box.confidence * 0.85),
                ),
                image_width,
                image_height,
            )
            if normalized is None:
                continue
            tiles.append(
                ObjectBox(
                    *normalized,
                    confidence=max(0.25, object_box.confidence * 0.85),
                )
            )

    return tiles


def _deduplicate_boxes(object_boxes: list[ObjectBox]) -> list[ObjectBox]:
    """Remove nearly identical boxes while preserving the original order."""

    deduplicated: list[ObjectBox] = []
    for object_box in object_boxes:
        if any(_box_iou(object_box, existing) >= 0.9 for existing in deduplicated):
            continue
        deduplicated.append(object_box)
    return deduplicated


def _non_max_suppression(
    object_boxes: list[ObjectBox],
    *,
    iou_threshold: float,
) -> list[ObjectBox]:
    """Keep high-confidence boxes and drop lower-confidence overlaps."""

    selected: list[ObjectBox] = []
    for candidate in sorted(
        object_boxes,
        key=lambda object_box: object_box.confidence,
        reverse=True,
    ):
        if any(
            _box_iou(candidate, existing) > iou_threshold
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _box_iou(first: ObjectBox, second: ObjectBox) -> float:
    """Return intersection-over-union for two boxes."""

    intersection_left = max(first.left, second.left)
    intersection_top = max(first.top, second.top)
    intersection_right = min(first.right, second.right)
    intersection_bottom = min(first.bottom, second.bottom)
    intersection = _box_area(
        ObjectBox(
            left=intersection_left,
            top=intersection_top,
            right=intersection_right,
            bottom=intersection_bottom,
            confidence=0.0,
        )
    )
    if intersection <= 0:
        return 0.0

    union = _box_area(first) + _box_area(second) - intersection
    return intersection / max(1, union)


def _boxes_should_merge(first: ObjectBox, second: ObjectBox) -> bool:
    intersection_left = max(first.left, second.left)
    intersection_top = max(first.top, second.top)
    intersection_right = min(first.right, second.right)
    intersection_bottom = min(first.bottom, second.bottom)
    intersection_area = _box_area(
        ObjectBox(
            left=intersection_left,
            top=intersection_top,
            right=intersection_right,
            bottom=intersection_bottom,
            confidence=0.0,
        )
    )
    if intersection_area <= 0:
        return False

    smaller_area = max(1, min(_box_area(first), _box_area(second)))
    return intersection_area / smaller_area >= 0.2


def _model_path_from_env(env_name: str, *, default_path: Path) -> Path | None:
    """Read an optional model path from the environment."""

    configured_path = os.environ.get(env_name)
    if configured_path:
        return Path(configured_path)
    if default_path.is_file():
        return default_path
    return None


def _model_format(weights_path: Path, config_path: Path | None) -> str:
    if (
        weights_path.suffix == ".pb"
        and config_path is not None
        and config_path.suffix == ".pbtxt"
    ):
        return "tensorflow"
    return "generic"


def _is_upper_background_box(
    box: tuple[int, int, int, int],
    *,
    image_height: int,
) -> bool:
    return box[3] < image_height * 0.16


def _float_from_env(env_name: str, default: float) -> float:
    configured_value = os.environ.get(env_name)
    if configured_value is None:
        return default
    try:
        return float(configured_value)
    except ValueError:
        LOGGER.warning("Invalid float value for %s: %s", env_name, configured_value)
        return default


def _int_from_env(env_name: str, default: int) -> int:
    configured_value = os.environ.get(env_name)
    if configured_value is None:
        return default
    try:
        return int(configured_value)
    except ValueError:
        LOGGER.warning("Invalid integer value for %s: %s", env_name, configured_value)
        return default


def _scale_object_box(object_box: ObjectBox, *, scale_x: float, scale_y: float) -> ObjectBox:
    """Scale a box from resized-image coordinates back to source coordinates."""

    return ObjectBox(
        left=max(0, int(round(object_box.left * scale_x))),
        top=max(0, int(round(object_box.top * scale_y))),
        right=max(0, int(round(object_box.right * scale_x))),
        bottom=max(0, int(round(object_box.bottom * scale_y))),
        confidence=object_box.confidence,
    )


def _normalize_box(
    object_box: ObjectBox,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Clamp a detected box to image boundaries and reject empty crops."""

    left = max(0, min(int(object_box.left), image_width))
    top = max(0, min(int(object_box.top), image_height))
    right = max(0, min(int(object_box.right), image_width))
    bottom = max(0, min(int(object_box.bottom), image_height))

    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _expand_crop_box_bottom_right(
    box: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Add a small bottom-right margin so crops do not clip object edges."""

    left, top, right, bottom = box
    padding = max(
        2,
        min(
            DEFAULT_CROP_BOTTOM_RIGHT_PADDING_MAX,
            int(
                round(
                    min(image_width, image_height)
                    * DEFAULT_CROP_BOTTOM_RIGHT_PADDING_RATIO
                )
            ),
        ),
    )
    return (
        left,
        top,
        min(image_width, right + padding),
        min(image_height, bottom + padding),
    )


def _box_area(object_box: ObjectBox) -> int:
    return max(0, object_box.right - object_box.left) * max(
        0,
        object_box.bottom - object_box.top,
    )


def _full_image_box(image_path: Path, *, confidence: float) -> list[ObjectBox]:
    """Return one fallback box covering the whole image."""

    from PIL import Image, ImageOps

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size

    if width <= 0 or height <= 0:
        return []
    return [
        ObjectBox(
            left=0,
            top=0,
            right=width,
            bottom=height,
            confidence=confidence,
        )
    ]
