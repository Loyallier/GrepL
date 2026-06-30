"""TF-CLIP embedding engine for GrepL.

This module owns the CLIP retrieval unit used by the registration and ranking
pipelines.

- encode_text(): user description -> normalized text vector
- encode_image(): cropped item image -> normalized image vector
- register_item_image(): persist image vectors for registered found items
- match_text_to_images(): compute visual_similarity and return Candidate objects
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
from PIL import Image
from tfclip import create_model_and_transforms

from contracts import Candidate, LostItem, RegisterItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_STORE = PROJECT_ROOT / "data" / "generated" / "image_embeddings.json"
DEFAULT_LOCAL_WEIGHTS = PROJECT_ROOT / "data" / "weights" / "tfclip_vit_b_32_laion.h5"

MODEL_NAME = os.environ.get("GREPL_TFCLIP_MODEL_NAME", "ViT-B-32")
PRETRAINED = os.environ.get("GREPL_TFCLIP_PRETRAINED", "laion2b_s34b_b79k")
WEIGHTS_PATH = os.environ.get("GREPL_TFCLIP_WEIGHTS_PATH")
EXPECTED_EMBEDDING_DIMENSION = int(os.environ.get("GREPL_TFCLIP_EMBEDDING_DIMENSION", "512"))
DISPLAY_SIMILARITY_CENTER = float(os.environ.get("GREPL_TFCLIP_DISPLAY_CENTER", "0.23"))
DISPLAY_SIMILARITY_SCALE = float(os.environ.get("GREPL_TFCLIP_DISPLAY_SCALE", "0.05"))

LOGGER = logging.getLogger(__name__)

_model = None
_image_preprocess = None
_text_preprocess = None
_text_encoder = None
_image_encoder = None


def encode_text(description: str) -> np.ndarray:
    """Convert one user description into a normalized text vector.
    """
    text = description.strip()
    if not text:
        raise ValueError("description must not be empty")

    _ensure_model_loaded()
    tokens = _text_preprocess([text])
    vector = np.asarray(_text_encoder(tokens, training=False).numpy()[0], dtype="float32")
    return _l2_normalize(vector)


def encode_image(image_path: str | Path) -> np.ndarray:
    """Convert one cropped found-item image into a normalized image vector.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")

    _ensure_model_loaded()
    with Image.open(path) as image:
        image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)

    image_tensor = _image_preprocess(image_array)[None]
    vector = np.asarray(_image_encoder(image_tensor, training=False).numpy()[0], dtype="float32")
    return _l2_normalize(vector)


def register_item_image(
    registering_item: RegisterItem,
    *,
    store_path: str | Path = DEFAULT_EMBEDDING_STORE,
) -> bool:
    """Encode and persist an item's image embedding for later retrieval.
    """
    try:
        vector = encode_image(registering_item.image_path)
    except (FileNotFoundError, OSError) as error:
        LOGGER.warning(
            "Skip embedding registration for item %s because image is unavailable: %s",
            registering_item.item_id,
            error,
        )
        return False

    store_file = Path(store_path)
    store = _read_embedding_store(store_file)
    store[str(registering_item.item_id)] = _embedding_record(
        item_id=registering_item.item_id,
        image_path=registering_item.image_path,
        vector=vector,
    )
    _write_embedding_store(store_file, store)
    return True


def match_text_to_images(
    description: str,
    items: Iterable[LostItem],
    *,
    store_path: str | Path = DEFAULT_EMBEDDING_STORE,
) -> list[Candidate]:
    """Compare user text with found-item images and return ranker candidates.
    """
    text_vector = encode_text(description)
    store_file = Path(store_path)
    store = _read_embedding_store(store_file)
    candidates: list[Candidate] = []
    store_changed = False

    for item in items:
        if not _item_image_available(item):
            continue

        try:
            image_vector = _embedding_for_item(item, store)
            if image_vector is None:
                image_vector = encode_image(item.image_path)
                store[str(item.item_id)] = _embedding_record(
                    item_id=item.item_id,
                    image_path=item.image_path,
                    vector=image_vector,
                )
                store_changed = True
        except (FileNotFoundError, OSError, ValueError) as error:
            # Keep one bad record from breaking the whole search.
            LOGGER.warning(
                "Skip item %s because its image embedding cannot be prepared: %s",
                item.item_id,
                error,
            )
            continue

        cosine = float(np.dot(text_vector, image_vector))
        visual_similarity = _cosine_to_display_similarity(cosine)
        candidates.append(
            Candidate(
                item_id=item.item_id,
                image_path=item.image_path,
                found_time=item.found_time,
                found_location=item.found_location,
                visual_similarity=visual_similarity,
                bound_confidence=float(item.bound_confidence),
            )
        )

    if store_changed:
        _write_embedding_store(store_file, store)

    candidates.sort(key=lambda candidate: candidate.visual_similarity, reverse=True)
    return candidates



def _ensure_model_loaded() -> None:
    """Lazy-load the TF-CLIP model and split it into text/image encoders.
    """
    global _model, _image_preprocess, _text_preprocess, _text_encoder, _image_encoder

    if _model is not None:
        return

    kwargs: dict[str, Any] = {}
    weights_path = _resolve_weights_path()
    if weights_path is not None:
        kwargs["weights_path"] = str(weights_path)

    _model, _image_preprocess, _text_preprocess = create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
        **kwargs,
    )
    _text_encoder = keras.Model(
        inputs=_model.inputs[1],
        outputs=_model.get_layer("text_head_out").output,
        name="grepl_tfclip_text_encoder",
    )
    _image_encoder = keras.Model(
        inputs=_model.inputs[0],
        outputs=_model.get_layer("vision_head_out").output,
        name="grepl_tfclip_image_encoder",
    )



def _resolve_weights_path() -> Path | None:
    """Resolve an optional local weights file, otherwise let tfclip download.
    """
    if WEIGHTS_PATH:
        path = Path(WEIGHTS_PATH)
        if path.is_file():
            return path
        raise FileNotFoundError(f"GREPL_TFCLIP_WEIGHTS_PATH does not exist: {path}")
    if DEFAULT_LOCAL_WEIGHTS.is_file():
        return DEFAULT_LOCAL_WEIGHTS
    return None


def _embedding_for_item(item: LostItem, store: dict[str, dict[str, Any]]) -> np.ndarray | None:
    """Return a valid cached image embedding, or None when it must be rebuilt.
    """
    record = store.get(str(item.item_id))
    if not _record_matches_item(item, record):
        return None

    embedding = record.get("embedding")
    if not isinstance(embedding, list):
        return None

    try:
        vector = np.asarray(embedding, dtype="float32")
    except (TypeError, ValueError):
        return None

    if vector.ndim != 1 or vector.size == 0:
        return None
    if not np.all(np.isfinite(vector)):
        return None
    if not _dimension_matches(record.get("dimension"), vector.size):
        return None

    return _l2_normalize(vector)


def _record_matches_item(item: LostItem, record: dict[str, Any] | None) -> bool:
    """Validate cache metadata before trusting a stored vector.
    """
    if not isinstance(record, dict):
        return False
    if str(record.get("item_id")) != str(item.item_id):
        return False
    if str(record.get("image_path")) != str(item.image_path):
        return False
    if record.get("model_name") != MODEL_NAME:
        return False
    if record.get("pretrained") != PRETRAINED:
        return False
    return True


def _dimension_matches(recorded_dimension: Any, vector_size: int) -> bool:
    """Check stored and expected embedding dimensions.
    """
    dimension = _safe_int(recorded_dimension)
    if dimension is None or dimension != vector_size:
        return False
    if EXPECTED_EMBEDDING_DIMENSION > 0 and vector_size != EXPECTED_EMBEDDING_DIMENSION:
        return False
    return True


def _item_image_available(item: LostItem) -> bool:
    """Return whether the item's image path can be used for retrieval.
    """
    path = Path(item.image_path)
    if path.is_file():
        return True
    LOGGER.warning("Skip item %s because image does not exist: %s", item.item_id, path)
    return False


def _embedding_record(item_id: str, image_path: str | Path, vector: np.ndarray) -> dict[str, Any]:
    """Build one JSON-serializable embedding cache record.
    """
    return {
        "item_id": str(item_id),
        "image_path": str(image_path),
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dimension": int(vector.shape[0]),
        "embedding": vector.astype("float32").tolist(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _read_embedding_store(path: Path) -> dict[str, dict[str, Any]]:
    """Read the image embedding JSON store; invalid cache data is ignored.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        LOGGER.warning("Ignore corrupted embedding store %s: %s", path, error)
        return {}

    if not isinstance(data, dict):
        LOGGER.warning("Ignore embedding store because it is not a JSON object: %s", path)
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _write_embedding_store(path: Path, store: dict[str, dict[str, Any]]) -> None:
    """Write the embedding store atomically to reduce partial-file risk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector for cosine-similarity retrieval.
    """
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype("float32")
    return (vector / norm).astype("float32")


def _cosine_to_unit_interval(cosine: float) -> float:
    """Map cosine similarity from [-1, 1] to a UI-friendly [0, 1] score.
    """
    return round(max(0.0, min(1.0, (cosine + 1.0) / 2.0)), 4)



def _cosine_to_display_similarity(cosine: float) -> float:
    """Map raw CLIP cosine to a user-facing visual similarity score with sigmoid."""
    if not np.isfinite(cosine):
        return 0.0
    if DISPLAY_SIMILARITY_SCALE <= 1e-12:
        return _cosine_to_unit_interval(cosine)

    scaled = (float(cosine) - DISPLAY_SIMILARITY_CENTER) / DISPLAY_SIMILARITY_SCALE
    scaled = max(-60.0, min(60.0, scaled))
    similarity = 1.0 / (1.0 + float(np.exp(-scaled)))
    return round(max(0.0, min(1.0, similarity)), 4)

def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
