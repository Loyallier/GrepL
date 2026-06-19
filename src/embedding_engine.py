"""TF-CLIP embedding engine for GrepL.

This module owns the CLIP retrieval unit used by the registration and ranking
pipelines:

- encode_text(): user description -> normalized text vector
- encode_image(): cropped item image -> normalized image vector
- register_item_image(): persist image vectors for registered found items
- match_text_to_images(): compute visual_similarity and return Candidate objects
"""

from __future__ import annotations

import json
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

_model = None
_image_preprocess = None
_text_preprocess = None
_text_encoder = None
_image_encoder = None


def encode_text(description: str) -> np.ndarray:
    """Convert one user description into a 512-dimensional normalized vector."""
    text = description.strip()
    if not text:
        raise ValueError("description must not be empty")

    _ensure_model_loaded()
    tokens = _text_preprocess([text])
    vector = np.asarray(_text_encoder(tokens, training=False).numpy()[0], dtype="float32")
    return _l2_normalize(vector)


def encode_image(image_path: str | Path) -> np.ndarray:
    """Convert one cropped found-item image into a normalized image vector."""
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
    """Encode and persist an item's image embedding for later retrieval."""
    vector = encode_image(registering_item.image_path)
    store_file = Path(store_path)
    store = _read_embedding_store(store_file)
    store[registering_item.item_id] = {
        "item_id": registering_item.item_id,
        "image_path": str(registering_item.image_path),
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "dimension": int(vector.shape[0]),
        "embedding": vector.tolist(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_embedding_store(store_file, store)
    return True


def match_text_to_images(
    description: str,
    items: Iterable[LostItem],
    *,
    store_path: str | Path = DEFAULT_EMBEDDING_STORE,
) -> list[Candidate]:
    """Compare user text with found-item images and return ranker candidates."""
    text_vector = encode_text(description)
    store_file = Path(store_path)
    store = _read_embedding_store(store_file)
    candidates: list[Candidate] = []
    store_changed = False

    for item in items:
        image_vector = _embedding_for_item(item, store)
        if image_vector is None:
            image_vector = encode_image(item.image_path)
            store[item.item_id] = {
                "item_id": item.item_id,
                "image_path": item.image_path,
                "model_name": MODEL_NAME,
                "pretrained": PRETRAINED,
                "dimension": int(image_vector.shape[0]),
                "embedding": image_vector.tolist(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            store_changed = True

        cosine = float(np.dot(text_vector, image_vector))
        visual_similarity = _cosine_to_unit_interval(cosine)
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
    if WEIGHTS_PATH:
        path = Path(WEIGHTS_PATH)
        if path.is_file():
            return path
        raise FileNotFoundError(f"GREPL_TFCLIP_WEIGHTS_PATH does not exist: {path}")
    if DEFAULT_LOCAL_WEIGHTS.is_file():
        return DEFAULT_LOCAL_WEIGHTS
    return None


def _embedding_for_item(item: LostItem, store: dict[str, dict[str, Any]]) -> np.ndarray | None:
    record = store.get(item.item_id)
    if not record:
        return None
    if str(record.get("image_path")) != str(item.image_path):
        return None
    embedding = record.get("embedding")
    if not isinstance(embedding, list):
        return None
    vector = np.asarray(embedding, dtype="float32")
    if vector.ndim != 1 or vector.size == 0:
        return None
    return _l2_normalize(vector)


def _read_embedding_store(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Embedding store must be a JSON object: {path}")
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}


def _write_embedding_store(path: Path, store: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype("float32")
    return (vector / norm).astype("float32")


def _cosine_to_unit_interval(cosine: float) -> float:
    return round(max(0.0, min(1.0, (cosine + 1.0) / 2.0)), 4)

