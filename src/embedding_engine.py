"""TF-CLIP embedding engine for GrepL.
GrepL 的 TF-CLIP 向量化引擎。

This module owns the CLIP retrieval unit used by the registration and ranking
pipelines.
本模块负责登记与排序流程中的 CLIP 检索单元。

- encode_text(): user description -> normalized text vector
  用户描述 -> 归一化文本向量
- encode_image(): cropped item image -> normalized image vector
  裁剪后的物品图片 -> 归一化图像向量
- register_item_image(): persist image vectors for registered found items
  为已登记物品持久化图像向量
- match_text_to_images(): compute visual_similarity and return Candidate objects
  计算 visual_similarity，并返回 Candidate 对象
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
try:
    from tfclip import create_model_and_transforms
except Exception:
    create_model_and_transforms = None

from contracts import Candidate, LostItem, RegisterItem


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_STORE = PROJECT_ROOT / "data" / "generated" / "image_embeddings.json"
DEFAULT_LOCAL_WEIGHTS = PROJECT_ROOT / "data" / "weights" / "tfclip_vit_b_32_laion.h5"

MODEL_NAME = os.environ.get("GREPL_TFCLIP_MODEL_NAME", "ViT-B-32")
PRETRAINED = os.environ.get("GREPL_TFCLIP_PRETRAINED", "laion2b_s34b_b79k")
WEIGHTS_PATH = os.environ.get("GREPL_TFCLIP_WEIGHTS_PATH")
EXPECTED_EMBEDDING_DIMENSION = int(os.environ.get("GREPL_TFCLIP_EMBEDDING_DIMENSION", "512"))

LOGGER = logging.getLogger(__name__)

_model = None
_image_preprocess = None
_text_preprocess = None
_text_encoder = None
_image_encoder = None


def encode_text(description: str) -> np.ndarray:
    """Convert one user description into a normalized text vector.
    将一条用户描述转换为归一化文本向量。
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
    将一张裁剪后的拾获物品图片转换为归一化图像向量。
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
    生成并保存单个物品的图像向量，供后续检索使用。
    """
    if create_model_and_transforms is None:
        return False
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
    比较用户文本与拾获物品图片，并返回可交给 ranker 的 Candidate 列表。
    """
    if create_model_and_transforms is None:
        return []
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
            # 单条坏数据不应导致整次搜索失败。
            LOGGER.warning(
                "Skip item %s because its image embedding cannot be prepared: %s",
                item.item_id,
                error,
            )
            continue

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
    """Lazy-load the TF-CLIP model and split it into text/image encoders.
    延迟加载 TF-CLIP 模型，并拆分出文本编码器和图像编码器。
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
    解析可选的本地权重文件；如果没有，则交给 tfclip 自动下载。
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
    返回有效缓存图像向量；如果缓存失效，则返回 None 以便重新生成。
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
    信任缓存向量之前，先校验缓存元信息。
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
    检查缓存记录维度和当前期望维度是否一致。
    """
    dimension = _safe_int(recorded_dimension)
    if dimension is None or dimension != vector_size:
        return False
    if EXPECTED_EMBEDDING_DIMENSION > 0 and vector_size != EXPECTED_EMBEDDING_DIMENSION:
        return False
    return True


def _item_image_available(item: LostItem) -> bool:
    """Return whether the item's image path can be used for retrieval.
    判断物品图片路径是否可用于检索。
    """
    path = Path(item.image_path)
    if path.is_file():
        return True
    LOGGER.warning("Skip item %s because image does not exist: %s", item.item_id, path)
    return False


def _embedding_record(item_id: str, image_path: str | Path, vector: np.ndarray) -> dict[str, Any]:
    """Build one JSON-serializable embedding cache record.
    构建一条可写入 JSON 的向量缓存记录。
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
    读取图像向量 JSON 存储；无效缓存数据会被忽略。
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
    原子化写入向量库，降低文件写到一半损坏的风险。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector for cosine-similarity retrieval.
    对向量做 L2 归一化，便于计算余弦相似度。
    """
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype("float32")
    return (vector / norm).astype("float32")


def _cosine_to_unit_interval(cosine: float) -> float:
    """Map cosine similarity from [-1, 1] to a UI-friendly [0, 1] score.
    将余弦相似度从 [-1, 1] 映射为更适合展示的 [0, 1] 分数。
    """
    return round(max(0.0, min(1.0, (cosine + 1.0) / 2.0)), 4)


def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible.
    尽可能将输入转换为整数。
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
