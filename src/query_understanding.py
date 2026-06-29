"""User-side query understanding helpers for guided search confirmation."""

from __future__ import annotations

# Use project's own tfclip-based embedding engine instead of KerasHub:
# https://github.com/shkarupa-alex/tfclip

from dataclasses import dataclass, field
from functools import lru_cache
import sys
from pathlib import Path
import re

import numpy as np

# Import from embedding_engine
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from embedding_engine import encode_text


ITEM_LABELS = {
    "Bottle": {
        "keywords": ["bottle", "cup", "mug", "thermos", "tumbler", "flask", "water bottle", "杯子", "水杯", "杯", "水瓶", "保温杯"],
        "prompts": ["a photo of a bottle", "a photo of a cup", "a photo of a water bottle"],
    },
    "Keys": {
        "keywords": ["keys", "key", "钥匙", "钥匙串", "门钥匙"],
        "prompts": ["a photo of keys", "a photo of a keychain"],
    },
    "Earphones": {
        "keywords": ["earphones", "earbuds", "headphones", "airpods", "耳机", "蓝牙耳机"],
        "prompts": ["a photo of earphones", "a photo of earbuds", "a photo of headphones"],
    },
    "Student Card": {
        "keywords": ["student card", "campus card", "id card", "access card", "学生卡", "校园卡", "门禁卡", "卡证"],
        "prompts": ["a photo of a student card", "a photo of an ID card", "a photo of a campus card"],
    },
    "Wallet": {
        "keywords": ["wallet", "purse", "card holder", "钱包", "卡包", "钱夹"],
        "prompts": ["a photo of a wallet", "a photo of a purse"],
    },
    "Umbrella": {
        "keywords": ["umbrella", "伞", "雨伞"],
        "prompts": ["a photo of an umbrella"],
    },
    "Bag": {
        "keywords": ["bag", "backpack", "handbag", "tote", "包", "书包", "背包", "手提包"],
        "prompts": ["a photo of a bag", "a photo of a backpack", "a photo of a handbag"],
    },
    "Phone": {
        "keywords": ["phone", "smartphone", "iphone", "android", "手机", "电话"],
        "prompts": ["a photo of a phone", "a photo of a smartphone"],
    },
    "Laptop": {
        "keywords": ["laptop", "notebook computer", "macbook", "笔记本电脑", "电脑", "笔记本"],
        "prompts": ["a photo of a laptop", "a photo of a notebook computer"],
    },
    "Charger": {
        "keywords": ["charger", "charging cable", "adapter", "充电器", "充电线", "数据线", "插头", "转换头"],
        "prompts": ["a photo of a charger", "a photo of a charging cable", "a photo of a power adapter"],
    },
    "Power Bank": {
        "keywords": ["power bank", "portable charger", "充电宝"],
        "prompts": ["a photo of a power bank", "a photo of a portable charger"],
    },
    "USB Drive": {
        "keywords": ["usb", "usb drive", "flash drive", "thumb drive", "u盘", "优盘"],
        "prompts": ["a photo of a usb drive", "a photo of a flash drive"],
    },
    "Notebook": {
        "keywords": ["notebook", "notepad", "journal", "exercise book", "笔记本", "本子", "记事本"],
        "prompts": ["a photo of a notebook", "a photo of a notepad"],
    },
    "Pen": {
        "keywords": ["pen", "pencil", "marker", "ballpoint", "钢笔", "圆珠笔", "笔", "铅笔", "记号笔"],
        "prompts": ["a photo of a pen", "a photo of a pencil"],
    },
    "Glasses": {
        "keywords": ["glasses", "spectacles", "eyeglasses", "眼镜"],
        "prompts": ["a photo of eyeglasses", "a photo of glasses"],
    },
    "Watch": {
        "keywords": ["watch", "smart watch", "apple watch", "手表", "智能手表"],
        "prompts": ["a photo of a watch", "a photo of a smartwatch"],
    },
    "Mouse": {
        "keywords": ["mouse", "wireless mouse", "bluetooth mouse", "computer mouse", "鼠标", "无线鼠标", "蓝牙鼠标"],
        "prompts": ["a photo of a computer mouse", "a photo of a wireless mouse"],
    },
}

COLOR_LABELS = {
    "Black": ["black", "dark black", "黑色"],
    "White": ["white", "off white", "白色"],
    "Blue": ["blue", "navy", "sky blue", "蓝色", "深蓝", "浅蓝"],
    "Red": ["red", "burgundy", "红色"],
    "Green": ["green", "dark green", "绿色"],
    "Pink": ["pink", "rose pink", "粉色"],
    "Purple": ["purple", "violet", "紫色"],
    "Yellow": ["yellow", "mustard", "黄色"],
    "Silver": ["silver", "gray", "grey", "银色", "灰色"],
    "Brown": ["brown", "tan", "棕色", "褐色"],
}

PART_LABELS = {
    "lid": ["lid", "cap", "cover", "top", "杯盖", "盖子", "盖"],
    "case": ["case", "shell", "cover", "外壳", "壳", "保护壳", "保护套", "套子"],
    "strap": ["strap", "lanyard", "string", "belt", "肩带", "背带", "挂绳", "绳子"],
    "handle": ["handle", "grip", "把手", "提手"],
    "button": ["button", "key", "按键", "按钮"],
    "screen": ["screen", "display", "屏幕", "屏"],
    "left": ["left", "left earbud", "左耳", "左耳机", "左边"],
    "right": ["right", "right earbud", "右耳", "右耳机", "右边"],
}

LOCATION_KEYWORDS = {
    "Library": ["library", "图书馆"],
    "Cafeteria": ["cafeteria", "canteen", "食堂"],
    "Classroom": ["classroom", "lecture hall", "教室"],
    "Dormitory": ["dorm", "dormitory", "宿舍"],
    "Lab": ["lab", "laboratory", "实验室"],
    "Sports Centre": ["gym", "sports center", "体育馆", "操场"],
}

TIME_KEYWORDS = [
    "today",
    "yesterday",
    "last night",
    "this morning",
    "this afternoon",
    "tonight",
    "今天",
    "昨天",
    "前天",
    "昨晚",
    "今天上午",
    "今天下午",
    "昨天上午",
    "昨天下午",
]

SPECIAL_NOTE_KEYWORDS = [
    "sticker",
    "label",
    "tag",
    "name tag",
    "keychain",
    "scratch",
    "dent",
    "engraving",
    "吊牌",
    "标签",
    "贴纸",
    "挂件",
    "刻字",
    "划痕",
    "破损",
]

FUZZY_WORDS = [
    "maybe",
    "probably",
    "perhaps",
    "i think",
    "not sure",
    "好像",
    "可能",
    "大概",
    "应该",
    "不太确定",
    "反正",
]

PUNCTUATION_SPLIT_RE = re.compile(r"[,.!?;，。！？；\n]+")
SPACE_RE = re.compile(r"\s+")

_COLOR_KEYWORD_TO_LABEL: dict[str, str] = {}
for _label, _keywords in COLOR_LABELS.items():
    for _keyword in _keywords:
        _COLOR_KEYWORD_TO_LABEL[_keyword.lower()] = _label


@dataclass
class QueryAnalysis:
    """Structured hints extracted from the user's natural-language query."""

    raw_query: str
    normalized_query: str
    reconstructed_query: str
    item_type: str | None = None
    color: str | None = None
    time_hint: str | None = None
    location_hint: str | None = None
    special_notes: list[str] = field(default_factory=list)
    component_colors: dict[str, str] = field(default_factory=dict)
    needs_confirmation: bool = False
    follow_up_question: str | None = None
    follow_up_options: list[str] = field(default_factory=list)
    follow_up_target: str | None = None
    confidence_summary: str = "No analysis has been run yet."


@dataclass(frozen=True)
class _Prediction:
    label: str | None
    score: float = 0.0
    runner_up: str | None = None
    margin: float = 0.0


def analyze_query(
    query: str,
    *,
    lost_location: str | None = None,
    lost_time_hint: str | None = None,
) -> QueryAnalysis:
    """Extract stable query hints without rewriting the user's original sentence."""

    normalized = _normalize_text(query)
    item_prediction = _predict_item_type(normalized)
    color_prediction = _predict_color(normalized)
    time_hint = _clean_optional(lost_time_hint) or _extract_time_hint(normalized)
    location_hint = _clean_optional(lost_location) or _extract_location_hint(normalized)
    special_notes = _extract_special_notes(normalized)
    component_colors = _extract_component_colors(normalized, default_color=color_prediction.label)
    fuzzy_detected = _contains_fuzzy_language(normalized)
    reconstructed_query = build_reconstructed_query(
        item_type=item_prediction.label,
        color=color_prediction.label,
        component_colors=component_colors,
        special_notes=special_notes,
        location_hint=location_hint,
        time_hint=time_hint,
    )

    needs_confirmation = False
    follow_up_question: str | None = None
    follow_up_options: list[str] = []
    follow_up_target: str | None = None

    if item_prediction.label is None:
        needs_confirmation = True
        follow_up_target = "item_type"
        follow_up_question = "Which item category is the best match?"
        follow_up_options = _dedupe_options(
            ["Bottle", "Keys", "Earphones", "Student Card", "Wallet", "Umbrella", "Bag", "Phone", "Laptop", "Mouse"]
        )
    elif item_prediction.margin < 0.12 or fuzzy_detected:
        needs_confirmation = True
        follow_up_target = "item_type"
        follow_up_question = "Please confirm the item category before searching."
        follow_up_options = _dedupe_options(
            [
                item_prediction.label,
                item_prediction.runner_up,
                "Bottle",
                "Keys",
                "Earphones",
                "Student Card",
                "Wallet",
                "Phone",
                "Laptop",
                "Mouse",
            ]
        )
    elif color_prediction.label is None and special_notes:
        needs_confirmation = True
        follow_up_target = "special_notes"
        follow_up_question = "Should the special mark be kept as an important search hint?"
        follow_up_options = ["Yes, keep it", "No, ignore it"]

    confidence_summary = _build_confidence_summary(
        item_prediction=item_prediction,
        color_prediction=color_prediction,
        fuzzy_detected=fuzzy_detected,
    )

    return QueryAnalysis(
        raw_query=query,
        normalized_query=normalized,
        reconstructed_query=reconstructed_query,
        item_type=item_prediction.label,
        color=color_prediction.label,
        time_hint=time_hint,
        location_hint=location_hint,
        special_notes=special_notes,
        component_colors=component_colors,
        needs_confirmation=needs_confirmation,
        follow_up_question=follow_up_question,
        follow_up_options=follow_up_options,
        follow_up_target=follow_up_target,
        confidence_summary=confidence_summary,
    )


def build_reconstructed_query(
    *,
    item_type: str | None,
    color: str | None,
    component_colors: dict[str, str],
    special_notes: list[str],
    location_hint: str | None,
    time_hint: str | None,
) -> str:
    """Reconstruct a stable, compact query string from extracted labels and hints."""

    parts: list[str] = []
    if color:
        parts.append(color.lower())
    if item_type:
        parts.append(item_type.lower())
    for part, part_color in component_colors.items():
        if part_color:
            parts.append(f"{part_color.lower()} {part}".strip())
        else:
            parts.append(part)
    for note in special_notes:
        cleaned = note.strip()
        if cleaned:
            parts.append(cleaned)
    if location_hint and location_hint != "any":
        parts.append(f"near {location_hint.lower()}")
    if time_hint:
        parts.append(time_hint.lower())
    return " ".join(_dedupe_options(parts))


def _predict_item_type(text: str) -> _Prediction:
    keyword_scores = _score_label_catalog(text, {label: data["keywords"] for label, data in ITEM_LABELS.items()})
    clip_scores = _clip_label_scores(text, {label: data["prompts"] for label, data in ITEM_LABELS.items()})
    return _merge_scores(keyword_scores, clip_scores)


def _predict_color(text: str) -> _Prediction:
    keyword_scores = _score_label_catalog(text, COLOR_LABELS)
    return _merge_scores(keyword_scores, {})


def _score_label_catalog(text: str, catalog: dict[str, list[str]]) -> dict[str, float]:
    lowered = text.lower()
    scores: dict[str, float] = {}
    for label, keywords in catalog.items():
        score = 0.0
        for keyword in keywords:
            if keyword.lower() in lowered:
                score += 1.0 if len(keyword) > 2 else 0.4
        if score > 0:
            scores[label] = score
    return _normalize_scores(scores)


def _merge_scores(keyword_scores: dict[str, float], clip_scores: dict[str, float]) -> _Prediction:
    merged: dict[str, float] = {}
    labels = set(keyword_scores) | set(clip_scores)
    for label in labels:
        keyword_score = keyword_scores.get(label, 0.0)
        clip_score = clip_scores.get(label, 0.0)
        if keyword_score and clip_score:
            merged[label] = 0.7 * keyword_score + 0.3 * clip_score
        else:
            merged[label] = keyword_score or clip_score

    if not merged:
        return _Prediction(label=None)

    ranking = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranking[0]
    runner_up_label, runner_up_score = ranking[1] if len(ranking) > 1 else (None, 0.0)
    if top_score < 0.18:
        return _Prediction(label=None, score=top_score, runner_up=runner_up_label, margin=top_score - runner_up_score)
    return _Prediction(
        label=top_label,
        score=top_score,
        runner_up=runner_up_label,
        margin=top_score - runner_up_score,
    )


def _clip_label_scores(text: str, prompts_by_label: dict[str, list[str]]) -> dict[str, float]:
    try:
        text_vector = encode_text(text)
    except Exception:
        return {}

    prompts: list[str] = []
    prompt_labels: list[str] = []
    for label, label_prompts in prompts_by_label.items():
        for prompt in label_prompts:
            prompts.append(prompt)
            prompt_labels.append(label)

    prompt_vectors = []
    for p in prompts:
        try:
            vec = encode_text(p)
            prompt_vectors.append(vec)
        except Exception:
            prompt_vectors.append(np.zeros_like(text_vector.shape))

    per_label: dict[str, list[float]] = {}
    for label, vector in zip(prompt_labels, prompt_vectors, strict=True):
        cosine = float(np.dot(text_vector, vector))
        per_label.setdefault(label, []).append((cosine + 1.0) / 2.0)

    return {label: max(scores) for label, scores in per_label.items()}


def _extract_time_hint(text: str) -> str | None:
    lowered = text.lower()
    for keyword in TIME_KEYWORDS:
        if keyword.lower() in lowered:
            return keyword
    return None


def _extract_location_hint(text: str) -> str | None:
    lowered = text.lower()
    for label, keywords in LOCATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in lowered:
                return label
    return None


def _extract_component_colors(text: str, *, default_color: str | None) -> dict[str, str]:
    lowered = text.lower()
    component_colors: dict[str, str] = {}
    for part, keywords in PART_LABELS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            start = 0
            while True:
                idx = lowered.find(keyword_lower, start)
                if idx == -1:
                    break
                window_left = max(0, idx - 10)
                window_right = min(len(lowered), idx + len(keyword_lower) + 10)
                window = lowered[window_left:window_right]
                color_label = _find_color_label(window) or default_color
                if color_label:
                    component_colors[part] = color_label
                start = idx + len(keyword_lower)
    return component_colors


def _find_color_label(text_window: str) -> str | None:
    for keyword, label in _COLOR_KEYWORD_TO_LABEL.items():
        if keyword in text_window:
            return label
    return None


def _extract_special_notes(text: str) -> list[str]:
    notes: list[str] = []
    for keyword in SPECIAL_NOTE_KEYWORDS:
        pattern = re.compile(
            rf"[^,.!?;，。！？；]{{0,5}}{re.escape(keyword)}[^,.!?;，。！？；]{{0,5}}",
            flags=re.IGNORECASE,
        )
        for match in pattern.findall(text):
            cleaned = match.strip()
            if cleaned:
                notes.append(cleaned)
    for segment in PUNCTUATION_SPLIT_RE.split(text):
        cleaned = segment.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(keyword.lower() in lowered for keyword in SPECIAL_NOTE_KEYWORDS) and len(cleaned) <= 12:
            notes.append(cleaned)
    return _dedupe_options(notes)


def _contains_fuzzy_language(text: str) -> bool:
    lowered = text.lower()
    return any(word.lower() in lowered for word in FUZZY_WORDS)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {}
    return {label: score / max_score for label, score in scores.items()}


def _normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _build_confidence_summary(
    *,
    item_prediction: _Prediction,
    color_prediction: _Prediction,
    fuzzy_detected: bool,
) -> str:
    parts: list[str] = []
    if item_prediction.label:
        parts.append(f"Item type: {item_prediction.label}")
    else:
        parts.append("Item type is still unclear")
    if color_prediction.label:
        parts.append(f"Color: {color_prediction.label}")
    if fuzzy_detected:
        parts.append("The description contains uncertain wording")
    if item_prediction.margin and item_prediction.margin < 0.12:
        parts.append("Top item-type candidates are close, so confirmation is recommended")
    return ". ".join(parts) + "."


def _dedupe_options(options: list[str | None]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for option in options:
        if not option:
            continue
        key = option.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique
