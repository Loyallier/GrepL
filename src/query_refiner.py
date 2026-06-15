"""Lightweight English query refinement before CLIP text embedding.

The refiner keeps visual English keywords from a noisy lost-item description and
rewrites them into short phrases that are easier for CLIP-like models to compare
with images.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryRefinement:
    """Result of cleaning a user description for CLIP retrieval."""

    original_text: str
    clip_text: str
    confidence: float
    extracted_terms: tuple[str, ...]
    used_fallback: bool = False


_COLOR_TERMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("blue",), "blue"),
    (("red",), "red"),
    (("black",), "black"),
    (("white",), "white"),
    (("green",), "green"),
    (("yellow",), "yellow"),
    (("pink",), "pink"),
    (("purple",), "purple"),
    (("gray", "grey"), "gray"),
    (("orange",), "orange"),
    (("silver",), "silver"),
    (("gold", "golden"), "gold"),
    (("transparent", "clear"), "transparent"),
)

_OBJECT_TERMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("water bottle", "bottle", "cup", "tumbler"), "water bottle"),
    (("backpack", "schoolbag"), "backpack"),
    (("handbag", "tote", "bag"), "bag"),
    (("umbrella",), "umbrella"),
    (("phone", "iphone", "smartphone"), "phone"),
    (("airpods", "earbuds", "headphones", "earphones"), "earphones"),
    (("keys", "key"), "keys"),
    (("campus card", "student card", "meal card", "id card", "card"), "campus card"),
    (("glasses", "eyeglasses"), "glasses"),
    (("wallet", "purse"), "wallet"),
    (("laptop", "notebook"), "laptop"),
    (("ipad", "tablet"), "tablet"),
    (("jacket", "coat", "hoodie"), "jacket"),
    (("cap", "hat"), "hat"),
    (("pencil", "pen"), "pen"),
    (("textbook", "book"), "book"),
)

_FEATURE_TERMS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("sticker", "stickers"), "with stickers"),
    (("striped", "stripe"), "striped"),
    (("plaid", "checkered"), "plaid"),
    (("pattern", "patterned"), "patterned"),
    (("cartoon",), "cartoon print"),
    (("metal", "metallic"), "metal"),
    (("plastic",), "plastic"),
    (("glass",), "glass"),
    (("leather",), "leather"),
    (("zipper",), "with zipper"),
    (("damaged", "broken"), "damaged"),
    (("old",), "old"),
    (("new",), "new"),
    (("large", "big"), "large"),
    (("small",), "small"),
)

_NOISE_PATTERNS: tuple[str, ...] = (
    r"\b(yesterday|today|tomorrow|near|around|maybe|probably|possibly|not sure|kind of|sort of)\b",
    r"\b(library|cafeteria|dormitory|classroom|gate|entrance|campus|building|sports center)\b",
)

_ENGLISH_VISUAL_WORD = re.compile(r"\b[a-z][a-z0-9-]{2,}\b")
_SPACES = re.compile(r"\s+")
_STOP_WORDS = {
    "and",
    "are",
    "but",
    "for",
    "have",
    "has",
    "lost",
    "maybe",
    "probably",
    "the",
    "this",
    "that",
    "was",
    "with",
}


def refine_query_for_clip(description: str, *, min_confidence: float = 0.34) -> QueryRefinement:
    """Return a CLIP-friendly short text and fall back when extraction is weak."""

    original = description.strip()
    if not original:
        return QueryRefinement("", "", 0.0, (), used_fallback=True)

    normalized = _normalize_text(original)
    colors = _collect_terms(normalized, _COLOR_TERMS)
    objects = _collect_terms(normalized, _OBJECT_TERMS)
    features = _collect_terms(normalized, _FEATURE_TERMS)
    extra_words = _collect_extra_english_words(normalized, colors + objects + features)

    terms = _dedupe((*colors, *features, *objects, *extra_words))
    confidence = _estimate_confidence(colors, objects, features, extra_words, normalized)
    if not terms or confidence < min_confidence:
        return QueryRefinement(original, original, confidence, (), used_fallback=True)

    clip_text = _build_clip_phrase(colors, features, objects, extra_words)
    return QueryRefinement(original, clip_text, confidence, terms, used_fallback=False)


def _normalize_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[，。！？、；：,.!?;:()（）\[\]{}]", " ", normalized)
    for pattern in _NOISE_PATTERNS:
        normalized = re.sub(pattern, " ", normalized)
    return _SPACES.sub(" ", normalized).strip()


def _collect_terms(text: str, table: tuple[tuple[tuple[str, ...], str], ...]) -> tuple[str, ...]:
    found: list[str] = []
    for aliases, canonical in table:
        if any(_contains_english_term(text, alias) for alias in aliases):
            found.append(canonical)
    return tuple(found)


def _contains_english_term(text: str, term: str) -> bool:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return re.search(rf"\b{escaped}\b", text) is not None


def _collect_extra_english_words(text: str, known_terms: tuple[str, ...]) -> tuple[str, ...]:
    known_words = {word for term in known_terms for word in term.split()}
    blocked = {
        "found",
        "item",
        "library",
        "cafeteria",
        "dormitory",
        "classroom",
        "yesterday",
        "today",
        "tomorrow",
    }
    blocked.update(_STOP_WORDS)
    words: list[str] = []
    for word in _ENGLISH_VISUAL_WORD.findall(text):
        if word in known_words or word in blocked:
            continue
        words.append(word)
    return tuple(words[:4])


def _estimate_confidence(
    colors: tuple[str, ...],
    objects: tuple[str, ...],
    features: tuple[str, ...],
    extra_words: tuple[str, ...],
    normalized_text: str,
) -> float:
    score = 0.0
    if objects:
        score += 0.45
    if colors:
        score += 0.25
    if features:
        score += 0.2
    if extra_words:
        score += min(0.1, 0.03 * len(extra_words))
    if len(normalized_text) <= 80:
        score += 0.05
    return min(score, 1.0)


def _build_clip_phrase(
    colors: tuple[str, ...],
    features: tuple[str, ...],
    objects: tuple[str, ...],
    extra_words: tuple[str, ...],
) -> str:
    object_text = objects[0] if objects else "object"
    descriptors = _dedupe((*colors, *features, *extra_words))
    before_object = [term for term in descriptors if not term.startswith("with ")]
    after_object = [term for term in descriptors if term.startswith("with ")]
    parts = [*before_object, object_text, *after_object]
    return " ".join(parts).strip()


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
