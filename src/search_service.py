"""Stable bridge between the browser UI and the matching backend."""

from __future__ import annotations

import importlib
from typing import Callable, Iterable

from config.options import LOCATION_OPTIONS, SelectOption
from contracts import Candidate, LostItem, MatchResult, SearchQuery
from ranker import evaluate_matches


DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 10

EmbeddingMatcher = Callable[[str, Iterable[LostItem]], Iterable[Candidate]]


def search_items(query: SearchQuery) -> list[MatchResult]:
    """Search registered found items and return ranked candidate matches."""

    if not query.description.strip() and not (query.search_text or "").strip():
        return [] 

    normalized_query = SearchQuery(
        description=query.description.strip(),
        search_text=_clean_optional(query.search_text),
        lost_time_range=query.lost_time_range,
        lost_location=_clean_option(query.lost_location, LOCATION_OPTIONS),
        result_limit=_clean_result_limit(query.result_limit),
        item_type_hint=_clean_optional(query.item_type_hint),
        color_hint=_clean_optional(query.color_hint),
        special_notes=[note.strip() for note in query.special_notes if note.strip()],
        component_color_hints={
            key.strip(): value.strip()
            for key, value in (query.component_color_hints or {}).items()
            if key.strip() and value.strip()
        },
    )

    registered_items = _load_registered_items()
    if not registered_items:
        return []

    matcher = _load_embedding_matcher() 
    if matcher is None:
        return []

    search_text = normalized_query.search_text or normalized_query.description
    candidates = _match_text_to_images(matcher, search_text, registered_items)
    return evaluate_matches(candidates, normalized_query)


def _load_registered_items() -> list[LostItem]:
    registration_service = _optional_module("registration_service")
    if registration_service is None:
        return []

    loader = getattr(registration_service, "load_registered_items", None)
    if not callable(loader):
        return []
    return list(loader()) 


def _load_embedding_matcher() -> EmbeddingMatcher | None:
    embedding_engine = _optional_module("embedding_engine")
    if embedding_engine is None:
        return None

    matcher = getattr(embedding_engine, "match_text_to_images", None)
    if not callable(matcher):
        return None
    return matcher


def _match_text_to_images(
    matcher: EmbeddingMatcher,
    description: str,
    registered_items: Iterable[LostItem],
) -> list[Candidate]:
    return list(matcher(description, registered_items))


def _clean_option(value: str | None, options: dict[str, SelectOption]) -> str:
    if value is None:
        return "any"
    cleaned = value.strip()
    return cleaned if cleaned in options else "any"


def _clean_result_limit(value: int | str | None) -> int:
    try:
        return max(1, min(int(value), MAX_RESULT_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_RESULT_LIMIT


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _optional_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            return None
        raise


