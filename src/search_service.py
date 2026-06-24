"""Stable bridge between the browser UI and the matching backend."""

from __future__ import annotations

import importlib
from typing import Callable, Iterable

from config.options import LOCATION_OPTIONS, SelectOption
from contracts import Candidate, FollowUpQuestion, LostItem, SearchQuery, SearchResponse
from ranker import evaluate_matches


DEFAULT_RESULT_LIMIT = 5
MAX_RESULT_LIMIT = 10

EmbeddingMatcher = Callable[[str, Iterable[LostItem]], Iterable[Candidate]]
_SPECIAL_NOTES_IGNORE_SENTINEL = "__IGNORE__"
_NONE_OF_ABOVE_OPTION = "None of the above"


try:
    from query_understanding import analyze_query, build_reconstructed_query
except ModuleNotFoundError:
    analyze_query = None
    build_reconstructed_query = None


def search_items(query: SearchQuery) -> SearchResponse:
    """Search registered found items and return ranked candidate matches."""

    if not query.description.strip() and not (query.search_text or "").strip():
        return SearchResponse(results=[])

    normalized_query = SearchQuery(
        description=query.description.strip(),
        search_text=_clean_optional(query.search_text),
        use_original_query=bool(getattr(query, "use_original_query", False)),
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

    resolved = _resolve_query_understanding(normalized_query)
    if resolved.follow_up is not None:
        return SearchResponse(results=[], follow_up=resolved.follow_up)

    registered_items = _load_registered_items()
    if not registered_items:
        return SearchResponse(results=[])

    matcher = _load_embedding_matcher() 
    if matcher is None:
        return SearchResponse(results=[])

    search_text = resolved.search_text
    candidates = _match_text_to_images(matcher, search_text, registered_items)
    results = evaluate_matches(candidates, resolved.query_for_ranking)
    return SearchResponse(results=results)


class _ResolvedQuery:
    def __init__(self, *, query_for_ranking: SearchQuery, search_text: str, follow_up: FollowUpQuestion | None):
        self.query_for_ranking = query_for_ranking
        self.search_text = search_text
        self.follow_up = follow_up


def _resolve_query_understanding(query: SearchQuery) -> _ResolvedQuery:
    if getattr(query, "use_original_query", False):
        search_text = query.description.strip() or query.search_text or ""
        query_for_ranking = SearchQuery(
            description=query.description,
            search_text=None,
            use_original_query=True,
            lost_time_range=query.lost_time_range,
            lost_location=query.lost_location,
            result_limit=query.result_limit,
        )
        return _ResolvedQuery(query_for_ranking=query_for_ranking, search_text=search_text, follow_up=None)

    if analyze_query is None or build_reconstructed_query is None:
        search_text = query.search_text or query.description
        return _ResolvedQuery(query_for_ranking=query, search_text=search_text, follow_up=None)

    lost_location_hint = None if query.lost_location in {"any", "not_sure", None} else query.lost_location
    analysis = analyze_query(query.description, lost_location=lost_location_hint)

    if analysis.needs_confirmation and analysis.follow_up_question and analysis.follow_up_target == "item_type":
        if query.item_type_hint is None:
            return _ResolvedQuery(
                query_for_ranking=query,
                search_text=query.search_text or query.description,
                follow_up=FollowUpQuestion(
                    target="item_type_hint",
                    question=analysis.follow_up_question,
                    options=_append_none_of_above(analysis.follow_up_options),
                    multi_select=False,
                ),
            )

    if analysis.needs_confirmation and analysis.follow_up_question and analysis.follow_up_target == "special_notes":
        if not query.special_notes:
            return _ResolvedQuery(
                query_for_ranking=query,
                search_text=query.search_text or query.description,
                follow_up=FollowUpQuestion(
                    target="special_notes",
                    question=analysis.follow_up_question,
                    options=list(analysis.follow_up_options),
                    multi_select=False,
                ),
            )

    item_type = query.item_type_hint or analysis.item_type
    color = query.color_hint or analysis.color

    special_notes = query.special_notes
    ignore_special_notes = _SPECIAL_NOTES_IGNORE_SENTINEL in special_notes
    if ignore_special_notes:
        special_notes = []
    elif not special_notes:
        special_notes = analysis.special_notes

    component_colors = query.component_color_hints or analysis.component_colors

    reconstructed_query = build_reconstructed_query(
        item_type=item_type,
        color=color,
        component_colors=component_colors,
        special_notes=special_notes,
        location_hint=analysis.location_hint or query.lost_location,
        time_hint=analysis.time_hint,
    )
    search_text = reconstructed_query.strip() or query.search_text or query.description

    query_for_ranking = SearchQuery(
        description=query.description,
        search_text=search_text,
        use_original_query=False,
        lost_time_range=query.lost_time_range,
        lost_location=query.lost_location,
        result_limit=query.result_limit,
        item_type_hint=item_type,
        color_hint=color,
        special_notes=special_notes,
        component_color_hints=component_colors,
    )
    return _ResolvedQuery(query_for_ranking=query_for_ranking, search_text=search_text, follow_up=None)


def _append_none_of_above(options: list[str]) -> list[str]:
    cleaned: list[str] = []
    for option in options:
        candidate = option.strip()
        if candidate and candidate not in cleaned:
            cleaned.append(candidate)
    if _NONE_OF_ABOVE_OPTION not in cleaned:
        cleaned.append(_NONE_OF_ABOVE_OPTION)
    return cleaned


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


