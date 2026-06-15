"""Stable bridge between the browser UI and the matching backend."""

from __future__ import annotations

from contracts import MatchResult, SearchQuery
from config.options import LOCATION_OPTIONS, SelectOption
from query_refiner import refine_query_for_clip

from mock_data import mock_search_items


def search_items(query: SearchQuery) -> list[MatchResult]:
    """Search found items for a user query.

    Current implementation returns mock data so the UI can be developed and
    demonstrated before the TensorFlow and ranking modules are complete.

    Future integration point:
        refined_query = query_refiner.refine_query_for_clip(query.description)
        items = database.load_items()
        items_with_similarity = embedding_engine.match_text_to_images(refined_query.clip_text, items)
        results = ranker.evaluate_matches(
            items_with_similarity,
            query.lost_time_range,
            query.lost_location,
            top_k=query.result_limit,
        )
    """

    if not query.description.strip():
        return []
    normalized_query = SearchQuery(
        description=query.description.strip(),
        lost_time_range=query.lost_time_range,
        lost_location=_clean_option(query.lost_location, LOCATION_OPTIONS),
        result_limit=max(1, min(int(query.result_limit), 10)),
    )
    query_refinement = refine_query_for_clip(normalized_query.description)
    clip_query = SearchQuery(
        description=query_refinement.clip_text,
        lost_time_range=normalized_query.lost_time_range,
        lost_location=normalized_query.lost_location,
        result_limit=normalized_query.result_limit,
    )
    return mock_search_items(clip_query, query_refinement=query_refinement)


def _clean_option(value: str | None, options: dict[str, SelectOption]) -> str:
    if value is None:
        return "any"
    cleaned = value.strip()
    return cleaned if cleaned in options else "any"
