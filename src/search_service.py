"""Stable bridge between the browser UI and the matching backend."""

from __future__ import annotations

from contracts import MatchResult, SearchQuery
from config.options import LOCATION_OPTIONS, SelectOption

from mock_data import mock_search_items # 导入 mock，方便在没有算法时，前端依然可以单独运行测试（实现后删除）


def search_items(query: SearchQuery) -> list[MatchResult]:
    """Search found items for a user query.

    Current implementation returns mock data so the UI can be developed and
    demonstrated before the TensorFlow and ranking modules are complete.

    Future integration point:
        items = database.load_items()
        items_with_similarity = embedding_engine.match_text_to_images(query.description, items)
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
    return mock_search_items(normalized_query)


def _clean_option(value: str | None, options: dict[str, SelectOption]) -> str:
    if value is None:
        return "any"
    cleaned = value.strip()
    return cleaned if cleaned in options else "any"
