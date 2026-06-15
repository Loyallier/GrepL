"""Small local dataset used while model modules are being integrated."""

from __future__ import annotations

from pathlib import Path

from config.options import LOCATION_OPTIONS, option_keywords
from contracts import MatchResult, SearchQuery, TimePoint
from query_refiner import QueryRefinement


DEMO_ASSET_DIR = Path(__file__).resolve().parent / "demo_assets"
DEMO_ASSET_DIR.mkdir(exist_ok=True)


_FOUND_ITEMS = (
    {
        "item_id": "demo-bottle-blue",
        "image_path": DEMO_ASSET_DIR / "blue_bottle.jpg",
        "found_time": TimePoint(date="2026-06-14", hour=16),
        "found_location": "library",
        "tags": ("blue", "water", "bottle", "stickers", "plastic"),
    },
    {
        "item_id": "demo-umbrella-black",
        "image_path": DEMO_ASSET_DIR / "black_umbrella.jpg",
        "found_time": TimePoint(date="2026-06-13", hour=11),
        "found_location": "cafeteria",
        "tags": ("black", "umbrella", "folding"),
    },
    {
        "item_id": "demo-card",
        "image_path": DEMO_ASSET_DIR / "campus_card.jpg",
        "found_time": TimePoint(date="2026-06-14", hour=9),
        "found_location": "classroom",
        "tags": ("campus", "card", "student", "white"),
    },
    {
        "item_id": "demo-backpack",
        "image_path": DEMO_ASSET_DIR / "gray_backpack.jpg",
        "found_time": TimePoint(date="2026-06-12", hour=18),
        "found_location": "dormitory",
        "tags": ("gray", "backpack", "zipper", "large"),
    },
)


def mock_search_items(
    query: SearchQuery,
    query_refinement: QueryRefinement | None = None,
) -> list[MatchResult]:
    """Return deterministic demo results with simple text, time, and place scores."""

    query_terms = _tokenize(query.description)
    results: list[MatchResult] = []
    for item in _FOUND_ITEMS:
        visual_similarity = _visual_score(query_terms, item["tags"])
        location_match = _location_score(query.lost_location, item["found_location"])
        time_match = _time_score(query, item["found_time"])
        overall = _overall_score(visual_similarity, location_match, time_match)
        reasons = _match_reasons(query_refinement, visual_similarity, location_match, time_match)
        notes = _mismatch_notes(visual_similarity, location_match, time_match)
        results.append(
            MatchResult(
                item_id=str(item["item_id"]),
                image_path=str(item["image_path"]),
                found_time=item["found_time"],
                found_location=str(item["found_location"]),
                visual_similarity=visual_similarity,
                time_match=time_match,
                location_match=location_match,
                overall_match=overall,
                confidence_label=_confidence_label(overall),
                reasons=reasons,
                mismatch_notes=notes,
            )
        )
    results.sort(key=lambda result: result.overall_match, reverse=True)
    return results[: query.result_limit]


def _tokenize(text: str) -> set[str]:
    return {part for part in text.lower().replace("-", " ").split() if len(part) >= 2}


def _visual_score(query_terms: set[str], item_tags: tuple[str, ...]) -> float:
    if not query_terms:
        return 0.1
    tag_set = set(item_tags)
    overlap = len(query_terms & tag_set)
    return min(0.98, 0.28 + 0.18 * overlap)


def _location_score(query_location: str | None, found_location: str | None) -> float | None:
    if not query_location or query_location in {"any", "not_sure"}:
        return None
    if query_location == found_location:
        return 1.0
    query_keywords = set(option_keywords(query_location, LOCATION_OPTIONS))
    found_keywords = set(option_keywords(found_location, LOCATION_OPTIONS))
    if query_keywords & found_keywords:
        return 0.75
    return 0.25


def _time_score(query: SearchQuery, found_time: TimePoint | None) -> float | None:
    time_range = query.lost_time_range
    if time_range is None or found_time is None:
        return None
    selected_dates = {
        point.date
        for point in (time_range.start, time_range.end)
        if point is not None and point.date is not None
    }
    if selected_dates and found_time.date in selected_dates:
        return 0.9
    selected_hours = {
        point.hour
        for point in (time_range.start, time_range.end)
        if point is not None and point.hour is not None
    }
    if selected_hours and found_time.hour is not None:
        closest_gap = min(abs(found_time.hour - hour) for hour in selected_hours)
        return max(0.2, 1.0 - closest_gap / 12)
    return 0.5


def _overall_score(visual: float, location: float | None, time: float | None) -> float:
    weighted_sum = visual * 0.7
    weight = 0.7
    if location is not None:
        weighted_sum += location * 0.2
        weight += 0.2
    if time is not None:
        weighted_sum += time * 0.1
        weight += 0.1
    return round(weighted_sum / weight, 3)


def _match_reasons(
    refinement: QueryRefinement | None,
    visual: float,
    location: float | None,
    time: float | None,
) -> list[str]:
    reasons: list[str] = []
    if refinement is not None and not refinement.used_fallback:
        reasons.append(f'Visual query refined to "{refinement.clip_text}".')
    if visual >= 0.46:
        reasons.append("Visual tags overlap with the refined description.")
    if location is not None and location >= 0.75:
        reasons.append("Found location is close to the selected lost location.")
    if time is not None and time >= 0.75:
        reasons.append("Found time is close to the selected lost time.")
    return reasons or ["Candidate kept for manual visual review."]


def _mismatch_notes(visual: float, location: float | None, time: float | None) -> list[str]:
    notes: list[str] = []
    if visual < 0.46:
        notes.append("Visual overlap is weak in the demo scorer.")
    if location is not None and location < 0.5:
        notes.append("Found location differs from the selected lost location.")
    if time is not None and time < 0.5:
        notes.append("Found time may be outside the selected time range.")
    return notes


def _confidence_label(score: float) -> str:
    if score >= 0.78:
        return "Strong Match"
    if score >= 0.62:
        return "Likely Match"
    if score >= 0.45:
        return "Possible Match"
    return "Weak Match"
