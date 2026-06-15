"""Comprehensive ranking algorithm for GrepL search candidates."""

from __future__ import annotations

from dataclasses import fields
from datetime import date, datetime
from math import isfinite
from typing import Any, Iterable

from config.options import LOCATION_OPTIONS
from contracts import Candidate, MatchResult, SearchQuery, TimePoint, TimeRange


BASE_WEIGHTS = {
    "visual": 0.60,
    "time": 0.20,
    "location": 0.15,
    "bound": 0.05,
}

MIN_VISUAL_SIMILARITY = 0.20
MIN_BOUND_CONFIDENCE = 0.20
MIN_OVERALL_MATCH = 0.15

SPECIAL_LOCATION_KEYS = {"any", "not_sure"}
VALID_LOCATION_KEYS = frozenset(LOCATION_OPTIONS) - SPECIAL_LOCATION_KEYS

SAME_LOCATION_PAIRS = {
    frozenset(("library", "library_entrance")),
}

NEARBY_LOCATION_PAIRS = {
    frozenset(("library", "block_a")),
    frozenset(("block_a", "classroom")),
    frozenset(("library_entrance", "block_a")),
    frozenset(("library_entrance", "classroom")),
}

RELATIVELY_CLOSE_LOCATION_PAIRS = {
    frozenset(("cafeteria", "dormitory")),
    frozenset(("library", "classroom")),
    frozenset(("sports_center", "block_a")),
}


def evaluate_matches(
    candidates: Iterable[Candidate],
    query: SearchQuery | TimeRange | None = None,
    lost_location: str | None = None,
    top_k: int | None = None,
) -> list[MatchResult]:
    """Rank candidates and return UI-ready match results.

    Preferred call:
        evaluate_matches(candidates, query)

    Backward-compatible call:
        evaluate_matches(candidates, lost_time_range, lost_location, top_k)
    """

    lost_time_range, normalized_location, result_limit = _resolve_query(query, lost_location, top_k)
    ranked_results: list[MatchResult] = []

    for candidate in candidates:
        visual_similarity = clamp_score(_field(candidate, "visual_similarity", 0.0))
        bound_confidence = clamp_score(_field(candidate, "bound_confidence", 0.0))

        if visual_similarity < MIN_VISUAL_SIMILARITY or bound_confidence < MIN_BOUND_CONFIDENCE:
            continue

        found_time = _field(candidate, "found_time", None)
        found_location = _field(candidate, "found_location", None)

        time_match = calculate_time_match(lost_time_range, found_time)
        location_match = calculate_location_match(normalized_location, found_location)
        overall_match = calculate_overall_match(
            visual_similarity,
            time_match,
            location_match,
            bound_confidence,
        )

        if overall_match < MIN_OVERALL_MATCH:
            continue

        ranked_results.append(
            _make_match_result(
                candidate,
                found_time,
                found_location,
                visual_similarity,
                time_match,
                location_match,
                overall_match,
                confidence_label(overall_match),
                positive_reasons(
                    visual_similarity,
                    time_match,
                    location_match,
                    bound_confidence,
                ),
                mismatch_notes(
                    visual_similarity,
                    time_match,
                    location_match,
                    bound_confidence,
                    lost_time_range,
                    normalized_location,
                    found_time,
                    found_location,
                ),
            )
        )

    return rank_and_select_top_n(ranked_results, result_limit)


def rank_and_select_top_n(results: Iterable[MatchResult], top_n: int | None = None) -> list[MatchResult]:
    """Sort match results by score descending and return only the top N items."""

    ranked_results = sorted(
        results,
        key=lambda result: (-result.overall_match, -result.visual_similarity, result.item_id),
    )
    if top_n is None:
        return ranked_results
    return ranked_results[:top_n]


def calculate_time_match(lost_time_range: TimeRange | None, found_time: TimePoint | None) -> float | None:
    """Calculate the time consistency score, or None when time is not usable."""

    if not has_time_range(lost_time_range) or not has_time_point(found_time):
        return None

    found_datetime = _time_point_to_datetime(found_time)
    range_start_datetime, range_end_datetime = _time_range_to_datetimes(lost_time_range)
    if found_datetime is not None and (range_start_datetime is not None or range_end_datetime is not None):
        start_datetime = range_start_datetime or range_end_datetime
        end_datetime = range_end_datetime or range_start_datetime
        if start_datetime is not None and end_datetime is not None:
            return _score_datetime_distance(found_datetime, start_datetime, end_datetime)

    found_date = _parse_date(found_time.date)
    range_start_date, range_end_date = _time_range_to_dates(lost_time_range)
    if found_date is not None and (range_start_date is not None or range_end_date is not None):
        start_date = range_start_date or range_end_date
        end_date = range_end_date or range_start_date
        if start_date is not None and end_date is not None:
            return _score_date_distance(found_date, start_date, end_date)

    found_hour = _clean_hour(found_time.hour)
    range_start_hour, range_end_hour = _time_range_to_hours(lost_time_range)
    if found_hour is not None and (range_start_hour is not None or range_end_hour is not None):
        start_hour = range_start_hour if range_start_hour is not None else range_end_hour
        end_hour = range_end_hour if range_end_hour is not None else range_start_hour
        if start_hour is not None and end_hour is not None:
            return _score_hour_distance(found_hour, start_hour, end_hour)

    return None


def calculate_location_match(lost_location: str | None, found_location: str | None) -> float | None:
    """Calculate the campus location consistency score, or None when unused."""

    lost_key = _clean_location_key(lost_location)
    found_key = _clean_location_key(found_location)
    if not _is_scored_location(lost_key) or not _is_scored_location(found_key):
        return None

    if lost_key == found_key:
        return 1.0

    location_pair = frozenset((lost_key, found_key))
    if location_pair in SAME_LOCATION_PAIRS:
        return 1.0
    if location_pair in NEARBY_LOCATION_PAIRS:
        return 0.75
    if location_pair in RELATIVELY_CLOSE_LOCATION_PAIRS:
        return 0.5
    return 0.1


def calculate_overall_match(
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
    bound_confidence: float,
) -> float:
    """Calculate the weighted final score with dynamic weight transfer."""

    weights = adjusted_weights(time_match, location_match)
    score = (
        clamp_score(visual_similarity) * weights["visual"]
        + (0.0 if time_match is None else clamp_score(time_match) * weights["time"])
        + (0.0 if location_match is None else clamp_score(location_match) * weights["location"])
        + clamp_score(bound_confidence) * weights["bound"]
    )
    return round(clamp_score(score), 4)


def adjusted_weights(time_match: float | None, location_match: float | None) -> dict[str, float]:
    """Return scoring weights after moving unused time/location weight to visual."""

    weights = dict(BASE_WEIGHTS)
    if time_match is None:
        weights["visual"] += weights["time"]
        weights["time"] = 0.0
    if location_match is None:
        weights["visual"] += weights["location"]
        weights["location"] = 0.0
    return weights


def confidence_label(overall_match: float) -> str:
    if overall_match >= 0.80:
        return "High"
    if overall_match >= 0.55:
        return "Medium"
    return "Low"


def positive_reasons(
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
    bound_confidence: float,
) -> list[str]:
    reasons: list[str] = []
    if visual_similarity >= 0.80:
        reasons.append("The item has high visual similarity with the user description.")
    elif visual_similarity >= 0.60:
        reasons.append("The item has moderate visual similarity with the user description.")

    if time_match is not None and time_match >= 0.80:
        reasons.append("The item's found time is close to the selected lost time.")

    if location_match == 1.0:
        reasons.append("The item's found location matches the selected lost location.")
    elif location_match == 0.75:
        reasons.append("The item's found location is near the selected lost location.")
    elif location_match == 0.5:
        reasons.append("The item's found location is relatively close to the selected lost location.")

    return reasons


def mismatch_notes(
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
    bound_confidence: float,
    lost_time_range: TimeRange | None,
    lost_location: str | None,
    found_time: TimePoint | None,
    found_location: str | None,
) -> list[str]:
    notes: list[str] = []
    user_provided_time = has_time_range(lost_time_range)
    user_provided_location = _is_scored_location(_clean_location_key(lost_location))

    if visual_similarity < 0.60:
        notes.append("The item is not highly similar to the user description.")

    if user_provided_time:
        if found_time_is_before_lost_range(lost_time_range, found_time):
            notes.append("The item was found before the selected lost time.")
        elif time_match is not None and time_match < 0.40:
            notes.append("The item's found time is far from the selected lost time.")
        elif not has_time_point(found_time):
            notes.append("The item does not have found-time information.")

    if user_provided_location:
        if location_match == 0.10:
            notes.append("The item's found location is different from the selected lost location.")
        elif not _is_scored_location(_clean_location_key(found_location)):
            notes.append("The item does not have found-location information.")

    if bound_confidence < 0.50:
        notes.append("The detected item region may be inaccurate.")

    return notes


def has_time_range(time_range: TimeRange | None) -> bool:
    if time_range is None:
        return False
    return has_time_point(time_range.start) or has_time_point(time_range.end)


def has_time_point(time_point: TimePoint | None) -> bool:
    if time_point is None:
        return False
    return _parse_date(time_point.date) is not None or _clean_hour(time_point.hour) is not None


def found_time_is_before_lost_range(
    lost_time_range: TimeRange | None,
    found_time: TimePoint | None,
) -> bool:
    """Return True when found time clearly predates the selected lost time."""

    if not has_time_range(lost_time_range) or not has_time_point(found_time):
        return False

    found_datetime = _time_point_to_datetime(found_time)
    range_start_datetime, range_end_datetime = _time_range_to_datetimes(lost_time_range)
    first_lost_datetime = _earliest_not_none(range_start_datetime, range_end_datetime)
    if found_datetime is not None and first_lost_datetime is not None:
        return found_datetime < first_lost_datetime

    found_date = _parse_date(found_time.date)
    range_start_date, range_end_date = _time_range_to_dates(lost_time_range)
    first_lost_date = _earliest_not_none(range_start_date, range_end_date)
    if found_date is not None and first_lost_date is not None:
        return found_date < first_lost_date

    return False


def clamp_score(value: Any) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not isfinite(numeric_value):
        return 0.0
    return max(0.0, min(1.0, numeric_value))


def _resolve_query(
    query: SearchQuery | TimeRange | None,
    lost_location: str | None,
    top_k: int | None,
) -> tuple[TimeRange | None, str | None, int | None]:
    if isinstance(query, SearchQuery):
        result_limit = top_k if top_k is not None else query.result_limit
        return query.lost_time_range, query.lost_location, _clean_result_limit(result_limit)
    return query, lost_location, _clean_result_limit(top_k)


def _clean_result_limit(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _field(candidate: Any, name: str, default: Any) -> Any:
    if isinstance(candidate, dict):
        return candidate.get(name, default)
    return getattr(candidate, name, default)


def _make_match_result(
    candidate: Any,
    found_time: TimePoint | None,
    found_location: str | None,
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
    overall_match: float,
    label: str,
    reasons: list[str],
    notes: list[str],
) -> MatchResult:
    values = {
        "item_id": str(_field(candidate, "item_id", "")),
        "title": str(_field(candidate, "title", "Untitled item")),
        "image_path": str(_field(candidate, "image_path", "")),
        "found_time": found_time,
        "found_location": found_location,
        "visual_similarity": visual_similarity,
        "time_match": time_match,
        "location_match": location_match,
        "overall_match": overall_match,
        "confidence_label": label,
        "reasons": reasons,
        "mismatch_notes": notes,
    }
    result_fields = {field.name for field in fields(MatchResult)}
    return MatchResult(**{key: value for key, value in values.items() if key in result_fields})


def _earliest_not_none(first: Any, second: Any) -> Any:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _score_datetime_distance(found: datetime, start: datetime, end: datetime) -> float:
    if start > end:
        start, end = end, start
    if start <= found <= end:
        return 1.0

    if found < start:
        hours_before_start = (start - found).total_seconds() / 3600
        return 0.6 if hours_before_start <= 1 else 0.1

    hours_after_end = (found - end).total_seconds() / 3600
    if hours_after_end <= 1:
        return 0.9
    if hours_after_end <= 3:
        return 0.8
    if found.date() == end.date():
        return 0.65

    date_gap = (found.date() - end.date()).days
    if date_gap <= 1:
        return 0.6
    if date_gap <= 3:
        return 0.3
    return 0.1


def _score_date_distance(found: date, start: date, end: date) -> float:
    if start > end:
        start, end = end, start
    if start <= found <= end:
        return 1.0

    day_gap = (start - found).days if found < start else (found - end).days
    if day_gap == 1:
        return 0.6
    if 2 <= day_gap <= 3:
        return 0.3
    return 0.1


def _score_hour_distance(found: int, start: int, end: int) -> float:
    if _hour_in_range(found, start, end):
        return 1.0
    hour_gap = min(_hour_gap(found, start), _hour_gap(found, end))
    if hour_gap == 1:
        return 0.8
    if 2 <= hour_gap <= 3:
        return 0.6
    return 0.3


def _hour_in_range(found: int, start: int, end: int) -> bool:
    if start <= end:
        return start <= found <= end
    return found >= start or found <= end


def _hour_gap(first: int, second: int) -> int:
    direct_gap = abs(first - second)
    return min(direct_gap, 24 - direct_gap)


def _time_range_to_datetimes(time_range: TimeRange | None) -> tuple[datetime | None, datetime | None]:
    if time_range is None:
        return None, None
    return _time_point_to_datetime(time_range.start), _time_point_to_datetime(time_range.end)


def _time_range_to_dates(time_range: TimeRange | None) -> tuple[date | None, date | None]:
    if time_range is None:
        return None, None
    start_date = _parse_date(time_range.start.date) if time_range.start is not None else None
    end_date = _parse_date(time_range.end.date) if time_range.end is not None else None
    return start_date, end_date


def _time_range_to_hours(time_range: TimeRange | None) -> tuple[int | None, int | None]:
    if time_range is None:
        return None, None
    start_hour = _clean_hour(time_range.start.hour) if time_range.start is not None else None
    end_hour = _clean_hour(time_range.end.hour) if time_range.end is not None else None
    return start_hour, end_hour


def _time_point_to_datetime(time_point: TimePoint | None) -> datetime | None:
    if time_point is None:
        return None
    point_date = _parse_date(time_point.date)
    point_hour = _clean_hour(time_point.hour)
    if point_date is None or point_hour is None:
        return None
    return datetime(point_date.year, point_date.month, point_date.day, point_hour)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _clean_hour(value: int | str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        hour = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= hour <= 23:
        return hour
    return None


def _clean_location_key(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned_value = str(value).strip()
    return cleaned_value or None


def _is_scored_location(location_key: str | None) -> bool:
    return location_key in VALID_LOCATION_KEYS


__all__ = [
    "adjusted_weights",
    "calculate_location_match",
    "calculate_overall_match",
    "calculate_time_match",
    "clamp_score",
    "confidence_label",
    "evaluate_matches",
    "found_time_is_before_lost_range",
    "has_time_point",
    "has_time_range",
    "mismatch_notes",
    "positive_reasons",
    "rank_and_select_top_n",
]
