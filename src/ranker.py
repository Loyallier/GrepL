"""
Comprehensive ranking algorithm for GrepL search candidates.
GrepL 搜索候选项的综合排序算法。
"""

from __future__ import annotations

from collections import deque
from dataclasses import fields
from datetime import date, datetime
from math import isfinite
from typing import Any, Iterable

from config.options import LOCATION_OPTIONS
from contracts import Candidate, MatchResult, SearchQuery, TimePoint, TimeRange


# Default factor weights used when both time and location evidence are available.
# 当时间和地点证据都可用时使用的默认因子权重。
BASE_WEIGHTS = {
    "visual": 0.65,
    "time": 0.20,
    "location": 0.15,
}

# Low-quality candidate filtering thresholds.
# 低质量候选项的过滤阈值。
MIN_VISUAL_SIMILARITY = 0.20
MIN_OVERALL_MATCH = 0.15

# Location keys that mean the user did not provide usable location evidence.
# 表示用户没有提供可用地点证据的地点键。
SPECIAL_LOCATION_KEYS = {"any", "not_sure"}

# Location keys that can participate in location scoring.
# 可以参与地点评分的地点键。
VALID_LOCATION_KEYS = frozenset(LOCATION_OPTIONS) - SPECIAL_LOCATION_KEYS

# Campus location graph edges used to estimate spatial closeness by hop count.
# 用于通过跳数估计空间接近程度的校园地点图边。
LOCATION_EDGES = (
    ("a1", "a2"),
    ("a2", "library"),
    ("library", "a4"),
    ("library", "a3_classroom"),
    ("a4", "a5"),
    ("a5", "playground"),
    ("playground", "b1"),
    ("b1", "d_dormitory"),
    ("d_dormitory", "d6_cafeteria"),
    ("d_dormitory", "ly_dormitory"),
    ("ly_dormitory", "ly3_cafeteria"),
    ("ly3_cafeteria", "music_island"),
    ("music_island", "library"),
    ("music_island", "a2"),
)

# Location scores mapped from shortest-path hop counts.
# 由最短路径跳数映射得到的地点分数。
LOCATION_HOP_SCORES = {
    0: 1.00,
    1: 0.85,
    2: 0.70,
}

# Minimum location score for valid locations that are far apart or disconnected.
# 有效地点相距较远或不连通时使用的最低地点分数。
LOCATION_MIN_SCORE = 0.40

# Complete date-and-hour time scores for found time versus the selected lost range.
# 拾获时间与所选丢失时间范围进行完整日期和小时比较时的时间分数。
TIME_SCORE_WITHIN_RANGE = 1.00
TIME_SCORE_LATER_WITHIN_1_HOUR = 0.95
TIME_SCORE_LATER_WITHIN_3_HOURS = 0.85
TIME_SCORE_SAME_DAY_FAR = 0.75
TIME_SCORE_1_DAY_AWAY = 0.65
TIME_SCORE_2_TO_3_DAYS_AWAY = 0.50
TIME_SCORE_MORE_THAN_3_DAYS_AWAY = 0.35
TIME_SCORE_EARLIER_WITHIN_1_HOUR = 0.75
TIME_SCORE_EARLIER_MORE_THAN_1_HOUR = 0.35

# Date-only time scores used when at least one side lacks an hour value.
# 当至少一侧缺少小时值时使用的仅日期时间分数。
DATE_SCORE_SAME_DATE = 1.00
DATE_SCORE_1_DAY_AWAY = 0.75
DATE_SCORE_2_TO_3_DAYS_AWAY = 0.55
DATE_SCORE_MORE_THAN_3_DAYS_AWAY = 0.35

# Hour-only time scores used when date information is unavailable.
# 当日期信息不可用时使用的仅小时时间分数。
HOUR_SCORE_SAME_HOUR = 1.00
HOUR_SCORE_1_HOUR_AWAY = 0.90
HOUR_SCORE_2_TO_3_HOURS_AWAY = 0.75
HOUR_SCORE_MORE_THAN_3_HOURS_AWAY = 0.55


def _build_location_graph(edges: Iterable[tuple[str, str]]) -> dict[str, set[str]]:
    """
    Build an undirected campus location graph.
    构建一个无向的校园地点图。
    """

    graph: dict[str, set[str]] = {}
    for first, second in edges:
        graph.setdefault(first, set()).add(second)
        graph.setdefault(second, set()).add(first)
    return graph


LOCATION_GRAPH = _build_location_graph(LOCATION_EDGES)


def evaluate_matches(
    candidates: Iterable[Candidate],
    query: SearchQuery | TimeRange | None = None,
    lost_location: str | None = None,
    top_k: int | None = None,
) -> list[MatchResult]:
    """
    Rank candidates and return UI-ready match results.
    对候选项排序并返回可直接供界面使用的匹配结果。
    """

    lost_time_range, normalized_location, result_limit = _resolve_query(query, lost_location, top_k)
    ranked_results: list[MatchResult] = []

    for candidate in candidates:
        visual_similarity = clamp_score(_field(candidate, "visual_similarity", 0.0))

        if visual_similarity < MIN_VISUAL_SIMILARITY:
            continue

        found_time = _field(candidate, "found_time", None)
        found_location = _field(candidate, "found_location", None)

        time_match = calculate_time_match(lost_time_range, found_time)
        location_match = calculate_location_match(normalized_location, found_location)
        overall_match = calculate_overall_match(
            visual_similarity,
            time_match,
            location_match,
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
                ),
                mismatch_notes(
                    visual_similarity,
                    time_match,
                    location_match,
                    lost_time_range,
                    normalized_location,
                    found_time,
                    found_location,
                ),
            )
        )

    return rank_and_select_top_n(ranked_results, result_limit)


def rank_and_select_top_n(results: Iterable[MatchResult], top_n: int | None = None) -> list[MatchResult]:
    """
    Sort match results by score descending and return only the top N items.
    按分数从高到低排序匹配结果，并只返回前 N 个结果。
    """

    ranked_results = sorted(
        results,
        key=lambda result: (-result.overall_match, -result.visual_similarity, result.item_id),
    )
    if top_n is None:
        return ranked_results
    return ranked_results[:top_n]


def calculate_time_match(lost_time_range: TimeRange | None, found_time: TimePoint | None) -> float | None:
    """
    Calculate the time consistency score, or None when time is not usable.
    计算时间一致性分数；当时间不可用时返回 None。
    """

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
    """
    Calculate the campus location consistency score by shortest-path hop count.
    通过最短路径跳数计算校园地点一致性分数。
    """

    lost_key = _clean_location_key(lost_location)
    found_key = _clean_location_key(found_location)
    if not _is_scored_location(lost_key) or not _is_scored_location(found_key):
        return None

    hops = _location_hops(lost_key, found_key)
    if hops is None:
        return LOCATION_MIN_SCORE

    return LOCATION_HOP_SCORES.get(hops, LOCATION_MIN_SCORE)


def _location_hops(start: str, target: str) -> int | None:
    """
    Return the shortest hop count between two valid locations.
    返回两个有效地点之间的最短跳数。
    """

    if start == target:
        return 0

    visited = {start}
    queue: deque[tuple[str, int]] = deque([(start, 0)])

    while queue:
        current, hops = queue.popleft()
        for neighbor in LOCATION_GRAPH.get(current, ()):
            if neighbor in visited:
                continue
            if neighbor == target:
                return hops + 1
            visited.add(neighbor)
            queue.append((neighbor, hops + 1))

    return None


def calculate_overall_match(
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
) -> float:
    """
    Calculate the weighted final score with dynamic weight transfer.
    通过动态权重转移计算加权最终分数。
    """

    weights = adjusted_weights(time_match, location_match)
    score = (
        clamp_score(visual_similarity) * weights["visual"]
        + (0.0 if time_match is None else clamp_score(time_match) * weights["time"])
        + (0.0 if location_match is None else clamp_score(location_match) * weights["location"])
    )
    return round(clamp_score(score), 4)


def adjusted_weights(time_match: float | None, location_match: float | None) -> dict[str, float]:
    """
    Return scoring weights after moving unused time/location weight to visual.
    将未使用的时间或地点权重转移到视觉分数后返回评分权重。
    """

    weights = dict(BASE_WEIGHTS)
    if time_match is None:
        weights["visual"] += weights["time"]
        weights["time"] = 0.0
    if location_match is None:
        weights["visual"] += weights["location"]
        weights["location"] = 0.0
    return weights


def confidence_label(overall_match: float) -> str:
    """
    Convert an overall score into a confidence label.
    将总分转换为置信度标签。
    """

    if overall_match >= 0.80:
        return "High"
    if overall_match >= 0.55:
        return "Medium"
    return "Low"


def positive_reasons(
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
) -> list[str]:
    """
    Generate positive human-readable matching reasons.
    生成正向的人类可读匹配理由。
    """

    reasons: list[str] = []
    if visual_similarity >= 0.80:
        reasons.append("The item has high visual similarity with the user description.")
    elif visual_similarity >= 0.60:
        reasons.append("The item has moderate visual similarity with the user description.")

    if time_match is not None and time_match >= 0.80:
        reasons.append("The item's found time is close to the selected lost time.")

    if location_match == 1.0:
        reasons.append("The item's found location matches the selected lost location.")
    elif location_match is not None and location_match >= 0.85:
        reasons.append("The item's found location is near the selected lost location.")
    elif location_match is not None and location_match >= 0.70:
        reasons.append("The item's found location is relatively close to the selected lost location.")

    return reasons


def mismatch_notes(
    visual_similarity: float,
    time_match: float | None,
    location_match: float | None,
    lost_time_range: TimeRange | None,
    lost_location: str | None,
    found_time: TimePoint | None,
    found_location: str | None,
) -> list[str]:
    """
    Generate human-readable notes for possible mismatches.
    生成可能不匹配情况的人类可读说明。
    """

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
        if location_match == LOCATION_MIN_SCORE:
            notes.append("The item's found location is different from the selected lost location.")
        elif not _is_scored_location(_clean_location_key(found_location)):
            notes.append("The item does not have found-location information.")

    return notes


def has_time_range(time_range: TimeRange | None) -> bool:
    """
    Return True when a time range contains any usable time information.
    当时间范围包含任何可用时间信息时返回 True。
    """

    if time_range is None:
        return False
    return has_time_point(time_range.start) or has_time_point(time_range.end)


def has_time_point(time_point: TimePoint | None) -> bool:
    """
    Return True when a time point contains a valid date or hour.
    当时间点包含有效日期或小时时返回 True。
    """

    if time_point is None:
        return False
    return _parse_date(time_point.date) is not None or _clean_hour(time_point.hour) is not None


def found_time_is_before_lost_range(
    lost_time_range: TimeRange | None,
    found_time: TimePoint | None,
) -> bool:
    """
    Return True when found time clearly predates the selected lost time.
    当拾获时间明确早于所选丢失时间时返回 True。
    """

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
    """
    Clamp a numeric-like value into the 0 to 1 score range.
    将类似数字的值限制在 0 到 1 的分数范围内。
    """

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
    """
    Normalize supported query call styles into one internal tuple.
    将支持的查询调用方式规范化为一个内部元组。
    """

    if isinstance(query, SearchQuery):
        result_limit = top_k if top_k is not None else query.result_limit
        return query.lost_time_range, query.lost_location, _clean_result_limit(result_limit)
    return query, lost_location, _clean_result_limit(top_k)


def _clean_result_limit(value: int | None) -> int | None:
    """
    Convert a result limit into a non-negative integer or None.
    将结果数量限制转换为非负整数或 None。
    """

    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _field(candidate: Any, name: str, default: Any) -> Any:
    """
    Read a named field from a dataclass-like object or dictionary.
    从类似数据类的对象或字典中读取指定字段。
    """

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
    """
    Create a MatchResult while respecting the current contract fields.
    在遵守当前契约字段的前提下创建 MatchResult。
    """

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
    """
    Return the earlier non-None value from two comparable values.
    从两个可比较的值中返回较早的非 None 值。
    """

    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def _score_datetime_distance(found: datetime, start: datetime, end: datetime) -> float:
    """
    Score a full date-and-hour found time against a lost time range.
    根据丢失时间范围为包含完整日期和小时的拾获时间评分。
    """

    if start > end:
        start, end = end, start
    if start <= found <= end:
        return TIME_SCORE_WITHIN_RANGE

    if found < start:
        hours_before_start = (start - found).total_seconds() / 3600
        return TIME_SCORE_EARLIER_WITHIN_1_HOUR if hours_before_start <= 1 else TIME_SCORE_EARLIER_MORE_THAN_1_HOUR

    hours_after_end = (found - end).total_seconds() / 3600
    if hours_after_end <= 1:
        return TIME_SCORE_LATER_WITHIN_1_HOUR
    if hours_after_end <= 3:
        return TIME_SCORE_LATER_WITHIN_3_HOURS
    if found.date() == end.date():
        return TIME_SCORE_SAME_DAY_FAR

    date_gap = (found.date() - end.date()).days
    if date_gap <= 1:
        return TIME_SCORE_1_DAY_AWAY
    if date_gap <= 3:
        return TIME_SCORE_2_TO_3_DAYS_AWAY
    return TIME_SCORE_MORE_THAN_3_DAYS_AWAY


def _score_date_distance(found: date, start: date, end: date) -> float:
    """
    Score a found date against a lost date range.
    根据丢失日期范围为拾获日期评分。
    """

    if start > end:
        start, end = end, start
    if start <= found <= end:
        return DATE_SCORE_SAME_DATE

    day_gap = (start - found).days if found < start else (found - end).days
    if day_gap == 1:
        return DATE_SCORE_1_DAY_AWAY
    if 2 <= day_gap <= 3:
        return DATE_SCORE_2_TO_3_DAYS_AWAY
    return DATE_SCORE_MORE_THAN_3_DAYS_AWAY


def _score_hour_distance(found: int, start: int, end: int) -> float:
    """
    Score a found hour against a lost hour range.
    根据丢失小时范围为拾获小时评分。
    """

    if _hour_in_range(found, start, end):
        return HOUR_SCORE_SAME_HOUR
    hour_gap = min(_hour_gap(found, start), _hour_gap(found, end))
    if hour_gap == 1:
        return HOUR_SCORE_1_HOUR_AWAY
    if 2 <= hour_gap <= 3:
        return HOUR_SCORE_2_TO_3_HOURS_AWAY
    return HOUR_SCORE_MORE_THAN_3_HOURS_AWAY


def _hour_in_range(found: int, start: int, end: int) -> bool:
    """
    Return True when an hour falls within a possibly overnight range.
    当小时落在可能跨夜的小时范围内时返回 True。
    """

    if start <= end:
        return start <= found <= end
    return found >= start or found <= end


def _hour_gap(first: int, second: int) -> int:
    """
    Return the circular clock-hour gap between two hours.
    返回两个小时之间按 24 小时循环计算的间隔。
    """

    direct_gap = abs(first - second)
    return min(direct_gap, 24 - direct_gap)


def _time_range_to_datetimes(time_range: TimeRange | None) -> tuple[datetime | None, datetime | None]:
    """
    Convert a time range into start and end datetimes when possible.
    在可能时将时间范围转换为开始和结束 datetime。
    """

    if time_range is None:
        return None, None
    return _time_point_to_datetime(time_range.start), _time_point_to_datetime(time_range.end)


def _time_range_to_dates(time_range: TimeRange | None) -> tuple[date | None, date | None]:
    """
    Convert a time range into start and end dates when possible.
    在可能时将时间范围转换为开始和结束日期。
    """

    if time_range is None:
        return None, None
    start_date = _parse_date(time_range.start.date) if time_range.start is not None else None
    end_date = _parse_date(time_range.end.date) if time_range.end is not None else None
    return start_date, end_date


def _time_range_to_hours(time_range: TimeRange | None) -> tuple[int | None, int | None]:
    """
    Convert a time range into start and end hours when possible.
    在可能时将时间范围转换为开始和结束小时。
    """

    if time_range is None:
        return None, None
    start_hour = _clean_hour(time_range.start.hour) if time_range.start is not None else None
    end_hour = _clean_hour(time_range.end.hour) if time_range.end is not None else None
    return start_hour, end_hour


def _time_point_to_datetime(time_point: TimePoint | None) -> datetime | None:
    """
    Convert a time point into a datetime when it has both date and hour.
    当时间点同时包含日期和小时时将其转换为 datetime。
    """

    if time_point is None:
        return None
    point_date = _parse_date(time_point.date)
    point_hour = _clean_hour(time_point.hour)
    if point_date is None or point_hour is None:
        return None
    return datetime(point_date.year, point_date.month, point_date.day, point_hour)


def _parse_date(value: str | None) -> date | None:
    """
    Parse an ISO date string, returning None when invalid or missing.
    解析 ISO 日期字符串；无效或缺失时返回 None。
    """

    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _clean_hour(value: int | str | None) -> int | None:
    """
    Normalize an hour value to an integer from 0 to 23.
    将小时值规范化为 0 到 23 之间的整数。
    """

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
    """
    Trim a location key and return None when it is empty.
    清理地点键，并在其为空时返回 None。
    """

    if value is None:
        return None
    cleaned_value = str(value).strip()
    return cleaned_value or None


def _is_scored_location(location_key: str | None) -> bool:
    """
    Return True when a location key can be used for location scoring.
    当地点键可用于地点评分时返回 True。
    """

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
