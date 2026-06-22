"""Tracked fallback demo data for local UI testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config.options import LOCATION_OPTIONS, option_keywords
from contracts import MatchResult, SearchQuery, TimePoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ASSET_DIR = PROJECT_ROOT / "data" / "demo"


@dataclass(frozen=True)
class _DemoItem:
    item_id: str
    image_path: str
    category: str
    color: str
    found_time: TimePoint | None
    found_location: str | None
    special_notes: list[str]


_DEMO_ITEMS = [
    _DemoItem(
        item_id="F001",
        image_path="",
        category="Bottle",
        color="Blue",
        found_time=TimePoint(date="2026-06-17", hour=14),
        found_location="library",
        special_notes=["sticker", "cartoon"],
    ),
    _DemoItem(
        item_id="F002",
        image_path="",
        category="Keys",
        color="Black",
        found_time=TimePoint(date="2026-06-18", hour=9),
        found_location="d6_cafeteria",
        special_notes=["keychain", "metal ring"],
    ),
    _DemoItem(
        item_id="F003",
        image_path="",
        category="Earphones",
        color="White",
        found_time=TimePoint(date="2026-06-17", hour=20),
        found_location="a3_classroom",
        special_notes=["case", "scratch"],
    ),
    _DemoItem(
        item_id="F004",
        image_path="",
        category="Student Card",
        color="Purple",
        found_time=TimePoint(date="2026-06-18", hour=11),
        found_location="d_dormitory",
        special_notes=["card holder"],
    ),
    _DemoItem(
        item_id="F005",
        image_path="",
        category="Wallet",
        color="Brown",
        found_time=TimePoint(date="2026-06-17", hour=22),
        found_location="playground",
        special_notes=["leather", "name tag"],
    ),
]


def mock_search_items(query: SearchQuery) -> list[MatchResult]:
    ranked = sorted(
        (_build_result(query, item) for item in _DEMO_ITEMS),
        key=lambda result: result.overall_match,
        reverse=True,
    )
    return ranked[: max(1, min(int(query.result_limit), len(ranked)))]


def _build_result(query: SearchQuery, item: _DemoItem) -> MatchResult:
    search_text = (query.search_text or query.description).lower()
    visual_score = 0.42
    reasons: list[str] = ["Uses the reconstructed query hints for matching."]

    if item.category.lower() in search_text:
        visual_score += 0.18
        reasons.append("Reconstructed query mentions the item category.")
    if query.item_type_hint and query.item_type_hint == item.category:
        visual_score += 0.18
        reasons.append("Confirmed item type matches this candidate.")
    if query.color_hint and query.color_hint == item.color:
        visual_score += 0.12
        reasons.append("Confirmed color matches this candidate.")

    component_bonus = 0.0
    for part, color in (query.component_color_hints or {}).items():
        if part in search_text and color.lower() in search_text:
            component_bonus += 0.03
    if component_bonus:
        reasons.append("Color-part binding hints are consistent with the query.")

    special_bonus = 0.0
    for note in query.special_notes:
        if note.lower() in search_text:
            special_bonus += 0.03
    if special_bonus:
        reasons.append("Special mark hints appear in the query.")

    time_match = 0.35
    if query.lost_time_range and item.found_time and query.lost_time_range.start and query.lost_time_range.start.date:
        if query.lost_time_range.start.date == item.found_time.date:
            time_match = 0.75
            reasons.append("Found date matches the selected time range.")

    location_match = 0.3
    query_location = query.lost_location or "any"
    if query_location != "any" and item.found_location:
        keywords = option_keywords(query_location, LOCATION_OPTIONS)
        if query_location == item.found_location or any(word in item.found_location for word in keywords):
            location_match = 0.85
            reasons.append("Found location is consistent with the selected area.")

    overall_match = min(
        0.99,
        visual_score * 0.65 + time_match * 0.15 + location_match * 0.15 + special_bonus + component_bonus,
    )

    mismatch_notes: list[str] = []
    if query.item_type_hint and query.item_type_hint != item.category:
        mismatch_notes.append("Item type differs from the confirmed hint.")
    if query.color_hint and query.color_hint != item.color:
        mismatch_notes.append("Color differs from the confirmed hint.")

    return MatchResult(
        item_id=item.item_id,
        image_path=item.image_path,
        found_time=item.found_time,
        found_location=item.found_location,
        visual_similarity=round(min(0.99, visual_score + special_bonus + component_bonus), 2),
        time_match=round(time_match, 2),
        location_match=round(location_match, 2),
        overall_match=round(overall_match, 2),
        confidence_label=_confidence_label(overall_match),
        reasons=reasons,
        mismatch_notes=mismatch_notes,
    )


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "Strong Match"
    if score >= 0.65:
        return "Likely Match"
    if score >= 0.5:
        return "Possible Match"
    return "Weak Match"
