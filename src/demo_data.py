"""Tracked fallback demo data for local UI testing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from contracts import MatchResult, SearchQuery


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_ASSET_DIR = PROJECT_ROOT / "data" / "demo"


@dataclass(frozen=True)
class _DemoItem:
    item_id: str
    title: str
    category: str
    color: str
    found_time: str
    found_location: str
    special_notes: list[str]


_DEMO_ITEMS = [
    _DemoItem(
        item_id="F001",
        title="Blue Water Bottle With Sticker",
        category="Bottle",
        color="Blue",
        found_time="Yesterday afternoon",
        found_location="Library",
        special_notes=["Has cartoon sticker", "Standard bottle shape"],
    ),
    _DemoItem(
        item_id="F002",
        title="Black Keychain Set",
        category="Keys",
        color="Black",
        found_time="This morning",
        found_location="Cafeteria",
        special_notes=["Metal keychain ring", "One card holder attached"],
    ),
    _DemoItem(
        item_id="F003",
        title="White Earbuds Case",
        category="Earphones",
        color="White",
        found_time="Yesterday evening",
        found_location="Classroom",
        special_notes=["Charging case only", "Small scratch on the lid"],
    ),
    _DemoItem(
        item_id="F004",
        title="Purple Student Card Holder",
        category="Student Card",
        color="Purple",
        found_time="Today",
        found_location="Dormitory",
        special_notes=["Campus card inside", "Transparent holder"],
    ),
    _DemoItem(
        item_id="F005",
        title="Brown Wallet",
        category="Wallet",
        color="Brown",
        found_time="Last night",
        found_location="Sports Centre",
        special_notes=["Leather texture", "Contains a name tag"],
    ),
]


def mock_search_items(query: SearchQuery) -> list[MatchResult]:
    """Return demo ranking results driven by the confirmed query hints."""

    ranked = sorted(
        (_build_result(query, item) for item in _DEMO_ITEMS),
        key=lambda result: result.overall_match,
        reverse=True,
    )
    return ranked[: max(1, min(int(query.result_limit), len(ranked)))]


def _build_result(query: SearchQuery, item: _DemoItem) -> MatchResult:
    raw_text = query.description.lower()
    visual_score = 0.46
    reasons = ["Original query is preserved for matching."]

    if item.category.lower() in raw_text:
        visual_score += 0.12
        reasons.append(f"Raw description directly mentions a {item.category.lower()}.")
    if query.item_type_hint and query.item_type_hint == item.category:
        visual_score += 0.2
        reasons.append(f"Confirmed item type matches {item.category.lower()}.")
    if query.color_hint and query.color_hint == item.color:
        visual_score += 0.1
        reasons.append(f"Confirmed color matches {item.color.lower()}.")

    special_bonus = 0.0
    for note in query.special_notes:
        if _notes_overlap(note, item.special_notes):
            special_bonus += 0.05
    if special_bonus:
        reasons.append("Special mark hints overlap with the found-item notes.")

    time_match = 0.4
    if query.lost_time and query.lost_time.lower() in item.found_time.lower():
        time_match = 0.92
        reasons.append("Found time is close to the reported lost time.")

    location_match = 0.35
    if query.lost_location and query.lost_location.lower() in item.found_location.lower():
        location_match = 0.94
        reasons.append("Found location matches the reported area.")

    overall_match = min(0.99, visual_score * 0.6 + time_match * 0.15 + location_match * 0.15 + special_bonus)
    mismatch_notes: list[str] = []
    if query.item_type_hint and query.item_type_hint != item.category:
        mismatch_notes.append(f"Confirmed item type points to {query.item_type_hint.lower()}, not {item.category.lower()}.")
    if query.color_hint and query.color_hint != item.color:
        mismatch_notes.append(f"Confirmed color differs from the {item.color.lower()} item shown here.")
    if query.lost_location and query.lost_location.lower() not in item.found_location.lower():
        mismatch_notes.append("Found location does not fully match the reported area.")

    return MatchResult(
        item_id=item.item_id,
        title=item.title,
        image_path="",
        found_time=item.found_time,
        found_location=item.found_location,
        visual_similarity=round(min(0.99, visual_score + special_bonus), 2),
        time_match=round(time_match, 2),
        location_match=round(location_match, 2),
        overall_match=round(overall_match, 2),
        confidence_label=_confidence_label(overall_match),
        reasons=reasons,
        mismatch_notes=mismatch_notes,
    )


def _notes_overlap(note: str, item_notes: list[str]) -> bool:
    lowered = note.lower()
    return any(token in lowered for token in _tokenize_notes(item_notes))


def _tokenize_notes(item_notes: list[str]) -> set[str]:
    tokens: set[str] = set()
    for note in item_notes:
        for part in note.lower().replace("-", " ").split():
            if len(part) > 2:
                tokens.add(part)
    return tokens


def _confidence_label(score: float) -> str:
    if score >= 0.8:
        return "Strong Match"
    if score >= 0.65:
        return "Likely Match"
    if score >= 0.5:
        return "Possible Match"
    return "Weak Match"
