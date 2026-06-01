"""Shared data contracts for the GrepL browser interface."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchQuery:
    """User search input collected from the browser interface."""

    description: str
    lost_time: str | None = None
    lost_location: str | None = None
    result_limit: int = 20


@dataclass
class LostItem:
    """A found item stored by the lost-and-found system."""

    item_id: str
    title: str
    image_path: str
    found_time: str | None
    found_location: str | None
    category: str | None = None


@dataclass
class MatchResult:
    """A ranked match returned to the UI."""

    item_id: str
    title: str
    image_path: str
    found_time: str | None
    found_location: str | None
    visual_similarity: float
    time_match: float
    location_match: float
    overall_match: float
    confidence_label: str
    reasons: list[str] = field(default_factory=list)
    mismatch_notes: list[str] = field(default_factory=list)
