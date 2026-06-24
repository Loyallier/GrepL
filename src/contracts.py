"""Shared data contracts for the GrepL browser interface."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimePoint:
    """An optional objective time point selected from the UI."""

    date: str | None = None
    hour: int | None = None


@dataclass
class TimeRange:
    """An optional objective time range selected from the UI."""

    start: TimePoint | None = None
    end: TimePoint | None = None


@dataclass
class SearchQuery:
    """User search input collected from the browser interface."""

    description: str
    search_text: str | None = None
    use_original_query: bool = False
    lost_time_range: TimeRange | None = None
    lost_location: str | None = None
    result_limit: int = 20
    item_type_hint: str | None = None
    color_hint: str | None = None
    special_notes: list[str] = field(default_factory=list)
    component_color_hints: dict[str, str] = field(default_factory=dict)


@dataclass
class FollowUpQuestion:
    target: str
    question: str
    options: list[str] = field(default_factory=list)
    multi_select: bool = False


@dataclass
class SearchResponse:
    results: list[MatchResult] = field(default_factory=list)
    follow_up: FollowUpQuestion | None = None


@dataclass
class LostItem:
    """A found item stored by the lost-and-found system."""

    item_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None
    bound_confidence: float
    raw_id: str | None = None
    category: str | None = None  # Unknown attritube


@dataclass
class MatchResult:
    """A ranked match returned to the UI."""

    item_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None
    visual_similarity: float
    time_match: float | None
    location_match: float | None
    overall_match: float
    confidence_label: str
    reasons: list[str] = field(default_factory=list)
    mismatch_notes: list[str] = field(default_factory=list)



@dataclass
class RawFoundItem:
    """A raw found-item photo containing one or more physical items."""

    raw_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None


@dataclass
class RowItem:
    """ Initial information for each identified item from one picture. """

    image_path: str
    bound_confidence: float


@dataclass
class RegisterItem:
    """ To encapsulate all information required for the cropped image of an item to being converts into an image embedding vector. """

    item_id: str
    image_path: str


@dataclass
class Candidate:
    """ Transfor item information required for ranker.py. """

    item_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None 
    visual_similarity: float 
    bound_confidence: float
