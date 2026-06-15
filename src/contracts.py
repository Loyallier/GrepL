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
    lost_time_range: TimeRange | None = None
    lost_location: str | None = None
    result_limit: int = 20


@dataclass
class LostItem:
    """A found item stored by the lost-and-found system."""

    item_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None
    category: str | None = None


@dataclass(frozen=True)
class BoundingBox:
    """Pixel-space object box in the original image."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class DetectedObject:
    """A detector output before cropping and registration."""

    label: str
    confidence: float
    bbox: BoundingBox


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
class RowItem:
    """Initial information for each identified item from one picture."""

    image_path: str
    bound_confidence: float
    bbox: BoundingBox | None = None
    label: str | None = None


@dataclass
class RegisterItem:
    """ To encapsulate all information required for the cropped image of an item to being converts into an image embedding vector. """

    item_id: str
    image_path: str


@dataclass
class ClipResult:
    """ The comparison result returned from embedding_engine.py. """

    item_id: str
    visual_similarity: float


@dataclass
class Candidate:
    """ Transfor item information required for ranker.py. """

    item_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None 
    visual_similarity: float 
    bound_confidence: float
