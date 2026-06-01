# GrepL Interface Specification

The browser GUI talks to the rest of the project through one stable function:

```python
def search_items(query: SearchQuery) -> list[MatchResult]:
    ...
```

Current implementation uses mock data in `src/mock_data.py`. When the real
database, image matching, and ranking modules are ready, only
`src/search_service.py` should need to change.

## Query

```python
@dataclass
class SearchQuery:
    description: str
    lost_time: str | None = None
    lost_location: str | None = None
    result_limit: int = 5
```

## Result

```python
@dataclass
class MatchResult:
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
    reasons: list[str]
    mismatch_notes: list[str]
```

## Backend Integration

```python
items = database.load_items()
items_with_similarity = embedding_engine.match_text_to_images(query.description, items)
results = ranker.evaluate_matches(
    items_with_similarity,
    query.lost_time,
    query.lost_location,
    top_k=query.result_limit,
)
return results
```

User-facing UI labels should stay simple: use "Visual Similarity",
"Overall Match", "Number of Results", and "Why This May Not Match".
