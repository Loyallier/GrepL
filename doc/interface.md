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


## 1. Image Processing & Storage Pipeline

Establish a database for "Found Items".

### 1.1 Object Detection Interface

- **Caller**: `main.py`
    
- **Provider**: `detector.py`
    
- **Function Prototype**: `detector.detect_objects(image_path: str) -> list[dict]`
    
- **Input Data**: The file path of the image taken for the found item (e.g., `"./raw_images/room_302.jpg"`)
    
- **Return Data (dict format)**:
    
``` Python
[
    {
        "box": [xmin, ymin, xmax, ymax],  # Bounding box coordinates
        "confidence": 0.92,               # Confidence score
    },
    ...
]
```

### 1.2 Image Vectorization & Registration Interface

- **Task**: `main.py` passes the cropped sub-image path and its uniquely corresponding `item_id` to `embedding_engine.py`. Then, `embedding_engine.py` converts it into an image embedding vector and caches it. Returns `True` if successful. 
- `embedding_engine.py` extracts the image feature vector and binds/caches it with the `item_id`.
   
- **Caller**: `main.py`
    
- **Provider**: `embedding_engine.py`
    
- **Function Prototype**: `embedding_engine.register_item_image(item_id: str, cropped_img_path: str) -> bool`
    
- **Input Data**: Path of the cropped sub-image, uniquely mapped the ID `item_id`
    
- **Return Data**: Boolean indicator


## 2. Retrieval & Matching Strategy Pipeline

Used during the stage when students look for lost items, managing data communication after a student initiates a search request.

### 2.1 Multimodal Text Retrieval Interface

Links the implementations of Member 5 and Member 6.

- **Caller**: `main.py`
    
- **Provider**: `embedding_engine.py`
    
- **Function Prototype**: `embedding_engine.match_text_to_images(description: str, items: list[dict]) -> list[dict]`
    
- **Input Data**: The description string entered by the student, containing visual appearance information, etc.
    
- **Return Data (includes original data + CLIP score)**:
    

``` Python
[
    {
        "item_id": "item_001",
        "clip_score": 0.285  # CLIP-calculated similarity score between text and image
    },
    {
        "item_id": "item_002",
        "clip_score": 0.192
    }
    ...
]
```

### 2.2 Comprehensive Strategy Ranking Interface

- **Caller**: `main.py`
    
- **Provider**: `ranker.py`
    
- **Function Prototype**: `ranker.evaluate_matches(candidates: list[dict], lost_time: str | None, lost_location: str | None, top_k: int) -> tuple[list[dict], dict]`
    
- **Input Data**:
    
    - `candidates`: A newly combined list composed of the list returned by `embedding_engine.py` along with meta-information like the actual time the item was found (`found_time`) and the actual location (`found_location`):
        
``` Python
[
    {
        "item_id": "item_20260527_001",
        "clip_score": 0.2854,
        "found_time": "2026-05-27 14:30",  
        "found_location": "Library",  
        "confidence": 0.94 # Detection confidence score, used for weight trust deduction
    },
    ... 
]
```
- 
	- `lost_time`: The time the item was lost as filled in by the student (e.g., `"2026-05-27 14:00"`)
        
    - `lost_location`: The location where the item was lost as filled in by the student (e.g., `"Library"`)
		
	- `top_k`: The maximum number of most-matching results to keep in the final list
	
- **Return Data (Dual return values: Final ranking list + Elimination/Ranking reasons)**:
    
    - **Return Value 1 (list)**: The final Top-N recommendation list combined with appearance and spatio-temporal weighting, including multidimensional score breakdowns.
        
    - **Return Value 2 (dict)**: The specific evaluation reasons for each item's ranking (including both positive and negative aspects).
        

``` Python
# Return Value 1
[
    {
        "item_id": "item_001", 
        "final_score": 0.88,
        "time_match": 0.95,        
        "location_match": 1.0      
    },
    {
        "item_id": "item_002", 
        "final_score": 0.45,
        "time_match": 0.80,
        "location_match": 0.1
    }
]

# Return Value 2
{
    "item_001": {
        "reasons": ["Appearance matches highly.", "The found time (14:30) is close to your lost time."],
        "mismatch_notes": []
    },
    "item_002": {
        "reasons": ["Appearance has some similarity."],
        "mismatch_notes": ["This item was found at 'Academic Building A', which is far from the 'Library' you filled in."]
    }
}
``` 


