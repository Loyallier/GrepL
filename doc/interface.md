# GrepL Interface Specification

-  try to use data class to encapsulate the input and output data for each interface

## 1. GUI Related Interface

The browser GUI talks to the rest of the project through one stable function:

```python
def search_items(query: SearchQuery) -> list[MatchResult]:
    ...
```

Current implementation uses mock data (prefer `src/mock_data.py`, fallback to
`src/demo_data.py` if the mock module is unavailable). When the real
database, image matching, and ranking modules are ready, only
`src/search_service.py` should need to change.

The search result is a candidate match, not an object-recognition label. The UI
must not display item names inferred from the matching model.

### Query (Input)

```python
@dataclass
class TimePoint:
    date: str | None = None
    hour: int | None = None


@dataclass
class TimeRange:
    start: TimePoint | None = None
    end: TimePoint | None = None


@dataclass
class SearchQuery:
    description: str
    lost_time_range: TimeRange | None = None
    lost_location: str | None = None
    result_limit: int = 20
```

- `lost_time_range`: Objective time input selected by the user. Dates use ISO strings such as `"2026-06-14"`. Hours are integers from `0` to `23`. Partial values are allowed and passed through directly.
        
- `lost_location`: The selected key from `LOCATION_OPTIONS` in `src/config/options.py` (e.g., `"library"`). Use `"any"` when unknown.
		
- `result_limit`: The maximum number of most-matching results to keep in the final list

Structured dropdown dictionaries:

```python
date_options() -> dict[str, str]
hour_options() -> dict[str, str]
LOCATION_OPTIONS: dict[str, SelectOption]
```

Time dropdown values are not normalized by the GUI. Blank date/hour selections remain `None`, and hour-only or date-only input remains valid. Backend modules decide how to interpret incomplete time information.
  
### Result (Output)

```python
@dataclass
class MatchResult:
    item_id: str
    image_path: str
    found_time: TimePoint | None
    found_location: str | None
    visual_similarity: float
    time_match: float | None
    location_match: float | None
    overall_match: float
    confidence_label: str
    reasons: list[str]
    mismatch_notes: list[str]
```

### Backend Integration 

```python
items = database.load_items()
items_with_similarity = embedding_engine.match_text_to_images(query.description, items)
results = ranker.evaluate_matches(
    items_with_similarity,
    query.lost_time_range,
    query.lost_location,
    top_k=query.result_limit,
)
return results
```

User-facing UI labels should stay simple: use "Visual Similarity",
"Overall Match", "Number of Results", and "Why This May Not Match".
Result cards should use neutral labels such as "Candidate #1" instead of item
names, because CLIP retrieves similar images but does not identify the object.

---


## 2. Image Processing & Storage Pipeline

Establish a database for "Found Items".

### 2.1 Object Detection & Cropping Interface

流程：模型加载 --> 推理计算图片坐标--> 裁剪出子图返回文件夹（后处理部分，image_path替换） 

子图裁剪成功之后才能创建对应的 LostItem 对象

- **Caller**: `main.py`
    
- **Provider**: `detector.py`
    
- **Function Prototype**: `detector.detect_objects(
  row_image_path: str) -> list[RowItem]` (一个组合函数，结合坐标推理和子图裁剪流程各自的函数)
    
- **Input Data** (`str`): The file path of the row image taken for the found items (e.g., `"./raw_images/room_302.jpg"`)
    
- **Return Data (`list[RowItem]`)**:  
`RowItem` 包含一张图片裁剪出所有子图的路径，便于后续对象逐一创建。以及对应子图的裁剪坐标置信度
    ```python
    @dataclass
    class RowItem:
        """ Initial found item information"""
        image_path: str
        bound_confidence: float
    ```


### 2.2 Image Vectorization & Registration Interface

 `main.py` passes the cropped sub-image path and its uniquely corresponding `item_id` to `embedding_engine.py`. Then, `embedding_engine.py` converts it into an image embedding vector and caches it in some data storage. Returns `True` if successful. 

`embedding_engine.py` extracts the image feature vector and binds it with the `item_id`.
   
- **Caller**: `main.py`
    
- **Provider**: `embedding_engine.py`
    
- **Function Prototype**: `embedding_engine.register_item_image(registering_item: RegisterItem) -> bool`
    
    
- **Input Data (`RegisterItem`)**: 
    ``` Python
    @dataclass
    class RegisterItem:
        item_id: str
        image_path: str
    ```

- **Return Data (`bool`)**: a boolen indicator 

---
## 3. Retrieval & Matching Strategy Pipeline

Used during the stage when students look for lost items, managing data communication after a student initiates a search request.

### 3.1 Multimodal Text Retrieval Interface

Links the implementations of Member 5 and Member 6.

- **Caller**: `main.py`
    
- **Provider**: `embedding_engine.py`
    
- **Function Prototype**: `embedding_engine.match_text_to_images(description: str) -> list[ClipResult]`
    
- **Input Data (`str`)**: The description string is entered by the student, containing visual appearance information, etc.
    
- **Return Data (`list[ClipResult]`)**: includes item_id + CLIP score
    ``` Python
    @dataclass
    class ClipResult:
        item_id: str
        visual_similarity: float
    ```


### 3.2 Comprehensive Strategy Ranking Interface

- **Caller**: `main.py`
    
- **Provider**: `ranker.py`
    
- **Function Prototype**: `ranker.evaluate_matches(candidates: list[Candidate], query: SearchQuery) -> list[MatchResult]`
    
- **Input Data**:
    
    `candidates`: A newly combined list composed of `Candidate` objects.
   
    ``` Python
    @dataclass
    class Candidate:
        item_id: str
        image_path: str
        found_time: TimePoint | None 
        found_location: str | None 
        visual_similarity: float 
        bound_confidence: float

    ```
    `query`:  `SearchQuery` object, the corresponding data class is defined in the above section
    	
- **Return Data (`MatchResult`)**:
    according to data class defined in the above section



- 当 `query.lost_time_range` 或 `query.lost_location` 为 None（即用户未填写时间或地点）时，`ranker.py` 应按以下策略调整权重计算公式（降级处理）,一共三种情况：

  1. 若 lost_time_range 缺失 (None)：

  -  处理逻辑：自动取消“时间邻近度评分”。不对候选物品进行时间跨度扣分。

  - 权重转移：原本分配给时间维度的权重，将等比例转移/合并到 `visual_similarity` 上，使系统退化为以“视觉特征为主、空间特征为辅”的检索模式。

  2. 若 lost_location 缺失 (None)：

  - 处理逻辑：自动取消“空间地理位置距离/区域匹配评分”。不再考虑具体丢失地点与拾获地点的距离差。

  - 权重转移：原本分配给地理地点的权重，将等比例转移/合并到 `visual_similarity` 上。

  3. 若两者皆缺失 (None)：

  - 处理逻辑：完全不使用任何时空过滤与加权。

  - 降级结果：`ranker.py` 的评分逻辑直接等价于 Candidate.visual_similarity * Bound_Weight（仅参考视觉相似度与目标检测置信度，该公式仅是作为例子），退化为单纯的图像外观检索。
  
        
