"""NiceGUI browser interface for GrepL."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from nicegui import app, ui

from mock_data import DEMO_ASSET_DIR
from config.options import LOCATION_OPTIONS, date_options, hour_options, option_label, select_labels
from contracts import MatchResult, SearchQuery, TimePoint, TimeRange
from search_service import search_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "src" / "static"
PLACEHOLDER_COLORS = ("#dce9f6", "#f1f5f9")


def run_app() -> None:
    """Configure and start the browser GUI."""

    app.add_static_files("/static", str(STATIC_DIR))
    app.add_static_files("/demo-assets", str(DEMO_ASSET_DIR))
    _register_pages()
    ui.run(
        title="GrepL Lost & Found",
        host="127.0.0.1",
        port=int(os.environ.get("GREPL_PORT", "5000")),
        reload=False,
        show=os.environ.get("GREPL_NO_BROWSER") != "1",
    )


def _register_pages() -> None:
    @ui.page("/")
    def index() -> None:
        ui.add_head_html('<link rel="stylesheet" href="/static/styles.css">')

        with ui.element("main").classes("app-shell"):
            with ui.element("section").classes("search-panel"):
                with ui.element("div").classes("search-copy"):
                    ui.label("GrepL").classes("brand")
                    ui.label("Campus Lost & Found").classes("page-title")
                    ui.label("Describe your lost item and review candidate matches from the found-item library.").classes(
                        "intro-text"
                    )

                # 动态折叠搜索栏容器
                with ui.element("div").classes("search-container"):
                    with ui.row().classes("search-main-row"):
                        description = (
                            ui.input(
                                label="Item Description",
                                placeholder="Example: blue bottle with stickers",
                            )
                            .classes("description-field search-control")
                            .props("borderless clearable")
                        )
                        with ui.row().classes("main-action-group"):
                            advanced_toggle = ui.button("Filters", icon="tune").classes("filter-action").props("flat no-caps")
                            search_button = ui.button("Search", icon="search").classes("primary-action").props("unelevated no-caps")

                    with ui.element("div").classes("search-advanced-panel") as advanced_panel:
                        advanced_panel.set_visibility(False)
                        ui.separator().classes("panel-divider")
                        
                        with ui.element("div").classes("advanced-grid"):
                            with ui.element("div").classes("search-segment"):
                                with ui.column().classes("time-range-group"):
                                    ui.label("Lost Time Range").classes("field-group-title")
                                    with ui.row().classes("time-range-row"):
                                        start_date = ui.select(options=date_options(), label="Start Date", value="").classes("time-select search-control").props("borderless")
                                        start_hour = ui.select(options=hour_options(), label="Start Hour", value="").classes("time-select search-control").props("borderless")
                                    with ui.row().classes("time-range-row"):
                                        end_date = ui.select(options=date_options(), label="End Date", value="").classes("time-select search-control").props("borderless")
                                        end_hour = ui.select(options=hour_options(), label="End Hour", value="").classes("time-select search-control").props("borderless")
                            
                            with ui.element("div").classes("search-segment"):
                                lost_location = ui.select(
                                    options=select_labels(LOCATION_OPTIONS), label="Lost Location", value="any"
                                ).classes("w-full search-control").props("borderless")
                            
                            with ui.element("div").classes("search-segment"):
                                result_limit = ui.number(
                                    label="Number of Results", value=5, min=1, max=10, step=1
                                ).classes("w-full search-control").props("borderless")
                        
                        with ui.row().classes("advanced-footer"):
                            reset_button = ui.button("Reset Filters", icon="refresh").classes("secondary-action").props("flat no-caps")

                with ui.row().classes("loading-row") as loading_row:
                    ui.spinner("dots", size="md", color="primary")
                    ui.label("Searching found items...").classes("loading-text")
                loading_row.set_visibility(False)

            with ui.element("section").classes("results-panel"):
                with ui.row().classes("results-header"):
                    with ui.column().classes("header-copy"):
                        ui.label("Candidate Matches").classes("section-title")
                        status_label = ui.label("Enter a description to begin.").classes("status-text")
                    ui.icon("inventory_2").classes("header-icon")

                results_container = ui.column().classes("results-grid")
                _render_empty_state(results_container)

        # 交互逻辑
        def toggle_advanced() -> None:
            is_visible = not advanced_panel.visible
            advanced_panel.set_visibility(is_visible)
            if is_visible:
                advanced_toggle.classes(add="active-filter", remove="filter-action")
            else:
                advanced_toggle.classes(add="filter-action", remove="active-filter")

        advanced_toggle.on_click(toggle_advanced)

        async def handle_search() -> None:
            query_text = (description.value or "").strip()
            if not query_text:
                ui.notify("Please enter an item description.", color="warning", position="top")
                description.props("error error-message='Description is required'")
                return

            description.props(remove="error error-message")
            search_button.disable()
            loading_row.set_visibility(True)
            status_label.text = "Finding possible matches..."
            results_container.clear()

            try:
                query = SearchQuery(
                    description=query_text,
                    lost_time_range=_build_time_range(
                        start_date.value,
                        start_hour.value,
                        end_date.value,
                        end_hour.value,
                    ),
                    lost_location=lost_location.value or "any",
                    result_limit=int(result_limit.value or 5),
                )
                results = await asyncio.to_thread(search_items, query)
                results_container.clear()
                if results:
                    status_label.text = f"Showing {len(results)} possible match{'es' if len(results) != 1 else ''}."
                    _render_results(results_container, results)
                else:
                    status_label.text = "No matches found."
                    _render_empty_state(results_container, "No candidates yet", "Try adding a color, visual feature, or location.")
            except Exception:
                status_label.text = "Search failed."
                results_container.clear()
                _render_error_state(results_container)
            finally:
                loading_row.set_visibility(False)
                search_button.enable()

        def handle_reset() -> None:
            description.value = ""
            start_date.value = ""
            start_hour.value = ""
            end_date.value = ""
            end_hour.value = ""
            lost_location.value = "any"
            result_limit.value = 5
            status_label.text = "Enter a description to begin."
            results_container.clear()
            _render_empty_state(results_container)

        search_button.on_click(handle_search)
        reset_button.on_click(handle_reset)


def _render_results(container: ui.column, results: list[MatchResult]) -> None:
    with container:
        for index, result in enumerate(results, start=1):
            _render_result_card(index, result)


def _render_result_card(index: int, result: MatchResult) -> None:
    with ui.card().classes("result-card"):
        with ui.row().classes("result-main"):
            image_url = _image_url(result.image_path)
            if image_url:
                ui.image(image_url).classes("item-image")
            else:
                with ui.element("div").classes("item-image image-fallback"):
                    ui.icon("image_not_supported").classes("fallback-icon")
                    ui.label("Image unavailable").classes("fallback-text")

            with ui.column().classes("result-content"):
                with ui.row().classes("result-topline"):
                    ui.label(f"#{index}").classes("rank-badge")
                    ui.label(result.confidence_label).classes(_confidence_class(result.confidence_label))
                ui.label(f"Candidate #{index}").classes("item-title")
                ui.label("Review this item visually before claiming it.").classes("candidate-note")
                ui.label(_found_summary(result)).classes("item-meta")

                with ui.row().classes("score-row"):
                    _score_pill("Overall Match", result.overall_match)
                    _score_pill("Visual Similarity", result.visual_similarity)
                    _score_pill("Place Match", result.location_match)

                _reason_list("Why It May Match", result.reasons, "check_circle")
                if result.mismatch_notes:
                    _reason_list("Why This May Not Match", result.mismatch_notes, "info")


def _score_pill(label: str, value: float) -> None:
    with ui.element("div").classes("score-pill"):
        ui.label(label).classes("score-label")
        ui.label(_percent(value)).classes("score-value")


def _reason_list(title: str, items: list[str], icon_name: str) -> None:
    with ui.element("div").classes("reason-block"):
        with ui.row().classes("reason-title-row"):
            ui.icon(icon_name).classes("reason-icon")
            ui.label(title).classes("reason-title")
        for item in items:
            ui.label(item).classes("reason-line")


def _render_empty_state(container: ui.column, title: str = "Ready to search", detail: str = "Results will appear here after you submit a lost item description.") -> None:
    with container:
        with ui.element("div").classes("empty-state"):
            ui.icon("manage_search").classes("empty-icon")
            ui.label(title).classes("empty-title")
            ui.label(detail).classes("empty-detail")


def _render_error_state(container: ui.column) -> None:
    with container:
        with ui.element("div").classes("empty-state error-state"):
            ui.icon("error").classes("empty-icon")
            ui.label("Something went wrong").classes("empty-title")
            ui.label("Please try again or check whether the backend modules are available.").classes("empty-detail")


def _build_time_range(
    start_date: str | None,
    start_hour: str | int | None,
    end_date: str | None,
    end_hour: str | int | None,
) -> TimeRange | None:
    start = _build_time_point(start_date, start_hour)
    end = _build_time_point(end_date, end_hour)
    if start is None and end is None:
        return None
    return TimeRange(start=start, end=end)


def _build_time_point(selected_date: str | None, selected_hour: str | int | None) -> TimePoint | None:
    date_value = selected_date or None
    hour_value = _selected_hour_to_int(selected_hour)
    if date_value is None and hour_value is None:
        return None
    return TimePoint(date=date_value, hour=hour_value)


def _selected_hour_to_int(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _image_url(image_path: str) -> str | None:
    path = Path(image_path)
    if not path.is_file():
        return None
    try:
        relative_path = path.relative_to(DEMO_ASSET_DIR).as_posix()
    except ValueError:
        return None
    return f"/demo-assets/{relative_path}"


def _found_summary(result: MatchResult) -> str:
    time_text = _format_time_point(result.found_time)
    location_text = option_label(result.found_location, LOCATION_OPTIONS) or result.found_location or "Location unknown"
    return f"Found at {location_text} · {time_text}"


def _format_time_point(time_point: TimePoint | None) -> str:
    if time_point is None:
        return "Time unknown"
    if time_point.date and time_point.hour is not None:
        return f"{time_point.date} {time_point.hour:02d}:00"
    if time_point.date:
        return time_point.date
    if time_point.hour is not None:
        return f"{time_point.hour:02d}:00"
    return "Time unknown"


def _percent(value: float) -> str:
    return f"{round(value * 100):d}%"


def _confidence_class(label: str) -> str:
    normalized = label.lower().replace(" ", "-")
    return f"confidence-badge {normalized}"