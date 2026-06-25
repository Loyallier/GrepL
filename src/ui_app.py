"""NiceGUI browser interface for GrepL."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from nicegui import app, ui

from config.options import LOCATION_OPTIONS, option_label, select_labels
from contracts import FollowUpQuestion, MatchResult, SearchQuery, SearchResponse, TimePoint, TimeRange
from search_service import search_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "src" / "static"
ITEM_IMAGE_DIR = PROJECT_ROOT / "data" / "cropped_item_image"
PROCESS_STAGES = (
    "Analyzing description...",
    "Scanning visual database...",
    "Ranking results by time and location...",
)
LOGGER = logging.getLogger(__name__)


def run_app() -> None:
    """Configure and start the browser GUI."""

    app.add_static_files("/static", str(STATIC_DIR))
    app.add_static_files("/item-images", str(ITEM_IMAGE_DIR))
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

        with ui.element("main").classes("app-shell") as app_shell:
            with ui.element("section").classes("search-section"):
                ui.label("GrepL").classes("brand")
                with ui.element("div").classes("search-body"):
                    ui.label("Campus Lost & Found").classes("page-title")
                    ui.label("Describe what you lost. Add time or location only when it helps.").classes("intro-text")

                    with ui.element("div").classes("omnibox"):
                        with ui.row().classes("omnibox-main"):
                            description = (
                                ui.input(placeholder="Describe the item you lost")
                                .classes("description-input")
                                .props("borderless clearable")
                            )
                            filters_button = ui.button(icon="tune").classes("icon-button").props(
                                "flat round type=button aria-label='Filters'"
                            )
                            reset_button = ui.button(icon="refresh").classes("icon-button reset-icon").props(
                                "flat round type=button aria-label='Reset'"
                            )
                            search_button = ui.button("Search", icon="search").classes("search-button").props(
                                "unelevated no-caps type=button"
                            )

                        with ui.element("div").classes("filters-panel") as filters_panel:
                            filters_panel.set_visibility(False)
                            with ui.element("div").classes("filters-grid"):
                                start_date = _date_input("Start date")
                                end_date = _date_input("End date")
                                lost_location = ui.select(
                                    options=select_labels(LOCATION_OPTIONS),
                                    label="Lost location",
                                    value="any",
                                ).classes("filter-control").props("borderless")
                                result_limit = ui.number(
                                    label="Number of results",
                                    value=5,
                                    min=1,
                                    max=10,
                                    step=1,
                                ).classes("filter-control").props("borderless")
                    with ui.element("div").classes("clarification-banner") as clarification_banner:
                        clarification_banner.set_visibility(False)

                    with ui.element("div").classes("process-panel") as process_panel:
                        process_steps = []
                        for label in PROCESS_STAGES:
                            with ui.element("div").classes("process-step") as step:
                                ui.icon("check").classes("process-check")
                                ui.element("span").classes("process-dot")
                                ui.label(label).classes("process-label")
                            process_steps.append(step)
                    process_panel.set_visibility(False)

            with ui.element("section").classes("results-section") as results_section:
                with ui.row().classes("results-header"):
                    ui.label("Candidate Matches").classes("section-title")
                    status_label = ui.label("").classes("status-text")
                results_container = ui.element("div").classes("results-grid")
            results_section.set_visibility(False)

        def toggle_filters() -> None:
            next_visible = not filters_panel.visible
            filters_panel.set_visibility(next_visible)
            if next_visible:
                filters_button.classes(add="icon-button-active")
            else:
                filters_button.classes(remove="icon-button-active")

        def close_filters() -> None:
            filters_panel.set_visibility(False)
            filters_button.classes(remove="icon-button-active")

        def set_process_stage(index: int, *, done: bool = False, error: bool = False) -> None:
            process_panel.set_visibility(True)
            for step_index, step in enumerate(process_steps):
                step.classes(remove="process-step-active process-step-complete process-step-error")
                if error and step_index == index:
                    step.classes(add="process-step-error")
                elif done or step_index < index:
                    step.classes(add="process-step-complete")
                elif step_index == index:
                    step.classes(add="process-step-active")

        def clear_clarification() -> None:
            clarification_banner.clear()
            clarification_banner.set_visibility(False)

        def current_query() -> SearchQuery | None:
            query_text = (description.value or "").strip()
            if not query_text:
                description.props("error error-message='Description is required'")
                ui.notify("Please enter an item description.", color="warning", position="top")
                return None
            description.props(remove="error error-message")
            return SearchQuery(
                description=query_text,
                lost_time_range=_build_time_range(start_date.value, end_date.value),
                lost_location=lost_location.value or "any",
                result_limit=int(result_limit.value or 5),
            )

        async def run_search(query: SearchQuery) -> None:
            clear_clarification()
            close_filters()
            app_shell.classes(add="app-shell-active")
            results_section.set_visibility(True)
            results_container.clear()
            status_label.text = "Searching for possible matches..."
            search_button.disable()

            try:
                set_process_stage(0)
                await asyncio.sleep(0.18)
                response = await asyncio.to_thread(search_items, query)

                if response.follow_up is not None:
                    status_label.text = "Clarification needed."
                    set_process_stage(0)
                    render_clarification(query, response.follow_up)
                    return

                set_process_stage(1)
                await asyncio.sleep(0.18)
                set_process_stage(2)
                await asyncio.sleep(0.18)
                set_process_stage(2, done=True)
                render_response(response)
            except Exception:
                LOGGER.exception("Search failed while rendering results.")
                status_label.text = "Search failed."
                results_container.clear()
                set_process_stage(2, error=True)
                _render_empty_state(
                    results_container,
                    "Something went wrong",
                    "Please try again or check whether the backend modules are available.",
                    icon="error",
                )
            finally:
                search_button.enable()

        def render_clarification(query: SearchQuery, follow_up: FollowUpQuestion) -> None:
            clarification_banner.clear()
            clarification_banner.set_visibility(True)
            with clarification_banner:
                with ui.row().classes("clarification-content"):
                    ui.icon("help_outline").classes("clarification-icon")
                    with ui.column().classes("clarification-copy"):
                        ui.label("Clarification").classes("clarification-title")
                        ui.label(follow_up.question).classes("clarification-question")
                    with ui.row().classes("chip-row"):
                        for option in follow_up.options:
                            ui.button(
                                option,
                                on_click=lambda selected=option: asyncio.create_task(
                                    run_search(_query_with_follow_up(query, follow_up, selected))
                                ),
                            ).classes("choice-chip").props("flat no-caps")

        def render_response(response: SearchResponse) -> None:
            results_container.clear()
            results = response.results
            if not results:
                status_label.text = "No matches found."
                _render_empty_state(
                    results_container,
                    "No candidates found",
                    "Try adding a color, visual feature, or location.",
                )
                return

            match_text = "match" if len(results) == 1 else "matches"
            status_label.text = f"Showing {len(results)} possible {match_text}."
            _render_results(results_container, results)

        async def handle_search() -> None:
            query = current_query()
            if query is not None:
                await run_search(query)

        def handle_reset() -> None:
            description.value = ""
            description.props(remove="error error-message")
            start_date.value = None
            end_date.value = None
            lost_location.value = "any"
            result_limit.value = 5
            status_label.text = ""
            results_container.clear()
            clear_clarification()
            close_filters()
            results_section.set_visibility(False)
            process_panel.set_visibility(False)
            app_shell.classes(remove="app-shell-active")
            description.update()
            start_date.update()
            end_date.update()
            lost_location.update()
            result_limit.update()

        filters_button.on_click(toggle_filters)
        search_button.on_click(handle_search)
        reset_button.on_click(handle_reset)


def _date_input(label: str) -> ui.input:
    today = date.today().isoformat()
    return (
        ui.input(label=label)
        .classes("filter-control")
        .props(f'borderless type="date" max="{today}"')
    )


def _query_with_follow_up(query: SearchQuery, follow_up: FollowUpQuestion, answer: str) -> SearchQuery:
    next_query = SearchQuery(
        description=query.description,
        search_text=query.search_text,
        use_original_query=query.use_original_query,
        lost_time_range=query.lost_time_range,
        lost_location=query.lost_location,
        result_limit=query.result_limit,
        item_type_hint=query.item_type_hint,
        color_hint=query.color_hint,
        special_notes=list(query.special_notes),
        component_color_hints=dict(query.component_color_hints),
    )

    if follow_up.target == "item_type_hint":
        if answer == "None of the above":
            next_query.use_original_query = True
            next_query.search_text = None
            next_query.item_type_hint = None
            next_query.color_hint = None
            next_query.special_notes = []
            next_query.component_color_hints = {}
        else:
            next_query.item_type_hint = answer
    elif follow_up.target == "special_notes" and answer.lower().startswith("no"):
        next_query.special_notes = ["__IGNORE__"]

    return next_query


def _render_results(container: ui.element, results: list[MatchResult]) -> None:
    with container:
        for index, result in enumerate(results, start=1):
            _render_result_card(index, result)


def _render_result_card(index: int, result: MatchResult) -> None:
    with ui.element("article").classes("result-card"):
        with ui.element("div").classes("card-summary"):
            with ui.element("div").classes("image-frame"):
                _render_result_image(result)
            with ui.element("div").classes("summary-content"):
                with ui.row().classes("card-topline"):
                    ui.label(f"Candidate #{index}").classes("candidate-title")
                    ui.label(_confidence_text(result.confidence_label)).classes(
                        f"confidence-tag confidence-{_confidence_key(result.confidence_label)}"
                    )
                with ui.row().classes("meta-row"):
                    _metadata("place", _format_location(result.found_location))
                    _metadata("schedule", _format_time_point(result.found_time))
                with ui.element("div").classes("score-ring").style(_score_ring_style(result.overall_match)):
                    ui.label(_percent(result.overall_match)).classes("score-ring-value")

        with ui.expansion("Details").classes("details-expansion").props("dense expand-icon=keyboard_arrow_down"):
            with ui.element("div").classes("details-panel"):
                _metric_bar("Visual Similarity", result.visual_similarity)
                _metric_bar("Time Match", result.time_match)
                _metric_bar("Location Match", result.location_match)
                _reason_list("Why it matched", result.reasons, "check_circle", "reason-positive")
                if result.mismatch_notes:
                    _reason_list("Why this may not match", result.mismatch_notes, "info", "reason-warning")


def _render_result_image(result: MatchResult) -> None:
    image_url = _image_url(result.image_path)
    if image_url:
        ui.image(image_url).classes("item-image")
    else:
        with ui.element("div").classes("image-fallback"):
            ui.icon("image_not_supported").classes("fallback-icon")
            ui.label("Image unavailable").classes("fallback-text")


def _metadata(icon_name: str, text: str) -> None:
    with ui.row().classes("metadata"):
        ui.icon(icon_name).classes("metadata-icon")
        ui.label(text).classes("metadata-text")


def _metric_bar(label: str, value: float | None) -> None:
    score = _clamp(value)
    with ui.element("div").classes("metric"):
        with ui.row().classes("metric-header"):
            ui.label(label).classes("metric-label")
            ui.label(_percent(value)).classes("metric-value")
        with ui.element("div").classes("metric-track"):
            ui.element("div").classes("metric-fill").style(f"width: {round(score * 100)}%")


def _reason_list(title: str, items: list[str], icon_name: str, classes: str) -> None:
    with ui.element("div").classes(f"reason-block {classes}"):
        ui.label(title).classes("reason-heading")
        if not items:
            ui.label("No additional notes.").classes("reason-line")
            return
        for item in items:
            with ui.row().classes("reason-row"):
                ui.icon(icon_name).classes("reason-icon")
                ui.label(item).classes("reason-line")


def _render_empty_state(container: ui.element, title: str, detail: str, *, icon: str = "manage_search") -> None:
    with container:
        with ui.element("div").classes("empty-state"):
            ui.icon(icon).classes("empty-icon")
            ui.label(title).classes("empty-title")
            ui.label(detail).classes("empty-detail")


def _build_time_range(start_date: str | None, end_date: str | None) -> TimeRange | None:
    start = _build_time_point(start_date)
    end = _build_time_point(end_date)
    if start is None and end is None:
        return None
    return TimeRange(start=start, end=end)


def _build_time_point(selected_date: str | None) -> TimePoint | None:
    date_value = selected_date or None
    if date_value is None:
        return None
    return TimePoint(date=date_value)


def _image_url(image_path: str) -> str | None:
    path = Path(image_path)
    if not path.is_file():
        return None
    try:
        relative_path = path.relative_to(ITEM_IMAGE_DIR).as_posix()
    except ValueError:
        return None
    return f"/item-images/{relative_path}"


def _format_location(value: str | None) -> str:
    return option_label(value, LOCATION_OPTIONS) or value or "Location unknown"


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


def _percent(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{round(_clamp(value) * 100):d}%"


def _clamp(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def _score_ring_style(value: float | None) -> str:
    percent = round(_clamp(value) * 100)
    return f"--score: {percent}%;"


def _confidence_text(label: str) -> str:
    normalized = label.strip().lower()
    if "high" in normalized or "strong" in normalized:
        return "High"
    if "medium" in normalized or "likely" in normalized or "possible" in normalized:
        return "Medium"
    return "Low"


def _confidence_key(label: str) -> str:
    return _confidence_text(label).lower()
