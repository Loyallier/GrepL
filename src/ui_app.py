"""NiceGUI browser interface for GrepL."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from nicegui import app, ui

from config.options import LOCATION_OPTIONS, date_options, hour_options, option_label, select_labels
from contracts import FollowUpQuestion, MatchResult, SearchQuery, TimePoint, TimeRange
from search_service import search_items


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "src" / "static"
ITEM_IMAGE_DIR = PROJECT_ROOT / "data" / "cropped_item_image"
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

        with ui.element("main").classes("app-shell"):
            with ui.element("section").classes("search-panel"):
                with ui.row().classes("topbar"):
                    with ui.element("div").classes("brand-lockup"):
                        ui.label("GrepL").classes("brand")
                        ui.label("AI-assisted lost item retrieval").classes("brand-subtitle")
                    with ui.row().classes("signal-group"):
                        _signal_chip("visual_search", "Visual matching")
                        _signal_chip("inventory_2", "Live library")

                with ui.element("div").classes("search-copy"):
                    ui.label("Campus Lost & Found").classes("page-title")
                    ui.label(
                        "Describe the item you lost. GrepL compares your description with registered "
                        "found-item images and ranks likely candidates."
                    ).classes("intro-text")

                with ui.element("div").classes("search-container"):
                    with ui.row().classes("search-label-row"):
                        ui.label("Search workspace").classes("eyebrow")
                        ui.label("Filters stay hidden until you need them.").classes("microcopy")

                    with ui.row().classes("search-main-row"):
                        description = (
                            ui.input(
                                label="Item description",
                                placeholder="Example: blue bottle with stickers",
                            )
                            .classes("description-field search-control")
                            .props("borderless clearable")
                        )
                        with ui.row().classes("main-action-group"):
                            advanced_toggle = ui.button("Filters", icon="tune").classes("filter-action").props(
                                "flat no-caps"
                            )
                            search_button = ui.button("Search", icon="search").classes("primary-action").props(
                                "unelevated no-caps"
                            )

                    with ui.element("div").classes("search-advanced-panel") as advanced_panel:
                        advanced_panel.set_visibility(False)
                        ui.separator().classes("panel-divider")

                        with ui.element("div").classes("advanced-grid"):
                            with ui.element("div").classes("search-segment"):
                                with ui.column().classes("time-range-group"):
                                    ui.label("Lost time range").classes("field-group-title")
                                    with ui.row().classes("time-range-row"):
                                        start_date = (
                                            ui.select(options=date_options(), label="Start date", value="")
                                            .classes("time-select search-control")
                                            .props("borderless")
                                        )
                                        start_hour = (
                                            ui.select(options=hour_options(), label="Start hour", value="")
                                            .classes("time-select search-control")
                                            .props("borderless")
                                        )
                                    with ui.row().classes("time-range-row"):
                                        end_date = (
                                            ui.select(options=date_options(), label="End date", value="")
                                            .classes("time-select search-control")
                                            .props("borderless")
                                        )
                                        end_hour = (
                                            ui.select(options=hour_options(), label="End hour", value="")
                                            .classes("time-select search-control")
                                            .props("borderless")
                                        )

                            with ui.element("div").classes("search-segment"):
                                lost_location = ui.select(
                                    options=select_labels(LOCATION_OPTIONS), label="Lost location", value="any"
                                ).classes("w-full search-control").props("borderless")

                            with ui.element("div").classes("search-segment"):
                                result_limit = ui.number(
                                    label="Number of results", value=5, min=1, max=10, step=1
                                ).classes("w-full search-control").props("borderless")

                        with ui.row().classes("advanced-footer"):
                            reset_button = ui.button("Reset filters", icon="refresh").classes("secondary-action").props(
                                "flat no-caps"
                            )

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

        def toggle_advanced() -> None:
            is_visible = not advanced_panel.visible
            advanced_panel.set_visibility(is_visible)
            if is_visible:
                advanced_toggle.classes(add="filter-action-active")
            else:
                advanced_toggle.classes(remove="filter-action-active")

        advanced_toggle.on_click(toggle_advanced)

        async def handle_search() -> None:
            query_text = (description.value or "").strip()
            if not query_text:
                ui.notify("Please enter an item description.", color="warning", position="top")
                description.props("error error-message='Description is required'")
                return

            async def prompt_follow_up(follow_up: FollowUpQuestion) -> str | list[str] | None:
                selection: str | list[str] | None = None
                waiter = asyncio.Event()

                with ui.dialog() as dialog, ui.card().classes("follow-up-card"):
                    ui.label("One more detail").classes("dialog-kicker")
                    ui.label(follow_up.question).classes("dialog-title")
                    if follow_up.multi_select:
                        control = ui.select(follow_up.options, multiple=True).classes("w-full dialog-control")
                    else:
                        control = ui.radio(follow_up.options).classes("dialog-control").props("inline")

                    def confirm() -> None:
                        nonlocal selection
                        selection = control.value
                        dialog.close()
                        waiter.set()

                    with ui.row().classes("dialog-actions"):
                        ui.button("Confirm", icon="check", on_click=confirm).classes("primary-action").props(
                            "unelevated no-caps"
                        )

                dialog.open()
                await waiter.wait()
                return selection

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
                response = await asyncio.to_thread(search_items, query)
                if response.follow_up is not None:
                    answer = await prompt_follow_up(response.follow_up)
                    if answer is None:
                        results_container.clear()
                        status_label.text = "Search cancelled."
                        _render_empty_state(
                            results_container,
                            "Cancelled",
                            "Please confirm the missing details and search again.",
                        )
                        return

                    follow_up_query = SearchQuery(
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
                    if response.follow_up.target == "item_type_hint" and isinstance(answer, str):
                        if answer == "None of the above":
                            follow_up_query.use_original_query = True
                            follow_up_query.item_type_hint = None
                            follow_up_query.color_hint = None
                            follow_up_query.special_notes = []
                            follow_up_query.component_color_hints = {}
                            follow_up_query.search_text = None
                        else:
                            follow_up_query.item_type_hint = answer
                    elif response.follow_up.target == "special_notes" and isinstance(answer, str):
                        if answer.lower().startswith("no"):
                            follow_up_query.special_notes = ["__IGNORE__"]

                    response = await asyncio.to_thread(search_items, follow_up_query)

                results_container.clear()
                results = response.results
                if results:
                    match_text = "match" if len(results) == 1 else "matches"
                    status_label.text = f"Showing {len(results)} possible {match_text}."
                    _render_results(results_container, results)
                else:
                    status_label.text = "No matches found."
                    _render_empty_state(
                        results_container,
                        "No candidates yet",
                        "Try adding a color, visual feature, or location.",
                    )
            except Exception:
                LOGGER.exception("Search failed while rendering results.")
                status_label.text = "Search failed."
                results_container.clear()
                _render_error_state(results_container)
            finally:
                loading_row.set_visibility(False)
                search_button.enable()

        def handle_reset() -> None:
            description.value = ""
            description.props(remove="error error-message")
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
            _render_result_image(result, "item-image")

            with ui.column().classes("result-content"):
                with ui.row().classes("result-topline"):
                    ui.label(f"#{index}").classes("rank-badge")
                    ui.label(result.confidence_label).classes(_confidence_class(result.confidence_label))
                ui.label(f"Candidate #{index}").classes("item-title")
                ui.label("Review the image and match signals before claiming it.").classes("candidate-note")
                ui.label(_found_summary(result)).classes("item-meta")

                with ui.row().classes("score-row"):
                    _score_pill("Overall Match", result.overall_match)
                    _score_pill("Visual Similarity", result.visual_similarity)
                    _score_pill("Place Match", result.location_match)

                with ui.row().classes("card-footer"):
                    ui.label(_compact_reason(result)).classes("compact-reason")
                    ui.button(
                        "View details",
                        icon="open_in_new",
                        on_click=lambda r=result, i=index: _open_result_detail(i, r),
                    ).classes("detail-action").props("flat no-caps")


def _render_result_image(result: MatchResult, classes: str) -> None:
    image_url = _image_url(result.image_path)
    if image_url:
        ui.image(image_url).classes(classes)
    else:
        with ui.element("div").classes(f"{classes} image-fallback"):
            ui.icon("image_not_supported").classes("fallback-icon")
            ui.label("Image unavailable").classes("fallback-text")


def _open_result_detail(index: int, result: MatchResult) -> None:
    with ui.dialog() as dialog, ui.card().classes("detail-dialog-card"):
        with ui.row().classes("detail-dialog-header"):
            with ui.column().classes("detail-title-group"):
                ui.label(f"Candidate #{index}").classes("dialog-title")
                ui.label(_found_summary(result)).classes("item-meta")
            ui.button(icon="close", on_click=dialog.close).classes("icon-action").props("flat round dense")

        with ui.row().classes("detail-layout"):
            _render_result_image(result, "detail-image")

            with ui.column().classes("detail-content"):
                with ui.row().classes("result-topline"):
                    ui.label(f"Rank #{index}").classes("rank-badge")
                    ui.label(result.confidence_label).classes(_confidence_class(result.confidence_label))

                with ui.element("div").classes("detail-score-grid"):
                    _score_pill("Overall Match", result.overall_match)
                    _score_pill("Visual Similarity", result.visual_similarity)
                    _score_pill("Time Match", result.time_match)
                    _score_pill("Place Match", result.location_match)

                _reason_list("Why It May Match", result.reasons, "check_circle")
                if result.mismatch_notes:
                    _reason_list("Why This May Not Match", result.mismatch_notes, "info")
                else:
                    ui.label("No mismatch notes were reported for this candidate.").classes("reason-line")

    dialog.open()


def _score_pill(label: str, value: float | None) -> None:
    with ui.element("div").classes("score-pill"):
        ui.label(label).classes("score-label")
        ui.label(_percent(value)).classes("score-value")


def _reason_list(title: str, items: list[str], icon_name: str) -> None:
    with ui.element("div").classes("reason-block"):
        with ui.row().classes("reason-title-row"):
            ui.icon(icon_name).classes("reason-icon")
            ui.label(title).classes("reason-title")
        if items:
            for item in items:
                ui.label(item).classes("reason-line")
        else:
            ui.label("No additional notes.").classes("reason-line")


def _signal_chip(icon_name: str, label: str) -> None:
    with ui.row().classes("signal-chip"):
        ui.icon(icon_name).classes("signal-icon")
        ui.label(label)


def _compact_reason(result: MatchResult) -> str:
    if result.reasons:
        return result.reasons[0]
    return "Open details to review the match evidence."


def _render_empty_state(
    container: ui.column,
    title: str = "Ready to search",
    detail: str = "Results will appear here after you submit a lost item description.",
) -> None:
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
        relative_path = path.relative_to(ITEM_IMAGE_DIR).as_posix()
    except ValueError:
        return None
    return f"/item-images/{relative_path}"


def _found_summary(result: MatchResult) -> str:
    time_text = _format_time_point(result.found_time)
    location_text = option_label(result.found_location, LOCATION_OPTIONS) or result.found_location or "Location unknown"
    return f"Found at {location_text} - {time_text}"


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
    return f"{round(value * 100):d}%"


def _confidence_class(label: str) -> str:
    normalized = label.lower().replace(" ", "-")
    return f"confidence-badge {normalized}"
