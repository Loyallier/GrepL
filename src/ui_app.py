"""NiceGUI browser interface for GrepL."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from nicegui import app, ui

from mock_data import DEMO_ASSET_DIR
from contracts import MatchResult, SearchQuery
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
                ui.label("GrepL").classes("brand")
                ui.label("Campus Lost & Found").classes("page-title")
                ui.label("Describe your lost item and review the most likely matches from the found-item library.").classes(
                    "intro-text"
                )

                description = (
                    ui.textarea(
                        label="Item Description",
                        placeholder="Example: blue water bottle with stickers",
                    )
                    .classes("w-full")
                    .props("outlined autogrow clearable")
                )
                lost_time = (
                    ui.input(
                        label="Lost Time",
                        placeholder="Example: yesterday afternoon",
                    )
                    .classes("w-full")
                    .props("outlined clearable")
                )
                lost_location = (
                    ui.input(
                        label="Lost Location",
                        placeholder="Example: library",
                    )
                    .classes("w-full")
                    .props("outlined clearable")
                )
                result_limit = (
                    ui.number(label="Number of Results", value=5, min=1, max=10, step=1)
                    .classes("w-full")
                    .props("outlined")
                )

                with ui.row().classes("action-row"):
                    search_button = ui.button("Search", icon="search").classes("primary-action").props("unelevated no-caps")
                    reset_button = ui.button("Reset", icon="refresh").classes("secondary-action").props("outline no-caps")

                with ui.row().classes("loading-row") as loading_row:
                    ui.spinner("dots", size="md", color="primary")
                    ui.label("Searching found items...").classes("loading-text")
                loading_row.set_visibility(False)

            with ui.element("section").classes("results-panel"):
                with ui.row().classes("results-header"):
                    with ui.column().classes("header-copy"):
                        ui.label("Match Results").classes("section-title")
                        status_label = ui.label("Enter a description to begin.").classes("status-text")
                    ui.icon("inventory_2").classes("header-icon")

                results_container = ui.column().classes("results-grid")
                _render_empty_state(results_container)

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
                    lost_time=lost_time.value,
                    lost_location=lost_location.value,
                    result_limit=int(result_limit.value or 5),
                )
                results = await asyncio.to_thread(search_items, query)
                results_container.clear()
                if results:
                    status_label.text = f"Showing {len(results)} possible match{'es' if len(results) != 1 else ''}."
                    _render_results(results_container, results)
                else:
                    status_label.text = "No matches found."
                    _render_empty_state(results_container, "No matches yet", "Try adding a color, item type, or location.")
            except Exception:
                status_label.text = "Search failed."
                results_container.clear()
                _render_error_state(results_container)
            finally:
                loading_row.set_visibility(False)
                search_button.enable()

        def handle_reset() -> None:
            description.value = ""
            lost_time.value = ""
            lost_location.value = ""
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
                ui.label(result.title).classes("item-title")
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
    time_text = result.found_time or "Time unknown"
    location_text = result.found_location or "Location unknown"
    return f"Found at {location_text} · {time_text}"


def _percent(value: float) -> str:
    return f"{round(value * 100):d}%"


def _confidence_class(label: str) -> str:
    normalized = label.lower().replace(" ", "-")
    return f"confidence-badge {normalized}"
