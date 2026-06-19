"""Structured option dictionaries used by the UI and ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class SelectOption:
    """A stable selectable value with display text and matching hints."""

    label: str
    keywords: tuple[str, ...] = ()


LOCATION_OPTIONS: dict[str, SelectOption] = {
    "any": SelectOption("Any location"),
    "a1": SelectOption("A1", ("a1",)),
    "a2": SelectOption("A2", ("a2",)),
    "a3_classroom": SelectOption("A3 Classroom", ("a3", "classroom")),
    "a4": SelectOption("A4", ("a4",)),
    "a5": SelectOption("A5", ("a5",)),
    "library": SelectOption("Library", ("library",)),
    "playground": SelectOption("Playground", ("playground",)),
    "b1": SelectOption("B1", ("b1",)),
    "d_dormitory": SelectOption("D Block Dormitory", ("d", "dormitory")),
    "d6_cafeteria": SelectOption("D6 Cafeteria", ("d6", "cafeteria")),
    "ly_dormitory": SelectOption("LY Block Dormitory", ("ly", "dormitory")),
    "ly3_cafeteria": SelectOption("LY3 Cafeteria", ("ly3", "cafeteria")),
    "music_island": SelectOption("Music Island", ("music", "island")),
    "not_sure": SelectOption("Not Sure")
}


def select_labels(options: dict[str, SelectOption]) -> dict[str, str]:
    """Return labels in the dictionary shape expected by NiceGUI select."""

    return {key: option.label for key, option in options.items()}


def date_options(lookback_days: int = 14) -> dict[str, str]:
    """Return optional ISO date values for recent campus lost-item searches."""

    today = date.today()
    options = {"": "No date"}
    for offset in range(lookback_days + 1):
        value = (today - timedelta(days=offset)).isoformat()
        options[value] = value
    return options


def hour_options() -> dict[str, str]:
    """Return optional hour values from 00:00 through 23:00."""

    return {"": "No hour"} | {str(hour): f"{hour:02d}:00" for hour in range(24)}


def option_keywords(value: str | None, options: dict[str, SelectOption]) -> tuple[str, ...]:
    """Convert a selected option code into matching keywords."""

    if not value or value == "any":
        return ()
    return options.get(value, SelectOption(value, (value,))).keywords


def option_label(value: str | None, options: dict[str, SelectOption]) -> str | None:
    """Convert a selected option code into a user-facing label."""

    if not value or value == "any":
        return None
    return options.get(value, SelectOption(value)).label
