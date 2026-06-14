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
    "library": SelectOption("Library", ("library",)),
    "library_entrance": SelectOption("Library entrance", ("library", "entrance")),
    "cafeteria": SelectOption("Cafeteria", ("cafeteria",)),
    "block_a": SelectOption("Block A", ("block", "a")),
    "classroom": SelectOption("Classroom", ("classroom",)),
    "sports_center": SelectOption("Sports center", ("sports", "center")),
    "dormitory": SelectOption("Dormitory", ("dormitory",)),
    "not_sure": SelectOption("Not sure"),
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
