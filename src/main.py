"""Application entrypoint for GrepL."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from ui_app import run_app
    except ModuleNotFoundError as error:
        if error.name == "nicegui":
            print("NiceGUI is not installed. Run: pip install -r requirements.txt")
            sys.exit(1)
        raise

    run_app()


if __name__ in {"__main__", "__mp_main__"}:
    main()
