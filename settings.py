import json
import sys
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    """Loads shared defaults from settings.json. CLI flags override these values."""
    if not path.exists():
        print(f"Error: settings file not found at '{path}'.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
