import argparse
import json
import sys
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


def load_settings(path: Path = SETTINGS_PATH) -> dict:
    """Loads shared defaults from settings.json. CLI flags override these values."""
    if not path.exists():
        print(f"Error: settings file not found at '{path}'.", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def add_selection_args(parser: argparse.ArgumentParser, settings: dict, head: bool = True):
    """Adds --model (and --head unless head=False), which pick an entry from settings.json `models` / `heads`."""
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Base model to use, a key of settings.json models: {', '.join(settings['models'])} "
        f"(default: {settings['default_model']})",
    )
    if not head:
        return
    parser.add_argument(
        "--head",
        type=str,
        default=None,
        help=f"Probe head (task) to use, a key of settings.json heads: {', '.join(settings['heads'])} "
        f"(default: {settings['default_head']})",
    )


def _pick(settings: dict, section: str, name: str, default_key: str) -> tuple:
    name = name or settings[default_key]
    entries = settings[section]
    if name not in entries:
        print(f"Error: unknown {section[:-1]} '{name}'. Available: {', '.join(entries)} (see settings.json '{section}').", file=sys.stderr)
        sys.exit(1)
    return name, entries[name]


def resolve_selection(settings: dict, model: str = None, head: str = None) -> dict:
    """Resolves the chosen model and head into concrete values.

    Returns model_name, model_path, layer, head_name, weights_path, prep_dir, done_dir and
    regressions_path. A "{model}" placeholder in a head's weights_path becomes the model name, so
    one head can keep a separate probe per base model.
    """
    model_name, m = _pick(settings, "models", model, "default_model")
    head_name, h = _pick(settings, "heads", head, "default_head")
    return {
        "model_name": model_name,
        "model_path": m["path"],
        "layer": m["layer"],
        "head_name": head_name,
        "weights_path": h["weights_path"].replace("{model}", model_name),
        "prep_dir": h["prep_dir"],
        "done_dir": h["done_dir"],
        "regressions_path": h["regressions_path"],
    }


def preparse_selection(settings: dict) -> dict:
    """Reads --model/--head from argv before the real parser is built, so the other flags can
    default to the selected model's and head's values. The real parser still validates them."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default=None)
    pre.add_argument("--head", default=None)
    known, _ = pre.parse_known_args()
    return resolve_selection(settings, known.model, known.head)
