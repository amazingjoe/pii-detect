import argparse
import sys
from pathlib import Path

from settings import add_selection_args, load_settings, resolve_selection


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description="Download a base model listed in settings.json into its local path (models/ is not in the repo)."
    )
    add_selection_args(parser, settings, head=False)
    parser.add_argument("--list", action="store_true", help="Show every configured model and whether it is downloaded")
    parser.add_argument("--force", action="store_true", help="Download again even if the model directory exists")
    args = parser.parse_args()

    if args.list:
        for name, m in settings["models"].items():
            state = "downloaded" if (Path(m["path"]) / "config.json").exists() else "missing"
            default = " (default)" if name == settings["default_model"] else ""
            print(f"  {name}{default}: {state}  path={m['path']}  hf_repo={m.get('hf_repo', '-')}")
        return

    sel = resolve_selection(settings, args.model)
    entry = settings["models"][sel["model_name"]]
    repo = entry.get("hf_repo")
    if not repo:
        print(f"Error: model '{sel['model_name']}' has no \"hf_repo\" in settings.json. Download it manually to '{sel['model_path']}'.")
        sys.exit(1)

    target = Path(sel["model_path"])
    if (target / "config.json").exists() and not args.force:
        print(f"'{target}' already holds a model. Use --force to download again.")
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub is not installed. Run 'pip install -r requirements.txt' first.")
        sys.exit(1)

    print(f"Downloading https://huggingface.co/{repo} to '{target}' ...")
    snapshot_download(repo_id=repo, local_dir=str(target))
    print(f"Done. Check it with: python predict.py --model {sel['model_name']} \"hello\"")


if __name__ == "__main__":
    main()
