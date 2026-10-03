import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from extract_features import FeatureExtractor
from settings import add_selection_args, load_settings, preparse_selection
from train import load_dataset_from_json

REPORT_TEMPLATE = Path(__file__).resolve().parent / "sweep_report_template.html"
METRICS = ["roc_auc", "f1", "accuracy", "log_loss"]


def collect_json_files(dirs: List[Path]) -> List[Path]:
    files = []
    for d in dirs:
        files.extend(sorted(d.glob("*.json")))
    return files


def extract_all(extractor: FeatureExtractor, texts: List[str], batch_size: int) -> np.ndarray:
    """Returns features of shape (num_hidden_states, num_samples, hidden_dim), one forward pass per batch."""
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunks.append(extractor.extract_all_layers(texts[i : i + batch_size]))
        counter_str = f"  [{min(i + batch_size, len(texts))}/{len(texts)} samples]"
        if sys.stdout.isatty():
            sys.stdout.write(f"\r{counter_str}")
            sys.stdout.flush()
        else:
            print(counter_str)
    if sys.stdout.isatty():
        print()
    return np.concatenate(chunks, axis=1)


def new_probe() -> LogisticRegression:
    # Same probe configuration as train.py, so sweep scores reflect what train.py would produce
    return LogisticRegression(max_iter=1000, C=1.0)


def cross_validate_layer(X: np.ndarray, y: np.ndarray, folds: StratifiedKFold) -> Dict:
    """Per-fold metrics plus every sample's out-of-fold probability (each sample is scored by a probe that never saw it)."""
    per_fold = {m: [] for m in METRICS}
    oof_prob = np.zeros(len(y))
    converged = True
    for train_idx, test_idx in folds.split(X, y):
        probe = new_probe()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            probe.fit(X[train_idx], y[train_idx])
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            converged = False

        prob = probe.predict_proba(X[test_idx])[:, 1]
        oof_prob[test_idx] = prob
        pred = (prob > 0.5).astype(int)
        y_test = y[test_idx]
        per_fold["roc_auc"].append(roc_auc_score(y_test, prob))
        per_fold["f1"].append(f1_score(y_test, pred, zero_division=0))
        per_fold["accuracy"].append(accuracy_score(y_test, pred))
        per_fold["log_loss"].append(log_loss(y_test, prob, labels=[0, 1]))

    return {
        "mean": {m: float(np.mean(v)) for m, v in per_fold.items()},
        "std": {m: float(np.std(v)) for m, v in per_fold.items()},
        "folds": {m: [float(x) for x in v] for m, v in per_fold.items()},
        "converged": converged,
        "oof_prob": oof_prob,
    }


def summarize_errors(result: Dict, y: np.ndarray, categories: List[str]):
    """Adds total and per-category error counts, and the misclassified samples, from the out-of-fold predictions."""
    prob = result.pop("oof_prob")
    wrong = np.where((prob > 0.5).astype(int) != y)[0]
    result["errors"] = int(len(wrong))
    counts = Counter(categories[i] for i in wrong)
    result["by_category"] = {c: counts.get(c, 0) for c in sorted(set(categories))}
    # [sample index, probability], most confidently wrong first
    result["misclassified"] = sorted(
        ([int(i), round(float(prob[i]), 4)] for i in wrong), key=lambda m: -abs(m[1] - 0.5)
    )


def rank_layers(results: List[Dict]) -> List[int]:
    """Best first: fewest out-of-fold errors, then lowest mean log loss.

    ROC-AUC is reported but not used for ranking: it saturates near 1.0 for most layers, so tiny
    differences in it are noise, while errors at the real threshold are what predictions show.
    """
    order = sorted(range(len(results)), key=lambda i: (results[i]["errors"], results[i]["mean"]["log_loss"]))
    return [results[i]["layer"] for i in order]


def print_terminal_chart(results: List[Dict], ranking: List[int], category_sizes: Dict[str, int]):
    color = sys.stdout.isatty()
    best = ranking[0]
    top = set(ranking[:3])
    cats = list(category_sizes)

    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    max_err = max(max(r["errors"] for r in results), 1)
    width = 24
    n = sum(category_sizes.values())

    print("\n" + paint(f" Layer sweep: out-of-fold errors by layer, out of {n} samples (shorter bar is better) ", "1;97;44"))
    print(paint("  Category columns show errors per category: " + ", ".join(f"{c} (n={category_sizes[c]})" for c in cats), "2"))
    header = f"  {'layer':>9}  {'':{width}}  {'errors':>6}  {'ROC-AUC':>7}  {'log loss':>8}  " + "  ".join(f"{c[:10]:>10}" for c in cats)
    print(paint(header, "2"))
    for r in results:
        L = r["layer"]
        filled = round(width * r["errors"] / max_err)
        bar_code = "92" if L == best else "96" if L in top else "34"
        bar = paint("█" * filled, bar_code) + paint("·" * (width - filled), "2")
        tag = paint(" ★ best", "1;92") if L == best else paint(" ▲ top 3", "96") if L in top else ""
        warn = paint(" (not converged)", "33") if not r["converged"] else ""
        cells = []
        for c in cats:
            e = r["by_category"][c]
            cells.append(paint(f"{e:>10}", "91" if e else "2"))
        label = f"{L:>3} ({L - len(results):>3})"
        print(
            f"  {label}  {bar}  {r['errors']:>6}  {r['mean']['roc_auc']:>7.3f}  {r['mean']['log_loss']:>8.4f}  "
            + "  ".join(cells) + tag + warn
        )


def write_report(path: Path, payload: Dict):
    html = REPORT_TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(payload).replace("</", "<\\/")
    path.write_text(html.replace("/*__SWEEP_DATA__*/null", data), encoding="utf-8")


def main():
    settings = load_settings()
    train_cfg = settings["train"]
    sweep_cfg = settings["sweep"]
    sel = preparse_selection(settings)

    parser = argparse.ArgumentParser(
        description="Cross-validate a linear probe on every hidden layer and report which layer performs best."
    )
    add_selection_args(parser, settings)
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Also use JSON files in the done directory (files are never moved by the sweep)",
    )
    parser.add_argument("--prep-dir", type=str, default=sel["prep_dir"], help="(default: the selected head's prep_dir)")
    parser.add_argument("--done-dir", type=str, default=sel["done_dir"], help="(default: the selected head's done_dir)")
    parser.add_argument("--model-path", type=str, default=sel["model_path"], help="(default: the selected model's path)")
    parser.add_argument("--batch-size", type=int, default=train_cfg["batch_size"], help="(default: settings.json train.batch_size)")
    parser.add_argument("--random-state", type=int, default=train_cfg["random_state"], help="(default: settings.json train.random_state)")
    parser.add_argument("--folds", type=int, default=sweep_cfg["folds"], help="Cross-validation folds (default: settings.json sweep.folds)")
    parser.add_argument(
        "--save-top",
        type=int,
        default=sweep_cfg["save_top"],
        help="Refit the N best layers on all data and save them as probe_L{layer}.npz; -1 saves every layer (default: settings.json sweep.save_top)",
    )
    parser.add_argument("--output-dir", type=str, default=sweep_cfg["output_dir"], help="(default: settings.json sweep.output_dir)")
    parser.add_argument("--report", type=str, default=sweep_cfg["report_path"], help="(default: settings.json sweep.report_path)")
    args = parser.parse_args()

    dirs = [Path(args.prep_dir)] + ([Path(args.done_dir)] if args.include_done else [])
    json_files = collect_json_files(dirs)
    if not json_files:
        hint = "" if args.include_done else " Add --include-done to also use previously trained files."
        print(f"No JSON files found in {', '.join(map(str, dirs))}.{hint}")
        sys.exit(0)

    texts: List[str] = []
    labels: List[int] = []
    categories: List[str] = []
    print(f"Using {len(json_files)} file(s):")
    for f in json_files:
        items = load_dataset_from_json(f, with_category=True)
        print(f"  {f} ({len(items)} samples)")
        for t, l, c in items:
            texts.append(t)
            labels.append(l)
            categories.append(c)
    y = np.array(labels)

    classes, counts = np.unique(y, return_counts=True)
    if set(classes) != {0, 1}:
        print(f"Error: the sweep expects binary labels 0/1, found {classes.tolist()}.")
        sys.exit(1)
    if counts.min() < args.folds:
        print(f"Error: the smallest class has {counts.min()} samples, fewer than --folds {args.folds}.")
        sys.exit(1)
    category_sizes = dict(sorted(Counter(categories).items()))

    print(f"\nInitializing feature extractor using '{args.model_path}'...")
    extractor = FeatureExtractor(model_path=args.model_path)

    print(f"\nExtracting every hidden layer for {len(texts)} samples in batches of {args.batch_size}...")
    features = extract_all(extractor, texts, args.batch_size)
    num_states = features.shape[0]
    print(f"Feature tensor shape: {features.shape} (hidden states × samples × hidden_dim)")

    # One fold assignment reused for every layer, so layers are compared on identical splits
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)

    print(f"\nCross-validating {num_states} layers with {args.folds} folds...")
    results = []
    for layer in range(num_states):
        r = cross_validate_layer(features[layer], y, folds)
        summarize_errors(r, y, categories)
        r["layer"] = layer
        results.append(r)
        if sys.stdout.isatty():
            sys.stdout.write(f"\r  layer {layer + 1}/{num_states}")
            sys.stdout.flush()
    if sys.stdout.isatty():
        print()

    ranking = rank_layers(results)
    print_terminal_chart(results, ranking, category_sizes)

    saved = []
    n_save = num_states if args.save_top < 0 else min(args.save_top, num_states)
    if n_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for layer in ranking[:n_save]:
            probe = new_probe()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                probe.fit(features[layer], y)
            out = out_dir / f"probe_L{layer}.npz"
            np.savez(
                out,
                weights=probe.coef_,
                bias=probe.intercept_,
                layer=np.array(layer),
                model_path=np.array(args.model_path),
            )
            saved.append(str(out))
        print(f"\nSaved {len(saved)} probe(s) refit on all data to '{out_dir}':")
        for s in saved:
            print(f"  {s}")

    payload = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model_path": args.model_path,
        "files": [str(f) for f in json_files],
        "n_samples": int(len(y)),
        "class_counts": {"clean": int(counts[0]), "pii": int(counts[1])},
        "category_sizes": category_sizes,
        "samples": [{"text": t, "label": l, "category": c} for t, l, c in zip(texts, labels, categories)],
        "folds": args.folds,
        "num_states": num_states,
        "ranking": ranking,
        "saved": saved,
        "layers": results,
    }
    report_path = Path(args.report)
    write_report(report_path, payload)
    print(f"\nReport written to '{report_path}'. Open it with: open {report_path}")

    best = results[ranking[0]]
    print(
        f"\nBest layer: {ranking[0]} (= {ranking[0] - num_states}) | "
        f"{best['errors']} errors of {len(y)} | "
        f"ROC-AUC {best['mean']['roc_auc']:.3f} ± {best['std']['roc_auc']:.3f} | "
        f"log loss {best['mean']['log_loss']:.4f}"
    )
    print(f"To make it the default, set \"layer\": {ranking[0]} on model '{sel['model_name']}' in settings.json and run train.py.")


if __name__ == "__main__":
    main()
