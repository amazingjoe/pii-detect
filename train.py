import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from extract_features import FeatureExtractor
from predict import ProbeClassifier
from regressions import find_leaks, load_cases, print_report, run_regressions
from settings import add_selection_args, load_settings, preparse_selection


def load_dataset_from_json(file_path: Path, with_category: bool = False) -> List[Tuple]:
    """Loads text and label pairs from a JSON file.

    With with_category=True, returns (text, label, category) triples instead. The category
    comes from each item's "category" field, falling back to the file name.
    
    Accepts:
      - A list of objects: [{"text": "...", "label": 1}, ...]
      - A dictionary with a key containing a list: {"samples": [...]} or {"data": [...]}
      - A single object: {"text": "...", "label": 1}
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    samples = []
    if isinstance(content, list):
        raw_items = content
    elif isinstance(content, dict):
        if "samples" in content and isinstance(content["samples"], list):
            raw_items = content["samples"]
        elif "data" in content and isinstance(content["data"], list):
            raw_items = content["data"]
        else:
            raw_items = [content]
    else:
        raise ValueError(f"Unsupported JSON structure in {file_path}")

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        label = item.get("label")
        if text is not None and label is not None:
            if with_category:
                samples.append((str(text).strip(), int(label), str(item.get("category") or file_path.stem)))
            else:
                samples.append((str(text).strip(), int(label)))

    return samples


def extract_file_features(
    extractor: FeatureExtractor,
    texts: List[str],
    file_idx: int,
    total_files: int,
    filename: str,
    batch_size: int,
    layer_index: int,
) -> np.ndarray:
    """Extracts features for a file in batches and displays a progress counter."""
    total_entries = len(texts)
    embeddings = []
    for i in range(0, total_entries, batch_size):
        chunk = texts[i : i + batch_size]
        features = extractor.extract(chunk, layer_index=layer_index)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        embeddings.append(features)

        current = min(i + len(chunk), total_entries)
        counter_str = f"  [{current}/{total_entries} of {file_idx}/{total_files} training files] ({filename})"
        if sys.stdout.isatty():
            sys.stdout.write(f"\r{counter_str}")
            sys.stdout.flush()
        else:
            print(counter_str)

    if sys.stdout.isatty():
        print()  # Newline after completing file
    return np.vstack(embeddings)


def move_file_to_done(source_file: Path, done_dir: Path) -> Path:
    """Moves a file to the done directory, resolving any name collisions."""
    dest_file = done_dir / source_file.name
    if dest_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest_file = done_dir / f"{source_file.stem}_{timestamp}{source_file.suffix}"
    shutil.move(str(source_file), str(dest_file))
    return dest_file


def main():
    settings = load_settings()
    train_cfg = settings["train"]
    sel = preparse_selection(settings)

    parser = argparse.ArgumentParser(
        description="Extract features using Qwen model and train a linear probe for PII detection."
    )
    add_selection_args(parser, settings)
    parser.add_argument(
        "--prep-dir",
        type=str,
        default=sel["prep_dir"],
        help="Directory containing JSON files ready to be trained on (default: the selected head's prep_dir)",
    )
    parser.add_argument(
        "--done-dir",
        type=str,
        default=sel["done_dir"],
        help="Directory where processed JSON files are moved (default: the selected head's done_dir)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=sel["model_path"],
        help="Path to the base model, overriding --model (default: the selected model's path)",
    )
    parser.add_argument(
        "--output-weights",
        type=str,
        default=sel["weights_path"],
        help="Destination path for trained probe weights (default: the selected head's weights_path)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=sel["layer"],
        help="Transformer layer index for feature extraction; saved into the weights file (default: the selected model's layer)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=train_cfg["batch_size"],
        help="Batch size for feature extraction forward pass (default: settings.json train.batch_size)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=train_cfg["test_size"],
        help="Proportion of the dataset to include in the validation split (default: settings.json train.test_size)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=train_cfg["random_state"],
        help="Random state for reproducibility (default: settings.json train.random_state)",
    )
    move = parser.add_mutually_exclusive_group()
    move.add_argument(
        "--move",
        dest="move",
        action="store_true",
        default=train_cfg["move_to_done"],
        help="Move processed JSON files from prep to done after training (default: settings.json train.move_to_done)",
    )
    move.add_argument(
        "--no-move",
        dest="move",
        action="store_false",
        help="Leave processed JSON files in prep",
    )
    args = parser.parse_args()

    prep_dir = Path(args.prep_dir)
    done_dir = Path(args.done_dir)

    prep_dir.mkdir(parents=True, exist_ok=True)
    done_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(prep_dir.glob("*.json")))
    if not json_files:
        print(f"No JSON files found in '{prep_dir}'. Place training data files there and re-run.")
        sys.exit(0)

    total_files = len(json_files)
    print(f"Found {total_files} file(s) in '{prep_dir}':")
    for idx, f in enumerate(json_files, start=1):
        print(f"  [{idx}/{total_files}] {f.name}")

    # 1. Initialize Feature Extractor
    print(f"\nInitializing feature extractor using '{args.model_path}'...")
    extractor = FeatureExtractor(model_path=args.model_path)

    all_embeddings: List[np.ndarray] = []
    all_labels: List[int] = []
    all_texts: List[str] = []
    processed_files: List[Path] = []

    print(f"\nExtracting features from layer {args.layer} in batches of {args.batch_size}...")
    for file_idx, f in enumerate(json_files, start=1):
        items = load_dataset_from_json(f)
        if not items:
            print(f"Warning: No valid samples found in {f.name}, skipping.")
            continue

        file_texts = [item[0] for item in items]
        file_labels = [item[1] for item in items]
        total_entries = len(file_texts)

        print(f"\nProcessing file {file_idx}/{total_files}: '{f.name}' ({total_entries} entries)")
        embeddings = extract_file_features(
            extractor=extractor,
            texts=file_texts,
            file_idx=file_idx,
            total_files=total_files,
            filename=f.name,
            batch_size=args.batch_size,
            layer_index=args.layer,
        )
        all_embeddings.append(embeddings)
        all_labels.extend(file_labels)
        all_texts.extend(file_texts)
        processed_files.append(f)

    if not all_embeddings:
        print("Error: No valid training samples could be extracted from the JSON files.")
        sys.exit(1)

    X = np.vstack(all_embeddings)
    y = np.array(all_labels)

    print(f"\nTotal dataset size across {len(processed_files)} file(s): {len(y)} samples")
    unique_classes, counts = np.unique(y, return_counts=True)
    for cls, count in zip(unique_classes, counts):
        class_name = "PII (1)" if cls == 1 else "Clean (0)" if cls == 0 else f"Class {cls}"
        print(f"  {class_name}: {count} samples")

    if len(unique_classes) < 2:
        print("Error: Dataset must contain at least two classes to train a classifier.")
        sys.exit(1)

    print(f"Feature matrix X shape: {X.shape}")

    # 2. Train / Test Split
    stratify = y if min(counts) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )
    print(f"Training set: {len(X_train)} samples | Validation set: {len(X_test)} samples")

    # 3. Train the Linear Probe
    print("\nFitting logistic regression probe...")
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train, y_train)

    # 4. Evaluation
    preds = probe.predict(X_test)
    print("\n--- Validation Results ---")
    target_names = ["Clean", "PII"] if set(unique_classes) == {0, 1} else [str(c) for c in unique_classes]
    print(classification_report(y_test, preds, target_names=target_names))

    # The split above only measures quality; the saved probe learns from every sample
    print(f"Refitting on all {len(y)} samples for the saved probe...")
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X, y)

    # 5. Test Inference Sample
    test_prompt = "Reach out to Jane at jane.doe@corp.net or check our public API docs."
    test_vec = extractor.extract(test_prompt, layer_index=args.layer).reshape(1, -1)
    prob_pii = probe.predict_proba(test_vec)[0][1]

    print("--- Test Inference ---")
    print(f"Unseen text: '{test_prompt}'")
    print(f"PII Confidence Score: {prob_pii * 100:.2f}%")
    print(f"Prediction: {'PII Detected' if prob_pii > 0.5 else 'Clean'}")

    # 6. Save Probe Weights
    # The layer and model are stored alongside the weights so predict.py can't use mismatched features.
    # The layer is stored as a positive index, which means the same thing on a truncated model.
    layer = extractor.resolve_layer(args.layer)
    np.savez(
        args.output_weights,
        weights=probe.coef_,
        bias=probe.intercept_,
        layer=np.array(layer),
        model_path=np.array(args.model_path),
    )
    print(f"\nProbe weights saved to '{args.output_weights}' (layer {layer}).")

    # 7. Regression cases: past failures, never trained on, scored the same way predict.py scores
    cases_path = Path(sel["regressions_path"])
    if cases_path.exists():
        cases = load_cases(cases_path)
        classifier = ProbeClassifier(
            weights_path=args.output_weights,
            model_path=args.model_path,
            fallback_layer=args.layer,
            chunk_tokens=settings["predict"]["chunk_tokens"],
            chunk_overlap=settings["predict"]["chunk_overlap"],
            batch_size=args.batch_size,
            extractor=extractor,
        )
        threshold = settings["predict"]["threshold"]
        results = run_regressions(classifier, cases, threshold)
        print_report(results, threshold, leaks=find_leaks(cases, all_texts))

    # 8. Move processed JSON files to done/
    if args.move:
        print(f"\nMoving processed files from '{prep_dir}' to '{done_dir}':")
        for f in processed_files:
            moved_path = move_file_to_done(f, done_dir)
            print(f"  Moved: {f.name} -> {moved_path}")
    else:
        print(f"\nLeft training files in '{prep_dir}' (use --move to archive them to '{done_dir}').")


if __name__ == "__main__":
    main()
