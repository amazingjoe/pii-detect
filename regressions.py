"""Regression suite: past failures kept as permanent test cases.

Cases live in settings.json eval.regressions_path and are never trained on. They are scored
through ProbeClassifier.predict_chunks, the same path predict.py uses, so a pass here means
the case passes for real users of the probe.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from predict import ProbeClassifier
from settings import load_settings


def load_cases(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    cases = content["samples"] if isinstance(content, dict) else content
    return [c for c in cases if "text" in c and "label" in c]


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def find_leaks(cases: List[Dict], training_texts: Iterable[str]) -> List[Dict]:
    """Cases whose text also appears in the training data, so they no longer test generalization."""
    seen = {normalize(t) for t in training_texts}
    return [c for c in cases if normalize(c["text"]) in seen]


def run_regressions(classifier: ProbeClassifier, cases: List[Dict], threshold: float) -> List[Dict]:
    results = []
    for case in cases:
        chunks = classifier.predict_chunks(case["text"])
        top = max(chunks, key=lambda c: c.prob)
        predicted = int(top.prob > threshold)
        results.append({**case, "prob": top.prob, "passed": predicted == int(case["label"]), "top_chunk": top.text})
    return results


def print_report(results: List[Dict], threshold: float, leaks: Optional[List[Dict]] = None):
    color = sys.stdout.isatty()

    def paint(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if color else text

    passed = sum(r["passed"] for r in results)
    print(f"\n--- Regression Cases ({passed}/{len(results)} passing, threshold {threshold}) ---")
    for r in results:
        tag = paint("PASS", "1;92") if r["passed"] else paint("FAIL", "1;91")
        expected = "PII" if int(r["label"]) == 1 else "clean"
        snippet = r["text"] if len(r["text"]) <= 90 else r["text"][:87] + "..."
        print(f"  {tag}  expected {expected:5}  p(PII)={r['prob'] * 100:6.2f}%  {snippet!r}")
        if not r["passed"]:
            if r["top_chunk"] != r["text"]:
                top = r["top_chunk"] if len(r["top_chunk"]) <= 90 else r["top_chunk"][:87] + "..."
                print(f"        highest sentence: {top!r}")
            if r.get("note"):
                print(paint(f"        note: {r['note']}", "2"))
    for c in leaks or []:
        snippet = c["text"] if len(c["text"]) <= 90 else c["text"][:87] + "..."
        print(paint(f"  Warning: regression case is also in the training data, so passing it proves little: {snippet!r}", "33"))


def main():
    settings = load_settings()
    predict_cfg = settings["predict"]

    parser = argparse.ArgumentParser(description="Score a trained probe against the regression cases.")
    parser.add_argument("--weights", type=str, default=settings["weights_path"], help="(default: settings.json weights_path)")
    parser.add_argument("--cases", type=str, default=settings["eval"]["regressions_path"], help="(default: settings.json eval.regressions_path)")
    parser.add_argument("--model-path", type=str, default=settings["model_path"], help="(default: settings.json model_path)")
    parser.add_argument("--threshold", type=float, default=predict_cfg["threshold"], help="(default: settings.json predict.threshold)")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        print(f"No regression cases found at '{cases_path}'.")
        sys.exit(0)
    cases = load_cases(cases_path)

    classifier = ProbeClassifier(
        weights_path=args.weights,
        model_path=args.model_path,
        fallback_layer=settings["layer"],
        chunk_tokens=predict_cfg["chunk_tokens"],
        chunk_overlap=predict_cfg["chunk_overlap"],
        batch_size=settings["train"]["batch_size"],
    )
    results = run_regressions(classifier, cases, args.threshold)
    print_report(results, args.threshold)
    # Non-zero exit when anything fails, so this can gate a script or CI step
    sys.exit(0 if all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
