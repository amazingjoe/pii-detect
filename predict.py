import argparse
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from extract_features import FeatureExtractor
from settings import load_settings

# Sentence boundary: terminal punctuation followed by whitespace, or a line break.
# Requiring whitespace keeps emails, decimals and IP addresses in one piece.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
# Fragments shorter than this (e.g. "Dr." or "No.") are merged into the following sentence
MIN_SENTENCE_TOKENS = 4


class Chunk(NamedTuple):
    start: int   # character offset into the original text
    end: int
    text: str
    prob: float


class ProbeClassifier:
    """Inference engine combining the LLM feature extractor and linear probe weights.

    Multi-sentence inputs are split into sentences and each one is scored separately,
    because the probe is trained on short, whole sentences: a single mean-pooled vector
    over a long input drifts away from what it learned, and fixed token windows that
    start mid-sentence cause false positives. Sentences longer than chunk_tokens (logs,
    JSON, run-ons) are split further into overlapping token windows. The text's score
    is the highest chunk score.
    """

    def __init__(
        self,
        weights_path: str,
        model_path: str,
        fallback_layer: int,
        chunk_tokens: int,
        chunk_overlap: int,
        batch_size: int,
        extractor: Optional[FeatureExtractor] = None,
    ):
        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Probe weights not found at '{weights_path}'. Run 'python train.py' first."
            )
        if chunk_tokens > 0 and not 0 <= chunk_overlap < chunk_tokens:
            raise ValueError("chunk_overlap must be at least 0 and smaller than chunk_tokens.")

        self.weights_path = weights_path
        self.model_path = model_path
        data = np.load(weights_path)
        self.weights = data["weights"]  # shape: (1, hidden_dim)
        self.bias = data["bias"]        # shape: (1,)

        # The probe only makes sense on the layer it was trained on, so prefer the stored value
        if "layer" in data:
            self.layer_index = int(data["layer"])
        else:
            self.layer_index = fallback_layer
            print(
                f"Warning: '{weights_path}' has no stored layer (trained before settings.json); "
                f"using layer {fallback_layer} from settings. Retrain to embed it."
            )
        if "model_path" in data and str(data["model_path"]) != model_path:
            print(
                f"Warning: probe was trained with model '{data['model_path']}' "
                f"but '{model_path}' is being loaded."
            )
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        chunking = (
            f"sentence chunks, max {chunk_tokens} tokens, {chunk_overlap} overlap" if chunk_tokens > 0 else "chunking off"
        )
        # Reuse an already-loaded model when the caller has one (e.g. train.py). Otherwise load
        # one that stops at the probe's layer, since layers above it never affect the result.
        self.extractor = extractor or FeatureExtractor(model_path=model_path, max_layer=self.layer_index)
        self.layer_index = self.extractor.resolve_layer(self.layer_index)
        print(f"Using layer {self.layer_index} ({chunking}).")

    def chunk_spans(self, text: str) -> List[Tuple[int, int]]:
        """Returns (start, end) character spans: one per sentence, with long sentences windowed."""
        if self.chunk_tokens <= 0:
            return [(0, len(text))]

        sentences, pos = [], 0
        for m in SENTENCE_BREAK.finditer(text):
            sentences.append((pos, m.start()))
            pos = m.end()
        sentences.append((pos, len(text)))
        sentences = [(s, e) for s, e in sentences if text[s:e].strip()]
        if not sentences:
            return [(0, len(text))]

        merged = []
        carry_start = None
        for i, (s, e) in enumerate(sentences):
            s = s if carry_start is None else carry_start
            is_last = i == len(sentences) - 1
            if self._num_tokens(text[s:e]) < MIN_SENTENCE_TOKENS and not is_last:
                carry_start = s
                continue
            carry_start = None
            merged.append((s, e))

        spans = []
        for s, e in merged:
            spans.extend(self._token_windows(text, s, e))
        return spans

    def _num_tokens(self, text: str) -> int:
        return len(self.extractor.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _token_windows(self, text: str, start: int, end: int) -> List[Tuple[int, int]]:
        """Splits text[start:end] into overlapping windows of at most chunk_tokens tokens."""
        offsets = self.extractor.tokenizer(
            text[start:end], return_offsets_mapping=True, add_special_tokens=False
        )["offset_mapping"]
        if len(offsets) <= self.chunk_tokens:
            return [(start, end)]

        stride = self.chunk_tokens - self.chunk_overlap
        windows = []
        for first in range(0, len(offsets), stride):
            last = min(first + self.chunk_tokens, len(offsets)) - 1
            windows.append((start + offsets[first][0], start + offsets[last][1]))
            if last == len(offsets) - 1:
                break
        return windows

    def predict_chunks(self, text: str) -> List[Chunk]:
        """Scores every chunk of the text, in document order."""
        spans = self.chunk_spans(text)
        pieces = [text[s:e].strip() or text[s:e] for s, e in spans]

        probs = []
        for i in range(0, len(pieces), self.batch_size):
            vecs = self.extractor.extract(pieces[i : i + self.batch_size], layer_index=self.layer_index)
            logits = vecs @ self.weights[0] + self.bias[0]
            probs.extend((1.0 / (1.0 + np.exp(-logits))).tolist())

        return [Chunk(s, e, piece, float(p)) for (s, e), piece, p in zip(spans, pieces, probs)]

    def predict(self, text: str, threshold: float = 0.5) -> Tuple[float, bool]:
        """Predicts whether the given text contains PII.

        Args:
            text: Input string to evaluate.
            threshold: Probability decision threshold (default: 0.5).

        Returns:
            Tuple[float, bool]: (highest chunk probability, is_pii_detected)
        """
        prob = max(c.prob for c in self.predict_chunks(text))
        return prob, prob > threshold


def print_chunk_details(chunks: List[Chunk], threshold: float, indent: str = ""):
    """Lists the chunks above the threshold, or the highest-scoring one if none are."""
    if len(chunks) == 1:
        return
    flagged = [c for c in chunks if c.prob > threshold]
    print(f"{indent}Chunks:      {len(chunks)} scored, {len(flagged)} above threshold")
    shown = flagged or [max(chunks, key=lambda c: c.prob)]
    label = "Flagged" if flagged else "Highest"
    for c in shown:
        snippet = c.text if len(c.text) <= 160 else c.text[:157] + "..."
        print(f"{indent}  {label} chars {c.start}-{c.end} ({c.prob * 100:6.2f}%): {snippet!r}")


def run_interactive(classifier: ProbeClassifier, threshold: float = 0.5):
    """Runs a persistent interactive REPL where the model stays loaded in memory."""
    print("\n" + "=" * 60)
    print("  Interactive Probe Shell (Model loaded in memory)")
    ext = classifier.extractor
    print(f"  Model:      {classifier.model_path}")
    print(f"  Head:       {classifier.weights_path}")
    print(
        f"  Layer:      {classifier.layer_index} of {ext.full_num_layers} "
        f"({classifier.layer_index - ext.full_num_layers - 1} from top)"
    )
    print(f"  Threshold:  {threshold}")
    print("  Type any text to classify. Type 'exit', 'quit', or Ctrl+C to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit", "q"}:
                print("Exiting.")
                break

            chunks = classifier.predict_chunks(user_input)
            prob = max(c.prob for c in chunks)
            status = "\033[91m[PII DETECTED]\033[0m" if prob > threshold else "\033[92m[CLEAN]\033[0m"
            print(f"  Result:     {status} (Confidence: {prob * 100:.2f}%)")
            print_chunk_details(chunks, threshold, indent="  ")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def classify_document(classifier: ProbeClassifier, text: str, source: str, threshold: float):
    chunks = classifier.predict_chunks(text)
    prob = max(c.prob for c in chunks)
    print("\n--- Inference Result ---")
    print(f"Input:       {source} ({len(text)} characters)")
    print(f"Confidence:  {prob * 100:.2f}%")
    print(f"Prediction:  {'[PII DETECTED]' if prob > threshold else '[CLEAN]'}")
    print_chunk_details(chunks, threshold)


def main():
    settings = load_settings()
    predict_cfg = settings["predict"]

    parser = argparse.ArgumentParser(
        description="Run inference on text using the trained linear probe (single-shot, file, or interactive mode)."
    )
    parser.add_argument(
        "text",
        nargs="?",
        type=str,
        default=None,
        help="Text to classify (if omitted or '-', enters interactive/piped mode)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch an interactive shell (keeps the model loaded for instant responses)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to a text file with lines to classify one by one",
    )
    parser.add_argument(
        "--whole",
        action="store_true",
        help="Treat --file or piped stdin as one document instead of one text per line",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=settings["weights_path"],
        help="Path to trained probe weights (default: settings.json weights_path)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=settings["model_path"],
        help="Path to base LLM (default: settings.json model_path)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=predict_cfg["threshold"],
        help="Classification probability threshold (default: settings.json predict.threshold)",
    )
    parser.add_argument(
        "--chunk-tokens",
        type=int,
        default=predict_cfg["chunk_tokens"],
        help="Input is scored one sentence at a time; sentences longer than this many tokens are split into windows. 0 scores the whole input as one vector (default: settings.json predict.chunk_tokens)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=predict_cfg["chunk_overlap"],
        help="Tokens shared by neighboring windows inside a long sentence, so an identifier on a boundary lands whole in one window (default: settings.json predict.chunk_overlap)",
    )
    args = parser.parse_args()

    classifier = ProbeClassifier(
        weights_path=args.weights,
        model_path=args.model_path,
        fallback_layer=settings["layer"],
        chunk_tokens=args.chunk_tokens,
        chunk_overlap=args.chunk_overlap,
        batch_size=settings["train"]["batch_size"],
    )

    # 1. File mode
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' not found.")
            sys.exit(1)
        content = file_path.read_text(encoding="utf-8")
        if args.whole:
            classify_document(classifier, content, f"'{args.file}'", args.threshold)
            return
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        print(f"Classifying {len(lines)} lines from '{args.file}':\n")
        for line in lines:
            prob, is_pii = classifier.predict(line, threshold=args.threshold)
            tag = "[PII]" if is_pii else "[CLEAN]"
            print(f"{tag:7} ({prob * 100:6.2f}%) | {line}")
        return

    # 2. Piped stdin mode (e.g. echo "text" | python predict.py)
    if (args.text == "-" or args.text is None) and not sys.stdin.isatty():
        if args.whole:
            classify_document(classifier, sys.stdin.read(), "stdin", args.threshold)
            return
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            prob, is_pii = classifier.predict(line, threshold=args.threshold)
            tag = "[PII]" if is_pii else "[CLEAN]"
            print(f"{tag:7} ({prob * 100:6.2f}%) | {line}")
        return

    # 3. Interactive REPL mode
    if args.interactive or args.text is None:
        run_interactive(classifier, threshold=args.threshold)
        return

    # 4. Single-shot CLI argument mode
    chunks = classifier.predict_chunks(args.text)
    prob = max(c.prob for c in chunks)
    print("\n--- Inference Result ---")
    print(f"Input:       '{args.text}'")
    print(f"Confidence:  {prob * 100:.2f}%")
    print(f"Prediction:  {'[PII DETECTED]' if prob > args.threshold else '[CLEAN]'}")
    print_chunk_details(chunks, args.threshold)


if __name__ == "__main__":
    main()
