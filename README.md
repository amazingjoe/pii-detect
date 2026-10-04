<p align="center">
  <img src="assets/hero-banner.jpg" alt="PII Detect — LLM Feature Extraction & Linear Probe Training" width="100%" />
</p>

# PII Detect: LLM Feature Extraction & Linear Probe Training

A lightweight pipeline that detects PII by reading the internal hidden states of a local **Qwen2.5-1.5B** model and scoring them with a linear probe. There's no fine-tuning and no text generation: inference is a partial forward pass plus a dot product and a sigmoid.

The same pipeline works for any binary text classification (topic, policy compliance, ...) given labeled data.

---

## How it works

1. **Extract features.** Text goes through the base model; the hidden states of one transformer layer are mean-pooled over the tokens (padding excluded) into a single 1,536-dimensional vector.
2. **Train a probe.** A `LogisticRegression` classifier learns to separate PII (label `1`) from clean text (label `0`) in that vector space.
3. **Predict.** Input is split into sentences, each sentence is scored, and the text is flagged if any sentence crosses the threshold. The model only runs up to the probe's layer, since later layers can't affect the result.

```
 training/prep/*.json ──► sweep.py ──► sweep_report.html   (which layer works best?)
          │                   └──────► probes/probe_L{n}.npz
          ▼
      train.py ──► probe_weights.npz ──► predict.py        (sentence-by-sentence scoring)
          │                │
          │                └──────────► regressions.py     (past failures must keep passing)
          ▼
 training/done/  (archive, only with --move)
```

---

## Setup

Requires Python 3.14 (the pinned versions in `requirements.txt` were built on it) and about 4 GB of disk for the default model.

```bash
python3 -m venv .venv               # macOS has no bare `python`; use python3 to create the venv
source .venv/bin/activate           # prompt now starts with (.venv); `python` works inside it
which python                        # must print .../pii-detect/.venv/bin/python
pip install -r requirements.txt     # exact versions the probes were trained with (torch is large)

python setup_model.py               # downloads the default base model into models/
python predict.py "Call Renée at 541-555-0176."   # smoke test
```

If `which python` points anywhere else, or you get `ModuleNotFoundError: numpy`, the venv isn't the one being used. Virtual environments can't be moved or renamed after creation (their `activate` script hard-codes the original path), so rebuild it:

```bash
deactivate 2>/dev/null; rm -rf .venv
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**The base model is not in this repo** (`models/` is gitignored; only the small probe heads are). `setup_model.py` reads the model's `hf_repo` and `path` from `settings.json` and downloads it from Hugging Face:

```bash
python setup_model.py --list                  # configured models and whether each is downloaded
python setup_model.py --model qwen2.5-3b      # a different configured model
python setup_model.py --force                 # download again
```

Or do it by hand with the Hugging Face CLI (installed with `requirements.txt`), putting the files at the model's `path`:

```bash
hf download Qwen/Qwen2.5-1.5B --local-dir models/Qwen2.5-1.5B
```

The default model is [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) (Apache 2.0). The shipped `probe_weights.npz` head was trained on it at layer 12 and works only with that model. To use another model, add it to `settings.json` (see [Choosing a model and head](#choosing-a-model-and-head)), download it, run `sweep.py --model <name>` to pick a layer, and train a head with `train.py --model <name>`.

Apple Silicon (MPS) is used automatically when available, otherwise CPU. Run every script from the repo root; paths are relative.

---

## Project structure

```text
.
├── settings.json               # Models, heads and defaults for every script (CLI flags override)
├── settings.py                 # Loads settings.json
├── setup_model.py              # Download a configured base model from Hugging Face
├── extract_features.py         # FeatureExtractor: pooling, all-layer extraction, early stopping
├── train.py                    # Validate, refit on all data, save probe_weights.npz, run regressions
├── predict.py                  # ProbeClassifier + CLI: sentence chunking, interactive shell
├── sweep.py                    # Cross-validate every layer, per-category errors, HTML report
├── sweep_report_template.html  # Report page that sweep.py fills with results
├── regressions.py              # Score a probe against past failures
├── probe_weights.npz           # Production probe (weights, bias, layer, model_path)
├── requirements.txt            # Pinned package versions
├── models/                     # Local model weights (gitignored)
└── training/
    ├── LABELING.md             # What counts as PII
    ├── prep/                   # Data to train on next
    ├── done/                   # Optional archive (train.py --move)
    └── eval/regressions.json   # Past failures; never trained on
```

---

## Configuration (`settings.json`)

| Key | Used by | Meaning |
|---|---|---|
| `default_model` / `default_head` | all | Which `models` / `heads` entry is used when `--model` / `--head` isn't given |
| `models.<name>.hf_repo` | setup_model | Hugging Face repo to download the model from |
| `models.<name>.path` | all | Local model directory |
| `models.<name>.layer` | train, extract | Hidden-state index to train on (`0` = embeddings, `1`–`28` = layer outputs for Qwen2.5-1.5B; negative counts from the top). Layer counts differ per model, so it lives with the model |
| `heads.<name>.weights_path` | train, predict, regressions | Probe file for this head. A `{model}` placeholder becomes the model name, e.g. `probes/pii_{model}.npz` |
| `heads.<name>.prep_dir` / `done_dir` | train, sweep | Training data for this head, and its optional archive |
| `heads.<name>.regressions_path` | train, regressions | Regression cases for this head |
| `train.*` | train, sweep | Batch size, validation split, random seed, `move_to_done` |
| `predict.threshold` | predict, train, regressions | Probability above which text is flagged |
| `predict.chunk_tokens` / `chunk_overlap` | predict | Longest sentence scored whole, and window overlap for longer ones |
| `sweep.*` | sweep | Folds, output folder, report path, how many probes to save |

`predict.py` doesn't use the model's `layer`: it uses the layer stored in the probe file, so a probe can never be scored on the wrong layer.

### Choosing a model and head

A **model** is the frozen base LLM that produces the hidden states. A **head** is the linear probe trained on top of it (the task: PII, or anything else you have labeled data for). `settings.json` lists the ones you have and which pair is the default. Every script takes `--model` and `--head` to pick another pair for a single run:

```bash
python predict.py "some text"                          # default_model + default_head
python predict.py --model qwen2.5-3b --head pii "some text"
python train.py --head toxicity                        # trains on that head's prep_dir, writes its weights_path
python sweep.py --model qwen2.5-3b                     # which layer suits that model?
```

The low-level flags (`--model-path`, `--layer`, `--weights`, `--prep-dir`, `--output-weights`, ...) still work and override whatever the selection resolved to.

To add a **model**, add an entry (`hf_repo` is what `setup_model.py` downloads; the layer is a guess until you run `sweep.py --model <name>`), then `python setup_model.py --model <name>`:

```json
"models": {
  "qwen2.5-1.5b": { "hf_repo": "Qwen/Qwen2.5-1.5B", "path": "./models/Qwen2.5-1.5B", "layer": 12 },
  "qwen2.5-3b":   { "hf_repo": "Qwen/Qwen2.5-3B",   "path": "./models/Qwen2.5-3B",   "layer": 18 }
}
```

To add a **head**, give it its own data and output files, put labeled JSON in its `prep_dir`, and run `train.py --head <name>`:

```json
"heads": {
  "pii":      { "weights_path": "probe_weights.npz",  "prep_dir": "./training/prep",     "done_dir": "./training/done",     "regressions_path": "./training/eval/regressions.json" },
  "toxicity": { "weights_path": "probe_toxicity.npz", "prep_dir": "./training/toxicity", "done_dir": "./training/toxicity_done", "regressions_path": "./training/eval/toxicity.json" }
}
```

A probe only works with the model it was trained on. The probe file records the model path and layer, and `predict.py` warns on a model mismatch. Training one head on a second model overwrites the first probe unless its `weights_path` contains `{model}`. The labels (1 = positive class) are generic, but the output wording in `predict.py` ("PII DETECTED") is still PII-specific.

---

## Training data

One JSON file per category in `training/prep/`, following [`training/LABELING.md`](training/LABELING.md):

| File | Label | What it tests |
|---|---|---|
| `pii_easy.json` | 1 | Labeled identifiers in standard formats |
| `pii_medium.json` | 1 | Identifiers in natural prose and sensitive contexts |
| `pii_hard.json` | 1 | Obfuscated, code/logs, non-English, lookalike formats used for real people |
| `clean_easy.json` | 0 | Ordinary text |
| `false_positives.json` | 0 | Text that looks like PII but isn't |

```json
{
  "samples": [
    {"text": "My SSN is 219-44-6031.", "label": 1, "category": "pii_hard"},
    {"text": "Purchase order 219-44-6031 covers the laptops.", "label": 0, "category": "false_positives"}
  ]
}
```

A bare list of `{"text", "label"}` objects, or a list under `"data"`, also works. `category` is optional; the sweep falls back to the file name.

Files are combined and shuffled before splitting, so separate category files are fine. Never train on one category file alone: most contain a single label.

---

## Usage

### Pick a layer: `sweep.py`

```bash
python sweep.py                    # data in prep/
python sweep.py --include-done     # also data in done/
open sweep_report.html
```

Runs the model once, then cross-validates a probe on all 29 hidden states using identical folds. Layers are ranked by out-of-fold errors (ROC-AUC saturates near 1.0 and doesn't separate them). The report shows errors per category per layer, the misclassified texts, and per-fold scores. The top probes are saved to `probes/`. The sweep never moves files.

### Train the production probe: `train.py`

```bash
python train.py          # data stays in prep/, so every retrain sees the whole dataset
python train.py --move   # archive the files to done/ afterwards
```

1. Extracts features at `layer` for everything in `prep/`.
2. Reports precision/recall on a held-out 30% split.
3. Refits on 100% of the data and saves `probe_weights.npz` with its layer and model path.
4. Scores the regression cases and warns if any case also appears in the training data.
5. With `--move` (or `train.move_to_done: true`), moves the files to `done/`.

Training only sees `prep/`. That's why files stay there by default: the dataset is curated and keeps growing, and each retrain should see all of it. Archiving to `done/` suits one-off batches; to retrain on archived files, move them back first.

### Predict: `predict.py`

By default `predict.py` prints one JSON object to stdout and sends all status messages (model loading, layer) to stderr, so scripts and agents can parse stdout directly:

```bash
$ python predict.py "Call Renée at 541-555-0176."
{"pii": true, "confidence": 0.9995}
```

`confidence` is the highest sentence score (0 to 1), and `pii` is `confidence > threshold` (0.5 by default, set with `--threshold`). Add `--details` to also get the threshold and the flagged sentences with character offsets:

```json
{"pii": true, "confidence": 0.9959, "threshold": 0.5, "flagged": [{"start": 0, "end": 49, "confidence": 0.9959, "text": "..."}]}
```

```bash
python predict.py --pretty "Call Renée at 541-555-0176."   # human-readable output, for testing
python predict.py -i                                       # interactive shell (always human-readable), model stays loaded
python predict.py --file lines.txt                         # one JSON object per line, with a "line" number
python predict.py --file report.txt --whole                # whole document, one JSON object
cat email.txt | python predict.py --whole                  # piped document
python predict.py --weights probes/probe_L9.npz "..."      # any other probe file
python predict.py --model qwen2.5-3b --head pii "..."      # another model / head from settings.json
```

Each call loads the model, which takes a few seconds. To classify many texts from code, use `ProbeClassifier` directly (see [From Python](#from-python)) so the model loads once. Multi-sentence input is scored per sentence; `--chunk-tokens 0` scores the whole input as one vector instead (not recommended beyond a sentence or two; see below).

### Regression cases: `regressions.py`

```bash
python regressions.py                              # production probe
python regressions.py --weights probes/probe_L9.npz
```

Exits with status 1 if any case fails. When you find an edge case:

1. Add the exact text to `training/eval/regressions.json` with the expected label and a note.
2. Add 10–20 *varied* examples to the training files: the lookalike as clean, plus a matching real-PII version, so the probe learns the context and not the surface pattern.
3. Retrain. The regression report shows whether the fix generalized and whether older cases still pass.

### From Python

```python
from predict import ProbeClassifier
from settings import load_settings, resolve_selection

s = load_settings()
sel = resolve_selection(s)   # defaults; or resolve_selection(s, model="...", head="...")
clf = ProbeClassifier(
    weights_path=sel["weights_path"],
    model_path=sel["model_path"],
    fallback_layer=sel["layer"],
    chunk_tokens=s["predict"]["chunk_tokens"],
    chunk_overlap=s["predict"]["chunk_overlap"],
    batch_size=s["train"]["batch_size"],
)

prob, is_pii = clf.predict("Contact Bob at 555-0143 regarding your invoice.")
for chunk in clf.predict_chunks(document):   # per-sentence scores with character offsets
    print(chunk.start, chunk.end, f"{chunk.prob:.2%}", chunk.text)
```

---

## Technical notes

- **Mean pooling:** `embedding = Σ(H ⊙ M) / ΣM`, where `H` is `(batch, seq_len, hidden_dim)` and `M` is the expanded attention mask, so padding never contributes.
- **Why sentences:** the probe is trained on short texts (median 16 tokens). Pooling a long document into one vector gave 42–50% false positives in testing, and fixed token windows starting mid-sentence also misfired. Sentence scoring caught 100% of test documents with PII and flagged none of the clean ones. Splits happen at `.!?` + whitespace or line breaks (emails, decimals and IPs stay intact); fragments under 4 tokens merge into the next sentence; sentences over `chunk_tokens` are windowed.
- **Early stopping:** layers above the probe's layer never affect its features, so `predict.py` drops them after loading, and the model is loaded without its vocabulary head. Features are bit-identical to the full model's. Measured on 320 sentences: 2.2× faster at layer 14, 4.1× at layer 7. `train.py` and `sweep.py` always use the full model.
- **Layer indices:** probe files store a positive index (`27`, not `-2`), because on a truncated model a negative index would count from the wrong top.
- **Consistency:** training and prediction share `FeatureExtractor`. Any change to pooling, tokenization or dtype changes the feature space and requires retraining.
- **Versions:** `requirements.txt` pins the exact packages. Features depend on the `torch` and `transformers` versions (early stopping, for example, relies on how `transformers` 5.x records hidden states), so upgrade deliberately and retrain after.
- **Probe:** L2-regularized logistic regression (`C=1.0`), so inference after the forward pass is one dot product and a sigmoid.
