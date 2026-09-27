# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A representation-probing pipeline: mean-pooled hidden states from a local Qwen2.5-1.5B are fed to a `LogisticRegression` linear probe (currently trained for PII detection, label 1 = PII, 0 = clean). Only the probe's weights are saved (`probe_weights.npz`, keys `weights` shape `(1, hidden_dim)`, `bias`, `layer`, `model_path`); inference is the LLM forward pass plus a NumPy dot product and sigmoid.

There is no test suite or linter. Dependencies live in `.venv` (Python 3.14); `requirements.txt` pins the exact versions (`pip freeze`), since features depend on the torch/transformers versions. `models/`, `.venv/`, `probes*/` and generated sweep reports are gitignored. Use `source .venv/bin/activate` first. All scripts use relative default paths, so run them from the repo root.

## Commands

```bash
python train.py                       # validate on a 30% split, refit on 100%, write probe_weights.npz; data stays in prep/
python train.py --move                # same, then archive the JSON files to training/done/
python predict.py "some text"         # single-shot inference
python predict.py -i                  # interactive REPL (model stays loaded)
python predict.py --file lines.txt    # classify one line per row
python predict.py --file doc.txt --whole   # classify a whole document (also works with piped stdin)
echo "text" | python predict.py       # piped stdin mode
python extract_features.py "text"     # inspect the raw embedding
python sweep.py --include-done         # cross-validate a probe on every layer, write sweep_report.html + probes/probe_L{n}.npz
python regressions.py                  # score a probe (--weights) against the past-failure test cases; exits 1 if any fail
```

`train.py` flags (defaults from `settings.json`): `--layer`, `--batch-size`, `--test-size`, `--random-state`, `--model-path`, `--output-weights`, `--prep-dir`, `--done-dir`, `--move`/`--no-move` (`train.move_to_done`, default false). There is no single-test command because there are no tests.

## Architecture

- `extract_features.py`: `FeatureExtractor` loads the model in fp16 (on MPS if available, otherwise CPU). `extract()` returns attention-mask-aware mean-pooled hidden states from `hidden_states[layer_index]`. It returns `(hidden_dim,)` for a `str` and `(batch, hidden_dim)` for a `list`. It loads the base model with `AutoModel` (no LM head, so vocabulary logits are never computed). With `max_layer=L` it drops the transformer layers above hidden_states index `L` and replaces the final norm with `Identity`, because transformers ties `hidden_states[-1]` to the normed output. Features up to `L` are bit-identical to the full model's. A truncated extractor rejects negative layer indices; use `resolve_layer()` to convert `-2` to `27`. `predict.py` (and so `regressions.py`) truncates at the probe's stored layer. `train.py` and `sweep.py` always load the full model.
- `train.py` and `predict.py` both import `FeatureExtractor`, so training and serving share one feature path. Any change to pooling, tokenization or dtype in `extract()` changes the feature space, so `probe_weights.npz` must be retrained after it.
- `settings.json` (loaded by `settings.py`) holds the shared defaults: `model_path`, `layer`, `weights_path`, plus `train.*` and `predict.*` sections. Every argparse default reads from it, and CLI flags override it. There are no hardcoded fallbacks, so the scripts exit if the file is missing.
- `predict.py` scores input **one sentence at a time** and reports the highest sentence score, plus which sentences crossed the threshold. The probe is trained on short sentences (median 16 tokens). Pooling a long input into one vector gave 42–50% false positives in testing, and fixed token windows that start mid-sentence also cause false positives. Sentences are split at `.!?` followed by whitespace, or at line breaks, so emails, decimals and IP addresses stay in one piece. Fragments under 4 tokens ("Dr.") are merged into the next sentence. Sentences longer than `predict.chunk_tokens` are split into windows that overlap by `chunk_overlap` tokens. `--chunk-tokens 0` turns chunking off.
- **Regression cases** live in `training/eval/regressions.json` (`eval.regressions_path`): past failures with an expected label, a `note` and an `added` date. Nothing trains on them. `train.py` scores them after saving weights, using the same sentence-chunked path as `predict.py`, and warns if a case's text also appears in the training data. When a new edge case turns up, add the exact case here, and add *varied* examples (with matching PII counterparts) to the training files.
- The **layer index must match** between training and inference. `train.py` saves `layer` (as a positive index) and `model_path` into the `.npz`, and `predict.py` uses the stored layer (it has no `--layer` flag). It falls back to `settings.json` only for older weight files that lack the key, and it warns if the model path differs.
- `sweep.py` compares layers. `FeatureExtractor.extract_all_layers()` returns all 29 hidden states (embeddings + 28 layers) from one forward pass. Each layer gets the same `StratifiedKFold` splits and the same `LogisticRegression` config as `train.py`. Every sample gets an out-of-fold prediction. Layers are ranked by total out-of-fold errors at the 0.5 threshold, then mean log loss; ROC-AUC saturates near 1.0 and isn't used for ranking. Errors are also counted per category (the sample's `category` field, or its file name) and reported with the misclassified texts. The report's category heatmap caps its color scale at the 95th percentile, so layer 0 doesn't wash out the rest. It reads `prep/` (plus `done/` with `--include-done`) and never moves files. It fills `sweep_report_template.html` by replacing the `/*__SWEEP_DATA__*/null` placeholder with JSON, and saves the top `save_top` probes to `sweep.output_dir`. Saved probes store a positive layer index, which `predict.py --weights` uses directly.
- Data lifecycle: `train.py` reads `.json` files from `training/prep/`. Files stay there by default, because the dataset is curated and every retrain should see all of it. With `--move` (or `train.move_to_done: true`) they're moved to `training/done/` after training, with a timestamp appended on name collisions. Accepted formats are a list of `{"text", "label"}` objects, or the same list under a `samples` or `data` key. Running `train.py` again with an empty `prep/` exits without changing anything.
- Training data is split into one file per category (`pii_easy`, `pii_medium`, `pii_hard`, `clean_easy`, `false_positives`), following the labeling policy in `training/LABELING.md`. Each sample has an extra `category` field; `train.py` ignores it and `sweep.py` uses it for per-category errors. Never train on a single category file by itself: most are all one label.
- Training only sees what is in `prep/` at that moment. It does not re-read `training/done/`, so the probe is refit from scratch on the new files alone and overwrites `probe_weights.npz`. To retrain on the full history, copy or move the files from `done/` back into `prep/`.
