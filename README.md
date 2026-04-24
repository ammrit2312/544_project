# Grammatical error correction (GEC)

This repository implements **grammatical error correction** with two main model families:

1. **GECToR-style** — predict a discrete **edit label** per source word (`$KEEP`, `$DELETE`, `$REPLACE_*`, `$INSERT_*`), then apply rule-based string edits (optionally multiple passes).
2. **GECModel (seq2seq)** — **XLM-RoBERTa** encodes the noisy sentence; a **decoder** (custom transformer in `model/`, or an alternate stack in `model2.py`) generates the corrected sentence token by token.

There is also an **mT5** seq2seq baseline (`train_mt5.py` / `infer_mt5.py`) using `google/mt5-small` by default. - this was not implemented and tested.


---

## Data

Place processed CSVs under `data/processed/` (paths are hard-coded in the training scripts):

| Column       | Meaning                          |
|-------------|-----------------------------------|
| `incorrect` | noisy / learner text            |
| `correct`   | reference correction            |
| `lang`      | optional; e.g. `en`, `es` for filtered eval |

Typical files: `train.csv`, `val.csv`, `test.csv`.

---

## Environment setup

1. **Create a virtual environment (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   On Windows: `.\.venv\Scripts\activate`

2. **Upgrade pip**

   ```bash
   python -m pip install -U pip
   ```

3. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

   The root [requirements.txt](requirements.txt) uses the PyTorch **CUDA 12.4** wheel index. If you use **CPU-only** or **Apple Silicon**, see the comment block at the top of `requirements.txt` for alternative `pip install torch` commands, then install the remaining packages.


---

## GECModel (seq2seq)

There are **two** implementations; use the **matching** train and inference story for each.

### A. `model/` + `train_gec_ft.py`

- **Train:** `python train_gec_ft.py`  
  - Dataset: [dataset.py](dataset.py)  
  - Model: [model/gec_model.py](model/gec_model.py)  
  - Default checkpoint: `checkpoints/gec_model_encoder_ft.pt`

- **Infer:** `python infer_gec_ft.py`  
  - Loads `checkpoints/gec_model_encoder_ft.pt` and runs greedy decoding from the built-in decoder.

- **Evaluate:** `python evaluate_gec_ft.py`  
  - Uses `infer_gec_ft.correct` on `data/processed/test.csv` (default: samples 200 rows per language block in `__main__`).  
  - **Metric:** token-set overlap F0.5 (see script).

---

## GECToR

**Idea:** multi-class classification over a **finite label vocabulary** (frequent edits mined from data, plus `$KEEP` / `$DELETE`). Inference applies predicted labels to words; `GECTOR_MAX_PASSES` (used in the infer script) can repeat until the string stabilizes.

In this tree, **training entry points** call `from train_gector import main` (see [train_gector_es.py](train_gector_es.py)). You need a **`train_gector.py`** module in the project (or on `PYTHONPATH`) that defines `main`, `GECTorModel`, and related symbols. If that file is missing, install or add it before running the GECToR wrappers.

- **Train (wrapper example):** `python train_gector_es.py` — passes checkpoint / label paths and a `lang` filter into `train_gector.main` (see file for defaults).

- **Infer:** `python infer_gector_es.py` — loads checkpoint + label pickle, runs correction.

- **Evaluate:** `python evaluate_gector_es.py` — LCS-based F0.5 and exact match on Spanish rows by default (`lang == 'es'`).

For a **language-agnostic** workflow, use or author a `train_gector.py` that does not filter by `lang`, and point infer/eval scripts at the same artifacts (or duplicate the wrapper without the `es` suffix once the base module exists).

---

## mT5 baseline

- **Train:** `python train_mt5.py` — optional env: `MT5_MODEL`, `MT5_BATCH`, `MT5_EPOCHS`, `MT5_LR`. Checkpoint: `checkpoints/mt5_gec.pt`.

- **Infer:** `python infer_mt5.py`

- **Evaluate:** `python evaluate_mt5.py` — LCS-based F0.5 (same family of metric as GECToR eval script).

---

## Other utilities

- [data_loader.py](data_loader.py) — Hugging Face `datasets` → train/val/test CSV split (run as script if needed).
- [attention2.py](attention2.py) — standalone attention/decoder experiment with a `__main__` smoke block.

---

## Footnote: `*_es` scripts

Files like `train_gector_es.py`, `infer_gector_es.py`, and `evaluate_gector_es.py` are **thin wrappers** or evaluators focused on Spanish (`lang == 'es'` where applicable). The general flow is the same as the non-suffixed pattern once you use a full `train_gector` stack and shared CSV layout.
