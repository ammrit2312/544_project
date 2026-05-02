# Grammar correction (English / Spanish)

Small playground for fixing grammar using two different ideas: one learns **what edits to apply** to each word, the other **writes the corrected sentence** from scratch. Both read the same CSV files under `data/processed/` once you’ve built them.

Work from the **repo root** so paths and imports behave. GPU helps but isn’t required (scripts use MPS, CUDA, or CPU when available).

## Setup

```bash
python3 -m venv .venv && source .venv/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

Install deps from [`requirements.txt`](requirements.txt) as-is. The first line pulls PyTorch built for **CUDA 12.4**; if that doesn’t match your machine, install a suitable PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) first, then install the rest of the packages from the file (everything except the `torch` line).

## Data

```bash
python data_loader.py
```

This downloads English (`juancavallotti/bea-19-fine-tune`) and Spanish (`juancavallotti/multilingual-gec`), merges them, and writes `train.csv`, `val.csv`, and `test.csv` under `data/processed/` with an `lang` column (`en` / `es`).

## The two models (plain English)

**GECTOR** — A pretrained text encoder scores those actions; decoding just applies the edits, optionally running a few passes until the sentence stops changing.

**GECModel** — An encoder reads the input and a decoder generates the corrected text token by token (like a tiny translation model). The pretrained encoder starts frozen in [`train.py`](train.py); only the decoder side learns unless you use other scripts.

### Train

| What | Command |
|------|---------|
| GECTOR on **both** languages (all rows in `train.csv`) | `python train_gector.py` |
| GECTOR **Spanish only** | `python train_gector_es.py` |
| GECModel on the **merged** CSV | `python train.py` |

Optional: `GECTOR_EPOCHS`, `GECTOR_NUM_WORKERS` for training; `GECTOR_MAX_PASSES` affects iterative GECTOR inference.

### Inference

- GECTOR (default checkpoints): `python infer_gector.py`
- GECTOR (Spanish checkpoints from `train_gector_es.py`): `python infer_gector_es.py`
- GECModel: `python infer.py`

Match checkpoint paths at the top of each script to what you trained.

### Evaluation

GECTOR: `python evaluate_gector.py` or `python evaluate_gector_es.py`.

For **GECModel** metrics, copy the loop from `evaluate_gector.py` but use `from infer import correct` and [`gec_metrics.py`](gec_metrics.py). [`evaluate_gec_1.py`](evaluate_gec_1.py) currently imports `infer_gector`, so it isn’t scoring `train.py`’s checkpoint unless you change that locally.

## Main files

- [`data_loader.py`](data_loader.py) — build CSVs from Hugging Face
- [`dataset.py`](dataset.py) — batches for GECModel training
- [`train_gector.py`](train_gector.py), [`train_gector_es.py`](train_gector_es.py), [`train.py`](train.py)
- [`infer_gector.py`](infer_gector.py), [`infer_gector_es.py`](infer_gector_es.py), [`infer.py`](infer.py)
- [`evaluate_gector.py`](evaluate_gector.py), [`evaluate_gector_es.py`](evaluate_gector_es.py), [`gec_metrics.py`](gec_metrics.py)
- [`model/gec_model.py`](model/gec_model.py), [`model/decoder.py`](model/decoder.py), [`model/attention.py`](model/attention.py)

Other `*_gec_ft.py` scripts are alternate experiments.
