# Quick Start Guide

Complete walkthrough to go from a fresh clone to a trained LightRet-NER model.

---

## Step 0 — Clone and Install

```bash
git clone https://github.com/maheshbadari/LightRET.git
cd LightRET
pip install torch transformers datasets huggingface_hub reportlab
```

---

## Step 1 — Export RetVec Weights (one-time)

RetVec is a pretrained character embedder. This step downloads the TensorFlow
SavedModel and exports its weights to a NumPy archive that PyTorch can load.

```bash
pip install tensorflow retvec   # only needed for this step
python retvec_export.py
```

**Output:** `retvec_v1_weights.npz` (~1 MB)

> Skip if `retvec_v1_weights.npz` already exists in the project root.

---

## Step 2 — Smoke Test (optional but recommended)

Verifies that all modules import correctly, all models run forward passes,
and CoNLL-2003 loads. Uses mocked teacher outputs so no GPU or BERT download
is needed.

```bash
python dry_run.py
```

Expected output: `34 checks — all PASS`.

---

## Step 3 — Stage 1: BERT → RetBERT

Trains RetBERT to produce sentence embeddings matching frozen BERT-base.

```bash
python train_stage1.py
```

**Dataset:** WikiText-103 + CoNLL-2003 train (~3.9M sentences)  
**Duration:** ~8–10 hrs on T4 GPU, ~2–2.5 hrs on A6000  
**Output:** `weights/retbert_stage1.pt`

---

## Step 4 — Stage 2: RetBERT → LightRet

Compresses RetBERT's token-level hidden states into the compact LightRet backbone.

```bash
python train_stage2.py
```

**Requires:** `weights/retbert_stage1.pt`  
**Dataset:** Same as Stage 1  
**Duration:** ~4–5 hrs on T4 GPU, ~1–1.5 hrs on A6000  
**Output:** `weights/lightret_stage2.pt`

---

## Step 5 — Stage 3: Noisy-Student NER

Fine-tunes LightRet + NER head on noisy CoNLL-2003 using noisy-student training.

```bash
python train_stage3.py
```

**Requires:** `weights/lightret_stage2.pt`  
**Dataset:** CoNLL-2003 train/validation only (~15K sentences)  
**Duration:** ~20–40 min on T4 GPU, ~5–10 min on A6000  
**Outputs:**
- `weights/lightret_stage3.pt` — LightRet backbone
- `weights/ner_head_stage3.pt` — BiLSTM NER head

---

## Running on Kaggle

Three ready-to-run Kaggle notebooks are provided under `notebooks/`.

### Required Kaggle Datasets

| Dataset Name | Contents | How to create |
|---|---|---|
| `lightret-source` | Full project folder (`src/`, `train_*.py`, etc.) | Upload the repo |
| `lightret-weights` | `retvec_v1_weights.npz` | Upload after Step 1 |
| `wikitext-local` | WikiText-103 Arrow cache | Run `save_wikitext_local.py` then zip |

`conll2003_local.zip` is already in the GitHub repo — the notebooks download
and unzip it automatically.

### Notebook Order

```
stage1_kaggle.ipynb  →  download retbert_stage1.pt
stage2_kaggle.ipynb  →  download lightret_stage2.pt   (requires retbert_stage1.pt as input dataset)
stage3_kaggle.ipynb  →  download lightret_stage3.pt   (requires lightret_stage2.pt as input dataset)
```

### Internet Setting

| Stage | Internet required? |
|---|---|
| Stage 1 | ON (BERT download) — or provide `wikitext-local` dataset |
| Stage 2 | ON (if wikitext-local not attached) |
| Stage 3 | OFF — CoNLL is downloaded from GitHub automatically |

---

## Offline Dataset Preparation (optional)

Download WikiText-103 locally to avoid internet dependency on Kaggle:

```bash
python save_wikitext_local.py
# Windows:
Compress-Archive wikitext_local wikitext_local.zip
# Linux/Mac:
zip -r wikitext_local.zip wikitext_local/
```

Upload `wikitext_local.zip` to Kaggle as dataset named **`wikitext-local`**.

---

## GPU Time Estimates

| Stage | T4 (Kaggle) | A6000 | A6000 + AMP |
|---|---|---|---|
| Stage 1 (5 epochs) | 8–10 hrs | ~2–2.5 hrs | ~1.5 hrs |
| Stage 2 (5 epochs) | 4–5 hrs | ~1–1.5 hrs | ~45 min |
| Stage 3 (10 epochs)| 20–40 min | ~5–10 min | ~5 min |
| **Total** | **~13–16 hrs** | **~3–4 hrs** | **~2–2.5 hrs** |
