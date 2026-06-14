# LightRet — Project Overview

LightRet is a lightweight, vocabulary-free Named Entity Recognition model trained via
a three-stage progressive knowledge distillation pipeline. It uses **RetVec** as a
frozen character-level word embedder, eliminating any dependency on a tokenizer
vocabulary. The final model (~4M parameters) is robust to character-level noise
(typos, OCR errors, insertions, deletions) while remaining competitive on clean text.

---

## Pipeline at a Glance

```
retvec_export.py
      |
      v
retvec_v1_weights.npz
      |
      v
train_stage1.py  ──  BERT (frozen teacher)  ──>  RetBERT student
      |                  sentence-level cosine loss
      v
weights/retbert_stage1.pt
      |
      v
train_stage2.py  ──  RetBERT (frozen teacher)  ──>  LightRet student
      |                  token-level cosine loss
      v
weights/lightret_stage2.pt
      |
      v
train_stage3.py  ──  LightRet (frozen teacher, clean)
      |               LightRet + NERHead (student, noisy)
      |               compound loss: β·L_class + (1-β)·L_distill
      v
weights/lightret_stage3.pt
weights/ner_head_stage3.pt
```

---

## Repository Structure

```
lightret/
│
├── train_stage1.py          # Stage 1 training script
├── train_stage2.py          # Stage 2 training script
├── train_stage3.py          # Stage 3 training script
├── dry_run.py               # End-to-end smoke test (no GPU required)
├── retvec_export.py         # One-time RetVec weight export
├── save_wikitext_local.py   # One-time WikiText-103 download for offline use
│
├── src/
│   ├── config.py            # All hyperparameters and paths
│   ├── losses.py            # Loss functions for all three stages
│   ├── noise.py             # Character-level noise simulator
│   ├── models/
│   │   ├── retvec_embedder.py   # Frozen RetVec-v1 PyTorch port
│   │   ├── retbert.py           # RetBERT (Stage 1 student)
│   │   ├── lightret.py          # LightRet backbone (Stage 2 & 3)
│   │   └── ner_head.py          # BiLSTM NER classification head
│   └── data/
│       ├── dataset.py           # PretrainDataset and NERDataset
│       └── label_utils.py       # BIO label projection under noise
│
├── notebooks/
│   ├── stage1_kaggle.ipynb  # Kaggle notebook — Stage 1
│   ├── stage2_kaggle.ipynb  # Kaggle notebook — Stage 2
│   └── stage3_kaggle.ipynb  # Kaggle notebook — Stage 3
│
├── paper/
│   ├── main.tex             # LaTeX source (root)
│   ├── sections/            # One .tex file per paper section
│   ├── references.bib       # BibTeX bibliography
│   ├── build_pdf.py         # Python PDF builder (no LaTeX needed)
│   └── LightRet_paper.pdf   # Generated paper
│
├── conll2003_local.zip      # Pre-saved CoNLL-2003 Arrow dataset (~1.7 MB)
├── retvec_v1_weights.npz    # Exported RetVec weights (required for all stages)
└── weights/                 # Saved checkpoints (git-ignored)
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| Transformers | 4.36+ |
| datasets | 3.0+ |
| NumPy | 1.24+ |
| TensorFlow | 2.x (only for `retvec_export.py`) |
| retvec | latest (only for `retvec_export.py`) |

Install:
```bash
pip install torch transformers datasets huggingface_hub reportlab
```

---

## Quick Navigation

| What you want | Where to look |
|---|---|
| Run the full pipeline | [quickstart.md](quickstart.md) |
| Training scripts (Stage 1/2/3) | [scripts/training.md](scripts/training.md) |
| Utility scripts | [scripts/utilities.md](scripts/utilities.md) |
| Model architectures | [modules/models.md](modules/models.md) |
| Data loading & noise | [modules/data.md](modules/data.md) |
| Loss functions | [modules/losses.md](modules/losses.md) |
| All hyperparameters | [modules/config.md](modules/config.md) |
