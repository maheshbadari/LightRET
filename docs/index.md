# LightRet Documentation

## Contents

| Document | What it covers |
|---|---|
| [Overview](overview.md) | Project summary, repository layout, pipeline diagram |
| [Quick Start](quickstart.md) | Setup, prerequisites, full pipeline walkthrough (local + Kaggle) |
| **Scripts** | |
| [Training Scripts](scripts/training.md) | `train_stage1/2/3.py`, `dry_run.py` — what each does, inputs/outputs, hyperparameters |
| [Utility Scripts](scripts/utilities.md) | `retvec_export.py`, `save_wikitext_local.py`, `paper/build_pdf.py` |
| [Kaggle Notebooks](notebooks.md) | Setup cell logic, required datasets, per-stage notes, updating embedded `dataset.py` |
| **Modules** | |
| [config.py](modules/config.md) | All constants — paths, dimensions, noise probabilities, per-stage hyperparameters |
| [models/](modules/models.md) | `RetVecEmbedder`, `RetBERT`, `LightRet`, `NERHead` — architecture, forward signatures, factory methods |
| [data/](modules/data.md) | `PretrainDataset`, `NERDataset`, collation functions, CoNLL/WikiText loaders, label projection |
| [losses.py](modules/losses.md) | `stage1_loss`, `stage2_loss`, `stage3_loss` — formulas, design rationale |
| [noise.py](modules/noise.md) | `apply_noise`, `build_word_alignment`, `map_span`, visual similarity table |

---

## Five-Minute Summary

**LightRet** is a vocabulary-free NER model (~4M parameters) trained in three stages:

```
Stage 1  BERT (frozen) ──cosine loss──► RetBERT          [sentence-level]
Stage 2  RetBERT (frozen) ─cosine loss─► LightRet+Proj   [token-level]
Stage 3  LightRet (frozen, clean text)
           + LightRet+NERHead (trainable, noisy text)
           ─── β·L_class + (1-β)·L_distill ───►  LightRet-NER
```

Every model in the chain uses **RetVec** as its word embedder — a frozen,
pretrained 3-layer MLP that maps any Unicode word to a 256-d vector without
a vocabulary lookup. This makes LightRet robust to typos, OCR noise, and
out-of-vocabulary words by design.

---

## Key Files at a Glance

```
retvec_export.py          ← run once, produces retvec_v1_weights.npz
train_stage1.py           ← produces weights/retbert_stage1.pt
train_stage2.py           ← produces weights/lightret_stage2.pt
train_stage3.py           ← produces weights/lightret_stage3.pt + ner_head_stage3.pt
dry_run.py                ← smoke test, 34 checks, no GPU needed
src/config.py             ← change hyperparameters here
src/noise.py              ← character-level noise simulator
src/data/dataset.py       ← data loading (local-first, HuggingFace fallback)
src/losses.py             ← all three loss functions
```
