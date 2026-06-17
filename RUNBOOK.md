# LightRet — End-to-End Runbook

Complete script execution order from environment setup to final evaluation.
Every command is run from the **project root** (`lightret/`).

---

## Prerequisites

### 1. Install dependencies

```bash
pip install torch torchvision
pip install transformers datasets seqeval
pip install tensorflow retvec   # only needed for Step 2 (RetVec export)
pip install numpy
```

### 2. Verify GPU is available (optional but recommended)

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Phase 0 — One-Time Setup

### Step 0a: Export RetVec pretrained weights

Downloads the TF RetVec-v1 model and exports weights to `retvec_v1_weights.npz`.
Run **once** on a machine with internet access.

```bash
python retvec_export.py
```

**Output:** `retvec_v1_weights.npz` in the project root.

> **Kaggle:** Upload `retvec_v1_weights.npz` as a dataset and update
> `RETVEC_WEIGHTS` in `src/config.py` to point to the mount path.

---

### Step 0b: Cache WikiText-103 locally (optional but avoids repeated HF downloads)

```bash
python save_wikitext_local.py
```

**Output:** `wikitext_local/` directory (Arrow format).

> Skip this if you have a fast internet connection; the training scripts
> download it directly from HuggingFace if the local cache is absent.

---

### Step 0c: Smoke-test the pipeline (optional sanity check)

Runs a CPU dry run with synthetic data. No GPU or large downloads needed.

```bash
python dry_run.py
```

**Expected:** All lines print `PASS`.

---

## Phase 1 — Training (3 Stages)

Run stages **in order**. Each stage depends on the checkpoint from the previous one.

### Step 1: Stage 1 — Sentence distillation  (BERT-base → RetBERT)

**Corpus:** WikiText-103 + CoNLL-2003 train (clean sentences)
**Duration:** ~5 epochs, ~2–4 h on a single GPU

```bash
python train_stage1.py
```

**Output:** `weights/retbert_stage1.pt`

**What to expect:** Loss starts ~0.5–0.6, should settle below ~0.05
(cosine similarity > 0.95 is excellent — 0.0419 observed in testing).

---

### Step 2: Stage 2 — Token-level compression  (RetBERT → LightRet backbone)

**Prerequisite:** `weights/retbert_stage1.pt`
**Duration:** ~5 epochs, ~1–2 h on a single GPU

```bash
python train_stage2.py
```

**Output:** `weights/lightret_stage2.pt`

**What to expect:** Token cosine distance loss drops to ~0.05–0.15.

---

### Step 3: Stage 3 — Noisy-student NER fine-tuning  (LightRet + NER head)

**Prerequisite:** `weights/lightret_stage2.pt`
**Duration:** ~10 epochs, ~2–3 h on a single GPU

```bash
python train_stage3.py
```

**Outputs:**
- `weights/lightret_stage3.pt`  — LightRet backbone
- `weights/ner_head_stage3.pt`  — BiLSTM NER head

**What to expect:** Validation cross-entropy falls below 0.1; entity F1 on
clean CoNLL-2003 test should be in the high 80s–low 90s.

---

## Phase 2 — Evaluation

All evaluation scripts load `weights/lightret_stage3.pt` and
`weights/ner_head_stage3.pt` by default.

### Step 4: Full LightRet evaluation

Evaluates noise levels, per-entity-type F1, per-operator analysis, and
inference speed. Fills **Table 5, Table 6, Table 8, Table 9, and Abstract** of
the paper.

```bash
python evaluate.py
```

Optional flags:
```bash
python evaluate.py --backbone weights/lightret_stage3.pt \
                   --head     weights/ner_head_stage3.pt \
                   --seeds    5
```

**Console output covers:**
| Section | Paper location |
|---------|----------------|
| F1 at each noise level | Table 5 (clean) + Table 6 (noisy) |
| Per-entity-type F1 (clean vs medium) | Table 9 — LightRet rows |
| Per-noise-operator F1 | Analysis Table 8 — LightRet row |
| Inference speed ms/sentence | Table 8 — speed column |
| Abstract fill-in numbers | abstract.tex |

---

### Step 5: Baseline evaluation

Evaluates `dslim/bert-base-NER` and `elastic/distilbert-base-uncased-finetuned-conll03-english`
at the same noise conditions. Fills baseline rows in **Table 6, Table 8, Table 9**.

> Downloads ~440 MB (BERT-base) + ~260 MB (DistilBERT) on first run.

```bash
python evaluate_baselines.py
```

Optional flags:
```bash
python evaluate_baselines.py --models bert          # bert only
python evaluate_baselines.py --models distilbert    # distilbert only
python evaluate_baselines.py --seeds 3              # faster, fewer seeds
```

---

### Step 6: Label projection evaluation

Measures the space-insertion event rate and how accurately the dynamic BIO
label projection rule matches a BERT-large oracle. Fills **§7.3** of
analysis.tex.

> Downloads `dbmdz/bert-large-cased-finetuned-conll03-english` (~1.2 GB) on
> first run.

```bash
python label_proj_eval.py
```

Optional flags:
```bash
python label_proj_eval.py --seeds 3 --p-space-ins 0.02
```

---

## Phase 3 — Ablation Study

Ablation trains and evaluates 6 variants of Stage 3, each with one component
removed. Run after Phase 1 is complete (needs `weights/lightret_stage2.pt`).

### Step 7a: Train each ablation variant

Run one command per variant (can be parallelised on separate GPUs):

```bash
python ablation_train.py --variant no_noise
python ablation_train.py --variant beta1
python ablation_train.py --variant beta0
python ablation_train.py --variant random_emb
python ablation_train.py --variant no_stage2
python ablation_train.py --variant no_stage1   # needs weights/lightret_stage2_no_stage1.pt
```

**Variants explained:**

| Variant | What is removed | Purpose |
|---------|----------------|---------|
| `no_noise` | Character noise augmentation | Tests noise robustness contribution |
| `beta1` | Distillation loss (β=1, classification only) | Tests knowledge distillation contribution |
| `beta0` | Classification loss (β=0, distillation only) | Tests NER supervision contribution |
| `random_emb` | Pretrained RetVec weights (randomised) | Tests RetVec pretraining contribution |
| `no_stage2` | Stage 2 (fresh backbone, NER fine-tune only) | Tests Stage 2 contribution |
| `no_stage1` | Stage 1 (BERT used directly as Stage 2 teacher) | Tests Stage 1 contribution |

> `no_stage1` requires running Stage 2 with BERT as the direct teacher and
> saving to `weights/lightret_stage2_no_stage1.pt` before this step.

**Outputs per variant:**
- `weights/abl_<variant>.pt`
- `weights/abl_<variant>_head.pt`

---

### Step 7b: Evaluate all ablation variants

Evaluates all 7 rows (full model + 6 variants) and prints the comparison table
matching **Table 7** of the paper.

```bash
python ablation_eval.py
```

Optional:
```bash
python ablation_eval.py --seeds 3
```

Missing checkpoints are reported as `MISSING` and skipped; you can run
available variants without completing all.

---

## Checkpoint Summary

| File | Produced by | Used by |
|------|------------|---------|
| `retvec_v1_weights.npz` | `retvec_export.py` | All stages (frozen inside LightRet) |
| `weights/retbert_stage1.pt` | `train_stage1.py` | `train_stage2.py` |
| `weights/lightret_stage2.pt` | `train_stage2.py` | `train_stage3.py`, `ablation_train.py` |
| `weights/lightret_stage3.pt` | `train_stage3.py` | `evaluate.py`, `ablation_eval.py` |
| `weights/ner_head_stage3.pt` | `train_stage3.py` | `evaluate.py`, `ablation_eval.py` |
| `weights/abl_<variant>.pt` | `ablation_train.py` | `ablation_eval.py` |
| `weights/abl_<variant>_head.pt` | `ablation_train.py` | `ablation_eval.py` |

---

## Paper Numbers Mapping

| Script output | Paper section |
|--------------|--------------|
| `evaluate.py` — clean F1 | Table 5, Abstract |
| `evaluate.py` — noise-level F1 | Table 6 (LightRet rows) |
| `evaluate.py` — per-operator F1 | Analysis Table 8 (LightRet row) |
| `evaluate.py` — per-entity F1 | Table 9 (LightRet rows) |
| `evaluate.py` — inference speed | Table 8 (LightRet speed) |
| `evaluate_baselines.py` | Table 6, Table 8, Table 9 (baseline rows) |
| `label_proj_eval.py` | §7.3, analysis.tex |
| `ablation_eval.py` | Table 7 |

---

## Quick Reference (minimal run on GPU)

```bash
# One-time setup
python retvec_export.py

# Training
python train_stage1.py
python train_stage2.py
python train_stage3.py

# Evaluation
python evaluate.py
python evaluate_baselines.py
python label_proj_eval.py

# Ablation
python ablation_train.py --variant no_noise
python ablation_train.py --variant beta1
python ablation_train.py --variant beta0
python ablation_train.py --variant random_emb
python ablation_train.py --variant no_stage2
python ablation_eval.py
```
