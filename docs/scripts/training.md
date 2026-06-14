# Training Scripts

---

## train_stage1.py

**Purpose:** Stage 1 — Sentence-level knowledge distillation from BERT into RetBERT.

### What it does

Trains **RetBERT** (a 12-layer Transformer that uses RetVec character embeddings as
input) to produce sentence-level representations that match frozen BERT-base. Both
models process the same clean text. The distillation loss is the cosine distance
between the two sentence vectors.

```
BERT-base (frozen)  ─┐
                      ├── L₁ = 1 − cos(z_BERT, z_RetBERT)
RetBERT (trainable) ─┘
```

### Inputs / Outputs

| Item | Value |
|---|---|
| Prerequisite | `retvec_v1_weights.npz` |
| Dataset | WikiText-103-raw-v1 + CoNLL-2003 train |
| Output checkpoint | `weights/retbert_stage1.pt` |

### Key functions

| Function | Description |
|---|---|
| `bert_sentence_emb(batch_words)` | Tokenizes words, runs BERT, returns mean-pooled CLS embeddings `(B, 768)` |
| `make_scheduler(optimizer, warmup, total)` | Cosine annealing with linear warmup |

### Hyperparameters (from `src/config.py`)

| Name | Value | Description |
|---|---|---|
| `STAGE1_EPOCHS` | 5 | Training epochs |
| `STAGE1_BATCH_SIZE` | 32 | Sentences per batch |
| `STAGE1_LR` | 5e-5 | Peak learning rate |
| `STAGE1_WARMUP_STEPS` | 1000 | Linear warmup steps |
| `STAGE1_MAX_WORDS` | 64 | Max words per sentence (word-level truncation) |
| `STAGE1_MAX_SUBWORDS` | 128 | Max subword tokens fed to BERT |

### Run

```bash
python train_stage1.py
```

### Checkpointing

Saves `weights/retbert_stage1.pt` whenever the epoch average loss improves.
The file contains the full `RetBERT.state_dict()`.

---

## train_stage2.py

**Purpose:** Stage 2 — Token-level compression from RetBERT into the LightRet backbone.

### What it does

Trains **LightRet** (BiGRU + 4-layer Transformer, d=256) to reproduce RetBERT's
per-token hidden states. A temporary linear projector (256→768) aligns LightRet's
output dimension to RetBERT's before computing the loss. The projector is included
in the checkpoint but discarded before Stage 3.

```
RetBERT (frozen)  ─┐
                    ├── L₂ = (1/n) Σᵢ cos_dist(h_i^RetBERT, p_i)
LightRet+Proj     ─┘       where p_i = W_proj · h_i^LightRet
```

### Inputs / Outputs

| Item | Value |
|---|---|
| Prerequisite | `weights/retbert_stage1.pt` |
| Dataset | WikiText-103-raw-v1 + CoNLL-2003 train (same as Stage 1) |
| Output checkpoint | `weights/lightret_stage2.pt` |

### Key differences from Stage 1

- **Token-level** supervision (not sentence-level) — every word position is
  supervised independently.
- **Larger batch** (64 vs 32) — LightRet is much smaller than RetBERT so
  memory allows bigger batches.
- **Higher LR** (3e-4) — LightRet trains from scratch, not fine-tuning.

### Hyperparameters

| Name | Value | Description |
|---|---|---|
| `STAGE2_EPOCHS` | 5 | Training epochs |
| `STAGE2_BATCH_SIZE` | 64 | Sentences per batch |
| `STAGE2_LR` | 3e-4 | Peak learning rate |
| `STAGE2_WARMUP_STEPS` | 500 | Linear warmup steps |
| `STAGE2_MAX_WORDS` | 64 | Max words per sentence |

### Run

```bash
python train_stage2.py
```

### Checkpointing

Saves `weights/lightret_stage2.pt` on every epoch that improves the average
token cosine distance. Contains the full `LightRet.state_dict()` including
the projector weights (needed for loss computation during Stage 2 only).

---

## train_stage3.py

**Purpose:** Stage 3 — Noisy-student NER fine-tuning on CoNLL-2003.

### What it does

Fine-tunes LightRet + a BiLSTM NER head on CoNLL-2003. The **teacher** is a
frozen copy of LightRet from Stage 2 processing clean text; the **student** is
a trainable copy processing stochastically noisy text. Labels are dynamically
projected from clean word positions to noisy word positions using a shift log.

```
Teacher: LightRet (frozen)  ← clean words w      ─┐
                                                    ├── L_distill = cos_dist(aligned)
Student: LightRet+NERHead   ← noisy words w̃     ─┤
                                                    └── L_class   = CrossEntropy(ỹ)

L₃ = β · L_class + (1-β) · L_distill     (β = 0.5)
```

### Inputs / Outputs

| Item | Value |
|---|---|
| Prerequisite | `weights/lightret_stage2.pt` |
| Dataset | CoNLL-2003 train/validation (~15K sentences) |
| Output checkpoints | `weights/lightret_stage3.pt`, `weights/ner_head_stage3.pt` |

### Noise parameters

| Parameter | Value | Effect |
|---|---|---|
| `NOISE_P_SUB` | 0.10 | 10% of characters substituted with visual lookalike |
| `NOISE_P_INS` | 0.05 | 5% chance of random character insertion |
| `NOISE_P_DEL` | 0.05 | 5% chance of character deletion |
| `NOISE_P_SPACE_INS` | 0.02 | 2% chance of space insertion (word split) |

### Label projection

When `NOISE_P_SPACE_INS` splits a word (e.g., `London` → `Lon don`),
the BIO label must be projected to both fragments:

- First fragment keeps the original label: `B-LOC`
- Subsequent fragments get the continuation label: `I-LOC`

This logic lives in `src/data/label_utils.py:project_labels()` and
`src/noise.py:build_word_alignment()`.

### Hyperparameters

| Name | Value | Description |
|---|---|---|
| `STAGE3_EPOCHS` | 10 | Training epochs |
| `STAGE3_BATCH_SIZE` | 32 | Sentences per batch |
| `STAGE3_LR` | 2e-4 | Peak learning rate |
| `STAGE3_WARMUP_STEPS` | 200 | Linear warmup steps |
| `STAGE3_MAX_WORDS` | 64 | Max words per clean sentence |
| `STAGE3_BETA` | 0.5 | Weight of L_class vs L_distill |

### What is trainable in Stage 3

| Component | Trainable? |
|---|---|
| RetVec embedder | No (frozen always) |
| LightRet BiGRU | Yes (student only) |
| LightRet Transformer layers | Yes (student only) |
| LightRet projector | Dropped (not in checkpoint) |
| BiLSTM NER head | Yes |

### Run

```bash
python train_stage3.py
```

### Checkpointing

Saves both `lightret_stage3.pt` and `ner_head_stage3.pt` when the entity-level
validation F1 improves. The two files are loaded together at inference time.

### Validation

After each epoch, evaluates on CoNLL-2003 validation split with **noise disabled**
(`apply_noise_aug=False`) to measure clean-text entity F1.

---

## dry_run.py

**Purpose:** End-to-end smoke test that verifies the full pipeline on CPU in < 2 minutes.

### What it tests

| Section | What is checked |
|---|---|
| 0 — Imports | All `src.*` modules import without errors |
| 1 — RetVec | Weights load, forward pass shape, values in [-1,1], all params frozen |
| 2 — Stage 1 | RetBERT forward pass, stage1_loss computes, backward succeeds |
| 3 — Stage 2 | LightRet forward pass, stage2_loss computes, backward succeeds |
| 4 — Stage 3 setup | `from_stage2_checkpoint` loads correctly, projector removed, teacher frozen |
| 5 — CoNLL dataset | NERDataset loads, item shapes correct, alignment covers all noisy words |
| 6 — Stage 3 training | 3 full forward+backward+optimizer steps, gradients flow |
| 7 — Noise pipeline | apply_noise, build_word_alignment, project_labels all consistent |

Total: **34 checks**. All print `PASS` on a correctly configured environment.

### BERT is mocked

Stage 1 uses a `torch.randn(B, 768)` mock instead of downloading BERT (~440 MB).
This keeps the dry run fast and offline.

### Run

```bash
python dry_run.py
```

### Expected output

```
============================================================
  SUMMARY
============================================================
  All checks passed. Pipeline is ready for Kaggle.

  Checkpoints written to weights/
    retbert_stage1.pt    (X.X MB)
    lightret_stage2.pt   (X.X MB)
    lightret_stage3.pt   (X.X MB)
    ner_head_stage3.pt   (X.X MB)
```
