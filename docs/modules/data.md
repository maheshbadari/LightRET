# src/data/

Dataset classes, collation functions, and label projection utilities.

---

## dataset.py

### Overview

| Class / Function | Used in | Description |
|---|---|---|
| `PretrainDataset` | Stage 1, Stage 2 | WikiText-103 + CoNLL-2003 sentences |
| `NERDataset` | Stage 3 | CoNLL-2003 with on-the-fly noise augmentation |
| `pretrain_collate` | Stage 1, Stage 2 | Identity collation (returns list[list[str]]) |
| `ner_collate` | Stage 3 | Pads label tensors, bundles into dict |
| `_load_wikitext(split)` | internal | WikiText loader with local-first fallback |
| `_load_conll2003(split)` | internal | CoNLL loader with 4-strategy fallback chain |

---

### _load_wikitext(split)

Loads WikiText-103-raw-v1 with local-first, HuggingFace-fallback strategy:

```
1. /kaggle/input/wikitext-local/wikitext_local/{split}   ← Kaggle dataset
2. wikitext_local/{split}                                 ← local dev
3. HuggingFace Hub (requires internet)                    ← fallback
```

Prints which strategy succeeded so failures are immediately visible in notebook output.

---

### _load_conll2003(split)

Loads CoNLL-2003 without triggering the `conll2003.py` dataset script
(which `datasets >= 3.0` refuses to run). Four strategies attempted in order:

```
1. /kaggle/input/conll2003-local/conll2003_local/{split}           ← Kaggle dataset
2. /kaggle/input/datasets/.../lightret/conll2003_local/.../{split} ← lightret-source path
3. conll2003_local/{split}                                          ← local dev
4. hf://datasets/conll2003/data/{split}-00000-of-00001.parquet     ← parquet bypass
5. Dynamic parquet discovery via list_repo_files()                  ← fallback
```

The notebook setup cell also downloads `conll2003_local.zip` from GitHub and
unzips it to `/kaggle/working/conll2003_local/`, which strategy 3 then finds.

---

### class PretrainDataset

```python
PretrainDataset(
    split: str = "train",         # "train" or "validation"
    max_words: int = 64,          # truncate sentences longer than this
    verbose: bool = True,         # print loading progress
)
```

Loads WikiText-103 + CoNLL-2003 and extracts word-tokenized sentences.

**WikiText filtering:**
- Skip empty lines and section headings (lines starting with `=`)
- Split each line into sentences at `[.!?]` boundaries
- Keep sentences with ≥ 4 words; truncate to `max_words`

**CoNLL filtering:**
- Keep sentences with ≥ 2 words; truncate to `max_words`
- CoNLL is already tokenized — no splitting needed

**`__getitem__` returns:** `list[str]` — word tokens for one sentence

**Typical sizes:**
| Split | WikiText sentences | CoNLL sentences | Total |
|---|---|---|---|
| train | ~3,898,393 | ~14,987 | ~3,913,380 |
| validation | ~219,000 | ~3,466 | ~222,466 |

---

### class NERDataset

```python
NERDataset(
    split: str = "train",           # "train", "validation", or "test"
    max_words: int = 64,
    apply_noise_aug: bool = True,   # False for validation/inference
)
```

Loads CoNLL-2003 and applies fresh stochastic noise to each sentence on every
`__getitem__` call, so different epochs see different noise patterns automatically.

**`__getitem__` returns dict:**

| Key | Type | Description |
|---|---|---|
| `clean_words` | `list[str]` | Original word tokens |
| `clean_labels` | `list[int]` | BIO label IDs (clean coordinates) |
| `noisy_words` | `list[str]` | Word tokens after noise application |
| `noisy_labels` | `list[int]` | BIO label IDs projected to noisy coordinates |
| `alignment` | `list[tuple[list[int], list[int]]]` | `(C_k, N_k)` group pairs |

When `apply_noise_aug=False` (validation mode), `clean == noisy` and alignment
is trivial 1:1.

---

### pretrain_collate

```python
pretrain_collate(batch: list[list[str]]) -> list[list[str]]
```

Identity function — returns the batch as-is. Both `RetBERT` and `LightRet`
accept `list[list[str]]` directly and handle padding internally via the
`RetVecEmbedder` binarization step.

---

### ner_collate

```python
ner_collate(batch: list[dict]) -> dict
```

Collates a batch of `NERDataset` items into model-ready tensors.

**Returns dict with:**

| Key | Type | Description |
|---|---|---|
| `clean_words` | `list[list[str]]` | Batch of clean word lists |
| `clean_labels` | `(B, L_clean_max)` LongTensor | Padded with `NER_IGNORE_INDEX = -100` |
| `noisy_words` | `list[list[str]]` | Batch of noisy word lists |
| `noisy_labels` | `(B, L_noisy_max)` LongTensor | Padded with `NER_IGNORE_INDEX = -100` |
| `alignment` | `list[list[tuple]]` | Per-sentence alignment groups |
| `clean_lengths` | `list[int]` | Actual word count per clean sentence |
| `noisy_lengths` | `list[int]` | Actual word count per noisy sentence |

---

## label_utils.py

Handles BIO label projection from clean word coordinates to noisy word
coordinates when character-level noise changes word boundaries.

---

### continuation_label(label_id)

```python
continuation_label(label_id: int) -> int
```

Maps a B-X label to its I-X continuation:

| Input | Output |
|---|---|
| `B-PER` (1) | `I-PER` (2) |
| `B-ORG` (3) | `I-ORG` (4) |
| `B-LOC` (5) | `I-LOC` (6) |
| `B-MISC` (7) | `I-MISC` (8) |
| `O`, `I-*` | unchanged |

Used when a space insertion splits a single clean word into multiple noisy words —
all fragments after the first receive the continuation label.

---

### project_labels(clean_labels, alignment)

```python
project_labels(
    clean_labels: list[int],
    alignment: list[tuple[list[int], list[int]]],
) -> list[int]
```

Projects BIO labels from clean word positions to noisy word positions
using the alignment groups produced by `noise.build_word_alignment`.

**Projection rules:**

| Case | Clean words → Noisy words | Rule |
|---|---|---|
| 1:1 | 1 → 1 | `ỹ[N_k[0]] = y[C_k[0]]` |
| Merge | many → 1 | `ỹ[N_k[0]] = y[C_k[0]]` (first clean label) |
| Split | 1 → many | First: `ỹ[N_k[0]] = y[C_k[0]]`; rest: `ỹ[N_k[j]] = σ(y[C_k[0]])` |

**Example — split (space insertion):**
```
Clean:  ["London"]   labels: [B-LOC]
Noisy:  ["Lon", "don"]
Output: [B-LOC, I-LOC]
```

**Example — merge (deletion collapses word boundary):**
```
Clean:  ["New", "York"]   labels: [B-LOC, I-LOC]
Noisy:  ["NewYork"]
Output: [B-LOC]
```

**Safety fallback:** If the projected noisy length does not match the actual
number of noisy words (can happen in rare edge cases), the function returns
a 1:1 copy of the clean labels with any excess/missing positions handled by
padding. The `NERDataset.__getitem__` also has a fallback that reverts to
clean=noisy if the projected length mismatches.
