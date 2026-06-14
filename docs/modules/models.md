# src/models/

All model architectures for the LightRet pipeline.

---

## retvec_embedder.py — RetVecEmbedder

Frozen PyTorch port of the pretrained RetVec-v1 character embedder.

### Architecture

```
word string
    → binarize(word)
        • Take up to 16 Unicode code points
        • Encode each as 24-bit binary  → (16, 24) float32
    → Flatten                           → (384,)
    → Linear(384→256) + GELU            [dense_3 weights]
    → Linear(256→256) + GELU            [dense_4 weights]
    → Linear(256→256) + Tanh            [dense_5 weights]
Output: (256,) float32, values in [-1, 1]
```

### Class: `RetVecEmbedder(nn.Module)`

```python
RetVecEmbedder(weights_path: str)
```

| Parameter | Type | Description |
|---|---|---|
| `weights_path` | `str` | Path to `retvec_v1_weights.npz` |

**Key properties:**
- All parameters have `requires_grad = False` (permanently frozen)
- Handles variable-length sentences by padding to the longest sentence in the batch

### Forward

```python
embedder(batch_words: list[list[str]]) -> torch.Tensor  # (B, L_max, 256)
```

| Arg | Type | Description |
|---|---|---|
| `batch_words` | `list[list[str]]` | Batch of tokenized sentences |

Returns a tensor of shape `(B, L_max, 256)` where `L_max` is the length of the
longest sentence in the batch. Padding rows are zero vectors.

### Internal helpers

| Method | Description |
|---|---|
| `_binarize(word)` | Converts a word to `(16, 24)` binary float32 array |
| `_binarize_batch(batch_words)` | Vectorized binarization for a full batch |

### Usage

```python
from src.models.retvec_embedder import RetVecEmbedder

emb = RetVecEmbedder("retvec_v1_weights.npz").to(device)
out = emb([["London", "is", "the", "capital"], ["Microsoft"]])
# out.shape → (2, 4, 256)
```

---

## retbert.py — RetBERT

Stage 1 student model. Replaces BERT's WordPiece embedding lookup with frozen
RetVec, then uses 12 Transformer encoder layers to build contextual representations.

### Architecture

```
list[list[str]]   (batch of word-tokenized sentences)
    → RetVecEmbedder (frozen)        → (B, L, 256)
    → Linear projection (256→768)    → (B, L, 768)
    → Sinusoidal positional encoding → (B, L, 768)
    → 12× Pre-LN Transformer layer   → (B, L, 768)
    → Mean pool over valid tokens    → (B, 768)      ← sentence vector
```

**Pre-LN Transformer block:**
```
x' = x + MHA(LN(x))
x″ = x' + FFN(LN(x'))
FFN(v) = W₂ GELU(W₁v + b₁) + b₂
```

### Class: `RetBERT(nn.Module)`

```python
RetBERT(weights_path: str = None)
```

Uses `RETVEC_WEIGHTS` from `src.config` if `weights_path` is not given.

### Forward

```python
model(batch_words: list[list[str]]) -> tuple[Tensor, Tensor]
#                                       (B, L, 768), (B, 768)
```

Returns `(hidden_states, pooled_sentence_vector)`.

| Output | Shape | Description |
|---|---|---|
| `hidden` | `(B, L, 768)` | Per-token contextual representations |
| `pooled` | `(B, 768)` | Mean-pooled sentence embedding (Stage 1 target) |

### Trainable parameters

| Component | Trainable |
|---|---|
| RetVec embedder | No |
| `proj` — Linear(256→768) | Yes |
| `encoder` — 12× TransformerEncoderLayer | Yes |

Total trainable: ~86M parameters.

### Checkpointing

```python
torch.save(retbert.state_dict(), "weights/retbert_stage1.pt")
retbert.load_state_dict(torch.load("weights/retbert_stage1.pt"))
```

---

## lightret.py — LightRet

The final compact backbone model used in both Stage 2 (compression) and
Stage 3 (NER fine-tuning). 4× smaller d=256 space; BiGRU + 4-layer Transformer.

### Architecture

```
list[list[str]]
    → RetVecEmbedder (frozen)         → (B, L, 256)
    → BiGRU (128 per direction)       → (B, L, 256)
    → Sinusoidal positional encoding  → (B, L, 256)
    → 4× Pre-LN Transformer layer     → (B, L, 256)
    [Stage 2 only:]
    → Linear projector (256→768)      → (B, L, 768)
```

### Class: `LightRet(nn.Module)`

```python
LightRet(with_projector: bool = False, weights_path: str = None)
```

| Parameter | Description |
|---|---|
| `with_projector` | If `True`, adds the Stage 2 linear projector (256→768) |
| `weights_path` | Override for `RETVEC_WEIGHTS` path |

### Factory methods

| Method | Description |
|---|---|
| `LightRet.for_stage2()` | Creates with projector (`with_projector=True`) |
| `LightRet.for_stage3()` | Creates without projector (fresh backbone) |
| `LightRet.from_stage2_checkpoint(path, freeze=False)` | Loads Stage 2 checkpoint, drops projector |

```python
# Stage 2 training
model = LightRet.for_stage2().to(device)

# Stage 3 — teacher (frozen)
teacher = LightRet.from_stage2_checkpoint("weights/lightret_stage2.pt", freeze=True)

# Stage 3 — student (trainable backbone)
student = LightRet.from_stage2_checkpoint("weights/lightret_stage2.pt", freeze=False)
```

### Forward

```python
# Without projector (Stage 3):
model(batch_words) -> Tensor  # (B, L, 256)

# With projector (Stage 2):
model(batch_words) -> Tensor  # (B, L, 768)
```

### Trainable parameters

| Component | Stage 2 | Stage 3 |
|---|---|---|
| RetVec embedder | Frozen | Frozen |
| BiGRU | Yes | Yes |
| 4× Transformer | Yes | Yes |
| Projector (256→768) | Yes | Dropped |

Total (without projector): ~3.4M parameters.

---

## ner_head.py — NERHead

BiLSTM classification head for Stage 3 NER.

### Architecture

```
(B, L, 256)  ← LightRet backbone output
    → BiLSTM (128 per direction)  → (B, L, 256)
    → Dropout(0.1)
    → Linear (256 → num_classes)  → (B, L, 9)
```

Padding positions (beyond each sentence's actual length) are masked with
`NER_IGNORE_INDEX = -100`, which PyTorch's `CrossEntropyLoss` ignores.

### Class: `NERHead(nn.Module)`

```python
NERHead(
    input_dim: int   = LIGHTRET_DIM,       # 256
    hidden_dim: int  = NER_BILSTM_HIDDEN,  # 128 per direction
    num_classes: int = NER_NUM_CLASSES,    # 9
    dropout: float   = NER_DROPOUT,        # 0.1
)
```

### Forward

```python
head(
    hidden: Tensor,         # (B, L, 256) from LightRet
    lengths: list[int],     # actual word count per sentence
) -> Tensor                 # (B, L, 9) logits
```

Positions beyond `lengths[i]` for sentence `i` are set to `NER_IGNORE_INDEX`
in the returned logit tensor, ensuring they are excluded from the loss.

### Usage

```python
from src.models.ner_head import NERHead

head = NERHead().to(device)
hidden = student(noisy_words)          # (B, L, 256)
logits = head(hidden, noisy_lengths)   # (B, L, 9)
```

### Checkpointing

```python
torch.save(head.state_dict(), "weights/ner_head_stage3.pt")
head.load_state_dict(torch.load("weights/ner_head_stage3.pt"))
```
