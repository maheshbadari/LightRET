# src/config.py

Central configuration module. Every other module imports constants from here.
Change values here to adjust the training pipeline without touching any other file.

---

## Paths

| Constant | Default | Description |
|---|---|---|
| `ROOT` | `Path(__file__).parent.parent` | Project root (lightret/) |
| `WEIGHTS_DIR` | `ROOT / "weights"` | Directory for saved checkpoints |
| `RETVEC_WEIGHTS` | `ROOT / "retvec_v1_weights.npz"` | Exported RetVec weights file |
| `STAGE1_CKPT` | `WEIGHTS_DIR / "retbert_stage1.pt"` | Stage 1 output checkpoint |
| `STAGE2_CKPT` | `WEIGHTS_DIR / "lightret_stage2.pt"` | Stage 2 output checkpoint |
| `STAGE3_CKPT` | `WEIGHTS_DIR / "lightret_stage3.pt"` | Stage 3 output checkpoint |

**Kaggle override:** The notebooks patch `RETVEC_WEIGHTS` and `WEIGHTS_DIR` via
`os.environ` or direct attribute assignment after import.

---

## Device

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Auto-selects GPU on Kaggle, falls back to CPU locally.

---

## BERT Teacher (Stage 1)

| Constant | Value | Description |
|---|---|---|
| `BERT_MODEL_NAME` | `"bert-base-uncased"` | HuggingFace model ID |
| `BERT_DIM` | `768` | BERT hidden dimension |

---

## RetVec Embedder

| Constant | Value | Description |
|---|---|---|
| `RETVEC_DIM` | `256` | Output embedding dimension |
| `RETVEC_FROZEN` | `True` | Always frozen — never trained |

---

## RetBERT (Stage 1 Student)

| Constant | Value | Description |
|---|---|---|
| `RETBERT_DIM` | `768` | Hidden dimension (matches BERT) |
| `RETBERT_PROJ` | `256` | Input dim to projection layer |
| `RETBERT_LAYERS` | `12` | Number of Transformer encoder layers |
| `RETBERT_HEADS` | `12` | Attention heads (head_dim = 64) |
| `RETBERT_FFN_DIM` | `3072` | Feed-forward inner dimension (4 × 768) |
| `RETBERT_DROPOUT` | `0.1` | Dropout in attention and FFN |

---

## LightRet (Stages 2 & 3)

| Constant | Value | Description |
|---|---|---|
| `LIGHTRET_DIM` | `256` | Hidden dimension throughout backbone |
| `LIGHTRET_BIGRU_HIDDEN` | `128` | Per-direction GRU hidden size (concat = 256) |
| `LIGHTRET_LAYERS` | `4` | Number of Transformer encoder layers |
| `LIGHTRET_HEADS` | `4` | Attention heads (head_dim = 64) |
| `LIGHTRET_FFN_DIM` | `1024` | Feed-forward inner dimension (4 × 256) |
| `LIGHTRET_DROPOUT` | `0.1` | Dropout |
| `LIGHTRET_PROJ_DIM` | `768` | Stage 2 projector output dim (matches RetBERT) |

---

## NER Head (Stage 3)

| Constant | Value | Description |
|---|---|---|
| `NER_BILSTM_HIDDEN` | `128` | Per-direction LSTM hidden size (concat = 256) |
| `NER_DROPOUT` | `0.1` | Dropout before classifier |
| `NER_LABELS` | 9-element list | Full BIO label set |
| `NER_NUM_CLASSES` | `9` | Number of output classes |
| `NER_LABEL2ID` | dict | Label string → int |
| `NER_ID2LABEL` | dict | Int → label string |
| `NER_IGNORE_INDEX` | `-100` | Padding positions excluded from cross-entropy |

**Label set:**
```
O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
```

---

## Noise Parameters (Stage 3)

| Constant | Value | Description |
|---|---|---|
| `NOISE_P_SUB` | `0.10` | Character substitution probability |
| `NOISE_P_INS` | `0.05` | Character insertion probability |
| `NOISE_P_DEL` | `0.05` | Character deletion probability |
| `NOISE_P_SPACE_INS` | `0.02` | Space insertion probability (word splitting) |

---

## Stage 1 Hyperparameters

| Constant | Value |
|---|---|
| `STAGE1_EPOCHS` | `5` |
| `STAGE1_BATCH_SIZE` | `32` |
| `STAGE1_LR` | `5e-5` |
| `STAGE1_WARMUP_STEPS` | `1000` |
| `STAGE1_MAX_WORDS` | `64` |
| `STAGE1_MAX_SUBWORDS` | `128` |
| `STAGE1_WIKITEXT_DATASET` | `"wikitext"` |
| `STAGE1_WIKITEXT_CONFIG` | `"wikitext-103-raw-v1"` |
| `STAGE1_CONLL_DATASET` | `"eriktks/conll2003"` |

---

## Stage 2 Hyperparameters

| Constant | Value |
|---|---|
| `STAGE2_EPOCHS` | `5` |
| `STAGE2_BATCH_SIZE` | `64` |
| `STAGE2_LR` | `3e-4` |
| `STAGE2_WARMUP_STEPS` | `500` |
| `STAGE2_MAX_WORDS` | `64` |

Same dataset constants as Stage 1.

---

## Stage 3 Hyperparameters

| Constant | Value |
|---|---|
| `STAGE3_EPOCHS` | `10` |
| `STAGE3_BATCH_SIZE` | `32` |
| `STAGE3_LR` | `2e-4` |
| `STAGE3_WARMUP_STEPS` | `200` |
| `STAGE3_MAX_WORDS` | `64` |
| `STAGE3_BETA` | `0.5` |
| `STAGE3_CONLL_DATASET` | `"eriktks/conll2003"` |
