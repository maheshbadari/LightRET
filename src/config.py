"""
config.py — Central hyperparameters and paths for all LightRet training stages.

All other modules import from here. To run on Kaggle, update ROOT or override
RETVEC_WEIGHTS / WEIGHTS_DIR to match your dataset mount paths.
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT            = Path(__file__).parent.parent   # lightret/ (one level above src/)
WEIGHTS_DIR     = ROOT / "weights"               # saved model checkpoints
RETVEC_WEIGHTS  = ROOT / "retvec_v1_weights.npz"

STAGE1_CKPT     = WEIGHTS_DIR / "retbert_stage1.pt"
STAGE2_CKPT     = WEIGHTS_DIR / "lightret_stage2.pt"
STAGE3_CKPT     = WEIGHTS_DIR / "lightret_stage3.pt"

# ---------------------------------------------------------------------------
# Device  — auto-selects GPU on Kaggle, CPU locally
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# BERT teacher  (Stage 1)
# ---------------------------------------------------------------------------

BERT_MODEL_NAME = "bert-base-uncased"
BERT_DIM        = 768

# ---------------------------------------------------------------------------
# RetVec embedder  (frozen throughout all stages)
# ---------------------------------------------------------------------------

RETVEC_DIM    = 256
RETVEC_FROZEN = True

# ---------------------------------------------------------------------------
# RetBERT  (Stage 1 student)
#   Word-level tokenization: RetVec(256) → Linear(256→768) → 12 Transformer layers
#   d_model = 768 matches Teacher_BERT for token-level Stage 2 alignment
# ---------------------------------------------------------------------------

RETBERT_DIM     = 768
RETBERT_PROJ    = 256           # RetVec output dim fed into linear projection
RETBERT_LAYERS  = 12
RETBERT_HEADS   = 12            # 768 / 64 = 12 heads, head_dim = 64 (matches BERT)
RETBERT_FFN_DIM = 3072          # 4 × 768
RETBERT_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# LightRet  (Stages 2 & 3)
#   RetVec(256) → BiGRU(128×2=256) → 4 Transformer layers(d=256)
#   Stage 2 adds: Linear Projector(256→768) — discarded before Stage 3
# ---------------------------------------------------------------------------

LIGHTRET_DIM          = 256
LIGHTRET_BIGRU_HIDDEN = 128     # per direction; concat output = 256
LIGHTRET_LAYERS       = 4
LIGHTRET_HEADS        = 4       # 256 / 64 = 4 heads, head_dim = 64
LIGHTRET_FFN_DIM      = 1024    # 4 × 256
LIGHTRET_DROPOUT      = 0.1

LIGHTRET_PROJ_DIM     = 768     # Stage 2 projector output dim (matches RetBERT)

# ---------------------------------------------------------------------------
# BiLSTM NER Head  (Stage 3)
#   BiLSTM(256→128×2=256) → Linear(256→C)
# ---------------------------------------------------------------------------

NER_BILSTM_HIDDEN = 128         # per direction; concat output = 256
NER_DROPOUT       = 0.1

# CoNLL-2003 BIO label schema
NER_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC",
]
NER_NUM_CLASSES  = len(NER_LABELS)          # 9
NER_LABEL2ID     = {l: i for i, l in enumerate(NER_LABELS)}
NER_ID2LABEL     = {i: l for i, l in enumerate(NER_LABELS)}
NER_IGNORE_INDEX = -100     # masked positions excluded from cross-entropy

# ---------------------------------------------------------------------------
# Noise  (Stage 3 student input augmentation)
# ---------------------------------------------------------------------------

NOISE_P_SUB       = 0.10    # char substitution (visual homoglyph)
NOISE_P_INS       = 0.05    # char insertion
NOISE_P_DEL       = 0.05    # char deletion
NOISE_P_SPACE_INS = 0.02    # space insertion mid-word (word splitting)

# ---------------------------------------------------------------------------
# Stage 1  —  BERT → RetBERT sentence-level distillation
# ---------------------------------------------------------------------------

STAGE1_EPOCHS       = 5
STAGE1_BATCH_SIZE   = 32
STAGE1_LR           = 5e-5
STAGE1_WARMUP_STEPS = 1000
STAGE1_MAX_WORDS    = 64        # max words per sentence (word-level truncation)
STAGE1_MAX_SUBWORDS = 128       # max subword tokens fed to BERT

# Corpus: WikiText-103 + CoNLL-2003 train split (clean sentences only)
STAGE1_WIKITEXT_DATASET = "wikitext"
STAGE1_WIKITEXT_CONFIG  = "wikitext-103-raw-v1"
STAGE1_CONLL_DATASET    = "eriktks/conll2003"

# ---------------------------------------------------------------------------
# Stage 2  —  RetBERT → LightRet token-level compression
# ---------------------------------------------------------------------------

STAGE2_EPOCHS       = 5
STAGE2_BATCH_SIZE   = 64
STAGE2_LR           = 3e-4
STAGE2_WARMUP_STEPS = 500
STAGE2_MAX_WORDS    = 64

# Same corpus as Stage 1
STAGE2_WIKITEXT_DATASET = STAGE1_WIKITEXT_DATASET
STAGE2_WIKITEXT_CONFIG  = STAGE1_WIKITEXT_CONFIG
STAGE2_CONLL_DATASET    = STAGE1_CONLL_DATASET

# ---------------------------------------------------------------------------
# Stage 3  —  Noisy-student NER fine-tuning
# ---------------------------------------------------------------------------

STAGE3_EPOCHS       = 10
STAGE3_BATCH_SIZE   = 32
STAGE3_LR           = 2e-4
STAGE3_WARMUP_STEPS = 200
STAGE3_MAX_WORDS    = 64
STAGE3_BETA         = 0.5   # L_total = β·L_class + (1−β)·L_distill

STAGE3_CONLL_DATASET = "eriktks/conll2003"
