"""
train_stage1.py — Stage 1: Sentence-level distillation BERT -> RetBERT.

Objective:
    Teach RetBERT to produce sentence embeddings that match frozen BERT-Base.
    Both models see clean text. Loss: 1 - cosine_similarity(z_RetBERT, z_BERT).

Run:
    python train_stage1.py

Kaggle: upload retvec_v1_weights.npz as a dataset and set RETVEC_WEIGHTS in config.py.
Checkpoint saved to: weights/retbert_stage1.pt
"""

import sys
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import (
    DEVICE,
    BERT_MODEL_NAME,
    BERT_DIM,
    STAGE1_CKPT,
    STAGE1_EPOCHS,
    STAGE1_BATCH_SIZE,
    STAGE1_LR,
    STAGE1_WARMUP_STEPS,
    STAGE1_MAX_WORDS,
    STAGE1_MAX_SUBWORDS,
    WEIGHTS_DIR,
)
from src.models.retbert import RetBERT
from src.data.dataset import PretrainDataset, pretrain_collate
from src.losses import stage1_loss


# ---------------------------------------------------------------------------
# BERT teacher helpers
# ---------------------------------------------------------------------------

def load_bert_teacher(device: torch.device):
    """Load frozen BERT-Base and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device)
    return tokenizer, model


@torch.no_grad()
def bert_sentence_embedding(
    batch_words: list[list[str]],
    tokenizer,
    bert: nn.Module,
    device: torch.device,
    max_subwords: int = STAGE1_MAX_SUBWORDS,
) -> torch.Tensor:
    """
    Encode a batch of word-tokenized sentences with BERT.
    Joins words back to string, subword-tokenizes, MeanPools last hidden state.
    Returns (B, 768).
    """
    texts = [" ".join(words) for words in batch_words]
    enc = tokenizer(
        texts,
        max_length=max_subwords,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    out = bert(**enc)
    hidden = out.last_hidden_state           # (B, L_sub, 768)
    mask   = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return pooled                            # (B, 768)


# ---------------------------------------------------------------------------
# Scheduler: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading dataset ...")
    dataset  = PretrainDataset(split="train", max_words=STAGE1_MAX_WORDS)
    loader   = DataLoader(
        dataset,
        batch_size=STAGE1_BATCH_SIZE,
        shuffle=True,
        collate_fn=pretrain_collate,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("Loading BERT teacher ...")
    bert_tokenizer, bert = load_bert_teacher(DEVICE)

    print("Building RetBERT student ...")
    retbert = RetBERT().to(DEVICE)
    trainable = [p for p in retbert.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = AdamW(trainable, lr=STAGE1_LR, weight_decay=0.01)
    total_steps = STAGE1_EPOCHS * len(loader)
    scheduler   = make_scheduler(optimizer, STAGE1_WARMUP_STEPS, total_steps)

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, STAGE1_EPOCHS + 1):
        retbert.train()
        epoch_loss = 0.0

        for step, batch_words in enumerate(loader, 1):
            # Teacher: frozen BERT sentence embedding
            with torch.no_grad():
                z_bert = bert_sentence_embedding(
                    batch_words, bert_tokenizer, bert, DEVICE
                )

            # Student: RetBERT sentence embedding
            _, z_retbert = retbert(batch_words)

            loss = stage1_loss(z_bert, z_retbert)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()

            if step % 200 == 0:
                avg = epoch_loss / step
                lr  = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch}/{STAGE1_EPOCHS}  "
                    f"Step {step}/{len(loader)}  "
                    f"Loss {avg:.4f}  LR {lr:.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} done — avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(retbert.state_dict(), STAGE1_CKPT)
            print(f"  Saved checkpoint: {STAGE1_CKPT}")

    print(f"\nStage 1 complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {STAGE1_CKPT}")


if __name__ == "__main__":
    train()
