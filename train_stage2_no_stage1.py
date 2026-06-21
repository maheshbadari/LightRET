"""
train_stage2_no_stage1.py — Stage 2 variant: BERT-base as direct teacher.

Used for the "w/o Stage 1" ablation: skips RetBERT entirely and distils
BERT-base token-level hidden states directly into LightRet.

The key difference from train_stage2.py: BERT uses subword (WordPiece)
tokenisation while LightRet operates at word level.  We bridge this by
mean-pooling BERT's subword hidden states back to word level before
computing the cosine distillation loss.

Checkpoint saved to: weights/lightret_stage2_no_stage1.pt
Then run: python ablation_train.py --variant no_stage1
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModel

from src.config import (
    DEVICE,
    BERT_MODEL_NAME,
    STAGE2_EPOCHS,
    STAGE2_BATCH_SIZE,
    STAGE2_LR,
    STAGE2_WARMUP_STEPS,
    STAGE2_MAX_WORDS,
    WEIGHTS_DIR,
)
from src.models.lightret import LightRet
from src.data.dataset import PretrainDataset, pretrain_collate
from src.losses import stage2_loss


NO_STAGE1_CKPT = WEIGHTS_DIR / "lightret_stage2_no_stage1.pt"


# ---------------------------------------------------------------------------
# Subword → word alignment
# ---------------------------------------------------------------------------

def bert_to_word_level(
    hidden: torch.Tensor,           # (B, T, 768) — BERT last hidden states
    word_ids_list: list[list],      # B × T, word_ids() per sentence
    n_words: list[int],             # valid word count per sentence
) -> torch.Tensor:                  # (B, max_words, 768)
    """
    Mean-pool BERT subword hidden states to word level.
    Tokens with word_id=None (CLS, SEP, PAD) are ignored.
    """
    B, T, D = hidden.shape
    max_words = max(n_words)
    out = torch.zeros(B, max_words, D, device=hidden.device)

    for b in range(B):
        word_to_toks: dict[int, list[int]] = {}
        for tok_i, wid in enumerate(word_ids_list[b]):
            if wid is None or wid >= n_words[b]:
                continue
            word_to_toks.setdefault(wid, []).append(tok_i)
        for wid, tok_ids in word_to_toks.items():
            out[b, wid] = hidden[b, tok_ids].mean(dim=0)

    return out                       # (B, max_words, 768)


# ---------------------------------------------------------------------------
# Scheduler
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

    print(f"Device : {DEVICE}")
    print("w/o Stage 1 ablation: BERT-base used as direct Stage 2 teacher")

    # ---- Teacher: BERT-base (frozen) ----
    print(f"Loading BERT-base teacher ({BERT_MODEL_NAME}) ...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)
    print(f"  BERT params (frozen): {sum(p.numel() for p in bert.parameters()):,}")

    # ---- Student: LightRet with projector (256→768) ----
    print("Building LightRet student (with projector, random init) ...")
    lightret = LightRet.for_stage2().to(DEVICE)
    trainable = [p for p in lightret.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ---- Dataset ----
    print("Loading pretrain dataset ...")
    dataset = PretrainDataset(split="train", max_words=STAGE2_MAX_WORDS)
    loader  = DataLoader(
        dataset,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=True,
        collate_fn=pretrain_collate,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    print(f"  {len(dataset):,} sentences  |  {len(loader):,} steps/epoch")

    optimizer   = AdamW(trainable, lr=STAGE2_LR, weight_decay=0.01)
    total_steps = STAGE2_EPOCHS * len(loader)
    scheduler   = make_scheduler(optimizer, STAGE2_WARMUP_STEPS, total_steps)

    best_loss = float("inf")

    for epoch in range(1, STAGE2_EPOCHS + 1):
        lightret.train()
        epoch_loss = 0.0

        for step, batch_words in enumerate(loader, 1):
            n_words = [len(s) for s in batch_words]

            # ---- BERT: subword tokenise → hidden states → word-level ----
            with torch.no_grad():
                enc = tokenizer(
                    batch_words,
                    is_split_into_words=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(DEVICE)

                bert_out = bert(**enc)
                hidden   = bert_out.last_hidden_state   # (B, T, 768)

                # Collect word_ids per sentence (CPU, list of lists)
                word_ids_list = [
                    enc.word_ids(batch_index=b)
                    for b in range(len(batch_words))
                ]

                # Mean-pool subwords → word level  (B, max_words, 768)
                h_bert_word = bert_to_word_level(hidden, word_ids_list, n_words)

            # ---- LightRet: word-level → projected 768d ----
            h_proj = lightret(batch_words)              # (B, max_words, 768)

            # ---- Token-level cosine distillation loss ----
            loss = stage2_loss(h_bert_word, h_proj, n_words)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 200 == 0:
                avg = epoch_loss / step
                lr  = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch}/{STAGE2_EPOCHS}  "
                    f"Step {step}/{len(loader)}  "
                    f"Loss {avg:.4f}  LR {lr:.2e}",
                    flush=True,
                )

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch} done — avg loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(lightret.state_dict(), NO_STAGE1_CKPT)
            print(f"  Saved: {NO_STAGE1_CKPT}")

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {NO_STAGE1_CKPT}")
    print("Next: python ablation_train.py --variant no_stage1")


if __name__ == "__main__":
    train()
