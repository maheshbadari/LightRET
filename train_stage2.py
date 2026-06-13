"""
train_stage2.py — Stage 2: Token-level compression RetBERT -> LightRet.

Objective:
    Teach LightRet (with linear projector) to match RetBERT's token-level
    hidden states. Both models share the same word tokenization, so positions
    align directly — no shift alignment needed.
    Loss: mean cosine distance over all valid token positions.

Run:
    python train_stage2.py

Prerequisites:
    weights/retbert_stage1.pt  — produced by train_stage1.py

Checkpoint saved to: weights/lightret_stage2.pt
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import (
    DEVICE,
    STAGE1_CKPT,
    STAGE2_CKPT,
    STAGE2_EPOCHS,
    STAGE2_BATCH_SIZE,
    STAGE2_LR,
    STAGE2_WARMUP_STEPS,
    STAGE2_MAX_WORDS,
    WEIGHTS_DIR,
)
from src.models.retbert import RetBERT
from src.models.lightret import LightRet
from src.data.dataset import PretrainDataset, pretrain_collate
from src.losses import stage2_loss


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

    print(f"Device: {DEVICE}")

    # ---- Teacher: RetBERT (frozen) ----
    print(f"Loading RetBERT teacher from {STAGE1_CKPT} ...")
    if not STAGE1_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {STAGE1_CKPT}\n"
            "Run train_stage1.py first."
        )
    retbert = RetBERT().to(DEVICE)
    retbert.load_state_dict(
        torch.load(STAGE1_CKPT, map_location=DEVICE, weights_only=True)
    )
    retbert.eval()
    for p in retbert.parameters():
        p.requires_grad_(False)

    # ---- Student: LightRet with projector ----
    print("Building LightRet student (Stage 2, with projector) ...")
    lightret = LightRet.for_stage2().to(DEVICE)
    trainable = [p for p in lightret.parameters() if p.requires_grad]
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ---- Dataset ----
    print("Loading dataset ...")
    dataset = PretrainDataset(split="train", max_words=STAGE2_MAX_WORDS)
    loader  = DataLoader(
        dataset,
        batch_size=STAGE2_BATCH_SIZE,
        shuffle=True,
        collate_fn=pretrain_collate,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )

    optimizer   = AdamW(trainable, lr=STAGE2_LR, weight_decay=0.01)
    total_steps = STAGE2_EPOCHS * len(loader)
    scheduler   = make_scheduler(optimizer, STAGE2_WARMUP_STEPS, total_steps)

    best_loss = float("inf")

    for epoch in range(1, STAGE2_EPOCHS + 1):
        lightret.train()
        epoch_loss = 0.0

        for step, batch_words in enumerate(loader, 1):
            lengths = [len(s) for s in batch_words]

            # Teacher: RetBERT token-level hidden states (frozen)
            with torch.no_grad():
                h_retbert, _ = retbert(batch_words)   # (B, L_max, 768)

            # Student: LightRet projected to 768
            h_proj = lightret(batch_words)             # (B, L_max, 768)

            loss = stage2_loss(h_retbert, h_proj, lengths)

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
            torch.save(lightret.state_dict(), STAGE2_CKPT)
            print(f"  Saved checkpoint: {STAGE2_CKPT}")

    print(f"\nStage 2 complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {STAGE2_CKPT}")


if __name__ == "__main__":
    train()
