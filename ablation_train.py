"""
ablation_train.py — Train a single Stage 3 ablation variant of LightRet.

Each variant modifies one component of the full Stage 3 training (train_stage3.py).
Run once per variant, then evaluate all of them with ablation_eval.py.

Usage:
    python ablation_train.py --variant <name>

Variants:
    no_noise    Stage 3 without character noise augmentation (clean-only training)
    beta1       Stage 3 with β=1  (classification loss only, no distillation)
    beta0       Stage 3 with β=0  (distillation loss only, no classification)
    random_emb  Stage 3 with randomly initialized (untrained) RetVec weights
    no_stage2   NER fine-tune from scratch — fresh LightRet, no prior distillation
    no_stage1   Stage 2 + Stage 3 from scratch, BERT used directly as Stage 2 teacher

Checkpoints saved to weights/abl_<variant>.pt  and  weights/abl_<variant>_head.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import (
    DEVICE,
    STAGE2_CKPT,
    STAGE3_EPOCHS,
    STAGE3_BATCH_SIZE,
    STAGE3_LR,
    STAGE3_WARMUP_STEPS,
    STAGE3_BETA,
    WEIGHTS_DIR,
    NER_IGNORE_INDEX,
)
from src.models.lightret import LightRet
from src.models.ner_head import NERHead
from src.data.dataset import NERDataset, ner_collate
from src.losses import stage3_loss

import torch.nn.functional as F


VARIANTS = ["no_noise", "beta1", "beta0", "random_emb", "no_stage2", "no_stage1"]


# ---------------------------------------------------------------------------
# Cosine scheduler with linear warmup
# ---------------------------------------------------------------------------

def make_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Validation (cross-entropy proxy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(student, head, loader, device):
    student.eval(); head.eval()
    total, n = 0.0, 0
    for batch in loader:
        noisy_words   = batch["noisy_words"]
        noisy_labels  = batch["noisy_labels"].to(device)
        noisy_lengths = batch["noisy_lengths"]
        h = student(noisy_words)
        logits = head(h, noisy_lengths)
        B, L, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, C), noisy_labels.reshape(-1),
            ignore_index=NER_IGNORE_INDEX,
        )
        total += loss.item(); n += 1
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Shared Stage 3 training loop
# ---------------------------------------------------------------------------

def train_stage3_variant(
    student: nn.Module,
    ner_head: nn.Module,
    teacher: Optional[nn.Module],
    train_ds: NERDataset,
    valid_ds: NERDataset,
    beta: float,
    ckpt_backbone: Path,
    ckpt_head: Path,
    tag: str,
):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(
        train_ds, batch_size=STAGE3_BATCH_SIZE, shuffle=True,
        collate_fn=ner_collate, num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=STAGE3_BATCH_SIZE, shuffle=False,
        collate_fn=ner_collate, num_workers=2,
    )

    trainable = (
        [p for p in student.parameters() if p.requires_grad]
        + list(ner_head.parameters())
    )
    print(f"[{tag}] Trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer   = AdamW(trainable, lr=STAGE3_LR, weight_decay=0.01)
    total_steps = STAGE3_EPOCHS * len(train_loader)
    scheduler   = make_scheduler(optimizer, STAGE3_WARMUP_STEPS, total_steps)

    best_val = float("inf")

    for epoch in range(1, STAGE3_EPOCHS + 1):
        student.train(); ner_head.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            clean_words   = batch["clean_words"]
            noisy_words   = batch["noisy_words"]
            noisy_labels  = batch["noisy_labels"].to(DEVICE)
            alignment     = batch["alignment"]
            noisy_lengths = batch["noisy_lengths"]

            if teacher is not None:
                with torch.no_grad():
                    h_teacher = teacher(clean_words)
            else:
                # No distillation — build dummy teacher output of zeros
                h_teacher = torch.zeros(
                    len(noisy_words),
                    max(len(s) for s in clean_words),
                    256,
                    device=DEVICE,
                )

            h_student = student(noisy_words)
            logits    = ner_head(h_student, noisy_lengths)

            total, lc, ld = stage3_loss(
                logits, noisy_labels, h_teacher, h_student, alignment, beta
            )

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total.item()

            if step % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  [{tag}] E{epoch} S{step}/{len(train_loader)} "
                    f"loss={epoch_loss/step:.4f}  LR={lr:.2e}",
                    flush=True,
                )

        val_loss = validate(student, ner_head, valid_loader, DEVICE)
        print(f"  [{tag}] Epoch {epoch} — val_ce={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(student.state_dict(), ckpt_backbone)
            torch.save(ner_head.state_dict(), ckpt_head)
            print(f"    Saved {ckpt_backbone.name}  |  {ckpt_head.name}")

    print(f"[{tag}] Done. Best val CE: {best_val:.4f}")


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------

def run_no_noise():
    """Stage 3 without noise augmentation (clean-only NER fine-tuning)."""
    print("\n[no_noise] Stage 3 without character noise")
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {STAGE2_CKPT}")

    teacher = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student  = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    ner_head = NERHead().to(DEVICE)

    # apply_noise_aug=False on both splits
    train_ds = NERDataset(split="train",      apply_noise_aug=False)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)

    train_stage3_variant(
        student, ner_head, teacher, train_ds, valid_ds,
        beta=STAGE3_BETA,
        ckpt_backbone=WEIGHTS_DIR / "abl_no_noise.pt",
        ckpt_head=WEIGHTS_DIR / "abl_no_noise_head.pt",
        tag="no_noise",
    )


def run_beta1():
    """Stage 3 with β=1 (classification loss only)."""
    print("\n[beta1] Stage 3 with β=1 (no distillation signal)")
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {STAGE2_CKPT}")

    teacher = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student  = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    ner_head = NERHead().to(DEVICE)

    train_ds = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)

    train_stage3_variant(
        student, ner_head, teacher, train_ds, valid_ds,
        beta=1.0,                      # classification only
        ckpt_backbone=WEIGHTS_DIR / "abl_beta1.pt",
        ckpt_head=WEIGHTS_DIR / "abl_beta1_head.pt",
        tag="beta1",
    )


def run_beta0():
    """Stage 3 with β=0 (distillation loss only)."""
    print("\n[beta0] Stage 3 with β=0 (no classification signal)")
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {STAGE2_CKPT}")

    teacher = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student  = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    ner_head = NERHead().to(DEVICE)

    train_ds = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)

    train_stage3_variant(
        student, ner_head, teacher, train_ds, valid_ds,
        beta=0.0,                      # distillation only
        ckpt_backbone=WEIGHTS_DIR / "abl_beta0.pt",
        ckpt_head=WEIGHTS_DIR / "abl_beta0_head.pt",
        tag="beta0",
    )


def run_random_emb():
    """Stage 3 with randomly initialized (untrained) RetVec weights."""
    import numpy as np
    print("\n[random_emb] Stage 3 with random RetVec weights")
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(f"Stage 2 checkpoint not found: {STAGE2_CKPT}")

    teacher = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)

    # Randomize RetVec weights in the student (un-freeze temporarily, re-randomize, re-freeze)
    for name, param in student.retvec.named_parameters():
        nn.init.normal_(param.data, mean=0.0, std=0.02)
        param.requires_grad_(False)        # keep frozen — tests impact of pretrained weights

    ner_head = NERHead().to(DEVICE)

    train_ds = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)

    train_stage3_variant(
        student, ner_head, teacher, train_ds, valid_ds,
        beta=STAGE3_BETA,
        ckpt_backbone=WEIGHTS_DIR / "abl_random_emb.pt",
        ckpt_head=WEIGHTS_DIR / "abl_random_emb_head.pt",
        tag="random_emb",
    )


def run_no_stage2():
    """
    w/o Stage 2: fresh LightRet (no distillation pre-training), NER fine-tune only.
    β is forced to 1.0 — there is no meaningful teacher without Stage 2.
    """
    print("\n[no_stage2] Fresh LightRet backbone — NER fine-tune only (no prior distillation)")

    student  = LightRet.for_stage3().to(DEVICE)    # random init — no stage 2 weights
    ner_head = NERHead().to(DEVICE)

    train_ds = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)

    train_stage3_variant(
        student, ner_head, teacher=None, train_ds=train_ds, valid_ds=valid_ds,
        beta=1.0,                          # no teacher → classification loss only
        ckpt_backbone=WEIGHTS_DIR / "abl_no_stage2.pt",
        ckpt_head=WEIGHTS_DIR / "abl_no_stage2_head.pt",
        tag="no_stage2",
    )


def run_no_stage1():
    """
    w/o Stage 1: skip RetBERT pre-distillation.
    LightRet is initialized randomly and trained with BERT directly as the Stage 2 teacher,
    bypassing the intermediate RetBERT student.  This requires a separate Stage 2 run
    (train_stage2_no_stage1.py).  Here we run Stage 3 assuming a checkpoint produced by
    that run at weights/lightret_stage2_no_stage1.pt.
    """
    no_s1_ckpt = WEIGHTS_DIR / "lightret_stage2_no_stage1.pt"
    if not no_s1_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {no_s1_ckpt}\n"
            "Run train_stage2.py with --no-stage1 flag first."
        )

    print(f"\n[no_stage1] Using {no_s1_ckpt}")

    teacher = LightRet.from_stage2_checkpoint(str(no_s1_ckpt)).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    student  = LightRet.from_stage2_checkpoint(str(no_s1_ckpt)).to(DEVICE)
    ner_head = NERHead().to(DEVICE)

    train_ds = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)

    train_stage3_variant(
        student, ner_head, teacher, train_ds, valid_ds,
        beta=STAGE3_BETA,
        ckpt_backbone=WEIGHTS_DIR / "abl_no_stage1.pt",
        ckpt_head=WEIGHTS_DIR / "abl_no_stage1_head.pt",
        tag="no_stage1",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

VARIANT_FNS = {
    "no_noise":   run_no_noise,
    "beta1":      run_beta1,
    "beta0":      run_beta0,
    "random_emb": run_random_emb,
    "no_stage2":  run_no_stage2,
    "no_stage1":  run_no_stage1,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", required=True, choices=VARIANTS,
        help="Which ablation variant to train",
    )
    args = parser.parse_args()

    print(f"Device  : {DEVICE}")
    VARIANT_FNS[args.variant]()
