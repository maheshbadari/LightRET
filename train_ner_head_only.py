"""
train_ner_head_only.py — Quick recovery script.

If lightret_stage3.pt exists but ner_head_stage3.pt is missing, run this.
Freezes the Stage 3 backbone and trains only the NER head for a few epochs.

Usage:
    python train_ner_head_only.py
    python train_ner_head_only.py --epochs 3
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.config import (
    DEVICE, STAGE3_CKPT, NER_HEAD_CKPT, WEIGHTS_DIR,
    STAGE3_BATCH_SIZE, NER_IGNORE_INDEX,
)
from src.models.lightret import LightRet
from src.models.ner_head import NERHead
from src.data.dataset import NERDataset, ner_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=str(STAGE3_CKPT))
    parser.add_argument("--epochs",   type=int, default=3)
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device  : {DEVICE}")
    print(f"Loading backbone: {args.backbone}")

    backbone = LightRet.for_stage3()
    backbone.load_state_dict(
        torch.load(args.backbone, map_location="cpu", weights_only=True)
    )
    backbone.to(DEVICE).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)   # fully frozen

    head = NERHead().to(DEVICE)
    print(f"NER head params: {sum(p.numel() for p in head.parameters()):,}")

    train_ds = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds = NERDataset(split="validation", apply_noise_aug=False)
    print(f"Train: {len(train_ds):,}  Val: {len(valid_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=STAGE3_BATCH_SIZE,
                              shuffle=True,  collate_fn=ner_collate, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=STAGE3_BATCH_SIZE,
                              shuffle=False, collate_fn=ner_collate, num_workers=2)

    optimizer = AdamW(head.parameters(), lr=2e-4, weight_decay=0.01)
    best_val  = float("inf")

    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for step, batch in enumerate(train_loader, 1):
            noisy_words   = batch["noisy_words"]
            noisy_labels  = batch["noisy_labels"].to(DEVICE)
            noisy_lengths = batch["noisy_lengths"]

            with torch.no_grad():
                h = backbone(noisy_words)
            logits = head(h, noisy_lengths)
            B, L, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(-1, C), noisy_labels.reshape(-1),
                ignore_index=NER_IGNORE_INDEX,
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += loss.item()

            if step % 200 == 0:
                print(f"  E{epoch} S{step}/{len(train_loader)}  loss={total/step:.4f}", flush=True)

        # Validation
        head.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                noisy_words   = batch["noisy_words"]
                noisy_labels  = batch["noisy_labels"].to(DEVICE)
                noisy_lengths = batch["noisy_lengths"]
                h      = backbone(noisy_words)
                logits = head(h, noisy_lengths)
                B, L, C = logits.shape
                val_total += F.cross_entropy(
                    logits.reshape(-1, C), noisy_labels.reshape(-1),
                    ignore_index=NER_IGNORE_INDEX,
                ).item()

        val_ce = val_total / len(valid_loader)
        print(f"Epoch {epoch} — train={total/len(train_loader):.4f}  val_ce={val_ce:.4f}")

        if val_ce < best_val:
            best_val = val_ce
            torch.save(head.state_dict(), NER_HEAD_CKPT)
            print(f"  Saved: {NER_HEAD_CKPT}")

    print(f"\nDone. Best val CE: {best_val:.4f}")
    print(f"NER head saved to: {NER_HEAD_CKPT}")


if __name__ == "__main__":
    main()
