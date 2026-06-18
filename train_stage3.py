"""
train_stage3.py — Stage 3: Noisy-student NER fine-tuning.

Objective:
    Fine-tune LightRet + BiLSTM NER head on noisy CoNLL-2003.
    Teacher: LightRet loaded from Stage 2, frozen, processes CLEAN text.
    Student: LightRet + NERHead, processes NOISY text.

    Loss: beta * L_class + (1-beta) * L_distill
        L_class  : cross-entropy over BIO labels projected to noisy positions
        L_distill: per-word cosine distance with dynamic shift alignment

Run:
    python train_stage3.py

Prerequisites:
    weights/lightret_stage2.pt  — produced by train_stage2.py

Checkpoints:
    weights/lightret_stage3.pt  — LightRet backbone state dict
"""

import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import (
    DEVICE,
    STAGE2_CKPT,
    STAGE3_CKPT,
    NER_HEAD_CKPT,
    STAGE3_EPOCHS,
    STAGE3_BATCH_SIZE,
    STAGE3_LR,
    STAGE3_WARMUP_STEPS,
    STAGE3_BETA,
    WEIGHTS_DIR,
)
from src.models.lightret import LightRet
from src.models.ner_head import NERHead
from src.data.dataset import NERDataset, ner_collate
from src.losses import stage3_loss


# ---------------------------------------------------------------------------
# Evaluation (greedy argmax, token-level F1 via seqeval or simple accuracy)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    lightret_student: nn.Module,
    ner_head: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute mean cross-entropy on the validation set (quick proxy for NER quality).
    Full seqeval F1 can be plugged in here but requires seqeval package.
    """
    from src.config import NER_IGNORE_INDEX, STAGE3_BETA
    import torch.nn.functional as F

    lightret_student.eval()
    ner_head.eval()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        noisy_words   = batch["noisy_words"]
        noisy_labels  = batch["noisy_labels"].to(device)
        noisy_lengths = batch["noisy_lengths"]

        h_student = lightret_student(noisy_words)
        logits    = ner_head(h_student, noisy_lengths)

        B, L, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, C),
            noisy_labels.reshape(-1),
            ignore_index=NER_IGNORE_INDEX,
        )
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(1, n_batches)


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

def train(epochs: int = STAGE3_EPOCHS, resume: bool = False) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Epochs: {epochs}  |  Resume: {resume}")

    # ---- Teacher: LightRet from Stage 2 checkpoint, frozen ----
    print(f"Loading LightRet teacher from {STAGE2_CKPT} ...")
    if not STAGE2_CKPT.exists():
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {STAGE2_CKPT}\n"
            "Run train_stage2.py first."
        )
    teacher = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ---- Student: resume from Stage 3 checkpoint or start from Stage 2 ----
    if resume and STAGE3_CKPT.exists():
        print(f"Resuming student from {STAGE3_CKPT} ...")
        student = LightRet.for_stage3()
        student.load_state_dict(
            torch.load(STAGE3_CKPT, map_location="cpu", weights_only=True)
        )
        student.to(DEVICE)
    else:
        print("Building student from Stage 2 checkpoint ...")
        student = LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE)

    print("Building NER head ...")
    ner_head = NERHead().to(DEVICE)

    # RetVec is frozen (inside LightRet.__init__ already), trainable = BiGRU + Transformer + NERHead
    trainable = (
        [p for p in student.parameters()  if p.requires_grad]
        + list(ner_head.parameters())
    )
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ---- Datasets ----
    print("Loading CoNLL-2003 ...")
    train_ds  = NERDataset(split="train",      apply_noise_aug=True)
    valid_ds  = NERDataset(split="validation", apply_noise_aug=False)
    print(f"  Train: {len(train_ds):,}  Val: {len(valid_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=STAGE3_BATCH_SIZE,
        shuffle=True,
        collate_fn=ner_collate,
        num_workers=2,
        pin_memory=(DEVICE.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=STAGE3_BATCH_SIZE,
        shuffle=False,
        collate_fn=ner_collate,
        num_workers=2,
    )

    optimizer   = AdamW(trainable, lr=STAGE3_LR, weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    scheduler   = make_scheduler(optimizer, STAGE3_WARMUP_STEPS, total_steps)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        student.train()
        ner_head.train()
        epoch_loss = epoch_lc = epoch_ld = 0.0

        for step, batch in enumerate(train_loader, 1):
            clean_words   = batch["clean_words"]
            noisy_words   = batch["noisy_words"]
            noisy_labels  = batch["noisy_labels"].to(DEVICE)
            alignment     = batch["alignment"]
            noisy_lengths = batch["noisy_lengths"]

            # Teacher: clean hidden states (frozen)
            with torch.no_grad():
                h_teacher = teacher(clean_words)       # (B, L_clean, 256)

            # Student: noisy hidden states
            h_student = student(noisy_words)           # (B, L_noisy, 256)
            logits    = ner_head(h_student, noisy_lengths)

            total, lc, ld = stage3_loss(
                logits, noisy_labels, h_teacher, h_student, alignment, STAGE3_BETA
            )

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += total.item()
            epoch_lc   += lc.item()
            epoch_ld   += ld.item()

            if step % 100 == 0:
                avg_t  = epoch_loss / step
                avg_lc = epoch_lc  / step
                avg_ld = epoch_ld  / step
                lr     = scheduler.get_last_lr()[0]
                print(
                    f"  E{epoch} S{step}/{len(train_loader)}  "
                    f"total={avg_t:.4f}  lc={avg_lc:.4f}  ld={avg_ld:.4f}  "
                    f"LR={lr:.2e}",
                    flush=True,
                )

        # Validation
        val_loss = evaluate(student, ner_head, valid_loader, DEVICE)
        avg_train = epoch_loss / len(train_loader)
        print(
            f"Epoch {epoch} — train={avg_train:.4f}  val_ce={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(student.state_dict(),   STAGE3_CKPT)
            torch.save(ner_head.state_dict(),  NER_HEAD_CKPT)
            print(f"  Saved: {STAGE3_CKPT}  |  {NER_HEAD_CKPT}")

    print(f"\nStage 3 complete. Best val CE: {best_val_loss:.4f}")
    print(f"Final checkpoint: {STAGE3_CKPT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=STAGE3_EPOCHS,
                        help="Number of training epochs (default: STAGE3_EPOCHS from config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume student from weights/lightret_stage3.pt instead of Stage 2")
    args = parser.parse_args()
    train(epochs=args.epochs, resume=args.resume)
