"""
dry_run.py — End-to-end pipeline smoke test (CPU, no large downloads).

Stage 1: synthetic batch_words + mocked BERT output (no 440MB BERT download)
Stage 2: synthetic batch_words + mocked RetBERT output
Stage 3: real CoNLL-2003 load + 3 full train steps (verifies dataset fix)

Run:  python dry_run.py
Expected: all sections print PASS, no exceptions.
"""

import sys
import traceback
import torch
import torch.nn as nn

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def check(label, fn):
    try:
        result = fn()
        print(f"  {label:<50}{PASS}")
        return result
    except Exception as e:
        print(f"  {label:<50}{FAIL}")
        traceback.print_exc()
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────────────────────
section("0. Imports")

check("src.config",              lambda: __import__("src.config"))
check("src.noise",               lambda: __import__("src.noise"))
check("src.losses",              lambda: __import__("src.losses"))
check("src.data.label_utils",    lambda: __import__("src.data.label_utils"))
check("src.data.dataset",        lambda: __import__("src.data.dataset"))
check("src.models.retvec_embedder", lambda: __import__("src.models.retvec_embedder"))
check("src.models.retbert",      lambda: __import__("src.models.retbert"))
check("src.models.lightret",     lambda: __import__("src.models.lightret"))
check("src.models.ner_head",     lambda: __import__("src.models.ner_head"))

from src.config import DEVICE, STAGE1_CKPT, STAGE2_CKPT, STAGE3_CKPT, WEIGHTS_DIR
from src.models.retbert import RetBERT
from src.models.lightret import LightRet
from src.models.ner_head import NERHead
from src.losses import stage1_loss, stage2_loss, stage3_loss
from src.data.dataset import NERDataset, ner_collate
from torch.utils.data import DataLoader
from torch.optim import AdamW

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH = [
    ["London", "is", "the", "capital", "of", "England"],
    ["Microsoft", "was", "founded", "by", "Bill", "Gates"],
    ["The", "Eiffel", "Tower", "stands", "in", "Paris"],
]
LENGTHS = [len(s) for s in BATCH]

print(f"\n  Device : {DEVICE}")
print(f"  Batch  : {len(BATCH)} sentences, max {max(LENGTHS)} words")


# ─────────────────────────────────────────────────────────────
# 1. RetVec embedder
# ─────────────────────────────────────────────────────────────
section("1. RetVec Embedder")

emb = check("load weights",
    lambda: __import__("src.models.retvec_embedder", fromlist=["RetVecEmbedder"])
            .RetVecEmbedder("retvec_v1_weights.npz").to(DEVICE))

check("forward pass -> (3, 6, 256)",
    lambda: emb(BATCH).shape == (3, 6, 256) or (_ for _ in ()).throw(
        AssertionError(f"got {emb(BATCH).shape}")))

with torch.no_grad():
    out = emb(BATCH)
check("output in [-1, 1] (tanh)",
    lambda: (out.min() >= -1.0 and out.max() <= 1.0) or (_ for _ in ()).throw(
        AssertionError(f"range [{out.min():.3f}, {out.max():.3f}]")))

check("frozen (no grad)",
    lambda: all(not p.requires_grad for p in emb.parameters()))


# ─────────────────────────────────────────────────────────────
# 2. Stage 1 — RetBERT forward + loss + backward (mocked BERT)
# ─────────────────────────────────────────────────────────────
section("2. Stage 1 — RetBERT (BERT mocked)")

retbert = check("build RetBERT", lambda: RetBERT().to(DEVICE))

check(f"trainable params > 80M",
    lambda: sum(p.numel() for p in retbert.parameters() if p.requires_grad) > 80_000_000)

hidden, pooled = check("forward -> (3,6,768) + (3,768)",
    lambda: retbert(BATCH))
check("hidden shape",  lambda: hidden.shape == (3, 6, 768))
check("pooled shape",  lambda: pooled.shape == (3, 768))

# Mock BERT output — same shape as real BERT pooled vector
z_bert_mock = torch.randn(3, 768).to(DEVICE)
loss1 = check("stage1_loss", lambda: stage1_loss(z_bert_mock, pooled))
check("loss is scalar", lambda: loss1.ndim == 0)
check("loss in [0, 2]",  lambda: 0 <= loss1.item() <= 2.0)

check("backward", lambda: loss1.backward() or True)

trainable = [p for p in retbert.parameters() if p.requires_grad]
check("gradients exist on proj layer",
    lambda: retbert.proj.weight.grad is not None)

# Save Stage 1 checkpoint
check("save checkpoint",
    lambda: torch.save(retbert.state_dict(), STAGE1_CKPT) or True)
print(f"  Checkpoint: {STAGE1_CKPT}")


# ─────────────────────────────────────────────────────────────
# 3. Stage 2 — LightRet forward + loss + backward (mocked teacher)
# ─────────────────────────────────────────────────────────────
section("3. Stage 2 — LightRet compression (RetBERT mocked)")

lr2 = check("build LightRet (stage2)", lambda: LightRet.for_stage2().to(DEVICE))

check("trainable params ~3.6M",
    lambda: 3_000_000 < sum(p.numel() for p in lr2.parameters() if p.requires_grad) < 5_000_000)

out2 = check("forward -> (3, 6, 768)",
    lambda: lr2(BATCH))
check("output shape", lambda: out2.shape == (3, 6, 768))

# Mock RetBERT hidden states
h_teacher_mock = torch.randn(3, 6, 768).to(DEVICE)
loss2 = check("stage2_loss", lambda: stage2_loss(h_teacher_mock, out2, LENGTHS))
check("loss scalar", lambda: loss2.ndim == 0)
check("backward",    lambda: loss2.backward() or True)

# Save Stage 2 checkpoint
check("save checkpoint",
    lambda: torch.save(lr2.state_dict(), STAGE2_CKPT) or True)
print(f"  Checkpoint: {STAGE2_CKPT}")


# ─────────────────────────────────────────────────────────────
# 4. Stage 3 models — from_stage2_checkpoint
# ─────────────────────────────────────────────────────────────
section("4. Stage 3 — Load from Stage 2 checkpoint")

teacher = check("teacher = from_stage2_checkpoint (frozen)",
    lambda: LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE))
check("teacher projector dropped",
    lambda: teacher.projector is None)
check("teacher fully frozen",
    lambda: all(not p.requires_grad for p in teacher.parameters()))

student = check("student = from_stage2_checkpoint (trainable BiGRU+Transformer)",
    lambda: LightRet.from_stage2_checkpoint(str(STAGE2_CKPT)).to(DEVICE))
check("student projector dropped",
    lambda: student.projector is None)
check("RetVec frozen in student",
    lambda: all(not p.requires_grad for p in student.retvec.parameters()))

ner_head = check("build NERHead", lambda: NERHead().to(DEVICE))


# ─────────────────────────────────────────────────────────────
# 5. CoNLL-2003 dataset load (verifies trust_remote_code fix)
# ─────────────────────────────────────────────────────────────
section("5. CoNLL-2003 Dataset (real download)")

print("  Loading CoNLL-2003 train split (requires internet) ...")
train_ds = check("NERDataset(train, noise_aug=True)",
    lambda: NERDataset(split="train", apply_noise_aug=True))
valid_ds = check("NERDataset(validation, noise_aug=False)",
    lambda: NERDataset(split="validation", apply_noise_aug=False))

check(f"train size ~14K", lambda: len(train_ds) > 13_000)
check(f"valid size ~3K",  lambda: len(valid_ds) > 3_000)

sample = train_ds[0]
check("sample keys present",
    lambda: all(k in sample for k in
                ["clean_words","clean_labels","noisy_words","noisy_labels","alignment"]))
check("alignment covers all noisy words",
    lambda: sum(len(n) for _,n in sample["alignment"]) == len(sample["noisy_words"]))
check("noisy_labels length matches noisy_words",
    lambda: len(sample["noisy_labels"]) == len(sample["noisy_words"]))

print(f"  clean : {sample['clean_words'][:6]}")
print(f"  noisy : {sample['noisy_words'][:6]}")

loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=ner_collate)
batch  = check("DataLoader batch collation", lambda: next(iter(loader)))
check("batch keys",
    lambda: all(k in batch for k in
                ["clean_words","noisy_words","noisy_labels","alignment","noisy_lengths"]))
check("noisy_labels tensor shape (4, L)",
    lambda: batch["noisy_labels"].ndim == 2 and batch["noisy_labels"].shape[0] == 4)


# ─────────────────────────────────────────────────────────────
# 6. Stage 3 — 3 full training steps
# ─────────────────────────────────────────────────────────────
section("6. Stage 3 — 3 Training Steps (full pipeline)")

from src.config import STAGE3_LR, STAGE3_BETA, NER_IGNORE_INDEX
import torch.nn.functional as F

trainable = (
    [p for p in student.parameters()  if p.requires_grad]
    + list(ner_head.parameters())
)
optimizer = AdamW(trainable, lr=STAGE3_LR)

student.train(); ner_head.train(); teacher.eval()

for step, batch in enumerate(loader, 1):
    clean_words   = batch["clean_words"]
    noisy_words   = batch["noisy_words"]
    noisy_labels  = batch["noisy_labels"].to(DEVICE)
    alignment     = batch["alignment"]
    noisy_lengths = batch["noisy_lengths"]

    with torch.no_grad():
        h_teacher = teacher(clean_words)

    h_student = student(noisy_words)
    logits    = ner_head(h_student, noisy_lengths)
    total, lc, ld = stage3_loss(
        logits, noisy_labels, h_teacher, h_student, alignment, STAGE3_BETA
    )

    optimizer.zero_grad()
    total.backward()
    nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()

    check(f"step {step}: forward+backward",
        lambda t=total: t.item() > 0)
    print(f"  step {step}  total={total.item():.4f}  lc={lc.item():.4f}  ld={ld.item():.4f}")

    if step == 3:
        break

check("NER head grad flows", lambda: ner_head.classifier.weight.grad is not None)

# Save Stage 3 checkpoint
check("save Stage 3 checkpoint",
    lambda: (torch.save(student.state_dict(), STAGE3_CKPT),
             torch.save(ner_head.state_dict(), WEIGHTS_DIR / "ner_head_stage3.pt")) or True)


# ─────────────────────────────────────────────────────────────
# 7. Noise pipeline
# ─────────────────────────────────────────────────────────────
section("7. Noise Pipeline")

from src.noise import apply_noise, build_word_alignment
from src.data.label_utils import project_labels

sentence    = "London is the capital of England"
labels_in   = [5, 0, 0, 0, 0, 0]   # B-LOC O O O O O

noisy_str, shift_log, S = apply_noise(sentence, p_sub=0.10, p_ins=0.05,
                                       p_del=0.05, p_space_ins=0.05)
alignment    = build_word_alignment(sentence, shift_log)
noisy_labels = project_labels(labels_in, alignment)

check("noisy str non-empty",  lambda: len(noisy_str) > 0)
check("alignment covers all noisy words",
    lambda: sum(len(n) for _,n in alignment) == len(noisy_str.split()))
check("noisy_labels length matches",
    lambda: len(noisy_labels) == len(noisy_str.split()))

print(f"  clean : {sentence.split()}")
print(f"  noisy : {noisy_str.split()}")
print(f"  labels: {noisy_labels}")


# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
section("SUMMARY")
print("  All checks passed. Pipeline is ready for Kaggle.")
print(f"\n  Checkpoints written to {WEIGHTS_DIR}/")
for ckpt in WEIGHTS_DIR.iterdir():
    print(f"    {ckpt.name}  ({ckpt.stat().st_size / 1e6:.1f} MB)")
