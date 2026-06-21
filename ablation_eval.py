"""
ablation_eval.py — Evaluate all ablation variants and print the comparison table.

Loads each variant checkpoint, evaluates on CoNLL-2003 test at clean + medium
noise, and prints a single comparison table matching paper Table 7 (ablation).

Usage:
    python ablation_eval.py                  # evaluate all available variants
    python ablation_eval.py --seeds 3        # fewer seeds for speed

Expected checkpoints (produced by ablation_train.py):
    weights/lightret_stage3.pt    + weights/ner_head_stage3.pt      (full model)
    weights/abl_no_noise.pt       + weights/abl_no_noise_head.pt
    weights/abl_beta1.pt          + weights/abl_beta1_head.pt
    weights/abl_beta0.pt          + weights/abl_beta0_head.pt
    weights/abl_random_emb.pt     + weights/abl_random_emb_head.pt
    weights/abl_no_stage2.pt      + weights/abl_no_stage2_head.pt
    weights/abl_no_stage1.pt      + weights/abl_no_stage1_head.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import DEVICE, WEIGHTS_DIR, NER_ID2LABEL, NER_IGNORE_INDEX
from src.models.lightret import LightRet
from src.models.ner_head import NERHead
from src.data.dataset import NERDataset
from src.noise import apply_noise

try:
    from seqeval.metrics import f1_score as seqeval_f1
    HAS_SEQEVAL = True
except ImportError:
    HAS_SEQEVAL = False
    print("[warn] seqeval not installed — results will be token-level accuracy, not entity F1")


# ---------------------------------------------------------------------------
# Variant registry  (label, backbone_ckpt, head_ckpt)
# ---------------------------------------------------------------------------

VARIANTS = [
    ("LightRet (full, 3-stage)",          "lightret_stage3.pt",       "ner_head_stage3.pt"),
    ("w/o Stage 1 (skip RetBERT)",        "abl_no_stage1.pt",         "abl_no_stage1_head.pt"),
    ("w/o Stage 2 (direct NER finetune)", "abl_no_stage2.pt",         "abl_no_stage2_head.pt"),
    ("w/o noise augmentation",            "abl_no_noise.pt",          "abl_no_noise_head.pt"),
    ("L3 = L_class only (β=1)",           "abl_beta1.pt",             "abl_beta1_head.pt"),
    ("L3 = L_distill only (β=0)",         "abl_beta0.pt",             "abl_beta0_head.pt"),
    ("Random char embeddings",            "abl_random_emb.pt",        "abl_random_emb_head.pt"),
]

NOISE_CONFIGS = {
    "clean":  dict(p_sub=0.00, p_ins=0.00, p_del=0.00, p_space_ins=0.00),
    "medium": dict(p_sub=0.10, p_ins=0.05, p_del=0.05, p_space_ins=0.02),
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(backbone_path: Path, head_path: Path):
    backbone = LightRet.for_stage3()
    backbone.load_state_dict(
        torch.load(backbone_path, map_location="cpu", weights_only=True)
    )
    backbone.to(DEVICE).eval()

    head = NERHead()
    head.load_state_dict(
        torch.load(head_path, map_location="cpu", weights_only=True)
    )
    head.to(DEVICE).eval()
    return backbone, head


@torch.no_grad()
def predict(backbone, head, words):
    hidden = backbone([words])
    logits = head(hidden, [len(words)])
    ids    = logits[0, :len(words)].argmax(-1).cpu().tolist()
    return [NER_ID2LABEL[i] for i in ids]


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def entity_f1(all_true, all_pred):
    if HAS_SEQEVAL:
        return seqeval_f1(all_true, all_pred) * 100.0
    # Fallback: entity-token accuracy
    correct = total = 0
    for ts, ps in zip(all_true, all_pred):
        for t, p in zip(ts, ps):
            if t != "O":
                total   += 1
                correct += int(t == p)
    return 100.0 * correct / max(1, total)


# ---------------------------------------------------------------------------
# Single-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_variant(backbone, head, test_ds, noise_cfg, n_seeds):
    import numpy as np
    f1s = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        all_true, all_pred = [], []
        for item in test_ds:
            words       = item["clean_words"]
            true_labels = item["clean_labels"]

            if any(v > 0 for v in noise_cfg.values()):
                noisy_words = []
                for w in words:
                    nw, _, _ = apply_noise(w, rng=rng, **noise_cfg)
                    noisy_words.append(nw.strip() or w)
                eval_words = noisy_words
            else:
                eval_words = words

            valid_true = [NER_ID2LABEL[l] for l in true_labels
                          if l != NER_IGNORE_INDEX]
            preds      = predict(backbone, head, eval_words)[:len(valid_true)]

            all_true.append(valid_true)
            all_pred.append(preds)

        f1s.append(entity_f1(all_true, all_pred))

    mean = sum(f1s) / len(f1s)
    std  = float(torch.tensor(f1s).std().item()) if len(f1s) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5,
                        help="Seeds per noise level per variant (default 5)")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Seeds  : {args.seeds}")
    print("\nLoading CoNLL-2003 test split ...")
    test_ds = NERDataset(split="test", apply_noise_aug=False)
    print(f"  {len(test_ds):,} sentences\n")

    col_w = 38
    print("=" * (col_w + 30))
    print(f"{'Configuration':<{col_w}} {'Clean F1':>10}  {'Medium F1':>10}")
    print("=" * (col_w + 30))

    for label, bb_name, head_name in VARIANTS:
        bb_path   = WEIGHTS_DIR / bb_name
        head_path = WEIGHTS_DIR / head_name

        if not bb_path.exists() or not head_path.exists():
            missing = []
            if not bb_path.exists():   missing.append(bb_name)
            if not head_path.exists(): missing.append(head_name)
            print(f"{label:<{col_w}} {'MISSING':>10}  {'MISSING':>10}"
                  f"  [{', '.join(missing)}]")
            continue

        print(f"  evaluating: {label} ...", flush=True)
        backbone, head = load_model(bb_path, head_path)

        clean_f1,  clean_std  = evaluate_variant(
            backbone, head, test_ds, NOISE_CONFIGS["clean"],  args.seeds)
        medium_f1, medium_std = evaluate_variant(
            backbone, head, test_ds, NOISE_CONFIGS["medium"], args.seeds)

        # Free GPU memory between variants
        del backbone, head
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"{label:<{col_w}} "
            f"{clean_f1:>7.2f}±{clean_std:<4.2f}  "
            f"{medium_f1:>7.2f}±{medium_std:<4.2f}"
        )

    print("=" * (col_w + 30))
    metric_name = "entity-level F1 (seqeval)" if HAS_SEQEVAL else "entity-token accuracy"
    print(f"\nMetric: {metric_name}  |  Seeds per cell: {args.seeds}")
    print("Run `pip install seqeval` for proper entity-level F1.")


if __name__ == "__main__":
    main()
