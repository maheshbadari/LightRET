"""
evaluate.py — Full evaluation of the trained LightRet NER model.

Produces:
  1. Entity-level F1 on CoNLL-2003 test set at 4 noise levels
     (clean, low, medium, high)
  2. Per-entity-type F1 breakdown (PER, ORG, LOC, MISC)
  3. Inference speed benchmark (GPU + CPU)

Usage:
    python evaluate.py
    python evaluate.py --backbone weights/lightret_stage3.pt \
                       --head     weights/ner_head_stage3.pt

Requirements:
    pip install seqeval
"""

import argparse
import time
import torch
import torch.nn as nn

from src.config import (
    DEVICE,
    STAGE3_CKPT,
    NER_HEAD_CKPT,
    NER_ID2LABEL,
    NER_IGNORE_INDEX,
)
from src.models.lightret import LightRet
from src.models.ner_head import NERHead
from src.noise import apply_noise
from src.data.dataset import NERDataset, ner_collate
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Noise level configs (from paper Table 4)
# ---------------------------------------------------------------------------

NOISE_LEVELS = {
    "clean":  dict(p_sub=0.00, p_ins=0.00, p_del=0.00, p_space_ins=0.00),
    "low":    dict(p_sub=0.05, p_ins=0.02, p_del=0.02, p_space_ins=0.01),
    "medium": dict(p_sub=0.10, p_ins=0.05, p_del=0.05, p_space_ins=0.02),
    "high":   dict(p_sub=0.20, p_ins=0.10, p_del=0.10, p_space_ins=0.05),
}

# Each operator at medium intensity, applied in isolation (from paper analysis §7.1)
NOISE_OPERATORS = {
    "substitution":    dict(p_sub=0.10, p_ins=0.00, p_del=0.00, p_space_ins=0.00),
    "insertion":       dict(p_sub=0.00, p_ins=0.05, p_del=0.00, p_space_ins=0.00),
    "deletion":        dict(p_sub=0.00, p_ins=0.00, p_del=0.05, p_space_ins=0.00),
    "space insertion": dict(p_sub=0.00, p_ins=0.00, p_del=0.00, p_space_ins=0.02),
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(backbone_path: str, head_path: str, device: torch.device):
    backbone = LightRet.for_stage3()
    backbone.load_state_dict(
        torch.load(backbone_path, map_location="cpu", weights_only=True)
    )
    backbone.to(device).eval()

    head = NERHead()
    head.load_state_dict(
        torch.load(head_path, map_location="cpu", weights_only=True)
    )
    head.to(device).eval()

    return backbone, head


# ---------------------------------------------------------------------------
# Single-sentence inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    backbone: nn.Module,
    head: nn.Module,
    words: list[str],
    device: torch.device,
) -> list[str]:
    """Return predicted BIO label strings for a list of words."""
    hidden = backbone([words])                   # (1, L, 256)
    logits = head(hidden, [len(words)])          # (1, L, C)
    ids    = logits[0, :len(words)].argmax(-1).cpu().tolist()
    return [NER_ID2LABEL[i] for i in ids]


# ---------------------------------------------------------------------------
# Corrupt a CoNLL sentence at a given noise level
# ---------------------------------------------------------------------------

def corrupt_sentence(words: list[str], noise_cfg: dict) -> list[str]:
    """Apply noise to each word independently; return noisy word list."""
    if all(v == 0.0 for v in noise_cfg.values()):
        return words
    noisy = []
    for w in words:
        noisy_w, _, _ = apply_noise(w, **noise_cfg)
        noisy.append(noisy_w if noisy_w.strip() else w)   # never empty
    return noisy


# ---------------------------------------------------------------------------
# Entity-level F1 via seqeval
# ---------------------------------------------------------------------------

def compute_f1(all_true: list[list[str]], all_pred: list[list[str]]) -> dict:
    """
    Returns dict with overall F1 and per-type F1 (PER, ORG, LOC, MISC).
    Falls back to token-accuracy if seqeval is not installed.
    """
    try:
        from seqeval.metrics import classification_report, f1_score
        overall = f1_score(all_true, all_pred) * 100
        report  = classification_report(all_true, all_pred, output_dict=True)
        per_type = {
            etype: report.get(etype, {}).get("f1-score", 0.0) * 100
            for etype in ("PER", "ORG", "LOC", "MISC")
        }
        return {"overall": overall, **per_type}
    except ImportError:
        print("  [warn] seqeval not installed — reporting token accuracy instead.")
        correct = total = 0
        for true_seq, pred_seq in zip(all_true, all_pred):
            for t, p in zip(true_seq, pred_seq):
                if t != "O":
                    total   += 1
                    correct += int(t == p)
        acc = 100.0 * correct / max(1, total)
        return {"overall": acc, "PER": 0.0, "ORG": 0.0, "LOC": 0.0, "MISC": 0.0}


# ---------------------------------------------------------------------------
# Full evaluation at one noise level
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_noise_level(
    backbone: nn.Module,
    head: nn.Module,
    test_dataset: NERDataset,
    noise_cfg: dict,
    device: torch.device,
    num_seeds: int = 5,
) -> dict:
    """
    Evaluate model at a given noise level, averaged over `num_seeds` seeds.
    Returns metric dict from compute_f1.
    """
    seed_f1s: list[float] = []
    seed_metrics: list[dict] = []

    for seed in range(num_seeds):
        import numpy as np
        rng = np.random.default_rng(seed)
        all_true: list[list[str]] = []
        all_pred: list[list[str]] = []

        for item in test_dataset:
            clean_words = item["clean_words"]
            true_labels = item["clean_labels"]     # list[int]

            # Apply noise
            if all(v == 0.0 for v in noise_cfg.values()):
                eval_words = clean_words
                true_seqlabels = [NER_ID2LABEL[l] for l in true_labels
                                  if l != NER_IGNORE_INDEX]
            else:
                eval_words = []
                for w in clean_words:
                    noisy_w, _, _ = apply_noise(w, rng=rng, **noise_cfg)
                    eval_words.append(noisy_w.strip() or w)
                true_seqlabels = [NER_ID2LABEL[l] for l in true_labels
                                  if l != NER_IGNORE_INDEX]

            if not eval_words:
                continue

            pred_labels = predict(backbone, head, eval_words, device)

            # Align to valid (non-padding) positions
            valid_count = len([l for l in true_labels if l != NER_IGNORE_INDEX])
            pred_labels = pred_labels[:valid_count]

            all_true.append(true_seqlabels)
            all_pred.append(pred_labels)

        metrics = compute_f1(all_true, all_pred)
        seed_f1s.append(metrics["overall"])
        seed_metrics.append(metrics)

    # Average across seeds
    avg = {}
    for k in seed_metrics[0]:
        avg[k] = sum(m[k] for m in seed_metrics) / len(seed_metrics)
    avg["std"] = float(torch.tensor(seed_f1s).std().item())
    return avg


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed(
    backbone: nn.Module,
    head: nn.Module,
    device: torch.device,
    n_runs: int = 200,
    sentence: list[str] = None,
) -> float:
    """Returns mean inference time in milliseconds (single sentence, batch=1)."""
    if sentence is None:
        sentence = "The European Union agreed to lend Bosnia money .".split()

    # Warmup
    for _ in range(10):
        predict(backbone, head, sentence, device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        predict(backbone, head, sentence, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / n_runs * 1000.0   # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_noise_operators(
    backbone: nn.Module,
    head: nn.Module,
    test_dataset: NERDataset,
    device: torch.device,
    num_seeds: int = 5,
) -> dict[str, float]:
    """
    Evaluate each noise operator in isolation at medium intensity.
    Returns dict mapping operator name → mean F1.
    Fills paper analysis Table 8 (noise type breakdown) for LightRet.
    """
    results = {}
    for op_name, cfg in NOISE_OPERATORS.items():
        m = evaluate_noise_level(backbone, head, test_dataset, cfg, device, num_seeds)
        results[op_name] = m["overall"]
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=str(STAGE3_CKPT))
    parser.add_argument("--head",     default=str(NER_HEAD_CKPT))
    parser.add_argument("--seeds",    type=int, default=5)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Loading backbone : {args.backbone}")
    print(f"Loading NER head : {args.head}")

    backbone, head = load_model(args.backbone, args.head, DEVICE)

    print("\nLoading CoNLL-2003 test split ...")
    test_ds = NERDataset(split="test", apply_noise_aug=False)
    print(f"  {len(test_ds):,} sentences")

    # ------------------------------------------------------------------ #
    # 1. Noise-level evaluation  (fills Table 5 clean F1 + Table 6 noisy F1)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 66)
    print("Table 5 & 6 — LightRet F1 at each noise level")
    print(f"{'Level':<10} {'F1':>7}  {'±std':>6}  {'PER':>7}  {'ORG':>7}  "
          f"{'LOC':>7}  {'MISC':>7}")
    print("=" * 66)

    results = {}
    for level, cfg in NOISE_LEVELS.items():
        print(f"  evaluating [{level}] (seeds={args.seeds}) ...", flush=True)
        m = evaluate_noise_level(backbone, head, test_ds, cfg, DEVICE, args.seeds)
        results[level] = m
        print(
            f"{level:<10} {m['overall']:>7.2f}  {m['std']:>6.2f}  "
            f"{m['PER']:>7.2f}  {m['ORG']:>7.2f}  "
            f"{m['LOC']:>7.2f}  {m['MISC']:>7.2f}"
        )

    # ------------------------------------------------------------------ #
    # 2. Per-entity-type summary  (fills Table 9 LightRet rows)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 66)
    print("Table 9 — Per-entity-type F1 (clean vs medium)")
    print(f"{'Setting':<12} {'PER':>7}  {'ORG':>7}  {'LOC':>7}  {'MISC':>7}")
    print("=" * 66)
    for level in ("clean", "medium"):
        m = results[level]
        print(
            f"{level:<12} {m['PER']:>7.2f}  {m['ORG']:>7.2f}  "
            f"{m['LOC']:>7.2f}  {m['MISC']:>7.2f}"
        )

    # ------------------------------------------------------------------ #
    # 3. Noise-operator analysis  (fills analysis Table 8 LightRet row)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 66)
    print("Analysis Table 8 — LightRet F1 per noise operator (medium intensity)")
    print(f"{'Operator':<18} {'F1':>7}")
    print("=" * 66)
    op_results = evaluate_noise_operators(backbone, head, test_ds, DEVICE, args.seeds)
    for op_name, f1 in op_results.items():
        print(f"{op_name:<18} {f1:>7.2f}")

    # ------------------------------------------------------------------ #
    # 4. Inference speed  (fills Table 8 LightRet speed row)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 66)
    print("Table 8 — Inference speed  (single sentence, batch=1)")
    print("=" * 66)

    gpu_ms = benchmark_speed(backbone, head, DEVICE)
    print(f"  {DEVICE.type.upper():5}  :  {gpu_ms:.2f} ms")

    if DEVICE.type == "cuda":
        backbone_cpu = backbone.to("cpu")
        head_cpu     = head.to("cpu")
        cpu_ms = benchmark_speed(backbone_cpu, head_cpu, torch.device("cpu"))
        print(f"  CPU    :  {cpu_ms:.2f} ms")

    # ------------------------------------------------------------------ #
    # 5. Abstract numbers reminder
    # ------------------------------------------------------------------ #
    clean_f1  = results["clean"]["overall"]
    medium_f1 = results["medium"]["overall"]
    bert_f1   = 92.14   # known from literature
    print("\n" + "=" * 66)
    print("Abstract fill-in (copy these into abstract.tex):")
    print(f"  Clean F1              : {clean_f1:.1f}")
    print(f"  10% noise (medium) F1 : {medium_f1:.1f}")
    print(f"  Gap vs BERT-base      : +{medium_f1 - bert_f1:.1f} F1")
    print("=" * 66)
    print("\nDone.")


if __name__ == "__main__":
    main()
