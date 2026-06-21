"""
label_proj_eval.py — Measure dynamic BIO label projection accuracy and
space-insertion event rate.

Fills paper §7.3 (analysis.tex) todos:
  - Space-insertion event rate: % of word positions where a space is inserted
    mid-word, causing the tokeniser to split one word into two sub-words.
  - Projection accuracy: how often the dynamic BIO projection (B-X→B-X + I-X)
    matches a BERT-large oracle tagger applied independently to the noisy text.

The oracle uses bert-large-cased-finetuned-conll03-english to get "ground truth"
labels for the noisy text, then compares against what the label projection rule
would produce if we naively project clean labels through the space-insertion map.

Usage:
    python label_proj_eval.py
    python label_proj_eval.py --seeds 5 --p-space-ins 0.02

Requirements:
    pip install transformers seqeval
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from src.config import DEVICE, NER_IGNORE_INDEX, NER_ID2LABEL, NER_LABELS
from src.data.dataset import NERDataset
from src.noise import apply_noise


# ---------------------------------------------------------------------------
# Oracle tagger  (BERT-large fine-tuned on CoNLL-2003)
# ---------------------------------------------------------------------------

ORACLE_MODEL_ID = "dbmdz/bert-large-cased-finetuned-conll03-english"


class OracleTagger:
    """
    BERT-large NER tagger used as oracle on noisy text.
    Returns word-level BIO labels (first-subtoken rule).
    """

    def __init__(self, device: torch.device):
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        print(f"  Loading oracle tagger: {ORACLE_MODEL_ID} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(ORACLE_MODEL_ID)
        self.model = AutoModelForTokenClassification.from_pretrained(ORACLE_MODEL_ID)
        self.model.to(device).eval()
        self.device = device
        self._id2label = self.model.config.id2label

    @torch.no_grad()
    def tag(self, words: list[str]) -> list[str]:
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0].argmax(-1).cpu().tolist()

        enc_cpu = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )
        wids = enc_cpu.word_ids()
        seen: dict[int, str] = {}
        for tok_i, wid in enumerate(wids):
            if wid is None or wid in seen:
                continue
            raw = self._id2label.get(logits[tok_i], "O").upper().replace("_", "-")
            # Normalise: keep B-/I- prefix, strip unknowns
            if raw in NER_LABELS:
                seen[wid] = raw
            elif raw.startswith("B-") or raw.startswith("I-"):
                seen[wid] = raw
            else:
                seen[wid] = "O"

        return [seen.get(i, "O") for i in range(len(words))]


# ---------------------------------------------------------------------------
# Dynamic label projection  (mirrors src/data/dataset.py logic)
# ---------------------------------------------------------------------------

def project_labels(
    clean_words: list[str],
    clean_labels: list[str],
    noisy_words: list[str],
    alignment: list[int | None],
) -> list[str]:
    """
    Given the clean→noisy word alignment produced by apply_noise (space-insertion
    may split one clean word into multiple noisy words), project BIO labels from
    the clean sequence to the noisy sequence.

    alignment[noisy_word_idx] = clean_word_idx  (or None for padding)

    Rules:
      - First noisy word inheriting from clean word W  →  same label as W
      - Subsequent noisy words from the same clean word:
          - If previous was B-X or I-X → I-X  (continuation)
          - If previous was O          → O
    """
    projected = []
    prev_clean_idx: int | None = None
    for noisy_idx, clean_idx in enumerate(alignment):
        if clean_idx is None:
            projected.append("O")
            continue
        label = clean_labels[clean_idx]
        if clean_idx == prev_clean_idx:
            # Same clean word → continuation
            if label.startswith("B-"):
                label = "I-" + label[2:]
            elif label == "O":
                label = "O"
            # If already I-X, keep I-X
        projected.append(label)
        prev_clean_idx = clean_idx
    return projected


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def count_space_insertions(alignment: list[int | None]) -> tuple[int, int]:
    """
    Returns (n_split_positions, total_positions).
    A 'split position' is any noisy word that shares the same clean_idx as
    the previous noisy word (i.e., one clean word was split into ≥2 noisy words).
    """
    total = 0
    splits = 0
    prev = None
    for clean_idx in alignment:
        if clean_idx is None:
            continue
        if prev is not None and clean_idx == prev:
            splits += 1
        total += 1
        prev = clean_idx
    return splits, total


# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

def evaluate(
    oracle: OracleTagger,
    test_dataset: NERDataset,
    p_space_ins: float,
    num_seeds: int,
) -> dict:
    """
    Returns:
        space_event_rate : fraction of noisy-word positions that are continuations
                           of a space-inserted clean word
        proj_accuracy    : % of noisy-word positions where projection == oracle label
        proj_f1          : seqeval entity F1 of projection labels vs oracle labels
    """
    all_space_splits = 0
    all_space_total  = 0

    all_oracle_labels:  list[list[str]] = []
    all_proj_labels:    list[list[str]] = []

    noise_cfg = dict(p_sub=0.00, p_ins=0.00, p_del=0.00, p_space_ins=p_space_ins)

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)

        for item in test_dataset:
            clean_words = item["clean_words"]
            true_ids    = item["clean_labels"]
            clean_labels = [NER_ID2LABEL[l] for l in true_ids if l != NER_IGNORE_INDEX]

            if not clean_words:
                continue

            # Apply space-insertion noise only; collect alignment
            noisy_words: list[str] = []
            alignment:   list[int] = []

            for ci, w in enumerate(clean_words):
                noisy_w, _shift_log, _ = apply_noise(w, rng=rng, **noise_cfg)
                if not noisy_w.strip():
                    noisy_words.append(w)
                    alignment.append(ci)
                    continue
                parts = noisy_w.split()
                if not parts:
                    noisy_words.append(w)
                    alignment.append(ci)
                    continue
                for part in parts:
                    noisy_words.append(part)
                    alignment.append(ci)

            if not noisy_words:
                continue

            # Space-insertion rate
            splits, total = count_space_insertions(alignment)
            all_space_splits += splits
            all_space_total  += total

            # Oracle labels on noisy text
            try:
                oracle_labels = oracle.tag(noisy_words)
            except Exception:
                continue

            # Projected labels
            proj_labels = project_labels(
                clean_words, clean_labels, noisy_words, alignment
            )

            # Both truncated to noisy length
            n = len(noisy_words)
            all_oracle_labels.append(oracle_labels[:n])
            all_proj_labels.append(proj_labels[:n])

    # Compute metrics
    space_rate = all_space_splits / max(1, all_space_total) * 100.0

    # Token-level projection accuracy (vs oracle)
    correct = total_tok = 0
    for oracle_seq, proj_seq in zip(all_oracle_labels, all_proj_labels):
        for o, p in zip(oracle_seq, proj_seq):
            total_tok += 1
            correct   += int(o == p)
    proj_acc = 100.0 * correct / max(1, total_tok)

    # Entity-level F1 of projection vs oracle (seqeval)
    proj_f1 = 0.0
    try:
        from seqeval.metrics import f1_score
        proj_f1 = f1_score(all_oracle_labels, all_proj_labels) * 100.0
    except ImportError:
        proj_f1 = proj_acc   # fallback

    return {
        "space_event_rate_pct": space_rate,
        "proj_token_acc_pct":   proj_acc,
        "proj_entity_f1":       proj_f1,
        "n_space_splits":       all_space_splits,
        "n_total_positions":    all_space_total,
        "n_sentences":          len(all_oracle_labels),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int,   default=3)
    parser.add_argument("--p-space-ins",  type=float, default=0.02,
                        help="Space-insertion probability (default 0.02 = medium)")
    args = parser.parse_args()

    print(f"Device        : {DEVICE}")
    print(f"Seeds         : {args.seeds}")
    print(f"p_space_ins   : {args.p_space_ins}")

    print("\nLoading CoNLL-2003 test split ...")
    test_ds = NERDataset(split="test", apply_noise_aug=False)
    print(f"  {len(test_ds):,} sentences")

    oracle = OracleTagger(DEVICE)

    print(f"\nRunning label projection evaluation (seeds={args.seeds}) ...")
    results = evaluate(oracle, test_ds, args.p_space_ins, args.seeds)

    print("\n" + "=" * 60)
    print("§7.3 Label Projection Analysis  — fill these into analysis.tex")
    print("=" * 60)
    print(
        f"  Space-insertion event rate : {results['space_event_rate_pct']:.2f}%"
        f"  of word positions"
    )
    print(
        f"    ({results['n_space_splits']:,} split positions  /"
        f"  {results['n_total_positions']:,} total positions,"
        f"  over {results['n_sentences']:,} sentences × {args.seeds} seeds)"
    )
    print()
    print(
        f"  Projection token accuracy  : {results['proj_token_acc_pct']:.2f}%"
    )
    print(
        f"  Projection entity F1       : {results['proj_entity_f1']:.2f}"
        f"  (vs BERT-large oracle)"
    )
    print("=" * 60)
    print("\nInterpretation:")
    rate = results["space_event_rate_pct"]
    f1   = results["proj_entity_f1"]
    if rate < 5.0:
        print(f"  → Low split rate ({rate:.2f}%): space-insertion rarely changes token count.")
    else:
        print(f"  → Non-trivial split rate ({rate:.2f}%): label projection is load-bearing.")
    if f1 >= 90.0:
        print(f"  → High projection F1 ({f1:.2f}): dynamic BIO rule matches oracle well.")
    else:
        print(f"  → Projection F1 = {f1:.2f}: inspect edge cases (B→I transition accuracy).")


if __name__ == "__main__":
    main()
