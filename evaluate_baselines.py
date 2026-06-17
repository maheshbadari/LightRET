"""
evaluate_baselines.py — Evaluate HuggingFace baseline NER models on CoNLL-2003.

Baselines:
  1. BERT-base NER      : dslim/bert-base-NER
  2. DistilBERT NER     : elastic/distilbert-base-uncased-finetuned-conll03-english

Produces:
  - F1 at 4 noise levels (clean / low / medium / high)  → Table 6 baseline rows
  - Per-entity-type F1 (PER / ORG / LOC / MISC)        → Table 9 baseline rows
  - Per-noise-operator F1 (substitution / insertion /
    deletion / space insertion)                         → analysis Table 8 baselines
  - Inference speed in ms/sentence                      → Table 8 speed column

Usage:
    python evaluate_baselines.py
    python evaluate_baselines.py --seeds 3
    python evaluate_baselines.py --models bert distilbert

Requirements:
    pip install transformers seqeval
"""

import argparse
import time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from src.noise import apply_noise
from src.data.dataset import NERDataset
from src.config import NER_IGNORE_INDEX, NER_ID2LABEL

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

BASELINE_MODELS = {
    "bert": {
        "hf_id":  "dslim/bert-base-NER",
        "label":  "BERT-base NER",
        "param_count_M": 110,
    },
    "distilbert": {
        "hf_id":  "elastic/distilbert-base-uncased-finetuned-conll03-english",
        "label":  "DistilBERT NER",
        "param_count_M": 66,
    },
}

# ---------------------------------------------------------------------------
# Noise configurations  (mirror evaluate.py)
# ---------------------------------------------------------------------------

NOISE_LEVELS = {
    "clean":  dict(p_sub=0.00, p_ins=0.00, p_del=0.00, p_space_ins=0.00),
    "low":    dict(p_sub=0.05, p_ins=0.02, p_del=0.02, p_space_ins=0.01),
    "medium": dict(p_sub=0.10, p_ins=0.05, p_del=0.05, p_space_ins=0.02),
    "high":   dict(p_sub=0.20, p_ins=0.10, p_del=0.10, p_space_ins=0.05),
}

NOISE_OPERATORS = {
    "substitution":    dict(p_sub=0.10, p_ins=0.00, p_del=0.00, p_space_ins=0.00),
    "insertion":       dict(p_sub=0.00, p_ins=0.05, p_del=0.00, p_space_ins=0.00),
    "deletion":        dict(p_sub=0.00, p_ins=0.00, p_del=0.05, p_space_ins=0.00),
    "space insertion": dict(p_sub=0.00, p_ins=0.00, p_del=0.00, p_space_ins=0.02),
}

# CoNLL-2003 entity type labels (as seqeval entity strings)
ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# HuggingFace pipeline wrapper
# ---------------------------------------------------------------------------

class BaselineNERModel:
    """
    Wraps a HuggingFace NER pipeline so it exposes the same predict(words) API
    as the LightRet evaluate.py predict() function.

    The HF pipeline tokenises internally; we use word_ids() to align subword
    predictions back to the original word list.
    """

    def __init__(self, hf_id: str, device: torch.device):
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

        print(f"  Loading {hf_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model     = AutoModelForTokenClassification.from_pretrained(hf_id)
        self.model.to(device).eval()
        self.device    = device

        self._hf_id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, words: list[str]) -> list[str]:
        """
        Args:
            words: list of word strings (the sentence)
        Returns:
            list of BIO label strings, one per word.
        """
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits   # (1, T, C)
        pred_ids = logits[0].argmax(-1).cpu().tolist()
        word_ids = enc["input_ids"].cpu()   # unused; use word_ids from encoding

        # Re-encode on CPU for word_ids
        enc_cpu = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )
        token_word_ids = enc_cpu.word_ids()

        # Pick the label of the first subtoken for each word
        word_labels: dict[int, str] = {}
        for tok_i, word_idx in enumerate(token_word_ids):
            if word_idx is None:
                continue
            if word_idx not in word_labels:
                raw = self._hf_id2label.get(pred_ids[tok_i], "O")
                word_labels[word_idx] = _normalise_label(raw)

        return [word_labels.get(i, "O") for i in range(len(words))]

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self


def _normalise_label(raw: str) -> str:
    """
    HF models sometimes use different label formats.
    Map common variants to the CoNLL BIO format we use.
    """
    raw = raw.strip().upper()
    # Some models return "B_PER", "I_PER", or just "PER"
    raw = raw.replace("_", "-")
    if raw in ("O", ""):
        return "O"
    # Already correct: B-PER, I-ORG, etc.
    if raw.startswith("B-") or raw.startswith("I-"):
        return raw
    # Some models omit the B/I prefix — treat as B
    for etype in ENTITY_TYPES:
        if raw == etype:
            return f"B-{etype}"
    return "O"


# ---------------------------------------------------------------------------
# Noise application
# ---------------------------------------------------------------------------

def corrupt_words(words: list[str], noise_cfg: dict, rng: np.random.Generator) -> list[str]:
    if all(v == 0.0 for v in noise_cfg.values()):
        return words
    result = []
    for w in words:
        noisy_w, _, _ = apply_noise(w, rng=rng, **noise_cfg)
        result.append(noisy_w.strip() or w)
    return result


# ---------------------------------------------------------------------------
# F1 computation
# ---------------------------------------------------------------------------

def compute_f1(all_true: list[list[str]], all_pred: list[list[str]]) -> dict:
    try:
        from seqeval.metrics import f1_score, classification_report
        overall = f1_score(all_true, all_pred) * 100
        report  = classification_report(all_true, all_pred, output_dict=True)
        per_type = {
            et: report.get(et, {}).get("f1-score", 0.0) * 100
            for et in ENTITY_TYPES
        }
        return {"overall": overall, **per_type}
    except ImportError:
        # Token-level accuracy fallback
        correct = total = 0
        for ts, ps in zip(all_true, all_pred):
            for t, p in zip(ts, ps):
                if t != "O":
                    total   += 1
                    correct += int(t == p)
        acc = 100.0 * correct / max(1, total)
        return {"overall": acc, **{et: 0.0 for et in ENTITY_TYPES}}


# ---------------------------------------------------------------------------
# Evaluate one baseline at one noise configuration
# ---------------------------------------------------------------------------

def evaluate_one(
    model: BaselineNERModel,
    test_dataset: NERDataset,
    noise_cfg: dict,
    num_seeds: int,
) -> dict:
    """
    Returns metric dict: overall F1, per-type F1, std across seeds.
    """
    seed_f1s: list[float] = []
    seed_metrics: list[dict] = []

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)
        all_true: list[list[str]] = []
        all_pred: list[list[str]] = []

        for item in test_dataset:
            clean_words = item["clean_words"]
            true_ids    = item["clean_labels"]

            true_labels = [NER_ID2LABEL[l] for l in true_ids if l != NER_IGNORE_INDEX]
            eval_words  = corrupt_words(clean_words, noise_cfg, rng)

            if not eval_words:
                continue

            try:
                pred_labels = model.predict(eval_words)[:len(true_labels)]
            except Exception:
                pred_labels = ["O"] * len(true_labels)

            all_true.append(true_labels)
            all_pred.append(pred_labels)

        m = compute_f1(all_true, all_pred)
        seed_f1s.append(m["overall"])
        seed_metrics.append(m)

    avg: dict = {}
    for k in seed_metrics[0]:
        avg[k] = sum(sm[k] for sm in seed_metrics) / len(seed_metrics)
    avg["std"] = float(np.std(seed_f1s))
    return avg


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------

def benchmark_speed(
    model: BaselineNERModel,
    n_runs: int = 100,
    sentence: Optional[list[str]] = None,
) -> float:
    """Returns mean latency in milliseconds (single sentence, batch=1)."""
    if sentence is None:
        sentence = "The European Union agreed to lend Bosnia money .".split()

    for _ in range(5):
        model.predict(sentence)

    if model.device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_runs):
        model.predict(sentence)
    if model.device.type == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - start) / n_runs * 1000.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=list(BASELINE_MODELS.keys()),
        default=list(BASELINE_MODELS.keys()),
        help="Which baseline models to evaluate",
    )
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Seeds  : {args.seeds}")

    print("\nLoading CoNLL-2003 test split ...")
    test_ds = NERDataset(split="test", apply_noise_aug=False)
    print(f"  {len(test_ds):,} sentences")

    for model_key in args.models:
        cfg   = BASELINE_MODELS[model_key]
        label = cfg["label"]
        print(f"\n{'=' * 70}")
        print(f"{label}  ({cfg['param_count_M']}M params)")
        print(f"{'=' * 70}")

        model = BaselineNERModel(cfg["hf_id"], DEVICE)

        # ---------------------------------------------------------------- #
        # Table 6 — F1 at each noise level
        # ---------------------------------------------------------------- #
        print("\nTable 6 — Noise-level F1:")
        print(f"  {'Level':<10} {'F1':>7}  {'±std':>6}  {'PER':>7}  {'ORG':>7}  "
              f"{'LOC':>7}  {'MISC':>7}")
        print(f"  {'-' * 62}")

        level_results = {}
        for level, ncfg in NOISE_LEVELS.items():
            print(f"    evaluating [{level}] (seeds={args.seeds}) ...", flush=True)
            m = evaluate_one(model, test_ds, ncfg, args.seeds)
            level_results[level] = m
            print(
                f"  {level:<10} {m['overall']:>7.2f}  {m['std']:>6.2f}  "
                f"{m['PER']:>7.2f}  {m['ORG']:>7.2f}  "
                f"{m['LOC']:>7.2f}  {m['MISC']:>7.2f}"
            )

        # ---------------------------------------------------------------- #
        # Table 9 — Per-entity-type F1 (clean vs medium)
        # ---------------------------------------------------------------- #
        print("\nTable 9 — Per-entity-type F1 (clean vs medium):")
        print(f"  {'Setting':<12} {'PER':>7}  {'ORG':>7}  {'LOC':>7}  {'MISC':>7}")
        print(f"  {'-' * 46}")
        for lvl in ("clean", "medium"):
            m = level_results[lvl]
            print(
                f"  {lvl:<12} {m['PER']:>7.2f}  {m['ORG']:>7.2f}  "
                f"{m['LOC']:>7.2f}  {m['MISC']:>7.2f}"
            )

        # ---------------------------------------------------------------- #
        # Analysis Table 8 — Per-noise-operator F1
        # ---------------------------------------------------------------- #
        print("\nAnalysis Table 8 — Per-noise-operator F1 (medium intensity):")
        print(f"  {'Operator':<18} {'F1':>7}")
        print(f"  {'-' * 28}")
        for op_name, opcfg in NOISE_OPERATORS.items():
            print(f"    evaluating [{op_name}] ...", flush=True)
            m = evaluate_one(model, test_ds, opcfg, args.seeds)
            print(f"  {op_name:<18} {m['overall']:>7.2f}")

        # ---------------------------------------------------------------- #
        # Table 8 — Inference speed
        # ---------------------------------------------------------------- #
        print("\nTable 8 — Inference speed:")
        gpu_ms = benchmark_speed(model)
        print(f"  {DEVICE.type.upper():5}  :  {gpu_ms:.2f} ms/sentence")

        if DEVICE.type == "cuda":
            model.to(torch.device("cpu"))
            cpu_ms = benchmark_speed(model)
            print(f"  CPU    :  {cpu_ms:.2f} ms/sentence")
            model.to(DEVICE)  # restore

        # Free GPU memory before next model
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Done. Copy the numbers above into experiments.tex Tables 6/9")
    print("and analysis.tex Table 8 (baseline rows).")
    print("=" * 70)


if __name__ == "__main__":
    main()
