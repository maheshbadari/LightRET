"""
data/dataset.py — Dataset classes and collation for all three training stages.

PretrainDataset  (Stage 1 & 2)
    Combines WikiText-103-raw + CoNLL-2003 train into a flat list of
    word-tokenized sentences. Returns list[str] per item.

NERDataset       (Stage 3)
    Loads CoNLL-2003. On each __getitem__, applies stochastic noise to
    the clean sentence, projects BIO labels to noisy coordinates, and
    returns both representations plus the word-level alignment.

Collation:
    pretrain_collate — identity (models accept list[list[str]])
    ner_collate      — pads label tensors, bundles into a dict
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# CoNLL-2003 loader  (script-free, works with datasets >= 3.0)
# ---------------------------------------------------------------------------

def _load_conll2003(split: str):
    """
    Load CoNLL-2003 by reading parquet files directly from HuggingFace Hub.

    datasets >= 3.0 refuses to run dataset scripts (conll2003.py), even for
    third-party mirrors. Loading via the built-in 'parquet' loader bypasses
    the script-detection path entirely.

    Falls back through three strategies so this works in any environment.
    """
    from datasets import load_dataset

    # Strategy 1: direct parquet via hf:// URI (works on datasets >= 2.x, no script)
    base = "hf://datasets/conll2003/data"
    try:
        return load_dataset(
            "parquet",
            data_files={split: f"{base}/{split}-00000-of-00001.parquet"},
            split=split,
        )
    except Exception:
        pass

    # Strategy 2: dynamically discover parquet files in the repo
    try:
        from huggingface_hub import list_repo_files
        files = [
            f"hf://datasets/conll2003/{f}"
            for f in list_repo_files("conll2003", repo_type="dataset")
            if f.endswith(".parquet") and f"/{split}-" in f
        ]
        if files:
            return load_dataset("parquet", data_files={split: files}, split=split)
    except Exception:
        pass

    # Strategy 3: last resort — named dataset (works on older datasets versions)
    return load_dataset("conll2003", split=split, trust_remote_code=True)

import torch
from torch.utils.data import Dataset

from src.config import (
    NER_IGNORE_INDEX,
    STAGE1_MAX_WORDS,
    STAGE2_MAX_WORDS,
    STAGE3_MAX_WORDS,
    STAGE1_WIKITEXT_DATASET,
    STAGE1_WIKITEXT_CONFIG,
    STAGE1_CONLL_DATASET,
    STAGE3_CONLL_DATASET,
    NOISE_P_SUB,
    NOISE_P_INS,
    NOISE_P_DEL,
    NOISE_P_SPACE_INS,
)
from src.noise import apply_noise, build_word_alignment
from src.data.label_utils import project_labels

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sentences_from_wikitext(text: str, max_words: int) -> list[list[str]]:
    """
    Extract word-tokenized sentences from a WikiText-103 raw text chunk.

    Filtering:
    - Skip empty lines
    - Skip section headings (lines starting with '=')
    - Split each line into sentences at [.!?] boundaries
    - Keep sentences with >= 4 words; truncate to max_words
    """
    sentences: list[list[str]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("="):
            continue
        for part in _SENT_SPLIT_RE.split(line):
            words = part.split()
            if len(words) >= 4:
                sentences.append(words[:max_words])
    return sentences


def _pad_label_batch(label_lists: list[list[int]]) -> torch.Tensor:
    """
    Pad a list of label-id lists with NER_IGNORE_INDEX.
    Returns (B, L_max) LongTensor.
    """
    max_len = max(len(lbl) for lbl in label_lists)
    padded = torch.full((len(label_lists), max_len), NER_IGNORE_INDEX, dtype=torch.long)
    for i, lbl in enumerate(label_lists):
        padded[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)
    return padded


# ---------------------------------------------------------------------------
# PretrainDataset
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """
    Word-tokenized sentences for Stage 1 (BERT→RetBERT) and Stage 2
    (RetBERT→LightRet) distillation.

    Sources:
        WikiText-103-raw-v1  — broad vocabulary coverage
        CoNLL-2003 train     — domain overlap with Stage 3 NER data

    Args:
        split     : HuggingFace split name ('train' or 'validation')
        max_words : truncation length (default STAGE1_MAX_WORDS = 64)
        verbose   : print loading progress
    """

    def __init__(
        self,
        split: str = "train",
        max_words: int = STAGE1_MAX_WORDS,
        verbose: bool = True,
    ) -> None:
        from datasets import load_dataset  # imported here to keep top-level import light

        self.sentences: list[list[str]] = []

        # ---- WikiText-103 ----
        if verbose:
            print("Loading WikiText-103 ...")
        wiki = load_dataset(
            STAGE1_WIKITEXT_DATASET,
            STAGE1_WIKITEXT_CONFIG,
            split=split,
        )
        for item in wiki:
            self.sentences.extend(_sentences_from_wikitext(item["text"], max_words))

        if verbose:
            print(f"  WikiText sentences : {len(self.sentences):,}")

        # ---- CoNLL-2003 ----
        if verbose:
            print("Loading CoNLL-2003 ...")
        conll_split = "train" if split == "train" else "validation"
        conll = _load_conll2003(conll_split)
        n_before = len(self.sentences)
        for item in conll:
            words = item["tokens"]
            if len(words) >= 2:
                self.sentences.append(words[:max_words])

        if verbose:
            print(f"  CoNLL sentences    : {len(self.sentences) - n_before:,}")
            print(f"  Total              : {len(self.sentences):,}")

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> list[str]:
        return self.sentences[idx]


# ---------------------------------------------------------------------------
# NERDataset
# ---------------------------------------------------------------------------

class NERDataset(Dataset):
    """
    CoNLL-2003 NER dataset with on-the-fly noise augmentation for Stage 3.

    Each __getitem__ call applies fresh stochastic noise to the clean sentence,
    producing different noise patterns across epochs automatically.

    Returns a dict with:
        clean_words  : list[str]                              clean word tokens
        clean_labels : list[int]                              BIO label IDs (clean)
        noisy_words  : list[str]                              word tokens after noise
        noisy_labels : list[int]                              BIO label IDs (noisy coords)
        alignment    : list[tuple[list[int], list[int]]]      (C_k, N_k) groups

    Args:
        split      : 'train' | 'validation' | 'test'
        max_words  : truncate clean sentence to this many words
        noise_prob : if False, returns clean==noisy (for inference / evaluation)
    """

    def __init__(
        self,
        split: str = "train",
        max_words: int = STAGE3_MAX_WORDS,
        apply_noise_aug: bool = True,
    ) -> None:
        from datasets import load_dataset

        self.max_words       = max_words
        self.apply_noise_aug = apply_noise_aug

        dataset = _load_conll2003(split)

        # ner_tags may be int (ClassLabel) or str depending on the dataset source
        from src.config import NER_LABEL2ID
        def _to_int(tag) -> int:
            return tag if isinstance(tag, int) else NER_LABEL2ID[tag]

        self._data: list[tuple[list[str], list[int]]] = []
        for item in dataset:
            words  = item["tokens"][: max_words]
            labels = [_to_int(t) for t in item["ner_tags"][: max_words]]
            if len(words) >= 1:
                self._data.append((words, labels))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clean_words, clean_labels = self._data[idx]

        if not self.apply_noise_aug:
            # Evaluation mode: clean == noisy, trivial 1:1 alignment
            alignment = [([i], [i]) for i in range(len(clean_words))]
            return {
                "clean_words" : clean_words,
                "clean_labels": list(clean_labels),
                "noisy_words" : clean_words,
                "noisy_labels": list(clean_labels),
                "alignment"   : alignment,
            }

        # Apply character-level noise to the full sentence string
        clean_str = " ".join(clean_words)
        noisy_str, shift_log, _ = apply_noise(
            clean_str,
            p_sub=NOISE_P_SUB,
            p_ins=NOISE_P_INS,
            p_del=NOISE_P_DEL,
            p_space_ins=NOISE_P_SPACE_INS,
        )
        noisy_words = noisy_str.split()

        # Word-level alignment and label projection
        alignment    = build_word_alignment(clean_str, shift_log)
        noisy_labels = project_labels(clean_labels, alignment)

        # Safety: if alignment produced more/fewer noisy indices than actual words,
        # fall back to clean (avoids rare edge cases from unexpected whitespace)
        expected_noisy_len = sum(len(n) for _, n in alignment)
        if expected_noisy_len != len(noisy_words):
            alignment    = [([i], [i]) for i in range(len(clean_words))]
            noisy_words  = clean_words
            noisy_labels = list(clean_labels)

        return {
            "clean_words" : clean_words,
            "clean_labels": list(clean_labels),
            "noisy_words" : noisy_words,
            "noisy_labels": noisy_labels,
            "alignment"   : alignment,
        }


# ---------------------------------------------------------------------------
# Collation functions
# ---------------------------------------------------------------------------

def pretrain_collate(batch: list[list[str]]) -> list[list[str]]:
    """
    Collation for PretrainDataset.
    Models accept list[list[str]] directly — no tensor padding needed.
    """
    return batch


def ner_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collation for NERDataset.

    Pads label tensors with NER_IGNORE_INDEX; passes word lists through as-is.

    Returns dict with keys:
        clean_words  : list[list[str]]
        clean_labels : (B, L_clean_max) LongTensor
        noisy_words  : list[list[str]]
        noisy_labels : (B, L_noisy_max) LongTensor
        alignment    : list[list[tuple[list[int], list[int]]]]
        clean_lengths: list[int]
        noisy_lengths: list[int]
    """
    clean_words   = [item["clean_words"]  for item in batch]
    noisy_words   = [item["noisy_words"]  for item in batch]
    alignment     = [item["alignment"]    for item in batch]
    clean_lengths = [len(w) for w in clean_words]
    noisy_lengths = [len(w) for w in noisy_words]

    clean_labels = _pad_label_batch([item["clean_labels"] for item in batch])
    noisy_labels = _pad_label_batch([item["noisy_labels"] for item in batch])

    return {
        "clean_words"  : clean_words,
        "clean_labels" : clean_labels,
        "noisy_words"  : noisy_words,
        "noisy_labels" : noisy_labels,
        "alignment"    : alignment,
        "clean_lengths": clean_lengths,
        "noisy_lengths": noisy_lengths,
    }
