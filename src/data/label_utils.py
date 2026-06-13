"""
data/label_utils.py — BIO label projection for noise-shifted word boundaries.

The noise pipeline can delete inter-word spaces (merging two words) or insert
spaces inside a word (splitting it). Both break the 1:1 clean↔noisy word
alignment. This module projects clean BIO labels to noisy word positions using
the alignment groups produced by noise.build_word_alignment.

Three cases:
  1:1   clean_idxs=[k],   noisy_idxs=[j]    → direct copy
  merge clean_idxs=[k,k+1,...], noisy_idxs=[j] → first clean label
  split clean_idxs=[k],   noisy_idxs=[j,j+1,...] → first gets base label,
                                                     rest get continuation label
"""

from __future__ import annotations

from src.config import NER_LABELS, NER_LABEL2ID, NER_IGNORE_INDEX

# Build B→I continuation map from NER_LABELS (e.g. B-PER→I-PER)
_B_TO_I: dict[int, int] = {}
for _lbl in NER_LABELS:
    if _lbl.startswith("B-"):
        _i_lbl = "I-" + _lbl[2:]
        if _i_lbl in NER_LABEL2ID:
            _B_TO_I[NER_LABEL2ID[_lbl]] = NER_LABEL2ID[_i_lbl]


def continuation_label(label_id: int) -> int:
    """
    Return the inside (I-) label for a given label id.
    B-X  → I-X
    I-X  → I-X  (already inside, unchanged)
    O    → O    (unchanged)
    """
    return _B_TO_I.get(label_id, label_id)


def project_labels(
    clean_labels: list[int],
    alignment: list[tuple[list[int], list[int]]],
) -> list[int]:
    """
    Project clean BIO label IDs to noisy word positions.

    Args:
        clean_labels : BIO label id per clean word  (len = number of clean words)
        alignment    : output of noise.build_word_alignment —
                       list of (clean_word_indices, noisy_word_indices) groups.
                       All noisy indices cover [0 .. total_noisy_words - 1] exactly.

    Returns:
        noisy_labels : BIO label id per noisy word
                       (len = total number of noisy words after noise application)

    Label assignment rules:
        merge (multiple clean → one noisy):
            The merged noisy word takes the label of the FIRST clean word.
            Rationale: entity beginnings (B-X) must be preserved at the merge point.
        split (one clean → multiple noisy):
            First noisy word  : same label as the clean word.
            Remaining noisy words: continuation_label(base) — B-X→I-X, O→O, I-X→I-X.
            Rationale: the split keeps the entity span intact with valid BIO sequencing.
    """
    noisy_len = sum(len(n) for _, n in alignment)
    noisy_labels = [NER_IGNORE_INDEX] * noisy_len

    for clean_idxs, noisy_idxs in alignment:
        # Base label = first clean word in the group
        base_label = clean_labels[clean_idxs[0]]

        for k, ni in enumerate(noisy_idxs):
            noisy_labels[ni] = base_label if k == 0 else continuation_label(base_label)

    return noisy_labels
