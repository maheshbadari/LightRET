"""
noise.py — Character-level stochastic noise pipeline for LightRet Stage 3.

Three public functions:
  apply_noise          — corrupt a string, return (noisy_str, shift_log, S)
  map_span             — project clean char span [I_start, I_end] to noisy coords
  build_word_alignment — derive word-level (C_k, N_k) alignment groups from space mutations
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Visual similarity table  (OCR / typographic confusables)
# ---------------------------------------------------------------------------

VISUAL_SIMILARITY: dict[str, list[str]] = {
    # lowercase
    'a': ['@', '4'],
    'b': ['6'],
    'c': ['('],
    'e': ['3'],
    'g': ['9', 'q'],
    'i': ['1', 'l', '|'],
    'l': ['1', 'i', '|'],
    'o': ['0'],
    'q': ['9', 'g'],
    's': ['5', '$'],
    't': ['+'],
    'u': ['v'],
    'v': ['u'],
    'z': ['2'],
    # uppercase
    'A': ['4', '@'],
    'B': ['8'],
    'E': ['3'],
    'G': ['6'],
    'I': ['1', 'l', '|'],
    'O': ['0'],
    'S': ['5', '$'],
    'Z': ['2'],
    # digits
    '0': ['o', 'O'],
    '1': ['l', 'i', 'I'],
    '2': ['z', 'Z'],
    '3': ['e', 'E'],
    '5': ['s', 'S'],
    '6': ['b', 'G'],
    '8': ['B'],
    '9': ['g', 'q'],
}


# ---------------------------------------------------------------------------
# Shift log entry
# ---------------------------------------------------------------------------

@dataclass
class ShiftEntry:
    index: int   # original char position (0-indexed)
    op:    str   # 'INS' | 'DEL' | 'SUB'
    count: int   # chars added / removed / replaced  (always 1 in current pipeline)
    char:  str   # INS → inserted char
                 # DEL → deleted char  (needed for space-boundary detection)
                 # SUB → replacement char


# ---------------------------------------------------------------------------
# apply_noise
# ---------------------------------------------------------------------------

def apply_noise(
    text:        str,
    p_sub:       float = 0.10,
    p_ins:       float = 0.05,
    p_del:       float = 0.05,
    p_space_ins: float = 0.02,
    rng:         Optional[np.random.Generator] = None,
) -> tuple[str, list[ShiftEntry], np.ndarray]:
    """
    Corrupt `text` with stochastic character-level noise.

    Two independent noise processes per character:

    1. Main operation (mutually exclusive, one per character):
         substitution  p_sub        — replace with visual homoglyph
         insertion     p_ins        — keep char, append a similar char after it
         deletion      p_del        — remove char
         no change     1 - sum      — pass through unchanged

    2. Space insertion (independent of main op, non-space chars only):
         p_space_ins                — append a space after the character,
                                      splitting the word at that position.
                                      Skipped if the character was deleted or
                                      is itself a space.
         Simulates OCR word-splitting: "London" -> "Lon don"

    Both processes contribute to delta[i] independently, so a character can
    accumulate up to +2 offset (char insertion AND space insertion at same pos).

    Returns
    -------
    noisy_str  : corrupted string
    shift_log  : ShiftEntry list in original-index order
    S          : int32 ndarray shape (len(text),)
                 S[i] = cumulative net offset (insertions − deletions) over positions 0..i
    """
    assert p_sub + p_ins + p_del <= 1.0

    if rng is None:
        rng = np.random.default_rng()

    noisy_chars: list[str] = []
    shift_log:   list[ShiftEntry] = []
    delta = np.zeros(len(text), dtype=np.int32)

    t_sub = p_sub
    t_ins = p_sub + p_ins
    t_del = p_sub + p_ins + p_del

    for i, c in enumerate(text):
        r = float(rng.random())
        deleted = False

        if r < t_sub:                               # --- substitution
            new_c = _sample_similar(c, rng)
            noisy_chars.append(new_c)
            if new_c != c:
                shift_log.append(ShiftEntry(i, 'SUB', 1, new_c))

        elif r < t_ins:                             # --- char insertion
            ins_c = _sample_insert(c, rng)
            noisy_chars.append(c)
            noisy_chars.append(ins_c)
            shift_log.append(ShiftEntry(i, 'INS', 1, ins_c))
            delta[i] += 1

        elif r < t_del:                             # --- deletion
            shift_log.append(ShiftEntry(i, 'DEL', 1, c))
            delta[i] -= 1
            deleted = True                          # do not append c

        else:                                       # --- no change
            noisy_chars.append(c)

        # --- space insertion (independent of main op)
        # Only for non-space chars that survived deletion.
        # Appends a space after the (possibly substituted/inserted) char,
        # splitting the surrounding word at this position.
        if not deleted and c != ' ' and p_space_ins > 0:
            if float(rng.random()) < p_space_ins:
                noisy_chars.append(' ')
                shift_log.append(ShiftEntry(i, 'INS', 1, ' '))
                delta[i] += 1

    S = np.cumsum(delta).astype(np.int32)
    return ''.join(noisy_chars), shift_log, S


# ---------------------------------------------------------------------------
# map_span  — NER label coordinate projection
# ---------------------------------------------------------------------------

def map_span(
    I_start: int,
    I_end:   int,
    S:       np.ndarray,
) -> Optional[tuple[int, int]]:
    """
    Project a clean-text inclusive char span [I_start, I_end] to noisy-text coords.

    Spec §2C formulae:
      I'_start = I_start + S[I_start − 1]    (S[−1] ≡ 0 by convention)
      I'_end   = I_end   + S[I_end]

    Returns None when the span is invalidated by deletion (I'_end ≤ I'_start).
    """
    s_before     = int(S[I_start - 1]) if I_start > 0 else 0
    I_start_noisy = I_start + s_before
    I_end_noisy   = I_end   + int(S[I_end])

    if I_end_noisy <= I_start_noisy:
        return None  # span collapsed — invalidate per spec §2C
    return (I_start_noisy, I_end_noisy)


# ---------------------------------------------------------------------------
# build_word_alignment  — word-level (C_k, N_k) group builder
# ---------------------------------------------------------------------------

def build_word_alignment(
    clean_str:  str,
    shift_log:  list[ShiftEntry],
) -> list[tuple[list[int], list[int]]]:
    """
    Derive word-level alignment groups G = [(C_k, N_k)] from shift_log.

    Space deletion  → adjacent clean words merge → one noisy word
                       teacher repr  = MeanPool(H^Teacher[C_k])
    Space insertion → one clean word splits      → multiple noisy words
                       student repr = MeanPool(H^Student[N_k])
    No space change → direct 1-to-1

    Returns
    -------
    groups : list of (clean_word_indices, noisy_word_indices)
             Each index list is contiguous and sorted.
    """
    clean_words = clean_str.split()
    if not clean_words:
        return []

    # ------------------------------------------------------------------
    # Build position maps over the clean string
    # ------------------------------------------------------------------
    char_to_word:  dict[int, int]             = {}  # non-space char pos → word idx
    space_to_gap:  dict[int, tuple[int, int]] = {}  # space char pos → (left_w, right_w)

    pos = 0
    for w_idx, word in enumerate(clean_words):
        for _ in range(len(word)):
            char_to_word[pos] = w_idx
            pos += 1
        if w_idx < len(clean_words) - 1:
            space_to_gap[pos] = (w_idx, w_idx + 1)
            pos += 1                                 # advance past space

    # ------------------------------------------------------------------
    # Classify space-affecting mutations
    # ------------------------------------------------------------------
    deleted_boundaries: set[int]  = set()   # left-word idx of each deleted inter-word space
    word_extra_splits:  dict[int, int] = {} # word_idx → count of spaces injected inside it

    for entry in shift_log:
        if entry.op == 'DEL' and entry.char == ' ':
            gap = space_to_gap.get(entry.index)
            if gap is not None:
                deleted_boundaries.add(gap[0])              # left word of deleted gap

        elif entry.op == 'INS' and entry.char == ' ':
            w_idx = char_to_word.get(entry.index)
            if w_idx is not None:                           # insertion inside a word
                word_extra_splits[w_idx] = word_extra_splits.get(w_idx, 0) + entry.count

    # ------------------------------------------------------------------
    # Walk clean words and emit groups
    # ------------------------------------------------------------------
    groups:        list[tuple[list[int], list[int]]] = []
    noisy_cursor = 0
    i = 0

    while i < len(clean_words):
        clean_group = [i]

        # Greedily absorb following words whose left boundary was deleted
        while clean_group[-1] in deleted_boundaries:
            nxt = clean_group[-1] + 1
            if nxt >= len(clean_words):
                break
            clean_group.append(nxt)

        if len(clean_group) > 1:
            # Merged group → exactly one noisy word
            # Internal splits within the merged range are ignored
            noisy_count = 1
        else:
            # Single clean word → may split via inserted spaces
            noisy_count = 1 + word_extra_splits.get(clean_group[0], 0)

        noisy_group = list(range(noisy_cursor, noisy_cursor + noisy_count))
        groups.append((clean_group, noisy_group))

        noisy_cursor += noisy_count
        i = clean_group[-1] + 1

    return groups


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sample_similar(c: str, rng: np.random.Generator) -> str:
    """Return a visual homoglyph of c, or c itself if none defined."""
    candidates = VISUAL_SIMILARITY.get(c, [])
    if not candidates:
        return c
    return str(rng.choice(candidates))


def _sample_insert(c: str, rng: np.random.Generator) -> str:
    """
    Character to insert adjacent to c.
    70 %  → visual homoglyph of c (if available)
    30 %  → random lowercase letter
    """
    candidates = VISUAL_SIMILARITY.get(c, []) or VISUAL_SIMILARITY.get(c.lower(), [])
    if candidates and float(rng.random()) < 0.7:
        return str(rng.choice(candidates))
    return chr(int(rng.integers(ord('a'), ord('z') + 1)))
