# src/noise.py

Character-level stochastic noise pipeline used in Stage 3 noisy-student training.

---

## Overview

Three public functions:

| Function | Description |
|---|---|
| `apply_noise(text, ...)` | Corrupt a string; return `(noisy_str, shift_log, S)` |
| `map_span(shift_log, i_start, i_end)` | Project a clean character span to noisy coordinates |
| `build_word_alignment(clean_str, shift_log)` | Derive word-level `(C_k, N_k)` alignment groups |

---

## Visual Similarity Table

`VISUAL_SIMILARITY: dict[str, list[str]]`

Maps each character to visually similar substitutes used by
`apply_noise` for the substitution operator. Covers OCR-common confusables
and keyboard-adjacent characters.

```python
VISUAL_SIMILARITY = {
    'a': ['@', '4'],
    'b': ['6'],
    'c': ['('],
    'e': ['3'],
    'g': ['9', 'q'],
    'i': ['1', 'l', '|'],
    'l': ['1', 'i', '|'],
    'o': ['0'],
    's': ['5', '$'],
    't': ['+'],
    'z': ['2'],
    # uppercase mirrors + digits
    'A': ['4', '@'],
    'B': ['8'],
    'E': ['3'],
    'G': ['6'],
    'I': ['1', '|'],
    'O': ['0'],
    'S': ['5', '$'],
    'Z': ['2'],
    '0': ['O', 'o'],
    '1': ['l', 'I', '|'],
    ...
}
```

---

## ShiftEntry (dataclass)

```python
@dataclass
class ShiftEntry:
    pos:   int    # character position in the clean string where edit occurred
    delta: int    # length change: +1 (insertion), -1 (deletion), 0 (substitution)
    kind:  str    # 'sub' | 'ins' | 'del' | 'space'
```

The shift log is a `list[ShiftEntry]` that records every character-level
edit in order. It is the key data structure used to project clean positions
into noisy positions.

---

## apply_noise

```python
apply_noise(
    text:        str,
    p_sub:       float = 0.10,   # substitution probability per character
    p_ins:       float = 0.05,   # insertion probability per character
    p_del:       float = 0.05,   # deletion probability per character
    p_space_ins: float = 0.02,   # space-insertion probability per non-space character
    rng:         np.random.Generator = None,
) -> tuple[str, list[ShiftEntry], set[int]]
```

Applies four independent stochastic operations to each character of `text`:

| Operator | Probability | Effect | Delta |
|---|---|---|---|
| `sub` | `p_sub` | Replace `c` with `VISUAL_SIMILARITY[c]` (random choice) | 0 |
| `ins` | `p_ins` | Insert a random ASCII printable character after `c` | +1 |
| `del` | `p_del` | Delete `c` | −1 |
| `space` | `p_space_ins` | Insert a space after `c` (within a word only) | +1 |

Operations are applied left-to-right. Each character is evaluated independently.
A character can be affected by at most one operator (first match wins in priority
order: del > sub > ins > space).

**Returns:**

| Value | Type | Description |
|---|---|---|
| `noisy_str` | `str` | The corrupted string |
| `shift_log` | `list[ShiftEntry]` | Ordered record of all edits |
| `S` | `set[int]` | Set of clean positions where a space was inserted |

**Example:**
```python
clean = "London is great"
noisy, log, spaces = apply_noise(clean, p_sub=0.1, p_ins=0.05, p_del=0.05, p_space_ins=0.02)
# noisy might be: "L0ndon 1s greet"  (o→0, i→1, a→e)
```

---

## map_span

```python
map_span(
    shift_log: list[ShiftEntry],
    i_start:   int,
    i_end:     int,
) -> tuple[int, int]
```

Projects a clean character span `[i_start, i_end)` into the corresponding
noisy character span by replaying the shift log up to each boundary position.

**Algorithm:**
```
noisy_pos = clean_pos
for entry in shift_log:
    if entry.pos <= current_clean_pos:
        noisy_pos += entry.delta
```

Returns `(noisy_start, noisy_end)`.

Used internally by `build_word_alignment` to map each clean word's character
span into noisy coordinates.

---

## build_word_alignment

```python
build_word_alignment(
    clean_str:  str,
    shift_log:  list[ShiftEntry],
) -> list[tuple[list[int], list[int]]]
```

Derives a word-level alignment between the clean and noisy sentences.

**Returns:** `[(C_k, N_k), ...]` where:
- `C_k` = list of clean word indices in group `k`
- `N_k` = list of noisy word indices in group `k`

**Algorithm:**
1. Compute clean word boundaries (character start/end for each word).
2. Use `map_span` to project each boundary into noisy character coordinates.
3. Determine which noisy words each clean word overlaps with after projection.
4. Group overlapping clean↔noisy words into alignment groups.

**Three cases handled:**

| Case | Example | C_k | N_k |
|---|---|---|---|
| 1:1 | `"York"` → `"Y0rk"` | `[2]` | `[2]` |
| Split | `"London"` → `"Lon don"` | `[0]` | `[0, 1]` |
| Merge | `"New York"` → `"NewYork"` (del space) | `[1, 2]` | `[1]` |

**Usage in Stage 3:**
```python
clean_str = "New York is great"
noisy_str, shift_log, _ = apply_noise(clean_str)
alignment = build_word_alignment(clean_str, shift_log)
# e.g. [([0], [0]), ([1,2], [1]), ([3], [2])]
#       New→New   New+York→NewYork   is→is   great→gr3at

noisy_labels = project_labels(clean_labels, alignment)
```

---

## Integration with NERDataset

Inside `NERDataset.__getitem__`:

```python
clean_str = " ".join(clean_words)
noisy_str, shift_log, _ = apply_noise(clean_str, p_sub=..., p_ins=..., ...)
noisy_words  = noisy_str.split()
alignment    = build_word_alignment(clean_str, shift_log)
noisy_labels = project_labels(clean_labels, alignment)

# Safety check: revert to clean if alignment is inconsistent
if sum(len(n) for _, n in alignment) != len(noisy_words):
    noisy_words  = clean_words
    noisy_labels = clean_labels
    alignment    = [([i], [i]) for i in range(len(clean_words))]
```

The safety revert handles rare edge cases (e.g., multiple consecutive space
insertions or deletions within a single character) where the shift log
arithmetic produces an unexpected word count.
