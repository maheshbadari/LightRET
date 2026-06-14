# src/losses.py

Loss functions for all three training stages.

---

## stage1_loss

```python
stage1_loss(
    z_bert: Tensor,      # (B, 768) — BERT sentence embedding (teacher)
    z_retbert: Tensor,   # (B, 768) — RetBERT sentence embedding (student)
) -> Tensor              # scalar
```

**Formula:**
```
L₁ = 1 − mean_over_batch( cos(z_bert[i], z_retbert[i]) )
   = 1 − (1/B) Σᵢ (z_bert[i] · z_retbert[i]) / (‖z_bert[i]‖ · ‖z_retbert[i]‖)
```

Range: `[0, 2]`. Perfect alignment → 0. Opposite directions → 2.

**Why cosine and not MSE?**
Cosine distance is scale-invariant. BERT's sentence vectors and RetBERT's
projections may have different magnitudes early in training; cosine forces
directional alignment without penalizing scale differences, which improves
convergence stability.

---

## stage2_loss

```python
stage2_loss(
    h_teacher: Tensor,   # (B, L, 768) — RetBERT token hidden states
    h_student: Tensor,   # (B, L, 768) — LightRet projected token states
    lengths: list[int],  # actual word count per sentence (for masking)
) -> Tensor              # scalar
```

**Formula:**
```
L₂ = (1/N_valid) Σᵢ Σⱼ mask[i,j] · (1 − cos(h_teacher[i,j], h_student[i,j]))
```

where `mask[i,j] = 1` if `j < lengths[i]` (valid position), else `0`.

**Sequence-length masking:** RetVec pads shorter sentences to the batch maximum.
Padding positions produce zero vectors that would artificially push the cosine
distance toward zero. The length mask excludes these positions from the loss average.

---

## stage3_loss

```python
stage3_loss(
    logits: Tensor,          # (B, L_noisy, 9) — NER head output for noisy input
    noisy_labels: Tensor,    # (B, L_noisy)    — projected BIO labels (padded with -100)
    h_teacher: Tensor,       # (B, L_clean, 256) — teacher hidden states (clean)
    h_student: Tensor,       # (B, L_noisy, 256) — student hidden states (noisy)
    alignment: list,         # per-sentence list of (C_k, N_k) group tuples
    beta: float = 0.5,       # weight of classification loss
) -> tuple[Tensor, Tensor, Tensor]  # (total, L_class, L_distill)
```

**Classification loss:**
```
L_class = CrossEntropy(logits, noisy_labels)
        = −(1/m) Σⱼ log softmax(logits[j])[ỹⱼ]
```
Padding positions (`noisy_labels == -100`) are automatically excluded by
PyTorch's `CrossEntropyLoss(ignore_index=-100)`.

**Alignment-aware distillation loss:**
```
For each group k in alignment:
    h_T_(k) = mean( h_teacher[i] for i in C_k )   ← teacher, clean
    h_S_(k) = mean( h_student[j] for j in N_k )   ← student, noisy

L_distill = (1/K) Σ_k (1 − cos(h_T_(k), h_S_(k)))
```

Group-level pooling handles the case where noise changes word count:
- **Merge** (2 clean → 1 noisy): teacher pools 2 vectors; student uses 1 vector
- **Split** (1 clean → 2 noisy): teacher uses 1 vector; student pools 2 vectors
- **1:1**: direct cosine distance

**Combined loss:**
```
L₃ = β · L_class + (1−β) · L_distill     β = STAGE3_BETA = 0.5
```

**Returns:** `(total_loss, L_class, L_distill)` — individual components are
logged separately for monitoring training progress.

---

## Design notes

**Why return all three loss components from stage3_loss?**
Training logs print `total`, `lc`, and `ld` separately so you can diagnose
if one component dominates. If `ld` stays near 2.0 (max), the student is
not aligning with the teacher. If `lc` stays near `log(9)` ≈ 2.2 (random
baseline), the classifier is not learning entity labels.

**Why β=0.5?**
Empirically balanced in ablation studies. Setting β=1 (classification only)
loses noise robustness because the student gets no signal to match the clean
teacher's representations. Setting β=0 (distillation only) degrades entity
detection because the student is never rewarded for correct BIO predictions.
