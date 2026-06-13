"""
losses.py — Loss functions for all three LightRet training stages.

Stage 1  : sentence-level cosine distillation (BERT -> RetBERT)
Stage 2  : token-level cosine distillation    (RetBERT -> LightRet)
Stage 3  : compound loss = beta*L_class + (1-beta)*L_distill
             L_class   : cross-entropy NER labels on noisy tokens
             L_distill : token-level cosine with dynamic shift alignment
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config import NER_IGNORE_INDEX, STAGE3_BETA


# ---------------------------------------------------------------------------
# Stage 1 — sentence cosine loss
# ---------------------------------------------------------------------------

def stage1_loss(
    z_bert: torch.Tensor,
    z_retbert: torch.Tensor,
) -> torch.Tensor:
    """
    Sentence-level cosine distillation loss.

    Args:
        z_bert    : (B, 768)  — MeanPool of frozen BERT hidden states
        z_retbert : (B, 768)  — MeanPool of RetBERT hidden states

    Returns:
        scalar — mean over batch of (1 - cosine_similarity)
    """
    return (1.0 - F.cosine_similarity(z_retbert, z_bert, dim=-1)).mean()


# ---------------------------------------------------------------------------
# Stage 2 — token cosine loss
# ---------------------------------------------------------------------------

def stage2_loss(
    h_retbert: torch.Tensor,
    h_light_proj: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    """
    Token-level cosine distillation loss.

    Teacher and student share the same word tokenization, so positions align
    directly — no shift alignment needed here.

    Args:
        h_retbert     : (B, L_max, 768) — RetBERT hidden states (frozen teacher)
        h_light_proj  : (B, L_max, 768) — LightRet projected hidden states
        lengths       : valid token count per sentence

    Returns:
        scalar — mean cosine distance over all valid token positions
    """
    B, L_max, _ = h_retbert.shape

    mask = torch.zeros(B, L_max, dtype=torch.bool, device=h_retbert.device)
    for i, L in enumerate(lengths):
        mask[i, :L] = True

    cos_sim = F.cosine_similarity(h_light_proj, h_retbert, dim=-1)   # (B, L_max)
    return (1.0 - cos_sim).masked_select(mask).mean()


# ---------------------------------------------------------------------------
# Stage 3 — compound loss with dynamic shift alignment
# ---------------------------------------------------------------------------

def stage3_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    h_teacher: torch.Tensor,
    h_student: torch.Tensor,
    alignment: list[list[tuple[list[int], list[int]]]],
    beta: float = STAGE3_BETA,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compound NER loss with dynamic shift-aligned distillation.

    Args:
        logits     : (B, L_noisy, C)   — NER logits from student on noisy tokens
        labels     : (B, L_noisy)      — BIO labels in noisy coordinates
                                          (NER_IGNORE_INDEX=-100 for pad/invalid)
        h_teacher  : (B, L_clean, 256) — LightRet hidden states on clean text (frozen)
        h_student  : (B, L_noisy, 256) — LightRet hidden states on noisy text
        alignment  : per-sentence list of (clean_indices, noisy_indices) groups
                     from noise.build_word_alignment.
                     Each group represents one logical word in clean space.
                     - 1:1 groups: direct comparison
                     - space-deletion (merged) groups: MeanPool teacher tokens
                     - space-insertion (split) groups: MeanPool student tokens
        beta       : weight for L_class

    Returns:
        (total_loss, l_class, l_distill)
    """
    B, L_noisy, C = logits.shape

    # Classification loss
    l_class = F.cross_entropy(
        logits.reshape(-1, C),
        labels.reshape(-1),
        ignore_index=NER_IGNORE_INDEX,
    )

    # Distillation loss with dynamic alignment
    distill_terms: list[torch.Tensor] = []

    for b in range(B):
        for clean_idxs, noisy_idxs in alignment[b]:
            t_vec = h_teacher[b, clean_idxs].mean(dim=0, keepdim=True)  # (1, 256)
            s_vec = h_student[b, noisy_idxs].mean(dim=0, keepdim=True)  # (1, 256)
            distill_terms.append(1.0 - F.cosine_similarity(s_vec, t_vec, dim=-1))

    if distill_terms:
        l_distill = torch.stack(distill_terms).mean()
    else:
        l_distill = torch.zeros(1, device=logits.device).squeeze()

    total = beta * l_class + (1.0 - beta) * l_distill
    return total, l_class, l_distill
