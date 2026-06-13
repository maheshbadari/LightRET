"""
models/retbert.py — RetBERT: Stage 1 student model.

Architecture:
    word-tokenized sentence
        -> RetVecEmbedder (frozen, 256-dim)  -> (B, L, 256)
        -> Linear projection (256 -> 768)    -> (B, L, 768)
        -> positional encoding               -> (B, L, 768)
        -> 12 × Transformer encoder layers   -> (B, L, 768)
        -> MeanPool (over valid tokens)      -> (B, 768)   [Stage 1 distillation target]

Stage 1 trains only the projection + transformer layers.
RetVec embedder is permanently frozen.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.config import (
    RETVEC_WEIGHTS,
    RETVEC_DIM,
    RETBERT_DIM,
    RETBERT_LAYERS,
    RETBERT_HEADS,
    RETBERT_FFN_DIM,
    RETBERT_DROPOUT,
    STAGE1_MAX_WORDS,
)
from src.models.retvec_embedder import RetVecEmbedder


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class RetBERT(nn.Module):
    """
    RetBERT — Stage 1 student.

    Accepts word-tokenized sentences as list[list[str]].
    Returns token-level hidden states (B, L_max, 768) and sentence vectors (B, 768).

    Args:
        weights_path: path to retvec_v1_weights.npz
    """

    def __init__(self, weights_path: str | None = None) -> None:
        super().__init__()

        if weights_path is None:
            weights_path = str(RETVEC_WEIGHTS)

        self.retvec = RetVecEmbedder(weights_path)  # frozen inside __init__

        self.proj = nn.Linear(RETVEC_DIM, RETBERT_DIM)  # 256 -> 768

        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model=RETBERT_DIM,
            max_len=STAGE1_MAX_WORDS + 2,
            dropout=RETBERT_DROPOUT,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=RETBERT_DIM,
            nhead=RETBERT_HEADS,
            dim_feedforward=RETBERT_FFN_DIM,
            dropout=RETBERT_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,      # pre-LN: more stable training from scratch
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=RETBERT_LAYERS,
            enable_nested_tensor=False,
        )

    # ------------------------------------------------------------------
    # Padding mask helper
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_mask(lengths: list[int], max_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns (B, max_len) bool mask where True = position to IGNORE.
        (PyTorch TransformerEncoder convention)
        """
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            if L < max_len:
                mask[i, L:] = True
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, batch_words: list[list[str]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch_words: B sentences, each a list of word strings.

        Returns:
            hidden : (B, L_max, 768)  — token-level representations
            pooled : (B, 768)         — MeanPool over valid tokens (for Stage 1 loss)
        """
        lengths = [len(s) for s in batch_words]
        L_max   = max(lengths) if lengths else 1

        # RetVec: (B, L_max, 256), frozen
        x = self.retvec(batch_words)                  # (B, L_max, 256)

        # Project to BERT dim
        x = self.proj(x)                              # (B, L_max, 768)

        # Positional encoding
        x = self.pos_enc(x)                           # (B, L_max, 768)

        # Padding mask for transformer
        device = x.device
        pad_mask = self._pad_mask(lengths, L_max, device)  # (B, L_max)

        # Transformer encoder
        hidden = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, L_max, 768)

        # MeanPool over valid tokens
        valid = (~pad_mask).unsqueeze(-1).float()     # (B, L_max, 1)
        pooled = (hidden * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)  # (B, 768)

        return hidden, pooled

    def sentence_embedding(self, batch_words: list[list[str]]) -> torch.Tensor:
        """Convenience: returns only the pooled sentence vector (B, 768)."""
        _, pooled = self.forward(batch_words)
        return pooled
