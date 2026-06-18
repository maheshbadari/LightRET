"""
models/lightret.py — LightRet: Stage 2 student / Stage 3 noisy-student backbone.

Architecture:
    word-tokenized sentence
        -> RetVecEmbedder (frozen, 256-dim)      -> (B, L, 256)
        -> BiGRU (128 per direction → 256)       -> (B, L, 256)
        -> positional encoding                   -> (B, L, 256)
        -> 4 × Transformer encoder layers        -> (B, L, 256)
        [Stage 2 only]
        -> Linear projector (256 -> 768)         -> (B, L, 768)

Stage 2: projector is active; distillation target is (B, L, 768) matching RetBERT.
Stage 3: projector is disabled (with_projector=False); output is (B, L, 256).

RetVec is permanently frozen. BiGRU acts as a denoising filter in Stage 3 —
it maps noisy RetVec embeddings back toward the clean representation space,
which is why it must remain trainable in Stage 3.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.config import (
    RETVEC_WEIGHTS,
    RETVEC_DIM,
    LIGHTRET_DIM,
    LIGHTRET_BIGRU_HIDDEN,
    LIGHTRET_LAYERS,
    LIGHTRET_HEADS,
    LIGHTRET_FFN_DIM,
    LIGHTRET_DROPOUT,
    LIGHTRET_PROJ_DIM,
)
from src.models.retvec_embedder import RetVecEmbedder


class _SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

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


class LightRet(nn.Module):
    """
    LightRet backbone — used in Stage 2 (compression) and Stage 3 (NER tuning).

    Args:
        weights_path   : path to retvec_v1_weights.npz
        with_projector : if True, adds Linear(256→768) for Stage 2 distillation.
                         Set to False for Stage 3 (projector weights are discarded).

    Forward returns:
        Stage 2 (with_projector=True)  -> (B, L_max, 768)
        Stage 3 (with_projector=False) -> (B, L_max, 256)
    """

    def __init__(
        self,
        weights_path: str | None = None,
        with_projector: bool = False,
    ) -> None:
        super().__init__()

        if weights_path is None:
            weights_path = str(RETVEC_WEIGHTS)

        self.retvec = RetVecEmbedder(weights_path)  # frozen inside __init__

        # BiGRU: input=256, hidden=128 per direction, output=256
        self.bigru = nn.GRU(
            input_size=RETVEC_DIM,
            hidden_size=LIGHTRET_BIGRU_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,          # single layer — dropout has no effect anyway
        )

        self.pos_enc = _SinusoidalPositionalEncoding(
            d_model=LIGHTRET_DIM,
            max_len=512,
            dropout=LIGHTRET_DROPOUT,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=LIGHTRET_DIM,
            nhead=LIGHTRET_HEADS,
            dim_feedforward=LIGHTRET_FFN_DIM,
            dropout=LIGHTRET_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=LIGHTRET_LAYERS,
            enable_nested_tensor=False,
        )

        self.projector: nn.Linear | None = None
        if with_projector:
            self.projector = nn.Linear(LIGHTRET_DIM, LIGHTRET_PROJ_DIM)  # 256 -> 768

    # ------------------------------------------------------------------
    # Padding mask
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_mask(lengths: list[int], max_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
        for i, L in enumerate(lengths):
            if L < max_len:
                mask[i, L:] = True
        return mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_words: list[list[str]]) -> torch.Tensor:
        """
        Args:
            batch_words: B sentences, each a list of word strings.

        Returns:
            (B, L_max, 768) if with_projector else (B, L_max, 256)
        """
        lengths = [len(s) for s in batch_words]
        L_max   = max(lengths) if lengths else 1

        # RetVec embeddings (frozen)
        emb = self.retvec(batch_words)                # (B, L_max, 256)

        # BiGRU: pack to handle variable lengths efficiently
        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths=torch.tensor(lengths, dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        gru_packed, _ = self.bigru(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            gru_packed, batch_first=True, total_length=L_max
        )                                             # (B, L_max, 256)

        # Positional encoding
        x = self.pos_enc(x)                           # (B, L_max, 256)

        # Padding mask
        device   = x.device
        pad_mask = self._pad_mask(lengths, L_max, device)

        # Transformer encoder
        hidden = self.transformer(x, src_key_padding_mask=pad_mask)  # (B, L_max, 256)

        # Optional projection for Stage 2
        if self.projector is not None:
            return self.projector(hidden)             # (B, L_max, 768)

        return hidden                                 # (B, L_max, 256)

    # ------------------------------------------------------------------
    # Stage transition helpers
    # ------------------------------------------------------------------

    def drop_projector(self) -> None:
        """Remove the Stage 2 projector before Stage 3 fine-tuning."""
        self.projector = None

    def add_projector(self) -> None:
        """(Re-)attach a fresh projector for Stage 2 distillation."""
        self.projector = nn.Linear(LIGHTRET_DIM, LIGHTRET_PROJ_DIM)

    @classmethod
    def for_stage2(cls, weights_path: str | None = None) -> "LightRet":
        """Factory: Stage 2 model with projector active."""
        return cls(weights_path=weights_path, with_projector=True)

    @classmethod
    def for_stage3(cls, weights_path: str | None = None) -> "LightRet":
        """Factory: Stage 3 model without projector."""
        return cls(weights_path=weights_path, with_projector=False)

    @classmethod
    def from_stage2_checkpoint(cls, ckpt_path: str) -> "LightRet":
        """
        Load Stage 2 checkpoint and drop the projector for Stage 3.
        The projector weights are simply discarded — they are not needed further.
        """
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state.pop("pos_enc.pe", None)   # pe is deterministic; don't overwrite the 512-slot buffer
        model = cls(with_projector=True)
        model.load_state_dict(state, strict=False)
        model.drop_projector()
        return model
