"""
models/retvec_embedder.py — Frozen PyTorch port of pretrained RetVec-v1.

Architecture (verified by exhaustive forward-pass comparison, diff < 2e-6):
    word string
        -> binarize: Unicode codepoints -> 24-bit binary per char -> (16, 24) float32
        -> Flatten:   (16*24,) = (384,)
        -> Linear(384, 256) + GELU   [encoder1, dense_3 weights]
        -> Linear(256, 256) + GELU   [encoder2, dense_4 weights]
        -> Linear(256, 256) + Tanh   [tokenizer, dense_5 weights]
    output: 256-dim embedding, values in [-1, 1]

Usage:
    embedder = RetVecEmbedder("retvec_v1_weights.npz")
    emb = embedder(batch_words)   # list[list[str]] -> (B, L_max, 256)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetVecEmbedder(nn.Module):
    """
    Frozen RetVec-v1 word embedder.

    Converts a batch of tokenized sentences (list[list[str]]) into dense
    256-dim embeddings shaped (B, L_max, 256), zero-padded for shorter sentences.

    All parameters are frozen (requires_grad=False).
    The binarization step has no parameters — it is a deterministic bit-encoding
    of Unicode codepoints matching TF's RETVecBinarizer exactly.
    """

    WORD_LENGTH   = 16    # max chars per word (truncate/pad)
    ENCODING_SIZE = 24    # bits per character (Unicode codepoint)
    HIDDEN_DIM    = 256

    def __init__(self, weights_path: str) -> None:
        super().__init__()

        # Bit-mask buffer: masks[k] = 2^(23-k)
        # Used as: bit_k = min(codepoint & masks[k], 1)
        enc   = self.ENCODING_SIZE
        masks = torch.tensor(
            [2 ** (enc - 1 - i) for i in range(enc)], dtype=torch.int64
        )
        self.register_buffer("_bitmasks", masks)  # (24,)

        flat_dim = self.WORD_LENGTH * self.ENCODING_SIZE  # 384

        self.encoder1  = nn.Linear(flat_dim,        self.HIDDEN_DIM)  # 384 -> 256
        self.encoder2  = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)  # 256 -> 256
        self.tokenizer = nn.Linear(self.HIDDEN_DIM, self.HIDDEN_DIM)  # 256 -> 256

        self._load_pretrained(weights_path)

        for param in self.parameters():
            param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, weights_path: str) -> None:
        """
        Load weights exported by retvec_export.py.

        TF Dense kernel: (in_features, out_features)
        PyTorch Linear weight: (out_features, in_features)  -> transpose on load
        """
        data = np.load(weights_path)

        def _load(layer: nn.Linear, prefix: str) -> None:
            layer.weight.data = torch.from_numpy(data[f"{prefix}_kernel"].T.copy())
            layer.bias.data   = torch.from_numpy(data[f"{prefix}_bias"].copy())

        _load(self.encoder1,  "encoder1")
        _load(self.encoder2,  "encoder2")
        _load(self.tokenizer, "tokenizer")

    # ------------------------------------------------------------------
    # Binarizer (no parameters)
    # ------------------------------------------------------------------

    def _binarize_flat(self, flat_words: list[str]) -> torch.Tensor:
        """
        Convert N word strings to binarized float32 tensor of shape (N, 384).

        Matches TF RETVecBinarizer (word_length=16, encoding_size=24, UTF-8):
            - Each character -> Unicode codepoint via ord()
            - Extract 24 bits via descending-power bitmasks, clip to {0, 1}
            - Pad/truncate to WORD_LENGTH characters
        """
        W = self.WORD_LENGTH
        E = self.ENCODING_SIZE
        N = len(flat_words)

        masks_np = self._bitmasks.cpu().numpy()  # (24,)

        codepoints = np.zeros((N, W), dtype=np.int64)
        for i, word in enumerate(flat_words):
            chars = [ord(c) for c in word[:W]]
            codepoints[i, :len(chars)] = chars

        # (N, W, 1) & (1, 1, E) -> (N, W, E), then clip to {0, 1}
        bits = np.minimum(
            np.bitwise_and(
                codepoints[:, :, np.newaxis],
                masks_np[np.newaxis, np.newaxis, :]
            ),
            1,
        ).astype(np.float32)                      # (N, W, E)

        return torch.from_numpy(bits.reshape(N, W * E)).to(self._bitmasks.device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch_words: list[list[str]]) -> torch.Tensor:
        """
        Embed a batch of tokenized sentences.

        Args:
            batch_words: B sentences, each a list of word strings.

        Returns:
            (B, L_max, 256) float32, zero-padded for shorter sentences.
        """
        lengths = [len(s) for s in batch_words]
        L_max   = max(lengths) if lengths else 0

        if L_max == 0:
            return torch.zeros(
                len(batch_words), 0, self.HIDDEN_DIM,
                device=self._bitmasks.device
            )

        flat_words = [w for sent in batch_words for w in sent]

        # Binarize -> (total_words, 384)
        x = self._binarize_flat(flat_words)

        # MLP forward (matches TF: gelu -> gelu -> tanh)
        x = F.gelu(self.encoder1(x))      # (total_words, 256)
        x = F.gelu(self.encoder2(x))      # (total_words, 256)
        x = torch.tanh(self.tokenizer(x)) # (total_words, 256)

        # Scatter into (B, L_max, 256) with zero padding
        B   = len(batch_words)
        out = torch.zeros(B, L_max, self.HIDDEN_DIM, dtype=x.dtype, device=x.device)
        offset = 0
        for i, length in enumerate(lengths):
            if length > 0:
                out[i, :length] = x[offset : offset + length]
            offset += length

        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def embed_sentence(self, words: list[str]) -> torch.Tensor:
        """Embed a single sentence. Returns (L, 256)."""
        return self.forward([words])[0]
