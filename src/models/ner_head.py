"""
models/ner_head.py — BiLSTM NER head for Stage 3.

Architecture:
    (B, L, 256)  LightRet token representations
        -> BiLSTM (128 per direction -> 256)
        -> Dropout
        -> Linear (256 -> num_classes)
        -> (B, L, C) logits

Only valid (non-padded) positions are produced. Padding positions
are left as-is in the output tensor — the loss uses NER_IGNORE_INDEX=-100
to mask them out.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import (
    LIGHTRET_DIM,
    NER_BILSTM_HIDDEN,
    NER_DROPOUT,
    NER_NUM_CLASSES,
)


class NERHead(nn.Module):
    """
    BiLSTM NER classification head.

    Args:
        num_classes : number of BIO label classes (default: NER_NUM_CLASSES = 9)
        input_dim   : hidden dim from LightRet backbone (default: LIGHTRET_DIM = 256)

    Input:
        hidden  : (B, L_max, input_dim) — LightRet output
        lengths : list[int]             — valid token counts per sentence

    Output:
        logits  : (B, L_max, num_classes)
    """

    def __init__(
        self,
        num_classes: int = NER_NUM_CLASSES,
        input_dim: int = LIGHTRET_DIM,
    ) -> None:
        super().__init__()

        # BiLSTM: 256 -> 128 per direction -> concat 256
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=NER_BILSTM_HIDDEN,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(p=NER_DROPOUT)
        self.classifier = nn.Linear(NER_BILSTM_HIDDEN * 2, num_classes)  # 256 -> C

    def forward(
        self, hidden: torch.Tensor, lengths: list[int]
    ) -> torch.Tensor:
        """
        Args:
            hidden  : (B, L_max, 256)
            lengths : valid token count per sentence

        Returns:
            logits  : (B, L_max, num_classes)
        """
        L_max = hidden.size(1)

        # Pack for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            hidden,
            lengths=torch.tensor(lengths, dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_packed, _ = self.bilstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_packed, batch_first=True, total_length=L_max
        )                                     # (B, L_max, 256)

        x = self.dropout(x)
        logits = self.classifier(x)           # (B, L_max, C)
        return logits
