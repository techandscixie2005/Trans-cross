"""MLP head for Morgan fingerprint prediction."""

import torch
import torch.nn as nn


class FingerprintHead(nn.Module):
    """Predicts Morgan fingerprint logits from a pooled representation."""

    def __init__(self, d_model: int, n_bits: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_bits),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: (B, d_model) — pooled multimodal representation

        Returns:
            (B, n_bits) — fingerprint logits (before sigmoid)
        """
        pooled = self.norm(pooled)
        return self.mlp(pooled)
