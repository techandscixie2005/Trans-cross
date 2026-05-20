"""1D patch tokenization for IR and NMR spectra."""

import math
import torch
import torch.nn as nn


class PatchTokenizer1D(nn.Module):
    """Project 1D spectral patches to d_model tokens.

    Pads the input to a multiple of patch_size, then applies a linear
    projection to each non-overlapping patch.
    """

    def __init__(self, input_len: int, patch_size: int, d_model: int):
        super().__init__()
        self.input_len = input_len
        self.patch_size = patch_size
        self.d_model = d_model

        # Number of patches after padding
        self.num_patches = math.ceil(input_len / patch_size)

        # Linear projection per patch
        self.proj = nn.Linear(patch_size, d_model)

        # Positional embeddings (learnable, per modality)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_len) — 1D spectral vector

        Returns:
            (B, num_patches, d_model) — patch tokens
        """
        B = x.shape[0]
        L = x.shape[1]

        # Pad to multiple of patch_size
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x = torch.nn.functional.pad(x, (0, pad_len))
        else:
            pad_len = 0

        # Reshape to patches: (B, num_patches, patch_size)
        x = x.view(B, self.num_patches, self.patch_size)

        # Project: (B, num_patches, d_model)
        tokens = self.proj(x)

        # Add positional embedding
        tokens = tokens + self.pos_embed

        return tokens


class ModalityEmbedding(nn.Module):
    """Learnable modality type embedding added to all tokens of a modality."""

    def __init__(self, d_model: int):
        super().__init__()
        self.embed = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.embed
