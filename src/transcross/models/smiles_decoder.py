"""Autoregressive SMILES decoder with cross-attention to encoder memory."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, FeedForward


def _init_embedding(embed: nn.Embedding, std: float = 0.02) -> None:
    nn.init.normal_(embed.weight, std=std)


class SmilesDecoderLayer(nn.Module):
    """One decoder layer: masked self-attn, cross-attn, FFN (Pre-LN)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.norm_self1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm_cross_q = nn.LayerNorm(d_model)
        self.norm_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_memory: torch.Tensor,
        causal_mask: torch.Tensor,
        self_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Masked self-attention
        x = x + self.self_attn(
            self.norm_self1(x), self.norm_self1(x), self.norm_self1(x),
            key_padding_mask=self_padding_mask,
            attn_mask=causal_mask,
        )
        # Cross-attention to encoder memory
        x = x + self.cross_attn(
            self.norm_cross_q(x),
            self.norm_cross_kv(encoder_memory),
            self.norm_cross_kv(encoder_memory),
            key_padding_mask=memory_padding_mask,
        )
        # FFN
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TransformerSmilesDecoder(nn.Module):
    """Autoregressive transformer decoder for SMILES generation.

    Uses causal self-attention over SMILES tokens and cross-attention
    to the encoder memory. No additive bias except the causal mask
    (required for autoregressive decoding).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_len = max_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList([
            SmilesDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        _init_embedding(self.token_embed)
        _init_embedding(self.pos_embed)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Build causal mask: upper triangular with -inf, lower with 0."""
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) — SMILES token IDs with BOS prefix
            encoder_memory: (B, L_enc, d_model) — encoder output
            memory_padding_mask: (B, L_enc) — True for pad positions

        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token + positional embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Build causal mask for decoder self-attention
        causal_mask = self._build_causal_mask(T, device)

        # Self-padding mask for decoder input
        self_padding_mask = (input_ids == self.pad_id)

        for layer in self.layers:
            x = layer(x, encoder_memory, causal_mask,
                      self_padding_mask=self_padding_mask,
                      memory_padding_mask=memory_padding_mask)

        x = self.out_norm(x)
        logits = self.output_proj(x)
        return logits
