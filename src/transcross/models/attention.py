"""Bias-free custom multi-head attention and Transformer blocks.

All self-attention Q/K/V/O projections use Xavier uniform init.
Cross-attention Q/K/V projections use Xavier uniform, but output
projection uses near-zero Normal(0, 1e-4) init.
Cross-attention blocks include a learnable residual gate alpha=sigmoid(g),
with g initialized to -4.0 (alpha ≈ 0.018).

NO additive attention bias is allowed (no coordinate bias, no modality bias,
no relative position bias, no Graphormer-style bias).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear_xavier(linear: nn.Linear) -> None:
    """Xavier uniform init for self-attention projections."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_linear_near_zero(linear: nn.Linear) -> None:
    """Near-zero init for cross-attention output projection.

    Weight ~ Normal(0, 1e-4), bias = 0.
    This ensures cross-attention contributes minimally at initialization,
    allowing the model to first learn intra-modal representations.
    """
    nn.init.normal_(linear.weight, std=1e-4)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class MultiHeadAttention(nn.Module):
    """Custom multi-head scaled dot-product attention.

    No additive attention bias is allowed. Only key_padding_mask
    and an optional causal (attn_mask) mask are supported.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        zero_init_out_proj: bool = False,
        near_zero_out_proj: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters(zero_init_out_proj, near_zero_out_proj)

    def _reset_parameters(
        self, zero_init_out_proj: bool, near_zero_out_proj: bool
    ) -> None:
        _init_linear_xavier(self.q_proj)
        _init_linear_xavier(self.k_proj)
        _init_linear_xavier(self.v_proj)
        if near_zero_out_proj:
            _init_linear_near_zero(self.out_proj)
        elif zero_init_out_proj:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
        else:
            _init_linear_xavier(self.out_proj)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ):
        """Compute multi-head attention.

        Args:
            query: (B, L_q, d_model)
            key: (B, L_k, d_model)
            value: (B, L_k, d_model)
            key_padding_mask: (B, L_k) bool, True for positions to ignore
            attn_mask: (L_q, L_k) or (B*nhead, L_q, L_k) causal/attn mask
            need_weights: if True, return attention weights

        Returns:
            output: (B, L_q, d_model)
            attn_weights: optional (B, num_heads, L_q, L_k)
        """
        B = query.shape[0]
        L_q, L_k = query.shape[1], key.shape[1]

        # Project and reshape to multi-head
        q = self.q_proj(query).view(B, L_q, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(B, L_k, self.num_heads, self.d_head).transpose(1, 2)
        # q, k, v: (B, num_heads, L, d_head)

        # Scaled dot-product attention (no additive bias)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply key padding mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, L_k), True = ignore
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L_k)
            attn_weights = attn_weights.masked_fill(kpm, float("-inf"))

        # Apply attention mask (causal, etc.)
        if attn_mask is not None:
            # attn_mask: broadcastable to (..., L_q, L_k)
            attn_weights = attn_weights + attn_mask

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # (B, num_heads, L_q, d_head)
        out = out.transpose(1, 2).contiguous().view(B, L_q, self.d_model)

        out = self.out_proj(out)

        if need_weights:
            return out, attn_probs
        return out


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                _init_linear_xavier(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlockPreLN(nn.Module):
    """Standard Pre-LN Transformer block: self-attention + FFN.

    Self-attention uses Xavier uniform init for all projections.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        zero_init_out_proj: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, dropout, zero_init_out_proj=zero_init_out_proj
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with Pre-LN
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x),
                               key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # FFN with Pre-LN
        x = x + self.ffn(self.norm2(x))
        return x


class CrossAttentionBlockPreLN(nn.Module):
    """Pre-LN cross-attention block with learnable residual gate.

    Query tokens attend to key/value tokens from another source.
    Includes a learnable gate: alpha = sigmoid(g), g initialized to -4.0
    so that alpha ≈ 0.018 at initialization.

    Output projection uses near-zero Normal(0, 1e-4) init so that
    cross-attention contributes minimally at the start of training,
    allowing the model to first build useful intra-modal representations.

    Forward:
        attn_out = CrossAttn(LN_q(query), LN_kv(kv))
        query = query + alpha * attn_out
        query = query + FFN(LN(query))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        gate_init: float = -4.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads, dropout, near_zero_out_proj=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Learnable residual gate: alpha = sigmoid(g)
        # gate_init = -4.0 -> alpha ≈ 0.018 (near-closed, v1 default)
        # gate_init = -2.0 -> alpha ≈ 0.119 (partially open, v2)
        # gate_init =  0.0 -> alpha = 0.5   (fully open)
        self.gate_logit = nn.Parameter(torch.tensor(gate_init))

    def get_alpha(self) -> torch.Tensor:
        """Return current gate value alpha = sigmoid(g)."""
        return torch.sigmoid(self.gate_logit)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        alpha = torch.sigmoid(self.gate_logit)
        # Cross-attention with Pre-LN
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        query = query + alpha * self.cross_attn(
            q_norm, kv_norm, kv_norm, key_padding_mask=kv_padding_mask
        )
        # FFN with Pre-LN
        query = query + self.ffn(self.norm2(query))
        return query
