"""E1: Intra-modal + cross-modal attention encoder.

Each modality is first encoded with its own intra-modal self-attention,
then cross-attention exchanges information across modalities.
"""

import torch
import torch.nn as nn

from .tokenizers import PatchTokenizer1D, ModalityEmbedding
from .fingerprint_head import FingerprintHead


class IntraModalEncoder(nn.Module):
    """Small Transformer encoder for one modality."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class CrossModalAttention(nn.Module):
    """Cross-attention from one modality to others."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, d_model) — tokens from one modality
            kv: (B, N_kv, d_model) — tokens from other modalities

        Returns:
            (B, N_q, d_model) — updated query tokens
        """
        attn_out, _ = self.cross_attn(query, kv, kv)
        return self.norm(query + attn_out)


class IntraCrossEncoder(nn.Module):
    def __init__(
        self,
        ir_len: int = 1801,
        h1_len: int = 1501,
        c13_len: int = 2201,
        patch_size: int = 64,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        n_bits: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model

        # Patch tokenizers
        self.ir_tokenizer = PatchTokenizer1D(ir_len, patch_size, d_model)
        self.h1_tokenizer = PatchTokenizer1D(h1_len, patch_size, d_model)
        self.c13_tokenizer = PatchTokenizer1D(c13_len, patch_size, d_model)

        # Modality embeddings
        self.ir_mod = ModalityEmbedding(d_model)
        self.h1_mod = ModalityEmbedding(d_model)
        self.c13_mod = ModalityEmbedding(d_model)

        # Intra-modal encoders (single layer per modality)
        self.ir_intra = IntraModalEncoder(d_model, num_heads, dropout)
        self.h1_intra = IntraModalEncoder(d_model, num_heads, dropout)
        self.c13_intra = IntraModalEncoder(d_model, num_heads, dropout)

        # Cross-modal attention (one block for each modality's cross-attention)
        self.ir_cross = CrossModalAttention(d_model, num_heads, dropout)
        self.h1_cross = CrossModalAttention(d_model, num_heads, dropout)
        self.c13_cross = CrossModalAttention(d_model, num_heads, dropout)

        # Post-cross Transformer to refine combined representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.post_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Head
        self.head = FingerprintHead(d_model, n_bits, dropout)

    def forward(
        self, ir: torch.Tensor, nmr_1h: torch.Tensor, nmr_13c: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ir: (B, 1801)
            nmr_1h: (B, 1501)
            nmr_13c: (B, 2201)

        Returns:
            (B, n_bits) — fingerprint logits
        """
        B = ir.shape[0]

        # Tokenize and add modality embeddings
        ir_tok = self.ir_mod(self.ir_tokenizer(ir))
        h1_tok = self.h1_mod(self.h1_tokenizer(nmr_1h))
        c13_tok = self.c13_mod(self.c13_tokenizer(nmr_13c))

        # Intra-modal encoding
        ir_tok = self.ir_intra(ir_tok)
        h1_tok = self.h1_intra(h1_tok)
        c13_tok = self.c13_intra(c13_tok)

        # Cross-modal attention:
        # IR attends to (1H + 13C)
        # 1H attends to (IR + 13C)
        # 13C attends to (IR + 1H)
        kv_ir = ir_tok  # Other modalities will attend to IR
        kv_h1 = h1_tok
        kv_c13 = c13_tok

        # For each modality, cross-attend to the OTHER two modalities
        ir_cross_kv = torch.cat([h1_tok, c13_tok], dim=1)
        h1_cross_kv = torch.cat([ir_tok, c13_tok], dim=1)
        c13_cross_kv = torch.cat([ir_tok, h1_tok], dim=1)

        ir_tok = self.ir_cross(ir_tok, ir_cross_kv)
        h1_tok = self.h1_cross(h1_tok, h1_cross_kv)
        c13_tok = self.c13_cross(c13_tok, c13_cross_kv)

        # Concatenate all updated tokens
        all_tokens = torch.cat([ir_tok, h1_tok, c13_tok], dim=1)

        # Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        # Post-cross Transformer
        out = self.post_transformer(all_tokens)

        # CLS -> head
        cls_out = out[:, 0, :]
        return self.head(cls_out)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
