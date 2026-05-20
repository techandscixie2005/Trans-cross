"""E0: Direct concatenation encoder.

IR + 1H NMR + 13C NMR tokens are concatenated and processed
by a standard Transformer encoder. CLS token -> fingerprint head.
"""

import torch
import torch.nn as nn

from .tokenizers import PatchTokenizer1D, ModalityEmbedding
from .fingerprint_head import FingerprintHead


class ConcatEncoder(nn.Module):
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

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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

        # Tokenize each modality
        ir_tok = self.ir_tokenizer(ir)       # (B, N_ir, d_model)
        h1_tok = self.h1_tokenizer(nmr_1h)   # (B, N_h1, d_model)
        c13_tok = self.c13_tokenizer(nmr_13c)  # (B, N_c13, d_model)

        # Add modality embeddings
        ir_tok = self.ir_mod(ir_tok)
        h1_tok = self.h1_mod(h1_tok)
        c13_tok = self.c13_mod(c13_tok)

        # Concatenate all tokens
        all_tokens = torch.cat([ir_tok, h1_tok, c13_tok], dim=1)  # (B, N_total, d_model)

        # Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        # Transformer
        out = self.transformer(all_tokens)

        # CLS -> head
        cls_out = out[:, 0, :]  # (B, d_model)
        return self.head(cls_out)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
