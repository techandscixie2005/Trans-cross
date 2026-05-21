"""E1: Intra-modal + cross-modal attention SMILES generation model.

IR, 1H NMR, and 13C NMR are first processed separately by intra-modal
self-attention, then cross-modal attention exchanges information between
modalities. The fused encoder memory is passed to the SMILES decoder.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import TransformerBlockPreLN, CrossAttentionBlockPreLN
from .tokenizers import PatchTokenizer1D, ModalityEmbedding
from .smiles_decoder import TransformerSmilesDecoder


class IntraCrossSmilesModel(nn.Module):
    """Intra-modal + cross-modal encoder with SMILES decoder.

    Stage 1: Each modality processed by independent intra-modal blocks.
    Stage 2: Cross-modal attention exchanges information between modalities.
    Stage 3: Fused tokens optionally pass through fusion self-attention blocks.
    The resulting encoder memory is fed to the SMILES decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        ir_len: int = 1801,
        h1_len: int = 1501,
        c13_len: int = 2201,
        patch_size: int = 64,
        d_model: int = 128,
        encoder_layers: int = 2,
        cross_layers: int = 1,
        fusion_layers: int = 1,
        decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        zero_init_cross_out_proj: bool = True,
        pad_id: int = 0,
        max_smiles_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # Patch tokenizers
        self.ir_tokenizer = PatchTokenizer1D(ir_len, patch_size, d_model)
        self.h1_tokenizer = PatchTokenizer1D(h1_len, patch_size, d_model)
        self.c13_tokenizer = PatchTokenizer1D(c13_len, patch_size, d_model)

        # Modality embeddings
        self.ir_mod = ModalityEmbedding(d_model)
        self.h1_mod = ModalityEmbedding(d_model)
        self.c13_mod = ModalityEmbedding(d_model)

        # Intra-modal encoders (separate per modality)
        self.ir_intra = nn.ModuleList([
            TransformerBlockPreLN(d_model, num_heads, dropout=dropout)
            for _ in range(encoder_layers)
        ])
        self.h1_intra = nn.ModuleList([
            TransformerBlockPreLN(d_model, num_heads, dropout=dropout)
            for _ in range(encoder_layers)
        ])
        self.c13_intra = nn.ModuleList([
            TransformerBlockPreLN(d_model, num_heads, dropout=dropout)
            for _ in range(encoder_layers)
        ])

        # Cross-modal attention blocks
        self.ir_cross = nn.ModuleList([
            CrossAttentionBlockPreLN(
                d_model, num_heads, dropout=dropout,
                zero_init_out_proj=zero_init_cross_out_proj,
            )
            for _ in range(cross_layers)
        ])
        self.h1_cross = nn.ModuleList([
            CrossAttentionBlockPreLN(
                d_model, num_heads, dropout=dropout,
                zero_init_out_proj=zero_init_cross_out_proj,
            )
            for _ in range(cross_layers)
        ])
        self.c13_cross = nn.ModuleList([
            CrossAttentionBlockPreLN(
                d_model, num_heads, dropout=dropout,
                zero_init_out_proj=zero_init_cross_out_proj,
            )
            for _ in range(cross_layers)
        ])

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Optional fusion layers after concatenation
        self.fusion_layers = nn.ModuleList([
            TransformerBlockPreLN(d_model, num_heads, dropout=dropout)
            for _ in range(fusion_layers)
        ]) if fusion_layers > 0 else None

        # Shared SMILES decoder
        self.decoder = TransformerSmilesDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_smiles_len,
            pad_id=pad_id,
        )

    def _intra_encode(self, tokens: torch.Tensor, blocks: nn.ModuleList) -> torch.Tensor:
        """Run intra-modal self-attention blocks on tokens."""
        for block in blocks:
            tokens = block(tokens)
        return tokens

    def _cross_attend(
        self,
        query: torch.Tensor,
        kv_a: torch.Tensor,
        kv_b: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        """Cross-attend query to concatenation of kv_a and kv_b."""
        kv = torch.cat([kv_a, kv_b], dim=1)
        for block in blocks:
            query = block(query, kv)
        return query

    def _encode_spectra(
        self, ir: torch.Tensor, nmr_1h: torch.Tensor, nmr_13c: torch.Tensor
    ):
        """Encode spectra through intra-modal + cross-modal + fusion.

        Returns:
            encoder_memory: (B, total_tokens, d_model)
            memory_key_padding_mask: (B, total_tokens)
        """
        B = ir.shape[0]

        # Tokenize and add modality embeddings
        ir_tok = self.ir_mod(self.ir_tokenizer(ir))
        h1_tok = self.h1_mod(self.h1_tokenizer(nmr_1h))
        c13_tok = self.c13_mod(self.c13_tokenizer(nmr_13c))

        # Stage 1: Intra-modal encoding
        ir_tok = self._intra_encode(ir_tok, self.ir_intra)
        h1_tok = self._intra_encode(h1_tok, self.h1_intra)
        c13_tok = self._intra_encode(c13_tok, self.c13_intra)

        # Stage 2: Cross-modal attention
        # IR attends to 1H + 13C, etc.
        ir_tok = self._cross_attend(ir_tok, h1_tok, c13_tok, self.ir_cross)
        h1_tok = self._cross_attend(h1_tok, ir_tok, c13_tok, self.h1_cross)
        c13_tok = self._cross_attend(c13_tok, ir_tok, h1_tok, self.c13_cross)

        # Stage 3: Concatenate + CLS
        all_tokens = torch.cat([ir_tok, h1_tok, c13_tok], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        # No padding mask
        total_len = all_tokens.shape[1]
        memory_key_padding_mask = torch.zeros(
            B, total_len, dtype=torch.bool, device=all_tokens.device
        )

        # Optional fusion layers
        if self.fusion_layers is not None:
            for layer in self.fusion_layers:
                all_tokens = layer(all_tokens)

        return all_tokens, memory_key_padding_mask

    def forward(
        self,
        ir: torch.Tensor,
        nmr_1h: torch.Tensor,
        nmr_13c: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        encoder_memory, memory_mask = self._encode_spectra(ir, nmr_1h, nmr_13c)
        logits = self.decoder(
            input_ids, encoder_memory, memory_padding_mask=memory_mask
        )
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
