"""E1: Intra-modal + cross-modal attention SMILES generation model.

Supports configurable number of intra-cross blocks (intra_cross_blocks).
Each block contains per-modality:
  1. Intra-modal self-attention (Pre-LN)
  2. Cross-modal attention to other modalities (Pre-LN, learnable gate)
  3. Feed-forward network (Pre-LN)

Block stacking:
  H_m^{b+1} = FFN_m^b(CrossAttn_m^b(SelfAttn_m^b(H_m^b)))

If intra_cross_blocks=1 and encoder_layers=0, this matches the v2 architecture
where intra encoding happens once followed by cross attention once.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

from .attention import TransformerBlockPreLN, CrossAttentionBlockPreLN
from .tokenizers import PatchTokenizer1D, ModalityEmbedding
from .smiles_decoder import TransformerSmilesDecoder


class IntraCrossBlock(nn.Module):
    """One intra-cross block: self-attn + cross-attn + FFN per modality.

    For each modality m in a block b:
        H_m = H_m + SelfAttn_m(LN(H_m))
        H_m = H_m + alpha_m * CrossAttn_m(LN(H_m), LN(H_not_m))
        H_m = H_m + FFN_m(LN(H_m))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int = 512,
        dropout: float = 0.1,
        gate_init: float = -4.0,
    ):
        super().__init__()
        # Intra-modal self-attention
        self.ir_intra = TransformerBlockPreLN(d_model, num_heads, d_ff=d_ff, dropout=dropout)
        self.h1_intra = TransformerBlockPreLN(d_model, num_heads, d_ff=d_ff, dropout=dropout)
        self.c13_intra = TransformerBlockPreLN(d_model, num_heads, d_ff=d_ff, dropout=dropout)

        # Cross-modal attention with learnable gate
        self.ir_cross = CrossAttentionBlockPreLN(
            d_model, num_heads, d_ff=d_ff, dropout=dropout, gate_init=gate_init,
        )
        self.h1_cross = CrossAttentionBlockPreLN(
            d_model, num_heads, d_ff=d_ff, dropout=dropout, gate_init=gate_init,
        )
        self.c13_cross = CrossAttentionBlockPreLN(
            d_model, num_heads, d_ff=d_ff, dropout=dropout, gate_init=gate_init,
        )

    def forward(
        self,
        ir: torch.Tensor,
        h1: torch.Tensor,
        c13: torch.Tensor,
    ):
        """Process one intra-cross block.

        Args:
            ir, h1, c13: (B, L_m, d_model) modality tokens

        Returns:
            Updated ir, h1, c13 tokens
        """
        # Intra-modal self-attention
        ir = self.ir_intra(ir)
        h1 = self.h1_intra(h1)
        c13 = self.c13_intra(c13)

        # Cross-modal attention: each modality attends to the other two
        kv_for_ir = torch.cat([h1, c13], dim=1)
        kv_for_h1 = torch.cat([ir, c13], dim=1)
        kv_for_c13 = torch.cat([ir, h1], dim=1)

        ir = self.ir_cross(ir, kv_for_ir)
        h1 = self.h1_cross(h1, kv_for_h1)
        c13 = self.c13_cross(c13, kv_for_c13)

        return ir, h1, c13

    def get_alphas(self) -> dict:
        """Return gate values for this block."""
        return {
            "ir": self.ir_cross.get_alpha().item(),
            "1h": self.h1_cross.get_alpha().item(),
            "13c": self.c13_cross.get_alpha().item(),
        }


class IntraCrossSmilesModel(nn.Module):
    """Intra-modal + cross-modal encoder with SMILES decoder.

    Supports two modes:
    1. Legacy (intra_cross_blocks=0): separate intra_layers, cross_layers, fusion_layers
    2. Block mode (intra_cross_blocks>0): stacked IntraCrossBlock modules
    """

    def __init__(
        self,
        vocab_size: int,
        ir_len: int = 1801,
        h1_len: int = 1501,
        c13_len: int = 2201,
        patch_size: int = 64,
        d_model: int = 128,
        encoder_layers: int = 0,
        cross_layers: int = 0,
        fusion_layers: int = 0,
        intra_cross_blocks: int = 0,
        encoder_ffn_dim: int = 512,
        decoder_layers: int = 2,
        decoder_ffn_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
        cross_gate_init: float = -4.0,
        pad_id: int = 0,
        max_smiles_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.cross_gate_init = cross_gate_init
        self.intra_cross_blocks = intra_cross_blocks

        # Patch tokenizers
        self.ir_tokenizer = PatchTokenizer1D(ir_len, patch_size, d_model)
        self.h1_tokenizer = PatchTokenizer1D(h1_len, patch_size, d_model)
        self.c13_tokenizer = PatchTokenizer1D(c13_len, patch_size, d_model)

        # Modality embeddings
        self.ir_mod = ModalityEmbedding(d_model)
        self.h1_mod = ModalityEmbedding(d_model)
        self.c13_mod = ModalityEmbedding(d_model)

        # Build encoder: either new block-based or legacy
        if intra_cross_blocks > 0:
            self.blocks = nn.ModuleList([
                IntraCrossBlock(
                    d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout,
                    gate_init=cross_gate_init,
                )
                for _ in range(intra_cross_blocks)
            ])
            self._use_legacy = False
        else:
            self._use_legacy = True
            # Legacy mode: separate intra, cross, fusion
            self.ir_intra = nn.ModuleList([
                TransformerBlockPreLN(d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout)
                for _ in range(encoder_layers)
            ])
            self.h1_intra = nn.ModuleList([
                TransformerBlockPreLN(d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout)
                for _ in range(encoder_layers)
            ])
            self.c13_intra = nn.ModuleList([
                TransformerBlockPreLN(d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout)
                for _ in range(encoder_layers)
            ])

            self.ir_cross = nn.ModuleList([
                CrossAttentionBlockPreLN(
                    d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout,
                    gate_init=cross_gate_init,
                )
                for _ in range(cross_layers)
            ])
            self.h1_cross = nn.ModuleList([
                CrossAttentionBlockPreLN(
                    d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout,
                    gate_init=cross_gate_init,
                )
                for _ in range(cross_layers)
            ])
            self.c13_cross = nn.ModuleList([
                CrossAttentionBlockPreLN(
                    d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout,
                    gate_init=cross_gate_init,
                )
                for _ in range(cross_layers)
            ])

            self.fusion_layers = nn.ModuleList([
                TransformerBlockPreLN(d_model, num_heads, d_ff=encoder_ffn_dim, dropout=dropout)
                for _ in range(fusion_layers)
            ]) if fusion_layers > 0 else None

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Shared SMILES decoder
        self.decoder = TransformerSmilesDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=decoder_layers,
            num_heads=num_heads,
            d_ff=decoder_ffn_dim,
            dropout=dropout,
            max_len=max_smiles_len,
            pad_id=pad_id,
        )

    def _encode_spectra_block(self, ir, h1, c13):
        """Block-based encoding: stack intra-cross blocks."""
        B = ir.shape[0]
        ir_tok = self.ir_mod(self.ir_tokenizer(ir))
        h1_tok = self.h1_mod(self.h1_tokenizer(h1))
        c13_tok = self.c13_mod(self.c13_tokenizer(c13))

        for block in self.blocks:
            ir_tok, h1_tok, c13_tok = block(ir_tok, h1_tok, c13_tok)

        all_tokens = torch.cat([ir_tok, h1_tok, c13_tok], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        total_len = all_tokens.shape[1]
        memory_key_padding_mask = torch.zeros(
            B, total_len, dtype=torch.bool, device=all_tokens.device
        )
        return all_tokens, memory_key_padding_mask

    def _encode_spectra_legacy(self, ir, h1, c13):
        """Legacy encoding: intra + cross + fusion (v1/v2 compat)."""
        B = ir.shape[0]
        ir_tok = self.ir_mod(self.ir_tokenizer(ir))
        h1_tok = self.h1_mod(self.h1_tokenizer(h1))
        c13_tok = self.c13_mod(self.c13_tokenizer(c13))

        # Intra-modal encoding
        for block in self.ir_intra:
            ir_tok = block(ir_tok)
        for block in self.h1_intra:
            h1_tok = block(h1_tok)
        for block in self.c13_intra:
            c13_tok = block(c13_tok)

        # Cross-modal attention
        for block in self.ir_cross:
            kv = torch.cat([h1_tok, c13_tok], dim=1)
            ir_tok = block(ir_tok, kv)
        for block in self.h1_cross:
            kv = torch.cat([ir_tok, c13_tok], dim=1)
            h1_tok = block(h1_tok, kv)
        for block in self.c13_cross:
            kv = torch.cat([ir_tok, h1_tok], dim=1)
            c13_tok = block(c13_tok, kv)

        # Concatenate + CLS
        all_tokens = torch.cat([ir_tok, h1_tok, c13_tok], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        total_len = all_tokens.shape[1]
        memory_key_padding_mask = torch.zeros(
            B, total_len, dtype=torch.bool, device=all_tokens.device
        )

        if self.fusion_layers is not None:
            for layer in self.fusion_layers:
                all_tokens = layer(all_tokens)

        return all_tokens, memory_key_padding_mask

    def _encode_spectra(
        self, ir: torch.Tensor, nmr_1h: torch.Tensor, nmr_13c: torch.Tensor
    ):
        if self._use_legacy:
            return self._encode_spectra_legacy(ir, nmr_1h, nmr_13c)
        return self._encode_spectra_block(ir, nmr_1h, nmr_13c)

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

    def get_gate_alphas(self) -> dict:
        """Return cross-attention gate values.

        In block mode: block_0, block_1, ... each with ir/1h/13c keys.
        In legacy mode: ir, 1h, 13c keys with list of values per cross layer.
        """
        if not self._use_legacy:
            result = {}
            for i, block in enumerate(self.blocks):
                result[f"block_{i}"] = block.get_alphas()
            return result
        else:
            alphas = {}
            if self.ir_cross:
                alphas["ir"] = [block.get_alpha().item() for block in self.ir_cross]
            if self.h1_cross:
                alphas["1h"] = [block.get_alpha().item() for block in self.h1_cross]
            if self.c13_cross:
                alphas["13c"] = [block.get_alpha().item() for block in self.c13_cross]
            return alphas

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
