"""E0: Direct concatenation SMILES generation model.

IR + 1H NMR + 13C NMR tokens are concatenated and processed
by a standard Transformer encoder. A SMILES decoder attends to
the concatenated encoder memory.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .attention import TransformerBlockPreLN
from .tokenizers import PatchTokenizer1D, ModalityEmbedding
from .smiles_decoder import TransformerSmilesDecoder


class DirectConcatSmilesModel(nn.Module):
    """Direct concatenation encoder with SMILES decoder.

    IR, 1H, and 13C patch tokens are concatenated with a CLS token
    and processed jointly through self-attention layers. The resulting
    encoder memory is fed to the SMILES decoder.
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
        decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        pad_id: int = 0,
        max_smiles_len: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        # Patch tokenizers for each modality
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

        # Encoder layers (bias-free self-attention)
        self.encoder_layers = nn.ModuleList([
            TransformerBlockPreLN(d_model, num_heads, dropout=dropout)
            for _ in range(encoder_layers)
        ])

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

    def _encode_spectra(
        self, ir: torch.Tensor, nmr_1h: torch.Tensor, nmr_13c: torch.Tensor
    ):
        """Tokenize spectra and produce encoder memory.

        Returns:
            encoder_memory: (B, total_tokens, d_model)
            memory_key_padding_mask: (B, total_tokens)
        """
        B = ir.shape[0]

        ir_tok = self.ir_mod(self.ir_tokenizer(ir))
        h1_tok = self.h1_mod(self.h1_tokenizer(nmr_1h))
        c13_tok = self.c13_mod(self.c13_tokenizer(nmr_13c))

        # Concatenate all tokens
        all_tokens = torch.cat([ir_tok, h1_tok, c13_tok], dim=1)

        # Prepend CLS
        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_tokens = torch.cat([cls_tokens, all_tokens], dim=1)

        # No padding mask (all spectra tokens are valid)
        total_len = all_tokens.shape[1]
        memory_key_padding_mask = torch.zeros(
            B, total_len, dtype=torch.bool, device=all_tokens.device
        )

        # Process through encoder layers
        for layer in self.encoder_layers:
            all_tokens = layer(all_tokens, key_padding_mask=None)

        return all_tokens, memory_key_padding_mask

    def forward(
        self,
        ir: torch.Tensor,
        nmr_1h: torch.Tensor,
        nmr_13c: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ir: (B, 1801)
            nmr_1h: (B, 1501)
            nmr_13c: (B, 2201)
            input_ids: (B, T) — SMILES token IDs with BOS prefix

        Returns:
            logits: (B, T, vocab_size)
        """
        encoder_memory, memory_mask = self._encode_spectra(ir, nmr_1h, nmr_13c)
        logits = self.decoder(input_ids, encoder_memory, memory_padding_mask=memory_mask)
        return logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
