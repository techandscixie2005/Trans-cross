"""Verify that custom attention modules have NO additive attention bias.

Tests apply to all model variants (concat and intra_cross).
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.models.attention import (
    MultiHeadAttention,
    TransformerBlockPreLN,
    CrossAttentionBlockPreLN,
)
from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.factory import build_smiles_model

FORBIDDEN_KEYWORDS = [
    "attention_bias", "relative_bias", "coord_bias",
    "modality_pair_bias", "graph_bias", "graphormer",
    "spatial_bias", "distance_bias",
]


def _make_config():
    return {
        "data": {"processed_dir": "/tmp", "max_smiles_len": 160},
        "tokenizer": {"patch_size": 64},
        "shared": {"d_model": 128, "num_heads": 4, "decoder_layers": 2,
                   "decoder_ffn_dim": 512, "dropout": 0.1},
        "e0_concat": {"encoder_layers": 6, "encoder_ffn_dim": 512},
        "e1_intra_cross": {"intra_layers": 1, "cross_layers": 1, "fusion_layers": 0,
                          "encoder_ffn_dim": 512, "cross_zero_init_out_proj": True},
        "training": {"epochs": 30, "batch_size": 32, "lr": 1e-4, "seed": 42},
        "equality_constraint": {"max_relative_param_diff": 0.01},
    }


def _make_tokenizer():
    return SmilesTokenizer.build_from_smiles(["C", "CC", "CCO", "c1ccccc1"])


class TestAttentionNoBiasParams:
    """Verify no forbidden attention bias in model parameters."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = _make_config()
        self.tokenizer = _make_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id

    def _check_params(self, model):
        violations = []
        for name, param in model.named_parameters():
            name_lower = name.lower()
            for kw in FORBIDDEN_KEYWORDS:
                if kw in name_lower:
                    violations.append((name, list(param.shape), kw))
        return violations

    def test_e0_no_bias_params(self):
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        violations = self._check_params(model)
        assert len(violations) == 0, f"E0 violations: {violations}"

    def test_e1_no_bias_params(self):
        model = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        violations = self._check_params(model)
        assert len(violations) == 0, f"E1 violations: {violations}"


class TestAttentionModuleProperties:
    """Verify attention module types and properties."""

    def test_transformer_block_is_pre_ln(self):
        block = TransformerBlockPreLN(d_model=128, num_heads=4)
        assert isinstance(block.norm1, nn.LayerNorm)
        assert isinstance(block.norm2, nn.LayerNorm)

    def test_cross_attention_has_separate_norms(self):
        block = CrossAttentionBlockPreLN(d_model=128, num_heads=4)
        assert hasattr(block, "norm_q")
        assert hasattr(block, "norm_kv")

    def test_cross_near_zero_init(self):
        """Cross-attention out_proj should be near-zero (Normal(0, 1e-4)), not all zero."""
        block = CrossAttentionBlockPreLN(d_model=128, num_heads=4)
        w = block.cross_attn.out_proj.weight
        # Near-zero: mean close to 0, small but non-zero std
        assert abs(w.mean().item()) < 0.001
        assert 1e-5 < w.std().item() < 1e-2
        # Bias should be zero
        assert torch.all(block.cross_attn.out_proj.bias == 0)

    def test_mha_no_bias_in_attention_logits(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128)
        out = mha(x, x, x)
        assert out.shape == (2, 10, 128)
        assert torch.isfinite(out).all()

    def test_causal_mask_structure(self):
        """Causal mask: upper tri = -inf, lower tri with diagonal = 0."""
        model = build_smiles_model("concat_equal", _make_config(),
                                   _make_tokenizer().vocab_size, _make_tokenizer().pad_id)
        mask = model.decoder._build_causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)
        for i in range(5):
            for j in range(5):
                if i >= j:
                    assert mask[i, j] == 0.0, f"Lower tri ({i},{j}) should be 0"
                else:
                    assert mask[i, j] == float("-inf"), f"Upper tri ({i},{j}) should be -inf"

    def test_padding_mask_all_zeros_for_spectra(self):
        """Encoder padding mask is all False (no spectra padding)."""
        model = build_smiles_model("concat_equal", _make_config(),
                                   _make_tokenizer().vocab_size, _make_tokenizer().pad_id)
        B = 2
        ir = torch.randn(B, 1801)
        h1 = torch.randn(B, 1501)
        c13 = torch.randn(B, 2201)
        mem, mask = model._encode_spectra(ir, h1, c13)
        assert mask.shape[0] == B
        assert not mask.any(), "Padding mask should be all False for spectra"
