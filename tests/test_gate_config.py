"""Tests for configurable cross-attention gate initialization."""

import torch
import pytest

from src.transcross.models.attention import CrossAttentionBlockPreLN
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
from src.transcross.tokenization.spe_tokenizer import SPETokenizer


SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


@pytest.fixture
def tokenizer():
    tok = SPETokenizer()
    tok.train(SAMPLE_SMILES, vocab_size=32, min_frequency=1)
    return tok


class TestCrossAttentionGateInit:
    def test_default_gate_init(self):
        """Default gate_init=-4.0 gives alpha ≈ sigmoid(-4) ≈ 0.018."""
        block = CrossAttentionBlockPreLN(d_model=64, num_heads=4)
        assert torch.allclose(block.gate_logit, torch.tensor(-4.0))
        alpha = torch.sigmoid(block.gate_logit)
        assert 0.015 < alpha.item() < 0.02, f"alpha={alpha.item()}"

    def test_active_gate_init(self):
        """gate_init=-2.0 gives alpha ≈ sigmoid(-2) ≈ 0.119."""
        block = CrossAttentionBlockPreLN(d_model=64, num_heads=4, gate_init=-2.0)
        assert torch.allclose(block.gate_logit, torch.tensor(-2.0))
        alpha = torch.sigmoid(block.gate_logit)
        assert 0.10 < alpha.item() < 0.15, f"alpha={alpha.item()}"

    def test_zero_gate_init(self):
        """gate_init=0.0 gives alpha = 0.5."""
        block = CrossAttentionBlockPreLN(d_model=64, num_heads=4, gate_init=0.0)
        assert torch.allclose(block.gate_logit, torch.tensor(0.0))
        alpha = torch.sigmoid(block.gate_logit)
        assert abs(alpha.item() - 0.5) < 0.01

    def test_e1_default_v1_behavior(self, tokenizer):
        """E1 with default cross_gate_init=-4 preserves v1 backward compat."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
        )
        alphas = model.get_gate_alphas()
        for modality, vals in alphas.items():
            for v in vals:
                assert 0.015 < v < 0.02, f"{modality} alpha={v} not near 0.018"

    def test_e1_active_gate_v2(self, tokenizer):
        """E1 with cross_gate_init=-2 gives alpha ≈ 0.119."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            cross_gate_init=-2.0,
        )
        alphas = model.get_gate_alphas()
        for modality, vals in alphas.items():
            for v in vals:
                assert 0.10 < v < 0.15, f"{modality} alpha={v} not near 0.119"

    def test_e0_has_no_gate(self, tokenizer):
        """E0 (DirectConcatSmilesModel) has no cross-attention gate."""
        from src.transcross.models.smiles_concat import DirectConcatSmilesModel
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64,
        )
        for module in model.modules():
            assert not hasattr(module, "gate_logit"), \
                f"E0 should not have gate_logit, found in {type(module).__name__}"

    def test_no_modality_pair_bias(self, tokenizer):
        """Verify no modality-pair bias module exists."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            cross_gate_init=-2.0,
        )
        for name, _ in model.named_parameters():
            assert "modality_bias" not in name.lower(), \
                f"Found modality bias parameter: {name}"
            assert "pair_bias" not in name.lower(), \
                f"Found pair bias parameter: {name}"
