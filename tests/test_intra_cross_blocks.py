"""Tests for E1 with multiple intra-cross blocks (v3 diagnostic)."""

import torch
import pytest

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel


SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture
def tokenizer():
    tok = SPETokenizer()
    tok.train(SAMPLE_SMILES, vocab_size=32, min_frequency=1)
    return tok


class TestE1MultiBlock:
    """Tests for E1 with multiple intra-cross blocks."""

    def test_3block_forward_shape(self, tokenizer):
        """E1 with 3 blocks produces correct output shape."""
        B = 2
        ir = torch.randn(B, 1801)
        h1 = torch.randn(B, 1501)
        c13 = torch.randn(B, 2201)
        input_ids = torch.randint(1, tokenizer.vocab_size, (B, 8))

        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=3, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        logits = model(ir, h1, c13, input_ids)
        assert logits.shape == (B, 8, tokenizer.vocab_size)

    def test_3block_forward_no_nan(self, tokenizer):
        """E1 with 3 blocks produces no NaN."""
        ir = torch.randn(2, 1801)
        h1 = torch.randn(2, 1501)
        c13 = torch.randn(2, 2201)
        input_ids = torch.randint(1, tokenizer.vocab_size, (2, 8))

        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=3, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        logits = model(ir, h1, c13, input_ids)
        assert not torch.isnan(logits).any()

    def test_3block_encoder_memory_shape(self, tokenizer):
        """E1 with 3 blocks produces correct encoder memory shape."""
        B = 2
        ir = torch.randn(B, 1801)
        h1 = torch.randn(B, 1501)
        c13 = torch.randn(B, 2201)

        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=3, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        mem, mask = model._encode_spectra(ir, h1, c13)
        # 29 + 24 + 35 + 1 CLS = 89 tokens
        assert mem.shape == (B, 89, 64)
        assert mask.shape == (B, 89)

    def test_3block_has_9_gates(self, tokenizer):
        """E1 with 3 blocks has 9 gate values (3 modalities × 3 blocks)."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=3, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        alphas = model.get_gate_alphas()
        assert len(alphas) == 3, f"expected 3 blocks, got {len(alphas)}"
        total_gates = 0
        for block_key, block_gates in alphas.items():
            assert len(block_gates) == 3, f"{block_key}: expected 3 modalities"
            total_gates += len(block_gates)
        assert total_gates == 9, f"expected 9 gates, got {total_gates}"

    def test_3block_gates_init_at_neg2(self, tokenizer):
        """All gates initialize near sigmoid(-2) = 0.1192."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=3, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        alphas = model.get_gate_alphas()
        for block_key, block_gates in alphas.items():
            for mod, val in block_gates.items():
                assert 0.10 < val < 0.13, \
                    f"{block_key}/{mod}: alpha={val} not near 0.119"

    def test_1block_reproduces_v2_behavior(self, tokenizer):
        """E1 with 1 block should match v2 behavior (comparing gate count)."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=1, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        alphas = model.get_gate_alphas()
        assert len(alphas) == 1
        for v in alphas["block_0"].values():
            assert 0.10 < v < 0.13

    def test_3block_no_modality_pair_bias(self, tokenizer):
        """Verify no modality-pair bias in 3-block E1."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            intra_cross_blocks=3, cross_gate_init=-2.0,
            pad_id=tokenizer.pad_id,
        )
        for name, _ in model.named_parameters():
            assert "modality_bias" not in name.lower()
            assert "pair_bias" not in name.lower()

    def test_legacy_model_still_works(self, tokenizer):
        """Legacy mode (encoder_layers=1, cross_layers=1) still functional."""
        B = 2
        ir = torch.randn(B, 1801)
        h1 = torch.randn(B, 1501)
        c13 = torch.randn(B, 2201)
        input_ids = torch.randint(1, tokenizer.vocab_size, (B, 8))

        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            encoder_layers=1, cross_layers=1,
            cross_gate_init=-4.0,
            pad_id=tokenizer.pad_id,
        )
        logits = model(ir, h1, c13, input_ids)
        assert logits.shape == (B, 8, tokenizer.vocab_size)
        assert not torch.isnan(logits).any()
