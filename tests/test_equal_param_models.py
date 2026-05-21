"""Tests for equal-parameter SMILES generation ablation models.

Verifies:
1. E0 and E1 instantiate from config
2. Forward pass works
3. Decoder is identical
4. Parameter difference <= 1%
5. Causal mask present
6. Padding ignore_index works
7. Greedy generation works
"""

import os
import sys
import tempfile

import pytest
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.factory import build_smiles_model
from src.transcross.model_utils import compare_models
from src.transcross.generation import greedy_decode


def _make_config():
    """Return a minimal config for testing."""
    return {
        "data": {"processed_dir": "/tmp/test", "max_smiles_len": 160},
        "tokenizer": {
            "patch_size": 64,
            "use_modality_embedding": True,
            "use_absolute_position_embedding": True,
        },
        "shared": {
            "d_model": 128,
            "num_heads": 4,
            "decoder_layers": 2,
            "decoder_ffn_dim": 512,
            "dropout": 0.1,
        },
        "e0_concat": {
            "encoder_layers": 6,
            "encoder_ffn_dim": 512,
        },
        "e1_intra_cross": {
            "intra_layers": 1,
            "cross_layers": 1,
            "fusion_layers": 0,
            "encoder_ffn_dim": 512,
            "cross_zero_init_out_proj": True,
        },
        "training": {"epochs": 30, "batch_size": 32, "lr": 1e-4, "seed": 42},
        "equality_constraint": {"max_relative_param_diff": 0.01},
    }


def _make_tokenizer():
    """Create a minimal SMILES tokenizer for testing."""
    return SmilesTokenizer.build_from_smiles(["C", "CC", "CCO", "c1ccccc1"])


class TestEqualParamModels:
    """Test suite for equal-parameter SMILES ablation models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = _make_config()
        self.tokenizer = _make_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id

    def test_instantiate_e0(self):
        """E0 DirectConcat model instantiates from config."""
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        assert model is not None
        assert model.d_model == 128

    def test_instantiate_e1(self):
        """E1 IntraCross model instantiates from config."""
        model = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        assert model is not None
        assert model.d_model == 128

    def test_e0_forward_pass(self):
        """E0 forward pass produces logits with correct shape."""
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        B, T = 2, 10
        ir = torch.randn(B, 1801)
        nmr_1h = torch.randn(B, 1501)
        nmr_13c = torch.randn(B, 2201)
        input_ids = torch.randint(1, self.vocab_size, (B, T))

        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        assert logits.shape == (B, T, self.vocab_size)

    def test_e1_forward_pass(self):
        """E1 forward pass produces logits with correct shape."""
        model = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        B, T = 2, 10
        ir = torch.randn(B, 1801)
        nmr_1h = torch.randn(B, 1501)
        nmr_13c = torch.randn(B, 2201)
        input_ids = torch.randint(1, self.vocab_size, (B, T))

        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        assert logits.shape == (B, T, self.vocab_size)

    def test_decoder_identical(self):
        """Decoder parameter count is identical between E0 and E1."""
        model_e0 = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        model_e1 = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)

        e0_dec_params = sum(p.numel() for p in model_e0.decoder.parameters() if p.requires_grad)
        e1_dec_params = sum(p.numel() for p in model_e1.decoder.parameters() if p.requires_grad)
        assert e0_dec_params == e1_dec_params, (
            f"Decoder params differ: E0={e0_dec_params}, E1={e1_dec_params}"
        )

    def test_param_difference_within_tolerance(self):
        """Parameter difference between E0 and E1 is <= 1%."""
        model_e0 = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        model_e1 = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)

        result, within = compare_models(model_e0, model_e1, max_relative_diff=0.01)
        assert within, (
            f"Parameter difference {result['rel_diff_pct']:.4f}% exceeds 1%. "
            f"E0={result['e0_total']}, E1={result['e1_total']}"
        )

    def test_causal_mask_present(self):
        """Decoder has causal mask for autoregressive decoding."""
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        assert hasattr(model.decoder, "_build_causal_mask")

        # Verify causal mask is upper triangular with -inf
        mask = model.decoder._build_causal_mask(5, torch.device("cpu"))
        assert mask.shape == (5, 5)
        # Lower triangular should be 0
        assert mask[1, 0].item() == 0.0
        # Upper triangular should be -inf
        assert mask[0, 1].item() == float("-inf")

    def test_padding_ignore_index_in_loss(self):
        """CrossEntropyLoss with ignore_index works correctly."""
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        B, T = 4, 6
        ir = torch.randn(B, 1801)
        nmr_1h = torch.randn(B, 1501)
        nmr_13c = torch.randn(B, 2201)

        # Use small valid token IDs within vocab range
        valid_tok = min(3, self.vocab_size - 1)
        # input_ids: [BOS, t1, t2, EOS, PAD, PAD]
        bos = self.tokenizer.bos_id
        eos = self.tokenizer.eos_id
        pad = self.pad_id

        input_ids = torch.tensor([
            [bos, 1, 2, eos, pad, pad],
            [bos, 1, eos, pad, pad, pad],
        ])
        target_ids = torch.tensor([
            [1, 2, eos, pad, pad, pad],
            [1, eos, pad, pad, pad, pad],
        ])

        logits = model(ir[:2], nmr_1h[:2], nmr_13c[:2], input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=self.pad_id,
        )
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_greedy_generation(self):
        """Greedy decode produces token IDs for both models."""
        for model_key in ["concat_equal", "intra_cross_equal"]:
            model = build_smiles_model(model_key, self.config, self.vocab_size, self.pad_id)
            B = 2
            ir = torch.randn(B, 1801)
            nmr_1h = torch.randn(B, 1501)
            nmr_13c = torch.randn(B, 2201)

            result = greedy_decode(model, ir, nmr_1h, nmr_13c,
                                   self.tokenizer, max_len=50)
            assert len(result) == B
            for token_ids in result:
                assert len(token_ids) <= 50
                # EOS should not appear in decoded output (it's a stop token)
                for tid in token_ids:
                    assert tid != self.tokenizer.eos_id
            # Greedy decode should produce non-empty output (may include PAD for untrained models)
