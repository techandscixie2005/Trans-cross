"""Tests for model factory."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.factory import build_smiles_model
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel


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


class TestBuildSmilesModel:
    """Test build_smiles_model factory function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.config = _make_config()
        self.tokenizer = _make_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id

    def test_concat_equal_returns_correct_type(self):
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        assert isinstance(model, DirectConcatSmilesModel)

    def test_intra_cross_equal_returns_correct_type(self):
        model = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        assert isinstance(model, IntraCrossSmilesModel)

    def test_concat_legacy(self):
        model = build_smiles_model("concat", self.config, self.vocab_size, self.pad_id)
        assert isinstance(model, DirectConcatSmilesModel)

    def test_intra_cross_legacy(self):
        model = build_smiles_model("intra_cross", self.config, self.vocab_size, self.pad_id)
        assert isinstance(model, IntraCrossSmilesModel)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model_name"):
            build_smiles_model("nonexistent", self.config, self.vocab_size, self.pad_id)

    def test_same_vocab_pad_used(self):
        """Both models use the same vocab_size and pad_id."""
        m0 = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        m1 = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        assert m0.decoder.vocab_size == m1.decoder.vocab_size
        assert m0.pad_id == m1.pad_id

    def test_custom_d_model(self):
        config = dict(_make_config())
        config["shared"]["d_model"] = 256
        config["shared"]["num_heads"] = 8
        config["shared"]["decoder_ffn_dim"] = 1024
        config["e0_concat"]["encoder_ffn_dim"] = 1024
        config["e1_intra_cross"]["encoder_ffn_dim"] = 1024

        model = build_smiles_model("concat_equal", config, self.vocab_size, self.pad_id)
        assert model.d_model == 256
