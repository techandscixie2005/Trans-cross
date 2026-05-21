"""Tests for SPE equal-parameter models.

Verifies model instantiation, forward pass, parameter matching,
and attention bias audit for SPE-tokenizer models.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import yaml

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.models.factory import build_smiles_model
from src.transcross.model_utils import (
    count_trainable_parameters,
    count_parameters_by_module,
    compare_models,
)


TOY_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "C1CCCCC1",
    "C/C=C\\C",
    "CCCC(CC)CI",
    "BrCCBr",
    "C#N",
    "[NH4+]",
    "O=Cc1c(F)c(F)c(F)c(F)c1F",
    "C[C@H](O)Cl",
    "[Na+]",
]


def _make_toy_tokenizer():
    tokenizer = SPETokenizer()
    tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=1)
    return tokenizer


def _make_spe_config():
    return {
        "data": {
            "processed_dir": "/tmp/test",
            "max_smiles_len": 96,
        },
        "tokenizer": {
            "type": "spe",
            "vocab_path": "/tmp/test/spe_vocab_256.json",
            "target_vocab_size": 256,
            "min_frequency": 2,
            "patch_size": 64,
        },
        "shared": {
            "d_model": 128,
            "num_heads": 4,
            "decoder_layers": 2,
            "decoder_ffn_dim": 512,
            "dropout": 0.1,
            "init_std": 0.02,
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
        "equality_constraint": {
            "max_relative_param_diff": 0.01,
        },
    }


class TestSPEEqualParamModels:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tokenizer = _make_toy_tokenizer()
        self.config = _make_spe_config()
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_id = self.tokenizer.pad_id

    def test_e0_instantiate(self):
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        assert model is not None
        n_params = count_trainable_parameters(model)
        assert n_params > 0

    def test_e1_instantiate(self):
        model = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        assert model is not None
        n_params = count_trainable_parameters(model)
        assert n_params > 0

    def test_e0_forward_pass(self):
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        B = 2
        ir = torch.randn(B, 1801)
        nmr_1h = torch.randn(B, 1501)
        nmr_13c = torch.randn(B, 2201)
        input_ids = torch.randint(4, self.vocab_size, (B, 10))

        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        assert logits.shape == (B, 10, self.vocab_size)
        assert not torch.isnan(logits).any()

    def test_e1_forward_pass(self):
        model = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)
        B = 2
        ir = torch.randn(B, 1801)
        nmr_1h = torch.randn(B, 1501)
        nmr_13c = torch.randn(B, 2201)
        input_ids = torch.randint(4, self.vocab_size, (B, 10))

        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        assert logits.shape == (B, 10, self.vocab_size)
        assert not torch.isnan(logits).any()

    def test_parameter_diff_within_tolerance(self):
        e0 = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        e1 = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)

        result, within = compare_models(e0, e1, max_relative_diff=0.01)
        assert within, f"Parameter diff {result['rel_diff_pct']:.4f}% exceeds 1%"

    def test_decoder_params_identical(self):
        e0 = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        e1 = build_smiles_model("intra_cross_equal", self.config, self.vocab_size, self.pad_id)

        e0_by_mod = count_parameters_by_module(e0)
        e1_by_mod = count_parameters_by_module(e1)

        assert e0_by_mod.get("decoder", 0) == e1_by_mod.get("decoder", 0), \
            "Decoder params not identical between E0 and E1"

    def test_no_attention_bias(self):
        """Audit models for attention bias violations."""
        FORBIDDEN = [
            "attention_bias", "relative_bias", "coord_bias",
            "modality_pair_bias", "graph_bias", "graphormer",
            "spatial_bias", "distance_bias",
        ]

        for model_key in ["concat_equal", "intra_cross_equal"]:
            model = build_smiles_model(model_key, self.config, self.vocab_size, self.pad_id)
            for name, param in model.named_parameters():
                name_lower = name.lower()
                for kw in FORBIDDEN:
                    assert kw not in name_lower, \
                        f"{model_key}: param '{name}' contains forbidden keyword '{kw}'"

    def test_vocab_size_from_config(self):
        """Model vocab size should match tokenizer vocab size."""
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        # Check decoder embedding size
        assert model.decoder.token_embed.num_embeddings == self.vocab_size
        assert model.decoder.output_proj.out_features == self.vocab_size

    def test_gradient_flow(self):
        """Verify gradients flow through the model."""
        model = build_smiles_model("concat_equal", self.config, self.vocab_size, self.pad_id)
        B = 2
        ir = torch.randn(B, 1801)
        nmr_1h = torch.randn(B, 1501)
        nmr_13c = torch.randn(B, 2201)
        input_ids = torch.randint(4, self.vocab_size, (B, 10))
        target_ids = torch.randint(4, self.vocab_size, (B, 10))

        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_ids.reshape(-1),
            ignore_index=self.pad_id,
        )
        loss.backward()

        # Check that gradients exist
        grad_count = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
        assert grad_count > 0, "No gradients flowed through the model"
