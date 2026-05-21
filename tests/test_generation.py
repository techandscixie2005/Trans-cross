"""Tests for greedy SMILES generation."""

import pytest
import torch

from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
from src.transcross.generation import greedy_decode


SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture
def tokenizer():
    return SmilesTokenizer.build_from_smiles(SAMPLE_SMILES)


@pytest.fixture
def sample_spectra():
    B = 2
    ir = torch.randn(B, 1801)
    h1 = torch.randn(B, 1501)
    c13 = torch.randn(B, 2201)
    return ir, h1, c13


class TestGreedyDecodeConcat:
    def test_returns_token_lists(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=50)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_generated_tokens_in_vocab_range(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=50)
        for seq in results:
            for tok in seq:
                assert 0 <= tok < tokenizer.vocab_size

    def test_decoded_smiles_nonempty(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=50)
        for seq in results:
            smi = tokenizer.decode(seq, remove_special=True)
            assert len(smi) > 0

    def test_no_eos_in_decoded_output(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=50)
        for seq in results:
            assert tokenizer.eos_id not in seq


class TestGreedyDecodeIntraCross:
    def test_returns_token_lists(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
            cross_layers=1, fusion_layers=1,
        )
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=50)
        assert len(results) == 2

    def test_stops_before_max_len(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
            cross_layers=1, fusion_layers=1,
        )
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=100)
        for seq in results:
            assert len(seq) <= 100  # should stop at EOS or max_len


class TestMaxLenTruncation:
    def test_truncates_at_max_len(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        # Very short max_len to force truncation
        results = greedy_decode(model, ir, h1, c13, tokenizer, max_len=3)
        for seq in results:
            assert len(seq) <= 3
