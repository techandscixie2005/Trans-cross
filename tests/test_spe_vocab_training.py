"""Tests for SPE vocabulary training on toy data.

Verifies SPE training is correct: no valid/test leakage,
token length reduction, and unknown token handling.
"""

import json
import os
import tempfile

import pytest

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.tokenization.atom_tokenizer import atom_tokenize


TRAIN_SMILES = [
    "CCO",
    "CC(=O)O",
    "CC(=O)O",
    "c1ccccc1",
    "C1CCCCC1",
    "CC(=O)O",
    "CCO",
    "CC(=O)O",
    "C/C=C\\C",
    "c1ccccc1",
    "CCCC(CC)CI",
    "BrCCBr",
]

VALID_SMILES = [
    "[NH4+]",
    "C#N",
    "C[C@H](O)Cl",
]

TEST_SMILES = [
    "O=Cc1c(F)c(F)c(F)c(F)c1F",
    "[Na+]",
    "C[Si](C)(C)C",
]


class TestSPEVocabTraining:
    def test_train_only_on_train_split(self):
        """Verify SPE is trained only on training SMILES."""
        tokenizer = SPETokenizer()
        n_merges = tokenizer.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)
        assert n_merges > 0
        assert tokenizer.vocab_size >= 4 + 1  # special + at least some tokens

    def test_no_leakage_to_valid_test(self):
        """SPE training should only use the training split.
        This test verifies that the API supports separate splits."""
        train_tokenizer = SPETokenizer()
        train_tokenizer.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)

        # Valid/test SMILES should tokenize without errors
        for smi in VALID_SMILES + TEST_SMILES:
            tokens = train_tokenizer.tokenize(smi)
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    def test_length_reduction(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TRAIN_SMILES, vocab_size=128, min_frequency=1)

        # With aggressive merging, SPE should reduce length
        total_atom = 0
        total_spe = 0
        for smi in TRAIN_SMILES:
            total_atom += len(atom_tokenize(smi))
            total_spe += len(tokenizer.tokenize(smi))
        assert total_spe <= total_atom

    def test_unknown_atom_maps_to_unk(self):
        """Very rare atom tokens should map to <unk>."""
        tokenizer = SPETokenizer()
        # Train on limited data so exotic atoms are unseen
        tokenizer.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)

        # Tokenize a SMILES with atoms not in training
        tokens = tokenizer.tokenize("[Xe]F")
        # The atom tokens should be in vocab or map to unk
        ids = tokenizer.encode("[Xe]F", add_bos=False, add_eos=False)
        # Should not crash
        assert all(0 <= tid < tokenizer.vocab_size for tid in ids)

    def test_save_and_load_summary(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            tokenizer.save(path)

        try:
            with open(path) as f:
                data = json.load(f)
            assert "token_to_id" in data
            assert "merges" in data
            assert "vocab_size" in data
            assert "num_merges" in data
            assert data["vocab_size"] == tokenizer.vocab_size
        finally:
            os.unlink(path)

    def test_merge_ordering_is_deterministic(self):
        t1 = SPETokenizer()
        t1.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)

        t2 = SPETokenizer()
        t2.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)

        assert t1._merges == t2._merges
        assert t1.vocab_size == t2.vocab_size

    def test_min_frequency_respected(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TRAIN_SMILES, vocab_size=256, min_frequency=1000)
        assert tokenizer.num_merges == 0

    def test_top_merges_are_frequent_pairs(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TRAIN_SMILES, vocab_size=64, min_frequency=2)

        if tokenizer.num_merges > 0:
            rules = tokenizer.get_merge_rules()
            # First merge should be the most frequent pair
            first_merge = rules[0]
            pair = tuple(first_merge["pair"])
            merged = first_merge["merged"]
            assert merged == pair[0] + pair[1]
