"""Tests for SPE tokenizer: training, encoding, decoding, save/load."""

import json
import os
import tempfile

import pytest

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.tokenization.atom_tokenizer import atom_tokenize


TOY_SMILES = [
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
    "C#N",
    "CCO",
    "CC(=O)O",
]


class TestSPETokenizerTrain:
    def test_train_basic(self):
        tokenizer = SPETokenizer()
        n_merges = tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=2)
        assert tokenizer.vocab_size >= _NUM_SPECIAL + 1
        assert n_merges > 0
        # Should have some merges
        assert tokenizer.num_merges == n_merges

    def test_train_min_frequency(self):
        tokenizer = SPETokenizer()
        # With very high min_frequency, no merges should happen
        n_merges = tokenizer.train(TOY_SMILES, vocab_size=256, min_frequency=100)
        assert n_merges == 0
        # But atom tokens should still be in vocab
        assert tokenizer.vocab_size >= _NUM_SPECIAL

    def test_train_vocab_limit(self):
        tokenizer = SPETokenizer()
        n_merges = tokenizer.train(TOY_SMILES, vocab_size=_NUM_SPECIAL + 5, min_frequency=1)
        # Should merge at most 5 times (to fill vocab from _NUM_SPECIAL to _NUM_SPECIAL+5)
        assert tokenizer.vocab_size <= _NUM_SPECIAL + 5 + len(set(
            t for s in TOY_SMILES for t in atom_tokenize(s)
        ))  # plus atom tokens


class TestSPETokenizerEncodeDecode:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tokenizer = SPETokenizer()
        self.tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=2)

    def test_special_token_ids(self):
        assert self.tokenizer.pad_id == 0
        assert self.tokenizer.bos_id == 1
        assert self.tokenizer.eos_id == 2
        assert self.tokenizer.unk_id == 3

    def test_encode_with_bos_eos(self):
        ids = self.tokenizer.encode("CCO", add_bos=True, add_eos=True)
        assert ids[0] == self.tokenizer.bos_id
        assert ids[-1] == self.tokenizer.eos_id

    def test_encode_without_special(self):
        ids = self.tokenizer.encode("CCO", add_bos=False, add_eos=False)
        assert self.tokenizer.bos_id not in ids
        assert self.tokenizer.eos_id not in ids

    @pytest.mark.parametrize("smiles", TOY_SMILES)
    def test_decode_inverts_encode(self, smiles):
        ids = self.tokenizer.encode(smiles, add_bos=True, add_eos=True)
        decoded = self.tokenizer.decode(ids, remove_special=True)
        assert decoded == smiles, f"Round-trip failed: {smiles} -> {decoded}"

    def test_tokenize_detokenize_round_trip(self):
        for smi in TOY_SMILES[:5]:
            tokens = self.tokenizer.tokenize(smi)
            reconstructed = self.tokenizer.detokenize(tokens)
            assert reconstructed == smi, f"Token round-trip: {smi} -> {reconstructed}"

    def test_decode_removes_special(self):
        ids = [self.tokenizer.bos_id, 5, 6, self.tokenizer.eos_id, self.tokenizer.pad_id]
        decoded = self.tokenizer.decode(ids, remove_special=True)
        assert "<pad>" not in decoded
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded

    def test_decode_keeps_special(self):
        ids = [self.tokenizer.bos_id, 5, self.tokenizer.eos_id]
        decoded = self.tokenizer.decode(ids, remove_special=False)
        assert "<bos>" in decoded
        assert "<eos>" in decoded

    def test_unknown_token_handling(self):
        # A token that doesn't exist in vocab
        ids = self.tokenizer.encode("[Xe]F", add_bos=False, add_eos=False)
        # Unknown atom token should map to unk_id
        # (unless [Xe] happens to be in atom vocab from training)
        assert all(0 <= tid < self.tokenizer.vocab_size for tid in ids)


class TestSPETokenizerSaveLoad:
    def test_save_load_round_trip(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=2)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
            tokenizer.save(path)

        try:
            loaded = SPETokenizer.load(path)
            assert loaded.vocab_size == tokenizer.vocab_size
            assert loaded.num_merges == tokenizer.num_merges
            assert loaded.pad_id == tokenizer.pad_id
            assert loaded.bos_id == tokenizer.bos_id

            # Encode/decode should match
            for smi in TOY_SMILES[:3]:
                assert loaded.encode(smi) == tokenizer.encode(smi)
                assert loaded.decode(tokenizer.encode(smi)) == tokenizer.decode(tokenizer.encode(smi))
        finally:
            os.unlink(path)

    def test_merge_rules_output(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=2)
        rules = tokenizer.get_merge_rules()
        assert len(rules) == tokenizer.num_merges
        for rule in rules:
            assert "pair" in rule
            assert "merged" in rule
            assert len(rule["pair"]) == 2
            assert rule["merged"] == rule["pair"][0] + rule["pair"][1]


class TestSPEMergeBehavior:
    def test_merge_reduces_length(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=2)

        spe_len = len(tokenizer.tokenize("CC(=O)O"))
        atom_len = len(atom_tokenize("CC(=O)O"))
        # With enough training, SPE should be shorter or equal
        assert spe_len <= atom_len

    def test_tokenization_is_deterministic(self):
        tokenizer = SPETokenizer()
        tokenizer.train(TOY_SMILES, vocab_size=64, min_frequency=2)

        t1 = tokenizer.tokenize("CC(=O)Oc1ccccc1")
        t2 = tokenizer.tokenize("CC(=O)Oc1ccccc1")
        assert t1 == t2


# Constants from spe_tokenizer
_NUM_SPECIAL = 4
