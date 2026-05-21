"""Tests for SMILES tokenizer: tokenization, vocabulary building, encode/decode."""

import pytest
import json
import tempfile
import os

from src.transcross.smiles_tokenizer import SmilesTokenizer, _SPECIAL_TOKENS


SAMPLE_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "C[C@H](N)C(=O)O",
    "BrC(Br)(Br)Br",
    "[Na+].[O-]c1ccc(Cl)cc1",
    "CC[Si](CC)(CC)CC",
    "C%12C%13C",
    "CCN(CC)CC",
]


class TestSmilesTokenization:
    def test_simple_tokenization(self):
        tokens = SmilesTokenizer.tokenize_smiles("CCO")
        assert tokens == ["C", "C", "O"]

    def test_branch_tokenization(self):
        tokens = SmilesTokenizer.tokenize_smiles("CC(=O)O")
        assert "(" in tokens
        assert ")" in tokens

    def test_ring_closure(self):
        tokens = SmilesTokenizer.tokenize_smiles("c1ccccc1")
        assert "1" in tokens

    def test_two_letter_element(self):
        tokens = SmilesTokenizer.tokenize_smiles("BrC(Br)(Br)Br")
        assert tokens.count("Br") == 4

    def test_stereochemistry(self):
        tokens = SmilesTokenizer.tokenize_smiles("C[C@H](N)C(=O)O")
        assert "@" in "".join(tokens) or any("@" in t for t in tokens)

    def test_bracket_atom(self):
        tokens = SmilesTokenizer.tokenize_smiles("[Na+].[O-]c1ccc(Cl)cc1")
        assert "[Na+]" in tokens
        assert "[O-]" in tokens

    def test_percent_ring_closure(self):
        tokens = SmilesTokenizer.tokenize_smiles("C%12C%13C")
        assert "%12" in tokens
        assert "%13" in tokens

    def test_empty_smiles(self):
        tokens = SmilesTokenizer.tokenize_smiles("")
        assert tokens == []

    def test_whitespace(self):
        tokens = SmilesTokenizer.tokenize_smiles("  CCO  ")
        assert tokens == ["C", "C", "O"]


class TestVocabularyBuilding:
    def test_build_from_smiles(self):
        tokenizer = SmilesTokenizer.build_from_smiles(SAMPLE_SMILES)
        assert tokenizer.vocab_size > len(_SPECIAL_TOKENS)
        assert tokenizer.pad_id == 0
        assert tokenizer.bos_id == 1
        assert tokenizer.eos_id == 2
        assert tokenizer.unk_id == 3

    def test_all_special_tokens_present(self):
        tokenizer = SmilesTokenizer.build_from_smiles(["CCO"])
        # All special token IDs should map back correctly
        assert tokenizer.decode([0], remove_special=False) == "<pad>"
        # pad removed when special tokens removed
        assert tokenizer.decode([0], remove_special=True) == ""

    def test_common_tokens_in_vocab(self):
        tokenizer = SmilesTokenizer.build_from_smiles(SAMPLE_SMILES)
        # Encode a simple molecule
        ids = tokenizer.encode("CCO", add_bos=True, add_eos=True)
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id


class TestEncodeDecode:
    @pytest.fixture
    def tokenizer(self):
        return SmilesTokenizer.build_from_smiles(SAMPLE_SMILES)

    def test_encode_decode_roundtrip(self, tokenizer):
        for smi in ["CCO", "c1ccccc1", "CC(=O)O"]:
            ids = tokenizer.encode(smi, add_bos=False, add_eos=False)
            decoded = tokenizer.decode(ids, remove_special=True)
            assert decoded == smi, f"Roundtrip failed: {smi} -> {ids} -> {decoded}"

    def test_encode_with_bos_eos(self, tokenizer):
        ids = tokenizer.encode("CCO", add_bos=True, add_eos=True)
        assert len(ids) >= 4  # BOS + ['C', 'C', 'O'] + EOS

    def test_encode_without_bos_eos(self, tokenizer):
        ids = tokenizer.encode("CCO", add_bos=False, add_eos=False)
        assert len(ids) == 3  # ['C', 'C', 'O']

    def test_encode_unknown_tokens(self, tokenizer):
        # A string with characters that likely aren't in the vocab
        ids = tokenizer.encode("[Xe]C", add_bos=False, add_eos=False)
        # Should still return ids, possibly with unk
        assert len(ids) > 0

    def test_decode_removes_special(self, tokenizer):
        ids = [tokenizer.bos_id, *tokenizer.encode("C", add_bos=False, add_eos=False), tokenizer.eos_id]
        decoded = tokenizer.decode(ids, remove_special=True)
        assert "<bos>" not in decoded
        assert "<eos>" not in decoded

    def test_decode_keeps_special(self, tokenizer):
        ids = [tokenizer.bos_id, *tokenizer.encode("C", add_bos=False, add_eos=False), tokenizer.eos_id]
        decoded = tokenizer.decode(ids, remove_special=False)
        assert "<bos>" in decoded
        assert "<eos>" in decoded


class TestSaveLoad:
    def test_save_load_roundtrip(self):
        tokenizer = SmilesTokenizer.build_from_smiles(SAMPLE_SMILES)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            tokenizer.save(f.name)
            saved_path = f.name

        loaded = SmilesTokenizer.load(saved_path)
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.pad_id == tokenizer.pad_id
        assert loaded.bos_id == tokenizer.bos_id
        assert loaded.eos_id == tokenizer.eos_id
        assert loaded.unk_id == tokenizer.unk_id

        # Check encode produces same results
        smi = "c1ccccc1"
        assert loaded.encode(smi) == tokenizer.encode(smi)

        os.unlink(saved_path)
