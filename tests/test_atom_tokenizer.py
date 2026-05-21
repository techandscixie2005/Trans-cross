"""Tests for atom-level SMILES tokenizer.

Verifies correct tokenization of various SMILES patterns including
bracket atoms, two-letter elements, stereochemistry, bonds, rings,
and round-trip detokenization.
"""

import pytest
from src.transcross.tokenization.atom_tokenizer import atom_tokenize, atom_detokenize


class TestAtomTokenize:
    def test_simple_ethanol(self):
        tokens = atom_tokenize("CCO")
        assert tokens == ["C", "C", "O"]

    def test_benzene_aromatic(self):
        tokens = atom_tokenize("c1ccccc1")
        assert tokens == ["c", "1", "c", "c", "c", "c", "c", "1"]

    def test_acetic_acid(self):
        tokens = atom_tokenize("CC(=O)O")
        assert tokens == ["C", "C", "(", "=", "O", ")", "O"]

    def test_chiral_with_chlorine(self):
        tokens = atom_tokenize("C[C@H](O)Cl")
        assert tokens == ["C", "[C@H]", "(", "O", ")", "Cl"]

    def test_pentafluorobenzaldehyde(self):
        tokens = atom_tokenize("O=Cc1c(F)c(F)c(F)c(F)c1F")
        assert "[" not in "".join(tokens) or all(
            t.startswith("[") == t.endswith("]") for t in tokens if "[" in t
        )

    def test_iodo_alkane(self):
        tokens = atom_tokenize("CCCC(CC)CI")
        assert "I" in tokens
        assert "C" in tokens

    def test_ammonium_bracket(self):
        tokens = atom_tokenize("[NH4+]")
        assert tokens == ["[NH4+]"]

    def test_cyclohexane(self):
        tokens = atom_tokenize("C1CCCCC1")
        assert "1" in tokens
        assert tokens.count("1") == 2

    def test_cis_trans_double_bond(self):
        tokens = atom_tokenize("C/C=C\\C")
        assert "/" in tokens
        assert "\\" in tokens

    def test_bromine(self):
        tokens = atom_tokenize("BrCCBr")
        assert tokens[0] == "Br"
        assert tokens[-1] == "Br"

    def test_silicon(self):
        tokens = atom_tokenize("C[Si](C)(C)C")
        assert "[Si]" in tokens

    def test_sodium(self):
        tokens = atom_tokenize("[Na+]")
        assert tokens == ["[Na+]"]

    def test_stereo_double_at(self):
        tokens = atom_tokenize("C[C@@H](O)Cl")
        assert "[C@@H]" in tokens

    def test_triple_bond(self):
        tokens = atom_tokenize("C#N")
        assert "#" in tokens

    def test_aromatic_nitrogen(self):
        tokens = atom_tokenize("c1ccncc1")
        assert "n" in tokens

    def test_percent_ring_closure(self):
        tokens = atom_tokenize("C%11CC%11")
        assert "%11" in tokens
        assert tokens.count("%11") == 2

    def test_charged_oxygen(self):
        tokens = atom_tokenize("[O-]")
        assert tokens == ["[O-]"]


class TestAtomDetokenize:
    @pytest.mark.parametrize("smiles", [
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "C[C@H](O)Cl",
        "O=Cc1c(F)c(F)c(F)c(F)c1F",
        "CCCC(CC)CI",
        "[NH4+]",
        "C1CCCCC1",
        "C/C=C\\C",
        "BrCCBr",
        "C#N",
        "[Na+]",
        "C[Si](C)(C)C",
    ])
    def test_round_trip(self, smiles):
        tokens = atom_tokenize(smiles)
        reconstructed = atom_detokenize(tokens)
        assert reconstructed == smiles, f"Round-trip failed: {smiles} -> {reconstructed}"

    def test_empty(self):
        assert atom_detokenize([]) == ""

    def test_detokenize_basic(self):
        tokens = ["C", "C", "O"]
        assert atom_detokenize(tokens) == "CCO"
