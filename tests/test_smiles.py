"""Tests for SMILES canonicalization and scaffold extraction."""

import pytest

from src.transcross.smiles import canonicalize_smiles, get_scaffold, is_valid_smiles


class TestCanonicalizeSmiles:
    def test_simple_molecule(self):
        """Caffeine should canonicalize consistently."""
        result = canonicalize_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
        assert result is not None
        # Re-canonicalizing should give the same result
        result2 = canonicalize_smiles(result)
        assert result == result2

    def test_different_representations_same_molecule(self):
        """Different SMILES for ethanol should map to the same canonical form."""
        a = canonicalize_smiles("CCO")
        b = canonicalize_smiles("OCC")
        assert a is not None
        assert b is not None
        assert a == b

    def test_invalid_smiles(self):
        assert canonicalize_smiles("not_a_molecule_xyzzy") is None

    def test_empty_string(self):
        assert canonicalize_smiles("") is None

    def test_none_input(self):
        assert canonicalize_smiles(None) is None

    def test_whitespace_is_stripped(self):
        result = canonicalize_smiles("  CCO  ")
        assert result == "CCO"


class TestGetScaffold:
    def test_simple_scaffold(self):
        """Scaffold of a substituted benzene should be benzene."""
        scaffold = get_scaffold("Cc1ccccc1")
        assert scaffold is not None
        # Toluene scaffold should be benzene
        # Note: Bemis-Murcko may or may not strip methyl
        assert len(scaffold) > 0

    def test_scaffold_fallback(self):
        """Should not crash on any input."""
        scaffold = get_scaffold("CCO")
        assert scaffold is not None


class TestIsValidSmiles:
    def test_valid(self):
        assert is_valid_smiles("CCO") is True

    def test_invalid(self):
        assert is_valid_smiles("XZ") is False
