"""Tests for IR–NMR pairing logic."""

from src.transcross.pairing import build_pairs


class TestBuildPairs:
    def test_basic_pairing(self):
        """Molecules with both IR and NMR should be paired."""
        ir_catalog = {
            "CCO": {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "x": [500.0, 600.0],
                "y": [10.0, 20.0],
                "x_len": 2,
                "condition": "liquid",
                "temperature": "NONE",
                "pressure": "NONE",
            },
            "CCN": {
                "smiles": "CCN",
                "canonical_smiles": "CCN",
                "line_idx": 1,
                "x": [500.0, 600.0],
                "y": [10.0, 20.0],
                "x_len": 2,
                "condition": "gas",
                "temperature": "NONE",
                "pressure": "NONE",
            },
        }

        nmr_catalog = {
            ("CCO", "1H"): {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "nucleus": "1H",
                "peaks": [1.2, 3.7],
                "num_peaks": 2,
                "frequency": 300.0,
                "solvent": "CDCl3",
            },
            ("CCO", "13C"): {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 1,
                "nucleus": "13C",
                "peaks": [18.0, 58.0],
                "num_peaks": 2,
                "frequency": 75.0,
                "solvent": "CDCl3",
            },
        }

        pairs = build_pairs(ir_catalog, nmr_catalog)
        assert len(pairs) == 1  # Only CCO has NMR
        assert pairs[0]["canonical_smiles"] == "CCO"
        assert pairs[0]["sample_id"] == 0
        assert pairs[0]["nmr_1h_line_idx"] == 0
        assert pairs[0]["nmr_13c_line_idx"] == 1
        assert "1H" in pairs[0]["available_nuclei"]
        assert "13C" in pairs[0]["available_nuclei"]

    def test_no_nmr_no_pair(self):
        """Molecule with IR only should not be paired."""
        ir_catalog = {
            "CCO": {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "x": [500.0, 600.0],
                "y": [10.0, 20.0],
                "x_len": 2,
                "condition": "liquid",
                "temperature": "NONE",
                "pressure": "NONE",
            },
        }
        nmr_catalog = {}
        pairs = build_pairs(ir_catalog, nmr_catalog)
        assert len(pairs) == 0

    def test_partial_nmr(self):
        """Molecule with only 1H (no 13C) should still pair."""
        ir_catalog = {
            "CCO": {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "x": [500.0, 600.0],
                "y": [10.0, 20.0],
                "x_len": 2,
                "condition": "liquid",
                "temperature": "NONE",
                "pressure": "NONE",
            },
        }
        nmr_catalog = {
            ("CCO", "1H"): {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "nucleus": "1H",
                "peaks": [1.2, 3.7],
                "num_peaks": 2,
                "frequency": 300.0,
                "solvent": "CDCl3",
            },
        }
        pairs = build_pairs(ir_catalog, nmr_catalog)
        assert len(pairs) == 1
        assert pairs[0]["nmr_1h_line_idx"] is not None
        assert pairs[0]["nmr_13c_line_idx"] is None
        assert pairs[0]["nmr_13c_peaks"] == []

    def test_multiple_molecules(self):
        """Multiple molecules should be assigned sequential sample_ids."""
        ir_catalog = {
            "CCO": {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "x": [500.0, 600.0],
                "y": [10.0, 20.0],
                "x_len": 2,
                "condition": "liquid",
                "temperature": "NONE",
                "pressure": "NONE",
            },
            "CCN": {
                "smiles": "CCN",
                "canonical_smiles": "CCN",
                "line_idx": 1,
                "x": [500.0, 600.0],
                "y": [10.0, 20.0],
                "x_len": 2,
                "condition": "gas",
                "temperature": "NONE",
                "pressure": "NONE",
            },
        }
        nmr_catalog = {
            ("CCO", "1H"): {
                "smiles": "CCO",
                "canonical_smiles": "CCO",
                "line_idx": 0,
                "nucleus": "1H",
                "peaks": [1.2],
                "num_peaks": 1,
                "frequency": 300.0,
                "solvent": "CDCl3",
            },
            ("CCN", "13C"): {
                "smiles": "CCN",
                "canonical_smiles": "CCN",
                "line_idx": 1,
                "nucleus": "13C",
                "peaks": [18.0],
                "num_peaks": 1,
                "frequency": 75.0,
                "solvent": "CDCl3",
            },
        }
        pairs = build_pairs(ir_catalog, nmr_catalog)
        assert len(pairs) == 2
        assert {p["sample_id"] for p in pairs} == {0, 1}
