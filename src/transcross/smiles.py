"""SMILES canonicalization and scaffold extraction using RDKit."""

from typing import Optional

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def _require_rdkit():
    if not _HAS_RDKIT:
        raise ImportError(
            "RDKit is required for SMILES processing. "
            "Install with: conda install -c conda-forge rdkit"
        )


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit.

    Returns None if the SMILES is invalid or cannot be parsed.
    """
    _require_rdkit()
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_scaffold(smiles: str) -> str:
    """Extract the Bemis-Murcko scaffold for a SMILES string.

    Falls back to canonical SMILES if scaffold extraction fails.
    """
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return smiles.strip()
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, canonical=True)
        return Chem.MolToSmiles(scaffold, canonical=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string is valid and parseable by RDKit."""
    _require_rdkit()
    return canonicalize_smiles(smiles) is not None
