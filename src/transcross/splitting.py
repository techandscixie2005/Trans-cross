"""Train/valid/test splitting with scaffold-based or random strategy."""

import json
import warnings
from collections import defaultdict
from typing import List, Dict, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False

from .smiles import canonicalize_smiles, get_scaffold


def _get_scaffold_safe(smiles: str) -> str:
    """Get Bemis-Murcko scaffold, falling back to canonical SMILES."""
    if not _HAS_RDKIT:
        return smiles
    return get_scaffold(smiles)


def scaffold_split(
    smiles_list: List[str],
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Split molecules by Bemis-Murcko scaffold.

    Scaffolds are sorted by frequency (largest first) and assigned to
    train/valid/test in a round-robin fashion to keep scaffold diversity
    balanced across splits.

    Returns:
        {"train": [...], "valid": [...], "test": [...]}
        where values are lists of sample indices.
    """
    # Group by scaffold
    scaffold_groups = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = _get_scaffold_safe(smi)
        scaffold_groups[scaffold].append(idx)

    # Sort scaffolds by size (largest first)
    sorted_scaffolds = sorted(
        scaffold_groups.items(), key=lambda x: len(x[1]), reverse=True
    )

    total = len(smiles_list)
    target_train = int(total * train_ratio)
    target_valid = int(total * valid_ratio)
    target_test = total - target_train - target_valid

    train_idx, valid_idx, test_idx = [], [], []

    for scaffold, indices in sorted_scaffolds:
        if len(train_idx) < target_train:
            train_idx.extend(indices)
        elif len(valid_idx) < target_valid:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)

    return {
        "train": sorted(train_idx),
        "valid": sorted(valid_idx),
        "test": sorted(test_idx),
    }


def random_split(
    smiles_list: List[str],
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Randomly split molecule indices into train/valid/test."""
    import random

    rng = random.Random(seed)
    indices = list(range(len(smiles_list)))
    rng.shuffle(indices)

    total = len(indices)
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)

    return {
        "train": sorted(indices[:n_train]),
        "valid": sorted(indices[n_train : n_train + n_valid]),
        "test": sorted(indices[n_train + n_valid :]),
    }


def create_split(
    smiles_list: List[str],
    method: str = "scaffold",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, List[int]], Dict]:
    """Create a train/valid/test split.

    Returns (splits_dict, summary_dict).
    """
    actual_method = method
    if method == "scaffold" and not _HAS_RDKIT:
        warnings.warn(
            "RDKit not available, falling back to random split. "
            "Install RDKit for scaffold-based splitting."
        )
        actual_method = "random"

    if actual_method == "scaffold":
        splits = scaffold_split(
            smiles_list, train_ratio, valid_ratio, test_ratio, seed
        )
    else:
        splits = random_split(
            smiles_list, train_ratio, valid_ratio, test_ratio, seed
        )

    summary = {
        "method": method,
        "actual_method": actual_method,
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "test_ratio": test_ratio,
        "total_samples": len(smiles_list),
        "train_count": len(splits["train"]),
        "valid_count": len(splits["valid"]),
        "test_count": len(splits["test"]),
    }

    # Verify no overlap
    train_set = set(splits["train"])
    valid_set = set(splits["valid"])
    test_set = set(splits["test"])
    assert (
        len(train_set & valid_set) == 0
    ), "Overlap between train and valid"
    assert len(train_set & test_set) == 0, "Overlap between train and test"
    assert len(valid_set & test_set) == 0, "Overlap between valid and test"

    total_assigned = len(train_set) + len(valid_set) + len(test_set)
    if total_assigned != len(smiles_list):
        warnings.warn(
            f"Split assigned {total_assigned} samples but "
            f"{len(smiles_list)} exist."
        )

    return splits, summary
