"""PyTorch Dataset for paired IR + NMR multimodal spectra."""

import json

import numpy as np


class TranscrossDataset:
    """Dataset that loads preprocessed IR + NMR arrays.

    No tokenization or model code — returns raw tensors.
    """

    def __init__(
        self,
        processed_dir: str,
        split: str | None = None,
    ):
        """
        Args:
            processed_dir: Path to directory containing ir.npy, nmr_1h.npy,
                nmr_13c.npy, canonical_smiles.txt, splits.json.
            split: One of "train", "valid", "test", or None for all samples.
        """
        self.processed_dir = processed_dir
        self.split = split

        self.ir = np.load(f"{processed_dir}/ir.npy")
        self.nmr_1h = np.load(f"{processed_dir}/nmr_1h.npy")
        self.nmr_13c = np.load(f"{processed_dir}/nmr_13c.npy")

        with open(f"{processed_dir}/canonical_smiles.txt") as f:
            self.smiles = [line.strip() for line in f if line.strip()]

        n = len(self.smiles)
        assert self.ir.shape[0] == n, f"IR count {self.ir.shape[0]} != {n}"
        assert self.nmr_1h.shape[0] == n, (
            f"NMR 1H count {self.nmr_1h.shape[0]} != {n}"
        )
        assert self.nmr_13c.shape[0] == n, (
            f"NMR 13C count {self.nmr_13c.shape[0]} != {n}"
        )

        self.indices = list(range(n))
        if split is not None:
            with open(f"{processed_dir}/splits.json") as f:
                splits = json.load(f)
            self.indices = splits.get(split, [])
            if not self.indices:
                raise ValueError(
                    f"Split '{split}' not found in splits.json. "
                    f"Available: {list(splits.keys())}"
                )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]
        return {
            "ir": self.ir[real_idx].copy(),
            "nmr_1h": self.nmr_1h[real_idx].copy(),
            "nmr_13c": self.nmr_13c[real_idx].copy(),
            "smiles": self.smiles[real_idx],
            "idx": real_idx,
        }
