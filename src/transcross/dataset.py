"""PyTorch Dataset for paired IR + NMR multimodal spectra and fingerprints."""

import json
import os
from typing import Optional

import numpy as np

from .smiles_tokenizer import SmilesTokenizer


class TranscrossDataset:
    """Dataset that loads preprocessed IR + NMR arrays.

    No tokenization or model code — returns raw tensors.
    """

    def __init__(
        self,
        processed_dir: str,
        split: Optional[str] = None,
    ):
        """
        Args:
            processed_dir: Path to directory containing ir.npy, nmr_1h.npy,
                nmr_13c.npy, canonical_smiles.txt, splits.json.
            split: One of "train", "valid", "test", or None for all samples.
        """
        self.processed_dir = processed_dir
        self.split = split

        self.ir = np.load(os.path.join(processed_dir, "ir.npy"))
        self.nmr_1h = np.load(os.path.join(processed_dir, "nmr_1h.npy"))
        self.nmr_13c = np.load(os.path.join(processed_dir, "nmr_13c.npy"))

        # Optional fingerprints
        fp_path = os.path.join(processed_dir, "morgan_fp_2048.npy")
        if os.path.exists(fp_path):
            self.fp = np.load(fp_path)
        else:
            self.fp = None

        with open(os.path.join(processed_dir, "canonical_smiles.txt")) as f:
            self.smiles = [line.strip() for line in f if line.strip()]

        n = len(self.smiles)
        assert self.ir.shape[0] == n, f"IR count {self.ir.shape[0]} != {n}"
        assert self.nmr_1h.shape[0] == n, f"NMR 1H count {self.nmr_1h.shape[0]} != {n}"
        assert self.nmr_13c.shape[0] == n, f"NMR 13C count {self.nmr_13c.shape[0]} != {n}"
        if self.fp is not None:
            assert self.fp.shape[0] == n, f"FP count {self.fp.shape[0]} != {n}"

        self.indices = list(range(n))
        if split is not None:
            with open(os.path.join(processed_dir, "splits.json")) as f:
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

        ir_vec = self.ir[real_idx].copy()
        h1_vec = self.nmr_1h[real_idx].copy()
        c13_vec = self.nmr_13c[real_idx].copy()

        result = {
            "ir": ir_vec,
            "nmr_1h": h1_vec,
            "nmr_13c": c13_vec,
            "smiles": self.smiles[real_idx],
            "idx": real_idx,
            "mask_ir": 1.0 if ir_vec.sum() > 0 else 0.0,
            "mask_1h": 1.0 if h1_vec.sum() > 0 else 0.0,
            "mask_13c": 1.0 if c13_vec.sum() > 0 else 0.0,
        }

        if self.fp is not None:
            result["fp"] = self.fp[real_idx].copy()

        return result


class TransCrossSmilesDataset:
    """Dataset for SMILES generation from IR + NMR spectra.

    Loads preprocessed spectra, SMILES, splits, and vocabulary.
    Returns tokenized SMILES for teacher forcing.
    """

    def __init__(
        self,
        processed_dir: str,
        split: Optional[str] = None,
        max_smiles_len: int = 160,
        tokenizer: Optional[SmilesTokenizer] = None,
    ):
        self.processed_dir = processed_dir
        self.split = split
        self.max_smiles_len = max_smiles_len

        self.ir = np.load(os.path.join(processed_dir, "ir.npy"))
        self.nmr_1h = np.load(os.path.join(processed_dir, "nmr_1h.npy"))
        self.nmr_13c = np.load(os.path.join(processed_dir, "nmr_13c.npy"))

        with open(os.path.join(processed_dir, "canonical_smiles.txt")) as f:
            self.smiles = [line.strip() for line in f if line.strip()]

        n = len(self.smiles)
        assert self.ir.shape[0] == n
        assert self.nmr_1h.shape[0] == n
        assert self.nmr_13c.shape[0] == n

        # Load or build tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            vocab_path = os.path.join(processed_dir, "smiles_vocab.json")
            if os.path.exists(vocab_path):
                self.tokenizer = SmilesTokenizer.load(vocab_path)
            else:
                self.tokenizer = SmilesTokenizer.build_from_smiles(self.smiles)
                # Don't save here — caller should have saved it

        self.indices = list(range(n))
        if split is not None:
            with open(os.path.join(processed_dir, "splits.json")) as f:
                splits = json.load(f)
            self.indices = splits.get(split, [])
            if not self.indices:
                raise ValueError(
                    f"Split '{split}' not found in splits.json. "
                    f"Available: {list(splits.keys())}"
                )

        # Filter out SMILES longer than max_smiles_len
        filtered = []
        for idx in self.indices:
            token_ids = self.tokenizer.encode(
                self.smiles[idx], add_bos=True, add_eos=True
            )
            if len(token_ids) <= max_smiles_len:
                filtered.append(idx)
        self.indices = filtered

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        real_idx = self.indices[idx]

        ir_vec = self.ir[real_idx].copy()
        h1_vec = self.nmr_1h[real_idx].copy()
        c13_vec = self.nmr_13c[real_idx].copy()
        smi = self.smiles[real_idx]

        # Tokenize with BOS/EOS
        token_ids = self.tokenizer.encode(smi, add_bos=True, add_eos=True)
        if len(token_ids) > self.max_smiles_len:
            token_ids = token_ids[:self.max_smiles_len]
            if token_ids[-1] != self.tokenizer.eos_id:
                token_ids[-1] = self.tokenizer.eos_id

        # Teacher forcing: input = [BOS] + tokens, target = tokens + [EOS]
        input_ids = token_ids[:-1]  # includes BOS, excludes EOS
        target_ids = token_ids[1:]   # excludes BOS, includes EOS

        return {
            "ir": ir_vec,
            "nmr_1h": h1_vec,
            "nmr_13c": c13_vec,
            "input_ids": np.array(input_ids, dtype=np.int64),
            "target_ids": np.array(target_ids, dtype=np.int64),
            "smiles": smi,
            "idx": real_idx,
            "mask_ir": 1.0 if ir_vec.sum() > 0 else 0.0,
            "mask_1h": 1.0 if h1_vec.sum() > 0 else 0.0,
            "mask_13c": 1.0 if c13_vec.sum() > 0 else 0.0,
        }
