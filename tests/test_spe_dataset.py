"""Tests for dataset and collate with SPE tokenizer."""

import json
import os
import tempfile

import numpy as np
import pytest

from src.transcross.dataset import TransCrossSmilesDataset
from src.transcross.collate import smiles_collate_fn
from src.transcross.tokenization.spe_tokenizer import SPETokenizer


TOY_SMILES = [
    "CCO",
    "CC(=O)O",
    "c1ccccc1",
    "C1CCCCC1",
    "C/C=C\\C",
    "CCCC(CC)CI",
    "BrCCBr",
    "C#N",
    "[NH4+]",
    "O=Cc1c(F)c(F)c(F)c(F)c1F",
]


def _make_toy_processed_dir():
    """Create a temporary processed directory with minimal data for testing."""
    tmpdir = tempfile.mkdtemp()
    n = len(TOY_SMILES)

    # Create synthetic spectra
    ir = np.random.randn(n, 1801).astype(np.float32)
    nmr_1h = np.random.randn(n, 1501).astype(np.float32)
    nmr_13c = np.random.randn(n, 2201).astype(np.float32)

    np.save(os.path.join(tmpdir, "ir.npy"), ir)
    np.save(os.path.join(tmpdir, "nmr_1h.npy"), nmr_1h)
    np.save(os.path.join(tmpdir, "nmr_13c.npy"), nmr_13c)

    with open(os.path.join(tmpdir, "canonical_smiles.txt"), "w") as f:
        for smi in TOY_SMILES:
            f.write(smi + "\n")

    # Simple splits: first 6 train, next 2 valid, last 2 test
    splits = {
        "train": list(range(6)),
        "valid": list(range(6, 8)),
        "test": list(range(8, 10)),
    }
    with open(os.path.join(tmpdir, "splits.json"), "w") as f:
        json.dump(splits, f)

    # Train SPE tokenizer and save
    tokenizer = SPETokenizer()
    train_smiles = [TOY_SMILES[i] for i in splits["train"]]
    tokenizer.train(train_smiles, vocab_size=64, min_frequency=1)
    tokenizer.save(os.path.join(tmpdir, "spe_vocab_256.json"))

    return tmpdir


class TestSPEDataset:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = _make_toy_processed_dir()
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_dataset_loads_spe_tokenizer(self):
        ds = TransCrossSmilesDataset(
            self.tmpdir, split="train",
            max_smiles_len=64, tokenizer_type="spe",
        )
        assert len(ds) > 0
        assert ds.tokenizer_type == "spe"
        assert ds.tokenizer.vocab_size >= 4

    def test_dataset_returns_token_ids(self):
        ds = TransCrossSmilesDataset(
            self.tmpdir, split="train",
            max_smiles_len=64, tokenizer_type="spe",
        )
        sample = ds[0]
        assert "input_ids" in sample
        assert "target_ids" in sample
        assert "smiles" in sample
        assert len(sample["input_ids"]) > 0
        # input_ids should be 1 shorter than input_ids + eos
        assert len(sample["target_ids"]) == len(sample["input_ids"])

    def test_dataset_teacher_forcing_format(self):
        ds = TransCrossSmilesDataset(
            self.tmpdir, split="train",
            max_smiles_len=64, tokenizer_type="spe",
        )
        sample = ds[0]
        # input_ids starts with BOS
        assert sample["input_ids"][0] == ds.tokenizer.bos_id
        # target_ids ends with EOS
        assert sample["target_ids"][-1] == ds.tokenizer.eos_id

    def test_dataset_max_length_filter(self):
        ds = TransCrossSmilesDataset(
            self.tmpdir, split="train",
            max_smiles_len=8, tokenizer_type="spe",
        )
        # All samples should have tokenized length <= max_smiles_len
        for i in range(len(ds)):
            sample = ds[i]
            assert len(sample["input_ids"]) + 1 <= 8  # +1 for eos

    def test_dataset_all_splits(self):
        for split in ["train", "valid", "test"]:
            ds = TransCrossSmilesDataset(
                self.tmpdir, split=split,
                max_smiles_len=64, tokenizer_type="spe",
            )
            assert len(ds) > 0


class TestSPECollate:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.tmpdir = _make_toy_processed_dir()
        yield
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_collate_pads_with_spe_pad_id(self):
        import torch

        ds = TransCrossSmilesDataset(
            self.tmpdir, split="train",
            max_smiles_len=64, tokenizer_type="spe",
        )
        pad_id = ds.tokenizer.pad_id

        # Get two samples with different lengths
        samples = [ds[i] for i in range(min(4, len(ds)))]
        batch = smiles_collate_fn(samples, pad_id)

        assert "input_ids" in batch
        assert "target_ids" in batch
        assert batch["input_ids"].ndim == 2  # (B, T_max)

        # Check padding values equal pad_id
        for i in range(len(samples)):
            seq_len = len(samples[i]["input_ids"])
            if seq_len < batch["input_ids"].shape[1]:
                padding = batch["input_ids"][i, seq_len:]
                assert (padding == pad_id).all()

    def test_collate_stacks_spectra(self):
        import torch

        ds = TransCrossSmilesDataset(
            self.tmpdir, split="train",
            max_smiles_len=64, tokenizer_type="spe",
        )
        pad_id = ds.tokenizer.pad_id

        samples = [ds[i] for i in range(min(4, len(ds)))]
        batch = smiles_collate_fn(samples, pad_id)

        assert batch["ir"].ndim == 2
        assert batch["nmr_1h"].ndim == 2
        assert batch["nmr_13c"].ndim == 2
        assert batch["ir"].shape[0] == len(samples)
