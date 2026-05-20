#!/usr/bin/env python3
"""Inspect processed output: shapes, statistics, sample SMILES.

Usage:
  python scripts/inspect_processed.py \
    --processed-dir data/processed
"""

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Inspect processed preprocessing output")
    parser.add_argument("--processed-dir", required=True, help="Path to processed data directory")
    args = parser.parse_args()

    d = args.processed_dir

    # Load arrays
    print(f"Inspecting: {d}\n")

    files_to_check = [
        "ir.npy",
        "ir_x.npy",
        "nmr_1h.npy",
        "nmr_1h_x.npy",
        "nmr_13c.npy",
        "nmr_13c_x.npy",
        "canonical_smiles.txt",
        "pairs.csv",
        "paired_records.jsonl",
        "pairing_summary.json",
        "preprocess_summary.json",
        "splits.json",
        "split_summary.json",
    ]

    for fname in files_to_check:
        fpath = os.path.join(d, fname)
        if os.path.exists(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"  [EXISTS] {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {fname}")

    print()

    # Load arrays and compute stats
    ir = np.load(os.path.join(d, "ir.npy"))
    nmr_1h = np.load(os.path.join(d, "nmr_1h.npy"))
    nmr_13c = np.load(os.path.join(d, "nmr_13c.npy"))

    n = ir.shape[0]
    print(f"Number of samples: {n}")
    print(f"IR shape:        {ir.shape}")
    print(f"NMR 1H shape:    {nmr_1h.shape}")
    print(f"NMR 13C shape:   {nmr_13c.shape}")

    # NaN/Inf check
    for name, arr in [("ir", ir), ("nmr_1h", nmr_1h), ("nmr_13c", nmr_13c)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        print(f"  {name}: NaN={nan_count}, Inf={inf_count}")

    # Statistics
    for name, arr in [("ir", ir), ("nmr_1h", nmr_1h), ("nmr_13c", nmr_13c)]:
        flat = arr.ravel()
        nonzero_frac = (flat > 0).sum() / len(flat)
        print(
            f"  {name}: min={flat.min():.4f} max={flat.max():.4f} "
            f"mean={flat.mean():.4f} nonzero={nonzero_frac:.3f}"
        )

    # SMILES
    smiles_path = os.path.join(d, "canonical_smiles.txt")
    if os.path.exists(smiles_path):
        with open(smiles_path) as f:
            smiles = [line.strip() for line in f if line.strip()]
        print(f"\nSMILES count: {len(smiles)}")
        print(f"First 3 SMILES:")
        for i, smi in enumerate(smiles[:3]):
            print(f"  [{i}] {smi}")

    # Split info
    splits_path = os.path.join(d, "splits.json")
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        print(f"\nSplits:")
        for split_name, indices in splits.items():
            print(f"  {split_name}: {len(indices)} samples")

    # Summary
    summary_path = os.path.join(d, "preprocess_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f"\nPreprocessing summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
