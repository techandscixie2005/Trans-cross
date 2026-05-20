#!/usr/bin/env python3
"""Split paired samples into train/valid/test using scaffold-based or random splitting.

Usage:
  python scripts/split_data.py \
    --config configs/preprocessing.yaml \
    --pairs data/processed/pairs.csv \
    --out-dir data/processed
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transcross.io import read_yaml, write_json
from src.transcross.splitting import create_split


def main():
    parser = argparse.ArgumentParser(description="Split data into train/valid/test")
    parser.add_argument("--config", required=True, help="Path to preprocessing.yaml")
    parser.add_argument("--pairs", required=True, help="Path to pairs.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    args = parser.parse_args()

    config = read_yaml(args.config)
    split_cfg = config["split"]

    # Load SMILES from pairs.csv
    print(f"Loading SMILES from {args.pairs}...")
    smiles_list = []
    with open(args.pairs, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["canonical_smiles"])

    n = len(smiles_list)
    print(f"  Loaded {n} SMILES")

    # Create split
    print(f"\nCreating {split_cfg['method']} split (seed={split_cfg['seed']})...")
    splits, summary = create_split(
        smiles_list,
        method=split_cfg["method"],
        train_ratio=split_cfg["train"],
        valid_ratio=split_cfg["valid"],
        test_ratio=split_cfg["test"],
        seed=split_cfg["seed"],
    )

    if summary.get("actual_method") != summary.get("method"):
        print(
            f"  WARNING: Requested {split_cfg['method']} but used "
            f"{summary['actual_method']}"
        )

    print(f"  Train: {summary['train_count']}")
    print(f"  Valid: {summary['valid_count']}")
    print(f"  Test:  {summary['test_count']}")

    os.makedirs(args.out_dir, exist_ok=True)

    splits_path = os.path.join(args.out_dir, "splits.json")
    write_json(splits_path, splits)
    print(f"  Saved: {splits_path}")

    summary_path = os.path.join(args.out_dir, "split_summary.json")
    write_json(summary_path, summary)
    print(f"  Saved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
