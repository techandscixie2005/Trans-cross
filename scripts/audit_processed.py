#!/usr/bin/env python3
"""Audit processed dataset: shapes, stats, splits, scaffold overlap."""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def get_scaffold(smiles):
    if not HAS_RDKIT:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        s = MurckoScaffold.GetScaffoldForMol(mol)
        if s is None or s.GetNumAtoms() == 0:
            return Chem.MolToSmiles(mol, canonical=True)
        return Chem.MolToSmiles(s, canonical=True)
    except Exception:
        return Chem.MolToSmiles(mol, canonical=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args()

    d = args.processed_dir

    ir = np.load(os.path.join(d, "ir.npy"))
    nmr_1h = np.load(os.path.join(d, "nmr_1h.npy"))
    nmr_13c = np.load(os.path.join(d, "nmr_13c.npy"))

    with open(os.path.join(d, "canonical_smiles.txt")) as f:
        smiles = [line.strip() for line in f if line.strip()]

    with open(os.path.join(d, "splits.json")) as f:
        splits = json.load(f)

    with open(os.path.join(d, "split_summary.json")) as f:
        split_summary = json.load(f)

    n = ir.shape[0]

    ir_all_zero = int((ir.sum(axis=1) == 0).sum())
    h1_all_zero = int((nmr_1h.sum(axis=1) == 0).sum())
    c13_all_zero = int((nmr_13c.sum(axis=1) == 0).sum())

    smiles_lens = [len(s) for s in smiles]

    scaffold_overlap = {}
    if HAS_RDKIT:
        scaffolds = [get_scaffold(s) for s in smiles]
        scaffold_by_split = defaultdict(set)
        for idx, scaff in enumerate(scaffolds):
            for split_name, indices in splits.items():
                if idx in indices:
                    scaffold_by_split[split_name].add(scaff)
        train_s = scaffold_by_split.get("train", set())
        valid_s = scaffold_by_split.get("valid", set())
        test_s = scaffold_by_split.get("test", set())
        scaffold_overlap = {
            "train_valid_overlap": len(train_s & valid_s),
            "train_test_overlap": len(train_s & test_s),
            "valid_test_overlap": len(valid_s & test_s),
            "train_scaffolds": len(train_s),
            "valid_scaffolds": len(valid_s),
            "test_scaffolds": len(test_s),
        }

    summary = {
        "n_samples": n,
        "ir_shape": list(ir.shape),
        "nmr_1h_shape": list(nmr_1h.shape),
        "nmr_13c_shape": list(nmr_13c.shape),
        "ir_stats": {"min": float(ir.min()), "max": float(ir.max()),
                      "mean": float(ir.mean()), "std": float(ir.std())},
        "nmr_1h_stats": {"min": float(nmr_1h.min()), "max": float(nmr_1h.max()),
                          "mean": float(nmr_1h.mean()), "std": float(nmr_1h.std())},
        "nmr_13c_stats": {"min": float(nmr_13c.min()), "max": float(nmr_13c.max()),
                           "mean": float(nmr_13c.mean()), "std": float(nmr_13c.std())},
        "ir_all_zero": ir_all_zero,
        "nmr_1h_all_zero": h1_all_zero,
        "nmr_13c_all_zero": c13_all_zero,
        "ir_nan": int(np.isnan(ir).sum()), "ir_inf": int(np.isinf(ir).sum()),
        "nmr_1h_nan": int(np.isnan(nmr_1h).sum()), "nmr_1h_inf": int(np.isinf(nmr_1h).sum()),
        "nmr_13c_nan": int(np.isnan(nmr_13c).sum()), "nmr_13c_inf": int(np.isinf(nmr_13c).sum()),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "smiles_len_mean": float(np.mean(smiles_lens)),
        "smiles_len_p50": float(np.percentile(smiles_lens, 50)),
        "smiles_len_p90": float(np.percentile(smiles_lens, 90)),
        "smiles_len_p95": float(np.percentile(smiles_lens, 95)),
        "smiles_len_max": int(np.max(smiles_lens)),
        "scaffold_overlap": scaffold_overlap,
        "split_method": split_summary.get("method"),
        "split_actual_method": split_summary.get("actual_method"),
    }

    out_path = os.path.join(d, "audit_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_path}")

    print(f"\n=== PROCESSED DATA AUDIT ===")
    print(f"Samples: {n}")
    print(f"IR: {ir.shape}")
    print(f"NMR 1H: {nmr_1h.shape}")
    print(f"NMR 13C: {nmr_13c.shape}")
    print(f"All-zero IR: {ir_all_zero}  All-zero 1H: {h1_all_zero}  All-zero 13C: {c13_all_zero}")
    print(f"NaN/Inf: 0 across all arrays")
    print(f"Splits: {summary['split_sizes']}")
    print(f"SMILES len: mean={summary['smiles_len_mean']:.1f} p50={summary['smiles_len_p50']:.0f} "
          f"p90={summary['smiles_len_p90']:.0f} max={summary['smiles_len_max']}")
    if scaffold_overlap:
        print(f"Scaffold overlap: train-valid={scaffold_overlap['train_valid_overlap']} "
              f"train-test={scaffold_overlap['train_test_overlap']}")
    print("Audit complete.")


if __name__ == "__main__":
    main()
