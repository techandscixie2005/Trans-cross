#!/usr/bin/env python3
"""Resample IR and bin NMR peaks to fixed grids.

Usage:
  python scripts/preprocess_spectra.py \
    --config configs/preprocessing.yaml \
    --pairs data/processed/paired_records.jsonl \
    --out-dir data/processed \
    --limit 200
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transcross.io import read_yaml, write_json
from src.transcross.spectra import (
    make_ir_grid,
    resample_ir,
    normalize_minmax,
    make_nmr_grid,
    bin_nmr_peaks,
)


def load_paired_records(path: str, limit: int | None = None):
    """Load paired_records.jsonl as list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def main():
    parser = argparse.ArgumentParser(description="Preprocess IR and NMR spectra")
    parser.add_argument("--config", required=True, help="Path to preprocessing.yaml")
    parser.add_argument("--pairs", required=True, help="Path to paired_records.jsonl")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max pairs to process")
    args = parser.parse_args()

    config = read_yaml(args.config)

    # IR grid
    ir_cfg = config["ir"]
    ir_grid = make_ir_grid(ir_cfg["min_cm"], ir_cfg["max_cm"], ir_cfg["step_cm"])
    print(f"IR grid: {len(ir_grid)} points ({ir_cfg['min_cm']}–{ir_cfg['max_cm']} cm⁻¹)")

    # NMR grids
    nmr_cfg = config["nmr"]
    nmr_mode = nmr_cfg["mode"]
    h1_cfg = nmr_cfg["1H"]
    c13_cfg = nmr_cfg["13C"]
    h1_grid = make_nmr_grid(h1_cfg["min_ppm"], h1_cfg["max_ppm"], h1_cfg["step_ppm"])
    c13_grid = make_nmr_grid(c13_cfg["min_ppm"], c13_cfg["max_ppm"], c13_cfg["step_ppm"])
    print(f"1H grid: {len(h1_grid)} points ({h1_cfg['min_ppm']}–{h1_cfg['max_ppm']} ppm)")
    print(f"13C grid: {len(c13_grid)} points ({c13_cfg['min_ppm']}–{c13_cfg['max_ppm']} ppm)")
    print(f"NMR mode: {nmr_mode}")

    # Load pairs
    print(f"\nLoading pairs from {args.pairs}...")
    pairs = load_paired_records(args.pairs, limit=args.limit)
    n = len(pairs)
    print(f"  Loaded {n} pairs")

    os.makedirs(args.out_dir, exist_ok=True)

    # Allocate arrays
    ir_arr = np.zeros((n, len(ir_grid)), dtype=np.float32)
    nmr_1h_arr = np.zeros((n, len(h1_grid)), dtype=np.float32)
    nmr_13c_arr = np.zeros((n, len(c13_grid)), dtype=np.float32)
    smiles_list = []

    h1_sigma = h1_cfg.get("sigma") if nmr_mode == "gaussian" else None
    c13_sigma = c13_cfg.get("sigma") if nmr_mode == "gaussian" else None

    ir_nan_count = 0
    skipped = 0

    for i, pair in enumerate(pairs):
        if i % 500 == 0:
            print(f"  Processing {i}/{n}...")

        # IR
        ir_x = np.array(pair["ir_x"], dtype=np.float64)
        ir_y = np.array(pair["ir_y"], dtype=np.float64)
        ir_resampled = resample_ir(ir_x, ir_y, ir_grid)

        if np.any(np.isnan(ir_resampled)) or np.any(np.isinf(ir_resampled)):
            ir_nan_count += 1
            ir_resampled = np.nan_to_num(ir_resampled, nan=0.0, posinf=0.0, neginf=0.0)

        ir_normalized = normalize_minmax(ir_resampled)
        ir_arr[i] = ir_normalized.astype(np.float32)

        # 1H NMR
        h1_peaks = pair.get("nmr_1h_peaks", [])
        nmr_1h_arr[i] = bin_nmr_peaks(
            h1_peaks, h1_grid, mode=nmr_mode, sigma=h1_sigma
        )

        # 13C NMR
        c13_peaks = pair.get("nmr_13c_peaks", [])
        nmr_13c_arr[i] = bin_nmr_peaks(
            c13_peaks, c13_grid, mode=nmr_mode, sigma=c13_sigma
        )

        smiles_list.append(pair["canonical_smiles"])

    print(f"  Processed {n} pairs, {ir_nan_count} had NaN in resampled IR")

    # Save arrays
    print("\nSaving arrays...")
    np.save(os.path.join(args.out_dir, "ir.npy"), ir_arr)
    np.save(os.path.join(args.out_dir, "ir_x.npy"), ir_grid.astype(np.float32))
    np.save(os.path.join(args.out_dir, "nmr_1h.npy"), nmr_1h_arr)
    np.save(os.path.join(args.out_dir, "nmr_1h_x.npy"), h1_grid.astype(np.float32))
    np.save(os.path.join(args.out_dir, "nmr_13c.npy"), nmr_13c_arr)
    np.save(os.path.join(args.out_dir, "nmr_13c_x.npy"), c13_grid.astype(np.float32))

    with open(os.path.join(args.out_dir, "canonical_smiles.txt"), "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")

    # Summary
    summary = {
        "n_samples": n,
        "ir_shape": list(ir_arr.shape),
        "ir_grid_range": [float(ir_grid[0]), float(ir_grid[-1])],
        "ir_grid_step": ir_cfg["step_cm"],
        "ir_nan_count": ir_nan_count,
        "ir_normalization": ir_cfg["normalization"],
        "nmr_1h_shape": list(nmr_1h_arr.shape),
        "nmr_1h_grid_range": [float(h1_grid[0]), float(h1_grid[-1])],
        "nmr_1h_grid_step": h1_cfg["step_ppm"],
        "nmr_13c_shape": list(nmr_13c_arr.shape),
        "nmr_13c_grid_range": [float(c13_grid[0]), float(c13_grid[-1])],
        "nmr_13c_grid_step": c13_cfg["step_ppm"],
        "nmr_mode": nmr_mode,
    }
    summary_path = os.path.join(args.out_dir, "preprocess_summary.json")
    write_json(summary_path, summary)
    print(f"  Saved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
