#!/usr/bin/env python3
"""Build IR–NMR molecule pairs via canonical SMILES matching.

Usage:
  python scripts/build_pairs.py \
    --config configs/preprocessing.yaml \
    --raw-dir /data/home/sczc698/run/xxy/Trans-cross/ \
    --out-dir data/processed \
    --limit-ir 200 \
    --limit-nmr 500000
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transcross.io import read_yaml, write_json, write_jsonl
from src.transcross.pairing import scan_ir_records, scan_nmr_records, build_pairs


def main():
    parser = argparse.ArgumentParser(description="Build IR–NMR molecule pairs")
    parser.add_argument("--config", required=True, help="Path to preprocessing.yaml")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw JSONL files")
    parser.add_argument("--out-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--limit-ir", type=int, default=None, help="Max IR records to scan")
    parser.add_argument("--limit-nmr", type=int, default=None, help="Max NMR records to scan")
    args = parser.parse_args()

    config = read_yaml(args.config)

    ir_filename = config["raw"]["ir_filename"]
    nmr_filename = config["raw"]["nmr_filename"]
    ir_path = os.path.join(args.raw_dir, ir_filename)
    nmr_path = os.path.join(args.raw_dir, nmr_filename)

    limit_ir = args.limit_ir if args.limit_ir is not None else config["limits"].get("limit_ir")
    limit_nmr = args.limit_nmr if args.limit_nmr is not None else config["limits"].get("limit_nmr")

    print(f"IR file: {ir_path}")
    print(f"NMR file: {nmr_path}")
    print(f"IR limit: {limit_ir or 'none'}")
    print(f"NMR limit: {limit_nmr or 'none'}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Scan IR
    print("\n[1/3] Scanning IR records...")
    ir_catalog = scan_ir_records(ir_path, limit=limit_ir)
    print(f"  IR records loaded: {len(ir_catalog)} unique canonical SMILES")

    # Step 2: Scan NMR with allowed SMILES filter
    print("\n[2/3] Scanning NMR records...")
    allowed_smiles = set(ir_catalog.keys())
    nmr_catalog = scan_nmr_records(nmr_path, allowed_smiles=allowed_smiles, limit=limit_nmr)
    print(f"  NMR records loaded: {len(nmr_catalog)} (unique SMILES, nucleus pairs)")

    # Step 3: Build pairs
    print("\n[3/3] Building pairs...")
    pairs = build_pairs(ir_catalog, nmr_catalog)
    print(f"  Paired molecules: {len(pairs)}")

    # Statistics
    has_1h = sum(1 for p in pairs if p["nmr_1h_line_idx"] is not None)
    has_13c = sum(1 for p in pairs if p["nmr_13c_line_idx"] is not None)
    has_both = sum(
        1
        for p in pairs
        if p["nmr_1h_line_idx"] is not None and p["nmr_13c_line_idx"] is not None
    )
    print(f"  With 1H NMR: {has_1h}")
    print(f"  With 13C NMR: {has_13c}")
    print(f"  With both 1H and 13C: {has_both}")

    # Save pairs.csv (without large array fields)
    csv_path = os.path.join(args.out_dir, "pairs.csv")
    csv_fields = [
        "sample_id",
        "canonical_smiles",
        "ir_line_idx",
        "ir_num_points",
        "ir_condition",
        "nmr_1h_line_idx",
        "nmr_1h_num_peaks",
        "nmr_1h_frequency",
        "nmr_1h_solvent",
        "nmr_13c_line_idx",
        "nmr_13c_num_peaks",
        "nmr_13c_frequency",
        "nmr_13c_solvent",
        "available_nuclei",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for p in pairs:
            row = {k: p[k] for k in csv_fields}
            writer.writerow(row)
    print(f"  Saved: {csv_path}")

    # Save paired_records.jsonl (compact, with full arrays for preprocessing step)
    rec_path = os.path.join(args.out_dir, "paired_records.jsonl")
    compact_pairs = []
    for p in pairs:
        compact = {
            "sample_id": p["sample_id"],
            "canonical_smiles": p["canonical_smiles"],
            "original_smiles": p["original_smiles"],
            "ir_line_idx": p["ir_line_idx"],
            "ir_x": p["ir_x"],
            "ir_y": p["ir_y"],
            "ir_num_points": p["ir_num_points"],
            "ir_condition": p["ir_condition"],
            "nmr_1h_line_idx": p["nmr_1h_line_idx"],
            "nmr_1h_peaks": p["nmr_1h_peaks"],
            "nmr_1h_num_peaks": p["nmr_1h_num_peaks"],
            "nmr_1h_frequency": p["nmr_1h_frequency"],
            "nmr_1h_solvent": p["nmr_1h_solvent"],
            "nmr_13c_line_idx": p["nmr_13c_line_idx"],
            "nmr_13c_peaks": p["nmr_13c_peaks"],
            "nmr_13c_num_peaks": p["nmr_13c_num_peaks"],
            "nmr_13c_frequency": p["nmr_13c_frequency"],
            "nmr_13c_solvent": p["nmr_13c_solvent"],
            "available_nuclei": p["available_nuclei"],
        }
        compact_pairs.append(compact)
    write_jsonl(rec_path, compact_pairs)
    print(f"  Saved: {rec_path}")

    # Save pairing summary
    summary = {
        "ir_total_scanned": len(ir_catalog),
        "nmr_total_matched": len(nmr_catalog),
        "paired_count": len(pairs),
        "with_1h": has_1h,
        "with_13c": has_13c,
        "with_both": has_both,
        "ir_path": ir_path,
        "nmr_path": nmr_path,
        "limit_ir": limit_ir,
        "limit_nmr": limit_nmr,
    }
    summary_path = os.path.join(args.out_dir, "pairing_summary.json")
    write_json(summary_path, summary)
    print(f"  Saved: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
