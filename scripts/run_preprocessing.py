#!/usr/bin/env python3
"""Run the full preprocessing pipeline in sequence.

Smoke mode:
  python scripts/run_preprocessing.py \
    --config configs/preprocessing.yaml \
    --raw-dir /data/home/sczc698/run/xxy/Trans-cross/ \
    --out-dir data/processed_smoke \
    --smoke \
    --limit-ir 200 \
    --limit-nmr 500000 \
    --limit-preprocess 100

Full mode:
  python scripts/run_preprocessing.py \
    --config configs/preprocessing.yaml \
    --raw-dir /data/home/sczc698/run/xxy/Trans-cross/ \
    --out-dir data/processed
"""

import argparse
import os
import subprocess
import sys


def run_step(cmd: list[str], step_name: str) -> int:
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {step_name} failed with exit code {result.returncode}")
        return result.returncode
    print(f"\n{step_name} completed successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to preprocessing.yaml")
    parser.add_argument("--raw-dir", required=True, help="Directory containing raw JSONL files")
    parser.add_argument("--out-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--limit-ir", type=int, default=None, help="Max IR records")
    parser.add_argument("--limit-nmr", type=int, default=None, help="Max NMR records")
    parser.add_argument("--limit-preprocess", type=int, default=None, help="Max pairs to preprocess")
    args = parser.parse_args()

    script_dir = os.path.join(os.path.dirname(__file__))
    build_pairs_script = os.path.join(script_dir, "build_pairs.py")
    preprocess_script = os.path.join(script_dir, "preprocess_spectra.py")
    split_script = os.path.join(script_dir, "split_data.py")

    os.makedirs(args.out_dir, exist_ok=True)

    # Step 1: Build pairs
    build_cmd = [
        sys.executable, build_pairs_script,
        "--config", args.config,
        "--raw-dir", args.raw_dir,
        "--out-dir", args.out_dir,
    ]
    if args.limit_ir is not None:
        build_cmd += ["--limit-ir", str(args.limit_ir)]
    if args.limit_nmr is not None:
        build_cmd += ["--limit-nmr", str(args.limit_nmr)]

    ret = run_step(build_cmd, "build_pairs")
    if ret != 0:
        sys.exit(ret)

    # Step 2: Preprocess spectra
    pairs_path = os.path.join(args.out_dir, "paired_records.jsonl")
    preprocess_cmd = [
        sys.executable, preprocess_script,
        "--config", args.config,
        "--pairs", pairs_path,
        "--out-dir", args.out_dir,
    ]
    if args.limit_preprocess is not None:
        preprocess_cmd += ["--limit", str(args.limit_preprocess)]

    ret = run_step(preprocess_cmd, "preprocess_spectra")
    if ret != 0:
        sys.exit(ret)

    # Step 3: Split data
    split_cmd = [
        sys.executable, split_script,
        "--config", args.config,
        "--pairs", os.path.join(args.out_dir, "pairs.csv"),
        "--out-dir", args.out_dir,
    ]

    ret = run_step(split_cmd, "split_data")
    if ret != 0:
        sys.exit(ret)

    print(f"\n{'=' * 60}")
    print("All preprocessing steps completed successfully!")
    print(f"Output directory: {args.out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
