#!/usr/bin/env python
"""Run the full SPE SMILES generation ablation experiment.

Orchestrates training of both E0 (Direct Concat) and E1 (Intra-Cross) models,
then runs evaluation. Supports smoke mode for quick testing.

Smoke mode:
  python scripts/run_spe_smiles_ablation.py \\
    --processed-dir ... --out-root ... --smoke \\
    --epochs 1 --batch-size 8 --d-model 64 --num-heads 4 \\
    --limit-train-batches 5 --limit-eval-batches 2

Full mode:
  python scripts/run_spe_smiles_ablation.py \\
    --processed-dir ... --out-root ... \\
    --epochs 50 --batch-size 32 --d-model 128 --num-heads 4 \\
    --ffn-dim 512 --patch-size 64 --lr 1e-4 \\
    --warmup-steps 500 --grad-clip 1.0 \\
    --label-smoothing 0.05 --beam-size 5 --seed 42
"""

import argparse
import json
import os
import subprocess
import sys
import time


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"FAILED after {elapsed:.0f}s: {description}")
        return False
    print(f"DONE in {elapsed:.0f}s: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run SPE SMILES ablation.")
    # Paths
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-root", required=True)

    # Mode
    parser.add_argument("--smoke", action="store_true", default=False)

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=512)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    # Config file option
    parser.add_argument("--config", default=None, help="YAML config file path")

    # Limits
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-eval-batches", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # Find train script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_smiles_ablation.py")

    # Determine config
    if args.config:
        config_flag = ["--config", args.config]
    else:
        # Write a temporary config
        tmp_config = os.path.join(args.out_root, "run_config.yaml")
        import yaml
        config = {
            "data": {
                "processed_dir": args.processed_dir,
                "max_smiles_len": 120,
            },
            "tokenizer": {
                "type": "spe",
                "vocab_path": os.path.join(args.processed_dir, "spe_vocab.json"),
                "target_vocab_size": 1000,
                "min_frequency": 2,
                "patch_size": args.patch_size,
                "use_modality_embedding": True,
                "use_absolute_position_embedding": True,
            },
            "shared": {
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "decoder_layers": 2,
                "decoder_ffn_dim": args.ffn_dim,
                "dropout": 0.1,
            },
            "e0_concat": {
                "encoder_layers": 4,
                "encoder_ffn_dim": args.ffn_dim,
            },
            "e1_intra_cross": {
                "intra_layers": 1,
                "cross_layers": 1,
                "fusion_layers": 0,
                "encoder_ffn_dim": args.ffn_dim,
            },
            "training": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": 1e-4,
                "warmup_steps": args.warmup_steps,
                "grad_clip": args.grad_clip,
                "label_smoothing": args.label_smoothing,
                "seed": args.seed,
                "beam_size": args.beam_size,
            },
        }
        with open(tmp_config, "w") as f:
            yaml.dump(config, f)
        config_flag = ["--config", tmp_config]

    common_args = [
        "--processed-dir", args.processed_dir,
    ]

    # Train E0
    e0_out = os.path.join(args.out_root, "e0_direct_concat")
    e0_cmd = [sys.executable, train_script] + config_flag + [
        "--model", "concat_equal",
        "--out-dir", e0_out,
    ]
    if not run_command(e0_cmd, "Train E0 DirectConcat"):
        return 1

    # Train E1
    e1_out = os.path.join(args.out_root, "e1_intra_cross")
    e1_cmd = [sys.executable, train_script] + config_flag + [
        "--model", "intra_cross_equal",
        "--out-dir", e1_out,
    ]
    if not run_command(e1_cmd, "Train E1 IntraCross"):
        return 1

    # Run evaluation
    eval_script = os.path.join(script_dir, "evaluate_smiles_model.py")
    if os.path.exists(eval_script):
        for model_name, run_dir in [("E0 DirectConcat", e0_out), ("E1 IntraCross", e1_out)]:
            eval_cmd = [
                sys.executable, eval_script,
                "--model-dir", run_dir,
                "--processed-dir", args.processed_dir,
                "--out-dir", run_dir,
            ]
            run_command(eval_cmd, f"Evaluate {model_name}")

    print(f"\n{'='*60}")
    print(f"Ablation complete. Outputs in: {args.out_root}")
    print(f"  E0 (Direct Concat): {e0_out}")
    print(f"  E1 (Intra Cross):   {e1_out}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
