#!/usr/bin/env python
"""Compare model parameter counts for E0 and E1 equal-parameter models.

Supports both regex_atom and SPE tokenizers.
Instantiates both models from config, prints parameter tables, and checks
that the relative parameter difference is within the allowed tolerance.

Usage (SPE):
  python scripts/compare_model_params.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --vocab /data/home/sczc698/run/xxy/Trans-cross/data/processed/spe_vocab_256.json \
    --config configs/smiles_spe_equal_param.yaml

Usage (regex_atom):
  python scripts/compare_model_params.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --vocab /data/home/sczc698/run/xxy/Trans-cross/data/processed/smiles_vocab.json \
    --config configs/smiles_equal_param.yaml

Exit code is nonzero if relative parameter difference > 1.0%.
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.models.factory import build_smiles_model
from src.transcross.model_utils import (
    count_trainable_parameters,
    count_parameters_by_module,
    format_parameter_table,
    compare_models,
)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(vocab_path: str, config: dict):
    """Auto-detect tokenizer type from path or config."""
    tok_cfg = config.get("tokenizer", {})
    tok_type = tok_cfg.get("type", "regex_atom")

    if tok_type == "spe" or "spe_vocab" in os.path.basename(vocab_path):
        tokenizer = SPETokenizer.load(vocab_path)
        print(f"Loaded SPE tokenizer from {vocab_path}")
    else:
        tokenizer = SmilesTokenizer.load(vocab_path)
        print(f"Loaded regex_atom tokenizer from {vocab_path}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Compare parameter counts of E0 and E1 models."
    )
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--vocab", required=True, help="Path to vocab JSON")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    tokenizer = load_tokenizer(args.vocab, config)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    print(f"Vocab size: {vocab_size}, pad_id: {pad_id}")

    # Build E0 (concat_equal)
    print("\n" + "=" * 60)
    print("Building E0: DirectConcat (concat_equal)")
    print("=" * 60)
    model_e0 = build_smiles_model("concat_equal", config, vocab_size, pad_id)
    print(format_parameter_table(model_e0))
    e0_by_module = count_parameters_by_module(model_e0)
    e0_total = count_trainable_parameters(model_e0)

    # Build E1 (intra_cross_equal)
    print("\n" + "=" * 60)
    print("Building E1: IntraCross (intra_cross_equal)")
    print("=" * 60)
    model_e1 = build_smiles_model("intra_cross_equal", config, vocab_size, pad_id)
    print(format_parameter_table(model_e1))
    e1_by_module = count_parameters_by_module(model_e1)
    e1_total = count_trainable_parameters(model_e1)

    # Compare
    result, within = compare_models(
        model_e0, model_e1,
        max_relative_diff=config.get("equality_constraint", {}).get(
            "max_relative_param_diff", 0.01
        ),
    )

    max_diff_pct = config.get("equality_constraint", {}).get("max_relative_param_diff", 0.01) * 100

    print("\n" + "=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"E0 (concat_equal) total params:     {result['e0_total']:>12,}")
    print(f"E1 (intra_cross_equal) total params: {result['e1_total']:>12,}")
    print(f"Absolute difference:                 {result['abs_diff']:>12,}")
    print(f"Relative difference:                 {result['rel_diff_pct']:>10.4f}%")
    print(f"Within tolerance (<= {max_diff_pct}%): {within}")

    # Verify decoder equality
    print("\n" + "=" * 60)
    print("Decoder Component Check")
    print("=" * 60)
    e0_dec_params = e0_by_module.get("decoder", 0)
    e1_dec_params = e1_by_module.get("decoder", 0)
    print(f"E0 decoder params: {e0_dec_params:,}")
    print(f"E1 decoder params: {e1_dec_params:,}")
    dec_match = e0_dec_params == e1_dec_params
    print(f"Decoder params identical: {dec_match}")

    # Detailed component breakdown
    print("\n" + "=" * 60)
    print("Component Breakdown")
    print("=" * 60)
    all_keys = sorted(set(e0_by_module.keys()) | set(e1_by_module.keys()))
    print(f"{'Component':<30s} {'E0':>12s} {'E1':>12s}")
    print("-" * 56)
    for key in all_keys:
        e0v = e0_by_module.get(key, 0)
        e1v = e1_by_module.get(key, 0)
        print(f"{key:<30s} {e0v:>12,} {e1v:>12,}")

    if not dec_match:
        print("ERROR: Decoder parameters are not identical!")
        sys.exit(1)

    if not within:
        print(
            f"\nERROR: Parameter difference {result['rel_diff_pct']:.4f}% "
            f"exceeds maximum {max_diff_pct}%"
        )
        sys.exit(1)

    print("\nPASS: E0 and E1 have matched parameter counts within tolerance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
