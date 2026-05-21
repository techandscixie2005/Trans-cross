#!/usr/bin/env python
"""Build SPE (SMILES Pair Encoding) vocabulary from processed data.

Trains SPE only on the training split to avoid data leakage.
Computes token length statistics and audits unknown token rates.

Usage:
  python scripts/build_spe_vocab.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --out /data/home/sczc698/run/xxy/Trans-cross/data/processed/spe_vocab_256.json \
    --vocab-size 256 \
    --min-frequency 2 \
    --split train
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.tokenization.atom_tokenizer import atom_tokenize


def percentile(data, p):
    """Compute the p-th percentile of a list (p in 0..100)."""
    if not data:
        return 0
    return float(np.percentile(data, p))


def compute_stats(lengths):
    """Compute summary statistics for a list of sequence lengths."""
    if not lengths:
        return {"mean": 0, "p50": 0, "p90": 0, "p95": 0, "max": 0}
    return {
        "mean": round(float(np.mean(lengths)), 2),
        "p50": round(percentile(lengths, 50), 2),
        "p90": round(percentile(lengths, 90), 2),
        "p95": round(percentile(lengths, 95), 2),
        "max": int(np.max(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description="Build SPE vocabulary.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out", required=True, help="Output path for spe_vocab JSON")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--split", default="train", help="Which split to train on")
    args = parser.parse_args()

    # Load SMILES and splits
    smiles_path = os.path.join(args.processed_dir, "canonical_smiles.txt")
    with open(smiles_path) as f:
        all_smiles = [line.strip() for line in f if line.strip()]
    print(f"Total SMILES: {len(all_smiles)}")

    splits_path = os.path.join(args.processed_dir, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)
    print(f"Splits: { {k: len(v) for k, v in splits.items()} }")

    train_indices = splits.get(args.split, [])
    train_smiles = [all_smiles[i] for i in train_indices]
    print(f"Training SMILES ({args.split}): {len(train_smiles)}")

    # Train SPE only on training split
    tokenizer = SPETokenizer()
    n_merges = tokenizer.train(
        train_smiles,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Number of merges: {n_merges}")

    # Audit each split
    atom_lengths = {}
    spe_lengths = {}
    unk_rates = {}

    for split_name, indices in splits.items():
        split_smiles = [all_smiles[i] for i in indices]

        a_lens = []
        s_lens = []
        total_unk = 0
        total_tokens = 0

        for smi in split_smiles:
            a_tokens = atom_tokenize(smi)
            a_lens.append(len(a_tokens))

            s_tokens = tokenizer.tokenize(smi)
            s_lens.append(len(s_tokens))

            for t in s_tokens:
                if t not in tokenizer._token_to_id:
                    total_unk += 1
                total_tokens += 1

        atom_lengths[split_name] = compute_stats(a_lens)
        spe_lengths[split_name] = compute_stats(s_lens)
        unk_rates[split_name] = round(total_unk / max(total_tokens, 1), 6)

        print(f"\n{split_name}:")
        print(f"  atom lengths: {atom_lengths[split_name]}")
        print(f"  SPE lengths:  {spe_lengths[split_name]}")
        print(f"  unk rate:     {unk_rates[split_name]}")

    # Length reduction ratio (train)
    train_atom_mean = atom_lengths[args.split]["mean"]
    train_spe_mean = spe_lengths[args.split]["mean"]
    reduction = (
        round((1 - train_spe_mean / train_atom_mean) * 100, 1)
        if train_atom_mean > 0
        else 0
    )
    print(f"\nLength reduction (train): {reduction}%")

    # Example tokenizations
    examples = []
    for i in range(min(10, len(train_smiles))):
        smi = train_smiles[i]
        a_toks = atom_tokenize(smi)
        s_toks = tokenizer.tokenize(smi)
        examples.append({
            "smiles": smi,
            "atom_tokens": a_toks,
            "atom_len": len(a_toks),
            "spe_tokens": s_toks,
            "spe_len": len(s_toks),
        })

    # Token frequency counts on training set
    token_freq: dict = {}
    for smi in train_smiles:
        for tok in tokenizer.tokenize(smi):
            token_freq[tok] = token_freq.get(tok, 0) + 1

    top_tokens = sorted(token_freq.items(), key=lambda x: -x[1])[:50]

    # Save vocabulary
    tokenizer.save(args.out)
    print(f"\nSaved SPE vocab to {args.out}")

    # Save summary
    summary_path = args.out.replace(".json", "_summary.json")
    summary = {
        "vocab_size": tokenizer.vocab_size,
        "num_merges": n_merges,
        "target_vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "train_split": args.split,
        "unk_rates": unk_rates,
        "atom_length_stats": atom_lengths,
        "spe_length_stats": spe_lengths,
        "length_reduction_pct": reduction,
        "examples": examples,
        "top_50_spe_tokens": [
            {"token": tok, "frequency": freq}
            for tok, freq in top_tokens
        ],
        "merge_rules": tokenizer.get_merge_rules(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
