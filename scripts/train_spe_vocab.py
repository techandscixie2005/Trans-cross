#!/usr/bin/env python
"""Train SPE vocabulary for SMILES generation ablation.

Matches the spec-required CLI:
  python scripts/train_spe_vocab.py \\
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \\
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \\
    --vocab-size 1000 \\
    --min-frequency 2 \\
    --max-len 120

Only trains on the training split to avoid data leakage.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_spe_vocab import compute_stats, percentile


def main():
    parser = argparse.ArgumentParser(description="Train SPE vocabulary.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=120)
    parser.add_argument("--prefix", default="spe", help="Prefix for output filenames")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.prefix

    # Build SPE vocab using the core builder
    from src.transcross.tokenization.spe_tokenizer import SPETokenizer
    from src.transcross.tokenization.atom_tokenizer import atom_tokenize

    # Load SMILES and splits
    smiles_path = os.path.join(args.processed_dir, "canonical_smiles.txt")
    with open(smiles_path) as f:
        all_smiles = [line.strip() for line in f if line.strip()]

    splits_path = os.path.join(args.processed_dir, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    train_indices = splits["train"]
    train_smiles = [all_smiles[i] for i in train_indices]
    print(f"Training SMILES: {len(train_smiles)}")

    # Train SPE
    tokenizer = SPETokenizer()
    n_merges = tokenizer.train(
        train_smiles,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(f"SPE vocab_size: {tokenizer.vocab_size}, merges: {n_merges}")

    # Save
    vocab_path = os.path.join(args.out_dir, f"{prefix}_vocab.json")
    merges_path = os.path.join(args.out_dir, f"{prefix}_merges.txt")
    config_path = os.path.join(args.out_dir, f"{prefix}_tokenizer_config.json")
    summary_path = os.path.join(args.out_dir, f"{prefix}_tokenization_summary.json")

    tokenizer.save(vocab_path)
    print(f"Saved vocab: {vocab_path}")

    # Save merge rules as text
    with open(merges_path, "w") as f:
        for a, b in tokenizer._merges:
            f.write(f"{a} + {b} -> {a + b}\n")
    print(f"Saved merges: {merges_path}")

    # Save tokenizer config
    config = {
        "type": "spe",
        "vocab_size": tokenizer.vocab_size,
        "num_merges": n_merges,
        "target_vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "max_target_len": args.max_len,
        "special_tokens": {
            "pad": tokenizer.pad_id,
            "bos": tokenizer.bos_id,
            "eos": tokenizer.eos_id,
            "unk": tokenizer.unk_id,
        },
    }
    with open(config_path, "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Saved config: {config_path}")

    # Audit all splits
    attr = {}
    for sn, indices in splits.items():
        split_smiles = [all_smiles[i] for i in indices]
        lens = []
        dropped = 0
        kept = 0
        for smi in split_smiles:
            toks = tokenizer.tokenize(smi)
            lens.append(len(toks))
            if len(toks) > args.max_len:
                dropped += 1
            else:
                kept += 1
        attr[sn] = {
            "n_total": len(split_smiles),
            "n_kept": kept,
            "n_dropped": dropped,
            "dropped_pct": round(100 * dropped / len(split_smiles), 2) if split_smiles else 0,
        }
        print(f"{sn}: {attr[sn]}")

    # Compute overall stats
    all_lens = []
    for sn, indices in splits.items():
        split_smiles = [all_smiles[i] for i in indices]
        for smi in split_smiles:
            all_lens.append(len(tokenizer.tokenize(smi)))

    summary = {
        "vocab_size": tokenizer.vocab_size,
        "num_merges": n_merges,
        "target_vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "max_target_len": args.max_len,
        "length_stats": {
            "mean": round(sum(all_lens) / len(all_lens), 2) if all_lens else 0,
            "median": sorted(all_lens)[len(all_lens) // 2] if all_lens else 0,
            "p90": int(percentile(all_lens, 90)),
            "p95": int(percentile(all_lens, 95)),
            "max": max(all_lens) if all_lens else 0,
        },
        "split_stats": attr,
        "merge_rules": [{"pair": list(p), "merged": p[0] + p[1]} for p in tokenizer._merges],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")


def percentile(data, p):
    if not data:
        return 0
    import numpy as np
    return float(np.percentile(data, p))


if __name__ == "__main__":
    main()
