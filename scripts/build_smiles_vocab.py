"""Build SMILES vocabulary from processed SMILES corpus and save to JSON."""

import argparse
import json
import os
import sys

# Add project root to path for local runs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.smiles_tokenizer import SmilesTokenizer


def main():
    parser = argparse.ArgumentParser(description="Build SMILES vocabulary.")
    parser.add_argument(
        "--processed-dir", required=True,
        help="Path to processed data directory containing canonical_smiles.txt"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output path for smiles_vocab.json"
    )
    args = parser.parse_args()

    smiles_path = os.path.join(args.processed_dir, "canonical_smiles.txt")
    if not os.path.exists(smiles_path):
        print(f"ERROR: canonical_smiles.txt not found at {smiles_path}")
        sys.exit(1)

    with open(smiles_path) as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(smiles_list)} SMILES strings.")

    tokenizer = SmilesTokenizer.build_from_smiles(smiles_list)
    tokenizer.save(args.out)
    print(f"Vocabulary saved to {args.out}")
    print(f"  vocab_size: {tokenizer.vocab_size}")

    # Compute statistics
    all_lens = []
    unk_count = 0
    total_tokens = 0
    for smi in smiles_list:
        ids = tokenizer.encode(smi, add_bos=False, add_eos=False)
        all_lens.append(len(ids))
        total_tokens += len(ids)
        unk_count += sum(1 for tid in ids if tid == tokenizer.unk_id)

    all_lens.sort()
    n = len(all_lens)
    summary = {
        "vocab_size": tokenizer.vocab_size,
        "num_smiles": len(smiles_list),
        "max_token_length": max(all_lens),
        "mean_token_length": sum(all_lens) / n,
        "p90_token_length": all_lens[int(n * 0.90)],
        "p95_token_length": all_lens[int(n * 0.95)],
        "unk_token_count": unk_count,
        "unk_token_frac": unk_count / total_tokens if total_tokens > 0 else 0.0,
    }

    summary_path = os.path.join(
        os.path.dirname(args.out), "smiles_vocab_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
