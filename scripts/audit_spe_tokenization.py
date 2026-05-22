#!/usr/bin/env python
"""Audit SPE tokenization quality and statistics.

Usage:
  python scripts/audit_spe_tokenization.py \\
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \\
    --tokenizer /data/home/sczc698/run/xxy/Trans-cross/data/processed/spe_vocab.json \\
    --out reports/spe_tokenization_audit.md
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.tokenization.atom_tokenizer import atom_tokenize


def percentile(data, p):
    if not data:
        return 0
    import numpy as np
    return float(np.percentile(data, p))


def main():
    parser = argparse.ArgumentParser(description="Audit SPE tokenization.")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--tokenizer", required=True, help="Path to spe_vocab JSON")
    parser.add_argument("--out", required=True, help="Output markdown report path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Load data
    smiles_path = os.path.join(args.processed_dir, "canonical_smiles.txt")
    with open(smiles_path) as f:
        all_smiles = [line.strip() for line in f if line.strip()]

    splits_path = os.path.join(args.processed_dir, "splits.json")
    with open(splits_path) as f:
        splits = json.load(f)

    # Load tokenizer
    tokenizer = SPETokenizer.load(args.tokenizer)

    lines = []
    def w(s=""):
        lines.append(s)

    w("# SPE Tokenization Audit")
    w()
    w(f"**Tokenizer path**: `{args.tokenizer}`")
    w(f"**Processed dir**: `{args.processed_dir}`")
    w()

    # Vocab info
    w("## 1. SPE Vocabulary")
    w()
    w(f"- **Vocab size**: {tokenizer.vocab_size}")
    w(f"- **Number of merges**: {tokenizer.num_merges}")
    w(f"- **Special tokens**: pad={tokenizer.pad_id}, bos={tokenizer.bos_id}, eos={tokenizer.eos_id}, unk={tokenizer.unk_id}")
    w()

    # Split-level statistics
    w("## 2. Token Length Statistics")
    w()
    w("| Split | N | Mean | Median | P90 | P95 | Max | Dropped |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|")

    for split_name in ["train", "valid", "test"]:
        indices = splits[split_name]
        split_smiles = [all_smiles[i] for i in indices]
        spe_lens = []
        atom_lens = []
        dropped = 0
        for smi in split_smiles:
            s_toks = tokenizer.tokenize(smi)
            a_toks = atom_tokenize(smi)
            spe_lens.append(len(s_toks))
            atom_lens.append(len(a_toks))
            if len(s_toks) > 120:
                dropped += 1

        mean_spe = sum(spe_lens) / len(spe_lens)
        med_spe = sorted(spe_lens)[len(spe_lens) // 2]
        p90 = int(percentile(spe_lens, 90))
        p95 = int(percentile(spe_lens, 95))
        max_spe = max(spe_lens)

        w(f"| {split_name} | {len(split_smiles)} | "
          f"{mean_spe:.1f} | {med_spe} | {p90} | {p95} | {max_spe} | {dropped} |")

    w()
    w("### Atom-Level Reference (NOT a model condition)")
    w()
    w("| Split | N | Mean atom length | Median | P90 | P95 | Max |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    for split_name in ["train", "valid", "test"]:
        indices = splits[split_name]
        split_smiles = [all_smiles[i] for i in indices]
        a_lens = [len(atom_tokenize(smi)) for smi in split_smiles]
        mean_a = sum(a_lens) / len(a_lens)
        med_a = sorted(a_lens)[len(a_lens) // 2]
        p90_a = int(percentile(a_lens, 90))
        p95_a = int(percentile(a_lens, 95))
        max_a = max(a_lens)
        w(f"| {split_name} | {len(split_smiles)} | "
          f"{mean_a:.1f} | {med_a} | {p90_a} | {p95_a} | {max_a} |")

    w()
    w("## 3. Dropped Samples by Split")
    w()
    w("| Split | Total | Dropped (>120 tokens) | Drop % |")
    w("|---|---:|---:|---:|")
    for split_name in ["train", "valid", "test"]:
        indices = splits[split_name]
        split_smiles = [all_smiles[i] for i in indices]
        dropped = sum(1 for smi in split_smiles if len(tokenizer.tokenize(smi)) > 120)
        drop_pct = 100 * dropped / len(split_smiles) if split_smiles else 0
        w(f"| {split_name} | {len(split_smiles)} | {dropped} | {drop_pct:.2f}% |")

    w()
    w("## 4. Examples of SPE-Tokenized SMILES")
    w()
    w("| SMILES | Atom tokens | SPE tokens | SPE len |")
    w("|---|---:|---:|")
    train_indices = splits["train"]
    for i in range(min(15, len(train_indices))):
        smi = all_smiles[train_indices[i]]
        a_toks = atom_tokenize(smi)
        s_toks = tokenizer.tokenize(smi)
        w(f"| `{smi}` | `{' '.join(a_toks[:10])}...` ({len(a_toks)}) | "
          f"`{' '.join(s_toks[:10])}...` ({len(s_toks)}) | {len(s_toks)} |")

    w()
    w("## 5. Validation and Unknown Token Rates")
    w()
    w("| Split | Unknown tokens | Total tokens | UNK rate |")
    w("|---|---:|---:|---:|")
    for split_name in ["train", "valid", "test"]:
        indices = splits[split_name]
        split_smiles = [all_smiles[i] for i in indices]
        total_unk = 0
        total_tok = 0
        for smi in split_smiles:
            for t in tokenizer.tokenize(smi):
                if t not in tokenizer._token_to_id:
                    total_unk += 1
                total_tok += 1
        rate = total_unk / max(total_tok, 1)
        w(f"| {split_name} | {total_unk} | {total_tok} | {rate:.6f} |")

    w()
    w("---")
    w("*Atom-level tokenization is NOT a model condition — shown for reference only.*")

    with open(args.out, "w") as f:
        f.write("\n".join(lines))
    print(f"Audit report saved to {args.out}")


if __name__ == "__main__":
    main()
