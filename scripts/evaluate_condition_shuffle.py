#!/usr/bin/env python
"""Evaluate whether SMILES generation models actually use IR/NMR spectral conditions.

Compares model predictions under PAIRED (correct spectra) vs SHUFFLED (wrong
spectra) conditions to measure condition sensitivity.

Three shuffle modes:
  - shuffle_all:       complete (IR, 1H, 13C) tuple permuted across samples
  - shuffle_nmr_only:  keep IR fixed, permute 1H/13C together
  - shuffle_ir_only:   keep NMR fixed, permute IR only

Usage:
  python scripts/evaluate_condition_shuffle.py \
    --run-dir /path/to/training/run \
    --processed-dir /path/to/processed \
    --split test \
    --shuffle-seed 999

Output:
  {run_dir}/condition_shuffle_summary.json
  {run_dir}/condition_shuffle_predictions.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.dataset import TransCrossSmilesDataset
from src.transcross.collate import smiles_collate_fn
from src.transcross.generation import greedy_decode
from src.transcross.chem_metrics import (
    canonicalize,
    is_valid,
    compute_tanimoto,
    scaffold_match,
    functional_group_f1,
    compute_summary_from_rows,
)

SHUFFLE_MODES = [
    "paired",
    "shuffle_all",
    "shuffle_nmr_only",
    "shuffle_ir_only",
]


# ---------------------------------------------------------------------------
# Device / tokenizer helpers (same as evaluate_smiles_model.py)
# ---------------------------------------------------------------------------


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer_for_eval(processed_dir: str, run_config: dict):
    from src.transcross.smiles_tokenizer import SmilesTokenizer
    from src.transcross.tokenization.spe_tokenizer import SPETokenizer

    tok_type = run_config.get("tokenizer_type", "regex_atom")
    if tok_type == "spe":
        config_content = run_config.get("config_content", {})
        tok_cfg = config_content.get("tokenizer", {})
        vocab_path = tok_cfg.get("vocab_path")
        if not vocab_path:
            vocab_path = os.path.join(processed_dir, "spe_vocab_256.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"SPE vocab not found: {vocab_path}")
        tokenizer = SPETokenizer.load(vocab_path)
    else:
        vocab_path = os.path.join(processed_dir, "smiles_vocab.json")
        tokenizer = SmilesTokenizer.load(vocab_path)
    return tokenizer, tok_type


# ---------------------------------------------------------------------------
# ShuffledDataset wrapper
# ---------------------------------------------------------------------------


class ShuffledDataset(Dataset):
    """Wraps TransCrossSmilesDataset and replaces spectral arrays via a
    pre-computed permutation, keeping SMILES / token IDs from the original
    sample.

    Three shuffle modes are supported:
      paired          — no shuffling (returns original spectra)
      shuffle_all     — permute (IR, 1H, 13C) as a tuple
      shuffle_nmr_only — keep IR fixed, permute (1H, 13C) together
      shuffle_ir_only  — keep (1H, 13C) fixed, permute IR only
    """

    def __init__(
        self,
        base_dataset: Dataset,
        shuffle_perm,
        shuffle_mode: str,
    ):
        self.base = base_dataset
        # shuffle_perm[i] = index of the sample whose spectra to use for position i
        self.shuffle_perm = shuffle_perm
        self.shuffle_mode = shuffle_mode

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        if self.shuffle_perm is not None and self.shuffle_mode != "paired":
            perm_idx = self.shuffle_perm[idx]
            perm_item = self.base[perm_idx]

            if self.shuffle_mode in ("all", "shuffle_all"):
                item["ir"] = perm_item["ir"]
                item["nmr_1h"] = perm_item["nmr_1h"]
                item["nmr_13c"] = perm_item["nmr_13c"]
                item["mask_ir"] = perm_item["mask_ir"]
                item["mask_1h"] = perm_item["mask_1h"]
                item["mask_13c"] = perm_item["mask_13c"]
            elif self.shuffle_mode in ("nmr_only", "shuffle_nmr_only"):
                item["nmr_1h"] = perm_item["nmr_1h"]
                item["nmr_13c"] = perm_item["nmr_13c"]
                item["mask_1h"] = perm_item["mask_1h"]
                item["mask_13c"] = perm_item["mask_13c"]
            elif self.shuffle_mode in ("ir_only", "shuffle_ir_only"):
                item["ir"] = perm_item["ir"]
                item["mask_ir"] = perm_item["mask_ir"]

        return item


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------


def generate_predictions(
    model,
    loader: DataLoader,
    tokenizer,
    max_len: int,
    device: torch.device,
) -> list:
    """Run greedy decode over a DataLoader and return prediction row dicts."""
    rows = []
    for batch in loader:
        ir = batch["ir"].to(device)
        nmr_1h = batch["nmr_1h"].to(device)
        nmr_13c = batch["nmr_13c"].to(device)

        pred_ids = greedy_decode(
            model, ir, nmr_1h, nmr_13c, tokenizer, max_len=max_len,
        )

        for i in range(len(batch["smiles"])):
            target_smi = batch["smiles"][i]
            idx_val = batch["idx"][i]
            pred_smi = tokenizer.decode(pred_ids[i], remove_special=True)

            exact = 1 if pred_smi == target_smi else 0
            valid = 1 if is_valid(pred_smi) else 0
            target_canon = canonicalize(target_smi)
            pred_canon = canonicalize(pred_smi)
            canon_exact = (
                1
                if (pred_canon and target_canon and pred_canon == target_canon)
                else 0
            )
            tanimoto = compute_tanimoto(target_smi, pred_smi)
            scaff = scaffold_match(target_smi, pred_smi)
            fg_p, fg_r, fg_f1 = functional_group_f1(target_smi, pred_smi)

            row = {
                "idx": idx_val,
                "target_smiles": target_smi,
                "pred_smiles": pred_smi,
                "exact_match": exact,
                "canonical_exact_match": canon_exact,
                "rdkit_valid": valid,
                "tanimoto": round(tanimoto, 6),
                "scaffold_match": scaff,
                "fg_precision": round(fg_p, 6),
                "fg_recall": round(fg_r, 6),
                "fg_f1": round(fg_f1, 6),
            }
            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate condition usage by shuffling spectra.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="test",
    )
    parser.add_argument("--shuffle-seed", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # -- Load config -----------------------------------------------------------
    config_path = os.path.join(args.run_dir, "config_used.json")
    checkpoint_path = os.path.join(args.run_dir, "best_model.pt")

    for p in [config_path, checkpoint_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found")
            sys.exit(1)

    with open(config_path) as f:
        run_config = json.load(f)

    model_type = run_config.get("model", "concat")
    yaml_config = run_config.get("config_content", {})
    device = get_device()
    os.makedirs(args.run_dir, exist_ok=True)

    d_model = run_config.get("d_model", 128)
    encoder_layers = run_config.get("encoder_layers", 2)
    decoder_layers = run_config.get("decoder_layers", 2)
    num_heads = run_config.get("num_heads", 4)
    patch_size = run_config.get("patch_size", 64)
    max_smiles_len = run_config.get("max_smiles_len", 160)

    # -- Tokenizer -------------------------------------------------------------
    tokenizer, tok_type = load_tokenizer_for_eval(args.processed_dir, run_config)
    pad_id = tokenizer.pad_id
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {tok_type}, vocab_size={vocab_size}")

    # -- Dataset ---------------------------------------------------------------
    spe_vocab_path = None
    if tok_type == "spe":
        config_content = run_config.get("config_content", {})
        tok_cfg = config_content.get("tokenizer", {})
        spe_vocab_path = tok_cfg.get("vocab_path")

    base_dataset = TransCrossSmilesDataset(
        args.processed_dir,
        split=args.split,
        max_smiles_len=max_smiles_len,
        tokenizer=tokenizer,
        tokenizer_type=tok_type,
        spe_vocab_path=spe_vocab_path,
    )
    n_samples = len(base_dataset)
    if n_samples == 0:
        print("ERROR: dataset is empty")
        sys.exit(1)
    print(f"Dataset: {n_samples} samples, split={args.split}")

    # -- Model -----------------------------------------------------------------
    if yaml_config and model_type in ("concat_equal", "intra_cross_equal"):
        from src.transcross.models.factory import build_smiles_model

        model = build_smiles_model(model_type, yaml_config, vocab_size, pad_id)
    else:
        model_kwargs = dict(
            vocab_size=vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_heads=num_heads,
            patch_size=patch_size,
            dropout=0.1,
            pad_id=pad_id,
            max_smiles_len=max_smiles_len,
        )
        if model_type in ("concat", "concat_equal"):
            from src.transcross.models.smiles_concat import DirectConcatSmilesModel

            model = DirectConcatSmilesModel(**model_kwargs)
        else:
            from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel

            model = IntraCrossSmilesModel(**model_kwargs)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # -- Shuffle permutation (one permutation, reused across all shuffle modes) -
    gen = torch.Generator().manual_seed(args.shuffle_seed)
    full_perm = torch.randperm(n_samples, generator=gen).tolist()
    # Eliminate fixed points: if perm[i] == i, swap with neighbour
    for i in range(n_samples):
        if full_perm[i] == i:
            swap_with = (i + 1) % n_samples
            full_perm[i], full_perm[swap_with] = full_perm[swap_with], full_perm[i]

    # -- Evaluate each mode ----------------------------------------------------
    all_rows_by_mode: dict = {}

    for mode in SHUFFLE_MODES:
        print(f"\n=== Mode: {mode} ===")

        if mode == "paired":
            dataset = base_dataset
        else:
            dataset = ShuffledDataset(base_dataset, full_perm, mode)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: smiles_collate_fn(b, pad_id),
            num_workers=2,
            pin_memory=True,
        )

        rows = generate_predictions(
            model, loader, tokenizer, max_smiles_len, device,
        )
        all_rows_by_mode[mode] = rows
        print(f"  Generated {len(rows)} predictions")

    # -- Save combined predictions CSV -----------------------------------------
    csv_path = os.path.join(args.run_dir, "condition_shuffle_predictions.csv")
    csv_fields = [
        "idx",
        "target_smiles",
        "mode",
        "pred_smiles",
        "rdkit_valid",
        "tanimoto",
        "scaffold_match",
        "fg_f1",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for mode, rows in all_rows_by_mode.items():
            for r in rows:
                writer.writerow(
                    {
                        "idx": r["idx"],
                        "target_smiles": r["target_smiles"],
                        "mode": mode,
                        "pred_smiles": r["pred_smiles"],
                        "rdkit_valid": r["rdkit_valid"],
                        "tanimoto": r["tanimoto"],
                        "scaffold_match": r["scaffold_match"],
                        "fg_f1": r["fg_f1"],
                    }
                )
    print(f"\nSaved predictions to {csv_path}")

    # -- Summaries per mode ----------------------------------------------------
    summaries = {}
    for mode, rows in all_rows_by_mode.items():
        summaries[mode] = compute_summary_from_rows(rows)

    # -- Build comparison JSON -------------------------------------------------
    METRIC_KEYS = (
        "rdkit_validity",
        "unique_generated",
        "unique_ratio",
        "canonical_exact_match",
        "mean_tanimoto",
        "mean_tanimoto_valid_only",
        "scaffold_match_rate",
        "mean_fg_f1",
        "mode_collapse_score",
        "prediction_entropy",
        "avg_pred_char_length",
        "mean_levenshtein",
    )

    comparison = {
        "run_dir": args.run_dir,
        "split": args.split,
        "shuffle_seed": args.shuffle_seed,
        "num_samples": n_samples,
        "modes": {},
    }

    for mode in SHUFFLE_MODES:
        s = summaries[mode]
        entry = {}
        for key in METRIC_KEYS:
            entry[key] = s.get(key, 0)
        comparison["modes"][mode] = entry

    # Paired-vs-shuffled deltas
    paired_s = summaries["paired"]
    deltas = {}
    for mode in ["shuffle_all", "shuffle_nmr_only", "shuffle_ir_only"]:
        mode_s = summaries[mode]
        md = {}
        for key in [
            "rdkit_validity",
            "unique_ratio",
            "canonical_exact_match",
            "mean_tanimoto",
            "scaffold_match_rate",
            "mean_fg_f1",
        ]:
            md[f"{key}_drop"] = round(float(paired_s.get(key, 0)) - float(mode_s.get(key, 0)), 6)
        md["mode_collapse_score_change"] = round(
            float(mode_s.get("mode_collapse_score", 0))
            - float(paired_s.get("mode_collapse_score", 0)),
            6,
        )
        md["prediction_entropy_change"] = round(
            float(mode_s.get("prediction_entropy", 0))
            - float(paired_s.get("prediction_entropy", 0)),
            6,
        )
        deltas[mode] = md
    comparison["paired_vs_shuffled_deltas"] = deltas

    # Interpretation guidance
    comparison["interpretation_notes"] = [
        "If paired metrics are approximately equal to shuffled metrics, "
        "the model is NOT strongly using spectral conditions.",
        "A larger paired-vs-shuffled metric drop indicates greater "
        "condition sensitivity.",
        "If mode collapse persists under both paired and shuffled "
        "conditions, the model may be decoder-prior dominated.",
    ]

    summary_path = os.path.join(args.run_dir, "condition_shuffle_summary.json")
    with open(summary_path, "w") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")

    # -- Print results table ---------------------------------------------------
    print()
    print("=" * 90)
    print(f"  Condition Shuffle Evaluation  |  split={args.split}  "
          f"seed={args.shuffle_seed}  num_samples={n_samples}")
    print("=" * 90)

    header = (
        f"{'Metric':<30} {'Paired':<14} {'Shuf All':<14} "
        f"{'Shuf NMR':<14} {'Shuf IR':<14}"
    )
    print(header)
    print("-" * len(header))

    display_keys = [
        ("rdkit_validity", "RDKit Validity"),
        ("unique_ratio", "Unique Ratio"),
        ("canonical_exact_match", "Canon Exact Match"),
        ("mean_tanimoto", "Mean Tanimoto"),
        ("scaffold_match_rate", "Scaffold Match"),
        ("mean_fg_f1", "Mean FG-F1"),
        ("avg_pred_char_length", "Avg Pred Length"),
        ("mode_collapse_score", "Mode Collapse"),
        ("prediction_entropy", "Prediction Entropy"),
        ("unique_generated", "Unique Generated"),
        ("mean_levenshtein", "Mean Levenshtein"),
    ]

    for key, label in display_keys:
        vals = [comparison["modes"][m].get(key, 0) for m in SHUFFLE_MODES]
        if isinstance(vals[0], float):
            vals_str = [f"{v:<14.4f}" for v in vals]
        else:
            vals_str = [str(v).ljust(14) for v in vals]
        print(f"{label:<30} {' '.join(vals_str)}")

    print()
    print("Deltas (paired minus shuffled, positive = paired better):")
    for mode in ["shuffle_all", "shuffle_nmr_only", "shuffle_ir_only"]:
        d = deltas[mode]
        print(f"  {mode}:")
        for k, v in d.items():
            print(f"    {k}: {v:+.6f}")

    print()
    print("Interpretation notes:")
    for note in comparison["interpretation_notes"]:
        print(f"  - {note}")


if __name__ == "__main__":
    main()
