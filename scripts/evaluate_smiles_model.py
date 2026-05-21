"""Evaluate a trained SMILES generation model.

Supports two modes:
1. Run-dir mode (preferred): --run-dir to load config + checkpoint automatically
2. Legacy mode: --checkpoint + --model + --d-model etc.

Usage (run-dir mode):
  python scripts/evaluate_smiles_model.py \
    --processed-dir /path/to/processed \
    --run-dir /path/to/training/run \
    --split valid \
    --save-predictions

Usage (legacy mode):
  python scripts/evaluate_smiles_model.py \
    --processed-dir /path/to/processed \
    --checkpoint /path/to/best_model.pt \
    --model concat \
    --d-model 128 \
    --encoder-layers 6 \
    --decoder-layers 2 \
    --num-heads 4 \
    --out-dir /path/to/output
"""

import argparse
import json
import os
import sys
import csv
import yaml

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.dataset import TransCrossSmilesDataset
from src.transcross.collate import smiles_collate_fn
from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
from src.transcross.models.factory import build_smiles_model
from src.transcross.generation import greedy_decode

try:
    from rdkit import Chem
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def canonicalize(smi):
    """RDKit canonicalize if available, else return input."""
    if not _HAS_RDKIT or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid(smi):
    """Check RDKit validity."""
    if not _HAS_RDKIT or not smi:
        return False
    mol = Chem.MolFromSmiles(smi)
    return mol is not None


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMILES generation model.")
    # Run-dir mode (preferred)
    parser.add_argument("--run-dir", default=None,
                       help="Path to training run directory (loads config + checkpoint)")
    # Legacy mode
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model", default="concat",
                       choices=["concat", "intra_cross", "concat_equal", "intra_cross_equal"])
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--max-smiles-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    # Common
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--save-predictions", action="store_true", default=False)
    args = parser.parse_args()

    # Resolve run-dir vs legacy mode
    if args.run_dir:
        # Run-dir mode: load config + checkpoint from run directory
        config_path = os.path.join(args.run_dir, "config_used.json")
        checkpoint_path = os.path.join(args.run_dir, "best_model.pt")
        if not os.path.exists(config_path):
            print(f"ERROR: config_used.json not found in {args.run_dir}")
            sys.exit(1)
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: best_model.pt not found in {args.run_dir}")
            sys.exit(1)

        with open(config_path) as f:
            run_config = json.load(f)
        model_type = run_config.get("model", "concat")
        # If config_content is present (YAML mode), use it
        yaml_config = run_config.get("config_content", None)
        args.checkpoint = checkpoint_path
        args.model = model_type
        out_dir = args.out_dir or args.run_dir

        # Resolve model params
        d_model = run_config.get("d_model", args.d_model or 128)
        encoder_layers = run_config.get("encoder_layers", args.encoder_layers or 2)
        decoder_layers = run_config.get("decoder_layers", args.decoder_layers or 2)
        num_heads = run_config.get("num_heads", args.num_heads or 4)
        patch_size = run_config.get("patch_size", args.patch_size or 64)
        max_smiles_len = run_config.get("max_smiles_len", args.max_smiles_len or 160)
    else:
        # Legacy mode: use CLI args
        if not args.checkpoint:
            parser.error("Either --run-dir or --checkpoint is required")
        out_dir = args.out_dir or os.path.dirname(args.checkpoint)
        yaml_config = None
        d_model = args.d_model or 128
        encoder_layers = args.encoder_layers or 2
        decoder_layers = args.decoder_layers or 2
        num_heads = args.num_heads or 4
        patch_size = args.patch_size or 64
        max_smiles_len = args.max_smiles_len or 160

    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # Load tokenizer
    vocab_path = os.path.join(args.processed_dir, "smiles_vocab.json")
    tokenizer = SmilesTokenizer.load(vocab_path)
    pad_id = tokenizer.pad_id

    # Dataset
    dataset = TransCrossSmilesDataset(
        args.processed_dir, split=args.split,
        max_smiles_len=max_smiles_len, tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: smiles_collate_fn(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # Model
    if yaml_config and args.model in ("concat_equal", "intra_cross_equal"):
        model = build_smiles_model(args.model, yaml_config, tokenizer.vocab_size, pad_id)
    else:
        model_kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_heads=num_heads,
            patch_size=patch_size,
            dropout=0.1,
            pad_id=pad_id,
            max_smiles_len=max_smiles_len,
        )

        if args.model in ("concat", "concat_equal"):
            model = DirectConcatSmilesModel(**model_kwargs)
        else:
            model = IntraCrossSmilesModel(**model_kwargs)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    predictions = []
    total_correct = 0
    total_canon_correct = 0
    total_valid = 0
    total_samples = 0
    total_pred_len = 0
    examples = []
    failure_cases = []

    for batch in loader:
        ir = batch["ir"].to(device)
        nmr_1h = batch["nmr_1h"].to(device)
        nmr_13c = batch["nmr_13c"].to(device)

        pred_ids = greedy_decode(model, ir, nmr_1h, nmr_13c,
                                  tokenizer, max_len=max_smiles_len)

        for i in range(len(batch["smiles"])):
            target_smi = batch["smiles"][i]
            idx = batch["idx"][i]
            pred_smi = tokenizer.decode(pred_ids[i], remove_special=True)
            pred_len = len(pred_ids[i])

            exact = int(pred_smi == target_smi)
            valid = int(is_valid(pred_smi)) if _HAS_RDKIT else -1
            target_canon = canonicalize(target_smi)
            pred_canon = canonicalize(pred_smi)
            canon_exact = int(pred_canon == target_canon) if pred_canon and target_canon else 0

            total_correct += exact
            total_valid += valid
            total_canon_correct += canon_exact
            total_samples += 1
            total_pred_len += pred_len

            pred_entry = {
                "idx": idx,
                "target_smiles": target_smi,
                "predicted_smiles": pred_smi,
                "exact_match": exact,
                "valid": valid,
                "canon_exact_match": canon_exact,
                "pred_length": pred_len,
            }
            predictions.append(pred_entry)

            if len(examples) < 20:
                examples.append({
                    "target": target_smi,
                    "predicted": pred_smi,
                    "exact": exact,
                    "valid": valid,
                })

            if not exact:
                failure_cases.append(pred_entry)

    # Save predictions CSV
    if args.save_predictions:
        pred_csv = os.path.join(out_dir, f"predictions_{args.split}.csv")
        with open(pred_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
        print(f"Saved predictions to {pred_csv}")

    # Summary
    summary = {
        "split": args.split,
        "num_samples": total_samples,
        "exact_string_match": round(total_correct / total_samples, 4) if total_samples > 0 else 0,
        "canonical_exact_match": round(total_canon_correct / total_samples, 4) if total_samples > 0 else 0,
        "rdkit_validity": round(total_valid / total_samples, 4) if _HAS_RDKIT and total_samples > 0 else ("RDKit not available" if not _HAS_RDKIT else 0),
        "avg_pred_length": round(total_pred_len / total_samples, 2) if total_samples > 0 else 0,
        "examples": examples[:10],
        "failure_cases": failure_cases[:20],
    }

    summary_path = os.path.join(out_dir, f"evaluation_summary_{args.split}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Summary ({args.split}):")
    for k, v in summary.items():
        if k not in ("examples", "failure_cases"):
            print(f"  {k}: {v}")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
