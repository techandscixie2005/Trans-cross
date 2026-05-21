"""Evaluate a trained SMILES generation model.

Loads a checkpoint and tokenizer, runs evaluation on valid/test splits,
and saves predictions and summary metrics.
"""

import argparse
import json
import os
import sys
import csv

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.dataset import TransCrossSmilesDataset
from src.transcross.collate import smiles_collate_fn
from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
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
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=["concat", "intra_cross"], required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--max-smiles-len", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()

    # Load tokenizer
    vocab_path = os.path.join(args.processed_dir, "smiles_vocab.json")
    tokenizer = SmilesTokenizer.load(vocab_path)
    pad_id = tokenizer.pad_id

    # Dataset
    dataset = TransCrossSmilesDataset(
        args.processed_dir, split=args.split,
        max_smiles_len=args.max_smiles_len, tokenizer=tokenizer,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: smiles_collate_fn(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # Model
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        dropout=0.1,
        pad_id=pad_id,
        max_smiles_len=args.max_smiles_len,
    )

    if args.model == "concat":
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

    for batch in loader:
        ir = batch["ir"].to(device)
        nmr_1h = batch["nmr_1h"].to(device)
        nmr_13c = batch["nmr_13c"].to(device)

        pred_ids = greedy_decode(model, ir, nmr_1h, nmr_13c,
                                  tokenizer, max_len=args.max_smiles_len)

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

            predictions.append({
                "idx": idx,
                "target_smiles": target_smi,
                "predicted_smiles": pred_smi,
                "exact_match": exact,
                "valid": valid,
                "canon_exact_match": canon_exact,
                "pred_length": pred_len,
            })

            if len(examples) < 20:
                examples.append({
                    "target": target_smi,
                    "predicted": pred_smi,
                    "exact": exact,
                    "valid": valid,
                })

    # Save predictions CSV
    pred_csv = os.path.join(args.out_dir, f"predictions_{args.split}.csv")
    with open(pred_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    print(f"Saved predictions to {pred_csv}")

    # Summary
    summary = {
        "split": args.split,
        "num_samples": total_samples,
        "token_accuracy": "N/A (use training metrics)",
        "exact_string_match": round(total_correct / total_samples, 4),
        "canonical_exact_match": round(total_canon_correct / total_samples, 4),
        "rdkit_validity": round(total_valid / total_samples, 4) if _HAS_RDKIT else "RDKit not available",
        "avg_pred_length": round(total_pred_len / total_samples, 2),
        "examples": examples[:10],
    }

    summary_path = os.path.join(args.out_dir, f"evaluation_summary_{args.split}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nEvaluation Summary ({args.split}):")
    for k, v in summary.items():
        if k != "examples":
            print(f"  {k}: {v}")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
