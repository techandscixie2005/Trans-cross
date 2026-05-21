#!/usr/bin/env python
"""Evaluate a trained SMILES generation model with full chemical metrics.

Supports regex_atom and SPE tokenizers. Auto-detects tokenizer type from run config.
Outputs predictions CSV and comprehensive evaluation summary JSON.

Usage (run-dir mode):
  python scripts/evaluate_smiles_model.py \
    --run-dir /path/to/training/run \
    --split test \
    --processed-dir /data/... \
    --save-predictions
"""

import argparse
import csv
import json
import os
import sys
from typing import Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.dataset import TransCrossSmilesDataset
from src.transcross.collate import smiles_collate_fn
from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.models.factory import build_smiles_model
from src.transcross.generation import greedy_decode
from src.transcross.chem_metrics import (
    canonicalize,
    is_valid,
    compute_tanimoto,
    scaffold_match,
    functional_group_f1,
    levenshtein,
    compute_summary_from_rows,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer_for_eval(processed_dir: str, run_config: dict):
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


def compute_token_accuracy(pred_ids, target_ids, pad_id):
    """Compute per-sample token accuracy (excluding pad)."""
    if len(pred_ids) == 0:
        return 0.0
    min_len = min(len(pred_ids), len(target_ids))
    correct = 0
    total = 0
    for i in range(min_len):
        if target_ids[i] != pad_id:
            total += 1
            if pred_ids[i] == target_ids[i]:
                correct += 1
    return correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate SMILES generation model.")
    parser.add_argument("--run-dir", default=None)
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
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--save-predictions", action="store_true", default=True)
    parser.add_argument("--model-name", default=None,
                       help="Override model name in output")
    parser.add_argument("--seed", type=int, default=None,
                       help="Override seed in output")
    args = parser.parse_args()

    # ── Load config from run directory ──────────────────────────────────
    if args.run_dir:
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
        yaml_config = run_config.get("config_content", {})
        args.checkpoint = checkpoint_path
        args.model = model_type
        out_dir = args.out_dir or args.run_dir

        d_model = run_config.get("d_model", 128)
        encoder_layers = run_config.get("encoder_layers", 2)
        decoder_layers = run_config.get("decoder_layers", 2)
        num_heads = run_config.get("num_heads", 4)
        patch_size = run_config.get("patch_size", 64)
        max_smiles_len = run_config.get("max_smiles_len", 160)
        tokenizer_type = run_config.get("tokenizer_type", "regex_atom")
        eval_seed = args.seed or run_config.get("seed", 42)
        eval_model_name = args.model_name or run_config.get("model_name", model_type)
    else:
        if not args.checkpoint:
            parser.error("Either --run-dir or --checkpoint is required")
        out_dir = args.out_dir or os.path.dirname(args.checkpoint)
        yaml_config = {}
        d_model = args.d_model or 128
        encoder_layers = args.encoder_layers or 2
        decoder_layers = args.decoder_layers or 2
        num_heads = args.num_heads or 4
        patch_size = args.patch_size or 64
        max_smiles_len = args.max_smiles_len or 160
        tokenizer_type = "regex_atom"
        run_config = {}
        eval_seed = args.seed or 0
        eval_model_name = args.model_name or args.model

    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    # ── Load tokenizer ──────────────────────────────────────────────────
    tokenizer, tok_type = load_tokenizer_for_eval(args.processed_dir, run_config)
    pad_id = tokenizer.pad_id
    eos_id = tokenizer.eos_id
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer: {tok_type}, vocab_size={vocab_size}")

    # ── Dataset ─────────────────────────────────────────────────────────
    spe_vocab_path = None
    if tok_type == "spe":
        config_content = run_config.get("config_content", {})
        tok_cfg = config_content.get("tokenizer", {})
        spe_vocab_path = tok_cfg.get("vocab_path")

    dataset = TransCrossSmilesDataset(
        args.processed_dir, split=args.split,
        max_smiles_len=max_smiles_len, tokenizer=tokenizer,
        tokenizer_type=tok_type, spe_vocab_path=spe_vocab_path,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: smiles_collate_fn(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # ── Build model ─────────────────────────────────────────────────────
    if yaml_config and args.model in ("concat_equal", "intra_cross_equal"):
        model = build_smiles_model(args.model, yaml_config, vocab_size, pad_id)
    else:
        model_kwargs = dict(
            vocab_size=vocab_size, d_model=d_model,
            encoder_layers=encoder_layers, decoder_layers=decoder_layers,
            num_heads=num_heads, patch_size=patch_size, dropout=0.1,
            pad_id=pad_id, max_smiles_len=max_smiles_len,
        )
        if args.model in ("concat", "concat_equal"):
            from src.transcross.models.smiles_concat import DirectConcatSmilesModel
            model = DirectConcatSmilesModel(**model_kwargs)
        else:
            from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
            model = IntraCrossSmilesModel(**model_kwargs)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # ── Evaluate ────────────────────────────────────────────────────────
    rows = []
    total_eos_hit = 0
    total_hit_max = 0
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        ir = batch["ir"].to(device)
        nmr_1h = batch["nmr_1h"].to(device)
        nmr_13c = batch["nmr_13c"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # Teacher-forcing loss & token accuracy
        with torch.no_grad():
            logits = model(ir, nmr_1h, nmr_13c, input_ids)
            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(B * T, V), target_ids.reshape(B * T),
                ignore_index=pad_id,
            )
            preds_tf = logits.argmax(dim=-1)
            non_pad = target_ids != pad_id
            acc = (preds_tf == target_ids) & non_pad
            batch_acc = acc.sum().float() / non_pad.sum().float() if non_pad.any() else 0.0

        total_loss += loss.item()
        total_acc += batch_acc.item()
        n_batches += 1

        # Greedy decode (with EOS info)
        pred_ids, batch_eos_hit, batch_eos_step = greedy_decode(
            model, ir, nmr_1h, nmr_13c, tokenizer,
            max_len=max_smiles_len, return_eos_info=True,
        )

        for i in range(len(batch["smiles"])):
            target_smi = batch["smiles"][i]
            idx_val = batch["idx"][i]
            pred_smi = tokenizer.decode(pred_ids[i], remove_special=True)
            pred_len_chars = len(pred_smi)
            pred_len_tokens = len(pred_ids[i])
            target_len_chars = len(target_smi)

            # Chemical metrics
            exact = 1 if pred_smi == target_smi else 0
            valid = 1 if is_valid(pred_smi) else 0
            target_canon = canonicalize(target_smi)
            pred_canon = canonicalize(pred_smi)
            canon_exact = 1 if (pred_canon and target_canon and pred_canon == target_canon) else 0
            tanimoto = compute_tanimoto(target_smi, pred_smi)
            scaff = scaffold_match(target_smi, pred_smi)
            fg_p, fg_r, fg_f1 = functional_group_f1(target_smi, pred_smi)
            lev = levenshtein(target_smi, pred_smi)
            tok_acc = compute_token_accuracy(pred_ids[i], target_ids[i].tolist(), pad_id)

            # EOS detection from decode info
            eos_hit = batch_eos_hit[i]
            eos_pos = batch_eos_step[i]
            hit_max = (not eos_hit) and (pred_len_tokens >= max_smiles_len)
            if eos_hit:
                total_eos_hit += 1
            if hit_max:
                total_hit_max += 1

            row = {
                "idx": idx_val,
                "target_smiles": target_smi,
                "pred_smiles": pred_smi,
                "target_len": target_len_chars,
                "pred_len": pred_len_chars,
                "exact_match": exact,
                "canonical_exact_match": canon_exact,
                "rdkit_valid": valid,
                "tanimoto": round(tanimoto, 6),
                "scaffold_match": scaff,
                "fg_precision": round(fg_p, 6),
                "fg_recall": round(fg_r, 6),
                "fg_f1": round(fg_f1, 6),
                "token_accuracy_sample": round(tok_acc, 6),
                "levenshtein": lev,
                "pred_token_length": pred_len_tokens,
                "eos_hit": int(eos_hit),
                "eos_position": eos_pos,
                "hit_max_len": int(hit_max),
                "split": args.split,
                "model_name": eval_model_name,
                "tokenizer_type": tok_type,
                "seed": eval_seed,
            }
            rows.append(row)

    n_samples = len(rows)

    # ── Save predictions CSV ────────────────────────────────────────────
    if args.save_predictions:
        pred_csv = os.path.join(out_dir, f"predictions_{args.split}.csv")
        fieldnames = [
            "idx", "target_smiles", "pred_smiles", "target_len", "pred_len",
            "exact_match", "canonical_exact_match", "rdkit_valid",
            "tanimoto", "scaffold_match", "fg_precision", "fg_recall", "fg_f1",
            "token_accuracy_sample", "levenshtein",
            "pred_token_length", "eos_hit", "eos_position", "hit_max_len",
            "split", "model_name", "tokenizer_type", "seed",
        ]
        with open(pred_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to {pred_csv} ({n_samples} rows)")

    # ── Compute full summary ────────────────────────────────────────────
    summary = compute_summary_from_rows(
        rows, split=args.split,
        model_name=eval_model_name,
        tokenizer_type=tok_type,
        seed=eval_seed,
    )

    # Add teacher-forcing metrics
    summary["teacher_forcing_loss"] = round(total_loss / n_batches, 6) if n_batches > 0 else 0
    summary["teacher_forcing_token_accuracy"] = round(total_acc / n_batches, 6) if n_batches > 0 else 0

    # Add EOS stats
    summary["pct_ending_with_eos"] = round(total_eos_hit / n_samples * 100, 2) if n_samples > 0 else 0
    summary["pct_hitting_max_len"] = round(total_hit_max / n_samples * 100, 2) if n_samples > 0 else 0

    # Example predictions
    examples = []
    for r in rows[:20]:
        examples.append({
            "target": r["target_smiles"],
            "predicted": r["pred_smiles"],
            "exact": r["exact_match"],
            "valid": r["rdkit_valid"],
            "tanimoto": r["tanimoto"],
        })
    summary["examples"] = examples

    # Failure cases (invalid predictions)
    invalid_rows = [r for r in rows if not r["rdkit_valid"]]
    failure_cases = []
    for r in invalid_rows[:20]:
        failure_cases.append({
            "target": r["target_smiles"],
            "predicted": r["pred_smiles"],
            "tanimoto": r["tanimoto"],
            "levenshtein": r["levenshtein"],
        })
    summary["failure_cases"] = failure_cases

    # Save summary
    summary_path = os.path.join(out_dir, f"evaluation_summary_{args.split}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\nEvaluation Summary ({args.split}):")
    print(f"  Samples:              {n_samples}")
    print(f"  Loss (TF):            {summary['teacher_forcing_loss']:.4f}")
    print(f"  Token acc (TF):       {summary['teacher_forcing_token_accuracy']:.4f}")
    print(f"  Exact string match:   {summary['exact_string_match']:.4f}")
    print(f"  Canonical exact:      {summary['canonical_exact_match']:.4f}")
    print(f"  RDKit validity:       {summary['rdkit_validity']:.4f}")
    print(f"  Unique generated:     {summary['unique_generated']}")
    print(f"  Unique ratio:         {summary['unique_ratio']:.4f}")
    print(f"  Mode collapse score:  {summary['mode_collapse_score']:.4f}")
    print(f"  Prediction entropy:   {summary['prediction_entropy']:.4f}")
    print(f"  Mean Tanimoto:        {summary['mean_tanimoto']:.4f}")
    print(f"  Mean Tanimoto (valid):{summary['mean_tanimoto_valid_only']:.4f}")
    print(f"  Scaffold match:       {summary['scaffold_match_rate']:.4f}")
    print(f"  FG-F1:                {summary['mean_fg_f1']:.4f}")
    print(f"  Mean Levenshtein:     {summary['mean_levenshtein']:.2f}")
    print(f"  Avg pred char len:    {summary['avg_pred_char_length']:.2f}")
    print(f"  % ending w/ EOS:      {summary['pct_ending_with_eos']:.1f}%")
    print(f"  % hitting max len:    {summary['pct_hitting_max_len']:.1f}%")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
