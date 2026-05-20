#!/usr/bin/env python3
"""Train fingerprint prediction ablation: concat vs intra-cross encoder.

Usage:
  python scripts/train_fingerprint_ablation.py \
    --processed-dir /path/to/data/processed \
    --model concat \
    --epochs 30 \
    --batch-size 64 \
    --d-model 128 \
    --num-layers 2 \
    --num-heads 4 \
    --patch-size 64 \
    --lr 1e-4 \
    --seed 42 \
    --out-dir /path/to/runs/exp_name
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.transcross.dataset import TranscrossDataset
from src.transcross.models.concat_encoder import ConcatEncoder
from src.transcross.models.intra_cross_encoder import IntraCrossEncoder


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """Compute BCE, bit accuracy, and Tanimoto similarity.

    Args:
        logits: (B, n_bits) float — model outputs before sigmoid
        targets: (B, n_bits) float — ground truth binary fp

    Returns:
        dict with loss, bit_acc, tanimoto
    """
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(logits, targets).item()

    preds = (torch.sigmoid(logits) > 0.5).float()
    acc = (preds == targets).float().mean().item()

    # Tanimoto: intersection / union for each sample, then mean
    intersection = (preds * targets).sum(dim=1)
    union = (preds + targets).clamp(0, 1).sum(dim=1)
    # Handle zero-union
    tanimoto = torch.where(
        union > 0,
        intersection / union,
        torch.ones_like(intersection),  # both all-zero -> perfect match
    )
    tanimoto_mean = tanimoto.mean().item()

    return {"loss": loss, "bit_acc": acc, "tanimoto": tanimoto_mean}


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ir = batch["ir"].to(device)
        h1 = batch["nmr_1h"].to(device)
        c13 = batch["nmr_13c"].to(device)
        fp = batch["fp"].to(device)

        optimizer.zero_grad()
        logits = model(ir, h1, c13)
        loss = nn.BCEWithLogitsLoss()(logits, fp)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * ir.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits = []
    all_targets = []
    for batch in loader:
        ir = batch["ir"].to(device)
        h1 = batch["nmr_1h"].to(device)
        c13 = batch["nmr_13c"].to(device)
        fp = batch["fp"].to(device)

        logits = model(ir, h1, c13)
        all_logits.append(logits.cpu())
        all_targets.append(fp.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return compute_metrics(all_logits, all_targets)


def main():
    parser = argparse.ArgumentParser(description="Train fingerprint ablation")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--model", required=True, choices=["concat", "intra_cross"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Create model
    model_kwargs = dict(
        patch_size=args.patch_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    if args.model == "concat":
        model = ConcatEncoder(**model_kwargs)
    else:
        model = IntraCrossEncoder(**model_kwargs)
    model = model.to(device)
    n_params = model.count_params()
    print(f"Parameters: {n_params:,}")

    # Datasets
    train_ds = TranscrossDataset(args.processed_dir, split="train")
    valid_ds = TranscrossDataset(args.processed_dir, split="valid")
    test_ds = TranscrossDataset(args.processed_dir, split="test")

    if train_ds.fp is None:
        print("ERROR: morgan_fp_2048.npy not found. Run build_fingerprints.py first.")
        sys.exit(1)

    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}  Test: {len(test_ds)}")

    # For small valid/test sets, use small batch to avoid GPU memory issues
    valid_bs = min(args.batch_size, len(valid_ds))
    test_bs = min(args.batch_size, len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=valid_bs, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=test_bs, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(args.out_dir, exist_ok=True)

    best_valid_tanimoto = -1.0
    best_epoch = 0
    best_state = None
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        valid_metrics = evaluate(model, valid_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_bit_acc": valid_metrics["bit_acc"],
            "valid_tanimoto": valid_metrics["tanimoto"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"valid_loss={valid_metrics['loss']:.4f}  "
            f"valid_acc={valid_metrics['bit_acc']:.4f}  "
            f"valid_tani={valid_metrics['tanimoto']:.4f}"
        )

        if valid_metrics["tanimoto"] > best_valid_tanimoto:
            best_valid_tanimoto = valid_metrics["tanimoto"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save training log
    log_path = os.path.join(args.out_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Saved: {log_path}")

    # Load best checkpoint and evaluate test
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best epoch: {best_epoch} (valid_tanimoto={best_valid_tanimoto:.4f})")

    test_metrics = evaluate(model, test_loader, device)
    print(
        f"Test: loss={test_metrics['loss']:.4f}  "
        f"acc={test_metrics['bit_acc']:.4f}  "
        f"tanimoto={test_metrics['tanimoto']:.4f}"
    )

    # Save best checkpoint
    ckpt_path = os.path.join(args.out_dir, "best.pt")
    torch.save({"model_state": best_state, "epoch": best_epoch, "valid_tanimoto": best_valid_tanimoto}, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # Save metrics
    metrics = {
        "model": args.model,
        "n_params": n_params,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_valid_tanimoto": float(best_valid_tanimoto),
        "test_loss": float(test_metrics["loss"]),
        "test_bit_acc": float(test_metrics["bit_acc"]),
        "test_tanimoto": float(test_metrics["tanimoto"]),
        "device": str(device),
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
