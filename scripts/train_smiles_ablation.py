"""Train SMILES generation ablation models (concat vs intra_cross).

Usage (config mode, preferred):
  python scripts/train_smiles_ablation.py \
    --config configs/smiles_equal_param.yaml \
    --model concat_equal \
    --out-dir /path/to/run

Usage (legacy CLI mode):
  python scripts/train_smiles_ablation.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --model concat \
    --epochs 30 \
    --batch-size 32 \
    --d-model 128 \
    --encoder-layers 2 \
    --decoder-layers 2 \
    --num-heads 4 \
    --patch-size 64 \
    --lr 1e-4 \
    --seed 42 \
    --max-smiles-len 160 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/smiles_concat_seed42
"""

import argparse
import json
import os
import sys
import csv
import time
import yaml
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.dataset import TransCrossSmilesDataset
from src.transcross.collate import smiles_collate_fn
from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
from src.transcross.models.factory import build_smiles_model
from src.transcross.model_utils import (
    count_trainable_parameters,
    format_parameter_table,
)
from src.transcross.generation import greedy_decode


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected, running on CPU")
    return device


def compute_loss_and_acc(logits, target_ids, pad_id):
    """Compute cross-entropy loss and token accuracy (excluding pad)."""
    # logits: (B, T, V), target_ids: (B, T)
    B, T, V = logits.shape
    logits = logits.reshape(B * T, V)
    targets = target_ids.reshape(B * T)

    loss = nn.functional.cross_entropy(logits, targets, ignore_index=pad_id)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        non_pad = targets != pad_id
        correct = (preds == targets) & non_pad
        acc = correct.sum().float() / non_pad.sum().float() if non_pad.any() else 0.0

    return loss, acc.item()


def train_epoch(model, dataloader, optimizer, pad_id, device,
                clip_grad: float = 1.0):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        ir = batch["ir"].to(device)
        nmr_1h = batch["nmr_1h"].to(device)
        nmr_13c = batch["nmr_13c"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        optimizer.zero_grad()
        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        loss, acc = compute_loss_and_acc(logits, target_ids, pad_id)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


@torch.no_grad()
def validate(model, dataloader, pad_id, device, tokenizer,
             max_len: int = 256):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    total_exact = 0
    total_samples = 0

    for batch in dataloader:
        ir = batch["ir"].to(device)
        nmr_1h = batch["nmr_1h"].to(device)
        nmr_13c = batch["nmr_13c"].to(device)
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(ir, nmr_1h, nmr_13c, input_ids)
        loss, acc = compute_loss_and_acc(logits, target_ids, pad_id)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        # Greedy decode
        predicted_ids = greedy_decode(model, ir, nmr_1h, nmr_13c,
                                       tokenizer, max_len=max_len)
        for i, smi_target in enumerate(batch["smiles"]):
            pred_smi = tokenizer.decode(predicted_ids[i], remove_special=True)
            if pred_smi == smi_target:
                total_exact += 1
            total_samples += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    exact_match = total_exact / total_samples if total_samples > 0 else 0.0
    return avg_loss, avg_acc, exact_match


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train SMILES generation ablation model."
    )
    # Config mode (preferred)
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--model", default="concat",
                       choices=["concat", "intra_cross", "concat_equal", "intra_cross_equal"])
    # Legacy CLI args (used when --config not provided, or override config values)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--encoder-layers", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-smiles-len", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = load_config(args.config)
        print(f"Loaded config from {args.config}")
        print(json.dumps(config, indent=2))

    # Resolve parameters: CLI overrides > config > defaults
    train_cfg = config.get("training", {}) if config else {}
    shared_cfg = config.get("shared", {}) if config else {}
    data_cfg = config.get("data", {}) if config else {}
    tok_cfg = config.get("tokenizer", {}) if config else {}

    processed_dir = args.processed_dir or data_cfg.get("processed_dir")
    if not processed_dir:
        parser.error("--processed-dir is required (or set in config)")

    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 32)
    lr = args.lr if args.lr is not None else train_cfg.get("lr", 1e-4)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    max_smiles_len = args.max_smiles_len if args.max_smiles_len is not None else data_cfg.get("max_smiles_len", 160)
    patch_size = args.patch_size if args.patch_size is not None else tok_cfg.get("patch_size", 64)

    set_seed(seed)
    device = get_device()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer
    vocab_path = os.path.join(processed_dir, "smiles_vocab.json")
    if not os.path.exists(vocab_path):
        print("Building SMILES vocabulary...")
        smiles_path = os.path.join(processed_dir, "canonical_smiles.txt")
        with open(smiles_path) as f:
            smiles_list = [l.strip() for l in f if l.strip()]
        tokenizer = SmilesTokenizer.build_from_smiles(smiles_list)
        tokenizer.save(vocab_path)
    else:
        tokenizer = SmilesTokenizer.load(vocab_path)

    pad_id = tokenizer.pad_id
    print(f"Vocab size: {tokenizer.vocab_size}, pad_id: {pad_id}")

    # Datasets
    train_ds = TransCrossSmilesDataset(
        processed_dir, split="train",
        max_smiles_len=max_smiles_len, tokenizer=tokenizer,
    )
    valid_ds = TransCrossSmilesDataset(
        processed_dir, split="valid",
        max_smiles_len=max_smiles_len, tokenizer=tokenizer,
    )
    test_ds = TransCrossSmilesDataset(
        processed_dir, split="test",
        max_smiles_len=max_smiles_len, tokenizer=tokenizer,
    )

    print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: smiles_collate_fn(b, pad_id),
        num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: smiles_collate_fn(b, pad_id),
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: smiles_collate_fn(b, pad_id),
        num_workers=2, pin_memory=True,
    )

    # Create model
    if config and args.model in ("concat_equal", "intra_cross_equal"):
        # Use factory for equal-param models
        model = build_smiles_model(args.model, config, tokenizer.vocab_size, pad_id)
        if args.model == "concat_equal":
            model_name = "DirectConcatSmilesModel-equal"
        else:
            model_name = "IntraCrossSmilesModel-equal"
    else:
        # Legacy model creation
        d_model = args.d_model or shared_cfg.get("d_model", 128)
        encoder_layers = args.encoder_layers or (
            config.get("e0_concat", {}).get("encoder_layers", 2) if config else 2
        )
        decoder_layers = args.decoder_layers or shared_cfg.get("decoder_layers", 2)
        num_heads = args.num_heads or shared_cfg.get("num_heads", 4)

        model_kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            ir_len=1801, h1_len=1501, c13_len=2201,
            patch_size=patch_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_heads=num_heads,
            dropout=0.1,
            pad_id=pad_id,
            max_smiles_len=max_smiles_len,
        )

        if args.model in ("concat", "concat_equal"):
            model = DirectConcatSmilesModel(**model_kwargs)
            model_name = "DirectConcatSmilesModel"
        else:
            model = IntraCrossSmilesModel(**model_kwargs)
            model_name = "IntraCrossSmilesModel"

    model = model.to(device)
    n_params = count_trainable_parameters(model)
    print(f"Model: {model_name}")
    print(f"Parameters: {n_params:,}")
    print(format_parameter_table(model))

    # Save full config snapshot
    run_config = {
        "model": args.model,
        "model_name": model_name,
        "vocab_size": tokenizer.vocab_size,
        "n_params": n_params,
        "processed_dir": processed_dir,
        "lr": lr,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_smiles_len": max_smiles_len,
        "patch_size": patch_size,
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
    }
    # Merge config into run_config for traceability
    if config:
        run_config["config_file"] = args.config
        run_config["config_content"] = config

    with open(os.path.join(args.out_dir, "config_used.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    # Save parameter count separately
    param_count = {
        "model_name": model_name,
        "total_params": n_params,
        "by_module": {
            k: v for k, v in sorted(
                type(model).__name__,
            )
        },
    }
    # Use our utility for proper per-module counts
    from src.transcross.model_utils import count_parameters_by_module
    param_count["by_module"] = count_parameters_by_module(model)
    with open(os.path.join(args.out_dir, "parameter_count.json"), "w") as f:
        json.dump(param_count, f, indent=2)

    weight_decay = train_cfg.get("weight_decay", 1e-4) if config else 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda") if args.mixed_precision else None

    best_valid_loss = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, pad_id, device,
        )
        valid_loss, valid_acc, valid_exact = validate(
            model, valid_loader, pad_id, device, tokenizer,
            max_len=max_smiles_len,
        )

        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_loss": round(valid_loss, 6),
            "train_token_acc": round(train_acc, 4),
            "valid_token_acc": round(valid_acc, 4),
            "valid_exact_match": round(valid_exact, 4),
        }
        history.append(entry)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_acc: {valid_acc:.4f} | "
            f"valid_exact: {valid_exact:.4f} | "
            f"{elapsed:.1f}s"
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch
            save_checkpoint(
                model, optimizer, epoch,
                {"valid_loss": valid_loss, "valid_acc": valid_acc},
                os.path.join(args.out_dir, "best_model.pt"),
            )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, epochs,
        {"valid_loss": valid_loss, "valid_acc": valid_acc},
        os.path.join(args.out_dir, "final_model.pt"),
    )

    # Save training log
    if history:
        with open(os.path.join(args.out_dir, "training_log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    # Load best model and evaluate on test set
    best_ckpt = torch.load(os.path.join(args.out_dir, "best_model.pt"),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc, test_exact = validate(
        model, test_loader, pad_id, device, tokenizer,
        max_len=max_smiles_len,
    )

    metrics = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "test_loss": round(test_loss, 6),
        "test_token_acc": round(test_acc, 4),
        "test_exact_match": round(test_exact, 4),
        "n_params": n_params,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nBest epoch: {best_epoch}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test token acc: {test_acc:.4f}")
    print(f"Test exact match: {test_exact:.4f}")


if __name__ == "__main__":
    main()
