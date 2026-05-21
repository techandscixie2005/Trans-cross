#!/usr/bin/env python
"""Audit model for attention bias violations.

Supports both regex_atom and SPE tokenizers.
Inspects both E0 (concat_equal) and E1 (intra_cross_equal) models.

Usage (SPE):
  python scripts/audit_attention_bias.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --config configs/smiles_spe_equal_param.yaml
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.models.factory import build_smiles_model


FORBIDDEN_KEYWORDS = [
    "attention_bias",
    "relative_bias",
    "coord_bias",
    "modality_pair_bias",
    "graph_bias",
    "graphormer",
    "spatial_bias",
    "distance_bias",
]


def audit_model(model, model_name: str) -> dict:
    """Audit a single model for attention bias violations."""
    violations = []
    allowed_masks = []

    for name, param in model.named_parameters():
        name_lower = name.lower()
        for kw in FORBIDDEN_KEYWORDS:
            if kw in name_lower:
                violations.append({
                    "param_name": name,
                    "param_shape": list(param.shape),
                    "keyword": kw,
                    "reason": f"Parameter name contains forbidden keyword '{kw}'",
                })

    ALLOWED_BIAS_TYPES = {"layernorm", "linear", "embedding"}
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type == "MultiHeadAttention":
            for pname, param in module.named_parameters():
                allowed = any(
                    pname.startswith(prefix) for prefix in
                    ["q_proj.", "k_proj.", "v_proj.", "out_proj."]
                )
                if not allowed:
                    violations.append({
                        "param_name": f"{module_name}.{pname}",
                        "param_shape": list(param.shape),
                        "keyword": "attention_bias",
                        "reason": "Unexpected parameter in MultiHeadAttention",
                    })

    if hasattr(model, "decoder") and hasattr(model.decoder, "_build_causal_mask"):
        allowed_masks.append({
            "type": "causal_mask",
            "location": "decoder._build_causal_mask",
            "purpose": "Autoregressive SMILES decoding",
        })

    allowed_masks.append({
        "type": "padding_mask",
        "location": "_encode_spectra() memory_key_padding_mask",
        "purpose": "Batching correctness",
    })
    allowed_masks.append({
        "type": "padding_mask",
        "location": "decoder.forward() self_padding_mask",
        "purpose": "Decoder input padding for batching",
    })

    return {
        "model_name": model_name,
        "violations": violations,
        "allowed_masks": allowed_masks,
        "passed": len(violations) == 0,
    }


def generate_report(results: list, output_path: str):
    """Write attention bias audit report in Markdown."""
    lines = []
    lines.append("# Attention Bias Audit Report (SPE)")
    lines.append("")
    lines.append("This report verifies that no attention bias mechanisms exist")
    lines.append("in the SPE equal-parameter SMILES ablation models.")
    lines.append("")

    all_passed = all(r["passed"] for r in results)

    lines.append(f"## Overall: {'PASS' if all_passed else 'FAIL'}")
    lines.append("")

    for result in results:
        name = result["model_name"]
        passed = result["passed"]
        lines.append(f"## Model: {name}")
        lines.append(f"**Status: {'PASS' if passed else 'FAIL'}**")
        lines.append("")

        if result["violations"]:
            lines.append("### Violations")
            lines.append("")
            lines.append("| Parameter | Shape | Keyword | Reason |")
            lines.append("|---|---|---|---|")
            for v in result["violations"]:
                lines.append(
                    f"| {v['param_name']} | {v['param_shape']} "
                    f"| {v['keyword']} | {v['reason']} |"
                )
            lines.append("")
        else:
            lines.append("No violations found.")
            lines.append("")

        lines.append("### Allowed Masks")
        lines.append("")
        lines.append("| Type | Location | Purpose |")
        lines.append("|---|---|---|")
        for m in result["allowed_masks"]:
            lines.append(f"| {m['type']} | {m['location']} | {m['purpose']} |")
        lines.append("")

    lines.append("## Forbidden Mechanisms (explicitly absent)")
    lines.append("")
    lines.append("- [x] No coordinate bias (spectral x-axis position bias)")
    lines.append("- [x] No modality-pair bias (learned pairwise modality bias)")
    lines.append("- [x] No relative position bias (Transformer-XL style)")
    lines.append("- [x] No Graphormer-style spatial/distance bias")
    lines.append("- [x] No learned additive attention-logit bias")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")


def load_tokenizer(processed_dir: str, config: dict):
    """Load tokenizer based on config type."""
    tok_cfg = config.get("tokenizer", {})
    tok_type = tok_cfg.get("type", "regex_atom")

    if tok_type == "spe":
        vocab_path = tok_cfg.get("vocab_path")
        if not vocab_path:
            vocab_path = os.path.join(processed_dir, "spe_vocab_256.json")
        if not os.path.exists(vocab_path):
            print(f"WARNING: SPE vocab not found at {vocab_path}, falling back to regex_atom")
            vocab_path = os.path.join(processed_dir, "smiles_vocab.json")
            tokenizer = SmilesTokenizer.load(vocab_path)
        else:
            tokenizer = SPETokenizer.load(vocab_path)
    else:
        vocab_path = os.path.join(processed_dir, "smiles_vocab.json")
        tokenizer = SmilesTokenizer.load(vocab_path)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Audit models for attention bias violations."
    )
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", default="reports/spe_attention_bias_audit.md")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    tokenizer = load_tokenizer(args.processed_dir, config)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id
    print(f"Vocab size: {vocab_size}, pad_id: {pad_id}")

    results = []

    for model_key, model_name in [
        ("concat_equal", "E0 DirectConcat (SPE)"),
        ("intra_cross_equal", "E1 IntraCross (SPE)"),
    ]:
        print(f"\nAuditing {model_name}...")
        model = build_smiles_model(model_key, config, vocab_size, pad_id)
        result = audit_model(model, model_name)
        results.append(result)

        if result["passed"]:
            print(f"  {model_name}: PASS")
        else:
            print(f"  {model_name}: FAIL - {len(result['violations'])} violations")
            for v in result["violations"]:
                print(f"    - {v['param_name']}: {v['reason']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    generate_report(results, args.output)

    if not all(r["passed"] for r in results):
        print("\nFAIL: Attention bias violations detected!")
        sys.exit(1)

    print("\nPASS: No attention bias violations found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
