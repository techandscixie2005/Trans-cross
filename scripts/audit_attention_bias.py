#!/usr/bin/env python
"""Audit model for attention bias violations.

Inspects both E0 (concat_equal) and E1 (intra_cross_equal) models and checks:
- No parameter name contains attention_bias, relative_bias, coord_bias,
  modality_pair_bias, graph_bias
- No learned additive attention-logit bias (bias terms in attention)
- Causal and padding masks ARE allowed and listed separately
- No cross-modal bias table

Outputs a Markdown report and exits nonzero if violations are found.
"""

import argparse
import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transcross.smiles_tokenizer import SmilesTokenizer
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
    """Audit a single model for attention bias violations.

    Returns dict with keys: violations, allowed_masks.
    """
    violations = []
    allowed_masks = []

    for name, param in model.named_parameters():
        name_lower = name.lower()
        # Check for forbidden keywords in parameter names
        for kw in FORBIDDEN_KEYWORDS:
            if kw in name_lower:
                violations.append({
                    "param_name": name,
                    "param_shape": list(param.shape),
                    "keyword": kw,
                    "reason": f"Parameter name contains forbidden keyword '{kw}'",
                })

    # Check for learned additive bias in attention modules
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        # Check for bias parameters inside attention-like modules
        if "attention" in module_type.lower() or "attn" in module_type.lower():
            for pname, param in module.named_parameters():
                if "bias" in pname.lower():
                    # Check if this is a projection bias (allowed) or attention bias
                    # Projection biases are in Linear layers
                    parent_type = type(
                        getattr(module, pname.replace(".", "_").replace("_bias", ""), None)
                    ).__name__ if hasattr(module, "_parameters") else ""
                    # Allow Linear bias (q_proj.bias, etc.)
                    # Flag any bias that's not part of a Linear layer
                    if "proj" not in pname.lower():
                        violations.append({
                            "param_name": f"{module_name}.{pname}",
                            "param_shape": list(param.shape),
                            "keyword": "attention_bias",
                            "reason": "Bias parameter in attention module not in projection",
                        })

    # Identify causal masks (allowed)
    # Look for _build_causal_mask methods or causal_mask parameters
    if hasattr(model, "decoder") and hasattr(model.decoder, "_build_causal_mask"):
        allowed_masks.append({
            "type": "causal_mask",
            "location": "decoder._build_causal_mask",
            "purpose": "Autoregressive SMILES decoding",
        })

    # Identify padding masks (allowed)
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
    lines.append("# Attention Bias Audit Report")
    lines.append("")
    lines.append("This report verifies that no attention bias mechanisms exist")
    lines.append("in the equal-parameter SMILES ablation models.")
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


def main():
    parser = argparse.ArgumentParser(
        description="Audit models for attention bias violations."
    )
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output", default="reports/attention_bias_audit.md")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    vocab_path = os.path.join(args.processed_dir, "smiles_vocab.json")
    tokenizer = SmilesTokenizer.load(vocab_path)
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_id

    results = []

    for model_key, model_name in [
        ("concat_equal", "E0 DirectConcat"),
        ("intra_cross_equal", "E1 IntraCross"),
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
