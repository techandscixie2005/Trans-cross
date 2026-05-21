"""Model parameter counting and comparison utilities."""

from typing import Dict, Tuple

import torch.nn as nn


def count_trainable_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_by_module(model: nn.Module) -> Dict[str, int]:
    """Count trainable parameters for each top-level submodule.

    Returns a dict mapping module name to parameter count.
    """
    counts: Dict[str, int] = {}
    for name, child in model.named_children():
        counts[name] = sum(
            p.numel() for p in child.parameters() if p.requires_grad
        )
    # Also add unassigned top-level parameters
    direct = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ) - sum(counts.values())
    if direct > 0:
        counts["_direct_params"] = direct
    return counts


def format_parameter_table(model: nn.Module) -> str:
    """Format a human-readable parameter table for a model.

    Returns a multi-line string with per-module and total counts.
    """
    by_module = count_parameters_by_module(model)
    total = sum(by_module.values())

    lines = []
    lines.append(f"{'Module':<40s} {'Parameters':>12s}")
    lines.append("-" * 54)
    for name, count in sorted(by_module.items()):
        lines.append(f"{name:<40s} {count:>12,}")
    lines.append("-" * 54)
    lines.append(f"{'TOTAL':<40s} {total:>12,}")
    return "\n".join(lines)


def compare_models(
    model_e0: nn.Module,
    model_e1: nn.Module,
    max_relative_diff: float = 0.01,
) -> Tuple[Dict, bool]:
    """Compare parameter counts of E0 and E1.

    Args:
        model_e0: E0 DirectConcat model
        model_e1: E1 IntraCross model
        max_relative_diff: maximum allowed relative difference (default 1%)

    Returns:
        (comparison_dict, is_within_tolerance)
    """
    e0_total = count_trainable_parameters(model_e0)
    e1_total = count_trainable_parameters(model_e1)

    e0_by_module = count_parameters_by_module(model_e0)
    e1_by_module = count_parameters_by_module(model_e1)

    abs_diff = abs(e0_total - e1_total)
    rel_diff = abs_diff / max(e0_total, e1_total) if max(e0_total, e1_total) > 0 else 0.0

    within = rel_diff <= max_relative_diff

    return {
        "e0_total": e0_total,
        "e1_total": e1_total,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "rel_diff_pct": round(rel_diff * 100, 4),
        "within_tolerance": within,
        "e0_by_module": e0_by_module,
        "e1_by_module": e1_by_module,
    }, within
