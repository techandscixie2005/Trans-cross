#!/usr/bin/env python
"""Audit generation behavior across runs: mode collapse, length bias, patterns.

Reads predictions_valid.csv and predictions_test.csv from each run directory,
computes length statistics, mode collapse scores, entropy, validity/length
buckets, Tanimoto/length buckets, top/bottom predictions, and produces:

  {run_dir}/generation_behavior_audit.json   (structured data)
  {run_dir}/generation_behavior_audit.md     (human-readable)
  {output_dir}/generation_behavior_audit_all.md  (aggregate report)

Usage:
  python scripts/audit_generation_behavior.py \
    --run-dirs /path/to/run1 /path/to/run2 ... \
    --output-dir /path/to/output
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.transcross.chem_metrics import (
    compute_length_stats,
    is_valid,
    compute_tanimoto,
    mode_collapse_score,
    prediction_entropy,
    top_frequencies,
    unique_stats,
    validity_by_length,
    tanimoto_by_length,
)


# ── CSV reading ──────────────────────────────────────────────────────────────


def read_predictions(csv_path: str) -> List[Dict]:
    """Read a predictions CSV and return list of row dicts.

    Returns empty list if file does not exist or is empty.
    """
    if not os.path.isfile(csv_path):
        return []
    rows: List[Dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coerce numeric fields
            for numeric_key in (
                "idx", "target_len", "pred_len", "exact_match",
                "canonical_exact_match", "rdkit_valid", "scaffold_match",
                "pred_token_length", "eos_hit", "eos_position", "hit_max_len",
                "seed", "levenshtein",
            ):
                val = row.get(numeric_key)
                if val is not None and val != "":
                    try:
                        row[numeric_key] = int(val)
                    except ValueError:
                        pass
            for float_key in ("tanimoto", "fg_precision", "fg_recall", "fg_f1",
                              "token_accuracy_sample"):
                val = row.get(float_key)
                if val is not None and val != "":
                    try:
                        row[float_key] = float(val)
                    except ValueError:
                        pass
            rows.append(row)
    return rows


def auto_detect_model_name(rows: List[Dict]) -> str:
    """Return the model_name from the first row, or 'unknown'."""
    if rows:
        name = rows[0].get("model_name", "")
        if name:
            return name.strip()
    return "unknown"


def auto_detect_tokenizer_type(rows: List[Dict]) -> str:
    """Return the tokenizer_type from the first row, or 'unknown'."""
    if rows:
        tt = rows[0].get("tokenizer_type", "")
        if tt:
            return tt.strip()
    return "unknown"


# ── Per-run audit computation ────────────────────────────────────────────────


def compute_eos_stats(rows: List[Dict]) -> Dict:
    """Compute EOS position statistics from prediction rows."""
    eos_hit_count = sum(1 for r in rows if r.get("eos_hit", 0))
    eos_positions = [
        r["eos_position"] for r in rows
        if r.get("eos_hit", 0) and isinstance(r.get("eos_position"), (int, float))
    ]
    hit_max_count = sum(1 for r in rows if r.get("hit_max_len", 0))
    n = len(rows)

    return {
        "eos_hit_count": eos_hit_count,
        "eos_hit_pct": round(eos_hit_count / n * 100, 2) if n > 0 else 0.0,
        "hit_max_len_count": hit_max_count,
        "hit_max_len_pct": round(hit_max_count / n * 100, 2) if n > 0 else 0.0,
        "eos_position_stats": compute_length_stats(eos_positions) if eos_positions else {},
    }


def compute_invalid_patterns(rows: List[Dict], top_n: int = 20) -> Dict:
    """Group invalid predictions by their first 20 characters (prefix)."""
    invalid_prefixes = Counter()
    for r in rows:
        if not r.get("rdkit_valid", 0):
            pred = r.get("pred_smiles", "")
            prefix = pred[:20]
            invalid_prefixes[prefix] += 1

    total_invalid = sum(invalid_prefixes.values())
    most_common = invalid_prefixes.most_common(top_n)
    return {
        "total_invalid": total_invalid,
        "top_invalid_patterns": [
            {"prefix": prefix, "count": count,
             "fraction": round(count / total_invalid, 4) if total_invalid > 0 else 0.0}
            for prefix, count in most_common
        ],
    }


def compute_top_bottom_predictions(
    rows: List[Dict], top_n: int = 20,
) -> Dict:
    """Compute top-N closest and worst predictions by Tanimoto."""
    scored = []
    for r in rows:
        scored.append({
            "target_smiles": r.get("target_smiles", ""),
            "pred_smiles": r.get("pred_smiles", ""),
            "tanimoto": r.get("tanimoto", 0.0),
            "rdkit_valid": r.get("rdkit_valid", 0),
        })

    # Top closest: highest Tanimoto (put valid first, then all)
    scored_sorted = sorted(
        scored, key=lambda x: (x["rdkit_valid"], x["tanimoto"]), reverse=True
    )
    top_closest = scored_sorted[:top_n]

    # Worst: lowest Tanimoto among valid, or invalid with 0
    scored_worst = sorted(
        scored, key=lambda x: (x["rdkit_valid"], x["tanimoto"])
    )
    top_worst = scored_worst[:top_n]

    return {
        "top_closest_by_tanimoto": [
            {
                "target": s["target_smiles"],
                "predicted": s["pred_smiles"],
                "tanimoto": s["tanimoto"],
                "valid": s["rdkit_valid"],
            }
            for s in top_closest
        ],
        "top_worst_by_tanimoto": [
            {
                "target": s["target_smiles"],
                "predicted": s["pred_smiles"],
                "tanimoto": s["tanimoto"],
                "valid": s["rdkit_valid"],
            }
            for s in top_worst
        ],
    }


def audit_run(rows: List[Dict]) -> Dict:
    """Compute full generation behavior audit for a single set of predictions."""
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}

    preds = [r["pred_smiles"] for r in rows]
    targets = [r["target_smiles"] for r in rows]
    valid_preds = [p for p in preds if is_valid(p)]
    invalid_preds = [p for p in preds if not is_valid(p)]

    # Target and predicted lengths
    target_lens = [len(s) for s in targets]
    pred_lens = [len(p) for p in preds]

    # Validity rate
    valid_count = len(valid_preds)
    validity_rate = valid_count / n if n > 0 else 0.0

    # Unique stats
    uniq = unique_stats(preds)
    valid_uniq = unique_stats(valid_preds) if valid_preds else {"unique": 0, "unique_ratio": 0.0, "total": 0}

    # Mode collapse
    collapse = mode_collapse_score(preds)
    entropy = prediction_entropy(preds)
    top20 = top_frequencies(preds, n=20)

    # Precision and recall from rows
    precisions = [r.get("fg_precision", 0.0) for r in rows]
    recalls = [r.get("fg_recall", 0.0) for r in rows]
    f1s = [r.get("fg_f1", 0.0) for r in rows]
    mean_fg_f1 = sum(f1s) / n if n > 0 else 0.0

    # Tanimoto stats
    tanimotos = [r.get("tanimoto", 0.0) for r in rows]
    mean_tanimoto = sum(tanimotos) / n if n > 0 else 0.0
    valid_tanimotos = [r["tanimoto"] for r in rows if r.get("rdkit_valid", 0)]
    mean_tanimoto_valid = (sum(valid_tanimotos) / len(valid_tanimotos)
                           if valid_tanimotos else 0.0)

    # Scaffold match rate
    scaffold_count = sum(1 for r in rows if r.get("scaffold_match", 0))
    scaffold_match_rate = scaffold_count / n if n > 0 else 0.0

    # Exact match rates
    exact_count = sum(1 for r in rows if r.get("exact_match", 0))
    exact_match_rate = exact_count / n if n > 0 else 0.0
    canon_exact_count = sum(1 for r in rows if r.get("canonical_exact_match", 0))
    canon_exact_rate = canon_exact_count / n if n > 0 else 0.0

    # Levenshtein
    levenshteins = [r.get("levenshtein", 0) for r in rows]
    mean_levenshtein = sum(levenshteins) / n if n > 0 else 0.0

    # Token accuracy
    tok_accs = [r.get("token_accuracy_sample", 0.0) for r in rows]
    mean_tok_acc = sum(tok_accs) / n if n > 0 else 0.0

    # Build result
    result = {
        "num_samples": n,
        "model_name": auto_detect_model_name(rows),
        "tokenizer_type": auto_detect_tokenizer_type(rows),
        "target_length_stats": compute_length_stats(target_lens),
        "pred_length_stats": compute_length_stats(pred_lens),
        "eos_stats": compute_eos_stats(rows),
        "validity_rate": round(validity_rate, 6),
        "unique_generated": uniq["unique"],
        "unique_ratio": round(uniq["unique_ratio"], 6),
        "unique_valid_generated": valid_uniq["unique"],
        "valid_unique_ratio": round(valid_uniq.get("unique_ratio", 0.0), 6),
        "mode_collapse_score": round(collapse, 6),
        "prediction_entropy": round(entropy, 4),
        "mean_tanimoto": round(mean_tanimoto, 6),
        "mean_tanimoto_valid_only": round(mean_tanimoto_valid, 6),
        "exact_match_rate": round(exact_match_rate, 6),
        "canonical_exact_match_rate": round(canon_exact_rate, 6),
        "scaffold_match_rate": round(scaffold_match_rate, 6),
        "mean_fg_f1": round(mean_fg_f1, 6),
        "mean_levenshtein": round(mean_levenshtein, 2),
        "mean_token_accuracy": round(mean_tok_acc, 6),
        "avg_pred_char_length": round(sum(pred_lens) / n, 2) if n > 0 else 0.0,
        "avg_target_char_length": round(sum(target_lens) / n, 2) if n > 0 else 0.0,
        "validity_by_length": validity_by_length(preds),
        "tanimoto_by_length": tanimoto_by_length(preds, targets),
        "top_20_predictions": [
            {"smiles": s, "count": c, "fraction": round(f, 6)}
            for s, c, f in top20
        ],
        "top_20_generated_smiles": [
            {"smiles": s, "count": c, "fraction": round(f, 6)}
            for s, c, f in top20
        ],
        "invalid_patterns": compute_invalid_patterns(rows),
        "top_bottom_predictions": compute_top_bottom_predictions(rows),
    }
    return result


# ── Per-run markdown report ──────────────────────────────────────────────────


def _fmt_list(items: List[str], indent: str = "  ") -> str:
    return "\n".join(f"{indent}- {item}" for item in items)


def _fmt_pct(val: float) -> str:
    return f"{val * 100:.2f}%"


def format_audit_md(audit: Dict, title: str = "Generation Behavior Audit") -> str:
    """Format audit results as a human-readable markdown report."""
    lines: List[str] = [
        f"# {title}",
        "",
        f"- **Samples**: {audit.get('num_samples', 0)}",
        f"- **Model**: {audit.get('model_name', 'unknown')}",
        f"- **Tokenizer**: {audit.get('tokenizer_type', 'unknown')}",
        "",
    ]

    # Validity and uniqueness
    lines.append("## Overview")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| Validity rate | {_fmt_pct(audit['validity_rate'])} |")
    lines.append(f"| Unique generated | {audit['unique_generated']} / {audit['num_samples']} ({_fmt_pct(audit['unique_ratio'])}) |")
    lines.append(f"| Unique valid generated | {audit['unique_valid_generated']} |")
    lines.append(f"| Mode collapse score | {audit['mode_collapse_score']} |")
    lines.append(f"| Prediction entropy | {audit['prediction_entropy']} |")
    lines.append(f"| Exact match rate | {_fmt_pct(audit['exact_match_rate'])} |")
    lines.append(f"| Canonical exact match rate | {_fmt_pct(audit['canonical_exact_match_rate'])} |")
    lines.append(f"| Scaffold match rate | {_fmt_pct(audit['scaffold_match_rate'])} |")
    lines.append(f"| Mean Tanimoto | {audit['mean_tanimoto']} |")
    lines.append(f"| Mean Tanimoto (valid only) | {audit['mean_tanimoto_valid_only']} |")
    lines.append(f"| Mean FG-F1 | {audit['mean_fg_f1']} |")
    lines.append(f"| Mean Levenshtein | {audit['mean_levenshtein']} |")
    lines.append(f"| Mean token accuracy | {audit['mean_token_accuracy']} |")
    lines.append(f"| Avg pred char length | {audit['avg_pred_char_length']} |")
    lines.append(f"| Avg target char length | {audit['avg_target_char_length']} |")
    lines.append("")

    # Length stats
    lines.append("## Target Length Statistics")
    lines.append("")
    tls = audit.get("target_length_stats", {})
    lines.append(f"| Stat | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| Mean | {tls.get('mean', 'N/A')} |")
    lines.append(f"| Min | {tls.get('min', 'N/A')} |")
    lines.append(f"| Max | {tls.get('max', 'N/A')} |")
    lines.append(f"| P50 | {tls.get('p50', 'N/A')} |")
    lines.append(f"| P90 | {tls.get('p90', 'N/A')} |")
    lines.append(f"| P95 | {tls.get('p95', 'N/A')} |")
    lines.append("")

    lines.append("## Predicted Length Statistics")
    lines.append("")
    pls = audit.get("pred_length_stats", {})
    lines.append(f"| Stat | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| Mean | {pls.get('mean', 'N/A')} |")
    lines.append(f"| Min | {pls.get('min', 'N/A')} |")
    lines.append(f"| Max | {pls.get('max', 'N/A')} |")
    lines.append(f"| P50 | {pls.get('p50', 'N/A')} |")
    lines.append(f"| P90 | {pls.get('p90', 'N/A')} |")
    lines.append(f"| P95 | {pls.get('p95', 'N/A')} |")
    lines.append("")

    # EOS stats
    lines.append("## EOS Statistics")
    lines.append("")
    eos = audit.get("eos_stats", {})
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---:|")
    lines.append(f"| EOS hit count | {eos.get('eos_hit_count', 0)} ({eos.get('eos_hit_pct', 0.0)}%) |")
    lines.append(f"| Hit max len count | {eos.get('hit_max_len_count', 0)} ({eos.get('hit_max_len_pct', 0.0)}%) |")
    epos = eos.get("eos_position_stats", {})
    if epos:
        lines.append(f"| EOS position mean | {epos.get('mean', 'N/A')} |")
        lines.append(f"| EOS position P50 | {epos.get('p50', 'N/A')} |")
        lines.append(f"| EOS position P95 | {epos.get('p95', 'N/A')} |")
    lines.append("")

    # Validity by length
    lines.append("## Validity Rate by Length Bucket")
    lines.append("")
    vbl = audit.get("validity_by_length", {})
    if vbl:
        lines.append(f"| Bucket | Count | Valid | Rate |")
        lines.append(f"|---|---|---:|---:|")
        for bucket, info in vbl.items():
            lines.append(f"| {bucket} | {info['count']} | {info['valid']} | {_fmt_pct(info['rate'])} |")
    else:
        lines.append("_(empty)_")
    lines.append("")

    # Tanimoto by length
    lines.append("## Tanimoto by Length Bucket")
    lines.append("")
    tbl = audit.get("tanimoto_by_length", {})
    if tbl:
        lines.append(f"| Bucket | Count | Mean Tanimoto |")
        lines.append(f"|---|---|---:|")
        for bucket, info in tbl.items():
            lines.append(f"| {bucket} | {info['count']} | {info['mean_tanimoto']} |")
    else:
        lines.append("_(empty)_")
    lines.append("")

    # Top 20 predictions
    lines.append("## Top 20 Most Frequent Generated SMILES")
    lines.append("")
    lines.append(f"| # | SMILES | Count | Fraction |")
    lines.append(f"|---|---:|---:|---:|")
    for i, entry in enumerate(audit.get("top_20_predictions", []), 1):
        lines.append(f"| {i} | `{entry['smiles']}` | {entry['count']} | {entry['fraction']} |")
    lines.append("")

    # Invalid patterns
    lines.append("## Most Common Invalid Patterns (first 20 chars)")
    lines.append("")
    inv = audit.get("invalid_patterns", {})
    total_inv = inv.get("total_invalid", 0)
    lines.append(f"**Total invalid**: {total_inv}")
    lines.append("")
    entries = inv.get("top_invalid_patterns", [])
    if entries:
        lines.append(f"| # | Prefix | Count | Fraction |")
        lines.append(f"|---|---:|---:|---:|")
        for i, entry in enumerate(entries, 1):
            lines.append(f"| {i} | `{entry['prefix']}` | {entry['count']} | {entry['fraction']} |")
    else:
        lines.append("_(no invalid predictions)_")
    lines.append("")

    # Top closest
    lines.append("## Top 20 Closest Predictions by Tanimoto")
    lines.append("")
    tb = audit.get("top_bottom_predictions", {})
    closest = tb.get("top_closest_by_tanimoto", [])
    if closest:
        lines.append(f"| # | Target | Predicted | Tanimoto | Valid |")
        lines.append(f"|---|---|---|---:|---:|")
        for i, entry in enumerate(closest, 1):
            lines.append(f"| {i} | `{entry['target']}` | `{entry['predicted']}` | {entry['tanimoto']} | {entry['valid']} |")
    lines.append("")

    # Top worst
    lines.append("## Top 20 Worst Predictions by Tanimoto")
    lines.append("")
    worst = tb.get("top_worst_by_tanimoto", [])
    if worst:
        lines.append(f"| # | Target | Predicted | Tanimoto | Valid |")
        lines.append(f"|---|---|---|---:|---:|")
        for i, entry in enumerate(worst, 1):
            lines.append(f"| {i} | `{entry['target']}` | `{entry['predicted']}` | {entry['tanimoto']} | {entry['valid']} |")
    lines.append("")

    return "\n".join(lines)


# ── Aggregate reporting ──────────────────────────────────────────────────────


def _fmt_val(v) -> str:
    """Format a numeric value for markdown table, handling None."""
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _parse_model_tag(run_dir: str) -> str:
    """Extract a short model tag from the run directory basename."""
    base = os.path.basename(os.path.normpath(run_dir))
    return base


def _make_comparison_table(
    label_a: str, audit_a: Dict,
    label_b: str, audit_b: Dict,
) -> str:
    """Create a side-by-side comparison table for two audits."""
    rows_text: List[str] = [
        f"### {label_a} vs {label_b}",
        "",
        f"| Metric | {label_a} | {label_b} | Better |",
        f"|---|---:|---:|---:|",
    ]

    comparisons = [
        ("Unique generated", "unique_generated", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Mode collapse score", "mode_collapse_score", lambda a, b, va, vb: label_b if va > vb else label_a if vb > va else "="),
        ("Prediction entropy", "prediction_entropy", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Validity rate", "validity_rate", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Avg pred char length", "avg_pred_char_length", lambda a, b, va, vb: label_a if abs(va - vb) < 0.01 else "diff"),
        ("Avg target char length", "avg_target_char_length", lambda a, b, va, vb: label_a if abs(va - vb) < 0.01 else "diff"),
        ("Exact match rate", "exact_match_rate", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Canonical exact match", "canonical_exact_match_rate", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Scaffold match rate", "scaffold_match_rate", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Mean Tanimoto", "mean_tanimoto", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Mean Tanimoto (valid)", "mean_tanimoto_valid_only", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Mean FG-F1", "mean_fg_f1", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Mean Levenshtein", "mean_levenshtein", lambda a, b, va, vb: label_b if va > vb else label_a if vb > va else "="),
        ("Mean token accuracy", "mean_token_accuracy", lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("EOS hit %", ("eos_stats", "eos_hit_pct"), lambda a, b, va, vb: label_a if va > vb else label_b if vb > va else "="),
        ("Hit max len %", ("eos_stats", "hit_max_len_pct"), lambda a, b, va, vb: label_b if va > vb else label_a if vb > va else "="),
    ]

    for metric_name, key, better_fn in comparisons:
        if isinstance(key, tuple):
            # Nested key path
            val_a = audit_a
            val_b = audit_b
            for k in key:
                val_a = val_a.get(k, {}) if isinstance(val_a, dict) else {}
                val_b = val_b.get(k, {}) if isinstance(val_b, dict) else {}
            if isinstance(val_a, dict) and isinstance(val_b, dict):
                continue
        else:
            val_a = audit_a.get(key, "N/A")
            val_b = audit_b.get(key, "N/A")

        if val_a == "N/A" and val_b == "N/A":
            continue
        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            better = better_fn(label_a, label_b, val_a, val_b)
        else:
            better = ""

        rows_text.append(
            f"| {metric_name} | {_fmt_val(val_a)} | {_fmt_val(val_b)} | {better} |"
        )

    rows_text.append("")
    return "\n".join(rows_text)


def build_aggregate_report(
    run_audits: Dict[str, Dict],
    output_dir: str,
) -> str:
    """Build aggregate comparison report across runs.

    Args:
        run_audits: Mapping of run_name -> audit dict (with split sub-keys).
        output_dir: Output directory for reference.
    """
    lines: List[str] = [
        "# Generation Behavior Audit: Aggregate Report",
        "",
        f"Generated from {len(run_audits)} run(s).",
        "",
        "## Per-Run Summary",
        "",
    ]

    # Per-run summary table
    lines.append("| Run | Split | Samples | Validity | Unique | Unique/Total | Mode Collapse | Entropy | Avg Tanimoto | Avg Pred Len |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for run_name, splits in sorted(run_audits.items()):
        for split_name, audit in sorted(splits.items()):
            if not audit or audit.get("num_samples", 0) == 0:
                continue
            lines.append(
                f"| {run_name} | {split_name} "
                f"| {audit['num_samples']} "
                f"| {_fmt_pct(audit['validity_rate'])} "
                f"| {audit['unique_generated']} "
                f"| {audit['unique_generated']}/{audit['num_samples']} ({_fmt_pct(audit['unique_ratio'])}) "
                f"| {audit['mode_collapse_score']} "
                f"| {audit['prediction_entropy']} "
                f"| {audit['mean_tanimoto']} "
                f"| {audit['avg_pred_char_length']} |"
            )
    lines.append("")

    # ── Direct comparisons: group runs into atom/SPE families ──────────
    lines.append("## Direct Comparisons")
    lines.append("")

    # Group runs by type and seed
    # We look for runs where model_name starts with "atom" or "spe"
    # and has a test split for comparisons.
    atom_runs: Dict[str, Dict] = {}
    spe256_runs: Dict[str, Dict] = {}
    spe512_runs: Dict[str, Dict] = {}

    for run_name, splits in run_audits.items():
        test_audit = splits.get("test")
        if test_audit is None or test_audit.get("num_samples", 0) == 0:
            continue
        model_name = test_audit.get("model_name", "").lower()
        # Determine group from model_name or tokenizer_type
        tok_type = test_audit.get("tokenizer_type", "").lower()
        if "atom" in model_name or tok_type == "regex_atom":
            # Extract seed
            seed = test_audit.get("seed", "?")
            atom_runs[f"{run_name} (seed={seed})"] = test_audit
        elif tok_type == "spe":
            # Determine vocab size from model_name or run_name
            vocab_size = None
            for source_str in [model_name, run_name.lower()]:
                if "256" in source_str:
                    vocab_size = 256
                    break
                elif "512" in source_str:
                    vocab_size = 512
                    break
            if vocab_size == 256:
                seed = test_audit.get("seed", "?")
                spe256_runs[f"{run_name} (seed={seed})"] = test_audit
            elif vocab_size == 512:
                seed = test_audit.get("seed", "?")
                spe512_runs[f"{run_name} (seed={seed})"] = test_audit
            else:
                # Infer from model_name
                seed = test_audit.get("seed", "?")
                spe256_runs[f"{run_name} (seed={seed})"] = test_audit

    def _pair_comparisons(group: Dict[str, Dict], group_label: str) -> List[str]:
        """Generate comparison blocks for a group, finding E0/E1 pairs."""
        items = list(group.items())
        if len(items) < 2:
            return []

        blocks: List[str] = []
        # Try to find seed-based pairs (E0=seed0, E1=seed1)
        e0 = None
        e1 = None
        e0_label = ""
        e1_label = ""
        for label, audit in items:
            seed = audit.get("seed", 0)
            if seed == 0 or seed == 42:
                e0 = audit
                e0_label = label
            else:
                e1 = audit
                e1_label = label

        if e0 is not None and e1 is not None:
            blocks.append(f"### {group_label}")
            blocks.append("")
            blocks.append(_make_comparison_table(e0_label, e0, e1_label, e1))
        elif len(items) >= 2:
            # Just compare first two
            labels = list(group.keys())
            blocks.append(f"### {group_label}")
            blocks.append("")
            blocks.append(_make_comparison_table(labels[0], group[labels[0]], labels[1], group[labels[1]]))
        return blocks

    if atom_runs:
        lines.extend(_pair_comparisons(atom_runs, "Atom (regex_atom) E0 vs E1"))
    if spe256_runs:
        lines.extend(_pair_comparisons(spe256_runs, "SPE-256 E0 vs E1"))
    if spe512_runs:
        lines.extend(_pair_comparisons(spe512_runs, "SPE-512 E0 vs E1"))

    lines.append("---")
    lines.append("")
    lines.append(f"_Report generated by audit_generation_behavior.py_")
    lines.append("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Audit generation behavior across runs."
    )
    parser.add_argument(
        "--run-dirs", nargs="+", required=True,
        help="One or more run directories containing predictions CSV files",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for aggregate report (default: cwd)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    split_names = ["valid", "test"]
    run_audits: Dict[str, Dict] = {}

    for run_dir in args.run_dirs:
        run_dir = os.path.abspath(run_dir)
        run_name = os.path.basename(run_dir)
        print(f"\n{'=' * 60}")
        print(f"Processing: {run_dir}")
        print(f"{'=' * 60}")

        run_audits[run_name] = {}

        for split in split_names:
            csv_path = os.path.join(run_dir, f"predictions_{split}.csv")
            rows = read_predictions(csv_path)

            if not rows:
                print(f"  [{split}] No predictions file or empty: {csv_path}")
                run_audits[run_name][split] = {"num_samples": 0}
                continue

            print(f"  [{split}] Loaded {len(rows)} rows from {csv_path}")
            audit = audit_run(rows)
            run_audits[run_name][split] = audit

            # Save JSON
            json_path = os.path.join(run_dir, f"generation_behavior_audit_{split}.json")
            with open(json_path, "w") as f:
                json.dump(audit, f, indent=2, ensure_ascii=False)
            print(f"  [{split}] Saved JSON: {json_path}")

            # Save MD
            title = f"Generation Behavior Audit - {run_name} ({split})"
            md_content = format_audit_md(audit, title=title)
            md_path = os.path.join(run_dir, f"generation_behavior_audit_{split}.md")
            with open(md_path, "w") as f:
                f.write(md_content)
            print(f"  [{split}] Saved MD: {md_path}")

        # Also save a combined per-run JSON (both splits)
        combined = {
            "run_name": run_name,
            "run_dir": run_dir,
        }
        for split in split_names:
            combined[f"{split}_audit"] = run_audits[run_name].get(split, {"num_samples": 0})

        combined_json_path = os.path.join(run_dir, "generation_behavior_audit.json")
        with open(combined_json_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"  Saved combined JSON: {combined_json_path}")

        # Combined MD per run
        md_lines: List[str] = []
        for split in split_names:
            audit = run_audits[run_name].get(split, {"num_samples": 0})
            if audit.get("num_samples", 0) > 0:
                title = f"{run_name} ({split})"
                md_lines.append(format_audit_md(audit, title=title))
                md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
        combined_md_path = os.path.join(run_dir, "generation_behavior_audit.md")
        with open(combined_md_path, "w") as f:
            f.write("\n".join(md_lines))
        print(f"  Saved combined MD: {combined_md_path}")

    # ── Aggregate report ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Building aggregate report...")
    print(f"{'=' * 60}")

    aggregate_md = build_aggregate_report(run_audits, output_dir)
    aggregate_path = os.path.join(output_dir, "generation_behavior_audit_all.md")
    with open(aggregate_path, "w") as f:
        f.write(aggregate_md)
    print(f"Saved aggregate report: {aggregate_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
