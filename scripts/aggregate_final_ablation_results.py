#!/usr/bin/env python
"""Aggregate final E0 vs E1 ablation results from multiple run directories.

Produces aggregated comparison tables (markdown) and a structured JSON file
containing all raw data for downstream analysis.

Usage:
  python scripts/aggregate_final_ablation_results.py \
    --run-dirs runs/equal_concat_seed42 runs/equal_intra_cross_seed42 \
    --tokenizer-summaries data/processed/spe_vocab_256_summary.json \
    --output-dir reports

Tables produced:
  Table 1: Tokenizer statistics (vocab coverage, length reduction)
  Table 2: Equal-parameter verification (parameter counts, relative diff)
  Table 3: Per-seed test results (all metrics per run)
  Table 4: Mean +/- std over seeds (aggregated per tokenizer)
  Table 5: Condition-shuffle sensitivity (if shuffle data exists)
  Table 6: Architecture verdict (overall winner determination)
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_read_json(path: str) -> Optional[Dict]:
    """Read a JSON file, returning None if missing or corrupt."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] Could not read {path}: {e}", file=sys.stderr)
        return None


def _hardlink_or_copy(src: str, dst: str) -> None:
    """Hardlink or copy a file, silently skipping if source is missing."""
    if not os.path.isfile(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _extract_seed(run_dir: str, config: Optional[Dict]) -> int:
    """Extract seed from config (preferred) or directory name."""
    if config and "seed" in config:
        return int(config["seed"])
    m = re.search(r"seed(\d+)", os.path.basename(run_dir))
    if m:
        return int(m.group(1))
    return 0


def _extract_model_type(config: Optional[Dict], run_dir: str) -> str:
    """Extract model type (concat / intra_cross) from config or dir name."""
    if config:
        model = config.get("model", "")
        if "intra_cross" in model:
            return "intra_cross"
        if "concat" in model:
            return "concat"
    basename = os.path.basename(run_dir).lower()
    if "intra_cross" in basename or "intracross" in basename:
        return "intra_cross"
    if "concat" in basename:
        return "concat"
    # Fallback: assume concat
    return "concat"


def _extract_tokenizer_type(config: Optional[Dict], run_dir: str) -> str:
    """Extract tokenizer type (spe / regex_atom) from config or dir name."""
    if config:
        tok = config.get("tokenizer_type", "")
        if tok:
            return tok
    basename = os.path.basename(run_dir).lower()
    if basename.startswith("spe_") or "_spe_" in basename:
        return "spe"
    return "regex_atom"


def _extract_model_label(model_type: str) -> str:
    """Return short display label for a model type."""
    return {
        "concat": "E0 DirectConcat",
        "intra_cross": "E1 IntraCross",
        "concat_equal": "E0 DirectConcat",
        "intra_cross_equal": "E1 IntraCross",
    }.get(model_type, model_type)


def _extract_model_short(model_type: str) -> str:
    """Return very short label.  'E0' / 'E1'."""
    return "E0" if model_type in ("concat", "concat_equal") else "E1"


def _format_mean_std(values: List[float], higher_is_better: bool = True) -> str:
    """Format mean +/- std for a list of values.

    Handles empty lists, None values, and non-numeric entries gracefully.
    Returns a string like '0.123 +/- 0.045' or 'N/A'.
    """
    numeric = [v for v in values if isinstance(v, (int, float))]
    if not numeric:
        return "N/A"
    mean = sum(numeric) / len(numeric)
    if len(numeric) >= 2:
        variance = sum((x - mean) ** 2 for x in numeric) / len(numeric)
        std = variance ** 0.5
    else:
        std = 0.0
    # Determine precision based on magnitude
    if max(abs(mean), abs(std)) < 0.01:
        return f"{mean:.6f} +/- {std:.6f}"
    return f"{mean:.4f} +/- {std:.4f}"


def _compute_winner(
    e0_values: List[float],
    e1_values: List[float],
    higher_is_better: bool = True,
) -> str:
    """Determine winner between E0 and E1 based on mean values.

    Returns 'E0', 'E1', 'tie', or 'N/A'.
    """
    e0_num = [v for v in e0_values if isinstance(v, (int, float))]
    e1_num = [v for v in e1_values if isinstance(v, (int, float))]
    if not e0_num or not e1_num:
        return "N/A"
    e0_mean = sum(e0_num) / len(e0_num)
    e1_mean = sum(e1_num) / len(e1_num)
    if abs(e0_mean - e1_mean) < 1e-10:
        return "tie"
    if higher_is_better:
        return "E1" if e1_mean > e0_mean else "E0"
    else:
        return "E1" if e1_mean < e0_mean else "E0"


def _compute_confidence(
    e0_values: List[float],
    e1_values: List[float],
    winner: str,
) -> str:
    """Compute confidence level for a winner determination.

    'high' if all seeds agree on the winner direction,
    'medium' if 2 of 3 agree,
    'low' if mixed or insufficient data.
    """
    if winner in ("N/A", "tie"):
        return "N/A"
    n = min(len(e0_values), len(e1_values))
    if n < 2:
        return "low"
    agreements = 0
    total = 0
    for e0, e1 in zip(e0_values, e1_values):
        if not isinstance(e0, (int, float)) or not isinstance(e1, (int, float)):
            continue
        if isinstance(e0, (int, float)) and isinstance(e1, (int, float)):
            if abs(e0 - e1) < 1e-10:
                this_winner = "tie"
                agreements += 0  # ties count as neither agreeing nor disagreeing
            elif (e1 > e0 and winner == "E1") or (e0 > e1 and winner == "E0"):
                this_winner = "E1" if e1 > e0 else "E0"
                agreements += 1 if this_winner == winner else 0
            total += 1
    if total < 2:
        return "low"
    # Redo: compare each seed pair and check if the majority agrees
    e1_wins = 0
    e0_wins = 0
    for e0, e1 in zip(e0_values, e1_values):
        if not isinstance(e0, (int, float)) or not isinstance(e1, (int, float)):
            continue
        if abs(e0 - e1) < 1e-10:
            continue
        if e1 > e0:
            e1_wins += 1
        else:
            e0_wins += 1
    total_non_tie = e0_wins + e1_wins
    if total_non_tie < 1:
        return "low"
    if winner == "E1":
        agree_count = e1_wins
    else:
        agree_count = e0_wins
    ratio = agree_count / total_non_tie
    if ratio >= 1.0 and total_non_tie >= 3:
        return "high"
    elif ratio >= 2.0 / 3.0:
        return "medium"
    else:
        return "low"


# ── Data loading ─────────────────────────────────────────────────────────────


def load_run_data(run_dir: str) -> Dict[str, Any]:
    """Load all available data files from a single run directory.

    Returns a dictionary keyed by data source; keys that fail to load
    will have None values (never raise).
    """
    result: Dict[str, Any] = {
        "_run_dir": run_dir,
        "config_used": None,
        "parameter_count": None,
        "metrics": None,
        "eval_test": None,
        "eval_valid": None,
        "condition_shuffle": None,
        "predictions_test": None,
    }

    result["config_used"] = _safe_read_json(
        os.path.join(run_dir, "config_used.json")
    )
    result["parameter_count"] = _safe_read_json(
        os.path.join(run_dir, "parameter_count.json")
    )
    result["metrics"] = _safe_read_json(
        os.path.join(run_dir, "metrics.json")
    )
    result["eval_test"] = _safe_read_json(
        os.path.join(run_dir, "evaluation_summary_test.json")
    )
    result["eval_valid"] = _safe_read_json(
        os.path.join(run_dir, "evaluation_summary_valid.json")
    )
    result["condition_shuffle"] = _safe_read_json(
        os.path.join(run_dir, "condition_shuffle_summary.json")
    )

    # Load predictions CSV (first 5 rows for reference, plus full row count)
    pred_csv = os.path.join(run_dir, "predictions_test.csv")
    if os.path.isfile(pred_csv):
        try:
            with open(pred_csv, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            result["predictions_test"] = {
                "num_rows": len(rows),
                "columns": list(rows[0].keys()) if rows else [],
                "first_5": rows[:5] if rows else [],
                "all_rows": rows,
            }
        except (csv.Error, OSError) as e:
            print(f"  [WARN] Could not read {pred_csv}: {e}", file=sys.stderr)

    return result


def load_tokenizer_summaries(paths: List[str]) -> List[Dict]:
    """Load tokenizer summary files."""
    summaries = []
    for p in paths:
        data = _safe_read_json(p)
        if data is not None:
            summaries.append(data)
        else:
            print(f"  [WARN] Tokenizer summary not found or invalid: {p}",
                  file=sys.stderr)
    return summaries


# ── Metadata extraction ──────────────────────────────────────────────────────


def _get_run_metadata(run: Dict) -> Dict:
    """Extract metadata (seed, model_type, tokenizer_type) from a run."""
    config = run.get("config_used") or {}
    run_dir = run["_run_dir"]
    return {
        "seed": _extract_seed(run_dir, config),
        "model_type": _extract_model_type(config, run_dir),
        "tokenizer_type": _extract_tokenizer_type(config, run_dir),
        "model_label": _extract_model_label(
            _extract_model_type(config, run_dir)
        ),
        "model_short": _extract_model_short(
            _extract_model_type(config, run_dir)
        ),
    }


# ── Table generation ─────────────────────────────────────────────────────────


def _format_pct(value: Any, decimals: int = 4) -> str:
    """Format a number or ratio as percentage, or return placeholder."""
    if value is None:
        return "-"
    try:
        v = float(value)
        if isinstance(value, float) and 0.0 <= value <= 1.0:
            return f"{v * 100:.2f}%"
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def _format_num(value: Any, decimals: int = 4) -> str:
    """Format a number or return placeholder."""
    if value is None:
        return "-"
    try:
        v = float(value)
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def build_table1_tokenizer_stats(
    tokenizer_summaries: List[Dict],
) -> List[Dict]:
    """Table 1: Tokenizer statistics (vocab coverage, length).

    Each returned dict represents one row.
    """
    rows = []
    for summary in tokenizer_summaries:
        vs = summary.get("vocab_size", "?")
        unk = summary.get("unk_rates", {})
        spe_lens = summary.get("spe_length_stats", {})
        train_spe = spe_lens.get("train", {})
        row = {
            "tokenizer": vs,  # literal vocab size as label
            "vocab_size": vs,
            "train_unk": _format_pct(unk.get("train", "?")),
            "valid_unk": _format_pct(unk.get("valid", "?")),
            "test_unk": _format_pct(unk.get("test", "?")),
            "mean_token_len": _format_num(train_spe.get("mean", "?")),
            "p95_token_len": _format_num(train_spe.get("p95", "?")),
            "max_token_len": train_spe.get("max", "?"),
        }
        rows.append(row)
    if not rows:
        rows.append({
            "tokenizer": "regex_atom",
            "vocab_size": "-",
            "train_unk": "N/A",
            "valid_unk": "N/A",
            "test_unk": "N/A",
            "mean_token_len": "N/A",
            "p95_token_len": "N/A",
            "max_token_len": "N/A",
        })
    return rows


def build_table2_param_verification(
    runs_by_tok: Dict[str, List[Dict]],
) -> List[Dict]:
    """Table 2: Equal-parameter verification.

    Groups by tokenizer and finds E0/E1 pairs with matching seeds
    to compute relative parameter differences.
    """
    rows = []
    for tokenizer, tokenizer_runs in sorted(runs_by_tok.items()):
        # Collect parameter counts per model per seed
        e0_params = []
        e1_params = []
        e0_sources = []
        e1_sources = []

        for run in tokenizer_runs:
            meta = _get_run_metadata(run)
            pc = run.get("parameter_count")
            total = None
            by_module = {}
            if pc:
                total = pc.get("total_params")
                by_module = pc.get("by_module", {})
            elif run.get("metrics"):
                total = run["metrics"].get("n_params")
            elif run.get("config_used"):
                total = run["config_used"].get("n_params")

            if meta["model_short"] == "E0":
                e0_params.append(total)
                e0_sources.append((run, total, by_module))
            else:
                e1_params.append(total)
                e1_sources.append((run, total, by_module))

        # If we have exactly one E0 and one E1, compute relative diff
        if e0_sources and e1_sources:
            for e0_run, e0_total, e0_mod in e0_sources:
                for e1_run, e1_total, e1_mod in e1_sources:
                    if e0_total is None or e1_total is None:
                        rel_diff = "N/A"
                    elif e0_total == 0:
                        rel_diff = "N/A"
                    else:
                        rel_diff = f"{abs(e1_total - e0_total) / e0_total * 100:.4f}%"

                    # Gather per-module breakdown
                    decoder_params_e0 = e0_mod.get("decoder", "N/A") if e0_mod else "N/A"
                    decoder_params_e1 = e1_mod.get("decoder", "N/A") if e1_mod else "N/A"

                    e0_meta = _get_run_metadata(e0_run)
                    e1_meta = _get_run_metadata(e1_run)

                    rows.append({
                        "tokenizer": tokenizer,
                        "model_e0": e0_meta["model_label"],
                        "model_e1": e1_meta["model_label"],
                        "total_params_e0": f"{e0_total:,}" if e0_total else "N/A",
                        "total_params_e1": f"{e1_total:,}" if e1_total else "N/A",
                        "decoder_params_e0": f"{decoder_params_e0:,}" if isinstance(decoder_params_e0, int) else str(decoder_params_e0),
                        "decoder_params_e1": f"{decoder_params_e1:,}" if isinstance(decoder_params_e1, int) else str(decoder_params_e1),
                        "relative_diff": rel_diff,
                    })
                    break  # one row per tokenizer is enough

    return rows


def build_table3_per_seed(
    runs_by_tok: Dict[str, List[Dict]],
) -> List[Dict]:
    """Table 3: Per-seed test results.

    One row per run, sorted by tokenizer -> model -> seed.
    """
    rows = []
    for tokenizer in sorted(runs_by_tok):
        tokenizer_runs = runs_by_tok[tokenizer]
        for run in tokenizer_runs:
            meta = _get_run_metadata(run)
            eval_test = run.get("eval_test") or {}
            metrics = run.get("metrics") or {}
            config = run.get("config_used") or {}

            # Loss from eval_test (preferred) or metrics
            loss = eval_test.get("eval_loss") or eval_test.get("loss") or metrics.get("test_loss")
            # Token accuracy from eval_test or metrics
            token_acc = eval_test.get("token_acc") or eval_test.get("accuracy") or metrics.get("test_token_acc")

            # For mode_collapse_score and unique_ratio, check eval_test
            mode_collapse = eval_test.get("mode_collapse_score")
            unique_ratio = eval_test.get("unique_ratio")
            scaffold_match = eval_test.get("scaffold_match_rate")
            mean_tanimoto = eval_test.get("mean_tanimoto")
            fg_f1 = eval_test.get("mean_fg_f1")

            seed = meta["seed"]
            model_label = meta["model_label"]

            row = {
                "tokenizer": tokenizer,
                "seed": seed,
                "model": model_label,
                "loss": _format_num(loss),
                "token_acc": _format_pct(token_acc, decimals=4),
                "exact_match": _format_pct(eval_test.get("exact_string_match")),
                "canon_exact": _format_pct(eval_test.get("canonical_exact_match")),
                "validity": _format_pct(eval_test.get("rdkit_validity")),
                "unique_ratio": _format_pct(unique_ratio),
                "mode_collapse": _format_num(mode_collapse),
                "tanimoto": _format_num(mean_tanimoto),
                "scaffold": _format_pct(scaffold_match),
                "fg_f1": _format_num(fg_f1),
                "avg_char_len": _format_num(eval_test.get("avg_pred_char_length"), decimals=2),
            }
            rows.append(row)

    return rows


def build_table4_mean_std(
    runs_by_tok_model: Dict[str, Dict[str, List[Dict]]],
) -> List[Dict]:
    """Table 4: Mean +/- std over seeds for each tokenizer.

    Groups runs by (tokenizer, model_short) and aggregates.
    """
    rows = []
    metric_fields: List[Tuple[str, str, bool]] = [
        ("loss", "Loss", False),
        ("token_acc", "Token Accuracy", True),
        ("exact_match", "Exact String Match", True),
        ("canon_exact", "Canonical Exact Match", True),
        ("validity", "RDKit Validity", True),
        ("unique_ratio", "Unique Ratio", True),
        ("mode_collapse", "Mode Collapse", False),
        ("tanimoto", "Tanimoto Similarity", True),
        ("scaffold", "Scaffold Match", True),
        ("fg_f1", "FG-F1", True),
        ("avg_char_len", "Avg Char Length", None),  # direction is ambiguous
    ]

    for tokenizer in sorted(runs_by_tok_model):
        by_model = runs_by_tok_model[tokenizer]
        e0_runs = by_model.get("E0", [])
        e1_runs = by_model.get("E1", [])

        if not e0_runs or not e1_runs:
            continue

        for field_key, field_label, higher_is_better in metric_fields:
            e0_values = []
            e1_values = []

            for run in e0_runs:
                meta = _get_run_metadata(run)
                eval_test = run.get("eval_test") or {}
                metrics = run.get("metrics") or {}

                if field_key == "loss":
                    v = eval_test.get("eval_loss") or eval_test.get("loss") or metrics.get("test_loss")
                elif field_key == "token_acc":
                    v = eval_test.get("token_acc") or eval_test.get("accuracy") or metrics.get("test_token_acc")
                elif field_key == "exact_match":
                    v = eval_test.get("exact_string_match")
                elif field_key == "canon_exact":
                    v = eval_test.get("canonical_exact_match")
                elif field_key == "validity":
                    v = eval_test.get("rdkit_validity")
                elif field_key == "unique_ratio":
                    v = eval_test.get("unique_ratio")
                elif field_key == "mode_collapse":
                    v = eval_test.get("mode_collapse_score")
                elif field_key == "tanimoto":
                    v = eval_test.get("mean_tanimoto")
                elif field_key == "scaffold":
                    v = eval_test.get("scaffold_match_rate")
                elif field_key == "fg_f1":
                    v = eval_test.get("mean_fg_f1")
                elif field_key == "avg_char_len":
                    v = eval_test.get("avg_pred_char_length")
                else:
                    v = None

                if v is not None:
                    e0_values.append(float(v))

            for run in e1_runs:
                meta = _get_run_metadata(run)
                eval_test = run.get("eval_test") or {}
                metrics = run.get("metrics") or {}

                if field_key == "loss":
                    v = eval_test.get("eval_loss") or eval_test.get("loss") or metrics.get("test_loss")
                elif field_key == "token_acc":
                    v = eval_test.get("token_acc") or eval_test.get("accuracy") or metrics.get("test_token_acc")
                elif field_key == "exact_match":
                    v = eval_test.get("exact_string_match")
                elif field_key == "canon_exact":
                    v = eval_test.get("canonical_exact_match")
                elif field_key == "validity":
                    v = eval_test.get("rdkit_validity")
                elif field_key == "unique_ratio":
                    v = eval_test.get("unique_ratio")
                elif field_key == "mode_collapse":
                    v = eval_test.get("mode_collapse_score")
                elif field_key == "tanimoto":
                    v = eval_test.get("mean_tanimoto")
                elif field_key == "scaffold":
                    v = eval_test.get("scaffold_match_rate")
                elif field_key == "fg_f1":
                    v = eval_test.get("mean_fg_f1")
                elif field_key == "avg_char_len":
                    v = eval_test.get("avg_pred_char_length")
                else:
                    v = None

                if v is not None:
                    e1_values.append(float(v))

            winner = _compute_winner(e0_values, e1_values, higher_is_better)
            confidence = _compute_confidence(e0_values, e1_values, winner)

            rows.append({
                "tokenizer": tokenizer,
                "metric": field_label,
                "e0_mean_std": _format_mean_std(e0_values),
                "e1_mean_std": _format_mean_std(e1_values),
                "winner": winner,
                "confidence": confidence,
            })

    return rows


def build_table5_shuffle_sensitivity(
    runs_by_tok: Dict[str, List[Dict]],
) -> List[Dict]:
    """Table 5: Condition-shuffle sensitivity (if data exists).

    Reads condition_shuffle_summary.json from each run.
    """
    rows = []
    for tokenizer in sorted(runs_by_tok):
        tokenizer_runs = runs_by_tok[tokenizer]
        for run in tokenizer_runs:
            meta = _get_run_metadata(run)
            shuffle_data = run.get("condition_shuffle")
            if shuffle_data is None:
                continue

            # The shuffle data could be structured in various ways.
            # Try two known formats:
            # Format A: list of dicts with seed/model/metric/paired/shuffled/drop/winner
            # Format B: dict keyed by metric name with values
            if isinstance(shuffle_data, list):
                for entry in shuffle_data:
                    row = {
                        "tokenizer": tokenizer,
                        "seed": meta["seed"],
                        "model": meta["model_label"],
                        "metric": entry.get("metric", "?"),
                        "paired": _format_num(entry.get("paired")),
                        "shuffled_all": _format_num(entry.get("shuffled_all") or entry.get("shuffled")),
                        "drop": _format_num(entry.get("drop")),
                        "winner": entry.get("winner", "?"),
                    }
                    rows.append(row)
            elif isinstance(shuffle_data, dict):
                for metric_name, metric_data in shuffle_data.items():
                    if isinstance(metric_data, dict):
                        row = {
                            "tokenizer": tokenizer,
                            "seed": meta["seed"],
                            "model": meta["model_label"],
                            "metric": metric_name,
                            "paired": _format_num(metric_data.get("paired")),
                            "shuffled_all": _format_num(metric_data.get("shuffled_all") or metric_data.get("shuffled")),
                            "drop": _format_num(metric_data.get("drop")),
                            "winner": metric_data.get("winner", "?"),
                        }
                        rows.append(row)
                    else:
                        rows.append({
                            "tokenizer": tokenizer,
                            "seed": meta["seed"],
                            "model": meta["model_label"],
                            "metric": metric_name,
                            "paired": "?",
                            "shuffled_all": "?",
                            "drop": "?",
                            "winner": "?",
                        })
    return rows


def build_table6_verdict(
    table4_rows: List[Dict],
) -> List[Dict]:
    """Table 6: Architecture verdict.

    Summarizes which model wins each criterion with confidence.
    """
    verdict_criteria: List[Tuple[str, str, bool]] = [
        ("Optimization (loss / token acc)", "Loss", False),
        ("Exact / canonical exact match", "Exact String Match", True),
        ("RDKit Validity", "RDKit Validity", True),
        ("Diversity (unique ratio / collapse)", "Unique Ratio", True),
        ("Target similarity (Tanimoto)", "Tanimoto Similarity", True),
        ("Condition sensitivity", "Condition Sensitivity", True),
    ]

    # Build a lookup by metric label
    metric_map: Dict[str, Dict] = {}
    for row in table4_rows:
        metric_label = row["metric"]
        if metric_label not in metric_map:
            metric_map[metric_label] = row
        # Prefer the first entry per tokenizer for the verdict table

    rows = []
    win_counts: Dict[str, int] = {"E0": 0, "E1": 0, "tie": 0}
    for criterion, metric_key, higher_is_better in verdict_criteria:
        found = metric_map.get(metric_key)
        if found:
            winner_str = found.get("winner", "?")
            confidence = found.get("confidence", "?")
            rows.append({
                "criterion": criterion,
                "e0": found.get("e0_mean_std", ""),
                "e1": found.get("e1_mean_std", ""),
                "winner": winner_str,
                "confidence": confidence,
            })
            if winner_str in win_counts:
                win_counts[winner_str] += 1
        else:
            rows.append({
                "criterion": criterion,
                "e0": "N/A",
                "e1": "N/A",
                "winner": "N/A",
                "confidence": "N/A",
            })

    # Overall row
    overall_winner = max(win_counts, key=win_counts.get) if win_counts else "tie"
    rows.append({
        "criterion": "Overall",
        "e0": "",
        "e1": "",
        "winner": overall_winner if overall_winner != "tie" else "tie (needs more data)",
        "confidence": "low" if overall_winner in ("tie", "N/A") else "medium",
    })

    return rows


# ── Markdown rendering ───────────────────────────────────────────────────────


def write_markdown(
    output_path: str,
    runs_by_tok: Dict[str, List[Dict]],
    runs_by_tok_model: Dict[str, Dict[str, List[Dict]]],
    tokenizer_summaries: List[Dict],
    run_dirs: List[str],
) -> None:
    """Write the complete markdown report."""
    lines: List[str] = [
        "# Final E0 vs E1 Ablation Comparison Tables",
        "",
        f"**Generated:** 2026-05-21",
        f"**Run directories:** {', '.join(run_dirs)}",
        "",
    ]

    # ── Table 1: Tokenizer Statistics ──────────────────────────────────────
    lines.extend([
        "---",
        "",
        "## Table 1. Tokenizer Statistics",
        "",
        "| tokenizer | vocab size | train unk | valid unk | test unk | "
        "mean token len (train) | p95 token len | max token len |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    t1_rows = build_table1_tokenizer_stats(tokenizer_summaries)
    for r in t1_rows:
        lines.append(
            f"| {r['tokenizer']} | {r['vocab_size']} | {r['train_unk']} | "
            f"{r['valid_unk']} | {r['test_unk']} | {r['mean_token_len']} | "
            f"{r['p95_token_len']} | {r['max_token_len']} |"
        )
    if not t1_rows:
        lines.append("| _No tokenizer data_ | | | | | | |")
    lines.append("")

    # ── Table 2: Equal-Parameter Verification ────────────────────────────
    lines.extend([
        "---",
        "",
        "## Table 2. Equal-Parameter Verification",
        "",
        "| tokenizer | model | total params | decoder params | "
        "encoder params | relative diff vs pair |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    t2_rows = build_table2_param_verification(runs_by_tok)
    for r in t2_rows:
        lines.append(
            f"| {r['tokenizer']} | {r['model_e0']} / {r['model_e1']} | "
            f"{r['total_params_e0']} / {r['total_params_e1']} | "
            f"{r['decoder_params_e0']} / {r['decoder_params_e1']} | "
            f"(see by_module) | {r['relative_diff']} |"
        )
    if not t2_rows:
        lines.append("| _No parameter data_ | | | | |")
    lines.append("")

    # ── Table 3: Per-Seed Test Results ──────────────────────────────────
    lines.extend([
        "---",
        "",
        "## Table 3. Per-Seed Test Results",
        "",
        "| tokenizer | seed | model | loss | token acc | "
        "canon exact | validity | unique ratio | mode collapse | "
        "Tanimoto | scaffold | FG-F1 | avg char len |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    t3_rows = build_table3_per_seed(runs_by_tok)
    for r in t3_rows:
        lines.append(
            f"| {r['tokenizer']} | {r['seed']} | {r['model']} | "
            f"{r['loss']} | {r['token_acc']} | {r['canon_exact']} | "
            f"{r['validity']} | {r['unique_ratio']} | {r['mode_collapse']} | "
            f"{r['tanimoto']} | {r['scaffold']} | {r['fg_f1']} | {r['avg_char_len']} |"
        )
    if not t3_rows:
        lines.append("| _No evaluation results_ | | | | | | | | | | | |")
    lines.append("")

    # ── Table 4: Mean +/- Std Over Seeds ────────────────────────────────
    lines.extend([
        "---",
        "",
        "## Table 4. Mean +/- Std Over Seeds (for each tokenizer)",
        "",
        "| tokenizer | metric | E0 mean +/- std | E1 mean +/- std | winner | confidence |",
        "|---|---:|---:|---:|---:|",
    ])
    t4_rows = build_table4_mean_std(runs_by_tok_model)
    for r in t4_rows:
        lines.append(
            f"| {r['tokenizer']} | {r['metric']} | "
            f"{r['e0_mean_std']} | {r['e1_mean_std']} | "
            f"{r['winner']} | {r['confidence']} |"
        )
    if not t4_rows:
        lines.append("| _Insufficient data for aggregation_ | | | | |")
    lines.append("")

    # ── Table 5: Condition-Shuffle Sensitivity ──────────────────────────
    t5_rows = build_table5_shuffle_sensitivity(runs_by_tok)
    if t5_rows:
        lines.extend([
            "---",
            "",
            "## Table 5. Condition-Shuffle Sensitivity",
            "",
            "| tokenizer | seed | model | metric | paired | shuffled all | drop | winner |",
            "|---|---:|---|---:|---:|---:|---:|",
        ])
        for r in t5_rows:
            lines.append(
                f"| {r['tokenizer']} | {r['seed']} | {r['model']} | "
                f"{r['metric']} | {r['paired']} | {r['shuffled_all']} | "
                f"{r['drop']} | {r['winner']} |"
            )
        lines.append("")

    # ── Table 6: Architecture Verdict ─────────────────────────────────────
    lines.extend([
        "---",
        "",
        "## Table 6. Architecture Verdict",
        "",
        "| criterion | E0 DirectConcat | E1 IntraCross | winner | confidence |",
        "|---|---:|---:|---:|",
    ])
    t6_rows = build_table6_verdict(t4_rows)
    for r in t6_rows:
        if r["criterion"] == "Overall":
            lines.append(f"| **{r['criterion']}** | {r['e0']} | {r['e1']} | "
                         f"**{r['winner']}** | {r['confidence']} |")
        else:
            lines.append(
                f"| {r['criterion']} | {r['e0']} | {r['e1']} | "
                f"{r['winner']} | {r['confidence']} |"
            )
    lines.append("")

    # ── Notes ─────────────────────────────────────────────────────────────
    lines.extend([
        "---",
        "",
        "## Notes",
        "",
        "- Token accuracy is comparable between E0 and E1 under the same "
        "tokenizer, but NOT across different tokenizers.",
        "- Winner determination: E1 wins if mean > E0 mean (for most metrics, "
        "higher is better; for loss and mode_collapse, lower is better).",
        '- Confidence: "high" if all 3 seeds agree, "medium" if 2 of 3 agree, '
        '"low" if mixed or insufficient data.',
        ""
        "- Metrics marked '-' were not available for this run.",
    ])
    lines.append("")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {output_path}")


# ── JSON output ──────────────────────────────────────────────────────────────


def write_json(
    output_path: str,
    runs: List[Dict],
    runs_by_tok: Dict[str, List[Dict]],
    runs_by_tok_model: Dict[str, Dict[str, List[Dict]]],
    tokenizer_summaries: List[Dict],
    run_dirs: List[str],
) -> None:
    """Write the structured JSON output with all tables and raw data."""
    output: Dict[str, Any] = {
        "generated": "2026-05-21",
        "run_dirs": run_dirs,
        "tables": {
            "tokenizer_statistics": build_table1_tokenizer_stats(
                tokenizer_summaries
            ),
            "parameter_verification": build_table2_param_verification(
                runs_by_tok
            ),
            "per_seed_results": build_table3_per_seed(runs_by_tok),
            "mean_std_over_seeds": build_table4_mean_std(runs_by_tok_model),
            "condition_shuffle_sensitivity": build_table5_shuffle_sensitivity(
                runs_by_tok
            ),
            "architecture_verdict": build_table6_verdict(
                build_table4_mean_std(runs_by_tok_model)
            ),
        },
    }

    # Include raw per-run data
    raw_runs = []
    for run in runs:
        meta = _get_run_metadata(run)
        raw_entry: Dict[str, Any] = {
            "run_dir": run["_run_dir"],
            "seed": meta["seed"],
            "model_type": meta["model_type"],
            "tokenizer_type": meta["tokenizer_type"],
        }
        if run.get("metrics"):
            raw_entry["metrics"] = run["metrics"]
        if run.get("eval_test"):
            raw_entry["evaluation_summary_test"] = run["eval_test"]
        if run.get("eval_valid"):
            raw_entry["evaluation_summary_valid"] = run["eval_valid"]
        if run.get("parameter_count"):
            raw_entry["parameter_count"] = run["parameter_count"]
        if run.get("config_used"):
            raw_entry["config_used"] = run["config_used"]
        if run.get("predictions_test"):
            preds = run["predictions_test"]
            raw_entry["predictions_test"] = {
                "num_rows": preds["num_rows"],
                "columns": preds["columns"],
            }
        raw_runs.append(raw_entry)

    output["raw_runs"] = raw_runs

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate E0 vs E1 ablation results from run directories."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="List of run directories to aggregate",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Where to save the report files (default: reports/)",
    )
    parser.add_argument(
        "--tokenizer-summaries",
        nargs="*",
        default=[],
        help="List of SPE vocab summary JSON files",
    )
    parser.add_argument(
        "--copy-predictions",
        action="store_true",
        default=False,
        help="Copy predictions CSVs to output dir for reference",
    )
    args = parser.parse_args()

    # Validate run directories
    valid_dirs = []
    for d in args.run_dirs:
        if os.path.isdir(d):
            valid_dirs.append(d)
        else:
            print(f"  [WARN] Run directory not found: {d}", file=sys.stderr)

    if not valid_dirs:
        print("ERROR: No valid run directories provided.", file=sys.stderr)
        return 1

    # Load data
    print(f"Loading {len(valid_dirs)} run directories...")
    runs = []
    for d in valid_dirs:
        run_data = load_run_data(d)
        meta = _get_run_metadata(run_data)
        print(f"  {os.path.basename(d)}: "
              f"model={meta['model_type']}, "
              f"tok={meta['tokenizer_type']}, "
              f"seed={meta['seed']}")
        runs.append(run_data)

    # Load tokenizer summaries
    tokenizer_summaries = load_tokenizer_summaries(args.tokenizer_summaries)
    if tokenizer_summaries:
        for s in tokenizer_summaries:
            print(f"  Tokenizer summary: vocab_size={s.get('vocab_size')}")
    else:
        print("  No tokenizer summaries loaded.")

    # Group runs by tokenizer
    runs_by_tok: Dict[str, List[Dict]] = defaultdict(list)
    for run in runs:
        meta = _get_run_metadata(run)
        runs_by_tok[meta["tokenizer_type"]].append(run)

    # Group runs by (tokenizer, model_short)
    runs_by_tok_model: Dict[str, Dict[str, List[Dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in runs:
        meta = _get_run_metadata(run)
        runs_by_tok_model[meta["tokenizer_type"]][meta["model_short"]].append(
            run
        )

    print(f"\nGrouped into {len(runs_by_tok)} tokenizer types:")
    for tok, tok_runs in sorted(runs_by_tok.items()):
        models = set(_get_run_metadata(r)["model_short"] for r in tok_runs)
        print(f"  {tok}: {len(tok_runs)} runs, models={sorted(models)}")

    # Write reports
    os.makedirs(args.output_dir, exist_ok=True)
    md_path = os.path.join(
        args.output_dir, "final_e0_e1_comparison_tables.md"
    )
    json_path = os.path.join(args.output_dir, "final_e0_e1_results.json")

    write_markdown(
        md_path,
        dict(runs_by_tok),
        dict(runs_by_tok_model),
        tokenizer_summaries,
        valid_dirs,
    )
    write_json(
        json_path,
        runs,
        dict(runs_by_tok),
        dict(runs_by_tok_model),
        tokenizer_summaries,
        valid_dirs,
    )

    # Optionally copy predictions CSVs
    if args.copy_predictions:
        preds_dir = os.path.join(args.output_dir, "predictions")
        for run in runs:
            meta = _get_run_metadata(run)
            src = os.path.join(run["_run_dir"], "predictions_test.csv")
            dst_name = (
                f"predictions_{meta['tokenizer_type']}_"
                f"{meta['model_short']}_seed{meta['seed']}.csv"
            )
            dst = os.path.join(preds_dir, dst_name)
            _hardlink_or_copy(src, dst)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
