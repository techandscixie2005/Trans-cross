"""Chemical metrics for SMILES generation evaluation.

Provides RDKit-based validity, Tanimoto similarity, scaffold matching,
functional group F1, and string-level metrics (Levenshtein, length stats).
All functions handle invalid SMILES gracefully.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Scaffolds
    _HAS_RDKIT = True
except ImportError:
    _HAS_RDKIT = False


# ── SMARTS patterns for common functional groups ──────────────────────────

_FUNCTIONAL_GROUP_SMARTS: Dict[str, str] = {
    "alcohol": "[OX2H][CX4;!$(C(=O))]",
    "phenol": "[OX2H]c",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[CX3](=O)[NX3H0,H1,H2]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
    "nitrile": "[CX2]#N",
    "nitro": "[NX3](=O)=O",
    "halide": "[F,Cl,Br,I]",
    "aromatic_ring": "a1aaaaa1",
    "ether": "[OD2]([#6])[#6]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "sulfonamide": "[SX4](=O)(=O)[NX3]",
}


def _get_fg_smarts() -> Dict[str, str]:
    return _FUNCTIONAL_GROUP_SMARTS


# ── RDKit utilities ────────────────────────────────────────────────────────


def canonicalize(smi: str) -> Optional[str]:
    """Return canonical SMILES, or None if invalid."""
    if not _HAS_RDKIT or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def is_valid(smi: str) -> bool:
    """Check if a SMILES string is RDKit-valid."""
    if not _HAS_RDKIT or not smi:
        return False
    return Chem.MolFromSmiles(smi) is not None


def get_num_atoms(smi: str) -> Optional[int]:
    """Return heavy atom count, or None if invalid."""
    if not _HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return mol.GetNumHeavyAtoms()


# ── Morgan fingerprint Tanimoto ────────────────────────────────────────────


def compute_tanimoto(
    target_smi: str,
    pred_smi: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Compute Morgan fingerprint Tanimoto similarity.

    Returns:
        Similarity in [0, 1]. Returns 0.0 if either molecule is invalid.
    """
    if not _HAS_RDKIT:
        return 0.0
    mol_t = Chem.MolFromSmiles(target_smi)
    mol_p = Chem.MolFromSmiles(pred_smi)
    if mol_t is None or mol_p is None:
        return 0.0
    fp_t = AllChem.GetMorganFingerprintAsBitVect(mol_t, radius, nBits=n_bits)
    fp_p = AllChem.GetMorganFingerprintAsBitVect(mol_p, radius, nBits=n_bits)
    return DataStructs.TanimotoSimilarity(fp_t, fp_p)


# ── Bemis-Murcko scaffold ──────────────────────────────────────────────────


def get_scaffold(smi: str) -> Optional[str]:
    """Return canonical Bemis-Murcko scaffold SMILES, or None if invalid."""
    if not _HAS_RDKIT or not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        scaffold = Scaffolds.MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=False
        )
        return canonicalize(scaffold)
    except Exception:
        return None


def scaffold_match(target_smi: str, pred_smi: str) -> int:
    """Return 1 if Bemis-Murcko scaffolds match, 0 otherwise."""
    if not _HAS_RDKIT:
        return 0
    t_scaffold = get_scaffold(target_smi)
    p_scaffold = get_scaffold(pred_smi)
    if t_scaffold is None or p_scaffold is None:
        return 0
    return 1 if t_scaffold == p_scaffold else 0


# ── Functional group detection ─────────────────────────────────────────────


def detect_functional_groups(smi: str) -> Dict[str, int]:
    """Count occurrences of each functional group in a SMILES.

    Returns:
        Dict mapping group name to count. Empty dict if invalid.
    """
    if not _HAS_RDKIT or not smi:
        return {}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    counts: Dict[str, int] = {}
    for name, smarts in _FUNCTIONAL_GROUP_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            counts[name] = 0
            continue
        matches = mol.GetSubstructMatches(pattern)
        counts[name] = len(matches)
    return counts


def functional_group_f1(
    target_smi: str,
    pred_smi: str,
) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 for functional group detection.

    Returns:
        (precision, recall, f1) tuple. All 0.0 if either molecule is invalid.
    """
    if not _HAS_RDKIT:
        return 0.0, 0.0, 0.0
    target_fg = detect_functional_groups(target_smi)
    pred_fg = detect_functional_groups(pred_smi)
    if not target_fg and not pred_fg:
        return 1.0, 1.0, 1.0

    tp = 0  # true positives at the group-presence level
    fp = 0
    fn = 0
    for name in set(list(target_fg.keys()) + list(pred_fg.keys())):
        t_present = target_fg.get(name, 0) > 0
        p_present = pred_fg.get(name, 0) > 0
        if t_present and p_present:
            tp += 1
        elif p_present and not t_present:
            fp += 1
        elif t_present and not p_present:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ── Levenshtein distance ────────────────────────────────────────────────────


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein (edit) distance via DP in O(len(a)*len(b))."""
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr.append(prev[j - 1])
            else:
                curr.append(1 + min(prev[j], curr[-1], prev[j - 1]))
        prev = curr
    return prev[-1]


# ── Length statistics ──────────────────────────────────────────────────────


def compute_length_stats(lengths: List[int]) -> Dict:
    """Compute summary statistics for a list of lengths."""
    if not lengths:
        return {"mean": 0, "min": 0, "max": 0, "p50": 0, "p90": 0, "p95": 0, "p99": 0}
    n = len(lengths)
    sorted_lens = sorted(lengths)
    return {
        "mean": round(sum(lengths) / n, 2),
        "min": sorted_lens[0],
        "max": sorted_lens[-1],
        "p50": sorted_lens[int(n * 0.50)],
        "p90": sorted_lens[int(n * 0.90)],
        "p95": sorted_lens[int(n * 0.95)],
        "p99": sorted_lens[int(n * 0.99)],
    }


def length_distribution(lengths: List[int], num_buckets: int = 10) -> Dict:
    """Return histogram of length distribution."""
    if not lengths:
        return {}
    min_l, max_l = min(lengths), max(lengths)
    if min_l == max_l:
        return {str(min_l): len(lengths)}
    step = max(1, (max_l - min_l) // num_buckets)
    dist: Dict[str, int] = {}
    for l in lengths:
        bucket = (l // step) * step
        key = f"{bucket}-{bucket + step - 1}"
        dist[key] = dist.get(key, 0) + 1
    return dist


# ── Mode collapse metrics ──────────────────────────────────────────────────


def mode_collapse_score(predictions: List[str]) -> float:
    """Return max_frequency / num_samples.
    Higher = more collapse. 1.0 = all outputs identical.
    """
    if not predictions:
        return 1.0
    freq = Counter(predictions)
    return freq.most_common(1)[0][1] / len(predictions)


def prediction_entropy(predictions: List[str]) -> float:
    """Shannon entropy of generated SMILES distribution."""
    if not predictions:
        return 0.0
    freq = Counter(predictions)
    n = len(predictions)
    entropy = 0.0
    for count in freq.values():
        p = count / n
        entropy -= p * math.log(p)
    return entropy


def top_frequencies(predictions: List[str], n: int = 20) -> List[Tuple[str, int, float]]:
    """Return top-N most frequent predictions with counts and fractions."""
    freq = Counter(predictions)
    total = len(predictions)
    return [(smi, count, count / total) for smi, count in freq.most_common(n)]


def unique_stats(predictions: List[str]) -> Dict:
    """Compute unique count and ratio."""
    n = len(predictions)
    n_unique = len(set(predictions))
    return {
        "total": n,
        "unique": n_unique,
        "unique_ratio": n_unique / n if n > 0 else 0.0,
    }


# ── Validity by length bucket ──────────────────────────────────────────────


def validity_by_length(predictions: List[str], num_buckets: int = 5) -> Dict:
    """Compute validity rate bucketed by SMILES character length."""
    if not predictions:
        return {}
    lens = [len(p) for p in predictions]
    min_l, max_l = min(lens), max(lens)
    if min_l == max_l:
        bucket_key = str(min_l)
        valid_count = sum(1 for p in predictions if is_valid(p))
        return {bucket_key: {"count": len(predictions), "valid": valid_count,
                              "rate": valid_count / len(predictions)}}

    step = max(1, (max_l - min_l) // num_buckets)
    buckets: Dict = {}
    for p in predictions:
        bucket = (len(p) // step) * step
        key = f"{bucket}-{bucket + step - 1}"
        if key not in buckets:
            buckets[key] = {"count": 0, "valid": 0}
        buckets[key]["count"] += 1
        if is_valid(p):
            buckets[key]["valid"] += 1
    for v in buckets.values():
        v["rate"] = v["valid"] / v["count"] if v["count"] > 0 else 0.0
    return dict(sorted(buckets.items(),
                       key=lambda x: int(x[0].split("-")[0])))


def tanimoto_by_length(predictions: List[str], targets: List[str],
                       num_buckets: int = 5) -> Dict:
    """Average Tanimoto bucketed by predicted SMILES length."""
    if not predictions:
        return {}
    lens = [len(p) for p in predictions]
    min_l, max_l = min(lens), max(lens)
    if min_l == max_l:
        bucket_key = str(min_l)
        t_sum = sum(compute_tanimoto(t, p) for t, p in zip(targets, predictions))
        n = len(predictions)
        return {bucket_key: {"count": n, "mean_tanimoto": t_sum / n if n > 0 else 0.0}}

    step = max(1, (max_l - min_l) // num_buckets)
    buckets: Dict = {}
    for t, p in zip(targets, predictions):
        bucket = (len(p) // step) * step
        key = f"{bucket}-{bucket + step - 1}"
        if key not in buckets:
            buckets[key] = {"count": 0, "tanimoto_sum": 0.0}
        buckets[key]["count"] += 1
        buckets[key]["tanimoto_sum"] += compute_tanimoto(t, p)
    for v in buckets.values():
        v["mean_tanimoto"] = v["tanimoto_sum"] / v["count"] if v["count"] > 0 else 0.0
        del v["tanimoto_sum"]
    return dict(sorted(buckets.items(),
                       key=lambda x: int(x[0].split("-")[0])))


# ── Aggregate metric computation from prediction rows ──────────────────────


def compute_summary_from_rows(
    rows: List[Dict],
    split: str = "test",
    model_name: str = "",
    tokenizer_type: str = "",
    seed: int = 0,
) -> Dict:
    """Compute full evaluation summary from prediction dict rows.

    Each row must contain keys from the predictions CSV specification.
    Returns a dict suitable for JSON serialization.
    """
    n = len(rows)
    if n == 0:
        return {"num_samples": 0}

    preds = [r["pred_smiles"] for r in rows]
    targets = [r["target_smiles"] for r in rows]
    valid_preds = [p for p in preds if is_valid(p)]
    invalid_preds = [p for p in preds if not is_valid(p)]

    # Exact & canonical exact
    exact_count = sum(1 for r in rows if r.get("exact_match", 0))
    canon_exact_count = sum(1 for r in rows if r.get("canonical_exact_match", 0))
    valid_count = sum(1 for p in preds if is_valid(p))

    # Tanimoto stats
    tanimotos = [r.get("tanimoto", 0.0) for r in rows]
    mean_tanimoto = sum(tanimotos) / n if n > 0 else 0.0
    valid_tanimotos = [r["tanimoto"] for r in rows if r.get("rdkit_valid", 0)]
    mean_tanimoto_valid = (sum(valid_tanimotos) / len(valid_tanimotos)
                           if valid_tanimotos else 0.0)

    # Scaffold match rate
    scaffold_count = sum(1 for r in rows if r.get("scaffold_match", 0))
    valid_scaffold_matches = sum(
        1 for r in rows if r.get("scaffold_match", 0) and r.get("rdkit_valid", 0)
    )

    # FG-F1
    fg_precisions = [r.get("fg_precision", 0.0) for r in rows]
    fg_recalls = [r.get("fg_recall", 0.0) for r in rows]
    fg_f1s = [r.get("fg_f1", 0.0) for r in rows]
    mean_fg_p = sum(fg_precisions) / n if n > 0 else 0.0
    mean_fg_r = sum(fg_recalls) / n if n > 0 else 0.0
    mean_fg_f1 = sum(fg_f1s) / n if n > 0 else 0.0

    # Lengths
    target_lens = [len(s) for s in targets]
    pred_lens = [len(p) for p in preds]

    # Unique stats
    uniq = unique_stats(preds)
    valid_uniq = unique_stats(valid_preds) if valid_preds else {"unique": 0, "unique_ratio": 0.0}

    # Mode collapse
    collapse = mode_collapse_score(preds)
    entropy = prediction_entropy(preds)
    top20 = top_frequencies(preds, n=20)

    # Levenshtein
    levenshteins = [r.get("levenshtein", 0) for r in rows]
    mean_levenshtein = sum(levenshteins) / n if n > 0 else 0.0

    # EOS stats (from pred_token_length if available)
    eos_rates: Dict = {}
    pct_hit_max = 0.0

    summary = {
        "split": split,
        "model_name": model_name,
        "tokenizer_type": tokenizer_type,
        "seed": seed,
        "num_samples": n,
        "exact_string_match": round(exact_count / n, 6) if n > 0 else 0.0,
        "canonical_exact_match": round(canon_exact_count / n, 6) if n > 0 else 0.0,
        "rdkit_validity": round(valid_count / n, 6) if n > 0 else 0.0,
        "unique_generated": uniq["unique"],
        "unique_valid_generated": valid_uniq["unique"],
        "unique_ratio": round(uniq["unique_ratio"], 6),
        "valid_unique_ratio": round(valid_uniq["unique_ratio"], 6),
        "mode_collapse_score": round(collapse, 6),
        "prediction_entropy": round(entropy, 4),
        "avg_pred_char_length": round(sum(pred_lens) / n, 2) if n > 0 else 0.0,
        "avg_target_char_length": round(sum(target_lens) / n, 2) if n > 0 else 0.0,
        "mean_tanimoto": round(mean_tanimoto, 6),
        "mean_tanimoto_valid_only": round(mean_tanimoto_valid, 6),
        "scaffold_match_rate": round(scaffold_count / n, 6) if n > 0 else 0.0,
        "mean_fg_precision": round(mean_fg_p, 6),
        "mean_fg_recall": round(mean_fg_r, 6),
        "mean_fg_f1": round(mean_fg_f1, 6),
        "mean_levenshtein": round(mean_levenshtein, 2),
        "target_length_stats": compute_length_stats(target_lens),
        "pred_length_stats": compute_length_stats(pred_lens),
        "target_length_dist": length_distribution(target_lens),
        "pred_length_dist": length_distribution(pred_lens),
        "top_20_predictions": [
            {"smiles": s, "count": c, "fraction": round(f, 4)}
            for s, c, f in top20
        ],
        "validity_by_length": validity_by_length(preds),
        "tanimoto_by_length": tanimoto_by_length(preds, targets),
    }
    return summary
