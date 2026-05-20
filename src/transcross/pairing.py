"""IR–NMR molecule pairing via canonical SMILES."""

import json
from collections import defaultdict
from typing import Optional

from .smiles import canonicalize_smiles
from .io import iter_jsonl, safe_get_spectrum, safe_get_smiles


def scan_ir_records(
    ir_path: str, limit: Optional[int] = None
) -> dict:
    """Stream IR JSONL and build a canonical-SMILES-indexed catalog.

    For each canonical SMILES, keeps the record with the largest
    number of x-points (highest spectral resolution).

    Returns:
        Dict mapping canonical_smiles -> {
            "smiles": original SMILES,
            "canonical_smiles": canonical SMILES,
            "line_idx": original line index (0-based),
            "x": list of wavenumbers,
            "y": list of intensities,
            "x_len": number of x points,
            "condition": phase condition string,
            "temperature": temperature field,
            "pressure": pressure field,
        }
    """
    catalog: dict[str, dict] = {}
    invalid_count = 0
    total = 0

    for line_idx, record in enumerate(iter_jsonl(ir_path)):
        total += 1
        if limit is not None and total > limit:
            break

        smiles = safe_get_smiles(record)
        if not smiles:
            invalid_count += 1
            continue

        canon = canonicalize_smiles(smiles)
        if canon is None:
            invalid_count += 1
            continue

        spectrum = safe_get_spectrum(record)
        if spectrum is None:
            invalid_count += 1
            continue

        x = spectrum["x"]
        y = spectrum["y"]
        if not x or not y or len(x) != len(y):
            invalid_count += 1
            continue

        x_len = len(x)

        # Keep the record with the most x-points for each canonical SMILES
        if canon not in catalog or x_len > catalog[canon]["x_len"]:
            catalog[canon] = {
                "smiles": smiles,
                "canonical_smiles": canon,
                "line_idx": line_idx,
                "x": x,
                "y": y,
                "x_len": x_len,
                "condition": record.get("condition", "NONE"),
                "temperature": record.get("temperature", "NONE"),
                "pressure": record.get("pressure", "NONE"),
            }

    return catalog


def scan_nmr_records(
    nmr_path: str,
    allowed_smiles: Optional[set] = None,
    limit: Optional[int] = None,
) -> dict:
    """Stream NMR JSONL and build a (canonical SMILES, nucleus)-indexed catalog.

    For each (canonical SMILES, nucleus), keeps the record with the most peaks.

    Args:
        nmr_path: Path to NMR JSONL file.
        allowed_smiles: If provided, only keep records whose canonical SMILES
            is in this set. Use this to avoid loading the full NMR file into
            the pairing dict. Note: the full file is still streamed, but
            non-matching records are discarded immediately.
        limit: Maximum number of records to process.

    Returns:
        Dict mapping (canonical_smiles, nucleus) -> {
            "smiles": original SMILES,
            "canonical_smiles": canonical SMILES,
            "line_idx": original line index (0-based),
            "nucleus": nucleus type,
            "peaks": list of chemical shifts,
            "num_peaks": number of peaks,
            "frequency": spectrometer frequency (MHz),
            "solvent": NMR solvent,
        }
    """
    catalog: dict[tuple[str, str], dict] = {}
    invalid_count = 0
    total = 0

    for line_idx, record in enumerate(iter_jsonl(nmr_path)):
        total += 1
        if limit is not None and total > limit:
            break

        smiles = safe_get_smiles(record)
        if not smiles:
            invalid_count += 1
            continue

        canon = canonicalize_smiles(smiles)
        if canon is None:
            invalid_count += 1
            continue

        # Early filtering: skip if not in allowed_smiles
        if allowed_smiles is not None and canon not in allowed_smiles:
            continue

        nucleus = record.get("nucleus")
        if not nucleus:
            invalid_count += 1
            continue

        # Focus on 1H and 13C for the first version
        if nucleus not in ("1H", "13C"):
            continue

        spectrum = safe_get_spectrum(record)
        if spectrum is None:
            invalid_count += 1
            continue

        peaks = spectrum["x"]
        if not peaks:
            invalid_count += 1
            continue

        num_peaks = len(peaks)
        key = (canon, nucleus)

        # Keep the record with the most peaks
        if key not in catalog or num_peaks > catalog[key]["num_peaks"]:
            catalog[key] = {
                "smiles": smiles,
                "canonical_smiles": canon,
                "line_idx": line_idx,
                "nucleus": nucleus,
                "peaks": peaks,
                "num_peaks": num_peaks,
                "frequency": record.get("frequency"),
                "solvent": record.get("solvent", "NONE"),
            }

    return catalog


def build_pairs(
    ir_catalog: dict,
    nmr_catalog: dict,
) -> list:
    """Inner join IR and NMR catalogs on canonical SMILES.

    Returns a list of paired sample dicts, one per molecule.
    A molecule is included if it has IR and at least one NMR channel (1H or 13C).
    """
    pairs = []
    sample_id = 0

    for canon_smiles, ir_rec in ir_catalog.items():
        nmr_1h = nmr_catalog.get((canon_smiles, "1H"))
        nmr_13c = nmr_catalog.get((canon_smiles, "13C"))

        if nmr_1h is None and nmr_13c is None:
            continue

        available_nuclei = []
        if nmr_1h is not None:
            available_nuclei.append("1H")
        if nmr_13c is not None:
            available_nuclei.append("13C")

        pair = {
            "sample_id": sample_id,
            "canonical_smiles": canon_smiles,
            "original_smiles": ir_rec["smiles"],
            "ir_line_idx": ir_rec["line_idx"],
            "ir_num_points": ir_rec["x_len"],
            "ir_condition": ir_rec["condition"],
            "ir_x": ir_rec["x"],
            "ir_y": ir_rec["y"],
            "nmr_1h_line_idx": nmr_1h["line_idx"] if nmr_1h else None,
            "nmr_1h_num_peaks": nmr_1h["num_peaks"] if nmr_1h else None,
            "nmr_1h_frequency": nmr_1h["frequency"] if nmr_1h else None,
            "nmr_1h_solvent": nmr_1h["solvent"] if nmr_1h else None,
            "nmr_1h_peaks": nmr_1h["peaks"] if nmr_1h else [],
            "nmr_13c_line_idx": nmr_13c["line_idx"] if nmr_13c else None,
            "nmr_13c_num_peaks": nmr_13c["num_peaks"] if nmr_13c else None,
            "nmr_13c_frequency": nmr_13c["frequency"] if nmr_13c else None,
            "nmr_13c_solvent": nmr_13c["solvent"] if nmr_13c else None,
            "nmr_13c_peaks": nmr_13c["peaks"] if nmr_13c else [],
            "available_nuclei": ",".join(available_nuclei),
        }
        pairs.append(pair)
        sample_id += 1

    return pairs
