"""Safe I/O for JSONL streaming and config loading."""

import json
from typing import Optional, Union

import yaml


def iter_jsonl(path: str):
    """Stream lines from a JSONL file as parsed dicts.

    Never loads the entire file into memory.
    """
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def safe_get_spectrum(record: dict) -> Optional[dict]:
    """Extract x, y arrays from an IR or NMR record.

    Returns dict with 'x' and 'y' keys, or None if spectrum is missing.
    """
    value = record.get("value")
    if not isinstance(value, dict):
        return None
    x = value.get("x")
    if x is None:
        return None
    y = value.get("y", [])
    return {"x": x, "y": y}


def safe_get_smiles(record: dict) -> Optional[str]:
    """Extract SMILES string from a record."""
    return record.get("smiles")


def read_yaml(path: str) -> dict:
    """Read a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_json(path: str, obj: Union[dict, list]):
    """Write an object as JSON file."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_jsonl(path: str, records: list):
    """Write a list of records as JSONL file."""
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
