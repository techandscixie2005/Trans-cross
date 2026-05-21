"""Collate functions for SMILES generation DataLoader."""

import torch
import numpy as np
from typing import List, Dict


def smiles_collate_fn(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Collate a batch of SMILES generation items.

    Pads input_ids and target_ids to max length in batch,
    stacks spectra into tensor.

    Args:
        batch: list of dicts from TransCrossSmilesDataset
        pad_id: token ID used for padding

    Returns:
        collated dict with stacked tensors
    """
    ir = torch.stack([torch.as_tensor(item["ir"], dtype=torch.float32) for item in batch])
    nmr_1h = torch.stack([torch.as_tensor(item["nmr_1h"], dtype=torch.float32) for item in batch])
    nmr_13c = torch.stack([torch.as_tensor(item["nmr_13c"], dtype=torch.float32) for item in batch])

    mask_ir = torch.stack([torch.as_tensor(item["mask_ir"], dtype=torch.float32) for item in batch])
    mask_1h = torch.stack([torch.as_tensor(item["mask_1h"], dtype=torch.float32) for item in batch])
    mask_13c = torch.stack([torch.as_tensor(item["mask_13c"], dtype=torch.float32) for item in batch])

    # Pad input_ids and target_ids
    max_len = max(item["input_ids"].shape[0] for item in batch)

    def pad_tensor(t, max_len, pad_id):
        if t.shape[0] < max_len:
            pad = torch.full((max_len - t.shape[0],), pad_id, dtype=t.dtype)
            return torch.cat([t, pad])
        return t

    input_ids = torch.stack([
        pad_tensor(torch.as_tensor(item["input_ids"], dtype=torch.long), max_len, pad_id)
        for item in batch
    ])
    target_ids = torch.stack([
        pad_tensor(torch.as_tensor(item["target_ids"], dtype=torch.long), max_len, pad_id)
        for item in batch
    ])

    return {
        "ir": ir,
        "nmr_1h": nmr_1h,
        "nmr_13c": nmr_13c,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "mask_ir": mask_ir,
        "mask_1h": mask_1h,
        "mask_13c": mask_13c,
        "smiles": [item["smiles"] for item in batch],
        "idx": [item["idx"] for item in batch],
    }
