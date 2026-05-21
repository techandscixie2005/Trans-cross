"""Greedy decoding and evaluation utilities for SMILES generation."""

from typing import List, Optional

import torch


@torch.no_grad()
def greedy_decode(
    model,
    ir: torch.Tensor,
    nmr_1h: torch.Tensor,
    nmr_13c: torch.Tensor,
    tokenizer,
    max_len: int = 256,
    device: Optional[torch.device] = None,
) -> List[List[int]]:
    """Greedy decode SMILES from spectral inputs.

    Args:
        model: DirectConcatSmilesModel or IntraCrossSmilesModel
        ir: (B, 1801)
        nmr_1h: (B, 1501)
        nmr_13c: (B, 2201)
        tokenizer: SmilesTokenizer instance
        max_len: maximum generation length
        device: torch device

    Returns:
        list of token ID lists for each sample in batch
    """
    if device is not None:
        model = model.to(device)
        ir = ir.to(device)
        nmr_1h = nmr_1h.to(device)
        nmr_13c = nmr_13c.to(device)

    model.eval()
    B = ir.shape[0]

    # Encode spectra once
    encoder_memory, memory_mask = model._encode_spectra(ir, nmr_1h, nmr_13c)

    # Start with BOS token
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    pad_id = tokenizer.pad_id

    input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=ir.device)
    finished = torch.zeros(B, dtype=torch.bool, device=ir.device)
    results: List[List[int]] = [[] for _ in range(B)]

    for _ in range(max_len):
        logits = model.decoder(input_ids, encoder_memory,
                               memory_padding_mask=memory_mask)
        next_logits = logits[:, -1, :]  # (B, vocab_size)
        next_tokens = next_logits.argmax(dim=-1)  # (B,)

        for i in range(B):
            if not finished[i]:
                tok = next_tokens[i].item()
                if tok == eos_id:
                    finished[i] = True
                else:
                    results[i].append(tok)

        if finished.all():
            break

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=1)

    return results
