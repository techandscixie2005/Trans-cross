"""Model factory for SMILES generation ablation models.

Supports config-driven instantiation of equal-parameter model variants:
- concat_equal: DirectConcat encoder with configurable layers/FFN
- intra_cross_equal: IntraCross encoder with configurable intra/cross/fusion layers
"""

from typing import Dict, Any

import torch.nn as nn

from .smiles_concat import DirectConcatSmilesModel
from .smiles_intra_cross import IntraCrossSmilesModel


def build_smiles_model(
    model_name: str,
    config: Dict[str, Any],
    vocab_size: int,
    pad_id: int,
) -> nn.Module:
    """Build a SMILES generation model from config.

    Args:
        model_name: One of "concat_equal", "intra_cross_equal",
            "concat", "intra_cross" (legacy).
        config: Full config dict (parsed from YAML or argparse).
        vocab_size: SMILES tokenizer vocabulary size.
        pad_id: Padding token ID.

    Returns:
        Instantiated model.

    Raises:
        ValueError: if model_name is unknown or config is invalid.
    """
    shared = config.get("shared", {})
    d_model = shared.get("d_model", 128)
    num_heads = shared.get("num_heads", 4)
    dropout = shared.get("dropout", 0.1)
    decoder_layers = shared.get("decoder_layers", 2)
    decoder_ffn_dim = shared.get("decoder_ffn_dim", 512)
    max_smiles_len = config.get("data", {}).get("max_smiles_len", 160)
    patch_size = config.get("tokenizer", {}).get("patch_size", 64)

    if model_name in ("concat_equal", "concat"):
        e0 = config.get("e0_concat", {})
        encoder_layers = e0.get("encoder_layers", 6)
        encoder_ffn_dim = e0.get("encoder_ffn_dim", 512)

        return DirectConcatSmilesModel(
            vocab_size=vocab_size,
            ir_len=1801, h1_len=1501, c13_len=2201,
            patch_size=patch_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            num_heads=num_heads,
            dropout=dropout,
            pad_id=pad_id,
            max_smiles_len=max_smiles_len,
        )

    elif model_name in ("intra_cross_equal", "intra_cross"):
        e1 = config.get("e1_intra_cross", {})
        intra_layers = e1.get("intra_layers", 1)
        cross_layers = e1.get("cross_layers", 1)
        fusion_layers = e1.get("fusion_layers", 0)
        encoder_ffn_dim = e1.get("encoder_ffn_dim", 512)
        return IntraCrossSmilesModel(
            vocab_size=vocab_size,
            ir_len=1801, h1_len=1501, c13_len=2201,
            patch_size=patch_size,
            d_model=d_model,
            encoder_layers=intra_layers,
            cross_layers=cross_layers,
            fusion_layers=fusion_layers,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_layers=decoder_layers,
            decoder_ffn_dim=decoder_ffn_dim,
            num_heads=num_heads,
            dropout=dropout,
            pad_id=pad_id,
            max_smiles_len=max_smiles_len,
        )

    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported: concat_equal, intra_cross_equal, concat, intra_cross"
        )
