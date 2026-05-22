"""Tests for SMILES generation models (concat and intra_cross)."""

import pytest
import torch

from src.transcross.smiles_tokenizer import SmilesTokenizer
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel
from src.transcross.models.smiles_decoder import TransformerSmilesDecoder


SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture
def tokenizer():
    return SmilesTokenizer.build_from_smiles(SAMPLE_SMILES)


@pytest.fixture
def sample_spectra():
    B = 2
    ir = torch.randn(B, 1801)
    h1 = torch.randn(B, 1501)
    c13 = torch.randn(B, 2201)
    return ir, h1, c13


def _get_input_ids(tokenizer, smi_list, B=2):
    """Helper: create batched input_ids for teacher forcing."""
    ids_list = []
    for smi in smi_list[:B]:
        ids = tokenizer.encode(smi, add_bos=True, add_eos=True)
        ids_list.append(torch.tensor(ids))
    max_len = max(len(ids) for ids in ids_list)
    padded = torch.zeros(B, max_len, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
    return padded


class TestDirectConcatSmilesModel:
    def test_forward_shape(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        input_ids = _get_input_ids(tokenizer, SAMPLE_SMILES, B=2)

        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )

        logits = model(ir, h1, c13, input_ids)
        B, T = input_ids.shape
        assert logits.shape == (B, T, tokenizer.vocab_size)

    def test_forward_no_nan(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        input_ids = _get_input_ids(tokenizer, SAMPLE_SMILES)

        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        logits = model(ir, h1, c13, input_ids)
        assert not torch.isnan(logits).any()

    def test_count_params(self, tokenizer):
        model = DirectConcatSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64,
        )
        n = model.count_params()
        assert n > 0


class TestIntraCrossSmilesModel:
    def test_forward_shape(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        input_ids = _get_input_ids(tokenizer, SAMPLE_SMILES, B=2)

        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
            cross_layers=1, fusion_layers=1,
        )

        logits = model(ir, h1, c13, input_ids)
        B, T = input_ids.shape
        assert logits.shape == (B, T, tokenizer.vocab_size)

    def test_forward_no_nan(self, tokenizer, sample_spectra):
        ir, h1, c13 = sample_spectra
        input_ids = _get_input_ids(tokenizer, SAMPLE_SMILES)

        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
        )
        logits = model(ir, h1, c13, input_ids)
        assert not torch.isnan(logits).any()

    def test_cross_attn_near_zero_init(self, tokenizer):
        """Cross-attention out_proj should be near-zero (Normal(0, 1e-4))."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
        )
        for cross_blocks in [model.ir_cross, model.h1_cross, model.c13_cross]:
            for block in cross_blocks:
                w = block.cross_attn.out_proj.weight
                # Mean should be close to 0, std should be close to 1e-4
                assert abs(w.mean().item()) < 0.001, f"mean {w.mean().item()} too large"
                assert 1e-5 < w.std().item() < 1e-2, f"std {w.std().item()} unexpected"

    def test_cross_attn_gate_init(self, tokenizer):
        """Cross-attention gate should be initialized to sigmoid(-4) ≈ 0.018."""
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, num_heads=4, patch_size=64,
            encoder_layers=1, cross_layers=1,
        )
        alpha = model.ir_cross[0].get_alpha()
        assert 0.015 < alpha.item() < 0.02, f"alpha={alpha.item()} not near 0.018"

    def test_count_params(self, tokenizer):
        model = IntraCrossSmilesModel(
            vocab_size=tokenizer.vocab_size,
            d_model=64, encoder_layers=1, decoder_layers=1,
            num_heads=4, patch_size=64,
        )
        n = model.count_params()
        assert n > 0


class TestSharedDecoder:
    def test_causal_mask_shape(self):
        decoder = TransformerSmilesDecoder(
            vocab_size=50, d_model=64, num_layers=1, num_heads=4,
        )
        mask = decoder._build_causal_mask(10, torch.device("cpu"))
        assert mask.shape == (10, 10)
        # Lower triangular should be 0 (not masked)
        assert mask[1, 0] == 0.0
        # Upper triangular should be -inf (masked)
        assert mask[0, 1] == float("-inf")

    def test_decoder_forward_shape(self):
        vocab_size = 50
        decoder = TransformerSmilesDecoder(
            vocab_size=vocab_size, d_model=64, num_layers=1, num_heads=4,
        )
        B, T_enc, T_dec = 2, 20, 8
        encoder_memory = torch.randn(B, T_enc, 64)
        input_ids = torch.randint(1, vocab_size, (B, T_dec))

        logits = decoder(input_ids, encoder_memory)
        assert logits.shape == (B, T_dec, vocab_size)

    def test_loss_computable(self):
        """Cross-entropy loss should be computable from decoder logits."""
        vocab_size = 50
        pad_id = 0
        decoder = TransformerSmilesDecoder(
            vocab_size=vocab_size, d_model=64, num_layers=1, num_heads=4,
            pad_id=pad_id,
        )
        B, T_enc, T_dec = 2, 20, 8
        encoder_memory = torch.randn(B, T_enc, 64)
        input_ids = torch.randint(1, vocab_size, (B, T_dec))
        target_ids = torch.randint(1, vocab_size, (B, T_dec))
        # Pad last few positions
        target_ids[0, -2:] = pad_id

        logits = decoder(input_ids, encoder_memory)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(B * T_dec, vocab_size),
            target_ids.reshape(B * T_dec),
            ignore_index=pad_id,
        )
        assert not torch.isnan(loss)
        assert loss > 0
