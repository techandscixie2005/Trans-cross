"""Tests: E0 and E1 encoder memory interface conformity.

Verifies that E1 (both legacy and block modes) produces memory with the
same shape, token order, and decoder interface as E0 direct concat.
"""

import torch
import pytest

from src.transcross.tokenization.spe_tokenizer import SPETokenizer
from src.transcross.models.smiles_concat import DirectConcatSmilesModel
from src.transcross.models.smiles_intra_cross import IntraCrossSmilesModel


SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]


@pytest.fixture
def tokenizer():
    tok = SPETokenizer()
    tok.train(SAMPLE_SMILES, vocab_size=32, min_frequency=1)
    return tok


@pytest.fixture
def batch():
    B = 2
    return (
        torch.randn(B, 1801),
        torch.randn(B, 1501),
        torch.randn(B, 2201),
    )


def _make_e0(tokenizer, **kwargs):
    defaults = dict(
        vocab_size=tokenizer.vocab_size, d_model=128, encoder_layers=1,
        decoder_layers=1, num_heads=4, patch_size=64, pad_id=tokenizer.pad_id,
    )
    defaults.update(kwargs)
    return DirectConcatSmilesModel(**defaults)


def _make_e1_legacy(tokenizer, **kwargs):
    defaults = dict(
        vocab_size=tokenizer.vocab_size, d_model=128,
        encoder_layers=1, cross_layers=1, fusion_layers=0,
        decoder_layers=1, num_heads=4, cross_gate_init=-2.0,
        patch_size=64, pad_id=tokenizer.pad_id,
    )
    defaults.update(kwargs)
    return IntraCrossSmilesModel(**defaults)


def _make_e1_block(tokenizer, blocks=1, **kwargs):
    defaults = dict(
        vocab_size=tokenizer.vocab_size, d_model=128,
        intra_cross_blocks=blocks,
        decoder_layers=1, num_heads=4, cross_gate_init=-2.0,
        patch_size=64, pad_id=tokenizer.pad_id,
    )
    defaults.update(kwargs)
    return IntraCrossSmilesModel(**defaults)


class TestMemoryShape:
    """Verify E0 and E1 produce identical memory shapes."""

    def test_e0_memory_shape(self, tokenizer, batch):
        ir, h1, c13 = batch
        e0 = _make_e0(tokenizer)
        mem, mask = e0._encode_spectra(ir, h1, c13)
        assert mem.shape == (2, 89, 128), f"E0 shape {mem.shape} != (2, 89, 128)"
        assert mask.shape == (2, 89)

    def test_e1_legacy_memory_shape(self, tokenizer, batch):
        ir, h1, c13 = batch
        e1 = _make_e1_legacy(tokenizer)
        mem, mask = e1._encode_spectra(ir, h1, c13)
        assert mem.shape == (2, 89, 128), f"E1 legacy shape {mem.shape} != (2, 89, 128)"
        assert mask.shape == (2, 89)

    def test_e1_1block_memory_shape(self, tokenizer, batch):
        ir, h1, c13 = batch
        e1 = _make_e1_block(tokenizer, blocks=1)
        mem, mask = e1._encode_spectra(ir, h1, c13)
        assert mem.shape == (2, 89, 128), f"E1 1block shape {mem.shape} != (2, 89, 128)"

    def test_e1_3block_memory_shape(self, tokenizer, batch):
        ir, h1, c13 = batch
        e1 = _make_e1_block(tokenizer, blocks=3)
        mem, mask = e1._encode_spectra(ir, h1, c13)
        assert mem.shape == (2, 89, 128), f"E1 3block shape {mem.shape} != (2, 89, 128)"


class TestMemoryTokenCount:
    """Verify both models produce exactly 88 spectral tokens (+1 CLS)."""

    def test_e0_token_count(self, tokenizer, batch):
        ir, h1, c13 = batch
        e0 = _make_e0(tokenizer)
        mem, _ = e0._encode_spectra(ir, h1, c13)
        # 29 IR + 24 1H + 35 13C + 1 CLS = 89
        assert mem.shape[1] == 89

    def test_e1_block_token_count(self, tokenizer, batch):
        ir, h1, c13 = batch
        e1 = _make_e1_block(tokenizer, blocks=3)
        mem, _ = e1._encode_spectra(ir, h1, c13)
        assert mem.shape[1] == 89

    def test_no_pooling_before_decoder(self, tokenizer, batch):
        """Memory token count is 89, not 1 (pooled) or 3 (per-modality pooled)."""
        ir, h1, c13 = batch
        e0 = _make_e0(tokenizer)
        e1 = _make_e1_block(tokenizer, blocks=1)
        mem0, _ = e0._encode_spectra(ir, h1, c13)
        mem1, _ = e1._encode_spectra(ir, h1, c13)
        # Not pooled
        assert mem0.shape[1] > 3, f"E0 memory seems pooled: {mem0.shape[1]} tokens"
        assert mem1.shape[1] > 3, f"E1 memory seems pooled: {mem1.shape[1]} tokens"
        # Not single modality (must be all 88 spectral tokens)
        assert mem0.shape[1] == 89, f"E0: expected 89 tokens, got {mem0.shape[1]}"
        assert mem1.shape[1] == 89, f"E1: expected 89 tokens, got {mem1.shape[1]}"


class TestDecoderInterface:
    """Verify E0 and E1 use the same decoder interface."""

    def test_same_decoder_class(self, tokenizer):
        e0 = _make_e0(tokenizer)
        e1_legacy = _make_e1_legacy(tokenizer)
        e1_block = _make_e1_block(tokenizer, blocks=3)
        cls = type(e0.decoder).__name__
        assert type(e1_legacy.decoder).__name__ == cls
        assert type(e1_block.decoder).__name__ == cls

    def test_decoder_config_identical(self, tokenizer):
        e0 = _make_e0(tokenizer, decoder_layers=2, decoder_ffn_dim=512)
        e1 = _make_e1_block(tokenizer, blocks=3, decoder_layers=2, decoder_ffn_dim=512)
        for attr in ['d_model', 'vocab_size', 'pad_id']:
            assert getattr(e0.decoder, attr) == getattr(e1.decoder, attr), \
                f"decoder.{attr} differs: {getattr(e0.decoder, attr)} vs {getattr(e1.decoder, attr)}"
        assert len(e0.decoder.layers) == len(e1.decoder.layers), \
            "decoder layer count differs"

    def test_logits_shape(self, tokenizer, batch):
        ir, h1, c13 = batch
        T = 8
        input_ids = torch.randint(1, tokenizer.vocab_size, (2, T))
        e0 = _make_e0(tokenizer)
        e1 = _make_e1_block(tokenizer, blocks=3)
        logits0 = e0(ir, h1, c13, input_ids)
        logits1 = e1(ir, h1, c13, input_ids)
        assert logits0.shape == (2, T, tokenizer.vocab_size)
        assert logits1.shape == logits0.shape

    def test_single_memory_tensor(self, tokenizer, batch):
        """Decoder receives a single memory tensor, not separate modality tensors."""
        ir, h1, c13 = batch
        T = 5
        input_ids = torch.randint(1, tokenizer.vocab_size, (2, T))

        # Check E1 forward: decoder receives single memory
        e1 = _make_e1_block(tokenizer, blocks=3)
        mem, mask = e1._encode_spectra(ir, h1, c13)
        logits = e1.decoder(input_ids, mem, memory_padding_mask=mask)
        assert logits.shape == (2, T, tokenizer.vocab_size)


class TestTokenOrder:
    """Verify both models use IR→1H→13C token order."""

    def test_token_segment_counts(self, tokenizer, batch):
        """Verify individual modality token counts before concatenation."""
        ir, h1, c13 = batch
        e0 = _make_e0(tokenizer)
        ir_t = e0.ir_tokenizer(ir)
        h1_t = e0.h1_tokenizer(h1)
        c13_t = e0.c13_tokenizer(c13)
        assert ir_t.shape[1] == 29, f"IR tokens: {ir_t.shape[1]} != 29"
        assert h1_t.shape[1] == 24, f"1H tokens: {h1_t.shape[1]} != 24"
        assert c13_t.shape[1] == 35, f"13C tokens: {c13_t.shape[1]} != 35"

    def test_e1_same_token_counts(self, tokenizer, batch):
        ir, h1, c13 = batch
        e1 = _make_e1_block(tokenizer, blocks=3)
        ir_t = e1.ir_tokenizer(ir)
        h1_t = e1.h1_tokenizer(h1)
        c13_t = e1.c13_tokenizer(c13)
        assert ir_t.shape[1] == 29
        assert h1_t.shape[1] == 24
        assert c13_t.shape[1] == 35


class TestNoModalityBias:
    """Verify no modality-pair bias or coordinate bias exists."""

    def test_e0_no_bias(self, tokenizer):
        e0 = _make_e0(tokenizer)
        for name, _ in e0.named_parameters():
            assert "modality" not in name.lower(), f"E0 has modality param: {name}"
            assert "coordinate" not in name.lower(), f"E0 has coordinate param: {name}"

    def test_e1_no_bias(self, tokenizer):
        e1 = _make_e1_block(tokenizer, blocks=3)
        for name, _ in e1.named_parameters():
            assert "modality_bias" not in name.lower(), f"E1 has modality bias: {name}"
            assert "pair_bias" not in name.lower(), f"E1 has pair bias: {name}"
