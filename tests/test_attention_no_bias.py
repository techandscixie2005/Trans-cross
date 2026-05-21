"""Verify that custom attention modules have NO additive attention bias."""

import pytest
import torch

from src.transcross.models.attention import (
    MultiHeadAttention,
    FeedForward,
    TransformerBlockPreLN,
    CrossAttentionBlockPreLN,
)


class TestNoAdditiveAttentionBias:
    def test_mha_no_bias_attribute(self):
        """MultiHeadAttention must not have an attention_bias parameter."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        params = dict(mha.named_parameters())
        # No parameter should be named "attention_bias" or similar
        for name in params:
            assert "bias" not in name.lower() or "proj" in name.lower(), \
                f"Found potential attention bias param: {name}"

    def test_mha_forward_no_bias(self):
        """Forward pass should work without any attention bias."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        B, L = 2, 10
        x = torch.randn(B, L, 64)

        # Self-attention
        out = mha(x, x, x)
        assert out.shape == (B, L, 64)
        assert not torch.isnan(out).any()

        # Cross-attention
        kv = torch.randn(B, 15, 64)
        out = mha(x, kv, kv)
        assert out.shape == (B, L, 64)

    def test_mha_with_key_padding_mask(self):
        """Key padding mask should work."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        B, L_q, L_k = 2, 5, 8
        q = torch.randn(B, L_q, 64)
        k = v = torch.randn(B, L_k, 64)
        mask = torch.zeros(B, L_k, dtype=torch.bool)
        mask[:, -2:] = True  # mask last 2 positions

        out = mha(q, k, v, key_padding_mask=mask)
        assert out.shape == (B, L_q, 64)
        assert not torch.isnan(out).any()

    def test_mha_return_weights(self):
        """Should optionally return attention weights."""
        mha = MultiHeadAttention(d_model=64, num_heads=4)
        x = torch.randn(2, 5, 64)
        out, weights = mha(x, x, x, need_weights=True)
        assert out.shape == (2, 5, 64)
        assert weights.shape == (2, 4, 5, 5)

    def test_no_graphormer_bias(self):
        """Model should not contain any graph distance or pair bias parameters."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        for name, _ in mha.named_parameters():
            for banned in ["spatial", "graph", "pair", "distance", "coordinate",
                           "relative", "edge", "adj"]:
                assert banned not in name.lower(), \
                    f"Found {banned} related parameter: {name}"


class TestTransformerBlockPreLN:
    def test_forward(self):
        block = TransformerBlockPreLN(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_causal_forward(self):
        block = TransformerBlockPreLN(d_model=64, num_heads=4)
        x = torch.randn(2, 10, 64)
        causal = torch.triu(torch.full((10, 10), float("-inf")), diagonal=1)
        out = block(x, attn_mask=causal)
        assert out.shape == x.shape


class TestCrossAttentionBlockPreLN:
    def test_forward(self):
        block = CrossAttentionBlockPreLN(d_model=64, num_heads=4)
        q = torch.randn(2, 5, 64)
        kv = torch.randn(2, 15, 64)
        out = block(q, kv)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_zero_init_out_proj(self):
        block = CrossAttentionBlockPreLN(d_model=64, num_heads=4,
                                          zero_init_out_proj=True)
        # After zero init, cross_attn.out_proj.weight should be all zeros
        assert torch.allclose(
            block.cross_attn.out_proj.weight,
            torch.zeros_like(block.cross_attn.out_proj.weight)
        )


class TestFeedForward:
    def test_forward(self):
        ffn = FeedForward(d_model=64, d_ff=256)
        x = torch.randn(2, 10, 64)
        out = ffn(x)
        assert out.shape == x.shape

    def test_default_d_ff(self):
        ffn = FeedForward(d_model=64)
        # Default d_ff should be 4 * d_model = 256
        # Check first linear layer
        assert ffn.net[0].out_features == 256


class TestInitialization:
    def test_qkv_normal_init(self):
        """Q/K/V projections should be initialized with Normal(0, 0.02)."""
        mha = MultiHeadAttention(d_model=128, num_heads=4)
        for proj in [mha.q_proj, mha.k_proj, mha.v_proj]:
            # Weight std should be close to 0.02
            assert abs(proj.weight.std().item() - 0.02) < 0.01
            # Bias should be zero
            assert torch.allclose(proj.bias, torch.zeros_like(proj.bias))

    def test_out_proj_zero_init_when_requested(self):
        mha = MultiHeadAttention(d_model=128, num_heads=4, zero_init_out_proj=True)
        assert torch.allclose(mha.out_proj.weight, torch.zeros_like(mha.out_proj.weight))
        assert torch.allclose(mha.out_proj.bias, torch.zeros_like(mha.out_proj.bias))
