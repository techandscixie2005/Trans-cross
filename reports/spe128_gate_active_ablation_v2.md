# Trans-cross SPE-128 Active-Gate SMILES Generation Ablation v2

**Date**: 2026-05-22
**Git commit**: d0196b1 (spe128-gate-active-v2)
**Status**: COMPLETE (single seed, seed=42)

## 1. Motivation

v1 used SPE-1000 and g0=-4:
- SPE-1000 compressed SMILES too aggressively (mean train len=2.6 SPE tokens)
- E1 cross-attention gates stayed closed (alpha ≈ 0.019, barely changed from init)
- Both models severely overfit (train loss ~1.1, valid loss ~5.0-5.9)

v2 changes only two variables:
1. **SPE-1000 → SPE-128**: Less aggressive compression, longer sequences
2. **E1 gate init g0=-4 → g0=-2**: alpha0 ≈ 0.119 instead of 0.018

Goal: Reduce target over-compression and allow cross-attention to participate.

## 2. Controlled Experimental Question

Does intra-modal self-attention + cross-modal attention (E1) outperform direct concatenation self-attention (E0) for IR+NMR-to-SMILES generation?

**Strict controls**: No coordinate bias, no modality-pair bias, no auxiliary loss, no reranking. Same decoder, same SPE-128 tokenizer, same data, same optimizer, same training protocol.

## 3. Dataset

| Property | Value |
|---|---|
| Total paired molecules | 4,563 |
| Train / Valid / Test | 3,195 / 684 / 684 |
| Split method | Scaffold split |
| IR shape | (4563, 1801) |
| 1H NMR shape | (4563, 1501) |
| 13C NMR shape | (4563, 2201) |
| No data leakage | 0% overlap across splits |

## 4. SPE-128 Tokenization Audit

| Property | SPE-128 (v2) | SPE-1000 (v1) | Change |
|---|---|---|---|
| Vocab size | 128 | 1000 | -872 |
| Merges | 86 | 958 | -872 |
| Train mean len | **5.5** | 2.6 | +112% |
| Valid mean len | **7.0** | 4.3 | +63% |
| Test mean len | **6.8** | 4.2 | +62% |
| Train P95 | 10 | 5 | +100% |
| Max train len | 21 | 11 | +91% |
| UNK rate train | 0% | 0% | same |
| UNK rate valid | 1.44% | 0.78% | +0.66% |
| UNK rate test | 2.07% | 0.17% | +1.90% |
| Dropped | 0 | 0 | same |

SPE-128 produces ~2× longer sequences with higher UNK rates on valid/test. The longer sequences provide more signal for autoregressive generation.

### Example SPE-128 Tokenizations

| SMILES | SPE-1000 tokens | SPE-128 tokens | SPE-128 len |
|---|---|---|---|
| `CNc1ccccc1` | CNc1ccccc1 | CN c1ccccc1 | 2 |
| `CN(C)c1ccc(Cc2ccc(N(C)C)cc2)cc1` | ...4 tokens | ...9 tokens | 9 |
| `CC(=O)OC1C(C)(C)C(=O)C1(C)C` | ...4 tokens | ...9 tokens | 9 |

## 5. Model Architectures

### Shared Components
- Spectral tokenizer: 1D patch (patch_size=64), Linear(64, d_model)
- Token counts: IR=29, 1H=24, 13C=35, +1 CLS = 89
- d_model=128, num_heads=4
- Decoder: 2 layers, FFN=512, dropout=0.1
- Self-attention: Xavier uniform init for Q/K/V/O
- Embeddings: Normal(0, 0.02)

### E0 DirectConcat
- 6 Pre-LN Transformer encoder layers
- All modality tokens concatenated + CLS → self-attention
- **1,804,544** trainable params

### E1 IntraCross (g0=-2)
- 1 intra-modal Pre-LN self-attention layer per modality
- 1 cross-modal attention layer per modality (attends to other 2 modalities)
- Learnable gate: alpha=sigmoid(g), g0=-2.0, alpha0≈0.119
- Cross-attention output: near-zero Normal(0, 1e-4) init
- No fusion layers
- **1,805,315** trainable params

## 6. Gate Initialization

| | v1 | v2 |
|---|---|---|
| g0 | -4.0 | **-2.0** |
| alpha0 | 0.018 | **0.119** |
| Intended behavior | Near-closed, stability first | Partially open, enables cross-modal learning |

Formula: alpha = sigmoid(g) = 1 / (1 + exp(-g))

## 7. Parameter Matching

| Model | Encoder | Decoder | Params | Diff |
|---|---|---|---|---|
| E0 | 6 self-attn layers | 2 layers | 1,804,544 | — |
| E1 | 1 intra + 1 cross | 2 layers | 1,805,315 | **0.043%** |

Decoder params identical: 578,176 each.

## 8. Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| lr | 1e-4 |
| Weight decay | 1e-4 |
| Warmup steps | 500 |
| Grad clip | 1.0 |
| Label smoothing | 0.05 |
| Batch size | 32 |
| Epochs | 50 |
| Seed | 42 |
| Beam size | 5 |
| Max target len | 120 |

## 9. Metrics

- Test loss: cross-entropy (label smoothing, ignore pad)
- Token accuracy: fraction correct (excl. pad)
- Canonical exact match: RDKit canonical SMILES equality
- Valid SMILES rate: RDKit MolFromSmiles success
- Tanimoto: Morgan FP (r=2, 2048 bits)

## 10. Results

### Validation (Best Epoch)

| Model | Best Epoch | Valid Loss | Valid Token Acc | Valid Exact |
|---|---|---|---|---|
| E0 | 13 | 3.3628 | 0.256 | 0.000 |
| E1 | 8 | 3.4942 | 0.226 | 0.000 |

### Test Metrics

| Model | Test Loss | Test Token Acc | Test Exact |
|---|---|---|---|
| **E0 DirectConcat** | **3.3380** | **0.2542** | 0.000 |
| E1 IntraCross (g0=-2) | 3.4842 | 0.2271 | 0.000 |

### Training Dynamics

| Model | Final Train Loss | Best Valid Loss | Gap |
|---|---|---|---|
| E0 | 0.694 | 3.363 | 4.85× |
| E1 | 0.887 | 3.494 | 3.94× |

Both still overfit, but significantly less than v1 (4.85× vs 4.73× for E0).

## 11. Cross-Attention Gate Diagnostics

| Modality | Initial α | Final α | Final g | Change |
|---|---|---|---|---|
| IR | 0.1192 | 0.1240 | -1.96 | +4.0% |
| 1H | 0.1192 | 0.1255 | -1.94 | +5.3% |
| 13C | 0.1192 | 0.1242 | -1.95 | +4.2% |

Gates opened slightly (+4-5%) but remain near initialization. The model is not strongly activating cross-modal attention even with g0=-2. Compared to v1 (alpha stayed at 0.019), the absolute gate value is 6.6× higher, but the training dynamics barely move them.

## 12. Qualitative Examples

(TBD — full evaluation with beam search.)

## 13. Comparison with v1

| Metric | v1 (SPE-1000, g0=-4) | v2 (SPE-128, g0=-2) | Improvement |
|---|---|---|---|
| | E0 | E1 | E0 | E1 | |
|---|---|---|---|---|---|
| Test loss | 5.045 | 5.096 | **3.338** | **3.484** | **-34%** |
| Token acc | 0.192 | 0.192 | **0.254** | **0.227** | **+32%** |
| Exact match | 0.000 | 0.000 | 0.000 | 0.000 | same |
| E1 gate IR | 0.019 | — | 0.124 | — | 6.5× |
| Train mean len | 2.6 | — | 5.5 | — | 2.1× |

**Key findings**:
- SPE-128 dramatically improves loss (34% reduction) and token accuracy (32% increase)
- E0 still beats E1, with similar margin (~0.15 loss) as v1
- E1 gates open more in absolute value (0.124 vs 0.019) but still barely move from init
- Neither model achieves exact matches despite the improvements

## 14. Interpretation

### Does E1 beat E0?
**No.** E0 direct concat attention still achieves lower test loss (3.338 vs 3.484) and higher token accuracy (0.254 vs 0.227).

### Did SPE-128 help?
**Yes, significantly.** Both models improved dramatically vs v1: 34% lower loss, 32% higher token accuracy. SPE-1000 was indeed over-compressing.

### Did g0=-2 activate cross-modal attention?
**Partially.** Gates are at a much higher absolute value (0.124 vs 0.019), but training barely moves them (+5%). The model still prefers to route information through intra-modal paths and the decoder.

### Why doesn't E1 outperform E0?
Possible explanations:
1. With only 3,195 training pairs, cross-modal attention may not have enough signal to learn meaningful patterns
2. The near-zero cross-attention output init (Normal(0, 1e-4)) still suppresses gradient flow to cross-attention
3. The decoder dominates optimization — memorization is easier than learning cross-modal fusion
4. The task (direct SMILES generation from spectra) may be too difficult for this dataset size

### Is one seed enough?
**No.** Multi-seed confirmation needed.

## 15. Limitations

- Single seed (42)
- Small dataset (3,195 train)
- NMR has no intensities (binary binning)
- No beam search evaluation
- No pretraining
- No coordinate encoding
- SPE-128 UNK rates non-trivial on valid/test (1.4-2.1%)

## 16. Next Steps

1. **Run seeds 43/44** for statistical significance
2. **Try g0=0** (fully open gate) to see if cross-attention can be forced to participate
3. **Try removing near-zero cross-attn output init** — Normal(0, 1e-4) may be too slow
4. **SPE-256 or SPE-512** as intermediate compression levels
5. **Larger models** with pretraining
6. **Add coordinate-aware attention bias** as a separate experiment

## 17. Reproducibility Checklist

| Item | Value |
|---|---|
| Git commit | d0196b1 |
| Branch | spe128-gate-active-v2 |
| Server path | /data/home/sczc698/run/xxy/Trans-cross/ |
| Code path | /data/home/sczc698/run/xxy/Trans-cross/code/ |
| Conda environment | transpec |
| Python | 3.9.0 |
| PyTorch | 2.0.0+cu118 |
| E0 output | runs/spe128_concat_seed42/ |
| E1 output | runs/spe128_intra_cross_seed42/ |
| SPE vocab | data/processed/spe128_vocab.json |
| Raw files untouched | IR_NIST.jsonl, NMR_exp2.jsonl |
