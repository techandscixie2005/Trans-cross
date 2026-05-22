# Trans-cross SPE SMILES Generation Ablation v1

**Date**: 2026-05-22
**Git commit**: 36d1911 (master)
**Status**: COMPLETE (single seed, seed=42)

## 1. Experimental Goal

**Does "intra-modal self-attention followed by cross-modal attention" outperform "direct concatenation self-attention" for IR+NMR-to-SMILES generation?**

Only two encoder fusion strategies are compared:

| Model | Strategy | Description |
|---|---|---|
| **E0** | Direct Concat Attention | IR + 1H + 13C tokens concatenated → standard Transformer encoder → decoder |
| **E1** | Intra-modal + Cross-modal Attention | IR, 1H, 13C processed by separate intra-modal encoders → cross-modal attention exchanges info → fused memory → decoder |

**Strict constraints enforced**:
- SPE-1000 tokenization for both models
- No atom-level baseline (atom-level is NOT a model condition)
- No coordinate bias
- No modality-pair bias
- No fingerprint auxiliary loss
- No reranking (no SpecGNN, no candidate reranking)
- Same decoder architecture for E0 and E1
- Same SPE tokenizer for E0 and E1
- Same data (4563 paired molecules, scaffold split)
- Same training protocol (AdamW, lr=1e-4, 50 epochs)
- Same evaluation metrics

## 2. Relation to Original TranSpec

This experiment is built as a controlled modification of the TranSpec codebase:

- **Reused**: Pre-LN Transformer decoder, 1D patch tokenization, scaffold splitting
- **Extended**: SPE tokenization module, IR+NMR dataset adapter, dual encoder architectures
- **New**: E0 (DirectConcatSmilesModel), E1 (IntraCrossSmilesModel), cross-attention with learnable gate
- **Modified**: Attention initialization (Xavier uniform), cross-attention output init (near-zero Normal(0,1e-4))

## 3. Dataset

| Property | Value |
|---|---|
| Total paired molecules | 4,563 |
| Train / Valid / Test | 3,195 / 684 / 684 |
| Split method | Scaffold split (Bemis-Murcko) |
| IR shape | (4563, 1801) |
| 1H NMR shape | (4563, 1501) |
| 13C NMR shape | (4563, 2201) |
| No data leakage | 0 overlapping indices across splits |

## 4. SPE Tokenization

SPE (SMILES Pair Encoding) starts from atom-level SMILES tokens, counts frequent adjacent token pairs, iteratively merges high-frequency pairs, and learns a vocabulary of chemically meaningful SMILES substrings.

| Property | Value |
|---|---|
| Vocab size | 1,000 |
| Number of merges | 958 |
| Min pair frequency | 2 |
| Max target length | 120 SPE tokens |
| Special tokens | pad=0, bos=1, eos=2, unk=3 |

### Token Length Statistics

| Split | N | Mean | Median | P90 | P95 | Max | Dropped |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 3,195 | 2.6 | 2 | 4 | 5 | 11 | 0 |
| valid | 684 | 4.3 | 4 | 7 | 9 | 26 | 0 |
| test | 684 | 4.2 | 3 | 7 | 9 | 18 | 0 |

### Atom-Level Reference (NOT a model condition)

| Split | N | Mean len |
|---|---:|---:|
| train | 3,195 | 18.8 |
| valid | 684 | 18.4 |
| test | 684 | 17.7 |

SPE reduces mean token length by 78% (from 18.8 to 2.6 for training set).

### Unknown Token Rate

| Split | UNK tokens | Total tokens | UNK rate |
|---|---:|---:|---:|
| train | 0 | 8,433 | 0.000% |
| valid | 23 | 2,955 | 0.778% |
| test | 5 | 2,901 | 0.172% |

### Example SPE Tokenizations

| SMILES | Atom Tokens | SPE Tokens | SPE Len |
|---|---|---|---|
| `CNc1ccccc1` | C N c 1 c c c c c 1 | CNc1ccccc1 | 1 |
| `Cc1ccc(N)cc1` | C c 1 c c c ( N ) c c 1 | Cc1ccc(N)cc1 | 1 |
| `CNC1CCCCC1` | C N C 1 C C C C C 1 | CN C1CCCCC1 | 2 |
| `CCCCCCOc1ccc(C(N)=O)cc1` | C C C C C C O c 1 c... | CCCCCCOc1ccc( C(N)=O)cc1 | 2 |

## 5. Task Definition

Autoregressive SMILES generation conditioned on IR + 1H NMR + 13C NMR:

$$L = -\sum_t \log p(y_t \mid y_{<t}, X_{IR}, X_{1H}, X_{13C})$$

where $y_t$ are SPE tokens.

## 6. Shared Architecture

| Component | Specification |
|---|---|
| Spectral tokenizer | 1D patch (patch_size=64), Linear(64, d_model) → tokens |
| IR tokens | ceil(1801/64) = 29 |
| 1H tokens | ceil(1501/64) = 24 |
| 13C tokens | ceil(2201/64) = 35 |
| Total encoder tokens | 88 + 1 CLS = 89 |
| d_model | 128 |
| num_heads | 4 |
| Decoder layers | 2 |
| Decoder FFN dim | 512 |
| Dropout | 0.1 |
| Positional embedding | Learnable, per-modality |
| Modality embedding | Learnable |

## 7. E0 Direct Concat Encoder

- IR, 1H, 13C tokens concatenated (88 tokens)
- Prepend learnable CLS token
- 6 Pre-LN Transformer encoder layers
- Self-attention: Xavier uniform init for Q/K/V/O
- Output memory: (B, 89, 128) → decoder cross-attention

**Architecture**: 2,028,648 trainable parameters

## 8. E1 Intra-Cross Encoder

- Each modality gets 1 intra-modal Pre-LN self-attention layer (separate weights)
- Then 1 cross-modal attention layer per modality:
  - IR attends to concat(1H, 13C)
  - 1H attends to concat(IR, 13C)
  - 13C attends to concat(IR, 1H)
- Learnable residual gate: $\alpha_m = \sigma(g_m)$, $g_m = -4.0$ initially ($\alpha \approx 0.018$)
- Cross-attention output: near-zero Normal(0, 1e-4) init
- Cross-attention Q/K/V: Xavier uniform init
- Fused memory: concat(updated_IR, updated_1H, updated_13C) + CLS
- No fusion layers after concatenation

**Formulas**:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$$H_m' = H_m + \alpha_m \cdot \text{CrossAttn}(\text{LN}(H_m), \text{LN}(H_{\neg m}))$$

$$\alpha_m = \sigma(g_m), \quad g_m^{(0)} = -4.0$$

**Architecture**: 2,029,419 trainable parameters

## 9. Parameter Matching

| Model | Encoder Setting | Decoder Setting | Trainable Params | Ratio | Diff % |
|---|---|---|---|---|---|
| E0 DirectConcat | 6 self-attn layers | 2 layers, d=128 | 2,028,648 | 1.0004 | — |
| E1 IntraCross | 1 intra + 1 cross | 2 layers, d=128 | 2,029,419 | 1.0000 | 0.038% |

**Difference: 0.038% — well within 10-15% tolerance.**

Decoder parameters identical: 802,280 for both models.

## 10. Initialization

| Component | Weight Init | Bias Init |
|---|---|---|
| Self-attention Q/K/V/O | Xavier uniform | Zero |
| Cross-attention Q/K/V | Xavier uniform | Zero |
| Cross-attention output | Normal(0, 1e-4) | Zero |
| Cross-attention gate g | -4.0 (constant) | — |
| Embeddings | Normal(0, 0.02) | — |
| LayerNorm | weight=1, bias=0 | — |
| Decoder output projection | Xavier uniform | Zero |
| Spectral tokenizer linear | Xavier uniform | Zero |
| Modality/positional embeddings | Truncated normal(0, 0.02) | — |

**No coordinate bias. No modality-pair bias.** Verified by code audit.

## 11. Training Protocol

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Warmup steps | 500 (linear warmup + linear decay) |
| Gradient clipping | 1.0 |
| Label smoothing | 0.05 |
| Batch size | 32 |
| Epochs | 50 |
| Seed | 42 |
| Beam size (decoding) | 5 |
| Max target length | 120 SPE tokens |

## 12. Metrics

| Metric | Definition |
|---|---|
| Test loss | Cross-entropy with label smoothing, ignoring pad |
| Token accuracy | Fraction of correctly predicted tokens (excluding pad) |
| Canonical exact match | RDKit canonical SMILES match |
| Valid SMILES rate | RDKit MolFromSmiles succeeds |
| Beam top-1 exact match | Greedy decode (beam=1) exact match |
| Beam top-5 exact match | At least one of top-5 beams is exact |
| Tanimoto similarity | Morgan FP (radius=2, 2048 bits) Tanimoto |
| Average generated length | Mean SPE token count in decoded SMILES |

## 13. Results

### Validation Metrics (Best Epoch)

| Model | Best Epoch | Valid Loss | Valid Token Acc | Valid Exact |
|---|---|---|---|---|
| E0 DirectConcat | 17 | 5.0742 | 0.198 | 0.000 |
| E1 IntraCross | 11 | 5.1155 | 0.192 | 0.000 |

### Test Metrics

| Model | Test Loss | Test Token Acc | Test Exact |
|---|---|---|---|
| **E0 DirectConcat** | **5.0449** | 0.1922 | 0.000 |
| E1 IntraCross | 5.0956 | 0.1921 | 0.000 |

### Training Dynamics

| Model | Final Train Loss | Final Valid Loss | Overfitting Gap |
|---|---|---|---|
| E0 | 1.103 | 5.836 | 4.73× |
| E1 | 1.089 | 5.911 | 5.43× |

Both models overfit severely: train loss drops to ~1.1 but validation loss plateaus around 5.0-5.9.

**E0 DirectConcat achieves lower test loss (5.045 vs 5.096) with less overfitting (4.73× vs 5.43× gap).**

## 14. Qualitative Examples

(To be generated by running full evaluation with beam search.)

## 15. Cross-Attention Diagnostics (E1)

| Modality | Initial α | Final α | Initial g | Final g | Gate Opened? |
|---|---|---|---|---|---|
| IR | 0.0180 | 0.0190 | -4.0 | -3.94 | **No** (barely) |
| 1H | 0.0180 | 0.0188 | -4.0 | -3.96 | **No** |
| 13C | 0.0180 | 0.0188 | -4.0 | -3.95 | **No** |

**E1's cross-attention gates remain essentially closed after 50 epochs.** The model barely uses cross-modal information. This explains why E1 does not outperform E0 — the cross-modal mechanism is not being activated.

## 16. Analysis

### Does E1 outperform E0?

**No.** E0 achieves slightly lower test loss (5.045 vs 5.096) with essentially identical token accuracy (0.1922 vs 0.1921). Neither model achieves any exact SMILES matches.

### Which metric improves?

E0 leads on test loss (Δ = -0.051). Token accuracy is tied. No model can recover molecular structures.

### Does E1 overfit more or less?

E1 overfits more (5.43× gap vs 4.73× for E0). Both models overfit substantially on this small dataset.

### Why are cross-attention gates still closed?

The near-zero init (alpha ≈ 0.018) combined with large overfitting means the model found an easier optimization path: memorize via the decoder rather than learn meaningful cross-modal fusion. The small dataset (3,195 training pairs) and aggressive SPE compression (mean 2.6 SPE tokens) may not provide enough signal for the cross-modal mechanism to activate.

### Is one seed enough?

**No.** Single-seed results have low confidence. Multi-seed (42/43/44) is needed to confirm the direction of the effect.

### Verdict

**Weak E0 edge** (single seed, low confidence). E0's simpler architecture achieves marginally lower loss with less overfitting. The theoretical advantage of cross-modal attention does not materialize under these experimental conditions — the gates remain closed and the model overfits more.

## 17. Limitations

- **Single seed only**: Seeds 43 and 44 not yet run
- **Small training set**: 3,195 paired molecules limits generalization
- **SPE compression may be too aggressive**: Mean train length of 2.6 tokens may lose too much structural information
- **Cross-attention gates stay closed**: Near-zero init prevents E1 from using cross-modal information
- **NMR has no intensities**: Binary NMR representation limits signal quality
- **No beam search evaluation**: Only greedy decoding metrics reported
- **No pretraining**: Models trained from scratch
- **No coordinate encoding**: Positional information not provided to encoder

## 18. Next Steps

1. **Run seeds 42/43/44** to confirm statistical significance
2. **Try higher gate initialization** (e.g., g=-2, α≈0.12) to encourage cross-modal use in E1
3. **Try SPE-512 or SPE-256** for less aggressive tokenization (longer sequences, more signal)
4. **Larger models** (d_model=256, more layers) with pretraining
5. **Improve NMR representation** (Gaussian peak smearing instead of binary binning)
6. **Add coordinate-aware attention bias** as a separate experiment (NOT in this comparison)
7. **Beam search top-5 evaluation** to check if exact matches are recoverable

## 19. Reproducibility Checklist

| Item | Value |
|---|---|
| Git commit | 36d1911 (master) |
| GitHub | github.com/techandscixie2005/Trans-cross |
| Server path | /data/home/sczc698/run/xxy/Trans-cross/ |
| Code path | /data/home/sczc698/run/xxy/Trans-cross/code/ |
| Conda environment | transpec |
| Python version | 3.9.0 |
| PyTorch version | 2.0.0+cu118 |
| Data path | /data/home/sczc698/run/xxy/Trans-cross/data/processed/ |
| Raw files untouched | IR_NIST.jsonl, NMR_exp2.jsonl |
| E0 output | runs/spe1000_concat_seed42/ |
| E1 output | runs/spe1000_intra_cross_seed42/ |
| SPE vocab | data/processed/spe_vocab.json |

### Commands

```bash
# Train SPE vocabulary
python scripts/train_spe_vocab.py \
  --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
  --out-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
  --vocab-size 1000 --min-frequency 2 --max-len 120

# Parameter matching
python scripts/compare_model_params.py \
  --processed-dir ... --vocab .../spe_vocab.json \
  --config configs/smiles_spe1000_ablation.yaml

# E0 training
python scripts/train_smiles_ablation.py \
  --config configs/smiles_spe1000_ablation.yaml \
  --model concat_equal --epochs 50 --batch-size 32 --seed 42 \
  --out-dir runs/spe1000_concat_seed42

# E1 training
python scripts/train_smiles_ablation.py \
  --config configs/smiles_spe1000_ablation.yaml \
  --model intra_cross_equal --epochs 50 --batch-size 32 --seed 42 \
  --out-dir runs/spe1000_intra_cross_seed42
```

## 20. Experiment Verification Audit Summary

**A. Passed Checks**: Data split integrity (0% overlap), tokenizer leakage (SPE trained on train only), model variant correctness (decoder identical, config consistent), evaluation consistency (same metrics, same test set), run completeness (checkpoints saved, metrics.json present), training logs (no NaN, best epoch recorded).

**B. Potential Issues**: Single seed only. E1 cross-attention gates essentially unused (α ≈ 0.019 after training). Overfitting gap large for both models.

**C. Serious Issues**: None.

**D. Are conclusions trustworthy?** Partially — the E0 > E1 direction is consistent with simpler models performing better on small data, but single-seed confidence is low.

**E. Should any result be rerun?** Multi-seed (43, 44) needed.
