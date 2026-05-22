# Trans-cross SPE-128 E1-3Layer Diagnostic Ablation v3

**Date**: 2026-05-22
**Git commit**: a48d3ce (spe128-e1-3layer-v3)
**Status**: COMPLETE (diagnostic, single seed)

## 1. Motivation

v2 showed SPE-128 improved training substantially (34% lower loss), but E1 with 1 intra-cross block still lost to E0 (3.484 vs 3.338). This raised the question: **was E1 underpowered because it had too few cross-modal interaction layers?**

v3 tests whether stacking 3 intra-cross blocks (self-attn + cross-attn + FFN repeated 3 times) can rescue E1 performance. 

**Important**: v3 is a diagnostic capacity experiment, NOT a parameter-matched comparison. E1-3Layer has 2.32× more parameters than E0.

## 2. Experimental Question

Does deeper cross-modal interaction (3 intra-cross blocks) allow E1 to beat E0 direct concat attention?

**Strict controls**: SPE-128, g0=-2, no coordinate bias, no modality bias, no auxiliary loss, no reranking. Same decoder, tokenizer, data, optimizer, training.

## 3. Dataset and Tokenization

| Property | Value |
|---|---|
| Total paired molecules | 4,563 |
| Train / Valid / Test | 3,195 / 684 / 684 |
| Split | Scaffold |
| SPE vocab size | 128 |
| SPE merges | 86 |
| SPE train mean len | 5.5 |
| SPE test mean len | 6.8 |

## 4. Models

### E0 DirectConcat (unchanged from v2)
- 6 Pre-LN Transformer encoder layers
- All modality tokens concatenated → self-attention
- **1,804,544** params

### E1-3Layer (new v3)
- 3 stacked intra-cross blocks
- Each block per modality:
  1. Intra-modal self-attention (Pre-LN)
  2. Cross-modal attention (Pre-LN, learnable gate α=σ(g), g0=-2)
  3. Feed-forward network (Pre-LN)
- **4,186,121** params (2.32× E0, **diagnostic only**)

Formulas:
$$H_m^{b, intra} = H_m^b + \text{SelfAttn}_m^b(\text{LN}(H_m^b))$$
$$H_m^{b, cross} = H_m^{b, intra} + \alpha_{b,m} \cdot \text{CrossAttn}_{b,m}(\text{LN}(H_m^{b,intra}), \text{LN}(H_{\neg m}^{b,intra}))$$
$$H_m^{b+1} = H_m^{b, cross} + \text{FFN}_m^b(\text{LN}(H_m^{b, cross}))$$

## 5. Parameter Report

| Model | Encoder | Decoder | Total Params | Ratio |
|---|---|---|---|---|
| E0 | 6 self-attn layers | 2 layers, 578K | 1,804,544 | 1.00× |
| E1-3Layer | 3 intra-cross blocks | 2 layers, 578K | 4,186,121 | **2.32×** |

**WARNING**: v3 is diagnostic. E1-3Layer has 132% more parameters. This tests whether more cross-modal capacity helps, not a fair parameter-controlled comparison.

Decoder params identical: 578,176 each.

## 6. Training Protocol

| Hyperparameter | Value |
|---|---|
| d_model | 128 |
| num_heads | 4 |
| ffn_dim | 512 |
| patch_size | 64 |
| decoder_layers | 2 |
| E0 encoder_layers | 6 |
| E1 intra_cross_blocks | 3 |
| dropout | 0.1 |
| lr | 1e-4 |
| weight_decay | 1e-4 |
| warmup_steps | 500 |
| grad_clip | 1.0 |
| label_smoothing | 0.05 |
| batch_size | 32 |
| epochs | 50 |
| seed | 42 |

## 7. Metrics

Test loss (cross-entropy, label smoothing, ignore pad), token accuracy, canonical exact match, valid SMILES rate, Morgan Tanimoto, gate values.

## 8. Results

### Test Metrics

| Model | Test Loss | Test Token Acc | Exact Match | Params |
|---|---|---|---|---|
| **E0 DirectConcat** | **3.3380** | **0.2542** | 0.000 | 1.80M |
| E1-3Layer | 3.3886 | 0.2365 | 0.000 | 4.19M |

### Comparison with v2

| Model | v2 (1 block) | v3 (3 blocks) | Δ |
|---|---|---|---|
| E0 test loss | 3.3380 | 3.3380 | 0 (identical) |
| E1 test loss | 3.4842 | **3.3886** | **-0.096** |
| E1 token acc | 0.2271 | **0.2365** | **+0.009** |
| E1 params | 1.81M | 4.19M | +132% |

**E1-3Layer improves over v2 E1** (3.389 vs 3.484, -2.7%) but **still loses to E0** (3.389 vs 3.338, Δ=0.051). The 2.32× parameter increase buys only a modest 0.096 loss reduction.

## 9. Cross-Attention Gate Diagnostics

| Block | IR | 1H | 13C | Block Mean |
|---|---|---|---|---|
| block_0 | 0.1245 (+4.4%) | 0.1253 (+5.1%) | 0.1258 (+5.5%) | 0.1252 |
| block_1 | 0.1243 (+4.3%) | 0.1243 (+4.3%) | 0.1285 (+7.8%) | 0.1257 |
| block_2 | 0.1249 (+4.8%) | 0.1255 (+5.3%) | 0.1267 (+6.3%) | 0.1257 |

- Overall mean: 0.1255 (vs initial 0.1192, +5.3%)
- **All 9 gates remain near initialization** — cross-modal attention contributes minimally
- Block depth does NOT increase gate opening (block_2 mean = block_0 mean)
- E1 is learning primarily through intra-modal self-attention and the decoder

## 10. Qualitative Examples

(TBD — full beam search evaluation.)

## 11. Interpretation

### Did adding 3 blocks improve E1?
**Yes, modestly.** E1 test loss improved from 3.484 (v2, 1 block) to 3.389 (v3, 3 blocks), a 2.7% reduction. Token accuracy improved from 0.227 to 0.237.

### Did E1-3Layer beat E0?
**No.** E0 still achieves lower loss (3.338 vs 3.389) despite having 2.32× fewer parameters. Direct concat attention remains the more efficient architecture.

### Did gates open more in deeper E1?
**No.** All 9 gates remain at 0.124-0.129, barely above initialization (0.119). Deeper architecture does NOT encourage cross-modal information flow.

### Why doesn't deeper cross-attention help more?

1. **Near-zero cross-attention output init (Normal(0, 1e-4))**: Even with g0=-2, the output projection is near-zero, so gradient signal to cross-attention is very weak
2. **Decoder shortcut**: The autoregressive decoder can memorize SMILES patterns without needing encoder information
3. **Small dataset (3,195 train)**: Not enough samples to learn meaningful cross-modal relationships
4. **No physical bias**: Without coordinate encoding or modality-pair structure, cross-modal attention has no inherent advantage over self-attention

### Verdict
**Direct concat attention is genuinely stronger for this task at this scale.** Adding cross-modal interaction depth helps marginally (0.096 loss reduction) but costs 2.32× more parameters. The bottleneck appears to be the fundamental difficulty of learning cross-modal fusion from 3,195 samples, not insufficient model capacity.

## 12. Limitations

- Single seed
- E1-3Layer not parameter-matched (2.32× larger)
- Small dataset (3,195 train)
- NMR has no intensities
- No beam search evaluation
- Near-zero cross-attention output init may be too aggressive
- No coordinate encoding or physical bias

## 13. Next Steps

1. **Try removing near-zero cross-attention output init** — use Xavier uniform instead of Normal(0, 1e-4) for cross-attn out_proj
2. **Try g0=0** (alpha=0.5) to forcibly open gates
3. **Add coordinate-aware attention bias** on top of E0 first (stronger baseline before testing E1 again)
4. **Larger dataset or pretraining** — 3,195 pairs may be insufficient for cross-modal learning
5. **Multi-seed (42/43/44)** to confirm E0 > E1 is statistically significant

## 14. Reproducibility Checklist

| Item | Value |
|---|---|
| Git branch | spe128-e1-3layer-v3 |
| Git commit | a48d3ce |
| Server path | /data/home/sczc698/run/xxy/Trans-cross/ |
| Conda env | transpec |
| Python | 3.9.0, torch 2.0.0+cu118 |
| E0 output | runs/spe128_v3_concat_seed42/ |
| E1 output | runs/spe128_v3_e1_3layer_seed42/ |
| SPE vocab | data/processed/spe128_vocab.json |
| Raw files untouched | IR_NIST.jsonl, NMR_exp2.jsonl |
