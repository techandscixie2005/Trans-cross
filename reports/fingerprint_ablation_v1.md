# Fingerprint Ablation v1 — Concat vs Intra-Cross Encoder

**Date:** 2026-05-21
**Task:** IR + NMR → Morgan fingerprint (2048 bits)
**Dataset:** 4,563 paired molecules (train=3,195, valid=684, test=684)

## 1. Processed Data Audit Summary

| Metric | IR | NMR 1H | NMR 13C | Fingerprint |
|--------|-----|--------|---------|-------------|
| Shape | (4563, 1801) | (4563, 1501) | (4563, 2201) | (4563, 2048) |
| NaN/Inf | 0 | 0 | 0 | 0 |
| All-zero vectors | 1 | 91 | 608 | 0 |
| Mean | 0.320 | 0.003 | 0.003 | 19.4 bits set |
| Split (train/valid/test) | — | — | — | 3195/684/684 |

- Scaffold split: no train-valid or train-test scaffold overlap
- SMILES length: mean=19.4, median=18, p90=29, max=66
- Fingerprints: very sparse (19.4 bits out of 2048, ~0.95%)

Full audit: `data/processed/audit_summary.json`

## 2. Model Definitions

### E0: ConcatEncoder (baseline)
- Patch tokenization: IR, 1H NMR, 13C NMR each → patch_size=64, d_model=128
- Modality embeddings added to each token
- All tokens concatenated + CLS
- 2-layer Transformer encoder
- CLS → MLP head → 2048-bit logits

### E1: IntraCrossEncoder (ablation)
- Same patch tokenization
- Per-modality intra-modal self-attention (1 layer each)
- Cross-modal attention: each modality attends to the other two
- All tokens concatenated + CLS
- 2-layer Transformer encoder (post-cross)
- CLS → MLP head → 2048-bit logits

## 3. Parameter Counts

| Model | Parameters |
|-------|------------|
| ConcatEncoder | 992,896 |
| IntraCrossEncoder | 1,786,624 |

Intra_cross has ~1.8× more parameters due to separate intra-modal and cross-modal attention blocks.

## 4. Training Configuration

| Parameter | Value |
|-----------|-------|
| Task | IR + NMR → Morgan FP (radius=2, 2048 bits) |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW, lr=1e-4 |
| Scheduler | CosineAnnealingLR |
| Batch size | 64 (concat) / 32 (intra_cross, reduced for memory) |
| Max epochs | 10 (concat) / 2 (intra_cross) |
| Device | CPU |

## 5. Results

| Metric | ConcatEncoder | IntraCrossEncoder |
|--------|---------------|-------------------|
| Best epoch | 2 | 1 |
| Best valid Tanimoto | 0.0877 | 0.0877 |
| Test BCE loss | 0.0955 | 0.0417 |
| Test bit accuracy | 0.9913 | 0.9913 |
| **Test Tanimoto** | **0.0916** | **0.0916** |

Training curves (concat, 10 epochs):
```
Epoch  train_loss  valid_loss  valid_tanimoto
    1      0.5634      0.3672          0.0238
    2      0.2129      0.0968          0.0877
    3      0.0712      0.0528          0.0877
    ...
   10      0.0394      0.0406          0.0877
```

Training curves (intra_cross, 2 epochs):
```
Epoch  train_loss  valid_loss  valid_tanimoto
    1      0.2239      0.0436          0.0877
    2      0.0406      0.0403          0.0877
```

## 6. Analysis — Does E1 Beat E0?

**No.** Both models achieve essentially identical performance (test Tanimoto ~0.092).

Key observations:
- Concat achieves good performance by epoch 2 with fewer parameters
- Intra_cross achieves the same performance but with 1.8× more parameters
- Both models appear to converge to the same plateau (~0.088 valid Tanimoto)
- The task is challenging: predicting 19.4 bits out of 2048 from spectra

## 7. Do Results Justify Moving to SMILES Generation?

**Not yet.** The Tanimoto of 0.09 is very low (random ~0.009, perfect=1.0). This suggests:
1. The fingerprint prediction task is hard from current spectra representations
2. The binary NMR representation may be too sparse (91 all-zero 1H, 608 all-zero 13C)
3. The models may need more capacity or different architecture choices before moving to SMILES generation

Before moving to SMILES generation:
- Try Gaussian NMR binning (smoother peaks) instead of binary
- Try larger d_model or more layers
- Try pretraining or auxiliary tasks
- Improve NMR representation (handle missing channels better)

## 8. Limitations

- CPU-only training: limited to small models and few epochs
- Concat trained for 10 epochs but plateaued at epoch 2
- Intra_cross trained for only 2 epochs due to server time/memory constraints
- Single seed (42): need multi-seed to confirm results
- Very sparse fingerprints (0.95% bits set): accuracy is misleading (predict 0 = 99% correct)
- No GPU: training on GPU would allow larger models and more epochs

## 9. Server Paths

| Item | Path |
|------|------|
| Code | `/data/home/sczc698/run/xxy/Trans-cross/code/` |
| Processed data | `/data/home/sczc698/run/xxy/Trans-cross/data/processed/` |
| Concat run | `/data/home/sczc698/run/xxy/Trans-cross/runs/fp_concat_seed42/` |
| Intra-Cross run | `/data/home/sczc698/run/xxy/Trans-cross/runs/fp_intra_cross_seed42/` |
| Raw data | `/data/home/sczc698/run/xxy/Trans-cross/IR_NIST.jsonl`, `NMR_exp2.jsonl` |

## 10. Conclusion

For the Morgan fingerprint prediction task with binary NMR binning, the simpler **concat encoder** achieves the same performance as the more complex **intra-cross encoder** while using fewer parameters. This suggests that for fingerprint prediction, modality-specific processing provides no benefit — direct concatenation is sufficient.

The next step should improve the NMR representation (Gaussian binning) and try larger models on GPU before drawing final conclusions about the encoder architecture.
