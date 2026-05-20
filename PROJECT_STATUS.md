# Trans-cross Project Status

**Last updated:** 2026-05-21

## Current Phase

**Fingerprint ablation v1 complete** — encoder comparison done.

## Completed

### Phase 1: Data Investigation
- [x] Remote data inspection
- [x] IR/NMR file analysis
- [x] Pairing analysis (4,567 overlapping SMILES)
- [x] Investigation report
- [x] Example cases extracted

### Phase 2: Preprocessing Pipeline
- [x] Full preprocessing code (pairing, resampling, binning, splitting)
- [x] Server smoke test passed
- [x] Full preprocessing: **4,563 paired molecules**
- [x] IR (4563, 1801), NMR 1H (4563, 1501), NMR 13C (4563, 2201)
- [x] Scaffold split: train=3,195, valid=684, test=684

### Phase 3: Fingerprint Ablation v1
- [x] Processed data audit
- [x] Morgan fingerprint generation (2048 bits, 0 invalid)
- [x] ConcatEncoder: 992,896 params, test_tanimoto=0.0916
- [x] IntraCrossEncoder: 1,786,624 params, test_tanimoto=0.0916
- [x] Ablation report (`reports/fingerprint_ablation_v1.md`)
- [x] Code committed and pushed to GitHub

## Key Finding

For Morgan fingerprint prediction from IR + NMR spectra, the simpler **concat encoder** achieves the same performance as the **intra-cross encoder** while using 1.8× fewer parameters. Modality-specific attention provides no benefit for this task with binary NMR representation.

## Next Steps

1. Improve NMR representation: try Gaussian peak smearing instead of binary binning
2. Run multi-seed experiments to confirm statistical significance
3. Train on GPU for larger models and more epochs
4. Consider pretraining or auxiliary tasks before SMILES generation

## Server Paths

| Item | Path |
|------|------|
| Code | `/data/home/sczc698/run/xxy/Trans-cross/code/` |
| Processed data | `/data/home/sczc698/run/xxy/Trans-cross/data/processed/` |
| Concat run | `/data/home/sczc698/run/xxy/Trans-cross/runs/fp_concat_seed42/` |
| Intra-Cross run | `/data/home/sczc698/run/xxy/Trans-cross/runs/fp_intra_cross_seed42/` |
| Raw data | `/data/home/sczc698/run/xxy/Trans-cross/IR_NIST.jsonl`, `NMR_exp2.jsonl` |
