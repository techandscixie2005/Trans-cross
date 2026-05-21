# Experiment Verification Audit — SPE Tokenization Ablation

**Audit Date:** 2026-05-21
**Experiment:** SPE Decoder Tokenization SMILES Generation Ablation
**Branch:** master, **Commit:** 17f66cb

---

## A. Passed Checks

### Step 1 — Git / Run Identity
- [x] Branch: master, Commit: 17f66cb
- [x] Run directories exist: `spe_equal_concat_seed42`, `spe_equal_intra_cross_seed42`
- [x] Expected runs (E0, E1) x seed (42) = 2 runs, 2 found
- [x] No unexpected extra runs

### Step 2 — Run Completeness
- [x] `metrics.json` exists for both runs
- [x] `parameter_count.json`, `config_used.json`, `training_log.csv` exist for both
- [x] `best_model.pt`, `final_model.pt` exist for both
- [x] `evaluation_summary_test.json`, `evaluation_summary_valid.json` exist for both
- [x] `predictions_test.csv`, `predictions_valid.csv` exist for both (685 lines each = 684 samples + header)
- [x] Slurm jobs 987740, 987741 completed with exit code 0:0
- [x] No error logs or crash indicators in Slurm output

### Step 3 — Data Split Consistency
- [x] Both conditions use same splits.json (verified on server)
- [x] Train: 3195, Valid: 684, Test: 684 (total: 4563)
- [x] Split sizes identical across conditions

### Step 4 — Train/Valid/Test Leakage
- [x] Train ∩ Test = 0
- [x] Train ∩ Valid = 0
- [x] Valid ∩ Test = 0
- [x] All 4563 indices uniquely assigned
- [x] No leakage detected

### Step 5 — Tokenizer Leakage
- [x] SPE tokenizer trained ONLY on train split (3195 SMILES)
- [x] `build_spe_vocab.py` passes `--split train` and uses only `train_indices`
- [x] Train unk rate: 0.0% (all atom tokens covered)
- [x] Valid unk rate: 0.6151% (minor, expected from unseen atom combinations)
- [x] Test unk rate: 0.1374% (minor)
- [x] No data leakage from valid/test into SPE vocabulary

### Step 6 — Model Variant Correctness
- [x] Config flags: tokenizer.type=spe, same shared hyperparameters for both
- [x] Non-ablated hyperparameters identical: d_model=128, num_heads=4, decoder_layers=2, decoder_ffn_dim=512, dropout=0.1
- [x] Decoder params identical: 608,000 for both E0 and E1
- [x] Encoder topologies as designed: E0=6-layer self-attention, E1=1 intra+1 cross per modality
- [x] Total params: E0=1,834,368, E1=1,835,136 (diff=768, 0.0418%)

### Step 7 — Evaluation Consistency
- [x] Same evaluation script (`evaluate_smiles_model.py`) used for both
- [x] Same test set (684 samples) and valid set (684 samples)
- [x] Max decode length auto-loaded from run config (max_smiles_len=96)
- [x] Greedy decoding (no beam search) used for both

### Step 8 — Metric Recomputation
- [x] E0 metrics.json: best_epoch=23, test_loss=3.880, test_token_acc=0.1923 → matches report
- [x] E1 metrics.json: best_epoch=15, test_loss=3.942, test_token_acc=0.1772 → matches report
- [x] Evaluation summaries independently verified against predictions CSV
- [x] Parameter counts independently verified from parameter_count.json files

### Step 9 — Aggregation Correctness
- [x] E0-E1 parameter diff = (1,835,136 - 1,834,368) / max = 768/1,835,136 = 0.0418% → correct
- [x] Validity comparison: E0=1.000, E1=0.728 → correct from evaluation summaries
- [x] Exact match: 0.000 for both → correct

---

## B. Potential Issues

### B1 — Severe Mode Collapse in E0-SPE (Step 10)
E0-SPE generates only **46 unique SMILES** across 684 test samples. The top prediction
"CCCC(=O)c1ccccc1" appears 151 times (22.1% of outputs). The 100% RDKit validity is
an artifact of the model memorizing a few valid SMILES templates — it does NOT indicate
genuinely better generation quality.

**Impact:** The validity metric for E0-SPE is not comparable to E1-SPE or the old
atom-level results. E0-SPE has effectively collapsed to a template-based generator.

### B2 — Overfitting (Step 11)
Both models show clear overfitting:
- E0: train_loss continues decreasing (4.82→2.18) while valid_loss plateaus at ~4.0
- E1: train_loss continues decreasing (4.85→1.61) while valid_loss stagnates at ~3.93-4.12

The models are memorizing training SMILES patterns but not generalizing to validation/test.
This is a known limitation of the task (exact match = 0 for all models so far).

### B3 — Single Seed Only (Step 12)
Only seed=42 was run. Cross-seed variance cannot be assessed. The mode collapse in E0
may or may not be seed-dependent. Running additional seeds (e.g., 43, 44) would
strengthen conclusions.

### B4 — E1 Generates Some Invalid SMILES (Step 10)
E1-SPE produces 27.2% invalid SMILES (unclosed rings, extra parentheses). Examples:
"CCCCCC1", "CCCCCCN1", "CCCCCO)cc1". This suggests the SPE decoder sometimes
produces token sequences that don't form valid SMILES when concatenated.

---

## C. Serious Issues
**None.** No data leakage, no config mismatches, no parameter equality violations,
no evaluation inconsistencies. The mode collapse in E0 is a real finding, not an
artifact of experimental error.

---

## D. Are the Conclusions Trustworthy?

**Yes**, with the caveat that the E0 100% validity is due to mode collapse, not
genuinely better generation. The key conclusions are:

1. SPE with vocab_size=256 (77.9% compression) induces mode collapse in DirectConcat (E0) → **supported**
2. IntraCross encoder (E1) resists mode collapse better than DirectConcat → **supported**
3. SPE does not improve exact/canonical exact match → **supported** (both 0.000)
4. SPE reduces generated sequence length → **supported** (18.52 vs 51.85 for E0, 14.83 vs 24.37 for E1)

The finding that IntraCross resists mode collapse is scientifically interesting and
would benefit from multi-seed verification.

---

## E. Should Any Result Be Rerun?
**No.** The current results are internally consistent and the audit found no errors.
However, additional seeds (43, 44) would strengthen the mode collapse conclusion.

---

## F. Suggested Next Checks or Ablations
1. **SPE with vocab_size=512**: Less aggressive compression (estimated ~50% reduction)
   may preserve diversity while still shortening sequences
2. **Multi-seed runs (42, 43, 44)**: Quantify seed variance for the mode collapse phenomenon
3. **Old atom-level diversity check**: Count unique predictions for old E0/E1 to compare
   diversity baselines
4. **Temperature sampling**: Try temperature > 1.0 during generation to break mode collapse
5. **Decoder dropout increase**: Higher dropout might reduce E0 memorization
