# SPE Tokenization SMILES Generation Ablation

## 1. Goal
Introduce atom-level + SPE (SMILES Pair Encoding) tokenization for the
decoder-side SMILES generation task, and rerun the equal-parameter
E0 (DirectConcat) vs E1 (IntraCross) ablation.

SPE is introduced to:
- Produce shorter decoder sequences (fewer autoregressive steps)
- Enable fragment-level SMILES generation (merged subword tokens)
- Potentially improve RDKit validity and generation stability
- Test whether decoder tokenization quality, rather than encoder attention
  bias, is the bottleneck for accurate SMILES generation

## 2. SPE Method Summary
1. **Atom-level base tokenization:** SMILES strings are split into chemically
   meaningful atom-level tokens (elements, bonds, brackets, stereochemistry, etc.)
2. **Pair frequency counting:** Adjacent token pair frequencies are counted
   across all training SMILES
3. **Iterative merging:** The most frequent token pair is merged into a new
   subword token; this repeats until the target vocabulary size is reached
   or the minimum frequency threshold is crossed
4. **Training only on train split:** SPE merge rules are learned exclusively
   from the training split to prevent data leakage into validation/test
5. **Greedy tokenization:** At inference time, atom-level tokens are first
   produced, then merge rules are applied in the order they were learned

## 3. Experimental Controls
- Same processed data (ir.npy, nmr_1h.npy, nmr_13c.npy)
- Same train/valid/test split (splits.json)
- Same E0/E1 encoder definitions
- Same decoder architecture (TransformerSmilesDecoder)
- Same parameter matching requirement (<= 1% relative diff)
- No attention bias of any kind
- Same training hyperparameters (epochs=30, batch_size=32, lr=1e-4, seed=42)

## 4. Planned Metrics
- Token accuracy (teacher-forcing)
- Exact string match
- Canonical exact match (RDKit)
- RDKit validity
- Average generated character length
- Average generated SPE token length
- SPE token length reduction ratio
- Generation examples (target vs predicted)

## 5. SPE Vocabulary Summary
- Vocab size: **256** (reached target)
- Number of merges: **214**
- Train unk rate: **0.0%** (perfect)
- Valid unk rate: **0.6151%**
- Test unk rate: **0.1374%**

### Atom Token Length Stats
| split | mean | p50 | p90 | p95 | max |
|---|---|---|---|---|---|
| train | 18.79 | 18 | 26 | 29 | 64 |
| valid | 18.36 | 17 | 31 | 34 | 61 |
| test | 17.73 | 16 | 30.7 | 35.85 | 50 |

### SPE Token Length Stats
| split | mean | p50 | p90 | p95 | max |
|---|---|---|---|---|---|
| train | 4.15 | 4 | 6 | 7 | 16 |
| valid | 5.47 | 5 | 9 | 11 | 28 |
| test | 5.32 | 4 | 9 | 11 | 20 |

### Length Reduction
- Train: **77.9%** (18.79 → 4.15 mean tokens)
- Valid: 70.2% (18.36 → 5.47 mean tokens)
- Test: 70.0% (17.73 → 5.32 mean tokens)

## 6. Equal Parameter Verification

| component | E0-SPE | E1-SPE |
|---|---:|---:|
| ir_tokenizer | 12,032 | 12,032 |
| h1_tokenizer | 11,392 | 11,392 |
| c13_tokenizer | 12,800 | 12,800 |
| ir_mod / h1_mod / c13_mod | 128 each | 128 each |
| encoder_layers (E0) | 1,189,632 | — |
| ir_intra + h1_intra + c13_intra (E1) | — | 198,272 each |
| ir_cross + h1_cross + c13_cross (E1) | — | 198,528 each |
| decoder | 608,000 | 608,000 |
| _direct_params (CLS) | 128 | 128 |
| **TOTAL** | **1,834,368** | **1,835,136** |
| **Relative diff** | | **0.0418%** |

- Decoder params identical: YES (608,000 each)
- Within 1% tolerance: PASS
- Within 0.5% tolerance: PASS

## 7. Attention Bias Audit

Status: **PASS**

- [x] No coordinate bias
- [x] No modality-pair bias
- [x] No relative position bias
- [x] No Graphormer-style spatial/distance bias
- [x] No learned additive attention-logit bias
- [x] Only causal mask (decoder autoregression) and padding masks present

Report: `reports/spe_attention_bias_audit.md`

## 8. Server Run Trace
- **Code path:** `/data/home/sczc698/run/xxy/Trans-cross/code/`
- **Git commit:** `0c02237` (fix: remove explicit --mem from SPE Slurm scripts)
- **Environment:** miniforge3/24.11, transpec conda env
- **Data:** `/data/home/sczc698/run/xxy/Trans-cross/data/processed/`
- **SPE vocab:** `spe_vocab_256.json` (256 tokens, 214 merges)

### Commands
```bash
# Build vocab
python scripts/build_spe_vocab.py \
  --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
  --out /data/home/sczc698/run/xxy/Trans-cross/data/processed/spe_vocab_256.json \
  --vocab-size 256 --min-frequency 2 --split train

# Parameter check
python scripts/compare_model_params.py \
  --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
  --vocab /data/home/sczc698/run/xxy/Trans-cross/data/processed/spe_vocab_256.json \
  --config configs/smiles_spe_equal_param.yaml

# Bias audit
python scripts/audit_attention_bias.py \
  --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
  --config configs/smiles_spe_equal_param.yaml
```

### Slurm Jobs
| Job ID | Model | Node | Status |
|---|---|---|---|
| 987740 | E0-SPE (concat_equal) | g0030 | Running |
| 987741 | E1-SPE (intra_cross_equal) | g0042 | Running |

- Partition: gpu
- GPUs: 1 each
- Time: 12:00:00 each
- Logs: `/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/`

### Smoke Training (1 epoch, CPU)
| Model | Train Loss | Valid Loss | Test Loss | Test Acc |
|---|---|---|---|---|
| E0-SPE | 4.6502 | 4.5449 | 4.5257 | 0.1676 |
| E1-SPE | 4.6807 | 4.5444 | 4.5272 | 0.1659 |

### Run Directories
- E0-SPE: `/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_concat_seed42`
- E1-SPE: `/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_intra_cross_seed42`

## 9. Results (placeholder — training in progress)

| tokenizer | model | test loss | token acc | exact | canonical exact | validity | avg char len | avg token len |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

Results pending completion of Slurm jobs 987740 and 987741 (30 epochs).

## 10. Prediction Examples (placeholder)
| target | atom E0 | atom E1 | SPE E0 | SPE E1 | notes |
|---|---|---|---|---|---|

Pending evaluation after training.

## 11. Conclusion (placeholder)
Pending experiment completion. Preliminary observations:
- SPE achieves 77.9% sequence length reduction
- Train unk rate is 0% (perfect coverage)
- Parameter matching within 0.0418% (well under 1% constraint)
- No attention bias violations
- Smoke training converges normally with finite loss
