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
- Prediction diversity (unique predictions / total)

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
| ir/h1/c13 mod | 128 each | 128 each |
| encoder (E0: 6 layers) | 1,189,632 | — |
| intra encoders (E1: 3×1 layer) | — | 594,816 |
| cross attn (E1: 3×1 layer) | — | 595,584 |
| decoder | 608,000 | 608,000 |
| CLS token | 128 | 128 |
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
- **Git commit:** `7d244db`
- **Environment:** miniforge3/24.11, transpec conda env
- **Data:** `/data/home/sczc698/run/xxy/Trans-cross/data/processed/`
- **SPE vocab:** `spe_vocab_256.json` (256 tokens, 214 merges)

### Slurm Jobs
| Job ID | Model | Node | Elapsed | Status |
|---|---|---|---|---|
| 987740 | E0-SPE (concat_equal) | g0030 | 2:20 | COMPLETED |
| 987741 | E1-SPE (intra_cross_equal) | g0042 | 2:22 | COMPLETED |

- Partition: gpu, GPUs: 1 each
- Logs: `/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/`

### Run Directories
- E0-SPE: `/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_concat_seed42`
- E1-SPE: `/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_intra_cross_seed42`

## 9. Results

### Test Split
| tokenizer | model | test loss | token acc | exact | canonical | validity | avg char len | unique preds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| regex_atom | E0 concat | 1.442 | 0.611 | 0.000 | — | 0.677 | 51.85 | — |
| regex_atom | E1 intra_cross | 1.478 | 0.577 | 0.000 | — | 0.721 | 24.37 | — |
| **spe** | **E0 concat** | **3.880** | **0.192** | **0.000** | **0.000** | **1.000** | **18.52** | **46/684** |
| **spe** | **E1 intra_cross** | **3.942** | **0.177** | **0.000** | **0.000** | **0.728** | **14.83** | **122/684** |

### Valid Split
| tokenizer | model | validity | avg char len | unique preds |
|---|---|---|---|---|
| **spe** | **E0 concat** | **1.000** | **18.37** | — |
| **spe** | **E1 intra_cross** | **0.756** | **15.23** | — |

### Training Curves
| model | best epoch | best valid loss | final train loss | final train acc |
|---|---|---|---|---|
| E0-SPE | 23 | 3.907 | 2.181 | 0.467 |
| E1-SPE | 15 | 3.930 | 1.609 | 0.614 |

Both models show significant overfitting (train loss continues decreasing while valid loss plateaus).

## 10. Prediction Examples
| target | atom E0 | atom E1 | SPE E0 | SPE E1 |
|---|---|---|---|---|
| C[C@@H](N)CO | — | — | CCCCCCOc1ccc(C)cc1 | CCCc1ccc(O)cc1 |
| CCCC(CC)CO | — | — | CCCCCCOc1ccc(C)cc1 | CCCCCC1 |
| c1ccc(CN2CCNCC2)cc1 | — | — | CCCC(=O)c1ccccc1 | Cc1ccc(C(=O)O)cc1 |
| O=C(Cc1ccc(-c2ccccc2)cc1)c1ccccc1 | — | — | Nc1cccc(C(=O)O)c1 | Cc1ccc(C(=O)c2ccccc2)cc1 |
| Cn1c(=O)sc2ccccc21 | — | — | Cc1ccc(C(=O)c2ccccc2)cc1 | O=C(O)c1ccccc1Cl |

SPE models tend to produce shorter, simpler SMILES. E0-SPE exhibits severe mode collapse
(repeating a small set of templates). E1-SPE has more diversity but generates many
invalid SMILES with unclosed rings and extra parentheses.

### E0-SPE Top Generated SMILES (mode collapse evidence)
| Generated SMILES | Count (of 684 test) |
|---|---|
| CCCC(=O)c1ccccc1 | 151 |
| CCCCCCOc1ccc(C)cc1 | 90 |
| Cc1ccc(C(=O)O)cc1 | 63 |
| Clc1ccc(Cc2ccccc2)cc1 | 61 |
| CCCCCCOc1ccccc1 | 50 |

### E1-SPE Top Generated SMILES
| Generated SMILES | Count (of 684 test) |
|---|---|
| Cc1ccc(C(=O)O)cc1 | 109 |
| CCCCCC1 | 89 |
| Cc1ccc(CO)cc1 | 42 |
| CCCCCCN1 | 40 |
| CCCCCC1CC1 | 24 |

## 11. Conclusion

### SPE improves validity?
**Yes, but with caveats.** E0-SPE achieves 100% RDKit validity (vs 67.7% atom-E0),
but this is due to severe **mode collapse** — the model generates only 46 unique
SMILES across 684 test samples. E1-SPE achieves 72.8% validity (vs 72.1% atom-E1),
a negligible improvement.

### SPE improves exact/canonical exact match?
**No.** Exact match remains 0.000 for both models under both tokenizers.

### SPE reduces generation length burden?
**Yes.** Average generated character length drops from 51.85→18.52 (E0) and
24.37→14.83 (E1). However, shorter outputs do not translate to correct outputs.

### E0 or E1 wins under SPE?
**E1-SPE is the better model** despite lower validity. E1 produces 122 unique
predictions (vs 46 for E0) and generates more chemically diverse SMILES.
E0-SPE's 100% validity is meaningless — the model collapsed to a few templates.

### Is exact match still zero?
**Yes.**

### Key Scientific Finding
The SPE tokenizer with vocab_size=256 and aggressive merging (77.9% length reduction)
**induces mode collapse in the DirectConcat encoder (E0)**. The IntraCross encoder
(E1) resists collapse, producing more diverse outputs. This suggests that:

1. Extremely short target sequences (mean 4.15 SPE tokens) may force the decoder
   into a low-diversity regime where a small set of template SMILES dominates.
2. The IntraCross encoder provides richer representations that help the decoder
   maintain output diversity even with a constrained tokenizer.
3. A larger SPE vocabulary (512 or 1024) might reduce mode collapse by allowing
   more fine-grained token sequences.

### Recommendation
Future work should try SPE with vocab_size=512 to trade off between sequence
length and output diversity. The current SPE configuration (256 tokens, 77.9%
reduction) is too aggressive for this task.
