# Final Equal-Parameter E0 vs E1 SMILES Generation Report

**Status**: INTERIM — Multi-seed runs (Slurm job 987794) pending on bjhpc GPU queue.
This report covers seed=42 across atom and SPE-256 tokenizers. SPE-512 training pending.

## 1. Executive Summary

Under seed=42 with equal parameters and no attention bias:

- **Neither model recovers molecular structures**: exact match = 0.000, canonical exact = 0.000, scaffold match = 0.000 across all conditions.
- **E0 (DirectConcat) has lower loss and higher token accuracy** in both atom and SPE-256 conditions, indicating easier optimization.
- **E1 (IntraCross) maintains higher output diversity** and strongly resists mode collapse under aggressive SPE tokenization (121 vs 45 unique SMILES).
- **E1 shows stronger condition sensitivity**: Tanimoto drops 34% under shuffled spectra vs 18% for E0.
- **E0 under SPE-256 achieves 100% validity but extreme mode collapse** (only 6.6% unique), making the high validity meaningless.
- **Mean Tanimoto is low across all conditions** (0.10–0.14), and the best valid-only Tanimoto is 0.15, indicating weak target proximity.

**Verdict (seed42 only)**: **Weak E1 win** — E1 shows better diversity and condition sensitivity, but neither model solves structure recovery. This conclusion requires multi-seed verification.

**Confidence**: LOW — single seed only, pending seeds 43 and 44.

## 2. Experimental Question

Does IntraCross (E1) outperform DirectConcat (E0) for SMILES generation from IR/NMR spectra when parameters, decoder, tokenizer, split, training, and attention bias are controlled?

## 3. Models

| Model | Encoder Architecture | Description |
|---|---|---|
| E0 DirectConcat | 6 self-attention layers | All spectral modalities concatenated and passed through shared self-attention |
| E1 IntraCross | 1 intra + 1 cross per modality | Separate intra-modal encoders per modality + cross-modal attention between modalities |

Both share:
- Identical Transformer decoder (2 layers, d_model=128, 4 heads)
- Same tokenizer per condition (atom or SPE)
- No attention bias of any kind (verified by audit)

## 4. Controls

| Variable | Status |
|---|---|
| Equal parameter count (≤1%) | PASS (0.04% diff) |
| Same decoder | PASS (identical 608K/668K params) |
| Same tokenizer per condition | PASS |
| Same data split | PASS (3195/684/684) |
| No data leakage | PASS (tokenizer train-only, 0% overlap) |
| No attention bias | PASS (audit verified) |
| Same training hyperparameters | PASS |
| Same evaluation script | PASS |

## 5. Tokenizer Conditions

### 5.1 Tokenizer Statistics

| tokenizer | vocab size | train unk | valid unk | test unk | mean train tok len | p95 tok len | max tok len |
|---|---:|---:|---:|---:|---:|---:|---:|
| atom (regex) | 84 | 0.0% | 0.0% | 0.0% | 18.79 | 29 | 64 |
| SPE-256 | 256 | 0.0% | 0.615% | 0.137% | 4.15 | 11 | 28 |
| SPE-512 | 512 | 0.0% | 0.689% | 0.154% | 3.28 | 10 | 27 |

SPE reduces mean token length by 78-83% vs atom tokenization. SPE-512 produces even shorter sequences than SPE-256 (3.28 vs 4.15 mean) due to more aggressive merging (470 merges vs 214).

### 5.2 Parameter Matching

| tokenizer | model | total params | decoder params | encoder params | relative diff |
|---|---:|---:|---:|---:|---:|
| atom | E0 DirectConcat | 1,638,016 | 608,000 | 1,007,744 | — |
| atom | E1 IntraCross | 1,638,784 | 608,000 | 1,008,512 | 0.0469% |
| SPE-256 | E0 DirectConcat | 1,834,368 | 608,000 | 1,204,096 | — |
| SPE-256 | E1 IntraCross | 1,835,136 | 608,000 | 1,204,864 | 0.0418% |
| SPE-512 | E0 DirectConcat | 1,894,016 | 667,648 | 1,204,096 | — |
| SPE-512 | E1 IntraCross | 1,894,784 | 667,648 | 1,204,864 | 0.0405% |

All within 0.5% tolerance. Decoder identical within each tokenizer condition.

## 6. Main Results (Seed 42)

### 6.1 Test Split Per-Seed Results

| tokenizer | seed | model | loss | token acc | canon exact | validity | unique ratio | mode collapse | Tanimoto | scaffold | FG-F1 | avg char len |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| atom | 42 | E0 | 1.442 | 0.611 | 0.000 | 0.677 | 0.320 | 0.078 | 0.104 | 0.000 | 0.188 | 51.9 |
| atom | 42 | E1 | 1.478 | 0.577 | 0.000 | 0.721 | 0.241 | 0.050 | 0.106 | 0.000 | 0.198 | 24.4 |
| spe256 | 42 | E0 | 3.880 | 0.192 | 0.000 | 1.000 | 0.066 | 0.221 | 0.140 | 0.000 | 0.301 | 18.5 |
| spe256 | 42 | E1 | 3.942 | 0.177 | 0.000 | 0.728 | 0.177 | 0.159 | 0.108 | 0.000 | 0.205 | 14.8 |

### 6.2 Valid Split

| tokenizer | seed | model | validity | unique ratio | mode collapse | Tanimoto |
|---|---:|---|---:|---:|---:|---:|
| atom | 42 | E0 | 0.670 | 0.319 | 0.075 | 0.109 |
| atom | 42 | E1 | 0.608 | 0.251 | 0.058 | 0.098 |
| spe256 | 42 | E0 | 1.000 | 0.070 | 0.174 | 0.141 |
| spe256 | 42 | E1 | 0.756 | 0.171 | 0.164 | 0.117 |

## 7. Mode Collapse Analysis

### Atom Tokenizer
- E0: 219 unique (32.0%), entropy 4.62, mode collapse 0.078
- E1: 165 unique (24.1%), entropy 4.52, mode collapse 0.050
- Both show reasonable diversity with atom tokenizer

### SPE-256 Tokenizer (CRITICAL)
- **E0**: 45 unique (6.6%), entropy 2.77, mode collapse 0.221
  - Top 5 outputs account for >60% of all predictions
  - Model generates templates: `CCCC(=O)c1ccccc1` (151×), `CCCCCCOc1ccc(C)cc1` (90×)
- **E1**: 121 unique (17.7%), entropy 3.70, mode collapse 0.159
  - More diverse, but still top templates dominate: `Cc1ccc(C(=O)O)cc1` (109×), `CCCCCC1` (89×)

**Finding**: SPE-256 induces extreme mode collapse in E0. E1 is more resistant, producing 2.7× more unique outputs. However, both models collapse to a small set of templates.

### SPE-512 (PENDING)
Training runs not yet completed.

## 8. Condition-Shuffle Test

Tests whether models actually use IR/NMR spectral conditions by shuffling spectra across samples.

### SPE-256 E0 (DirectConcat)

| Metric | Paired | Shuffle All | Drop |
|---|---:|---:|---:|
| Validity | 1.000 | 1.000 | 0.000 |
| Unique Ratio | 0.066 | 0.066 | 0.000 |
| Tanimoto | 0.140 | 0.115 | 0.025 (17.9%) |
| FG-F1 | 0.301 | 0.288 | 0.013 (4.3%) |
| Mode Collapse | 0.221 | 0.221 | 0.000 |

**E0 Interpretation**: Validity and diversity are unchanged under shuffled spectra — E0 is strongly decoder-prior dominated. The modest Tanimoto drop (18%) suggests weak conditional use. Mode collapse persists regardless of condition correctness.

### SPE-256 E1 (IntraCross)

| Metric | Paired | Shuffle All | Drop |
|---|---:|---:|---:|
| Validity | 0.728 | 0.728 | 0.000 |
| Unique Ratio | 0.177 | 0.177 | 0.000 |
| Tanimoto | 0.108 | 0.071 | **0.037 (34.0%)** |
| FG-F1 | 0.205 | 0.164 | **0.041 (20.1%)** |
| Mode Collapse | 0.159 | 0.159 | 0.000 |

**E1 Interpretation**: Tanimoto drops 34% when spectra are shuffled — E1 is **significantly more condition-sensitive** than E0. However, diversity metrics (unique ratio, entropy, validity) are unchanged under shuffle, suggesting the decoder prior still dominates the output vocabulary.

### Condition-Shuffle Verdict
- E1 uses spectral conditions more than E0 (34% vs 18% Tanimoto drop)
- Neither model shows diversity dependence on correct conditions
- Both models' output distributions are primarily decoder-prior driven

## 9. Target Similarity

| Metric | Atom E0 | Atom E1 | SPE-256 E0 | SPE-256 E1 |
|---|---:|---:|---:|---:|
| Canonical Exact | 0.000 | 0.000 | 0.000 | 0.000 |
| Scaffold Match | 0.000 | 0.000 | 0.000 | 0.000 |
| Mean Tanimoto | 0.104 | 0.106 | 0.140 | 0.108 |
| Mean Tanimoto (valid only) | 0.153 | 0.147 | 0.140 | 0.148 |
| Mean FG-F1 | 0.188 | 0.198 | 0.301 | 0.205 |
| Mean Levenshtein | 46.4 | 20.2 | 15.1 | 13.3 |

**Key findings**:
- Zero exact or scaffold matches across all conditions — **structure recovery is not solved**.
- SPE-256 E0's higher mean Tanimoto (0.140) is an artifact of 100% validity (all predictions are valid, so Tanimoto is never zeroed). E1's valid-only Tanimoto (0.148) is actually slightly higher.
- FG-F1 is moderate (0.19–0.30), suggesting some functional group preservation.
- Levenshtein distances are large relative to average target length (~18 chars).

## 10. Prediction Examples

| tokenizer | target | E0 prediction | E1 prediction | E0 Tanimoto | E1 Tanimoto |
|---|---|---|---|---|---|
| spe256 | C[C@@H](N)CO | CCCCCCOc1ccc(C)cc1 | CCCc1ccc(O)cc1 | 0.036 | 0.077 |
| spe256 | CCCC(CC)CO | CCCCCCOc1ccc(C)cc1 | CCCCCC1 | 0.077 | 0.133 |
| spe256 | c1ccc(CN2CCNCC2)cc1 | CCCC(=O)c1ccccc1 | Cc1ccc(C(=O)O)cc1 | 0.100 | 0.171 |
| spe256 | Cn1c(=O)sc2ccccc21 | Cc1ccc(C(=O)c2ccccc2)cc1 | O=C(O)c1ccccc1Cl | 0.095 | 0.067 |
| atom | C[C@@H](N)CO | CCCCCCCCCCCCCCCCCCCCCCCCCCCO | CCCCC(=O)c1ccccc1 | 0.034 | 0.040 |

Both models generate chemically plausible but incorrect molecules. E0 under SPE-256 strongly prefers templates with `c1ccc(...)cc1` benzene core. E1 generates more varied but still incorrect structures.

## 11. Interpretation

Following the pre-registered rules:

1. **Exact/canonical exact = 0**: Structure recovery is NOT solved. Neither model can be claimed to "recover correct molecular structures."

2. **E1 has higher diversity and condition sensitivity**: E1 better preserves conditional diversity. Under SPE-256, E1 produces 2.7× more unique SMILES and shows 2× larger condition-shuffle Tanimoto drop. E1 more genuinely uses spectral information.

3. **E0 has lower loss but collapses**: Under SPE-256, E0 achieves lower loss (3.880 vs 3.942) and higher token accuracy (0.192 vs 0.177), but this comes at the cost of extreme diversity collapse (6.6% unique). E0 finds a "lazy" optimum — output a small set of plausible templates regardless of input spectra.

4. **Tanimoto/scaffold/FG-F1**: No consistent winner. SPE-256 E0 has higher mean Tanimoto and FG-F1, but this is driven by 100% validity (no zero-Tanimoto penalties). E1's valid-only metrics are comparable or better.

## 12. Final Verdict

| Criterion | E0 DirectConcat | E1 IntraCross | Winner | Confidence |
|---|---|---|---|---|
| Optimization (loss/token acc) | Lower loss, higher acc | Higher loss, lower acc | E0 | High |
| Exact/canonical exact | 0.000 | 0.000 | Tie | High |
| Validity | Higher (100% SPE) | Lower (73% SPE) | E0 (artifact) | Low |
| Diversity/collapse | Severe collapse (SPE) | Moderate diversity | E1 | High |
| Target similarity (Tanimoto) | 0.140 (all valid) | 0.108 (0.148 valid) | Tie | Medium |
| Condition sensitivity | Weak (18% drop) | Strong (34% drop) | E1 | Medium |
| **Overall** | Easy optimization, collapses | Preserves diversity, more condition-sensitive | **Weak E1** | **Low** |

### Verdict: WEAK E1 WIN (seed42 only, low confidence)

E1 IntraCross is the better architecture under this experimental setup, primarily because it:
1. Maintains output diversity when the tokenizer is aggressive (SPE-256)
2. Shows genuine conditional use of spectral information
3. Does not sacrifice these properties just for easier optimization

However, this conclusion has **low confidence** because:
- Only seed=42 available (multi-seed pending)
- No SPE-512 results available
- Neither model achieves any exact structure matches
- The "win" is about resistance to collapse, not about solving the core task

## 13. Limitations

- **Single seed only**: Seeds 43 and 44 pending (Slurm job 987794 queued)
- **SPE-512 not trained**: Expected to reduce mode collapse by providing finer-grained tokens
- **No beam search**: Only greedy decoding evaluated; beam search could improve structure recovery
- **Small dataset**: 3195 training pairs limits generalization
- **No pretraining**: Models trained from scratch
- **No chemical constraints**: Formula, mass, valence not enforced
- **No coordinate encoding**: Positional information not provided to encoder
- **Only IR/NMR peak-binned input**: Limited spectral resolution

## 14. Next Steps

1. **Complete multi-seed runs** (SPE-256 seeds 43,44; SPE-512 seeds 42,43,44)
2. **Evaluate SPE-512** — expected to balance sequence length and diversity
3. **Run full aggregation** with multi-seed mean±std
4. **Add atom condition-shuffle** for baseline comparison
5. **Consider beam search** top-k evaluation
6. **Consider SPE vocab sweep** (128, 256, 512, 1024)
7. **Larger model** or pretraining

## Appendix A. Run Directory Map

| Run | Path | Status |
|---|---|---|
| atom E0 seed42 | runs/equal_concat_seed42 | Complete |
| atom E1 seed42 | runs/equal_intra_cross_seed42 | Complete |
| SPE-256 E0 seed42 | runs/spe_equal_concat_seed42 | Complete |
| SPE-256 E1 seed42 | runs/spe_equal_intra_cross_seed42 | Complete |
| SPE-256 seeds 43,44 (×4) | runs/spe_equal_{concat,intra_cross}_seed{43,44} | PENDING (grid 987794) |
| SPE-512 seeds 42,43,44 (×6) | runs/spe512_equal_{concat,intra_cross}_seed{42,43,44} | PENDING (grid 987794) |

## Appendix B. Git Information

- Commit: `efb8baf` on master
- Push: `github.com/techandscixie2005/Trans-cross` — pushed
- Server code: `/data/home/sczc698/run/xxy/Trans-cross/code/` — synced
