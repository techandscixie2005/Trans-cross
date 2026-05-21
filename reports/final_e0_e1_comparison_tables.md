# Final E0 vs E1 Ablation Comparison Tables

**Generated:** 2026-05-21
**Run directories:** /data/home/sczc698/run/xxy/Trans-cross/runs/equal_concat_seed42, /data/home/sczc698/run/xxy/Trans-cross/runs/equal_intra_cross_seed42, /data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_concat_seed42, /data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_intra_cross_seed42

---

## Table 1. Tokenizer Statistics

| tokenizer | vocab size | train unk | valid unk | test unk | mean token len (train) | p95 token len | max token len |
|---|---:|---:|---:|---:|---:|---:|---:|
| regex_atom | - | N/A | N/A | N/A | N/A | N/A | N/A |

---

## Table 2. Equal-Parameter Verification

| tokenizer | model | total params | decoder params | encoder params | relative diff vs pair |
|---|---:|---:|---:|---:|---:|
| regex_atom | E0 DirectConcat / E1 IntraCross | 1,789,361 / 1,790,129 | 562,993 / 562,993 | (see by_module) | 0.0429% |
| spe | E0 DirectConcat / E1 IntraCross | 1,834,368 / 1,835,136 | 608,000 / 608,000 | (see by_module) | 0.0419% |

---

## Table 3. Per-Seed Test Results

| tokenizer | seed | model | loss | token acc | canon exact | validity | unique ratio | mode collapse | Tanimoto | scaffold | FG-F1 | avg char len |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| regex_atom | 42 | E0 DirectConcat | 1.4421 | 61.08% | 0.00% | 67.69% | 32.02% | 0.0775 | 0.1037 | 0.00% | 0.1882 | 51.91 |
| regex_atom | 42 | E1 IntraCross | 1.4778 | 57.68% | 0.00% | 72.08% | 24.12% | 0.0497 | 0.1059 | 0.00% | 0.1980 | 24.42 |
| spe | 42 | E0 DirectConcat | 3.8801 | 19.23% | 0.00% | 100.00% | 6.58% | 0.2208 | 0.1400 | 0.00% | 0.3008 | 18.52 |
| spe | 42 | E1 IntraCross | 3.9418 | 17.72% | 0.00% | 72.81% | 17.69% | 0.1594 | 0.1081 | 0.00% | 0.2052 | 14.83 |

---

## Table 4. Mean +/- Std Over Seeds (for each tokenizer)

| tokenizer | metric | E0 mean +/- std | E1 mean +/- std | winner | confidence |
|---|---:|---:|---:|---:|
| regex_atom | Loss | 1.4421 +/- 0.0000 | 1.4778 +/- 0.0000 | E0 | low |
| regex_atom | Token Accuracy | 0.6108 +/- 0.0000 | 0.5768 +/- 0.0000 | E0 | low |
| regex_atom | Exact String Match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| regex_atom | Canonical Exact Match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| regex_atom | RDKit Validity | 0.6769 +/- 0.0000 | 0.7208 +/- 0.0000 | E1 | low |
| regex_atom | Unique Ratio | 0.3202 +/- 0.0000 | 0.2412 +/- 0.0000 | E0 | low |
| regex_atom | Mode Collapse | 0.0775 +/- 0.0000 | 0.0497 +/- 0.0000 | E1 | low |
| regex_atom | Tanimoto Similarity | 0.1037 +/- 0.0000 | 0.1059 +/- 0.0000 | E1 | low |
| regex_atom | Scaffold Match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| regex_atom | FG-F1 | 0.1882 +/- 0.0000 | 0.1980 +/- 0.0000 | E1 | low |
| regex_atom | Avg Char Length | 51.9100 +/- 0.0000 | 24.4200 +/- 0.0000 | E1 | low |
| spe | Loss | 3.8801 +/- 0.0000 | 3.9418 +/- 0.0000 | E0 | low |
| spe | Token Accuracy | 0.1923 +/- 0.0000 | 0.1772 +/- 0.0000 | E0 | low |
| spe | Exact String Match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| spe | Canonical Exact Match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| spe | RDKit Validity | 1.0000 +/- 0.0000 | 0.7281 +/- 0.0000 | E0 | low |
| spe | Unique Ratio | 0.0658 +/- 0.0000 | 0.1769 +/- 0.0000 | E1 | low |
| spe | Mode Collapse | 0.2208 +/- 0.0000 | 0.1594 +/- 0.0000 | E1 | low |
| spe | Tanimoto Similarity | 0.1400 +/- 0.0000 | 0.1081 +/- 0.0000 | E0 | low |
| spe | Scaffold Match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| spe | FG-F1 | 0.3008 +/- 0.0000 | 0.2052 +/- 0.0000 | E0 | low |
| spe | Avg Char Length | 18.5200 +/- 0.0000 | 14.8300 +/- 0.0000 | E1 | low |

---

## Table 5. Condition-Shuffle Sensitivity

| tokenizer | seed | model | metric | paired | shuffled all | drop | winner |
|---|---:|---|---:|---:|---:|---:|
| spe | 42 | E0 DirectConcat | run_dir | ? | ? | ? | ? |
| spe | 42 | E0 DirectConcat | split | ? | ? | ? | ? |
| spe | 42 | E0 DirectConcat | shuffle_seed | ? | ? | ? | ? |
| spe | 42 | E0 DirectConcat | num_samples | ? | ? | ? | ? |
| spe | 42 | E0 DirectConcat | modes | {'rdkit_validity': 1.0, 'unique_generated': 45, 'unique_ratio': 0.065789, 'canonical_exact_match': 0.0, 'mean_tanimoto': 0.140036, 'mean_tanimoto_valid_only': 0.140036, 'scaffold_match_rate': 0.0, 'mean_fg_f1': 0.300798, 'mode_collapse_score': 0.22076, 'prediction_entropy': 2.7726, 'avg_pred_char_length': 18.52, 'mean_levenshtein': 0.0} | - | - | ? |
| spe | 42 | E0 DirectConcat | paired_vs_shuffled_deltas | - | - | - | ? |
| spe | 42 | E0 DirectConcat | interpretation_notes | ? | ? | ? | ? |
| spe | 42 | E1 IntraCross | run_dir | ? | ? | ? | ? |
| spe | 42 | E1 IntraCross | split | ? | ? | ? | ? |
| spe | 42 | E1 IntraCross | shuffle_seed | ? | ? | ? | ? |
| spe | 42 | E1 IntraCross | num_samples | ? | ? | ? | ? |
| spe | 42 | E1 IntraCross | modes | {'rdkit_validity': 0.72807, 'unique_generated': 121, 'unique_ratio': 0.176901, 'canonical_exact_match': 0.0, 'mean_tanimoto': 0.108064, 'mean_tanimoto_valid_only': 0.148425, 'scaffold_match_rate': 0.0, 'mean_fg_f1': 0.205173, 'mode_collapse_score': 0.159357, 'prediction_entropy': 3.6945, 'avg_pred_char_length': 14.83, 'mean_levenshtein': 0.0} | - | - | ? |
| spe | 42 | E1 IntraCross | paired_vs_shuffled_deltas | - | - | - | ? |
| spe | 42 | E1 IntraCross | interpretation_notes | ? | ? | ? | ? |

---

## Table 6. Architecture Verdict

| criterion | E0 DirectConcat | E1 IntraCross | winner | confidence |
|---|---:|---:|---:|
| Optimization (loss / token acc) | 1.4421 +/- 0.0000 | 1.4778 +/- 0.0000 | E0 | low |
| Exact / canonical exact match | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | tie | N/A |
| RDKit Validity | 0.6769 +/- 0.0000 | 0.7208 +/- 0.0000 | E1 | low |
| Diversity (unique ratio / collapse) | 0.3202 +/- 0.0000 | 0.2412 +/- 0.0000 | E0 | low |
| Target similarity (Tanimoto) | 0.1037 +/- 0.0000 | 0.1059 +/- 0.0000 | E1 | low |
| Condition sensitivity | N/A | N/A | N/A | N/A |
| **Overall** |  |  | **E0** | medium |

---

## Notes

- Token accuracy is comparable between E0 and E1 under the same tokenizer, but NOT across different tokenizers.
- Winner determination: E1 wins if mean > E0 mean (for most metrics, higher is better; for loss and mode_collapse, lower is better).
- Confidence: "high" if all 3 seeds agree, "medium" if 2 of 3 agree, "low" if mixed or insufficient data.
- Metrics marked '-' were not available for this run.
