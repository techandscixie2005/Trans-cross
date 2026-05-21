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
- Same training hyperparameters (epochs, batch_size, lr, seed=42)

## 4. Planned Metrics
- Token accuracy (teacher-forcing)
- Exact string match
- Canonical exact match (RDKit)
- RDKit validity
- Average generated character length
- Average generated SPE token length
- SPE token length reduction ratio
- Generation examples (target vs predicted)

## 5. SPE Vocabulary Stats (placeholder — fill after server run)
- Vocab size: TBD
- Number of merges: TBD
- Train unk rate: TBD
- Valid unk rate: TBD
- Test unk rate: TBD
- Atom token length mean/p50/p90/p95/max: TBD
- SPE token length mean/p50/p90/p95/max: TBD
- Length reduction: TBD%

## 6. Equal Parameter Verification (placeholder)
| component | E0-SPE | E1-SPE |
|---|---:|---:|
| spectral tokenizer | TBD | TBD |
| encoder | TBD | TBD |
| decoder | TBD | TBD |
| output head | TBD | TBD |
| total | TBD | TBD |
| relative diff | TBD% | |

## 7. Attention Bias Audit (placeholder)
- Status: TBD
- Report: reports/spe_attention_bias_audit.md

## 8. Server Run Trace (placeholder)
- Code path: TBD
- Git commit: TBD
- Environment: TBD
- Slurm job IDs: TBD
- Run directories: TBD

## 9. Results (placeholder)
| tokenizer | model | test loss | token acc | exact | canonical exact | validity | avg char len | avg token len |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|

## 10. Prediction Examples (placeholder)
| target | atom E0 | atom E1 | SPE E0 | SPE E1 | notes |
|---|---|---|---|---|---|

## 11. Conclusion (placeholder)
TBD after experiment completion.
