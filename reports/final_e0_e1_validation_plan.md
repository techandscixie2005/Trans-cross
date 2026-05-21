# Final E0 vs E1 SMILES Generation Validation Plan

**Date:** 2026-05-21
**Repository:** https://github.com/techandscixie2005/Trans-cross
**Reference:** https://github.com/techandscixie2005/TranSpec (inspected, NOT modified)

---

## Primary Question

Under equal parameter count and no attention bias, does the IntraCross encoder topology (E1) outperform DirectConcat (E0) for SMILES generation from IR/NMR spectra?

This question is evaluated across three tokenization regimes (atom-level, SPE-256, SPE-512) and three random seeds to distinguish genuine architectural effects from initialization noise.

---

## Model Variants

Three tokenizers, each with E0 and E1 encoder topology:

| Variant | Tokenizer | Encoder | Status |
|---------|-----------|---------|--------|
| atom E0 | regex_atom | DirectConcat (6 layers) | Existing |
| atom E1 | regex_atom | IntraCross (1 intra + 1 cross) | Existing |
| SPE-256 E0 | SPE (vocab_size=256) | DirectConcat (6 layers) | Existing (seed 42) + new (seeds 43, 44) |
| SPE-256 E1 | SPE (vocab_size=256) | IntraCross (1 intra + 1 cross) | Existing (seed 42) + new (seeds 43, 44) |
| SPE-512 E0 | SPE (vocab_size=512) | DirectConcat (6 layers) | New |
| SPE-512 E1 | SPE (vocab_size=512) | IntraCross (1 intra + 1 cross) | New |

### Encoder Definitions (Unchanged Across All Variants)

**E0 (DirectConcat):**
- 6 standard Pre-LN self-attention blocks over concatenated spectral tokens
- d_model=128, num_heads=4, FFN dim=512

**E1 (IntraCross):**
- 1 intra-modal self-attention block per modality (3 total)
- 1 cross-modal attention block per modality (3 total)
- Cross-attention output projections zero-initialized
- No fusion layers
- d_model=128, num_heads=4, FFN dim=512

### Decoder (Identical Across All Variants)

- 2-layer TransformerSmilesDecoder, d_model=128, d_ff=512, num_heads=4
- Architecture and hyperparameters byte-identical between E0 and E1

---

## Seeds

Three seeds for SPE-256 and SPE-512 variants; two seeds minimum for atom variant (42 mandatory, 43 and 44 preferred):

| Seed | atom | SPE-256 | SPE-512 |
|------|------|---------|---------|
| 42 | Existing | Existing | New |
| 43 | Optional | New | New |
| 44 | Optional | New | New |

---

## Primary Metrics

These metrics determine the winner for each tokenizer condition. Improvement in multiple primary metrics across seeds constitutes evidence for architecture superiority.

### 1. Canonical Exact Match

RDKit canonicalization of both target and predicted SMILES, then exact string comparison. This is the strictest metric -- it requires the model to generate the correct molecule, not just any valid molecule.

- Calculation: `CanonSmiles(target) == CanonSmiles(prediction)`
- Interpretation: Non-zero values are significant; zero values mean the model has not learned molecule identity

### 2. RDKit Validity

Percentage of generated SMILES strings that parse successfully via `rdkit.Chem.MolFromSmiles` (non-None result).

- Calculation: `count(valid) / count(total)`
- Interpretation: Measures SMILES grammar learning independent of correctness

### 3. Morgan Tanimoto Similarity to Target

Fingerprint-based similarity between generated and target molecules using Morgan circular fingerprints (radius=2, nBits=2048). Only evaluated on valid SMILES pairs.

- Calculation: `TanimotoSimilarity(fp_pred, fp_target)` via RDKit
- Interpretation: Higher values indicate better functional group and ring structure proximity

### 4. Output Diversity / Unique Ratio

Ratio of unique generated SMILES to total generated SMILES. Measures whether the model collapses to a small set of outputs or produces diverse predictions.

- Calculation: `len(set(predictions)) / len(predictions)`
- Interpretation: Low ratio suggests mode collapse; high ratio shows the model respects input variation

### 5. Condition-Shuffle Sensitivity

Measure of how much output quality degrades when spectral conditions are shuffled (modality mismatch). A model with strong conditional dependence on each modality should show significant degradation.

- Calculation: Drop in canonical exact match, validity, and Tanimoto similarity under shuffled vs normal conditions
- Interpretation: Larger drops indicate stronger conditional use of each modality; small drops suggest the model ignores some modalities

---

## Secondary Metrics

These metrics provide supporting evidence and diagnostics. They do not independently determine the winner but inform interpretation of primary metrics.

### 1. Loss (Teacher-Forcing Cross-Entropy)

Standard cross-entropy loss on the test set. Comparable across E0/E1 within the same tokenizer. NOT comparable across different tokenizers.

### 2. Token Accuracy

Per-token prediction accuracy under teacher forcing. Comparable between E0 and E1 under the same tokenizer. NOT comparable across atom vs SPE tokenizers.

### 3. Character Length (Average and Distribution)

Length of generated SMILES strings in characters. Shorter sequences are generally preferred (fewer error opportunities).

- Report: mean, median, P10, P90
- Compare to target distribution to detect length bias

### 4. Token Length

Length of generated sequences in tokens (post-tokenization). Relevant for SPE analysis since SPE compresses long atom sequences into fewer tokens.

- Report: mean, median, P10, P90
- Compute compression ratio: `avg_char_len / avg_token_len`

### 5. Scaffold Match (Bemis-Murcko)

Whether the Bemis-Murcko scaffold (ring systems + linker atoms) of the generated molecule matches the target scaffold. Evaluated only on valid SMILES pairs.

- Calculation: `MurckoScaffoldSmiles(target) == MurckoScaffoldSmiles(predicted)`
- Interpretation: Indicates whether the model captures the core molecular architecture even when side chains differ

### 6. Functional Group F1 (Precision, Recall, F1)

Per-functional-group precision, recall, and F1 score comparing generated molecules to targets. Uses RDKit functional group decomposition.

- Calculation: Tokenize each molecule into functional group substructures; compute micro-averaged precision, recall, F1
- Interpretation: Reveals which functional groups are well-captured and which are systematically missed

---

## Controlled Variables

The following variables are held constant across all E0/E1 comparisons:

| Variable | Specification |
|----------|---------------|
| Data split | Identical train/valid/test indices from `splits.json` |
| Tokenizer | Same tokenizer within each tokenizer condition (atom, SPE-256, SPE-512) |
| Decoder architecture | Byte-identical across E0/E1 within each tokenizer condition |
| Parameter count | Within 1% relative difference between E0 and E1 |
| Optimizer | AdamW, same learning rate, same weight decay |
| Decoding method | Greedy decoding, same max length |
| Attention bias | None (verified by audit; see Strict Experimental Boundaries) |
| Spectral patch tokenization | Same patch size, same embedding dimension |
| Modality embeddings | Same configuration (learnable per-modality vector) |
| Epochs | Same for E0/E1 within each comparison pair |

---

## Strict Experimental Boundaries

These boundaries must not be violated. Any violation invalidates the corresponding comparison.

### Data Boundaries

- **No modification of raw data:** Raw IR, 1H NMR, and 13C NMR files are read-only
- **No modification of processed IR/NMR arrays:** Preprocessed `.npy` arrays are read-only
- **No modification of TranSpec repository:** The reference repository is inspected but never modified

### Model Boundaries

- **No attention bias of any kind:** No coordinate bias, modality-pair bias, relative bias, learned additive attention bias, Graphormer-style bias, or spectral x-axis bias
- **Allowed masks only:** Causal mask (autoregressive decoder only), padding masks (batching correctness only)
- **Same decoder:** Within each tokenizer condition, the decoder is byte-identical for E0 and E1
- **Parameter counts matched within 1%:** Between E0 and E1 for each tokenizer condition

### Evaluation Boundaries

- **Same split evaluation:** E0 and E1 evaluated on identical test indices
- **Same evaluation script:** The identical evaluation pipeline for E0 and E1
- **No post-hoc tuning:** No per-model hyperparameter optimization after results are observed

### Scope Boundaries

- No comparison across tokenizers for token accuracy (only E0 vs E1 within same tokenizer)
- No claim of molecule identity recovery if exact/canonical exact match is zero
- No modification of experimental design mid-run (pre-registered analysis)

---

## Interpretation Rules

### Rule 1: Exact Match Zero

If exact match and canonical exact match are zero for both E0 and E1, do NOT claim structure recovery. Report the fact explicitly and focus interpretation on validity, Tanimoto similarity, diversity, and conditional sensitivity.

### Rule 2: Higher Unique Ratio + Higher Condition-Shuffle Drop

If E1 has a higher unique ratio AND a larger condition-shuffle performance drop than E0, claim stronger conditional diversity. This pattern indicates that E1 produces more input-dependent outputs (respecting spectral variation) rather than falling back to a generic prediction distribution.

### Rule 3: Lower Loss / Higher Token Accuracy but Low Diversity

If E0 has lower loss and higher token accuracy but lower output diversity (lower unique ratio), claim easier optimization but possible mode collapse. This pattern suggests the DirectConcat encoder converges more readily but may produce less input-sensitive predictions.

### Rule 4: Higher Tanimoto / Scaffold / FG-F1

If E1 shows higher Morgan Tanimoto similarity, higher scaffold match rate, and higher functional group F1, claim better target proximity. This is the strongest evidence that IntraCross produces more chemically accurate molecules.

### Rule 5: Token Accuracy Comparability

- Token accuracy is NOT comparable across atom vs SPE tokenizers
- Token accuracy IS comparable between E0 and E1 under the same tokenizer
- Loss IS comparable across any condition (same vocabulary size within tokenizer)

### Rule 6: Winner Determination

For each tokenizer condition (atom, SPE-256, SPE-512), determine the winner by majority across:
1. Primary metrics that show a statistically meaningful difference (across seeds)
2. Consistency across seeds (both magnitude and direction)
3. Supporting evidence from secondary metrics

A model wins the condition if it leads on 3+ of the 5 primary metrics consistently across seeds.

---

## Experiment Matrix

| Tokenizer | Seeds | E0 | E1 | Total Runs |
|-----------|-------|----|----|------------|
| atom | 42 | Existing | Existing | 2 |
| atom | 43 | Optional (rerun if needed) | Optional (rerun if needed) | 0-2 |
| atom | 44 | Optional (rerun if needed) | Optional (rerun if needed) | 0-2 |
| SPE-256 | 42 | Existing | Existing | 2 |
| SPE-256 | 43 | New | New | 2 |
| SPE-256 | 44 | New | New | 2 |
| SPE-512 | 42 | New | New | 2 |
| SPE-512 | 43 | New | New | 2 |
| SPE-512 | 44 | New | New | 2 |

**Total runs:** 12-16 (depending on atom seed coverage)

### Run Naming Convention

```
{tokenizer}_{model}_seed{seed}
```

Examples:
- `atom_concat_seed42`
- `atom_intra_cross_seed42`
- `spe256_concat_seed42`
- `spe256_intra_cross_seed43`
- `spe512_concat_seed42`
- `spe512_intra_cross_seed44`

---

## Evaluation Pipeline (Per Run)

Each of the 12-16 runs follows this exact evaluation pipeline. The same pipeline scripts are used for both E0 and E1.

### Step 1: Evaluate Validation Split

```
python scripts/evaluate_smiles_model.py \
    --run-dir runs/{run_name} \
    --split valid \
    --out-dir runs/{run_name}/eval_valid/ \
    --metrics all
```

Outputs:
- `runs/{run_name}/eval_valid/predictions.csv` -- SMILES predictions with targets
- `runs/{run_name}/eval_valid/summary.json` -- aggregated metrics

### Step 2: Evaluate Test Split

```
python scripts/evaluate_smiles_model.py \
    --run-dir runs/{run_name} \
    --split test \
    --out-dir runs/{run_name}/eval_test/ \
    --metrics all
```

Outputs:
- `runs/{run_name}/eval_test/predictions.csv` -- SMILES predictions with targets
- `runs/{run_name}/eval_test/summary.json` -- aggregated metrics

### Step 3: Generation Behavior Audit

Run generation diagnostics on the test split to check for systematic artifacts.

```
python scripts/audit_generation.py \
    --run-dir runs/{run_name} \
    --split test \
    --out-dir runs/{run_name}/audit/
```

Checks:
- Repetition rate (fraction of predictions with repeated n-grams)
- Length distribution vs target distribution
- Frequency of common substructures (aromatic rings, carbonyls, etc.)
- Mode collapse detection (entropy of generated distribution)
- Invalid SMILES error categorization

### Step 4: Condition-Shuffle Test

Evaluate the model under shuffled spectral conditions to measure conditional dependence on each modality.

```
python scripts/condition_shuffle_test.py \
    --run-dir runs/{run_name} \
    --split test \
    --out-dir runs/{run_name}/shuffle/
```

Shuffle variants:
- Shuffle IR only (keep NMR intact)
- Shuffle 1H NMR only (keep IR and 13C intact)
- Shuffle 13C NMR only (keep IR and 1H intact)
- Shuffle all three modalities

For each variant, compute:
- Drop in canonical exact match vs normal conditions
- Drop in RDKit validity vs normal conditions
- Drop in average Tanimoto similarity vs normal conditions
- Drop in unique ratio vs normal conditions

---

## Deliverables

| File | Purpose |
|------|---------|
| `reports/final_e0_e1_validation_plan.md` | This file -- the experimental design and protocol |
| `reports/final_e0_e1_comparison_tables.md` | Consolidated results tables across all runs |
| `reports/final_e0_vs_e1_smiles_generation_report.md` | Full narrative report with analysis and conclusions |
| `reports/final_experiment_verification_audit.md` | Audit of experimental integrity (data, code, model, training) |
| All scripts and configs | Committed to git for reproducibility |

---

## Implementation Checklist

- [ ] SPE-512 tokenizer trained on training split only
- [ ] SPE-512 vocabulary verified (no unknown tokens on valid/test splits)
- [ ] SPE-512 E0/E1 parameter counts compared (< 1% difference)
- [ ] All 6 SPE-512 runs submitted (E0/E1 x seeds 42, 43, 44)
- [ ] SPE-256 seeds 43, 44 runs submitted (E0/E1)
- [ ] atom seeds 43, 44 runs submitted if needed
- [ ] All evaluation pipeline steps run for each completed training
- [ ] Comparison tables generated across all runs
- [ ] Interpretation rules applied to determine winner per tokenizer condition
- [ ] Audit completed and documented
- [ ] Final report written
