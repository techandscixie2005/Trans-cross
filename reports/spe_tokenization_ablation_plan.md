# SPE Decoder Tokenization Ablation — Experiment Plan

## Name
SPE Decoder Tokenization Ablation

## Purpose
Test whether atom-level + SPE (SMILES Pair Encoding) tokenization improves
SMILES generation quality, while preserving the equal-parameter E0 vs E1
encoder topology comparison.

## Scientific Motivation
SPE starts from atom-level SMILES tokens and merges frequent token pairs
into chemically meaningful SMILES substrings. This can:
- Shorten decoder output sequences (fewer autoregressive steps)
- Enable fragment-level generation (merged tokens represent chemical substructures)
- Potentially improve RDKit validity and generation stability
- Test whether decoder tokenization quality, not encoder attention bias,
  is the bottleneck for SMILES generation

## Controlled Variables (Unchanged)
- Same processed IR/NMR arrays (ir.npy, nmr_1h.npy, nmr_13c.npy)
- Same data split (train/valid/test from splits.json)
- Same E0 encoder definition: DirectConcat with 6 self-attention layers
- Same E1 encoder definition: IntraCross with 1 intra + 1 cross layer
- Same decoder architecture: TransformerSmilesDecoder (2 layers, d_model=128)
- Same decoder hyperparameters (d_ff=512, num_heads=4, dropout=0.1)
- Same training setup (epochs, batch_size, lr, seed=42)
- No attention bias (no coordinate, modality-pair, relative, or learned bias)
- Equal-parameter constraint: E0/E1 within 1%

## Changed Variable
**Decoder-side SMILES tokenization only:**
- Old: regex-based atom-level tokenizer (SmilesTokenizer, vocab ~30-40 tokens)
- New: atom-level + SPE subword tokenizer (SPETokenizer, vocab 256 tokens)

## Expected Benefits
1. **Shorter sequences:** SPE merges frequent pairs, reducing token count
2. **Higher validity:** Fragment-level generation may produce more valid SMILES
3. **Better exact/canonical match:** Shorter sequences are easier to generate correctly

## Failure Risks
1. SPE may not improve validity if the bottleneck is encoder representation,
   not decoder tokenization
2. Exact match may remain zero if the task is inherently too difficult
3. SPE vocabulary may contain unknown tokens if test SMILES have rare atoms
4. Token accuracy comparison across tokenizers is not meaningful

## Metrics
- Test loss (cross-entropy, comparable across tokenizers)
- Token accuracy (NOT comparable across tokenizers)
- Exact string match
- Canonical exact match (via RDKit)
- RDKit validity
- Average generated character length
- Average generated SPE token length
- Length reduction ratio (atom vs SPE)

## E0/E1 Definitions (Unchanged)
- **E0 (DirectConcat):** IR + 1H NMR + 13C NMR tokens concatenated,
  processed by 6-layer self-attention encoder
- **E1 (IntraCross):** Each modality first undergoes 1-layer intra-modal
  self-attention, then cross-modal attention, then concatenation

## Constraints
- Only decoder tokenization changes
- No attention bias of any kind
- Equal-parameter constraint (max 1% relative diff)
- E0/E1 encoder definitions unchanged
- SPE trained ONLY on training split (no valid/test leakage)

## Comparison
| tokenizer | model | test loss | token acc | exact | canonical exact | validity | avg char len |
|---|---|---|---|---|---|---|---|
| regex_atom | E0 concat | (old) | (old) | (old) | (old) | (old) | (old) |
| regex_atom | E1 intra_cross | (old) | (old) | (old) | (old) | (old) | (old) |
| spe | E0 concat | TBD | TBD | TBD | TBD | TBD | TBD |
| spe | E1 intra_cross | TBD | TBD | TBD | TBD | TBD | TBD |
