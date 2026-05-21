# Equal-Parameter SMILES Generation Ablation

**Date:** 2026-05-21
**Repository:** https://github.com/techandscixie2005/Trans-cross
**Reference:** https://github.com/techandscixie2005/TranSpec (inspected, NOT modified)

## 1. Goal

Compare two encoder attention topologies for SMILES generation from IR + 1H NMR + 13C NMR spectra under **strictly matched parameter budgets**.

- **E0 (DirectConcat):** All spectral tokens are directly concatenated and processed by a standard Transformer encoder with self-attention over all tokens.
- **E1 (IntraCross):** Each modality first undergoes intra-modal self-attention, followed by cross-modal attention between modalities. The fused representation is passed to the decoder.

## 2. Reference TranSpec Inspection

### Files Inspected

| File | Purpose |
|------|---------|
| `src/model.py` | Core `Model` class: standard Transformer encoder-decoder with optional CNN/MLP spectral frontends |
| `src/util.py` | `PositionalEncoding` (sinusoidal), data utilities |
| `src/train.py` | Training loop with teacher forcing |
| `src/evaluate.py` | Evaluation metrics |
| `src/search_methods.py` | Beam search and decoding methods |
| `src/scripts/SpecGNN.py` | GNN-based spectrum encoding (separate experiment) |
| `README.md` | Project overview |

### Key Observations

1. **Direct Concat Baseline:** TranSpec's `Model` class uses PyTorch's built-in `nn.TransformerEncoder` over concatenated spectral features. This is the direct-concat baseline we replicate conceptually as E0.
2. **No Intra-Cross:** TranSpec does NOT implement intra-modal or cross-modal attention — the architecture is purely standard Transformer encoder-decoder.
3. **Spectral Frontends:** TranSpec supports optional CNN/MLP spectral preprocessing before the Transformer encoder. Our implementation uses patch tokenization instead, which is a more modern approach.
4. **Reused Conceptually:** The encoder-decoder structure, teacher-forcing training, and autoregressive SMILES decoding patterns are conceptually reused in Trans-cross.
5. **Reimplemented:** All model code in Trans-cross is a clean reimplementation with custom bias-free attention modules, patch tokenization, and the novel IntraCross architecture.

### Statement of Non-Modification

**The TranSpec repository was NOT modified.** All new code is in the Trans-cross repository. TranSpec was read-only inspected under `/tmp/TranSpec_ref/`.

## 3. Experimental Controls

| Control | Status |
|---------|--------|
| Same decoder (architecture + hyperparams) | Enforced |
| Same tokenizer (SMILES vocabulary) | Enforced |
| Same spectral patch tokenization | Enforced |
| Same training/validation/test split | Enforced |
| Same optimizer (AdamW, lr=1e-4) | Enforced |
| Same batch size (32) | Enforced |
| Same seed (42) | Enforced |
| Same epochs (30) | Enforced |
| No attention bias of any kind | Enforced, audited |
| Matched parameter count | Enforced (within 1%) |

## 4. Model Definitions

### E0: DirectConcat Encoder

```
IR ──► [PatchTokenizer + ModalityEmbedding] ──┐
1H ──► [PatchTokenizer + ModalityEmbedding] ──┼──► [CLS + Concat] ──► [6× Self-Attention Blocks] ──► encoder_memory
13C ──► [PatchTokenizer + ModalityEmbedding] ──┘

encoder_memory ──► [2× Decoder Layers] ──► SMILES logits
                       ▲
              SMILES tokens (BOS + teacher forcing)
```

- 6 standard Pre-LN self-attention blocks (`TransformerBlockPreLN`)
- d_model=128, num_heads=4, FFN dim=512

### E1: IntraCross Encoder

```
IR ──► [PatchTokenizer + ModalityEmbedding] ──► [1× Intra Self-Attn] ──┐
1H ──► [PatchTokenizer + ModalityEmbedding] ──► [1× Intra Self-Attn] ──┤
13C ──► [PatchTokenizer + ModalityEmbedding] ──► [1× Intra Self-Attn] ──┘
                                                                         │
                    ┌────────────────────────────────────────────────────┘
                    ▼
            [Cross-Attention: IR attends to 1H+13C, etc.]
                    │
                    ▼
            [CLS + Concat] ──► encoder_memory

encoder_memory ──► [2× Decoder Layers] ──► SMILES logits
```

- 1 intra-modal self-attention block per modality (3 total)
- 1 cross-modal attention block per modality (3 total)
- No fusion layers
- Cross-attention output projections initialized to zero
- d_model=128, num_heads=4, FFN dim=512

## 5. Parameter Matching

### Strategy

E1 has 3× intra-modal self-attention blocks + 3× cross-modal attention blocks, creating extra parameters. To match E0, we increase the number of direct-concat encoder layers from 2 to 6.

- E0 receives 6 self-attention layers (instead of 2) to match E1's parameter budget
- E1 keeps a minimal design: 1 intra + 1 cross layer per modality, 0 fusion
- The decoder is byte-identical in both models

### Parameter Counts

| Component | E0 DirectConcat | E1 IntraCross |
|---|---|---:|
| Spectral tokenizers (3×) | 36,096 | 36,096 |
| Modality embeddings (3×) | 384 | 384 |
| CLS token | 128 | 128 |
| Encoder blocks | 1,189,632 | 1,190,400 |
| Decoder (2-layer) | 563,250 | 563,250 |
| **Total** | **1,789,490** | **1,790,258** |

| Metric | Value |
|--------|-------|
| Absolute difference | 768 |
| Relative difference | 0.043% |
| Within tolerance (≤1%) | Yes |

### Verification

```
python scripts/compare_model_params.py \
    --processed-dir /path/to/processed \
    --vocab /path/to/smiles_vocab.json \
    --config configs/smiles_equal_param.yaml
```

## 6. Attention Bias Audit

**Status: PASS** — No forbidden attention bias mechanisms detected.

### Explicitly Absent

- [x] No coordinate bias (spectral x-axis position bias)
- [x] No modality-pair bias (learned pairwise modality bias)
- [x] No relative position bias (Transformer-XL style)
- [x] No Graphormer-style spatial/distance bias
- [x] No learned additive attention-logit bias in any attention module

### Allowed Mechanisms

| Mechanism | Location | Purpose |
|-----------|----------|---------|
| Causal mask | `decoder._build_causal_mask()` | Autoregressive SMILES decoding |
| Padding mask | `_encode_spectra()` | Batching correctness |
| Padding mask | `decoder.forward()` | Decoder input padding |
| Absolute position embeddings | `PatchTokenizer1D.pos_embed` | Token position encoding |
| Modality embeddings | `ModalityEmbedding` | Modality type identification |

## 7. Local Tests

```
$ pytest tests/test_equal_param_models.py tests/test_param_counting.py tests/test_model_factory.py tests/test_attention_no_bias.py -q
................................
32 passed in 1.79s
```

| Test | Status |
|------|--------|
| E0/E1 instantiation from config | PASS |
| E0/E1 forward pass | PASS |
| Decoder parameter identity | PASS |
| Parameter difference ≤ 1% | PASS (0.043%) |
| No attention bias params | PASS |
| Causal mask exists and correct | PASS |
| Padding ignore_index in loss | PASS |
| Greedy generation | PASS |

## 8. Server Run Trace

### Environment

- **Server:** bjhpc (Beijing ParaCloud HPC)
- **Code path:** `/data/home/sczc698/run/xxy/Trans-cross/code/`
- **Data path:** `/data/home/sczc698/run/xxy/Trans-cross/data/processed/`
- **Environment:** `module load miniforge3/24.11 && source activate transpec`

### Git Commit

- **Commit:** `(to be filled after push)`
- **Branch:** `master`

### Server Verification Commands

```bash
ssh bjhpc 'cd /data/home/sczc698/run/xxy/Trans-cross/code/ && git pull'
ssh bjhpc 'cd /data/home/sczc698/run/xxy/Trans-cross/code/ && module load miniforge3/24.11 && source activate transpec && pytest -q'

# Parameter comparison
ssh bjhpc 'cd /data/home/sczc698/run/xxy/Trans-cross/code/ && module load miniforge3/24.11 && source activate transpec && python scripts/compare_model_params.py --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed --vocab /data/home/sczc698/run/xxy/Trans-cross/data/processed/smiles_vocab.json --config configs/smiles_equal_param.yaml'

# Attention bias audit
ssh bjhpc 'cd /data/home/sczc698/run/xxy/Trans-cross/code/ && module load miniforge3/24.11 && source activate transpec && python scripts/audit_attention_bias.py --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed --config configs/smiles_equal_param.yaml'
```

### Slurm Job Submission

```bash
ssh bjhpc 'cd /data/home/sczc698/run/xxy/Trans-cross/code/ && sbatch scripts/slurm_equal_concat.sh'
ssh bjhpc 'cd /data/home/sczc698/run/xxy/Trans-cross/code/ && sbatch scripts/slurm_equal_intra_cross.sh'
```

### Job IDs

- **E0 (concat_equal):** `(to be filled)`
- **E1 (intra_cross_equal):** `(to be filled)`

## 9. Results

### Training Metrics

| Metric | E0 DirectConcat | E1 IntraCross | Winner |
|---|---|---|---|
| Valid loss | (to be filled) | (to be filled) | |
| Test loss | (to be filled) | (to be filled) | |
| Valid token acc | (to be filled) | (to be filled) | |
| Test token acc | (to be filled) | (to be filled) | |
| Valid exact match | (to be filled) | (to be filled) | |
| Test exact match | (to be filled) | (to be filled) | |
| Valid canonical exact match | (to be filled) | (to be filled) | |
| Test canonical exact match | (to be filled) | (to be filled) | |
| Valid RDKit validity | (to be filled) | (to be filled) | |
| Test RDKit validity | (to be filled) | (to be filled) | |

### Prediction Examples

| Split | Target | E0 Prediction | E1 Prediction | Notes |
|-------|--------|---------------|---------------|-------|
| (to be filled) | | | | |

### Failure Cases

(to be filled)

## 10. Conclusion

(to be filled after training completes)

---

## File Manifest

| File | Purpose |
|------|---------|
| `configs/smiles_equal_param.yaml` | Equal-parameter ablation configuration |
| `src/transcross/model_utils.py` | Parameter counting and comparison utilities |
| `src/transcross/models/factory.py` | Model factory for config-driven instantiation |
| `src/transcross/models/smiles_concat.py` | E0: DirectConcatSmilesModel (updated) |
| `src/transcross/models/smiles_intra_cross.py` | E1: IntraCrossSmilesModel (updated) |
| `scripts/compare_model_params.py` | CLI for parameter comparison |
| `scripts/audit_attention_bias.py` | CLI for attention bias audit |
| `scripts/train_smiles_ablation.py` | Training script (updated with --config mode) |
| `scripts/evaluate_smiles_model.py` | Evaluation script (updated with --run-dir mode) |
| `scripts/slurm_equal_concat.sh` | Slurm script for E0 training |
| `scripts/slurm_equal_intra_cross.sh` | Slurm script for E1 training |
| `tests/test_equal_param_models.py` | Tests for equal-parameter models |
| `tests/test_param_counting.py` | Tests for parameter counting |
| `tests/test_model_factory.py` | Tests for model factory |
| `tests/test_attention_no_bias.py` | Tests for attention bias audit |
| `reports/equal_param_smiles_ablation.md` | This report |
