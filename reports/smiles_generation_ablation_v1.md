# SMILES Generation Ablation Experiment Report v1

## 1. Goal

Compare two encoder attention organizations for SMILES generation from triplet spectra (IR + ¹H NMR + ¹³C NMR):

- **E0 (DirectConcatSmilesTransformer)**: All spectral tokens concatenated + joint self-attention
- **E1 (IntraCrossSmilesTransformer)**: Intra-modal self-attention → cross-modal attention → fusion

The only variable tested is attention organization. All other factors (model size, initialization, optimizer, data, decoder) are held constant.

## 2. Data

- **Source**: 4563 IR-NMR pairs from NIST IR and experimental NMR
- **IR**: 1801-dimensional vector
- **¹H NMR**: 1501-dimensional vector
- **¹³C NMR**: 2201-dimensional vector
- **Target**: Canonical SMILES strings
- **Split**: train/valid/test from `splits.json`

### SMILES Vocabulary

| Metric | Value |
|--------|-------|
| Vocab size | 49 |
| Max token length | 64 |
| Mean token length | 18.6 |
| P90 token length | 27 |
| P95 token length | 31 |
| `<unk>` tokens | 0 (0.0%) |

## 3. Model E0: DirectConcatSmilesTransformer

```
IR → [PatchTokenizer1D] → IR tokens (with modality + position embed)
¹H → [PatchTokenizer1D] → ¹H tokens (with modality + position embed)
¹³C → [PatchTokenizer1D] → ¹³C tokens (with modality + position embed)

Concatenate: [CLS] + IR_tokens + ¹H_tokens + ¹³C_tokens

→ N layers TransformerBlockPreLN (self-attention)

Encoder memory → TransformerSmilesDecoder → SMILES logits
```

**Key characteristics**:
- All tokens attend to each other from the first layer
- Modality information is only present in the modality embeddings
- Standard Transformer self-attention over concatenated tokens

## 4. Model E1: IntraCrossSmilesTransformer

```
Stage 1 — Intra-modal self-attention:
  IR tokens → N layers TransformerBlockPreLN (self-attention, IR only)
  ¹H tokens → N layers TransformerBlockPreLN (self-attention, ¹H only)
  ¹³C tokens → N layers TransformerBlockPreLN (self-attention, ¹³C only)

Stage 2 — Cross-modal attention:
  IR attends to (¹H + ¹³C) via CrossAttentionBlockPreLN
  ¹H attends to (IR + ¹³C) via CrossAttentionBlockPreLN
  ¹³C attends to (IR + ¹H) via CrossAttentionBlockPreLN
  (cross-attention out_proj is zero-initialized by default)

Stage 3 — Fusion:
  Concatenate: [CLS] + IR_tokens + ¹H_tokens + ¹³C_tokens
  → optional fusion self-attention blocks

Encoder memory → TransformerSmilesDecoder → SMILES logits
```

**Key characteristics**:
- Modalities first develop independent representations
- Cross-attention with zero-init starts as near-no-op and learns gradually
- More structured information exchange than direct concatenation

## 5. Attention Bias Audit

**No attention bias of any kind was used.**

| Bias Type | Present? |
|-----------|----------|
| Coordinate bias | No |
| Spectral x-axis bias | No |
| Modality-pair bias | No |
| Learned additive attention bias | No |
| Relative position bias | No |
| Graphormer-style bias | No |
| Causal mask (decoder only) | Allowed (required for autoregressive) |

The only allowed embeddings:
- Modality embeddings (learnable per-modality vector)
- Learnable absolute positional embeddings for patch tokens
- SMILES token embeddings
- Decoder positional embeddings

## 6. Initialization Scheme

| Component | Weight Init | Bias Init |
|-----------|------------|-----------|
| Q/K/V projections | Normal(0, 0.02) | 0 |
| Out projections (standard) | Normal(0, 0.02) | 0 |
| Out projections (E1 cross-attention) | 0 (zero-init) | 0 |
| Feed-forward | Normal(0, 0.02) | 0 |
| Token/tokenizer/position embeddings | TruncNormal(0, 0.02) | N/A |
| SMILES token embeddings | Normal(0, 0.02) | N/A |
| LayerNorm | γ=1, β=0 | N/A |

## 7. Parameter Counts

| Model | d_model=64 (1L) | d_model=128 (2L) |
|-------|----------------|------------------|
| E0 (DirectConcat) | 151,921 | ~TBD |
| E1 (IntraCross) | 452,209 | ~TBD |

*(d_model=128 counts TBD after ablation runs)*

## 8. Training Commands

### Smoke Test (1 epoch, d_model=64)

```bash
# E0 Concat smoke
python scripts/train_smiles_ablation.py \
  --processed-dir /path/to/data/processed \
  --model concat --epochs 1 --batch-size 16 \
  --d-model 64 --encoder-layers 1 --decoder-layers 1 \
  --num-heads 4 --patch-size 64 --lr 1e-4 --seed 42 \
  --max-smiles-len 160 --out-dir runs/smiles_smoke_concat

# E1 Intra-cross smoke
python scripts/train_smiles_ablation.py \
  --processed-dir /path/to/data/processed \
  --model intra_cross --epochs 1 --batch-size 16 \
  --d-model 64 --encoder-layers 1 --decoder-layers 1 \
  --num-heads 4 --patch-size 64 --lr 1e-4 --seed 42 \
  --max-smiles-len 160 --out-dir runs/smiles_smoke_intra_cross
```

### Ablation Run (30 epochs, d_model=128)

```bash
# E0 Concat
python scripts/train_smiles_ablation.py \
  --processed-dir /path/to/data/processed \
  --model concat --epochs 30 --batch-size 32 \
  --d-model 128 --encoder-layers 2 --decoder-layers 2 \
  --num-heads 4 --patch-size 64 --lr 1e-4 --seed 42 \
  --max-smiles-len 160 --out-dir runs/smiles_concat_seed42

# E1 Intra-cross
python scripts/train_smiles_ablation.py \
  --processed-dir /path/to/data/processed \
  --model intra_cross --epochs 30 --batch-size 32 \
  --d-model 128 --encoder-layers 2 --decoder-layers 2 \
  --num-heads 4 --patch-size 64 --lr 1e-4 --seed 42 \
  --max-smiles-len 160 --out-dir runs/smiles_intra_cross_seed42
```

## 9. Smoke Test Results

### Local tests (pytest)

- **Local**: 80/80 passed in 1.51s
- **Server**: 80/80 passed in 7.66s

### Code smoke verification (CPU forward pass + loss)

| Check | Concat | IntraCross |
|-------|--------|------------|
| Forward pass shape | (4, 10, 49) OK | (4, 10, 49) OK |
| No NaN | True | True |
| Loss finite | 3.8862 (~ln(49)) | 3.9317 (~ln(49)) |

Both models produce valid forward passes and computable losses.
The initial losses (~3.89–3.93) are close to ln(49) ≈ 3.89, which is the
expected value for random prediction over 49 tokens.

### GPU availability

The server login node has no GPU. Slurm GPU partition is available
with running jobs on nodes g0013, g0015, g0045, g0057.
Training requires `sbatch` submission to the GPU partition.

## 10. Small Ablation Results

Training commands via Slurm:

```bash
# Submit concat ablation
sbatch scripts/slurm_concat.sh

# Submit intra_cross ablation
sbatch scripts/slurm_intra_cross.sh
```

*(Results TBD — Slurm jobs pending submission by user)*

## 11. Validation/Test Metrics

*(To be populated)*

## 12. Example Predictions

*(To be populated)*

## 13. Failure Cases

*(To be populated)*

## 14. E1 vs E0 Comparison

*(TBD — pending Slurm GPU training runs)*

## 15. Next Recommendation

1. Submit Slurm jobs for both models on the GPU partition
2. After training completes, run `scripts/evaluate_smiles_model.py` on both checkpoints
3. Compare results across both seeds (42, 43, 44) for statistical confidence
4. If E1 outperforms E0, the intra-modal + cross-modal architecture is validated
5. If E0 matches E1, direct concatenation with joint attention is sufficient

---

**Git commit**: fcedc3a
**GitHub**: https://github.com/techandscixie2005/Trans-cross
**Date**: 2026-05-21
**Server**: bjhpc (Beijing ParaCloud HPC)
**Environment**: Miniforge3-24.11, Python 3.9.0, conda env: transpec
**GPU partition**: gpu (nodes g0013-g0057)
