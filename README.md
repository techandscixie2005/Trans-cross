# Trans-cross

Cross-modal spectral representation learning with IR and NMR.

## Current Stage

**Preprocessing pipeline implemented** (2026-05-20).

The preprocessing pipeline converts raw IR and NMR JSONL files into paired, resampled, model-ready arrays. No model training code yet.

## Remote Data

Raw data are stored on the Beijing ParaCloud HPC server:

- **Remote path:** `/data/home/sczc698/run/xxy/Trans-cross/`
- **SSH alias:** `bjhpc`
- **IR file:** `IR_NIST.jsonl` (1.4 GB, 20,096 records)
- **NMR file:** `NMR_exp2.jsonl` (824 MB, 3,369,170 records)

Full raw data are **not** committed to this repository. Only tiny example cases are stored locally under `examples/`.

## Local Structure

```
Trans-cross/
  README.md
  PROJECT_STATUS.md
  .gitignore
  configs/
    preprocessing.yaml         # Grid parameters, split config
  scripts/
    build_pairs.py             # SMILES canonicalization + pairing
    preprocess_spectra.py      # IR resampling + NMR binning
    split_data.py              # Train/valid/test split
    run_preprocessing.py       # Full pipeline runner
    inspect_processed.py       # Output inspection tool
  src/transcross/
    __init__.py
    io.py                      # JSONL streaming, YAML reader
    smiles.py                  # RDKit SMILES canonicalization
    pairing.py                 # IR–NMR catalog scanning + pairing
    spectra.py                 # IR interpolation, NMR binning
    splitting.py               # Scaffold-based splitting
    dataset.py                 # PyTorch Dataset
  tests/
    test_smiles.py
    test_pairing.py
    test_spectra.py
  reports/
    data_investigation.md      # Full data investigation report
  examples/
    case_001/                  # Representative paired IR+NMR molecule
    case_002/                  # Second representative example
  data/                        # Git-ignored
    raw/                       # Local copies of raw data (not committed)
    processed/                 # Processed paired datasets (not committed)
```

## Preprocessing Pipeline

### Smoke Test (small subset)

```bash
python scripts/run_preprocessing.py \
  --config configs/preprocessing.yaml \
  --raw-dir /data/home/sczc698/run/xxy/Trans-cross/ \
  --out-dir data/processed_smoke \
  --smoke \
  --limit-ir 200 \
  --limit-nmr 500000 \
  --limit-preprocess 100
```

### Full Run

```bash
python scripts/run_preprocessing.py \
  --config configs/preprocessing.yaml \
  --raw-dir /data/home/sczc698/run/xxy/Trans-cross/ \
  --out-dir data/processed
```

### Inspect Output

```bash
python scripts/inspect_processed.py --processed-dir data/processed
```

### Expected Outputs

| File | Description |
|------|-------------|
| `pairs.csv` | Paired molecule metadata |
| `paired_records.jsonl` | Compact intermediate with full arrays |
| `pairing_summary.json` | Pairing statistics |
| `ir.npy` | (N, 1801) resampled IR intensity vectors |
| `ir_x.npy` | (1801,) shared IR wavenumber grid |
| `nmr_1h.npy` | (N, 1500) binned 1H NMR |
| `nmr_1h_x.npy` | (1500,) shared 1H grid |
| `nmr_13c.npy` | (N, 2200) binned 13C NMR |
| `nmr_13c_x.npy` | (2200,) shared 13C grid |
| `canonical_smiles.txt` | One SMILES per line |
| `splits.json` | Train/valid/test indices |
| `split_summary.json` | Split method and counts |
| `preprocess_summary.json` | Preprocessing parameters |

**Warning:** Raw data, processed `.npy` arrays, and `data/` directories are NOT committed to git.

## Planned Ablation

The first ablation will compare attention mechanisms for IR + NMR tokens:

1. **Direct concatenation attention** — all IR and NMR tokens attend to each other.
2. **Intra-modal + cross-modal attention** — tokens first attend within their own modality, then across modalities.

## See Also

- `reports/data_investigation.md` — Full data analysis and preprocessing plan
- `/transpec-ablation-project` — Skill for creating TranSpec ablation experiments
- `/bjhpc-safe-runner` — Skill for remote HPC operations
