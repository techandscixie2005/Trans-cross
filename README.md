# Trans-cross

Cross-modal spectral representation learning with IR and NMR.

## Current Stage

**Data investigation and preprocessing design** (2026-05-20).

We are analyzing raw IR and NMR spectra to design a reliable preprocessing pipeline before implementing a modified TranSpec encoder for multimodal spectra.

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
  reports/
    data_investigation.md      # Full data investigation report
  examples/
    case_001/                  # Representative paired IR+NMR molecule
      metadata.json
      ir.csv
      nmr.csv
      ir_preview.png
      nmr_preview.png
    case_002/                  # Second representative example
      metadata.json
      ir.csv
      nmr.csv
      ir_preview.png
      nmr_preview.png
  data/                        # Git-ignored
    raw/                       # Local copies of raw data (not committed)
    processed/                 # Processed paired datasets (not committed)
```

## Planned Ablation

The first ablation will compare attention mechanisms for IR + NMR tokens:

1. **Direct concatenation attention** — all IR and NMR tokens attend to each other.
2. **Intra-modal + cross-modal attention** — tokens first attend within their own modality, then across modalities.

## See Also

- `reports/data_investigation.md` — Full data analysis and preprocessing plan
- `/transpec-ablation-project` — Skill for creating TranSpec ablation experiments
- `/bjhpc-safe-runner` — Skill for remote HPC operations
