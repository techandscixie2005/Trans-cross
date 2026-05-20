# Trans-cross Project Status

**Last updated:** 2026-05-20

## Current Phase

**Preprocessing pipeline implemented** (COMPLETED)

## Completed

### Phase 1: Data Investigation
- [x] Remote data inspection (SSH to bjhpc)
- [x] IR file analysis (20,096 records, dense spectra, variable x-grids)
- [x] NMR file analysis (3,369,170 records, peak lists, no intensities)
- [x] IR–NMR pairing analysis (4,567 overlapping SMILES)
- [x] Investigation report (`reports/data_investigation.md`)
- [x] Example cases extracted (2 representative molecules)

### Phase 2: Preprocessing Code
- [x] `configs/preprocessing.yaml` — grid parameters, split config
- [x] `src/transcross/io.py` — JSONL streaming, YAML reader
- [x] `src/transcross/smiles.py` — RDKit SMILES canonicalization + scaffolds
- [x] `src/transcross/pairing.py` — IR/NMR catalog scanning + SMILES pairing
- [x] `src/transcross/spectra.py` — IR resampling, NMR binning, normalization
- [x] `src/transcross/splitting.py` — Scaffold/random train/valid/test split
- [x] `src/transcross/dataset.py` — PyTorch Dataset for processed arrays
- [x] `scripts/build_pairs.py` — CLI for IR–NMR pairing
- [x] `scripts/preprocess_spectra.py` — CLI for spectrum resampling/binning
- [x] `scripts/split_data.py` — CLI for data splitting
- [x] `scripts/run_preprocessing.py` — Full pipeline orchestrator
- [x] `scripts/inspect_processed.py` — Output inspection tool
- [x] 30 unit tests, all passing (smiles, pairing, spectra)
- [x] README.md and PROJECT_STATUS.md updated
- [x] Code committed and pushed to GitHub

## Pending

- [ ] Server smoke test (requires SSH to bjhpc)
- [ ] Full preprocessing run on server
- [ ] Model implementation (next phase)

## Output Files

All processed data is git-ignored and stored on the server:

| File | Shape | Description |
|------|-------|-------------|
| `ir.npy` | (N, 1801) | Min-max normalized IR, 400–4000 cm⁻¹ at 2 cm⁻¹ |
| `nmr_1h.npy` | (N, 1500) | Binary-binned 1H NMR, 0–15 ppm at 0.01 ppm |
| `nmr_13c.npy` | (N, 2200) | Binary-binned 13C NMR, 0–220 ppm at 0.1 ppm |

## Next Implementation Step

After preprocessing validation on server:
1. Design the 2x2 ablation experiment matrix
2. Implement the two encoder variants (direct concat + cross-modal attention)
3. Set up the TranSpec experiment structure
