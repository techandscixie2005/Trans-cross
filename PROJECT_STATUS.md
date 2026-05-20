# Trans-cross Project Status

**Last updated:** 2026-05-20

## Current Phase

**Data investigation and preprocessing design** (COMPLETED)

## Completed

- [x] Remote data inspection (SSH to bjhpc)
- [x] IR file analysis (`IR_NIST.jsonl` — 20,096 records, dense spectra, variable x-grids)
- [x] NMR file analysis (`NMR_exp2.jsonl` — 3,369,170 records, peak lists, no intensities)
- [x] IR–NMR pairing analysis (4,567 overlapping SMILES; 36.7% of IR records paired)
- [x] Preprocessing pipeline design (SMILES canonicalization, IR resampling, NMR binning)
- [x] Example cases extracted (2 representative molecules with IR + NMR)
- [x] Preview plots generated (IR and NMR for both cases)
- [x] Investigation report written (`reports/data_investigation.md`)
- [x] Project repository initialized with README, .gitignore, PROJECT_STATUS.md

## Files Extracted Locally

- `examples/case_001/` — `CCCCC(CC)CI` (iodoalkane), paired IR + ¹H NMR
- `examples/case_002/` — `O=Cc1c(F)c(F)c(F)c(F)c1F` (pentafluorobenzaldehyde), paired IR + ¹H/¹³C/¹⁹F NMR

## Preprocessing Recommendation

1. Canonicalize SMILES with RDKit for both IR and NMR
2. Deduplicate IR: keep highest-resolution record per canonical SMILES
3. Select best NMR: keep record with most peaks per (SMILES, nucleus)
4. Pair on canonical SMILES (inner join)
5. Resample IR to fixed 400–4000 cm⁻¹ grid at 2 cm⁻¹ spacing (1,801 points)
6. Bin NMR peaks to fixed grids (1H: 0–15 ppm, 13C: 0–220 ppm)
7. Scaffold-based train/valid/test split (70/15/15)
8. Save as .npy arrays + pairs.csv + splits.json

## Next Implementation Step

Write preprocessing scripts:
- `scripts/build_pairs.py`
- `scripts/preprocess_spectra.py`
- `scripts/split_data.py`
- `src/data/transcross_dataset.py`
- `configs/preprocessing.yaml`

## Known Issues

- Only ~4,567 molecules can be paired (33% of IR)
- NMR lacks intensity values (peak positions only)
- IR has 3,003 unique x-grid lengths (resampling required)
- Non-canonical SMILES in raw data (RDKit canonicalization needed)
