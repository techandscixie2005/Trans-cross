# Trans-cross Data Investigation Report

## 1. Goal

This stage investigates the raw IR and NMR data files on the remote HPC server to understand their structure, format, and content. The goal is to design a reliable preprocessing pipeline that converts raw spectra into paired, model-ready samples for a later attention mechanism ablation:

- **E0 (Baseline):** Direct concatenation attention over all IR + NMR tokens.
- **E1 (Ablation):** Intra-modal attention followed by cross-modal attention.

No model training or implementation is performed at this stage.

## 2. Remote Files

| Role | Filename | Format | Size | Records | Notes |
|------|----------|--------|------|---------|-------|
| IR spectra | `IR_NIST.jsonl` | JSONL | 1.4 GB | 20,096 | NIST IR database, dense spectra |
| NMR spectra | `NMR_exp2.jsonl` | JSONL | 824 MB | 3,369,170 | Peak lists, multiple nuclei per molecule |

**Remote path:** `/data/home/sczc698/run/xxy/Trans-cross/`
**Total size:** 2.2 GB

## 3. IR File Analysis

### 3.1 Schema

Each line is a JSON object with these keys:

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `smiles` | string | SMILES string (non-canonical) | `NNc1ccc([N+](=O)[O-])cc1` |
| `value.x` | float[] | Wavenumber grid (cm⁻¹) | `[629.56, 629.83, ...]` |
| `value.y` | float[] | Transmittance (%) | `[56.21, 56.21, ...]` |
| `temperature` | string | "NONE" or temperature value | `NONE` |
| `pressure` | string | "NONE" or pressure value | `NONE` |
| `condition` | string | Phase condition | `liquid`, `gas`, `solid` |
| `type` | string | Always `IR` | `IR` |
| `source` | string | Data source | `NIST` |

### 3.2 Statistics

- **Total records:** 20,096
- **Unique SMILES:** 13,716
- **Duplicate SMILES:** 6,380 (same molecule, different conditions)
- **All records have `y` (intensity) values**
- **x and y always same length within a record**

### 3.3 Spectrum Representation

- **Type:** Dense intensity vectors on variable x-grids
- **x range:** ~400 to ~4835 cm⁻¹ (varies by record)
- **y range:** ~0 to 100 (percent transmittance, scaled)
- **x-length distribution (top 10):**

  | Length | Count |
  |--------|-------|
  | 870 | 2,748 |
  | 1,780 | 2,508 |
  | 815 | 2,116 |
  | 1,779 | 498 |
  | 898 | 302 |
  | 2,469 | 296 |
  | 929 | 263 |
  | 2,465 | 174 |
  | 3,370 | 145 |
  | 885 | 135 |

- **Unique x-lengths:** 3,003 distinct values (high variability)
- **Implication:** IR spectra are NOT on a shared x-grid. Resampling to a common grid is required for model input.

### 3.4 Molecule Identifiers

- **Primary key:** `smiles` (non-canonical, may vary in representation)
- **No molecule name, formula, or InChI** — SMILES is the only identifier
- **No explicit molecule ID field** — must derive from SMILES

### 3.5 Extraction Method

For each IR record:
1. Parse JSON line.
2. Extract `smiles` (canonicalize with RDKit for consistent pairing).
3. Extract `value.x` and `value.y` arrays.
4. Resample to a fixed wavenumber grid (e.g., 400–4000 cm⁻¹ at 2 cm⁻¹ spacing = 1,801 points).
5. Handle duplicate SMILES by either: (a) keeping the first record, (b) averaging spectra, or (c) keeping the record with the highest resolution.

### 3.6 Potential Issues

- **Variable x-grids:** Must resample to fixed grid; interpolation method (linear, cubic) matters.
- **Duplicate SMILES:** 6,380 duplicates need a deduplication rule.
- **Non-canonical SMILES:** Same molecule may appear with different SMILES strings; canonicalization is required.
- **Missing metadata:** No temperature/pressure for most records.

## 4. NMR File Analysis

### 4.1 Schema

Each line is a JSON object representing **one nucleus spectrum for one molecule** (not the full molecule):

| Key | Type | Description | Example |
|-----|------|-------------|---------|
| `smiles` | string | SMILES string | `CCOC(=O)/C(Cc1ccc2ccccc2c1)=C(/F)[Si](CC)(CC)CC` |
| `value.x` | float[] | Chemical shifts (ppm) | `[3.0, 7.7, 14.3, ...]` |
| `value.y` | string | Always `"NONE"` | `NONE` |
| `nucleus` | string | NMR nucleus type | `1H`, `13C`, `19F`, `31P`, `11B`, `29Si` |
| `frequency` | float | Spectrometer frequency (MHz) | `101.0`, `500.0` |
| `solvent` | string | NMR solvent | `CDCl3`, `DMSO-d6` |
| `type` | string | Always `NMR` | `NMR` |
| `source` | string | Data source (usually `NONE`) | `NONE` |

### 4.2 Statistics

- **Total records:** 3,369,170
- **Unique SMILES:** 1,497,913
- **Records per SMILES:** 1 to 40+ (typically 1–28; average ~2.25)
- **All records have `value.y = "NONE"`** — no intensity values, only peak positions

### 4.3 Nucleus Distribution

| Nucleus | Count | Percentage |
|---------|-------|------------|
| 1H | 1,667,139 | 49.5% |
| 13C | 1,455,670 | 43.2% |
| 19F | 193,855 | 5.8% |
| 31P | 33,229 | 1.0% |
| 11B | 16,891 | 0.5% |
| 29Si | 2,386 | 0.1% |

### 4.4 Spectrum Representation

- **Type:** Peak lists (sparse) — only chemical shifts, **no intensities**
- **Peaks per record:** 1 to 17 (most records have 1–17 peaks)
- **Typical 1H:** 1–10 peaks at 300–500 MHz
- **Typical 13C:** 1–20 peaks at 75–126 MHz
- **Multiple records per molecule:** A single molecule can have 1H, 13C, 19F, etc. from different sources

### 4.5 Molecule Identifiers

- **Primary key:** `smiles` (same as IR)
- **No molecule name, formula, or InChI**
- **Nucleus + frequency + solvent** identify the specific NMR experiment

### 4.6 Extraction Method

For each molecule, combine all its NMR records:
1. Group NMR records by `smiles`.
2. For each nucleus type (1H, 13C), select the best spectrum (most peaks, or lowest frequency for 1H, highest for 13C).
3. Represent each nucleus channel as:
   - **Option A:** Binned dense vector on a fixed chemical shift grid (e.g., 0–200 ppm at 0.1 ppm = 2,000 bins).
   - **Option B:** Token sequence of (shift, nucleus) pairs.
   - **Option C:** Peak list kept as variable-length sequence, padded/bucketed for batching.
4. Merge 1H and 13C peak lists into a multimodal NMR token sequence.

### 4.7 Potential Issues

- **No intensities:** Cannot reconstruct full spectrum shape; peak positions only.
- **Multiple records per SMILES:** Need a selection/aggregation rule.
- **Different frequencies/solvents:** Same molecule may have spectra from different experimental conditions.
- **1H vs 13C:** Different chemical shift ranges and peak densities.
- **Large file (3.37M records):** Full loading into memory may be expensive; streaming processing is recommended.

## 5. Pairing Analysis

### 5.1 Pairing by Shared SMILES

| Metric | Value |
|--------|-------|
| IR unique SMILES | 13,716 |
| NMR unique SMILES | 1,497,913 |
| Overlapping SMILES | **4,567** |
| IR records with NMR match | 7,385 / 20,096 (36.7%) |
| NMR molecules with IR match | 4,567 / 1,497,913 (0.3%) |

### 5.2 Pairing Strategies

| Strategy | Paired Samples | Unmatched IR | Unmatched NMR | Duplicates | Reliability |
|----------|---------------|--------------|---------------|------------|-------------|
| Canonical SMILES exact match | 4,567 | 9,149 | 1,493,346 | 6,380 IR dupes | **High** (SMILES is a structural identifier) |
| Order-based pairing | 20,096 (risky) | 0 | 3,349,074 | Unknown | **Very Low** (files clearly have different record counts) |
| SMILES + condition matching | ~4,567 | ~9,149 | ~1.49M | 6,380 | High, but doesn't increase pairing |

### 5.3 Recommended Pairing Rule

**Canonical SMILES exact match** after canonicalization with RDKit.

- Canonicalize all SMILES in both files.
- Match on canonical SMILES.
- For IR duplicates: keep the record with the highest resolution (most x-points).
- For NMR duplicates (same SMILES, same nucleus): keep the record with the most peaks.
- Pairing yield: ~4,567 unique molecule pairs (reduced from 20,096 IR records after dedup).

### 5.4 Risks

- **Low pairing rate:** Only 33% of unique IR SMILES have NMR data, and only 0.3% of NMR SMILES have IR data. The NMR dataset is much larger and broader but not IR-targeted.
- **Duplicate IR records:** 6,380 IR records share SMILES with another IR record. A dedup rule is needed.
- **Non-canonical SMILES:** The current SMILES strings are not canonical. RDKit canonicalization may collapse some that look different.

## 6. Proposed Preprocessing Method

### Pipeline

```
Raw IR_NIST.jsonl ──┐                    Raw NMR_exp2.jsonl ──┐
                     │                                          │
              Parse & extract                             Parse & group
              canonicalize SMILES                        canonicalize SMILES
                     │                                          │
              Deduplicate IR                               Select best per nucleus
              (keep highest res)                          (most peaks per nucleus)
                     │                                          │
                     └─────── Canonical SMILES join ────────────┘
                                      │
                              Filter: keep only paired
                                      │
                        ┌─────────────┴─────────────┐
                        │                           │
                  Resample IR                    Bin NMR peaks
             (400–4000 cm⁻¹, 2 cm⁻¹)       (0–200 ppm, 0.1 ppm bins)
                        │                           │
                  Normalize IR                Normalize NMR
                  (min-max → [0,1])          (binary or gaussian bins)
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                              Save processed pairs
                                      │
                              Train/Valid/Test split
                              (scaffold-based, 70/15/15)
```

### Step-by-Step

1. **Load raw files safely** — stream from JSONL, never load full files into memory at once.
2. **Extract molecule key and SMILES** — parse each line, extract SMILES.
3. **Canonicalize SMILES** — use RDKit `Chem.MolToSmiles(Chem.MolFromSmiles(smi))`.
4. **Deduplicate IR** — for each canonical SMILES, keep the record with the largest `len(value.x)`.
5. **Select best NMR per nucleus** — for each (canonical SMILES, nucleus), keep the record with the most peaks.
6. **Pair IR and NMR** — inner join on canonical SMILES.
7. **Resample IR** — interpolate each spectrum to a fixed grid: 400–4000 cm⁻¹ at 2 cm⁻¹ spacing (1,801 points). Use linear interpolation for upsampling.
8. **Normalize IR** — min-max scale each spectrum to [0, 1].
9. **Bin NMR peaks** — for each nucleus channel:
   - ¹H: 0–15 ppm bins at 0.01 ppm (1,500 bins)
   - ¹³C: 0–220 ppm bins at 0.1 ppm (2,200 bins)
   - Set bin to 1.0 where a peak falls within ±half-bin-width, else 0.0.
10. **Filter invalid samples** — remove molecules where RDKit cannot parse the SMILES.
11. **Scaffold-based split** — use Bemis-Murcko scaffolds for train/valid/test (70/15/15) to prevent scaffold leakage.
12. **Save processed files.**

### IR Normalization

- Per-spectrum min-max normalization to [0, 1].
- Alternative: Standard scaling (z-score), but min-max preserves the relative peak heights better.

### NMR Representation

- Since no intensities are available, use **binary binning** or **Gaussian peak smearing** (σ = 0.5 ppm for ¹³C, σ = 0.05 ppm for ¹H).
- Each nucleus channel becomes a fixed-length binary or soft vector.

### Missing Values

- IR: all records have complete x,y pairs; no missing values.
- NMR: y is always "NONE"; handle by using peak-only representation (no intensities to impute).

### Invalid SMILES

- Use RDKit to validate: skip and log any SMILES that cannot be parsed.
- Track the count of rejected records.

## 7. Recommended Processed Data Format

### Output Files

```
data/processed/
  pairs.csv              # One row per paired molecule:
                         #   sample_id, canonical_smiles, ir_record_idx, nmr_record_indices_json,
                         #   ir_x_range, ir_y_range, nmr_nuclei_list, split
  ir.npy                 # Shape: (N_paired, 1801) — resampled IR intensity vectors
  ir_x.npy               # Shape: (1801,) — shared IR wavenumber grid
  nmr_1h.npy             # Shape: (N_paired, 1500) — binned 1H NMR
  nmr_13c.npy            # Shape: (N_paired, 2200) — binned 13C NMR
  nmr_1h_x.npy           # Shape: (1500,) — shared 1H chemical shift grid
  nmr_13c_x.npy          # Shape: (2200,) — shared 13C chemical shift grid
  canonical_smiles.txt   # One canonical SMILES per line
  splits.json            # {"train": [0, 1, ...], "valid": [...], "test": [...]}
  meta.json              # Preprocessing parameters, grid definitions, statistics
  preprocessing_log.json # Dropped records, reasons, counts
```

### Fields per Sample

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | int | 0-indexed sample identifier |
| `pairing_key` | string | Canonical SMILES |
| `canonical_smiles` | string | RDKit-canonicalized SMILES |
| `ir_vector` | float[1801] | Min-max normalized IR transmittance on fixed grid |
| `nmr_1h_vector` | float[1500] | Binary/gaussian-binned 1H NMR |
| `nmr_13c_vector` | float[2200] | Binary/gaussian-binned 13C NMR |
| `ir_x` | float[1801] | Shared IR wavenumber grid (cm⁻¹) |
| `nmr_1h_x` | float[1500] | Shared 1H chemical shift grid (ppm) |
| `nmr_13c_x` | float[2200] | Shared 13C chemical shift grid (ppm) |
| `split` | string | `train`, `valid`, or `test` |

## 8. Saved Example Cases

### case_001

- **Path:** `examples/case_001/`
- **SMILES:** `CCCCC(CC)CI` (iodoalkane)
- **IR:** 870 points, gas phase, ~490–3966 cm⁻¹
- **NMR:** 1 record — ¹H at 300 MHz in CDCl₃, 4 peaks
- **Pairing:** Confident (SMILES match)
- **Files:** `metadata.json`, `ir.csv`, `nmr.csv`, `ir_preview.png`, `nmr_preview.png`

### case_002

- **Path:** `examples/case_002/`
- **SMILES:** `O=Cc1c(F)c(F)c(F)c(F)c1F` (pentafluorobenzaldehyde)
- **IR:** 1,780 points, ~485–4000 cm⁻¹
- **NMR:** 5 records — ¹H (2 sources), ¹³C (2 sources), ¹⁹F (1 source) in CDCl₃
- **Pairing:** Confident (SMILES match)
- **Files:** `metadata.json`, `ir.csv`, `nmr.csv`, `ir_preview.png`, `nmr_preview.png`

## 9. Implications for the Later Ablation

### Direct Concat Attention (E0, Baseline)

- IR tokens: patches of the 1,801-point vector (e.g., 64-point patches → ~28 patches)
- NMR tokens: bin indices for each nucleus channel, positional encoded
- All tokens concatenated into one sequence → standard Transformer encoder
- Simple to implement; may suffer from modality interference

### Intra-modal + Cross-modal Attention (E1)

- IR tokens attend within IR modality first (self-attention)
- NMR tokens attend within NMR modality first (self-attention)
- Cross-attention layers then allow IR ↔ NMR information exchange
- Requires two separate encoder paths + cross-attention fusion
- May better preserve modality-specific features

### Data Requirements for Both

- Both models need fixed-dimension inputs → the resampling/binning pipeline proposed above
- Both need paired samples → the canonical SMILES join
- Both need a scaffold-based split → prevents overly optimistic evaluation
- Variable-length IR spectra → resampling to fixed grid is essential

## 10. Risks and Open Questions

| Risk | Severity | Mitigation |
|------|----------|------------|
| Low pairing rate (33% of IR) | Medium | Acceptable for proof-of-concept; can expand NMR coverage later |
| Missing NMR intensities | Medium | Use binary/gaussian binning; loses peak height information |
| Non-canonical SMILES in raw files | Medium | RDKit canonicalization collapses most variants |
| Variable IR grid lengths | Low | Resampling to fixed grid is standard practice |
| NMR multi-record selection bias | Medium | "Most peaks" heuristic may not select the best-quality spectrum |
| Duplicate IR SMILES (different conditions) | Low | Keep highest resolution; note discarded conditions |
| No atom-level NMR assignment | Low | Peak lists don't include atom indices; token-level only |
| Scaffold split may not generalize | Medium | Bemis-Murcko scaffolds are coarse; consider molecular weight stratification |
| Large NMR file (3.37M records) | Low | Stream processing; only load overlapping SMILES into memory |
| IR dataset dominated by common lengths (870, 1780) | Low | Resampling normalizes all lengths |

## 11. Next Step

The next implementation step is to write the preprocessing scripts:

1. **`scripts/build_pairs.py`** — load raw JSONL files, canonicalize SMILES, dedup IR, select NMR, pair, and save `pairs.csv`.
2. **`scripts/preprocess_spectra.py`** — resample IR to fixed grid, bin NMR peaks, normalize, and save `.npy` arrays.
3. **`scripts/split_data.py`** — scaffold-based train/valid/test split, save `splits.json`.
4. **`src/data/transcross_dataset.py`** — PyTorch Dataset that loads processed `.npy` files and returns (ir, nmr_1h, nmr_13c) tensors.
5. **`configs/preprocessing.yaml`** — configuration for grid parameters, normalization method, split ratios.

These scripts will be written and tested on the remote server using `/bjhpc-safe-runner` before any model code is written.

---

*Report generated 2026-05-20. Raw data on bjhpc; examples stored locally.*
