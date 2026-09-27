# Kaggle setup — Model 1 / Experiment 1

This experiment is **CPU-only**. It needs no GPU, no torch, and no internet.

---

## 1. What to upload

Create **one Kaggle Dataset** (Datasets → New Dataset), suggested name
`soil-gsd-processed`, containing exactly this tree:

```
soil-gsd-processed/
├── preprocess/                          <-- the whole package directory
│   ├── __init__.py  __main__.py  run.py
│   ├── config.py  inventory.py  cameras.py  geometry.py
│   ├── resample.py  colour.py  soilmask.py  tiling.py
│   ├── manifest.py  audit.py  qc_sheets.py  verify.py  final_check.py
└── data/
    ├── processed_meta/
    │   ├── manifest_images.csv          <-- single source of truth (162 rows)
    │   ├── manifest_tiles.csv
    │   ├── manifest_samples.csv
    │   ├── audit.json                   (config_hash cross-check)
    │   ├── golden_checks.json           (config_hash cross-check)
    │   └── target_ppm.json
    ├── tiles/                           1,976 materialised 256 px tiles   ~269 MB
    ├── training_down/                   127 canonical PNGs                ~342 MB
    ├── testing_down/                     35 canonical PNGs                ~103 MB
    ├── Training_labels_updated.csv
    ├── sample_submission.csv            <-- authoritative column names
    └── ppm_updated.csv
```

**Do NOT upload** `data/qc/` (230 MB of contact sheets), `data/Training/`, `data/Test/`,
`_look/` or `scratch/`. The raw folders must not be there — the notebook is forbidden
from reading them, and their absence makes that guarantee physical.

**Do NOT upload `data/processed_meta/masks/`** — Experiment 1 never reads masks; the
per-tile `soil_fraction` is already in `manifest_tiles.csv`.

Upload size ≈ **715 MB**.

### Why `preprocess/` must be included
The notebook imports `preprocess.verify.feats` and `pv.spectral_centroid` rather than
reimplementing them, so the model's features are byte-for-byte the definition that
`golden_checks.json` was validated against. A locally rewritten copy of the feature
function is the easiest way to make a result unreproducible.

---

## 2. Where things appear on Kaggle

Attached datasets are mounted read-only under `/kaggle/input/<dataset-slug>/`, and the
slug may gain a suffix, so **never hardcode it**. Cell A3 searches for
`data/processed_meta/manifest_images.csv` under every `/kaggle/input/*` entry and uses
the first match, so the notebook works whether the dataset is named
`soil-gsd-processed` or `username/soil-gsd-processed`.

Outputs are written to `/kaggle/working` (cell A1). `Submission_Model1_E1.csv` appears
in the notebook's Output panel.

---

## 3. Notebook settings

| setting | value |
|---|---|
| Accelerator | **None** (CPU is faster here — no GPU I/O overhead) |
| Internet | Off |
| Language / kernel | Python 3 |
| Persistence | optional |
| Dataset | attach `soil-gsd-processed`, pinned to a specific **version** |

Pinning the version matters: the dataset is the frozen preprocessing, and an
unpinned "latest version" attachment can silently change under an experiment.

---

## 4. Running

`Run All` (or Restart & Run All). The notebook is linear — no cell needs manual input
and no cell depends on state created outside the notebook.

Order of sections:

| section | cells | what it does |
|---|---|---|
| A setup | 1–4 | config, imports, input resolution, `config_hash` pin + seed |
| B data | 6–11 | manifests, labels, metric self-check, inspection, tile visuals, **derives CV families** |
| C features | 13–17 | per-tile extraction, tile→image→soil aggregation, inspection with permutation controls |
| D curves | 19–20 | rank-k ceilings, PCA basis + monotonicity projection |
| E validation | 22–25 | leave-one-family-out, results, error analysis, cross-camera diagnostic |
| F output | 27–32 | refit on all 24, predict test, submission gate, write files, summary |

Cells A4 and F3 are **assertion gates**. If the dataset version is wrong, or the
submission is malformed, the notebook stops rather than producing a file.

---

## 5. Runtime and cost

About **1–2 minutes** on Kaggle CPU. Feature extraction over 1,976 tiles dominates.
The tile features are cached to `.cache/tile_features_<size>_<config_hash>.csv` inside
the output folder, so a re-run within a persisted session skips extraction.
Ridge is closed-form; there are no epochs.

---

## 6. Outputs produced by the notebook

| file | contents |
|---|---|
| `Submission_Model1_E1.csv` | 10 rows, official IDs and column order, validated |
| `Experiment1.txt` | full experiment record, generated from live objects |
| `cv_families.csv` | the 16 curve-distance families used for CV |
| `features_soil.csv` | the per-soil feature matrix actually used |
| `cv_results.csv` | per-fold EMD, alpha, held-out soils |
| `feature_inspection.csv` | rho vs log D50, permutation control, camera ratio |
| `tile_features.csv` | per-tile features (also cached) |

`Experiment1.txt` is written by the notebook from the live config and result objects,
never by hand, so the record cannot drift from what actually ran.

---

## 7. After the run

Copy the public score into `Experiment1.txt` §16 by hand. Leave the private field blank
until the competition closes.

**Do not tune anything against the public leaderboard.** It is scored on a fixed 3 of
the 10 test soils and the final ranking is `0.30·public + 0.70·private`.

---

## 8. Local execution

The same notebook runs from the repository root on a machine that has the processed
data, with no changes — cell A3 falls back to the repo root when `/kaggle/input` is
absent, and cell A1 routes outputs to `Model 1/Model 1 Experiment 1/`.

Requires `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`. No torch.
