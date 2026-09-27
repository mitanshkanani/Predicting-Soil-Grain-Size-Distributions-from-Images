# Kaggle setup — Model 1 / Experiment 2

**CPU-only.** No GPU, no torch, no internet. Same attached dataset as Experiment 1 plus
three small CSVs from E1's output folder.

Experiment 2 calibrates the *selection instrument*, not the model. It is defined relative
to Experiment 1 and compares itself against E1's stored results, so those results have to
be reachable at run time (§1.2).

---

## 1. What to upload

### 1.1 The processed dataset — identical to Experiment 1

If you already created `soil-gsd-processed` for E1, **the existing version still works**.
Do not re-upload it. It contains:

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
    │   ├── manifest_samples.csv         <-- carries the 11 label columns; E2's ONLY
    │   │                                    label source. It never reads raw labels.
    │   ├── audit.json                   (config_hash cross-check)
    │   ├── golden_checks.json           (config_hash cross-check)
    │   └── target_ppm.json
    ├── tiles/                           1,976 materialised 256 px tiles   ~269 MB
    ├── training_down/                   127 canonical PNGs                ~342 MB
    ├── testing_down                      35 canonical PNGs                ~103 MB
    ├── Training_labels_updated.csv
    ├── sample_submission.csv            <-- authoritative column names
    └── ppm_updated.csv
```

**Do NOT upload** `data/qc/`, `data/Training/`, `data/Test/`, `_look/`, `scratch/`, or
`data/processed_meta/masks/`. The raw folders must not be present — the notebook is
forbidden from reading them and their absence makes that guarantee physical rather than
a promise.

Upload size ≈ **715 MB** (unchanged from E1).

### 1.2 NEW for E2 — Experiment 1's outputs, ~14 KB

Add this branch to the same dataset and **publish a new version**:

```
soil-gsd-processed/
└── experiment_history/
    └── Model1_E1/
        ├── features_soil.csv            (11.7 KB)  the per-soil matrix E1 used
        ├── cv_families.csv               (1.3 KB)  the 16 curve-distance families
        └── Submission_Model1_E1.csv      (1.5 KB)  for the prediction comparison plot
```

These live in `Model 1/Model 1 Experiment 1/` in the repo. Copy them, do not regenerate
them.

**Why they are required.** Cell E7 rebuilds E1's feature matrix from the tiles and asserts
it matches `features_soil.csv` — measured drift on the local run was **5.7e-14**, i.e. bit
identical. That assertion is what licenses the claim that any score difference between E1
and E2 is attributable to the ruler and the two excluded columns and to nothing else.
Without the file there is no control, and the experiment's central claim is unverifiable.
Cell C1 fails loudly with these paths in the message rather than silently skipping the
check.

---

## 2. Where things appear on Kaggle

Attached datasets mount read-only under `/kaggle/input/<dataset-slug>/`, and the slug may
gain a suffix, so nothing hardcodes it.

| cell | resolves | by |
|---|---|---|
| A3 | `INPUT_ROOT` | searches every `/kaggle/input/*` entry for `data/processed_meta/manifest_images.csv` |
| C1 | `E1_DIR` | tries `INPUT_ROOT/experiment_history/Model1_E1`, then `INPUT_ROOT/Model 1 Experiment 1` |

Outputs go to `/kaggle/working` (cell A1). `Submission_Model1_E2.csv` appears in the
notebook's Output panel.

---

## 3. Notebook settings

| setting | value |
|---|---|
| Accelerator | **None** — CPU is faster here; there is no learned model |
| Internet | Off |
| Language / kernel | Python 3 |
| Persistence | on, so the `.cache/` tile features survive a re-run |
| Dataset | attach `soil-gsd-processed`, pinned to a specific **version** |

Pin the version. This dataset *is* the frozen preprocessing, and an unpinned "latest
version" attachment can change underneath an experiment.

---

## 4. Running

`Restart & Run All`. The notebook is linear; no cell needs manual input and no cell
depends on state created outside the notebook.

| section | cells | what it does |
|---|---|---|
| A setup | 1–4 | config, imports, input resolution, `config_hash` pin + seed |
| B metric | 6–8 | manifests and labels, **three-way metric reconciliation**, reference baselines + the two Kaggle ground-truth scores |
| C E1 restated | 10–11 | load E1's artifacts, print the motivating table |
| D feature audit | 13–15 | training envelope vs test position, the `soil_fraction` defect assertion, texture-vs-colour shift |
| E instruments | 17–24 | tile features, **derived** resolution-match blur, synthetic-camera visuals, soil×camera table, **E1 reproduction assertion**, family copy, the four rulers, **leak assertion**, nested selection engine |
| F sweep | 26–32 | the factorial grid, the alpha-rank contradiction, nested procedure scores, **external-ordering test**, P2/P3 verdicts, shift decomposition plot |
| G verdict | 33–34 | select the ruler and the configuration, write `selection_rule.json` |
| H predict | 36–38 | fit, expected-score band, prediction comparison plot |
| I submission | 40–41 | 8 validation gates, write and re-read |
| J analysis | 42–43 | saturation readout, nearest-training-soil per row |
| K record | 44–45 | write `Experiment2.txt`, print the summary |

### Assertion gates — the notebook stops rather than emit a bad result

| cell | gate |
|---|---|
| A4 | one `config_hash` across manifest, audit and golden checks |
| B2 | our EMD reproduces the host's published 100.31 |
| D2 | `soil_fraction` is still degenerate in training and still excurses out of domain |
| E7 | rebuilt feature matrix matches E1 to < 1e-6 |
| E11 | no held-out soil's label enters its own fit, under either camera |
| G2 | the new ruler does not pick E1's alpha (if it did, the rulers are not really disagreeing) |
| I1 | all 8 submission checks |

---

## 5. Runtime and cost

About **2–3 minutes** on Kaggle CPU. Locally: 96 s.

Three tile-feature passes rather than one: 1,976 real tiles plus 2 × 1,541 synthetic
resolution-matched tiles. Ridge is closed form; there are no epochs. The nested
selection engine runs roughly 19,000 tiny ridge fits, which is negligible.

Caches land in `.cache/` inside the output folder:

```
tile_features_256_<config_hash>.csv                        (copied from E1 if present)
tile_features_res_256_<config_hash>_iPhone14.csv
tile_features_res_256_<config_hash>_iPhone16.csv
```

A persisted session skips all three on re-run.

---

## 6. Outputs produced by the notebook

| file | contents |
|---|---|
| `Submission_Model1_E2.csv` | 10 rows, official IDs and column order, validated |
| `Experiment2.txt` | full record, generated from live objects |
| **`criterion_comparison.csv`** | **the primary artifact** — nested score of each selection procedure |
| `alpha_sweep.csv` | every (ruler, feature set, alpha) triple scored by every ruler |
| `selection_rule.json` | the chosen ruler, feature set and the stated basis |
| `feature_range_audit.csv` | per-feature training envelope vs test position |
| `features_soil.csv` | the per-soil matrix actually used |
| `cv_families.csv` | copied from E1; E2 does not re-derive the grouping |

`Experiment2.txt` is written by the notebook from live objects, never by hand, so the
record cannot drift from what ran.

---

## 7. After the run

Copy the public score into `Experiment2.txt` §14 and fill in the P4 and P5 verdicts there.
Leave the private field blank until the competition closes.

**Do not tune anything against the public leaderboard.** It scores a fixed 3 of the 10
test soils and the final ranking is `0.30·public + 0.70·private`. E2's submission is a
calibration measurement for the ruler, not a target.

---

## 8. Local execution

Runs unchanged from the repository root: cell A3 falls back to the repo root when
`/kaggle/input` is absent, cell A1 routes outputs to `Model 1/Model 1 Experiment 2/`, and
cell C1 resolves E1's artifacts as a sibling folder.

Requires `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`. No torch,
no cv2, no skimage, no nbformat.

The notebook is generated from `scratch/build_m1e2.py`; edit that file and run
`python scratch/mk_run.py` rather than editing the `.ipynb` directly. `mk_run.py` also
flattens the code cells into `scratch/run_m1e2.py` for headless execution and compile-
checks the result, so the notebook and the verified local run cannot diverge.
