# Kaggle setup — Model 1 / Experiment 3

**CPU-only.** No GPU, no torch, no internet. Same attached dataset as E1/E2, with E2's
outputs added alongside E1's.

E3 changes one thing — the feature set — and judges it with the ruler E2 validated. It is
defined relative to E2 and asserts against E2's feature matrix, so E2's outputs must be
reachable at run time (§1.2).

---

## 1. What to upload

### 1.1 The processed dataset — unchanged from E1/E2

If `soil-gsd-processed` already exists with the `experiment_history/Model1_E1/` branch
created for E2, **you only need to add the E2 branch in §1.2 and publish a new version.**
The tree is:

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
    │   ├── manifest_samples.csv         <-- carries the 11 label columns; E3's ONLY
    │   │                                    label source
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
`data/processed_meta/masks/`. The raw folders must be absent — the notebook is forbidden
from reading them and their absence makes that a physical guarantee, not a promise.

Upload size ≈ **715 MB** (unchanged).

### 1.2 History branches — ~28 KB total

```
soil-gsd-processed/
├── experiment_history/
│   ├── Model1_E1/                 (already added for E2)
│   │   ├── features_soil.csv
│   │   ├── cv_families.csv
│   │   └── Submission_Model1_E1.csv
│   └── Model1_E2                  <-- NEW for E3
│       ├── features_soil.csv          (10.5 KB)  E2's per-soil matrix
│       ├── cv_families.csv             (1.3 KB)  the 16 families
│       └── Submission_Model1_E2.csv   (2.2 KB)  for the prediction comparison
```

These live in `Model 1/Model 1 Experiment 2/` in the repo. Copy them; do not regenerate.

**Why they are required.** Cell E5 rebuilds E2's soil matrix from the tiles and asserts it
matches `features_soil.csv` — measured drift on the local run was **2.8e-14**. That
assertion is what licenses the claim that any score difference between E2 and E3 is
attributable to the feature set and nothing else. It has already earned its keep: the
first local run failed it, exposing that E3 was pooling cameras before soils while E1/E2
pooled images. Without that gate the two experiments would have been silently
incomparable.

---

## 2. Where things appear on Kaggle

Attached datasets mount read-only under `/kaggle/input/<dataset-slug>/`; the slug may gain
a suffix, so nothing hardcodes it.

| cell | resolves | by |
|---|---|---|
| A3 | `INPUT_ROOT` | searches every `/kaggle/input/*` entry for `data/processed_meta/manifest_images.csv` |
| C1 | E2's outputs | tries `INPUT_ROOT/experiment_history/Model1_E2`, then `INPUT_ROOT/Model 1 Experiment 2` |

Outputs go to `/kaggle/working` (cell A1). `Submission_Model1_E3.csv` appears in the
notebook's Output panel.

---

## 3. Notebook settings

| setting | value |
|---|---|
| Accelerator | **None** — CPU only; there is no learned model |
| Internet | Off |
| Language / kernel | Python 3 |
| Persistence | **on** — strongly recommended, see §5 |
| Dataset | attach `soil-gsd-processed`, pinned to a specific **version** |

---

## 4. Running

`Restart & Run All`. Linear; no cell needs manual input.

| section | cells | what it does |
|---|---|---|
| A setup | 1–4 | config, imports, input resolution, `config_hash` pin + seed |
| B metric | 6–8 | manifests and labels, metric reconciliation, baselines + the three Kaggle ground-truth scores |
| C prior results | 10–11 | load E2's artifacts, print the 172.70 → 102.37 → 71.27 trajectory |
| D families | 13 | declare core / frequency / colour, assert disjointness and that core+freq+colour **is** E2's matrix |
| E instruments | 15–21 | tile features, derived resolution-match blur, image-level table, **E2 reproduction assertion**, family copy, four conditions + **leak assertion**, model primitives, **nested selection engine** |
| F grid | 23–24 | the flat 2×2 across every alpha and condition |
| G inference | 26–28 | **paired simple effects**, main effects + interaction, per-condition breakdown (this is what P1 is judged on) |
| H mechanism | 30 | per-family shift under camera change vs under sharpness change |
| I controls | 31 | each family alone, against a shuffled-label control |
| J selection | 33–34 | primary vs minimax, the bet/finding label, expected band |
| K submission | 36–38 | fit, 8 gates, write and re-read |
| M analysis | 39–40 | curve comparison vs E2, per-soil paired-difference strip plot |
| N record | 41–42 | write `Experiment3.txt`, print the summary |

### Assertion gates — the notebook stops rather than emit a bad result

| cell | gate |
|---|---|
| A4 | one `config_hash` across manifest, audit and golden checks |
| B2 | our EMD reproduces the host's published 100.31 |
| D1 | the three families are pairwise disjoint and sum to E2's 14 features |
| E5 | rebuilt feature matrix matches E2 to < 1e-6 |
| E7 | no evaluated soil's label enters its own fit, under **any** of the six conditions |
| G1 | every cell is aligned to the same shared soil vector |
| J2 | the submission uses an alpha that is actually on the grid |
| K2 | all 8 submission checks |

---

## 5. Runtime and cost

**First run: ~5 minutes** on CPU. **Re-run with persistence: ~45 seconds.**

The cost is the nested selection engine (cell E9): 4 cells × 16 outer folds × 10 alphas ×
15 inner folds × 4 conditions ≈ 38,000 closed-form ridge fits on ≤21 rows. Its result is
cached to `.cache/nested_<key>.json`, keyed by a SHA-256 of the cells, conditions, alpha
grid, `config_hash`, seed and PCA rank — so changing any one of them invalidates the cache
rather than silently reusing it.

No tile extraction happens if the caches resolve: E3 finds the base and both
resolution-matched caches in E2's `.cache` and copies them.

---

## 6. Outputs produced by the notebook

| file | contents |
|---|---|
| `Submission_Model1_E3.csv` | 10 rows, official IDs and column order, validated |
| `Experiment3.txt` | full record, generated from live objects |
| **`paired_effects.csv`** | **the primary artifact** — simple effects, main effects, interaction, all with soil-bootstrap CIs |
| `factorial_grid.csv` | every cell × alpha × condition |
| `mechanism_table.csv` | per-feature camera shift and sharpness shift in SD |
| `family_controls.csv` | each family alone vs its shuffled-label control |
| `selection_rule.json` | selected cell, primary vs minimax, bet/finding flag |
| `features_soil.csv` | the per-soil matrix actually used |
| `cv_families.csv` | copied from E2; E3 does not re-derive the grouping |

---

## 7. After the run

Copy the public score into `Experiment3.txt` §15 and record the P4, P5 and P6 verdicts
there. P4–P6 are the only blind predictions in this experiment — P1–P3 were already
answered by reconnaissance, which the record labels explicitly.

**Do not tune against the public leaderboard.** It scores a fixed 3 of the 10 test soils
and the final ranking is `0.30·public + 0.70·private`.

What each outcome means:

| outcome | consequence |
|---|---|
| score in 44–49 | the ruler extrapolates across kinds of change; Model 2 may be scoped against it |
| score < 71.27 but > 49 | the ruler's direction is right, its calibration band is too narrow; widen it |
| score ≥ 71.27 while P1–P3 held | the ruler ranks feature sets but cannot predict them — demote it to a diagnostic and do not scope Model 2 on it |

---

## 8. Local execution

Runs unchanged from the repository root: cell A3 falls back to the repo root when
`/kaggle/input` is absent, cell A1 routes outputs to `Model 1/Model 1 Experiment 3/`, and
cell C1 resolves E2's artifacts as a sibling folder.

Requires `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`. No torch,
no cv2, no skimage, no nbformat.

The notebook is generated from `scratch/build_m1e3.py`. Edit that file and run
`python scratch/mk_run3.py`, which regenerates the notebook, flattens the code cells into
`scratch/run_m1e3.py` for headless execution, and compile-checks the result — so the
notebook and the verified local run cannot diverge.
