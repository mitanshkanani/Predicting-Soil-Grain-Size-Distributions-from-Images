# Kaggle setup — Model 1 / Experiment 4

**CPU-only.** No GPU, no torch, no internet. Same attached dataset as E1/E2/E3, with E3's
outputs added.

E4 changes one thing — how colour is represented — and judges it with the CAM+RES ruler
that E3 demoted to **ordinal-only**. It is the first experiment whose central question is
one the ruler is forbidden by rule to answer.

---

## 1. What to upload

### 1.1 The processed dataset — unchanged

If `soil-gsd-processed` already exists with the `experiment_history/Model1_E1/` and
`Model1_E2/` branches, **you only need to add the E3 branch in §1.2 and publish a new
version.** The tree is:

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
    │   ├── manifest_samples.csv         <-- carries the 11 label columns; E4's ONLY
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
from reading them, and their absence makes that a physical guarantee rather than a promise.

Upload size ≈ **715 MB** (unchanged).

### 1.2 History branches — ~40 KB total

```
soil-gsd-processed/
└── experiment_history/
    ├── Model1_E1/                 (added for E2)
    │   ├── features_soil.csv   cv_families.csv   Submission_Model1_E1.csv
    ├── Model1_E2                (added for E3)
    │   ├── features_soil.csv   cv_families.csv   Submission_Model1_E2.csv
    └── Model1_E3                <-- NEW for E4
        ├── features_soil.csv          (9.9 KB)  E3's per-soil matrix
        ├── cv_families.csv            (1.3 KB)  the 16 families
        ├── Submission_Model1_E3.csv   (2.1 KB)  for the prediction comparison plot
        └── paired_effects.csv         (0.9 KB)  E3's headline effect, restated in C2
```

These live in `Model 1/Model 1 Experiment 3/`. Copy them; do not regenerate.

**Why they are required.** Cell D2 rebuilds E3's soil matrix from the tiles and asserts it
matches `features_soil.csv` — measured drift on the local run was **2.8e-14**. That
assertion has already earned its keep once: the first E4 run **failed** it at 2.18e+01 and
exposed that E4 was pooling cameras before soils while E1/E2/E3 pooled images. Without
the gate the two experiments would have been silently incomparable.

---

## 2. Where things appear on Kaggle

Attached datasets mount read-only under `/kaggle/input/<dataset-slug>/`; the slug may gain
a suffix, so nothing hardcodes it.

| cell | resolves | by |
|---|---|---|
| A3 | `INPUT_ROOT` | searches every `/kaggle/input/*` entry for `data/processed_meta/manifest_images.csv` |
| B2 | E3's outputs | tries `INPUT_ROOT/experiment_history/Model1_E3`, then the sibling folder locally |

Outputs go to `/kaggle/working` (cell A1). `Submission_Model1_E4.csv` appears in the
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

| section | what it does |
|---|---|
| A setup | config, imports, input resolution, `config_hash` pin + seed |
| B data | manifests, labels from the sample manifest, CV families copied from E3 |
| B metric | EMD with the three-way reconciliation against the host's published 100.31 |
| B ground truth | the four external scores: E1 172.70, baseline 102.37, E2 71.27, E3 61.24 |
| C ruler status | the 0.84 / 0.94 / 1.18 calibration table and the **ordinal-only ruling**, printed before any arm is scored |
| D cancellations | reproduces both cancelled hypotheses from artifacts, plus the matched-ruler circularity demo |
| E arms | the six colour arms, the device clustering for C2b, six evaluation conditions, leak assertion, nested selection engine |
| F paired effects | every arm vs C0 with soil-level bootstrap CIs and the "NOT RESOLVABLE" verdict |
| G predictions | P1–P4 verdicts printed as they are computed |
| H mechanism | per-feature camera shift vs sharpness shift |
| I circularity audit | demonstrates C2's CAM inflation numerically |
| J selection | primary vs minimax, the probe conversion, the alpha parsimony rule |
| K submission | fit, 8 gates, write and re-read |
| L analysis | curve comparison vs E3, per-soil paired-difference strip plot |
| M record | write `Experiment4.txt`, print the summary |

### Assertion gates — the notebook stops rather than emit a bad result

| cell | gate |
|---|---|
| A4 | one `config_hash` across manifest, audit and golden checks |
| B(metric) | our EMD reproduces the host's published 100.31 |
| C1 | the ruler's *ranking* is still correct on the three measured points — if not, it is not even ordinal and this experiment's premise collapses |
| E1 | the three families are disjoint and C0 is exactly E3's 12-feature cell |
| D2 | rebuilt feature matrix matches E3 to < 1e-6 |
| D1 | the real view still has both splits; no res view has test tiles |
| E2 | no NaN in any arm's table after normalisation |
| E3 | no evaluated soil's label enters its own fit, under **any** of the six conditions |
| F1 | every arm aligns to the same shared soil vector |
| K2 | all 8 submission checks |

---

## 5. Runtime and cost

**First run: ~3.5 minutes** on CPU. **Re-run with persistence: ~10 seconds.**

The cost is the nested selection engine: 6 arms × 16 outer folds × 10 alphas × 15 inner
folds × 4 conditions ≈ 57,600 closed-form ridge fits on ≤21 rows. Its result is cached to
`.cache/nested_<key>.json`, keyed by a SHA-256 over the arms, their normalisation mode, the
conditions, the alpha grid, `config_hash`, seed and PCA rank — so changing any one of them
invalidates the cache rather than silently reusing it.

No tile extraction happens if the caches resolve: E4 finds the base and both
resolution-matched caches in E3's `.cache` and copies them.

---

## 6. Outputs produced by the notebook

| file | contents |
|---|---|
| `Submission_Model1_E4.csv` | 10 rows, official IDs and column order, validated |
| `Experiment4.txt` | full record, generated from live objects |
| **`paired_effects.csv`** | **the primary artifact** — every arm vs C0 with CIs |
| `c2_circularity_audit.csv` | C2's CAM gain vs its CAM+RES gain |
| `circularity_demo.csv` | the matched-ruler artefact that killed resolution-matched training |
| `blur_sensitivity_matrix.csv` | the 3×3 train-blur × test-blur matrix for two feature sets |
| `mechanism_table.csv` | per-feature camera shift and sharpness shift in SD |
| `selection_rule.json` | selected arm, primary vs minimax, probe flag, bet/finding |
| `features_soil.csv` | the per-soil matrix actually used |
| `cv_families.csv` | copied from E3; E4 does not re-derive the grouping |

---

## 7. After the run

Copy the public score into `Experiment4.txt` §11 and record the P5 verdict. Leave the
private field blank until the competition closes.

**This submission is a discriminating probe, not an improvement claim.** The record says
so explicitly. What each outcome means:

| outcome | consequence |
|---|---|
| below 61.24 | colour was hurting after all; CAM and in-domain were right and CAM+RES was wrong. The ruler's ordinal authority is narrower than E3's ruling assumed |
| at 61.24 ± a few | colour genuinely does not matter. Keep E3's configuration and move on |
| above 61.24 | CAM+RES was right to prefer colour. **Model 1 has plateaued** — sharpness handled, aggregation exhausted, colour settled at "keep it". Model 2 becomes an evidence-backed decision |

Do not tune against the public leaderboard. It scores a fixed 3 of the 10 test soils and
the final ranking is `0.30·public + 0.70·private`.

---

## 8. Local execution

Runs unchanged from the repository root: cell A3 falls back to the repo root when
`/kaggle/input` is absent, cell A1 routes outputs to `Model 1/Model 1 Experiment 4/`, and
cell B2 resolves E3's artifacts as a sibling folder.

Requires `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`. No torch,
no cv2, no skimage, no nbformat.

The notebook is generated from `scratch/build_m1e4.py`. Edit that file and run
`python scratch/mk_run4.py`, which regenerates the notebook, flattens the code cells into
`scratch/run_m1e4.py` for headless execution, and compile-checks the result — so the
notebook and the verified local run cannot diverge.
