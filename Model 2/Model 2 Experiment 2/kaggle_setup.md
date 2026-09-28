# Kaggle setup — Model 2 / Experiment 2

Needs torch, which the development machine does not have. Locally only the `mock` arm runs;
the backbone arms run here.

---

## 1. What to upload — **a new dataset is required for E2**

The E1 dataset cannot be reused. E2 imports two modules that are not in it
(`crop_geometry.py`, `check_crop_geometry.py`), so the notebook would stop in cell A3.

> **`final_kaggle_upload_m2e2.zip`** in the repository root — **278.6 MB**, 2,023 members.
> Built and verified by `python scratch/make_kaggle_zip.py m2e2`.

### 1.1 The steps

1. Kaggle → **Datasets → New Dataset**
2. Title: `soil-gsd-m2-e2` · Visibility: **Private**
3. Drag `final_kaggle_upload_m2e2.zip` into the upload area
4. **Create**. Kaggle unzips it, so the file list must show `data/`, `preprocess/`,
   `backbones.py`, `check_backbones.py`, `crop_geometry.py`, `check_crop_geometry.py`,
   `Model 1 Experiment 3/`, `Model 2 Experiment 1/` — **not** the zip itself
5. Open the notebook → Input panel → **remove every dataset attachment** →
   **Add Input → Your Datasets → `soil-gsd-m2-e2`**

**Remove the old attachments.** Kaggle searches attachments alphabetically, and
`soil-gsd-processed-final-fixed` and `soil-gsd-m2-e1-code` both contain an older
`backbones.py` that sorts before this one. Leaving them attached can silently re-introduce
the 518 px bug that cost the last session. Attach exactly one dataset.

### 1.2 What is in the zip, and what is left out

```
data/tiles/            1,976 PNGs, 276 MB   <- every pixel the notebook reads
data/processed_meta/   manifests, audit.json, golden_checks.json, target_ppm.json
data/sample_submission.csv
preprocess/            the package; `from preprocess import verify` is the validated
                       hand-built feature definition, reused not reimplemented
backbones.py           E1's frozen torch surface, UNMODIFIED
check_backbones.py     the backbone contract test
crop_geometry.py       E2's one new variable (crop + patch geometry), no torch
check_crop_geometry.py 28 assertions pinning the sweep and the reachability finding
Model 1 Experiment 3/  features_soil.csv, cv_families.csv, Submission_Model1_E3.csv, .cache/
Model 2 Experiment 1/  instructions.txt, Submission_Model2_E1.csv, recovered transcript, .cache/
```

Left out, with the reason measured rather than guessed:

| excluded | size | why |
|---|---|---|
| `data/training_down`, `data/testing_down` | 445 MB | never opened; their paths are join keys only |
| `data/Training`, `data/Test` | — | raw sources; model notebooks are forbidden to read them |
| `data/qc` | 230 MB | contact sheets, nothing reads them |
| `data/processed_meta/masks` | — | never read; per-tile `soil_fraction` is in the tile manifest |
| any `local mock dry-run/` folder | — | plumbing artifacts with the same filenames as real results |
| `*.npy` embedding caches | 17 MB | regenerable, and a stale cache that still matches its key is worse than none |
| duplicate copies of the four root modules | — | two copies of `backbones.py` is how one silently wins |

---

## 2. Notebook settings

| setting | value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **ON** — timm fetches the DINOv2 weights once (~90 MB) |
| Language / kernel | Python 3 |
| Persistence | On — the `.npy` sweep caches are the expensive artifact |
| Datasets | `soil-gsd-m2-e2`, and nothing else |
| Environment variables | none needed |

Unset `BACKEND` defaults to `both` on Kaggle, which is what this experiment wants: R and D
across all four crops in one pass, so the gate is evaluable.

### 2.1 Getting the notebook onto Kaggle

**Kaggle → Your Work → New Notebook → upload notebook** → pick
`Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb`. Then **Save Version → Save & Run**
(time limit 2–3 hours). Not "Run All" — a draft session dies when the tab closes.

Do not paste `backbones.py` or `crop_geometry.py` into cells. If the code exists in two
places the notebook and the modules can disagree, and cell G2's reproduction assertion
would then be comparing a reimplementation against itself.

---

## 3. What the early cells will tell you

| cell | expect | if it fails |
|---|---|---|
| A1 | `crop sweep: (256, 128, 64, 32)` and `outputs -> /kaggle/working` | wrong BACKEND value |
| A3 | three resolved paths: `INPUT_ROOT`, `backbones.py`, `crop_geometry.py` | an attachment is missing from the dataset |
| A4 | `config_hash 010f44c36c74` | wrong dataset |
| C3 | the reachability table: 12 → 13 → 13 → 14 resolvable soils | — |
| D1 | `check_crop_geometry: ALL CHECKS PASSED` then `check_backbones: ALL CHECKS PASSED` | the crop or the backbone contract is broken |
| D2 | every crop `density ratio ~1.0`, `crops below 0.5 soil: 0` | centre crops are not soil-representative, so the sweep is confounded |
| **H3** | **`PASS: E2's crop-256 arm is E1's arm`** | **E2 is not comparable to E1. Read nothing below.** |
| I1 | the nine-arm sweep table | — |
| K1 | per-D50-group deltas — the section that decides whether any gain is real | — |
| O1 | the gate, printed as clauses | — |

**H3 is the gate before the gate.** It says which level it checked at. With this zip it
will print `check level: means only` because E1's per-soil file was lost locally — that is
still a real check (E2's crop-256 means must match E1's recorded 43.02 / 44.33 / 39.78),
just not the strongest one. To upgrade it, download run 2's `cv_per_soil.csv` from Kaggle
into `Model 2/Model 2 Experiment 1/` and rebuild the zip; H3 then compares all 24 per-soil
values.

Expected wall time: **12–20 minutes**. About 20,000 frozen forward passes across the sweep,
then 256 ridge fits per arm for nine arms.

---

## 4. Reading the outcome

Fixed before the run, executed as code in cell O1:

```
submit = exists a crop in {128, 64, 32} where
         D beats M1 in-domain
         AND D beats R at that crop with a significant paired CI
         AND the gain is concentrated in the patch-limited D50 groups
```

| outcome | meaning | do this |
|---|---|---|
| a crop satisfies all four | E1's refutation was partly a scale artefact | submit `Submission_Model2_E2.csv`; the corrected crop becomes frozen |
| D improves but never past R | the scale effect is real and is not about pretraining | do not submit; Model 2 closes, report the crop finding |
| no crop helps D | E1's refutation survives its declared confound | do not submit; **Model 2 is closed on evidence** |
| H3 fails | E2 is void | fix the pipeline, read nothing |

Three of four close Model 2, and the reachability table in cell C3 says that is the likely
answer. Read **section K (per-group), not section I (the mean)**, first: a mean over 24
soils where ten of them physically cannot respond to magnification is not a measurement of
the mechanism.

---

## 5. What to bring back

Download the Output files into `Model 2/Model 2 Experiment 2/`:
`Experiment2.txt`, `arms_config.csv`, `sweep_results.csv`, `sweep_trend.csv`,
`per_group_effects.csv`, `cv_per_fold.csv`, `cv_per_soil.csv`, `paired_effects.csv`,
`mechanism_table.csv`, `embedding_separation.csv`, `camres_report_only.csv`,
`predictions.csv`, `gate.json`, `features_soil_M1_control.csv`, `cv_families.csv`, and
`Submission_Model2_E2.csv` if the gate fired. Also keep the executed notebook.

**Do not tune anything against the public leaderboard.** It scores a fixed 3 of the 10 test
soils and the final ranking is `0.30·public + 0.70·private`.

---

## 6. Verified locally, and what was not

`BACKEND=mock`, two runs, all 15 emitted files byte-identical:

- `check_crop_geometry.py` 28 assertions pass, including the pinned geometry
  (3.5145 / 1.7573 / 0.8786 / 0.4393 mm) and the bounded-reachability finding;
- `check_backbones.py` passes and proves the torch arms **refuse** rather than fall back;
- the control reproduces E3's feature matrix to **2.842e-14** (gate < 1e-6);
- **H3 regression check passes**: E2's M1 nested score is 43.022 against E1's recorded
  43.02, gap 0.0017;
- mock in-domain is worse than the no-image floor 82.62 at **every** crop (87.82 / 94.95 /
  95.70 / 96.81) — P1 confirmed, and no crop lets a noise arm look informative;
- the gate reports `NOT EVALUABLE` and **writes no submission**;
- the submission fit and all eight validation gates **execute anyway** as a labelled
  plumbing check — E1's lesson that a gate-conditional section is untested code;
- no cell produces empty output, which is how a dead loop was caught in E1;
- `python -m preprocess.final_check` PASSED; nothing under `data/Training`, `data/Test` or
  `data/processed_meta` modified;
- the archive round-trips: all 1,976 tiles byte-identical after extraction, including the
  75 Münster names carrying U+00FC.

**Not verified anywhere yet:** the real embedding geometry at each crop, and every number
involving arms R and D. Cells D1 and H3 exist so those are checked before any conclusion is
drawn rather than after.
