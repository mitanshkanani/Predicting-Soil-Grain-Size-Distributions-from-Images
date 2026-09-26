# Soil Grain-Size Distributions from Images — Progress Report

**Competition:** [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos) (Kaggle id 139732)
**Host:** Lukas Leibold, BOKU / EU GRID project · $500 prize · ~366 participants · ends ~late Nov 2026
**Phase completed:** data audit + preprocessing pipeline
**Status:** pipeline built, run end-to-end, and verified. No modelling work started.

---

## 1. The task

Predict a soil sample's **cumulative grain-size distribution** at 11 fixed diameters
(0.002 mm → 200 mm, DIN EN ISO 14688-1) from photographs of soil spread in a dark tray.

Scored by **log-weighted Earth Mover's Distance** — the area between the true and predicted
cumulative curves plotted against log₁₀(diameter). Range 0–500, lower is better.

The hard constraint: **24 labelled soil samples behind 127 training photographs**, tested against
10 samples behind 35 photographs.

---

## 2. Measured baselines — the numbers any model must beat

Reimplemented the metric and measured reference points on the training set:

| approach | EMD |
|---|---|
| naive 2D connected-component granulometry (particle counting) | ~198 |
| copy some *other* sample's curve | 104.2 |
| **official trivial baseline** (9.09% equal mass per bin) | **100.31** |
| train global mean curve | 85.39 |
| **train global median curve — "ignore the images entirely"** | **82.62** |
| perfect | 0 |

**82.62 is the wall.** A model scoring above it contributes nothing beyond a constant.

### The label space is nearly one-dimensional

PCA on the 24 label curves: **PC1 carries 91.4% of the variance and correlates 0.995 with
log₁₀(D50)**. Reconstruction error by how many numbers you get right:

| numbers predicted | best achievable EMD |
|---|---|
| 1 (essentially D50) | 20.5 |
| 2 | 11.0 |
| 3 | 7.4 |
| 11 free values | 0, but needs ~1000 samples |

The labels are lab curves interpolated linearly on a log axis, so they are smooth and
low-dimensional by construction. With 24 samples the tractable problem is regressing **2–4
curve parameters**, not 11 free outputs.

### Two soil families, and near-duplicates

k=2 clustering splits cleanly into **11 fine-grained** samples (D50 0.021–0.21 mm) and **13
coarse** ones (D50 1.8–6.1 mm). Several curves are near-twins — H366≈H371 (distance 3.3),
H367≈H372 (4.3), H549≈H668 (6.8), H038≈H637 (7.3), H181≈H183 (8.2). **The effective number of
distinct soils is ~8–10, not 24**, which has consequences for validation (§8).

---

## 3. Four findings that changed the plan

### 3.1 The provided PPM table is wrong for the pixels we have

`ppm_updated.csv` quotes PPM at each camera's **native sensor resolution**, but the delivered
training JPEGs were downscaled before release. Correcting for that:

| camera | images | native | delivered | factor | **real px/mm** | scene FOV |
|---|---|---|---|---|---|---|
| Motorola Edge | 69 | 4000×1800 @11.492 | 1600×720 | 0.400 exact | **4.5968** | 348×157 mm |
| Samsung A52 | 55 | 9248×6936 @26.33 | 1599×1200 | 0.1729 | **4.5526 / 4.5554** | 351×264 mm |
| Edge 60 Fusion (H374 only) | 3 | 4096×2304 @12.465 | 4096×2304 | 1.0 | **12.4650** | 329×185 mm |
| iPhone 14 (test) | 14 | 4032×3024 @13.942 | 4032×3024 | 1.0 | **13.9420** | 289×217 mm |
| iPhone 16 (test) | 21 | 5712×4284 @19.525 | 5712×4284 | 1.0 | **19.5250** | 293×219 mm |

A 2 mm grain subtends **9 px** in most training images but **39 px** in an iPhone 16 test image.
Effective PPM is **exactly constant within each camera** (per-image spread 0.000000), so one
calibration per camera suffices and there is no per-image scale variation to model.

### 3.2 Grain counting is a dead end — proved, not assumed

A classical granulometer (adaptive threshold → connected components → equivalent-area diameters
converted to mm via PPM) scored **EMD ≈ 198**, twice as bad as ignoring the images. At 4.55 px/mm
a 0.063 mm particle is 0.3 px. For the 11 fine samples, individual grains are **physically
unresolvable** in train *and* test. Only texture and appearance can inform the fine tail.

### 3.3 Camera pipeline, not soil, dominates the fine texture bands — and this one I got wrong first

See §7: my original version of this finding was substantially a measurement artefact. The
corrected picture is in §6.3.

### 3.4 The leaderboard structure makes public feedback nearly worthless

From the host directly, in the competition forum:

- **Public LB uses a fixed 3 of the 10 test soils.** Final = `0.30·public + 0.70·private`.
- The sub-1.0 top public scores are **public-split overfitting/probing, not a leak** — the host
  reviewed the private board and considers a leak "close to impossible".
- Camera-to-soil distance is **always exactly 21 cm**; all photos under the same lighting;
  different cameras are **deliberate**, to force generalisation across phones.
- **H031 was deleted** (lab ran out of soil) — why there are 24 samples, not 25.
- **H374 was re-photographed** after a participant caught it looking mislabelled. It is the only
  Edge 60 Fusion sample, the only one at a different scale, and the reported worst CV score.

Consequence: judge models only by internal grouped cross-validation, never by public score.

---

## 4. Bugs found in the existing project files

1. **Stray test folder.** `test_image_organization.py` matched `^iPhone(?:14|16)_HPC_`, but one
   file is spelled `iPhone_16_HPC_...` with an extra underscore, so it landed in a folder of its
   own. Airbus BS10-4bis7 really has 3 images, not 2. **Fixed.**
2. **Submission IDs are not folder names.** `sample_submission.csv` wants
   `HPC_Muenster_BS6_9_0-10m`; the folder is `Münster_BS6_9,0-10m` — umlaut transliterated *and*
   comma→underscore *and* a prefix added. A naive `"HPC_"+folder` fails on exactly that row.
   **Solved** with a normalised mapping, verified 10/10, and stored as `submission_id` +
   `submission_row_order` so a submission can be written in the required order.
3. **Camera must come from the filename, not EXIF.** EXIF exists on only **38 of 162** images —
   all 124 Motorola/Samsung training files have no EXIF block at all. The intuitive
   "EXIF-first, filename fallback" rule is unsatisfiable for 77% of the corpus.
4. **Rotation-blind files.** 25 of the 69 Motorola files are stored portrait with no EXIF to say
   so, so pairing width-with-width reports a bogus 122% scale mismatch. All axis comparisons pair
   short-side with short-side.
5. **`labels.csv` inside every training folder** would be ingested as an image by a naive walker.
6. Two placeholder density fields **disagree** (EXIF says 72 dpi on all 38; JFIF says 300 dpi on
   30 of them). Neither is a physical scale; both are recorded and flagged
   `resolution_is_physical = False`.

---

## 5. What was built

A `preprocess/` package, run as `python -m preprocess --all | --dry-run | --stage NAME`, with 11
resumable stages:

```
inventory → cameras → geometry → resample → soilmask → colour
          → tiling → manifests → audit → qc → verify
```

Stages 1–3 are metadata-only and finish in under a second, so `flags.csv` and the derived scale
can be inspected **before** spending 233 Mpx on pixels.

### Outputs

| path | contents |
|---|---|
| `data/training_down/`, `data/testing_down/` | 162 canonical PNGs, one per source image, on a common physical scale |
| `data/tiles/` | 1,976 materialised 256 px tiles |
| `data/processed_meta/manifest_images.csv` | 162 rows × 103 columns — the single source of truth |
| `data/processed_meta/manifest_tiles.csv` | 16,228 tile rows across 4 sizes |
| `data/processed_meta/manifest_samples.csv` | 34 samples + the 11 raw label columns |
| `data/processed_meta/{audit.md, audit.json, golden_checks.json, camera_table.csv, flags.csv}` | reporting |
| `data/qc/sheets/`, `data/qc/paired/` | 162 contact sheets + 21 paired-camera sheets |

Total new footprint ≈ 1.2 GB. **`data/Training/` and `data/Test/` are byte-identical to the
original download**, proved by a sha256 baseline recorded before any pixel work and re-checked at
audit — not merely asserted.

### Key design decisions

| Decision | Reason |
|---|---|
| Canonical scale **derived** as `min(effective_ppm)` = **4.552516 px/mm**, not typed | The proposed 4.58 exceeds the Samsung's 4.554, which would force a 0.57% **upscale** on all 55 Samsung images, violating the no-upscale rule. At 4.5525 every image is downsampled and residual cross-camera scale error is **0.07%** |
| Downsample test to the training scale | Match **information content, not pixel count**. A 4.6 px/mm training image contains zero information below ~0.9 mm; upscaling it teaches on mush and then hands the model genuinely sharp test pixels |
| **Display P3 → sRGB baked in** | 38 files (all 35 test + all 3 H374) carry a Display P3 ICC profile; the other 124 are untagged and decode as sRGB. Zero fitted parameters, no population assumption |
| **Gray-world stored as parameters, not applied** | It is a choice, and its gains depend on the soil mask, which is expected to change. Baking it would bake a mask decision |
| Per-channel standardisation **rejected** | Measured to amplify a noise channel rather than help |
| Mask picks a **crop rectangle, never deletes pixels** | Deleting a pixel needs only that pixel dark; deleting a row needs the whole row non-soil. Isolated dark grains cannot be removed by construction |
| Tiles manifest-only for 3 of 4 sizes | Materialising all four would be 20k+ files, each a redundant re-encode of pixels already stored losslessly |
| No augmentation on disk, no random split | Deterministic preprocessing only; `sample_id`/`cv_group` preserved so future validation groups at soil level |

---

## 6. Verification results

### 6.1 Audit — 37/37 assertions pass

Highlights: source tree byte-identical (162 files, 0 changed) · zero upsamples · max scale
deviation 0.0566% · FOV preserved to 0.1045 mm · effective PPM constant within camera to
0.0000000000 · EXIF/filename disagreements 0 · clipping *introduced* by resampling 0.00049 ·
every output re-opens with manifest dimensions · all rotations multiples of 90° (lossless) ·
submission IDs reproduce `sample_submission.csv` exactly 10/10 · single-camera samples are exactly
H366/H374/H637.

### 6.2 Information preservation — all checks pass

- **Units/grating check** — a synthetic 2 mm grating through the identical code path returns
  exactly 9.105 px. This validates the *code*, and catches a ppm ↔ mm-per-pixel inversion that
  every relative check would silently agree with.
- **Aliasing stop-band** — power above the canonical Nyquist is **36.4 dB** below the source's own
  (requirement ≥ 20 dB). This is the rigorous version of "high-quality anti-aliased downsampling";
  a `NEAREST` resample or accidental upscale fails it immediately.
- **Preprocessing is signal-neutral** — Spearman(feature, log D50) is **identical to three
  decimals** before and after resampling for every usable band (e4 0.678→0.678, e8 0.764→0.764,
  e16 0.763→0.763).
- **Grouped-split foldability proven** with a hash assignment and no RNG.

### 6.3 The camera-effect result, corrected

Texture bands are defined in **millimetres** and converted to pixels, so every camera is filtered
identically. Under that definition:

| band | σ (mm) | wavelength | camera var. | soil var. | ρ with log D50 | real signal? |
|---|---|---|---|---|---|---|
| e1 | 0.22 | 2.5 px | 54% | 23% | −0.24 | **no — at Nyquist** |
| e2 | 0.44 | 5.0 px | — | — | +0.49 | **no** |
| e4 | 0.88 | 10.0 px | **1.6%** | **95.7%** | +0.68 | yes |
| e8 | 1.76 | 20.0 px | **1.2%** | **94.3%** | +0.76 | yes |
| e16 | 3.52 | 40.1 px | 23% | 67% | +0.76 | yes |
| saturation | — | — | 71% | 47% | −0.40 | — |

**The usable soil signal lives in e4/e8/e16 and is nearly camera-free.** Camera contamination is
concentrated in sub-resolution bands (which should simply not be used) and in saturation (a real
colour difference). This is materially better news than the earlier reading suggested.

### 6.4 A noise floor that applies to every future claim

With 24 labelled soils, **shuffled labels reach |ρ| ≈ 0.52**. Any correlation below ~0.5 in this
project is indistinguishable from chance.

---

## 7. Corrections issued during this work

Kept visible rather than quietly replaced, because each one is a live risk of being re-derived
wrong.

1. **"Camera explains up to 71% of texture variance" — wrong.** An artefact of the exploratory
   feature using `sigma = min(image_shape)//12`, which is frame-dependent (59 px on Motorola,
   99 px on Samsung) and so injected a per-camera offset unrelated to soil or camera. Actual: ~1.5%
   for the bands that carry signal.
2. **"Same soil differs +63% to +124%, Samsung always higher, 21/21" — wrong.** With
   camera-independent bands it is a **crossover**: Samsung reads *lower* at fine scales and higher
   at coarse, and is higher in only 1/23 soils on the finest band. The saturation figure (+74%)
   did reproduce, because it is frame-size-independent.
3. **"Camera-corrected e1 reaches ρ=0.83" — wrong.** e1 is unmeasurable at this resolution; ρ = −0.24.
4. **Three "preprocessing widened the gap" failures were false alarms** from comparing *signed*
   gaps where −32.5 → −31.9 is a narrowing.
5. **A deliberate-failure control I designed was vacuous** — it barely moved for e8. Replaced with a
   shuffled-label permutation control, which cannot be fooled.
6. **`config_hash()` was unstable across processes** — it stringified a Python set, and string
   hashing is randomised per process. Four identical runs gave three different hashes, so the
   staleness system declared everything stale every time and had therefore never tested anything.
   Fixed by sorting sets before serialising, then verified in **both** directions: identical config
   → identical hash; one changed constant → different hash.

---

## 8. Open caveats (non-blocking)

- **Grouped-CV folds are badly unbalanced** — hash assignment puts 10 of 24 soils in one fold
  ({5, 6, 10, 3}). Needs a balanced partition before model selection.
- **Leave-one-sample-out will overstate performance.** Because of the near-duplicate curves, the
  held-out soil usually has a twin in the training fold. Grouped CV over soil families is required.
- **iPhone 14 is the weakest masking case** — ~0.82 soil fraction inside its crops; 5 images fall
  back to full frame.
- **384 px tiles discard up to ~75 mm** at the partial edge, since a 713 px short axis fits one
  tile row. Making large tiles usable needs `TILE_STRIDE_FRAC = 0.5`.
- **`config_hash` embeds absolute paths**, so moving the repo forces a full ~13-minute rebuild.
- **Whether the test soils fall inside the training size range is unresolved.** Münster and
  Testfeld Lidl WHV look like rounded gravels with cobbles and have no training twin. An automated
  check was attempted and **discarded as unusable** — at a brightness threshold the regions merge,
  so the statistic measured blob coalescence rather than clast size.
- **`soil_family` (derived from ID letter-prefix) disagrees with curve clustering.** It is advisory
  only; `cv_group = sample_id` is the safe key.
- **Display P3 for the iPhones rests on matching primaries**, not a parsed label — the H374 profile
  does literally read `Display P3` and the iPhone profiles carry identical red primaries.
- **Environment:** no `torch`, `cv2`, `skimage` or `timm` installed; local GPU is 4 GB. Model
  training is targeted at Kaggle 2×T4 (16 GB each).

---

## 9. Next phase

Preprocessing is done and verified. Modelling has **not** started, by design.

The shape the evidence points to: load tiles through the manifest metadata (never by walking
folders), aggregate image→soil with the sample as the unit of supervision, draw on the
well-sampled bands, and predict a low-dimensional curve parameterisation with monotonicity
enforced by construction — validated by grouped CV over soil families against the 82.62 floor.

---

## 10. Reproducing

```bash
python -m preprocess --dry-run     # metadata only, writes no pixels
python -m preprocess --all         # incremental; rebuilds only what changed
python -m preprocess --all --force # full clean rebuild (~13 min)
python -m preprocess.final_check   # read-only confirmation of every invariant
```

Full log of the successful run: `_look/final_run.log`.
Human-readable audit: `data/processed_meta/audit.md`.
