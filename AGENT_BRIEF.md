# MASTER CONTEXT PROMPT — Soil Grain-Size Distributions from Images (Kaggle 139732)

**How to use this file:** paste everything below the horizontal rule into a new agent as its
opening context. It is written so an agent with zero prior conversation can work on this repo
without re-deriving anything, without repeating dead experiments, and without violating the
project's own rules. Every number below was measured, not guessed, and each is tagged:

- **[CONFIRMED]** — reproduced by an artifact in this repo or an actual Kaggle score.
- **[REFUTED]** — tested and false. Do not re-propose without new evidence.
- **[UNVERIFIED]** — plausible, recorded, not yet checked. Verify before acting on it.

Rule of engagement: **trust the artifacts, not this document, where they disagree.** If you find
a conflict, report it — do not silently pick one. This project has been corrected three times by
its own measurements (see §14), and "the record was wrong, here is the measurement" is the most
valuable thing you can produce here.

---

## 1. What this project is

A Kaggle community prediction competition, id **139732**, *Predicting Soil Grain Size
Distributions from Images*, hosted by Lukas Leibold (BOKU / EU GRID project), with Enrico Soranzo.
CC BY 4.0. 165 files, 395.38 MB. ~368 teams as of 2026-09-27; ends ~late Nov 2026.

The task: from a photograph of a soil sample, predict its **complete particle-size distribution
curve** — 11 cumulative-percent-fineness values at fixed diameters — for 10 unseen test samples.

The owner is an independent ML researcher, new to research, graduating mid-2026. They run
experiments one at a time, on a laptop plus Kaggle GPU quota, and they reason in leaderboard rank
targets. They value: plain language, explicit attribution of cause, pre-registered predictions,
null results treated as results, and one self-contained runnable artifact per experiment.

**Fixed acquisition facts (host statements, from the Kaggle pages — not recoverable from the repo):**
- Camera-to-soil distance is always **21 cm**; all photos under the same lighting.
- Different phones were used **deliberately**, to force cross-device generalisation.
- Host: "It's not really possible to determine how many particles are visible."
- **H031 was removed** from the dataset (lab ran out of soil); the original set was 25 samples.
- **H374 was re-photographed** (it looked mislabelled); it is the only sample shot on
  `Motorola Edge 60 Fusion` and the only one at a different delivered scale. Treat it as suspect,
  not merely an outlier.

## 2. Leaderboard structure — read this before trusting any score

- Public LB scores a **fixed 3 of the 10** test soils. `Final = 0.30 * public + 0.70 * private`.
- The near-zero top scores (0.00001, 0.09, 0.92) are **submission cycling / public probing, not a
  leak** — the host confirmed this directly and calls a leak "close to impossible".
- Therefore: **public LB feedback is nearly worthless for tuning fine differences**, and is the
  only true held-out-camera measurement available. It is used here as a handful of calibration
  points for internal proxies, never as a tuning signal.
- Snapshot 2026-09-27 (368 teams): rank 10 = 8.15124, rank 20 = 14.10551, rank 30 = 33.73941,
  rank 40 = 37.62858, **rank 49 = 40.22430** (three teams tie). Ranks 50+ unreachable without
  sign-in. Consequence: a score of ~50 is roughly rank 55–65, **not** top 100.
- **Reading the LB page technically:** it is JS-rendered. `WebFetch` returns only analytics scripts
  and zero rows. A real browser (`browser_evaluate` on `document.body.innerText`) returns the
  visible table. The "See 319 More" expander does not respond to synthetic clicks.

## 3. The metric — and the fact that the competition page prints it wrong

Log-weighted Earth's-Mover's-Distance between the two cumulative curves:

```python
sup = np.array([0.002,0.0063,0.02,0.063,0.2,0.63,2,6.3,20,63,200]); dl = np.log10(sup)
EMD = lambda P,T: np.trapezoid(np.abs(np.asarray(P,float)-np.asarray(T,float)), dl)  # mean over samples
```

**[CONFIRMED] The grader integrates the area (trapezoid). The formula printed on the competition
page is a left-endpoint Riemann sum and is WRONG.** Evidence: the page's own published reference
value for the trivial baseline is **100.31**; the trapezoid reproduces it at **100.3132** while the
page's literal formula gives **102.02**. A submission of the constant train-mean curve then scored
102.37 on the real LB, inside the band the trapezoid predicts. The two readings differ by only
`0.5 * |error at 0.002 mm|` (<= 3.7 EMD per soil), so no historical number needs recomputing.
**Do not "fix" the metric to match the page.**

Range is effectively 0–500. Lower is better.

**Submission validity (rejected outright if violated):** one row per `sample_id` in
`data/sample_submission.csv` (that file is the authoritative id list and row order — the data page's
`train.csv`/`test.csv` are stale names for files that do not exist); all eleven columns with exact
header names; numeric in [0,100]; monotonically non-decreasing; **exactly 100 at the 200 mm column**.
Note the bare column names `"2"`, `"20"`, `"63"`, `"200"` (not `"2.0"`).

## 4. The data

```
data/Training/<sample_id>/*.jpg     24 soils, 127 images   (F827 G190 H030 H037 H038 H126 H181
                                    H183 H366 H367 H368 H371 H372 H374 H405 H493 H516 H549
                                    H615 H616 H617 H637 H666 H668)
data/Test/<site name>/*.jpg          5 sites / 10 samples, 35 images
data/Training_labels_updated.csv     the 11 cumulative columns per training soil
data/ppm_updated.csv                 pixels-per-mm, at NATIVE sensor resolution
data/sample_submission.csv           test ids + row order
data/information.md                  host's data description
data/processed_meta/                 pipeline outputs (manifests, stages, audit, masks)
data/qc/                             QC contact sheets and paired-camera images
data/training_down/ data/testing_down/  canonical-resized PNGs (written by the pipeline)
data/tiles/train/ data/tiles/test/      materialised 256 px tiles, 1976 files
```
`data/` is **gitignored** — nothing under it is version-controlled; regenerated preprocessing
output is the only record. Do not modify or rename anything under `data/Training/`, `data/Test/`,
or `data/processed_meta/`. Never re-run preprocessing for a modelling experiment.

**Camera roster (critical for held-out-camera work):**

| camera | split | images | native px | native ppm | delivered px | **effective ppm (real)** | FOV long mm |
|---|---|---|---|---|---|---|---|
| Motorola Edge | train | 69 | 4000x1800 | 11.4920 | 1600x720 | **4.5968** | 348.07 |
| Samsung A52 | train | 55 | 9248x6936 | 26.3300 | 1599x1200 | **4.5539** | 351.23 |
| Motorola Edge 60 Fusion | train | 3 | 4096x2304 | 12.4650 | 4096x2304 | **12.4650** | 328.60 |
| iPhone 14 | test | 14 | 4032x3024 | 13.9420 | 4032x3024 | **13.9420** | 289.20 |
| iPhone 16 | test | 21 | 5712x4284 | 19.5250 | 5712x4284 | **19.5250** | 292.55 |

**[CONFIRMED] `ppm_updated.csv` is quoted at native resolution and is therefore WRONG for the
pixels we actually have.** The two dominant training cameras were delivered downscaled to ~4.55–4.60
px/mm while test images are native at 13.9 and 19.5 px/mm. Any pixel→mm conversion must use the
*effective* (delivered) column. Consequences that bound the whole project:
- At 4.55 px/mm a 0.063 mm particle is **0.3 px**. Minimum resolvable grain (4-px rule):
  0.87 mm Motorola/Samsung, 0.32 mm H374, 0.29 mm iPhone 14, 0.21 mm iPhone 16.
- Individual grains in the fine tail are **physically unresolvable in train and test alike**. Only
  texture/appearance can inform it. **Particle counting is dead** (§12).
- Test images resolve *finer* than training ever did.

**Colour/EXIF facts that constrain any colour normalisation:** 38 files (all 35 test + the 3 H374)
carry an embedded **Display P3** profile; the other 124 carry no profile and decode as sRGB, so
leaving them unconverted makes train/test partly a colourspace split. EXIF is **absent on every
Motorola/Samsung file** (and 25 of the 69 Motorola files are stored portrait with no orientation
tag), so the filename is the only usable camera key.

## 5. Preprocessing — exactly what was done, and why

The pipeline is `preprocess/`, run as `python -m preprocess --all` (or `--stage NAME`, `--dry-run`,
`--force`). **11 stages in dependency order**, each persisting its own CSV into
`data/processed_meta/stage_<name>.csv`, so any stage can be re-run standalone and `verify.py`'s
golden checks act as regression tests:

```
inventory -> cameras -> geometry -> resample -> soilmask -> colour -> tiling
          -> manifests -> audit -> qc -> verify
```

`preprocess/config.py` is the **single source of every constant**. Nothing is gitignored there; the
whole pipeline is fingerprinted by `config.config_hash()`, currently
**`010f44c36c74`**, which every experiment asserts. `PIPELINE_VERSION = "1.0.0"`.

### 5.1 Canonical physical scale is DERIVED, never typed
`TARGET_PPM` is `None` in config by design and is set by `geometry.run()` as
`min(effective_ppm over all cameras and both axes)` = **4.552516 px/mm**. Anything reading it
before the geometry stage raises. The brief's proposed `4.58` (kept in config as
`TARGET_PPM_CANDIDATE`, for comparison only) **would force a 0.57 % upscale on all 55 Samsung
images**, because Samsung's effective ppm is 4.5526 long-axis / 4.5554 short-axis. Deriving it makes
every one of the 162 images identity-or-downsample and leaves 0.07 % residual cross-camera scale
error — which is precisely what lets later stages attribute any remaining difference to the camera
pipeline rather than to geometry.
- Downscale factors are computed by pairing **short side with short side, long with long**; a naive
  width/width comparison reports a bogus 122 % mismatch on the 25 untagged portrait Motorola files.
- Tolerances: `TOL_AXIS_DISAGREEMENT 0.005`, `TOL_ASPECT_MISMATCH 0.005`, `TOL_FOV_MM 0.5`,
  `TOL_PPM_REL 5e-5`, `TOL_ACTUAL_PPM 0.005`.
- Match **information content, not pixel count**. Do NOT upscale training to test resolution: a
  4.6 px/mm image contains zero information below ~0.9 mm, so upscaling teaches on mush and then
  hands the model genuinely sharp test pixels.

### 5.2 Resample (`resample.py`)
Order is **EXIF-transpose → Display P3→sRGB → canonical grid → PNG**, and that order is load-bearing.
- Target is **equal physical scale, not equal pixel dimensions**. Canvas size per image comes from
  that image's own measured FOV, so real field-of-view differences (289–351 mm) are preserved
  rather than cropped/padded away.
- Resampler is **Pillow Lanczos**. Pillow rescales the Lanczos kernel support by the ratio, so it is
  genuinely band-limited at our non-integer factors (0.9994, 0.3652, 0.3265, 0.2332). BOX is only
  equivalent at integer factors; NEAREST would alias, which `verify.py`'s stop-band check exists to
  catch. Test images therefore take a real 3.1x / 4.3x anti-aliased downsample.
- Canonical frame is **"long side horizontal", applied uniformly** (not EXIF-honoured): 32 of the 38
  EXIF-bearing files are orientation 6, so honouring EXIF would make all test images portrait while
  the EXIF-less training files are an unknowable landscape/portrait mix. A 90° rotation preserves
  the pixel multiset, so nothing is lost. `gravity_known` and `rotation_applied_deg` are recorded so
  a gravity-aligned variant is rebuildable.
- **Enlarging raises `RuntimeError`** — the pipeline refuses to invent detail.
- Colour order matters because P3→sRGB is a nonlinear tone curve plus a matrix and does **not**
  commute with a linear anti-alias filter; converting after resampling would leave the 38 profiled
  files with a measurably different effective blur kernel, manufacturing the exact camera artefact
  this phase exists to isolate. Intent is `RELATIVE_COLORIMETRIC` (a compression curve would
  redistribute contrast per image and bias texture statistics; soil colours are near-neutral so
  almost nothing clips).
- Output is **PNG** (the file on disk is exactly the array `verify.py` measured), metadata stripped
  (`exif`, `dpi`, `jfif*`, `adobe*` — propagating them after transposing would be a lie), SHA-256
  recorded.
- Clipping is measured as **introduced, not present**: Samsung's own tone curve blows ~1.6 % of
  pixels in the source, which is a property of the photo. What is asserted is whether Lanczos
  ringing created *new* saturated pixels.

### 5.3 Tray masking (`soilmask.py`) — the mask picks a rectangle, never deletes pixels
Photos show soil in a dark tray. Whether the tray is worth cropping is **camera-dependent**, measured
as illumination-relative dark-pixel fraction from the outer border band down to the interior:

| camera | outer 2 % | 6 % | 13 % | interior | verdict |
|---|---|---|---|---|---|
| Motorola Edge | 0.48 | 0.37 | 0.37 | **0.34** | no rim, tray fills frame |
| Edge 60 Fusion | 0.37 | 0.44 | 0.39 | 0.35 | no rim |
| Samsung A52 | **0.82** | 0.64 | 0.47 | 0.32 | strong rim |
| iPhone 14 | 0.68 | 0.57 | 0.44 | 0.33 | strong rim |
| iPhone 16 | **0.79** | 0.55 | 0.41 | 0.32 | strong rim |

A single fixed-percentage border crop is wrong for three of five cameras at once. Therefore:
- **Rectangle, not pixel mask.** Deleting a *pixel* needs only that pixel dark; deleting a *row*
  needs the whole row non-soil. An inscribed axis-aligned rectangle therefore cannot remove an
  isolated dark grain, a shadow, or a gap between clasts — which is the stated risk when masking
  dark soil.
- Threshold on each pixel's ratio to **its own large-scale neighbourhood**, not absolute brightness:
  a genuinely dark grain is dark vs the frame but not vs its surroundings; the tray rim is dark in
  both senses.
- Row/col keep criterion is **relative to that image's own interior density**:
  `ROW_REL_COVERAGE = 0.85`, smoothed with a `COVERAGE_SMOOTH_FRAC = 0.05` window.
  **[CONFUTED approach]** An *absolute* 0.98 per-row soil-coverage threshold was unreachable — max
  per-row coverage is 0.945 (Motorola) / 0.951 (Samsung) because soil always contains dark grain
  gaps — so it matched no rows for exactly the cameras needing crops, silently returning full frames
  for them and cropping the no-rim cameras: the inverse of correct. Cost a full pipeline run.
- Crop caps: `MAX_CROP_FRACTION = 0.30` per axis, `MAX_ACCEPT_REMOVAL = 0.40` rejection guard,
  `MIN_CROP_SOIL_FRACTION = 0.85`. Chosen because measured tray fraction ranges 14.5 % (Motorola)
  to 41.6 % (iPhone 14); a small uniform cap would leave test tiles holding ~3x the empty tray of
  training tiles. Mask computed on a `PROXY_SHORT_SIDE = 480` proxy (a crop rectangle needs mm-level,
  not px-level, accuracy). `MIN_COMPONENT_FRACTION = 0.005` drops mask blobs; Otsu is implemented
  from a histogram because scikit-image is not installed.
- **Known open cost:** the 30 % cap now binds on 2/162 images (it bound on 60/162 under the earlier
  15 % policy), so roughly 0.6 % of generated tiles were pure tray. Expected, not a defect.
- **Diagnostic invariant:** if Samsung does not crop more than Motorola, the mask is broken.
  Measured `crop_area_fraction`: Motorola Edge 0.8446, Edge 60 Fusion 0.8046, Samsung 0.6997,
  iPhone 14 0.8930, iPhone 16 0.8737.

### 5.4 Colour normalisation (`colour.py`) — split by epistemic status
- **P3→sRGB is a correctness fix. Baked in** (0 fitted parameters, no assumption about the soil
  population).
- **Gray-world white balance is a choice. Computed and STORED, deliberately NOT applied** — its gains
  depend on the soil-mask policy, so baking them would silently bake a mask decision. See
  `data/processed_meta/stage_colour.csv` columns `gw_gain_r/g/b`.
- Gains are computed **on the soil region, not the frame** (border content is camera-dependent, so
  frame-level gains differ between cameras for non-white-balance reasons), and **normalised so the
  geometric mean gain is 1** — it is a chroma correction, not a brightness correction, and must not
  rescale luma.
- Measured effect: gray-world removes the R/G/B cast from −1.08 / 0.00 / +1.38 SD to
  +0.25 / +0.24 / +0.13 SD. **It does not touch texture.**
- **Rejected colour operations, with reasons:**
  - per-channel standardisation to fixed mean/SD — measured destructive (it amplifies a noise
    channel; see §8 on e1);
  - any colour op aimed at the texture gap — the same soil differs 63–124 % between the two training
    cameras at matched scale, unchanged by colour ops; it is MTF/tone, not chroma;
  - CLAHE / histogram equalisation — manufactures mid-scale contrast the sensor never recorded and
    breaks the monotone contrast↔grain-size link.

### 5.5 Tiling (`tiling.py`)
- Tiles partition the **cropped canonical image**, so a tile's physical extent is exactly
  `tile_px / TARGET_PPM` mm in every image from every camera. A 128 px tile is a 28 mm window of real
  soil. **Nothing is ever whole-image-resized for a model.**
- `TILE_SIZES = (128, 224, 256, 384)`; **only 256 px is materialised as files**
  (`MATERIALIZED_TILE_SIZE = 256`). Manifest rows exist for all four so a size change is a config
  edit, not a re-run. Materialising all four would be ~20k files, each a redundant re-encode of
  pixels already stored losslessly.
- `TILE_STRIDE_FRAC = 0.0` — **non-overlapping, partial edges dropped.** (Consequence: at 384 px a
  Motorola's 713 px short axis fits only one row, discarding ~72 mm of soil. Overlap is the fix if
  large tiles are ever wanted.)
- Tiles are **not** filtered by soil fraction at generation time (that threshold is a modelling
  decision; regeneration costs a full pass; a filter column costs nothing). The model-side filter is
  `min_tile_soil_fraction = 0.50`.
- Stale materialised tiles not named by the manifest are pruned, so a changed mask/crop policy cannot
  leave orphans that a directory glob would pick up.

**Tile counts (all four sizes):**

| tile px | mm square | tiles |
|---|---|---|
| 128 | 28.12 | 10247 |
| 224 | 49.20 | 3245 |
| **256** | **56.23** | **1976 (materialised)** |
| 384 | 84.35 | 760 |

Materialised by camera: Motorola 716, Samsung 825, iPhone 16 240, iPhone 14 183, plus **12 tiles whose
filename yields a bare `iPhone` token** — an irregularity to handle, not assume away. Tiles per soil:
train median 64, test median 36, so per-soil feature noise is ~1.3x higher on test.

### 5.6 Verification, audit, and invalidation
- `audit.py` emits `data/processed_meta/audit.md` / `audit.json`: currently **37/37 assertions pass**.
- `verify.py` (502 lines) holds the golden checks and the canonical feature definition (§6). It also
  contains a **`grating_check()`**: a synthetic 2 mm and 20 mm grating pushed through the geometry
  maths, which validates the code rather than the data and catches a ppm ↔ mm-per-pixel inversion that
  every relative check would silently agree with.
- `config.record_stage_done()` stamps each finished stage with its `config_hash` +
  `PIPELINE_VERSION`; `run.py` then rebuilds **any upstream stage whose recorded hash differs**.
  Without that, editing e.g. `MAX_CROP_FRACTION` and re-running `--stage manifests audit` would
  regenerate metadata over stale PNGs and report a clean audit over a corpus whose pixels and
  metadata describe different pipelines.
- **[CONFIRMED] `config_hash` was not stable at first.** Python randomises string hashing per
  process, so `str({...})` over a set came out in a different order across runs — four consecutive
  identical-config processes produced three different hashes, which made every stage look stale and
  meant the invalidation machinery rebuilt everything and tested nothing. Fixed by `_stable()`, which
  sorts sets/dicts before serialising. Lesson: any fingerprint over a collection must sort first.
- `config.assert_writable()` refuses any write resolving inside `data/Training/`, `data/Test/`, or
  onto the three input CSVs — the "source data is untouched, enforced not assumed" rule.
- **Preprocessing is signal-neutral [CONFIRMED]:** Spearman ρ of raw vs canonical features is
  identical to three decimals for e4, e8, e16.

## 6. Feature extraction — the hand-built model features

**Definition source of truth: `preprocess/verify.py::feats()`.** Model 1's extractors import it
rather than reimplementing it, because a silently divergent reimplementation is the easiest way to
make an experiment unrepeatable.

```python
BAND_SIGMA_MM = [0.22, 0.44, 0.88, 1.76, 3.52]   # e1 e2 e4 e8 e16
BASE_SIGMA_MM = 12.0                              # illumination base
```
- `g = RGB.mean(axis=2)`; soil mask `m = g > percentile(g, 12)` — the darkest 12 % (tray/shadow) is
  excluded from every statistic.
- `base = gaussian_filter(g, 12 mm * ppm)`; residual `e = (g - base) * m`.
- **Laplacian-pyramid octave bands:** `band_k = low_k - low_{k+1}` where `low_k` is `e` smoothed to
  `sigma_mm * ppm`; `e_k = std(band_k[m])`. (A earlier version subtracted **variances** in the wrong
  order, so every band was identically zero — the coarser low-pass always has less variance, making
  `var(fine) - var(coarse)` negative and floored to nothing. If bands ever read zero, check that.)
- `sat = mean((max_c - min_c) / (mean_c + 1e-6))` over soil pixels.
- `lum_p10/p50/p90/sd` = percentiles/std of masked luminance;
  `grad_mean = mean(|∇g|)` over masked pixels (edge density per px);
  `spec_centroid_cpm` = rotation-averaged radial power-spectrum centroid in cycles/mm
  (`spectral_centroid`), and `dom_wavelength_mm = 1 / cpm`.

**Feature families and what each was for:**

| family | columns | intent | status |
|---|---|---|---|
| texture bands | e1, e2, e4, e8, e16 | DoG octave energies at 0.22/0.44/0.88/1.76/3.52 mm | e4/e8/e16 usable; **e1/e2 diagnostic-only** |
| colour | R, G, B, sat | channel means + saturation; mineralogy/moisture co-vary with texture | kept (§9) |
| intensity | lum_p10, p50, p90, sd | brightness distribution of soil region | split across arms |
| gradient | grad_mean | finer grains pack more edges per mm | kept |
| absolute frequency | spec_centroid_cpm, dom_wavelength_mm | dominant texture wavelength, the most direct physical proxy for grain size | **REMOVED in E3 — costs ~27 EMD** |
| geometry | soil_fraction, crop_area_fraction | capture/mask QUALITY, flagged as a possible shortcut | **EXCLUDED — structurally degenerate** |

**Live feature sets:**
- E1 = 16 features. E2 = 14 (dropped `soil_fraction`, `crop_area_fraction`).
- **E3 / Model-2 anchor `M1` = 12** = texture core 5 `[e4, e8, e16, lum_sd, grad_mean]`
  + colour 7 `[R, G, B, sat, lum_p10, lum_p50, lum_p90]`.
  (Verified against the header of `Model 1/Model 1 Experiment 3/features_soil.csv`.)

**Per-soil aggregation — part of the frozen protocol, not an implementation detail:**
1. exclude tiles with `soil_fraction < 0.50`;
2. **median over tiles → per-image** value;
3. **median over images → per-soil** value.
Two-stage median, tile → image → soil. **[CONFIRMED]** One-stage vs two-stage moves CAM+RES for the
same matrix from 75.7 to 85.0, so any change here makes experiments incomparable. E3 once silently
aggregated tile → soil×camera → soil (pooling cameras before images); the reproduction gate (§7.4)
caught it.

## 7. How experiments are actually run here

### 7.1 The Model / Experiment hierarchy
- **Model** = a change of *kind*. **Experiment** = a limited, recorded set of variables changed inside
  that kind.
  - **Model 1** classical hand-built features + closed-form ridge. **CLOSED after E4** (promotion gate
    fired).
  - **Model 2** learned visual representation (frozen backbone) under the same head. E1, E2 done and
    both **REFUTED**; E3 planned, not run.
  - **Model 3** changed output parameterisation or task framing. **Model 4** changed aggregation /
    pooling as its own subject. **Model 5** explicit domain adaptation.
- **Promotion test:** if it can be described as "same model, different features", it is a new
  *experiment*. If it changes what the model fundamentally **is** — what it outputs, or how tiles are
  pooled — it is a new *Model*. A change of measurement instrument is **not** a change of what the
  model is (that is why replacing LOGO-CV with CAM+RES stayed inside Model 1).

### 7.2 The frozen head (identical across Model 1 E1–E4 and Model 2 E1–E3)
`StandardScaler` on features → **rank-3 PCA curve basis** → closed-form multi-output ridge on the
curve coefficients → reconstruct mean curve + 3 coefficients → **monotone projection: clip to
[0,100] → cumulative max → force exactly 100 at 200 mm.** The projection is load-bearing (a raw
rank-3 reconstruction breaks monotonicity on 9/24 samples and leaves [0,100] on 9/24) and costs a
little. Scaler and basis are **refit inside each training fold only**.
- Alpha grid: `(0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)`.
- `n_boot = 4000`, `n_perm = 200`, `ORDINAL_FLOOR = 15.0`.
- Seeds: E1 `20260930`, E2 `20260931`, E3(planned) `20260932`. **`CFG.embed_seed` stays 20260930** —
  the harvested Kaggle embedding cache filenames contain it; changing it does not just re-key the
  cache, it silently destroys E3's premise (the arrays would no longer be E2's).

### 7.3 Validation design and leakage rules
- **No random image-level split. The soil sample is the unit of supervision.**
- **Leave-One-Family-Out CV over 16 curve-distance families spanning the 24 soils.** Families are
  built deterministically inside the notebook by average linkage on pairwise **label-curve EMD**, cut
  at 14. LOF is used for **reporting**; it is never used to choose (§8).
  Do not use `sample_id` letter prefixes (F8, G1, H0…) as the group key — that grouping does **not**
  agree with the curve-distance clustering. `cv_group` in the tile manifest is `sample_id`.
- **No camera, ppm, EXIF, ICC, site or sample-id features as model inputs.** They identify the device
  or the sample; they inflate validation and are worthless on the private split.
- **Any column with near-zero training variance is excluded** (rule from E2 onward). Such a column is
  unlearnable in-domain yet free to move out-of-domain, so its fitted coefficient is a pure noise
  amplifier. Measured case: `soil_fraction` training std **0.00027** (mean 0.9999, range 0.0012) but
  reaches 0.829 on test — a **−63.9 SD** excursion. `crop_area_fraction` fails the same way, milder
  (8/10 test soils outside training range). Dropping both cut camera transfer 195.95 → 59.68 at
  alpha 0.03 and test saturation 59 % → 13 %.
- **Circularity rule:** no criterion may select on the same measurements it is judged by. Alpha is
  chosen inside each outer fold from inner folds only, so the reported number is the score of the
  **procedure**, never of the best alpha. Oracle-alpha comparisons across different feature sets
  manufacture differences that are not there.
- **The training-distribution circularity rule (the project's most transferable finding):** any
  treatment that **alters the training distribution** must be evaluated on a **held-out level of the
  nuisance it targets**. A ruler whose evaluation condition equals the treatment condition is circular
  and will report the treatment as free. Discovered, not planned: training on res16-blurred views and
  scoring with a ruler that also targets res16-blurred views made the frequency features E3 had
  removed look **helpful** (−3.08 EMD) when held-out blur says they cost **+35 to +58**. A **34-EMD
  artefact** that would have reversed a correct decision.

### 7.4 The reproduction gate (keep this in every experiment)
Each experiment rebuilds its predecessor's feature matrix from tiles and **asserts drift < 1e-6**
(measured **2.842e-14**). This has caught a real bug (E3's tile→soil×camera→soil aggregation, vs
E1/E2's tile→image→soil). Without it, two experiments are silently incomparable. Model 2 E2's
crop-256 arm likewise reproduced E1's M1 mean to **0.0017 EMD** before any E2 conclusion was drawn.
**Instrument proven, then hypothesis tested — in that order.**

### 7.5 Pre-registration and decision gates (non-negotiable method)
Every experiment plan is written **before** it runs, as a numbered spec at the repo root
(`model_plan*.md`), and must contain:
1. **Hypothesis H** and the one variable changed;
2. **Controls**, each with a stated reason it cannot be dropped;
3. **Pre-registered predictions P1…Pn**, each marked `blind = yes/no` — written before the score
   existed and never revised afterwards;
4. a **decision gate as code**, executed before any interpretation is written, whose failure means
   **the submission is not written at all**;
5. **explicit "what closes on which outcome"** table, so a null result is a finished experiment;
6. **cost** estimate.

Where an arm's real value cannot be measured internally, a **declared probe** is taken: a submission
spent to measure one specific question, labelled as a probe with its prediction written first — never
as a gate pass. Two probes so far (Model 1 E4's no-colour arm, Model 2 E1's arm D).

The honest prior is stated and is often *against* the hypothesis (Model 2 E3's P3 predicts its own
H3 is **refuted**). Predicting your own defeat is what makes the test real.

### 7.6 Artifacts and folder discipline
```
Model N/Model N Experiment M/
  instructions.txt          what this Model/Experiment means, its frozen surface, its results
  kaggle_setup.md           upload plan (see §11 — Model 1's were NEVER executed)
  Experiment{M}.txt         full run record: context, arms, sections, numbers, verdicts
  ModelN_ExperimentM.ipynb  the notebook
  gate.json                 machine-readable gate outcome
  predictions.csv           pre-registered predictions vs outcomes
  cv_families.csv cv_per_fold.csv cv_per_soil.csv paired_effects.csv mechanism_table.csv
  features_soil*.csv .cache/tile_features_<size>_<config_hash>.csv   (TRACKED: evidence for 7.4)
  Submission_ModelN_EM.csv  written only if the gate passes
  kaggle run 1 files/       authoritative downloaded Kaggle outputs (Model 2 E2)
  local mock dry-run/       plumbing only, with a README saying so — identical filenames to real runs
```
- Never overwrite a previous experiment's results. **Never delete a failed experiment.**
- A null result is a result and must be reported as one.
- Submissions are generated by the notebook under their final name
  (`Submission_Model2_E3.csv`), never renamed from a generic file; test ids come from
  `manifest_samples.csv` (`submission_id`, `submission_row_order`), never constructed from folder names.

### 7.7 Notebook generation (there is no Jupyter locally)
`.ipynb` files are **generated from a Python builder module** (`scratch/build_m1e1.py` …
`build_m2e2.py`) that holds the cell text; the builder is the source of truth and **hand-editing the
generated notebook is lost on the next regeneration**. The same code is flattened into a single
runnable `.py` (`scratch/run_*.py`) and executed headless with matplotlib on the Agg backend. That
flattened `.py` is the locally-executed artifact.

## 8. The selection instruments — the single most important section

**[CONFIRMED] Leave-one-family-out / leave-one-group-out in-domain LOGO-CV ranks ridge alpha
BACKWARDS.** E1 chose `alpha = 0.03` because LOGO CV said it was best, at CV **35.98**. That
configuration scored **172.70** on Kaggle against **102.37** for a submission that ignores the images
entirely — a 69 % loss to no-information.

| alpha | LOGO CV | transfer M→S | transfer S→M | % test cols pinned at 0/100 |
|---|---|---|---|---|
| **0.03** ← CV's pick | **36.03** | 96.9 | **136.6** | **59.1 %** |
| 1 | 40.77 | 80.8 | 137.7 | 30.9 % |
| 10 | 47.15 | 56.3 | 125.7 | 21.8 % |
| 100 | 65.07 | 65.1 | 92.4 | 17.3 % |
| 1000 | 87.32 (worse than no-image) | 88.2 | 80.8 | 16.4 % |

Across the whole grid, rank correlation between LOGO-CV and the replacement ruler is **−0.953**.
*Why:* within the training set the colour features co-vary with the soil (camera is nested inside
sample), so a near-unregularised linear fit on 16 features × 24 rows is genuinely good **in domain**
and amplifies noise catastrophically **out of domain**.

**CAM+RES** (the replacement ruler): fit on one training camera's real view of the non-held-out soils,
predict the **other** camera's view after an added anti-alias blur matched to the test cameras'
downsample factor, with the held-out soil's **LABEL excluded from the fit** (the notebook asserts this
per condition). 21 of 24 soils are shot by both Motorola Edge and Samsung A52;
`Motorola Edge 60 Fusion` (3 images, H374 only) is excluded. Evaluated in both camera directions and
against both test resolution targets = 4 conditions, with **paired per-soil differences and a
bootstrap that resamples SOILS, not observations** (the four conditions share the same 21 soils;
resampling observations gives falsely tight CIs; CIs are ±15 to ±35 EMD wide).

**[CONFIRMED] CAM+RES IS DEMOTED. It is an ORDINAL instrument and can no longer even rank reliably.**
Bias factor versus actual Kaggle score, at the four externally measured points:

| config | ruler reading | actual Kaggle | factor |
|---|---|---|---|
| M1 E1 (16 feats, alpha by CV) | 205.34 | 172.70 | 0.84 |
| M1 E2 (14 feats) | 75.73 | 71.27 | 0.94 |
| M1 E3 (12 feats) | 51.87 | 61.24 | **1.18** |
| M1 E4 (5 feats, no colour) | 56.98 | 77.12 | **1.35** |

- Ranking held 3/3 through E3, then **E4 broke it outright**: the ruler said E4's 5-feature arm was
  18.75 EMD *better* than E2; reality said 5.85 EMD *worse* (1 of 6 measured pairs ranked backwards).
- Magnitude forecasting failed: E3's pre-registered band was 44–49, actual 61.24.
- Marginal gains are inflated, and the inflation **grows as models improve** (E1→E2 ratio 0.78,
  E2→E3 ratio 0.42) — the signature of a proxy that models the nuisances it was built from and goes
  blind once those are removed.
- Its blind spot is specific: its only two colour-relevant nuisances are the Motorola–Samsung offset
  and a Gaussian approximation of a Lanczos prefilter, and **neither reproduces whatever makes colour
  informative on the iPhones. It under-penalises REMOVING colour.**

**BINDING RULES that follow:**
- Never select or compare configurations on LOGO-CV alone.
- **No experiment may be SELECTED by CAM+RES alone.** Use it to generate hypotheses and to reject
  configurations it calls catastrophic (it never missed a catastrophic one). Treat any finer ordering
  as a hypothesis only a submission can test. Spend submissions on **well-separated** candidates.
- It must not forecast a score, and must not choose between candidates differing by less than
  **~15 EMD**. Write expectations as "better or worse than X", never as a number.
- Any internal number in the 30–50 range is "unknown", not "good".
- **Exception, deliberately argued:** Model 2 E1/E2/E3 use **nested in-domain LOGO-CV as the primary
  selection metric**, because E1's claim was narrow — LOGO-CV is invalid for *predicting the
  leaderboard under camera shift*, not invalid for measuring how much curve signal a representation
  carries on devices already seen, and that quantity was the only thing constant across all four
  Model 1 experiments. It may decide an arm; **it may not forecast a Kaggle score.** CAM+RES is
  reported for every arm as a diagnostic and a further calibration point, never used to select.
- **Per-camera colour standardisation is structurally unmeasurable internally:** it removes exactly
  the between-camera difference the CAM ruler measures, so CAM would rate it near-perfect by
  construction. Score it only on CAM+RES and held-out blur, and settle its real value on the LB.
  It also needs to know which camera made each test image; the filenames do carry it, but camera
  identity is a banned input, so it must be run twice — true identity, and unsupervised clustering
  that never reads the label. **If it only works with identity, reject it on principle even if it
  scores.** (E4 measured cluster purity at **78.2 % for k=5** — unsupervised grouping does *not*
  recover the devices.)

**Alpha parsimony rule** (adopted before a score, not after): where a **larger** alpha lies within the
ruler's own 15-EMD resolution of the grid optimum, prefer it — the failure mode of too little
regularisation is known and catastrophic (E1), while the failure mode of too much is merely a worse
score. E4 used alpha 30 on this basis.

## 9. Results ledger — every externally measured number

**Model 1 (ran LOCALLY; only the CSV was ever uploaded to Kaggle):**

| experiment | what it was | in-domain LOGO-CV | CAM+RES | Kaggle |
|---|---|---|---|---|
| E1 | 16 features, alpha by LOGO-CV | 35.98 | 205.34 | **172.69929** |
| — | constant train-**mean** curve, all 10 rows (ignores images) | 85.39 (train) | — | **102.37237** |
| E2 | 14 feats, alpha by CAM+RES (instrument calibration) | 42.66 | 75.73 | **71.27346** |
| E3 | 12 feats, frequency family removed | 41.15 | 51.87 | **61.23560** |
| E4 | 5 feats, "no colour" arm — **declared probe** | 39.46 | 56.98 | **77.12257** |

**Model 2 (ran on Kaggle, 2x T4):**

| experiment | what it was | in-domain | CAM+RES (report-only) | Kaggle |
|---|---|---|---|---|
| E1 `M1` | 12 hand features (anchor) | 43.0217 | 54.98 | 61.24 (E3's file) |
| E1 `D` | frozen DINOv2 ViT-S/14, 384 dims | **44.3302** | 47.08 | **60.56167** ← best, **declared probe** |
| E1 `R` | same ViT, **random untrained weights** | **39.7769** | 78.55 | never submitted |
| E2 `D@256/128/64/32` | patch-magnification sweep | 44.33 / 43.32 / 43.83 / 44.87 | 54.01 / **51.33** / 53.51 / 58.59 | no submission (gate failed all clauses) |

Baselines on the 24 training soils: official trivial baseline (9.09 % equal mass per bin)
**100.31**; copy another training sample's curve 104.22; naive 2D connected-component granulometry
~**198**; train **mean** curve 85.39; train **median** curve **82.62** ← the "ignore the images
entirely" floor. **A real model must land well under 82.** Perfect = 0.

**Findings by experiment:**

- **M1 E2** — instrument calibration, not a model experiment. Compared LOGO-CV against three
  camera-based rulers on E1's unchanged model; LOGO-CV was the only one of four that got the one
  externally-checkable comparison wrong. Selected CAM+RES; dropped the two degenerate columns.
- **M1 E3** — 2×2 factorial {colour} × {absolute frequency} on a fixed 5-feature texture core,
  paired soil bootstrap across 4 conditions:

  | effect | estimate | 95 % CI | verdict |
  |---|---|---|---|
  | **MAIN frequency** | **+26.76** | **[+14.81, +39.07]** | significant (in 4/4 conditions: +21.0, +38.5, +23.4, +20.9) |
  | MAIN colour | −4.29 | [−14.64, +5.27] | **not distinguishable from zero** |
  | interaction | −1.64 | [−6.63, +3.17] | additive — the 2×2 was the right design |

  Selected cell C (core + colour, 12 features), alpha 10; paired advantage over E2 −27.6 EMD,
  CI [−39.5, −16.9], wins 18/21 soils. P4 CONFIRMED, P5/P6 REFUTED.
- **M1 E4** — colour-representation factorial, six arms on a fixed texture core. Nested CAM+RES:
  C0 raw 51.87, C1 none 56.98, C2a identity 49.57, C2b clustered 59.34, C3 chromaticity 62.63,
  C4 colour-only 82.13. **C1 vs C0: +5.11 EMD, CI [−6.23, +17.31], wins 12/21 — NOT RESOLVABLE, and
  the pre-registered P2 predicted exactly that null (CONFIRMED).** The selection rule returned the
  control, so the submission was redirected to C1 as a discriminating probe; it scored **77.12**,
  15.89 EMD worse than E3. **Colour earns its place; it stays.** P1/P2/P4 CONFIRMED, P3/P5 REFUTED.
  This is the experiment that demoted the ruler.
- **M2 E1** — the Model-2 premise (a pretrained ViT carries grain-size information the hand features
  miss) is **REFUTED as a replacement claim**: D 44.33 vs M1 43.02.
  - **The R trap.** Random-init weights scored 39.78, the best in-domain number in the project's
    history, and means nothing: (1) not significant (R−M1 = −3.24, CI [−12.53, +6.06]); (2) optimum at
    the **grid floor** alpha 0.03 with the curve rising monotonically under shrinkage — exactly E1's
    fatal configuration; (3) **worst** arm under transfer (78.55); (4) mechanism: its between-soil
    cosine distance is **0.0000**, so `StandardScaler` amplifies residual noise to unit variance and
    an unregularised ridge fits it. **Best in-domain + worst out-of-domain = overfit, not discovery.**
  - **D's durable finding: robustness and informativeness are independent.**

    | | camera shift z | camera/soil distance | blur/soil distance |
    |---|---|---|---|
    | M1 colour block | 1.541 | 1.105 | 0.818 |
    | R | 1.262 | degenerate (ratio of ~0s) | 0.108 |
    | **D** | **0.620** | **0.209** | **0.029** |

    Pretraining buys ~5x camera invariance and ~28x blur invariance **and zero extra grain-size
    signal.**
  - **The probe:** D's curves were genuinely different from E3's (mean 2.46 pp, max 13.10 pp,
    curve-to-curve EMD 13.40, no row agreeing within 1 pp), so 60.56167 is not E3 re-scored. But
    **0.674 EMD on a fixed 3-soil public set is a tie.** The informative reading is the transfer
    penalty: M1 goes 43.02 → 61.24 (+18.2); D goes 44.33 → 60.56 (+16.2). Two points on three soils
    *motivates* an invariance direction; it does not establish one. **Do not quote 60.56 as evidence
    that pretrained features work.**
  - Determinism evidence: run 2 reproduced run 1's entire primary table and the E3 reproduction line
    **byte-identically across two separate Kaggle sessions**.
- **M2 E2** — magnification rescue **REFUTED**. D lost to R at **every** crop (256 +4.55, 128 +0.96,
  64 +3.14, 32 +6.24; best case +0.96, no CI near excluding zero). No trend: Spearman ρ = −0.40,
  p = 0.60 for D, opposite sign for R. The 256→32 change was **not** concentrated in the
  patch-limited fines (fines −0.68, mid −2.88, coarse +2.13) — a systematic gain in the fines would
  have meant the crop was smuggling label information. **E2's machinery is valid (both regression
  checks passed); E2's hypothesis is dead.** Nothing may inherit any belief that magnification helped,
  and crop 128 is **not** carried forward as a preference despite being D's best crop — using CAM+RES
  to select it is itself cancelled (§12).

## 10. What the label space actually is (drives the output head)

PCA on the 24 label curves — best achievable EMD if you predict *k* numbers **perfectly**:

| numbers predicted | 1 | 2 | 3 | 4 | 11 (free) |
|---|---|---|---|---|---|
| best achievable EMD | 20.5 | 11.0 | **7.4** | 4.4 | 0, but needs ~1000 samples |

**PC1 carries 91.4 % of label variance and correlates 0.995 with log10(D50);** PC2 6.2 %, PC3 1.4 %.
So this is *"regress median grain size, plus two small shape corrections"*, not eleven free outputs.
With 24 labels, 3–4 numbers is tractable and 11 is not. Predict a low-dimensional parameterisation
(mean curve + 2–3 PCs, or D50 + sorting coefficients) and reconstruct. Note the parametric family's
own floor (~10–12 EMD for a 2–3 parameter fit) is at the level of a competitive LB score, so shape
residuals matter — skeleton + low-rank correction beats a pure 2-parameter model.

The curves are smooth **by construction**: labels come from sieve and sedimentation analyses whose
raw curves were interpolated linearly on log10(diameter) onto the 11 support points. They are not 11
independent measurements. (Sedimentation reaches clay sizes, so the fine tail is real lab data — but
still unresolvable from images.)

**Structure:** two families, recovered cleanly by k=2 clustering — **11 fine-grained** soils
(D50 0.021–0.21 mm, silt/clay/fine sand) and **13 coarse** (D50 1.8–6.1 mm, gravel). Near-duplicate
curve pairs: H366~H371 (EMD 3.3), H367~H372 (4.3), H368~H615 (7.0), H038~H637 (7.3), H549~H668 (6.8),
H181~H183 (8.2), H493~H666 (8.4). **Effective number of distinct soils ≈ 8–10, not 24** — hence the
family-based CV. Per-sample difficulty against the median baseline varies hugely:
H037 = 26, H637 = 33, H038 = 36 vs **F827 = 160, H405 = 151, H549 = 137** — the silt/clay samples are
where the no-image answer fails worst.

Training D50 is **bimodal**: ten soils at 0.021–0.1035 mm, only two between (0.211, 0.814), twelve at
1.795–6.126 mm. Any per-group effect must be reported: **the mean over all 24 soils is the wrong
headline number.**

**Noise floor at n = 24 — applies to every claim in this project:** with shuffled labels,
|Spearman ρ| reaches ≈ **0.52** (2.5/√(n−1)); observed permutation maxima 0.43–0.55.
**Any correlation below ~0.5 here is indistinguishable from chance.**

## 11. Which findings were WRONG first — read before trusting any of the above

`Model 2/.../` and `preprocess/` records contain superseded claims. The corrected picture:

| claim | status | actual |
|---|---|---|
| "camera explains up to 71 % of texture variance" | **WRONG** | ~1.5 % for the bands that carry soil signal |
| "camera-corrected e1 reaches ρ 0.83 with D50" | **WRONG** | e1 is unmeasurable; ρ = −0.24 |
| "same soil differs +63 % to +124 %, Samsung always higher" | **WRONG** | a crossover; Samsung higher in 1/23 fine bands |
| "saturation differs +74 %" | HOLDS | +74.4 %, 21/21 same sign |
| "colour carries the entire train→test shift, so drop colour" (E2's lever) | **WRONG as an action** | colour main effect on transfer −4.29, CI [−14.64, +5.27] |
| "CAM+RES may rank configurations" | **WRONG (E4)** | ordinal at best; ranked 1 of 6 measured pairs backwards |
| aliasing/sampling history explains the residual texture shift | **REFUTED** | synthetic matched low-pass induces only e4 −0.27, e8 −0.02, e16 +0.02 SD — ~2.4x too weak on e4, wrong sign on e16 |

**Root cause of the biggest one:** the exploratory feature used
`sigma = min(image_shape) // 12` for the illumination base — **frame-dependent** (59 px on a canonical
Motorola frame, 99 px on a Samsung frame), so every band carried a per-camera offset unrelated to
soil or camera. Fixed by defining bands in **millimetres** (`BAND_SIGMA_MM`), which is why the corpus
was put on a common physical scale first.

**The counter-intuitive lesson, stated plainly:** colour moves **most** of the three families under
camera change (1.54 z) yet removing it changes the score by nothing; the absolute-frequency features
barely shift on average yet cost ~27 EMD. **A feature can shift a lot and not be load-bearing once
the model is regularised.** Chasing the largest domain shift was the wrong heuristic; measuring the
**paired effect of removing the family** was right. Use the paired removal effect, never the size of
the shift.

**Domain-shift location (train → test, range-normalised, not z-scored):**
texture transfers cleanly — **1 of 70** test feature cells outside the training envelope, excess
0.00x range — vs **35 of 90** for colour+geometry (B **+2.63**, lum_p90 +1.34, sat **−1.50**,
crop_area_fraction +1.25, G +1.09). That is direct evidence the canonical-PPM resample,
P3→sRGB and gray-world balance did their job.
**Beware z-scores on near-constant columns:** an early version reported "max |z| = 641" as evidence of
extrapolation; that was entirely `soil_fraction`'s divide-by-noise. Use **excess-beyond-training-range
normalised by the training range**.

**`kaggle_setup.md` warning:** every Model 1 experiment folder contains a `kaggle_setup.md`
describing a `soil-gsd-processed` dataset upload. **That flow was NEVER used.** Model 1 E1–E4 ran
locally as a flattened `.py`; only the resulting CSV was uploaded on the submission page, and all five
Model-1 scores came from local runs. Reading one of those files as evidence of past actions produces
false premises — it already caused an invented data-loss risk that did not exist. There is no
pre-existing Model 1 Kaggle state to preserve.

## 12. Cancelled hypotheses — check here before proposing anything

| idea | status | evidence |
|---|---|---|
| Resolution-matched training (fit on test-blurred training tiles) | **CANCELLED** | all nine train-blur × test-blur cells span only **9.2 EMD**; matched training buys at most **2.3** on the deployment column, and the fully-matched row (53.51) is *worse* than partially-matched (51.54). E3's feature selection already absorbed the sharpness problem. (For the 14-feature set the matrix IS perfectly diagonal and real→res16 explodes to 105.18.) |
| Richer tile→soil aggregation | **CANCELLED** | median 12 cols → LOGO-CV **41.15**; median+mean+sd 36 cols → 42.13; +p10/p90 60 cols → **42.22**. Quintupling the space makes it slightly worse. Physics right, statistics at n=24 do not support it. |
| Aliasing explains the texture shift | **REFUTED** | see §11 |
| Particle counting / granulometry | **REFUTED** | connected components on 2D projected areas ≈ **198 EMD** vs ~100 baseline; 0.063 mm = 0.3 px; host agrees |
| Making the backbone patch **smaller** to rescue a pretrained representation | **CANCELLED (M2 E2)** | D loses to its own random-init twin at all four crops, no trend, only 2 of 24 soils newly resolvable |
| **Choosing a crop, or any arm, using CAM+RES** | **CANCELLED** | the ruler is ordinal-only and demoted; a coincidence (D's best crop = D's best CAM+RES) must be ignored, not followed |
| Per-camera colour standardisation | **not dead, but structurally unmeasurable internally** | see §8; must be run with and without camera identity |
| Band **ratios** (e4/e8) as blur-robust frequency surrogates | **cancelled from Model 1, weakly** | the frequency family alone scored LOGO-CV **97.7** — worse than ignoring images — so the ceiling on any ratio of it is low |
| Alternative classical regressors (robust, boosted trees) | **CANCELLED** | 24 samples, 12 features; a boosted tree overfits, Huber is second-order on a model dominated by the curve basis |

**What is NOT dead:**
- **In-domain headroom of ~34 EMD** (41.15 measured vs a rank-3 ceiling of **7.35**). Not reachable by
  re-summarising the same hand-built features — which is what motivated Model 2.
- **Whether DINOv2's 384 columns carry information the 12 hand features lack, tested by
  CONCATENATION rather than replacement.** This is Model 2 E3, the only reading of E1's invariance
  finding that leaves it mattering.
- Model 3/4/5 territory: output parameterisation, pooling as a subject, explicit domain adaptation.

## 13. Current state (as of 2026-09-29)

- **Model 1: CLOSED.** Best configuration to date: 12 features (5-feature texture core + 7 colour),
  rank-3 curve basis, ridge alpha 10, Kaggle **61.23560**.
- **Model 2: E1, E2 and E3 ALL REFUTED. MODEL 2 IS CLOSED AT THREE EXPERIMENTS.** E3 ran
  locally on 2026-09-29, CPU only, no Kaggle session, and produced no submission.
  - Spec: `model_plan_m2e3.md` (repo root) — read it in full before touching Model 2. Contract:
    `Model 2/Model 2 Experiment 3/instructions.txt`.
  - **What exists and is executed [CONFIRMED]:** `fusion.py` + `check_fusion.py` (33 assertions,
    `CHECK FUSION: PASSED`), `embedding_store.py` + `scratch/verify_e2_embeddings.py` +
    `scratch/check_e3_loader.py` (5 corruption classes fabricated and all 5 caught, repairs
    verified), `scratch/build_m2e3.py` → `Model2_Experiment3.ipynb` (47 cells, 35 code;
    **all 35 compile independently**; `scratch/mk_run5.py`), `scratch/run_m2e3.py`, and
    `scratch/run_nb_m2e3.py` (executes the shipped .ipynb, not the builder's output).
  - **E3 RESULT [CONFIRMED by execution]:** H3 **REFUTED**. Nested in-domain LOGO-CV,
    `fuse@c256` **43.43** vs `M1` **43.02** → fusion is 0.41 EMD *worse*, paired CI
    [-6.69, +7.46], 11/24 soils improved. Gate: G1 FAIL, G2 FAIL (FUSE−SHUF −9.50,
    CI [−23.56, +4.82], ns), G3 PASS, PROVISO PASS (oracle alpha 300, interior), VALID PASS,
    COMPLETE PASS → verdict `REFUTED`, `submit=False`, **no submission written** (expected:
    P3 predicted exactly this). Sensitivity crops `fuse@c128` 43.65, `@c64` 44.01, `@c32`
    44.76 — all worse than M1, so magnification does not rescue fusion either.
  - **The two findings worth carrying forward:**
    (1) `shuf@c256` (hand + row-permuted D block) scored **52.93**, 9.90 EMD worse than M1 —
    the permutation control behaved exactly as pre-registered, so the instrument was healthy
    and the null is a real null, not a broken ruler.
    (2) `fr@c256` (hand + **random-init** ViT block) scored **39.26** — the best in-domain
    number in the experiment — while its CAM+RES was **83.94**, the worst transfer of any fused
    arm. E1's capacity trap reproduced in fused form: in-domain CV prefers untrained random
    projections over pretrained features. **Do not read a 39-class in-domain number as a win.**
  - **E1's invariance did NOT survive fusion:** `fuse@c256` keeps D's *absolute* camera shift
    (0.64 vs D 0.62, M1 1.21) but its camera/**soil** ratio is 1.1044, indistinguishable from
    M1's 1.1048. P5 is marked CONFIRMED yet is near-vacuous — it lands inside its interval only
    by coinciding with M1. Flagged in `Experiment3.txt` section 12b.
  - **Two gate-code bugs found from run 1's own output and disclosed in the record:** VALID was
    coded with an inverted sign (so run 1 mislabelled a healthy instrument as INSTRUMENT
    INVALID), and the FUSE−SHUF contrast was bound to `None` (G2 evaluated nothing). Both were
    corrected to match the clause text **frozen in `instructions.txt` before the run**; no
    clause, threshold or comparator changed, and the verdict is identical under both versions
    because G1 fails either way. This is disclosed, not quietly fixed.

  - **15 of 35 code cells are E2's cell text imported byte-for-byte out of
    `Model2_Experiment2.ipynb`** (A2, A4, B1-B3, C1, F1, F3, G2, H1, H2, M1, N1, P2, Q3), so
    "the head/CV/metric are identical to E1-E2" is a fact rather than a promise. The builder
    prints the provenance table; E3's `Experiment3.txt` records it.
  - **H3:** concatenating 12 hand features + 384-dim DINOv2 embedding beats the 12 alone by more
    than the same concatenation with a **label-blind row-permuted** block, at crop 256.
    **P3 pre-predicts H3 is REFUTED.** Gate: G1 (FUSE<M1, CI excludes 0) ∧ G2 (FUSE<SHUF, CI) ∧
    G3 (FUSE<D) ∧ PROVISO (oracle alpha strictly inside the grid, else INCONCLUSIVE) ∧ VALID
    (SHUF must not beat M1, else INSTRUMENT INVALID) ∧ COMPLETE (M1/D/R present).
    `probe_arm = None` — E3 pre-authorises **no probe**.
  - Nine arms: `M1` 43.0217, `dinov2@c256` 44.3302, `vit_random@c256` 39.7769 (all three must
    reproduce ±0.05 in cell H3), gated `fuse@c256`, controls `shuf@c256` and `fr@c256`, and
    `fuse@c128/64/32` as **sensitivity only — cannot fire the gate**. R@256 was added because
    `Model 2/instructions.txt` makes M1+R+D a standing requirement, not optionally.
  - **Regression check is STRONGER than E2's:** E2 had to compare pinned means because E1's
    per-soil file was unattached. E2's own `kaggle run 1 files/cv_per_soil.csv` is on disk, so
    E3 compares **24-element per-soil vectors** for all three base arms. Verified live: the
    notebook prints `regression check level: per-soil vectors (24 soils) from kaggle run 1 files`.
  - **Local-only, decided with the owner:** no GPU, no torch import anywhere, no Kaggle run, no
    dataset, no zip. `final_kaggle_upload_m2e2.zip` is untouched. A Kaggle mirror would need a NEW
    ~358 MB dataset carrying the embeddings; that option is closed unless reopened. E3's A3 drops
    E2's `/kaggle/input` mount walk (kept only inside reused C1, where it is inert locally).
  - **E3 HAS NO MOCK MODE**, deliberately: `BACKEND=mock` is what hid E1's `ndarray.median()`
    crash and emptied sections of E2's dry run. Missing arrays raise and list every filename.
  - **BLOCKER RESOLVED 2026-09-29:** the 15 arrays + 15 `.paths.npy` (79.6 MB measured) were
    harvested from E2's Kaggle `/kaggle/working/.cache` into
    `Model 2/Model 2 Experiment 3/embeddings/`. **Kaggle's Output listing does surface the
    dot-folder** — the earlier "unverified" caveat is now answered. `E2_embedding_manifest.csv`
    (sha256 per file, committed) records provenance; the `.npy` themselves stay gitignored, so a
    fresh clone cannot run E3 without re-harvesting them. First download attempt brought the
    wrong subset (all `vit_random`, no `dinov2`) — the loader reported all 12 missing filenames
    and that is what made the correction trivial.
  - Determinism: the 15 emitted data files are **byte-identical across repeated runs**; every
    cell printed output (one figure cell that printed nothing was given an explicit proof line,
    since a silent cell is E1's failure mode). `preprocess.final_check` still PASSES; no file in
    Model 1, M2 E1 or M2 E2 was modified.

  - Two traps already found and fixed while building E3 — both worth remembering:
    **(1) `df.iloc[perm]` is NOT a permutation control.** Reordering rows preserves each
    label→row pairing, so the "shuffled" block realigns on a join and is byte-identical to the
    real arm; the control would measure itself. Correct form: permute the data, keep the index
    (`pd.DataFrame(arr[idx], index=df.index, ...)`). Caught by `check_fusion.py`.
    **(2) E2's arrays are float64, not float32** (`embed_tiles` ends in `.astype(np.float64)`),
    so the harvest is ~80 MB, not ~41 MB. Measured from same-shape local mock files.
  - Outcomes: G1 fails → **Model 2 closes at three experiments** (replacement, scale,
    complementarity is the complete set). G1 passes, G2 fails → capacity noise; closes.
    All pass, gain < 15 → real but not competition-relevant. All pass ≥ 15 → submit and open a
    crop follow-up as a *new* pre-registered experiment. SHUF beats M1 → instrument leaking,
    nothing else is read. Alpha at grid edge → INCONCLUSIVE, **not** a refutation.

- Git: `df3a4b7 model 2 experiment 2 REFUTED gates didnt open`, `19b30dc model 2 exp 1 completed`,
  `8a6f087 Experiment 4 completed colour was not hurting`, `c31b966 experiment 3 done succesfully`,
  `ba0bd43 experiment 2 succesfull`, `b8be3a5 experiment 1 model 1`,
  `bb0abae completed entire preprocessing and data is in data folder...`. The owner writes commit
  messages; **do not commit unless asked.**

## 14. Environment and compute

**Local machine:** Python 3.14, numpy 2.5, pandas 2.3, scipy 1.18, scikit-learn 1.9, matplotlib,
Pillow 12.3. GPU RTX 3050 Laptop **4 GB VRAM**.
**NOT installed: torch, torchvision, opencv, scikit-image, timm, albumentations, jupyter, nbformat,
jupyter_client.** So: no deep-learning package and no way to execute or even parse a `.ipynb` locally.

Consequences that will bite you:
- **Do not propose `pip install torch` as a step.** Do all auditing, preprocessing, feature
  measurement and classical-CV work locally with numpy/scipy/sklearn/PIL only.
- **scipy-only stand-ins** for the missing libraries: `ndimage.binary_opening/closing/fill_holes/label`,
  `uniform_filter`, `gaussian_filter`, plus a hand-written histogram Otsu.
- **`numpy 2 removed np.trapz`** — use `np.trapezoid`. Old Kaggle/Colab snippets pasted here break.
- **Python 3.14 is strict:** `f"{x:g}"` fails on Pillow's `IFDRational` (EXIF/JFIF density) — cast to
  `float()` first. Multi-line f-strings are rejected, which bites when generating notebook cell text.
- **Keep ALL printed text ASCII.** Local stdout is **cp1252**: an em dash, arrow or Greek letter in a
  `print()` raises `UnicodeEncodeError` and kills the run — including inside a
  `subprocess.run(capture_output=True)` child whose output the parent re-prints. Use `->` and `-`.
  This is invisible until output is redirected or captured.
- Model 1's whole ladder runs **CPU-only in 30 s–5 min** (closed-form ridge on ≤24 rows; the cost is
  tile feature extraction and nested CV, not fitting). Never reach for the GPU to run one.
- **Cache** expensive derived artifacts keyed by `config_hash`, and key any cached *selection* result
  by a hash of cells/conditions/alpha-grid/seed, so changing one thing invalidates it rather than
  silently reusing it. **Verify run-to-run determinism by hashing the emitted submission across two
  full runs** — it has caught drift already.
- Training target for anything with weights is **Kaggle 2× T4 (16 GB each)**: torch 2.10.0+cu128,
  timm 1.0.26, Python 3.12. Two T4s are best used for **parallel CV folds and two backbones**, not one
  large model — with ~24 labelled samples an epoch is seconds, so model parallelism buys nothing.
  E2 cost 12–20 GPU minutes plus two failed paste-and-run cycles.

**Kaggle run workflow (standing, applies to every future Kaggle experiment):**
- **One zip per experiment**, built by `scratch/make_kaggle_zip.py`, named
  `final_kaggle_upload_<model><exp>.zip` at the repo root; uploaded as a Kaggle **dataset** (Kaggle
  auto-unzips). Notebooks cannot import each other, so side-car modules must arrive as a dataset.
- **Reuse the existing zip; rebuild only when a file inside it actually changed.** Verify by hashing
  the member inside the archive against the local file. Current zips: `..._m2e1.zip` (277,807,341 B),
  `..._m2e2.zip` (278,557,346 B).
- The builder must **measure what is actually read**, not ship the data tree. The upload is
  **278 MB, not 721 MB**, because the notebooks open only tile images plus five `processed_meta`
  artifacts and `sample_submission.csv`. `data/training_down/` and `data/testing_down/` (445 MB) are
  **never read** — their paths are only join keys in `manifest_images.csv`. Never size an upload from
  the folder tree.
- The builder **round-trips the archive** and asserts every file extracts byte-identical, because
  **75 of the 1,976 tile filenames contain U+00FC** (from the Münster source names). A mangled
  non-ASCII name fails the tile-presence assertion at the start of a run, not silently.
- **Kaggle mounts this user's datasets THREE levels deep**:
  `/kaggle/input/datasets/<username>/<slug>`, not `/kaggle/input/<slug>`. Any input resolver must
  search deep, not assume a level. **A local run cannot catch this bug class** — locally the repo root
  *is* the data root, so the fallback silently succeeds and hides the failure. Test by building a fake
  three-deep tree and exec'ing the **actual generated cell source** against it:
  `scratch/check_m2e1_resolvers.py`, `scratch/check_m2e2_resolvers.py` do exactly this at both depths
  plus the nothing-attached case.
- **[CONFIRMED failure mode] A rewrite silently drops the fallback that made the original work.**
  Model 2 E2's cell A3 was a tidied-up version of E1's resolver and lost the deep-mount search, so it
  died on Kaggle with "could not find manifest_images.csv" while the dataset was correctly attached.
  **When porting a resolver, copy the edge cases with it, and ship a test that execs the real cell.**
- Attachments are searched **alphabetically**. An older dataset holding an older copy of a side-car
  module **silently wins** over the fixed one. Attach **exactly one** data-bearing dataset.
- Use **Save Version → Save & Run**, never "Run All" — a draft session dies with the tab.
- Coming back: download the notebook's Output files into `Model N/Model N Experiment M/` (the
  experiment folder root is the authoritative record), keep the executed notebook as evidence, and
  quarantine mock dry-run artifacts in `local mock dry-run/` with a README stating they are plumbing.
- `.gitignore` covers `data/`, `*.zip`, `**/.cache/*.npy`, `__pycache__/`, `scratch/*.log`. The
  tile-feature CSVs under experiment `.cache/` **stay tracked** — they are the evidence behind the
  reproduction assertion of §7.4.

**Backbone gotcha that cost a session:** Kaggle's `vit_small_patch14_dinov2.lvd142m` is the **RoPE
variant with a native 518 px input**, whose patch embedding asserts an exact size, so a 224 px forward
raises `AssertionError: Input height (224) doesn't match model (518)`. What works is
`create_model(..., img_size=224, dynamic_img_size=True)`. `Model 2/Model 2 Experiment 1/backbones.py`
probes four construction variants and **refuses rather than drifting to 518**, because 518 would change
patch magnification (3.51 → 1.54 mm) — the variable reserved for E2 — and silently break comparability
with Model 1. A pre-flight cell (`E1`) runs `check_backbones.py` against the **real** backbones before
any tile is extracted, which turned this into a 30-second failure instead of a lost session. Follow
that pattern: **any code whose only test environment is a remote GPU must surface every bug in one
run** — ship a gate/overfit check and a pre-flight probe rather than guessing a cause from a symptom.

## 15. Patch magnification facts (Model 2 geometry)

- **[CONFIRMED, and it corrected a record] `backbones.PATCH_MM = 14 / 4.552516 = 3.0752` was WRONG —
  it omits the 256 → 224 resize.** A 256 px tile is a **56.23 mm** window squeezed into 224 px, so one
  14 px patch covers `14 * (256/224) / 4.552516 = 3.5145 mm` — **14.3 % larger**.
- `backbones.py` was **deliberately not edited**: E1's record and cached embeddings must stay
  byte-reproducible, and rewriting a shipped module to fix a constant is how two experiments end up
  disagreeing about what they measured. **E2 computes the correct value and prints both.** This is the
  standing rule for any derived constant found wrong in a shipped experiment.
- "Resolvable" = the soil's median grain is ≥ 1/3 of a patch; below that a grain sits inside one
  token's receptive field and enters only as texture:

  | crop px | patch mm | window mm | soils resolvable | tile area used |
  |---|---|---|---|---|
  | 256 (E1) | 3.5145 | 56.23 | **12** | 100 % |
  | 128 | 1.7573 | 28.12 | 13 | 25 % |
  | 64 | 0.8786 | 14.06 | 13 | 6.2 % |
  | 32 | 0.4393 | 7.03 | **14** | **1.6 %** |
  | 16 | 0.2197 | 3.51 | 16 | 0.4 % |
  | 8 | 0.1098 | 1.76 | 19 | 0.1 % |

  **The whole sweep buys two soils and discards 98.4 % of the tile area.**
- **The 10 fines are unreachable, not under-sampled.** Giving the largest fine soil (D50 = 0.1035 mm) a
  one-third-patch grain needs a ~23×23 canonical-px window upsampled to 224 — 507 real pixels carrying
  no grain-level information. Upsampling re-spaces what is there; it cannot add resolution.
  **Design consequence:** any magnification gain must appear in the mid and coarse groups and be
  **absent in the fines**. A gain uniform across sizes is not explainable by the mechanism; a gain
  concentrated in the fines is physically impossible and **means a leak**. Build this split into the
  experiment as a **gate clause**, not as a post-hoc explanation.
- **Centre crops are unconfounded:** measured on 240 sampled tiles with the soil threshold taken from
  the **full** tile, the soil-density ratio inside-crop vs tile is 1.005 / 1.009 / 1.007 for crops of
  128 / 64 / 32 px, p05 0.93, p95 1.12, and **zero** crops below 0.5 soil fraction — tiles are cut from
  the inscribed soil rectangle, so soil fills them uniformly. (A *per-window* percentile is useless: it
  always excludes the same fraction by construction.) A future design varying visible soil area can use
  a centred crop of existing 256 px tiles with no mask and no re-tiling — but always measure
  representativeness with an absolute threshold first.

## 16. Hard rules for the agent

1. **Do not re-run preprocessing** for a modelling experiment. `config_hash 010f44c36c74` is pinned and
   asserted. Never modify or rename anything under `data/Training/`, `data/Test/`,
   `data/processed_meta/`.
2. **Do not edit shipped modules** (`backbones.py`, `check_backbones.py`, `crop_geometry.py`,
   `check_crop_geometry.py`, prior experiments' builders/CSVs). Compute corrections in the **new**
   experiment's own module and print both values.
3. **One experiment = one variable** (or a factorial that says so), declared in the plan before
   running. Do not change backbone, loss, aggregation, output representation and optimiser
   simultaneously. Anything out of scope is a *later, separately named experiment* — if it looks
   necessary mid-experiment, **STOP and record why** rather than folding it in.
4. **Never overwrite a previous experiment's results. Never delete a failed experiment.** A new
   experiment writes only into its own folder.
5. **A submission is written only if the pre-registered gate passes.** Otherwise the notebook writes
   nothing and says why. Never spend a submission on a fine distinction; well-separated candidates
   only. Label every submission **gate pass** or **probe**, never both, and write the prediction before
   the score exists.
6. **Do not tune against the public leaderboard.** It is 3 fixed soils.
7. **Report spec conflicts, do not route around them.** If a documented rule blocks what the evidence
   demands, quote the rule, give options, and **wait**. This has held up three times, including the
   case where `Model 1/instructions.txt` declared the validation protocol FIXED, E2 needed to change
   it, the conflict was escalated, and the user approved amending it to "fixed *as of E2*; E2 is the
   calibration experiment that established it" with the original text quoted and the evidence cited.
   An amendment records the old wording; it does not silently delete it.
8. **Verify estimates and premises before advising on them.** A measured number is required before
   "expensive/cheap", and a claimed past action ("you already did X") must be checked against a real
   artifact before being treated as history. Unverified premises have already produced a wasted
   three-message invented risk.
9. **A null result is a result.** Say what closes. Do not present a research menu when the evidence
   supports a decision — converge, and state the go/no-go.
10. **Quote evidence, not vibes.** Every claim in a run record should name the artifact that measured
    it. Never cite a paper or number you have not read; label unverified leads explicitly.
11. **Prefer one self-contained, runnable artifact** (one notebook generated from a builder module,
    plus its flattened local `.py`) over a package. Prove claims inside it and end with a
    checklist of what was proven.
12. **Plain language, and explicit causal attribution.** The owner is new to research; explain with
    analogies, name what caused what, and self-correction is treated as strength, not failure.

## 17. Checklist to run a new experiment here

- [ ] Read `Model 1/instructions.txt`, `Model 2/instructions.txt`, the newest
      `Model N/.../instructions.txt`, and the relevant `model_plan_*.md`.
- [ ] Check §12 / `project-cancelled-hypotheses` before proposing anything.
- [ ] State **hypothesis**, **the one variable**, **controls + why each is indispensable**, and a
      **capacity/degenerate control** if the feature count changes.
- [ ] Write pre-registered predictions with `blind` flags, as **ordinal** statements
      ("better/worse than X"), never score forecasts, and never closer than 15 EMD apart.
- [ ] Fix the gate as **code**, executed before interpretation, whose failure writes no submission.
      Include: reproduction of the predecessor's matrix (< 1e-6), an integrity proviso that turns a
      truncated/saturated alpha grid into INCONCLUSIVE, and — if any per-group physical claim exists —
      a clause that requires the gain to be **located** in specific groups.
- [ ] Keep the frozen surface (§7.2, §6 aggregation) byte-identical; assert `config_hash`.
- [ ] Confirm nothing needed is unavailable locally (torch/timm ⇒ Kaggle path, §14), ship one zip,
      round-trip it, and pre-flight the backbones/resolvers against a fake mount tree.
- [ ] Run **twice**, hash the submission, compare. Report drift if any.
- [ ] Write `Experiment{M}.txt` including: run context, arm table, paired bootstrap effects, the
      mechanism table, gate JSON, predictions vs outcomes, and an explicit "what this decides".
- [ ] Update the Model's `instructions.txt` with the result and any new binding ruling.

## 18. File map — where to look for what

```
model_plan.md, model_plan_exp{2,3,4}.md      Model 1 experiment specs (exp1..4)
model_plan_m2e1.md, _m2e2.md, _m2e3.md        Model 2 experiment specs; m2e3 is the live one
report.md                                     10-section preprocessing/verification report
preprocess/                                   3044 lines, 11 stages; config.py = all constants
  config.py   single source of constants, config_hash, staleness, assert_writable
  inventory.py cameras.py geometry.py         ids, camera calibration, derived TARGET_PPM + tolerances
  resample.py soilmask.py colour.py           Lanczos canonical grid; crop rectangle; P3->sRGB + stored gains
  tiling.py manifest.py audit.py qc_sheets.py verify.py   tiles, manifests, 37 assertions, golden checks
Scratch (local runners/builders, not shipped):
  scratch/build_m*.py    notebook cell text -> .ipynb (builder is the source of truth)
  scratch/run_m*.py      flattened, locally-executed version of the same experiment
  scratch/check_m2e*_resolvers.py   exec real generated cells against a fabricated 3-deep Kaggle mount
  scratch/make_kaggle_zip.py        measured, round-trip-verified upload builder
  scratch/cal_exp*.py, colour_norm_test.py, mk_run*.py   calibration / probe side-experiments
Model 1/Model 1 Experiment 1..4/               closed ladder; read-only inputs to Model 2
Model 2/Model 2 Experiment 1/                  backbones.py, check_backbones.py, executed run-2 notebook
Model 2/Model 2 Experiment 2/                  crop_geometry.py + `kaggle run 1 files/` (authoritative)
Model 2/Model 2 Experiment 3/                  fusion.py (new, torch-free) and an EMPTY embeddings/ dir
data/processed_meta/                           stage_*.csv, manifest_{images,samples,tiles}.csv,
                                               audit.md, golden_checks.json, run_state.json, target_ppm.json
_look/                                         eyeball crops + exploratory logs/CSVs from preprocessing
```

Persistent memory (read it too; it is written for future sessions):
- project: `~/.qoder/projects/C--Users-mitansh-Desktop-KAGGLE-HACKATHONS-Predicting-Soil-Grain-Size-Distributions-from-Images/memory/`
- user:    `~/.qoder/memory/`
Both have a `MEMORY.md` index. Memory is point-in-time — **verify against the repo before asserting**
file paths, constants, or "X exists".
