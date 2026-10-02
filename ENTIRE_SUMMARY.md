# ENTIRE SUMMARY — Soil Grain-Size Distributions from Images (Kaggle 139732)

**Purpose.** A complete, self-contained record of everything done on this project from Model 1
through Model 10: what was tried, the code that did it, the preprocessing, the strategies, every
score obtained, and **why each direction was closed**. It is written to be handed to another
model as the sole input for planning the next experiment.

**Scope note — deliberate omission.** **Model 9 is excluded.** A different agent is working on
it; its design and results are not in this document and must not be inferred from it. Model 3 and
Model 4 do not exist (the numbering has gaps; there are no such folders).

**Date of this record: 2026-10-02.** Everything below is measured, not estimated. Provenance
tags: `[M]` = measured by an artifact in this repo or an actual Kaggle score; `[U]` = unverified
claim read from an external source; `[X]` = two sources in the repo disagree (see §11).

---

## 1. The competition, in one page

| | |
|---|---|
| Competition | Kaggle **139732**, *Predicting Soil Grain Size Distributions from Images* |
| Hosts | Lukas Leibold (BOKU / EU GRID) with **Enrico Soranzo** — who is the author of the prior art on this exact task (§10) |
| Licence / size | CC BY 4.0; 165 files, 395.38 MB |
| Field | ~400 teams (2026-09-27 snapshot); ends ~late Nov 2026 |
| Task | From photographs of a soil sample's surface, predict the **full cumulative particle-size curve**: 11 percent-finer values at fixed diameters 0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200 mm (DIN EN ISO 14688-1) |
| Data | **24 labelled training soils (127 images)**, **10 unlabelled test soils (35 images)** |
| Metric | log-weighted Earth Mover's Distance between cumulative curves; lower better; range effectively 0–500 |
| Leaderboard | public = a **fixed 3 of the 10** test soils; `Final = 0.30*public + 0.70*private` |

**Fixed acquisition facts (host statements, not recoverable from the repo):** camera-to-soil
distance is always **21 cm**; all photos under the same lighting; **different phones were used
deliberately to force cross-device generalisation**; "It's not really possible to determine how
many particles are visible"; **H031 was removed** (lab ran out of soil, original set was 25);
**H374 was re-photographed** after a competitor showed its label contradicted its image.

### 1.1 The metric, and the fact that the competition page prints it wrong

`EMD = integral of |F_true(d) - F_pred(d)| d(log10 d)` = **trapezoidal area** between the two
cumulative curves against log10(diameter).

The page renders the formula as a **left-endpoint Riemann sum**. Implemented literally that scores
the trivial baseline at **102.02**, but the page's own published reference value for the same
baseline is **100.31**. The trapezoid gives **100.3132**. `[M]` Two independent confirmations
exist, so this is settled: **the grader integrates area; the page notation is loose.** Do not
"fix" the metric to match the printed formula — that would be the error.

The metric is **exactly a fixed-weight L1** on the 11 outputs: supports are spaced almost
uniformly in log10, so trapezoid weights are `w = [0.2492, 0.5×9, 0.2508]`, summing to 5.0.
Verified to 8.5e-14 over 12,000 valid monotone curve pairs. `[M]`

### 1.2 Measured reference points on the 24 training samples (lower is better)

| approach | EMD |
|---|---|
| official trivial baseline (9.09% equal mass per bin) | **100.31** |
| copy some *other* training sample's curve | 104.22 |
| naive 2D connected-component granulometry | ~198 |
| train global **mean** curve | 85.39 |
| train global **median** curve | **82.62** ← the "ignore the images entirely" floor |
| rank-3 PCA label basis, oracle coefficients | 6.25 (projected) / 7.35 (bare) |
| perfect | 0 |

### 1.3 Where we actually stand

**Best submission ever: 55.80591** (Model 8 E2-A). `[M]`

Context, from a 2026-09-27 snapshot of 368 teams: rank 10 = 8.15, rank 20 = 14.11, rank 30 =
33.74, rank 40 = 37.63, **rank 49 = 40.22**. Ranks 50+ were not reachable without sign-in.
So **55.81 is outside the top 49**, and reaching the top of the board needs roughly a 45 EMD
improvement, not a 5 EMD one.

Two frontier data points that reframe the whole problem — both `[U]`, neither reproduced by us:
- A competitor at rank 1 (Alexy) states *"My actual image model scores ≈10 public"*, and that the
  near-zero top scores (0.00001, 0.09, 0.92) are **the public test answers reconstructed by
  systematic submission probing** — because the metric is a separable sum, each submission is a
  noise-free linear measurement of the hidden truth. ~50 submissions recovered all 3 public
  curves. **The host confirmed this is probing, not a data leak**, and said "We are looking into
  how to address this."
- A competitor at rank 31 (zaoui Hamza) reported that passing the data to a large language model
  produced **34, then 29** on the public board **with no image model at all**.

**Consequence for the next design:** the honest achievable band appears to be roughly **~10 (good
image model) to ~30 (strong prior, no images)**. We are at 55.8. A no-image approach reportedly
reaching 29 means **we are not close to any ceiling — we are losing to methods that never look at
a photograph.** That should dominate the planning conversation, and §9 lists what we have
genuinely closed versus what merely *looks* closed.

---

## 2. The dataset and the preprocessing pipeline

All of it is reproducible: `preprocess/`, 11 stages, `PIPELINE_VERSION 1.0.0`,
`config_hash 010f44c36c74`, **37/37 audit assertions passing**. `[M]`

### 2.1 Stage order and what each does

`inventory → cameras → geometry → resample → soilmask → colour → tiling → manifests → audit → qc → verify`
(`preprocess/run.py` `STAGES`; CLI names differ from module files via `MODULE_MAP`.)

| stage | what is load-bearing |
|---|---|
| **inventory** | walks `data/Training`, `data/Test`; sha256 per file → `baseline_sources.json`; `image_id = sha256(source_path)[:12]` |
| **cameras** | camera identity is **filename-first, EXIF only as validator** — EXIF is absent on all 69 Motorola and 55 Samsung training files (present on only 38/162). Regex ordered so "Motorola Edge 60 Fusion" wins before "Motorola Edge" |
| **geometry** | axes paired **short-with-short / long-with-long** (25 Motorola files are stored portrait with no EXIF to say so; naive width/width reports a bogus 122% mismatch). `effective_ppm = sqrt(eff_short × eff_long)` |
| **resample** | `exif_transpose → P3→sRGB → canonicalise → LANCZOS resize → PNG`. **Refuses to upscale.** Records clipping *introduced* vs *already present* |
| **soilmask** | grey ÷ gaussian surround → Otsu → morphological close/open (disk r = max(2, 0.004·min(h,w))) → fill holes → largest component. Picks an **inscribed rectangle; never deletes pixels** |
| **colour** | `to_srgb()` with RELATIVE_COLORIMETRIC intent, applied **before** resampling. Grey-world gains computed on masked soil pixels, normalised to geometric mean 1, and **STORED BUT NOT APPLIED** (`gw_gain_r/g/b`; `apply_gains()` exists and is never called) |
| **tiling** | partitions the *cropped* canonical image; sizes 128/224/256/384 declared, **only 256 materialised**; `TILE_STRIDE_FRAC = 0.0` (non-overlapping); `tile_w_mm = tile_px / target_ppm` |
| **manifests / audit / qc / verify** | 162×103 source-of-truth table + 34-row sample table; 37 fail-loud assertions; 162 contact sheets; golden information-preservation checks (grating, stop-band, signal survival) |

### 2.2 The scale problem — the single most consequential preprocessing fact

`data/ppm_updated.csv` quotes pixels-per-mm at each camera's **native sensor resolution**, which
is **wrong for the delivered pixels**. The delivered files are smaller for both dominant training
cameras:

| camera | images | delivered px | **px/mm in the actual file** | scene FOV |
|---|---|---|---|---|
| Motorola Edge | 69 (train) | 1600×720 | **4.597** | 348 × 157 mm |
| Samsung A52 | 55 (train) | 1599×1200 | **4.553** | 351 × 264 mm |
| Motorola Edge 60 Fusion | 3 (train, H374 only) | 4096×2304 | 12.465 | 329 × 185 mm |
| iPhone 16 | 21 (**test only**) | 5712×4284 | **19.525** | 293 × 219 mm |
| iPhone 14 | 14 (**test only**) | 4032×3024 | **13.942** | 289 × 217 mm |

**The canonical scale is DERIVED, never typed:** `geometry.py:191` takes
`min(effective_ppm)` over cameras *and axes* = **4.552516 px/mm**, which makes all 162 images
identity-or-downsample. The earlier proposal of 4.58 would have forced a 0.57% **upscale** on all
55 Samsung images and was rejected on that ground. `[M]`

### 2.3 The physical resolution floor

At 4.55 px/mm a **0.063 mm particle is 0.3 px**. Individual grains in the fine tail are
**physically unresolvable in training *and* test** — only texture and appearance can inform them.
Minimum resolvable grain under a 4-pixel rule: 0.87 mm on Motorola/Samsung, 0.32 mm H374,
0.29 mm iPhone 14, 0.21 mm iPhone 16 — **the test images resolve finer than training ever did.**
Naive particle counting was tried and scored ~198 against a ~100 baseline. `[M]`

### 2.4 The features, and how they are actually computed

Source of truth is `preprocess/verify.py`. **They are not DoG bands** — they are
**Laplacian-pyramid octave bands**: `band_k = low_k − low_{k+1}`, `e_k = std(band_k[m])` where `m`
selects pixels above the 12th percentile of the gradient, with illumination base
`BASE_SIGMA_MM = 12.0`.

| feature | definition | sigma mm | wavelength px | camera-explained variance | ρ with log D50 |
|---|---|---|---|---|---|
| e1 | band std | 0.22 | 2.50 (at Nyquist) | 54% | −0.24 (fails shuffled-label) |
| e2 | band std | 0.44 | 5.01 | — | +0.49 (fails shuffled-label) |
| **e4** | band std | 0.88 | 10.02 | **1.6%** | +0.68 |
| **e8** | band std | 1.76 | 20.03 | **1.2%** | +0.76 |
| **e16** | band std | 3.52 | 40.06 | 23% | +0.76 |
| lum_sd, lum_p10/50/90 | luminance spread/percentiles over masked px | | | | |
| grad_mean | mean gradient magnitude over masked px | | | | |
| sat | mean((max−min)/(mean+1e-6)) | | | 71% | −0.40 |
| spec_centroid_cpm | spectral centroid × ppm (cycles/**mm**) | | | | |
| dom_wavelength_mm | 1/cpm — **the only feature with real-world mm units** | | | | |
| R, G, B | per-channel means (**implemented in the Model-1 extractors, not in `preprocess/`**) | | | | |

**e1 and e2 are excluded as sub-Nyquist** — at 2.5 px wavelength they can only report sensor
noise and in-camera sharpening. `[M]`

### 2.5 Other measured corpus facts

- **Colour space is a train/test split too:** 38 of 162 images carry an embedded **Display P3**
  ICC profile (all 35 test + the 3 H374); the other 124 are untagged sRGB. P3→sRGB is applied
  **before** resampling because it is a nonlinear tone curve plus a matrix and does not commute
  with a linear anti-alias filter — converting afterwards gives the 38 profiled files a different
  effective blur kernel, manufacturing the very camera artefact being isolated.
- **Highlight clipping differs 24× between cameras** in the *source* photos: Samsung A52 1.635%,
  iPhone 14 0.646%, Edge 60 Fusion 0.451%, Motorola Edge 0.305%, iPhone 16 0.067%. A blown
  highlight on a bright grain reads as low texture energy, so the Samsung's tone curve discards
  detail the iPhones keep. This is a **floor** on how well any Samsung-derived texture statistic
  can match an iPhone one.
- **Grey-world gains differ systematically by camera** (stored, not applied): Motorola Edge
  R 0.942 / B 1.063; Samsung A52 0.905 / 1.132; Edge 60 Fusion 0.898 / 1.264; iPhone 14/16 ≈0.98 /
  1.033. The Androids are the warm, saturated cameras; the iPhones near-neutral.
- **Tray fraction ranges 14.5% (Motorola) to 41.6% (iPhone 14)** of the frame, so a uniform small
  crop cap would manufacture a train/test asymmetry. Cap is 30%; it binds on 2/162 images.
- **Tiles:** 1,976 materialised at 256 px (a 256 px tile = **56.23 mm** of real soil); **1,946
  pass `soil_fraction ≥ 0.50`**; 104 are pure tray (<0.05). 75 tile filenames contain U+00FC.
- **Images per soil: train mean 5.29 (range 3–8), test mean 3.50 (range 3–5)** — so per-soil
  feature noise is ~1.3× higher at test. Measured consequence: scoring honest CV at k=3 images
  costs only **+1.14 EMD**. `[M]`
- **The labels are interpolated, not measured at the 11 supports** (Data page: raw curves were
  interpolated linearly on a log10 axis). The fine end comes from sedimentation (hydrometer)
  analysis. Part of the residual shape error is therefore **lab measurement noise, not soil**.

---

## 3. The frozen modelling stack (identical in Models 1–8, 10)

**Head** — closed-form, no gradient training anywhere in this project:

```
StandardScaler → rank-3 PCA curve basis (on labels) → multi-output Ridge on the 3 coefficients
→ reconstruct 11 supports → monotone projection: clip to [0,100] → np.maximum.accumulate
→ force exactly 100 at 200 mm
```
Scaler and basis are refit **inside each training fold only**. Alpha grid of 16 values
(0.03 … 1,000,000).

**Features (the "12")** = texture core `e4, e8, e16, lum_sd, grad_mean` + colour
`R, G, B, sat, lum_p10, lum_p50, lum_p90`.

**Aggregation** — **two-stage median, part of the frozen protocol**: tiles →(median)→ image →
(median)→ soil. One-stage vs two-stage moves CAM+RES for the same matrix from 75.7 to 85.0, so
this is not an implementation detail. `[M]`

**Validation** — leave-one-**family**-out over **16 curve-distance families**, built
deterministically by average linkage on pairwise label-curve EMD cut at 14.0. The six
near-duplicate pairs must land in the same family (verified twice, independently). **The soil is
the unit of supervision; no image-level splits.** Banned inputs: camera, ppm, EXIF, ICC, site,
sample_id, cv_group.

**Why families matter:** the 24 soils are **not 24 independent observations.** Six pairs are
near-duplicates in label space (H366~H371 at 3.3 EMD, H367~H372 4.3, H549~H668 6.8, H038~H637
7.3, H368~H615 7.0, H181~H183 8.2). **Effective n ≈ 8–10.**

**The label manifold is nearly one-dimensional:** PC1 carries **91.4%** of label variance and
correlates **0.9949** with log10 D50; PC2 6.2%, PC3 1.4%. D50 is bimodal — ten soils at
0.021–0.104 mm, two in between, twelve at 1.795–6.126 mm.

---

## 4. The durable instruments (this is the project's real intellectual output)

Every one of these exists because a number was believed and then destroyed.

| instrument | the measurement that created it | how to use it |
|---|---|---|
| **The alpha-selection inversion** | M1 E1 chose alpha 0.03 because LOGO CV said 36.03; it scored **172.70** externally while a no-image submission scored 102.37. Across the whole grid, rank-corr(LOGO CV, transfer) = **−0.953** | Never select on LOGO CV alone. Report it; select elsewhere |
| **CAM+RES ruler** | fit one training camera's view, predict the other camera's view after matched anti-alias blur, held-out soil's label excluded. Calibrated 3× externally: bias factors 0.84 / 0.94 / 1.18 | **ORDINAL ONLY.** May rank; must not forecast; must not choose between candidates <15 EMD apart. Demoted again by M1 E4 |
| **The circularity rule** | training on res16-blurred views and scoring with a res16-blurred ruler made the frequency features M1 E3 had removed look *helpful* (−3.08) when held-out blur says they cost **+57.60** — a **34 EMD artefact** | Any treatment that alters the training distribution must be evaluated on a **held-out level of the nuisance it targets** |
| **One-parameter null** | a motivating "+8.35 EMD" gain turned out to be an oracle-fitted scalar; tuning *any* one parameter on 6 random soils buys median **+3.82**, p90 **+11.71**, max **+40.18** | Compare a fitted-parameter gain to the chance gain of fitting a parameter, never to zero |
| **Best-of-N head null** | picking the winner of 15 fitted heads on **permuted** labels still shows a median apparent gain of **15.29 EMD** (p90 18.29) under the family-honest protocol | Any sweep result below ~15 EMD is selection, not signal |
| **Same-dimensionality noise comparator** | M7-A's real 4-column block gained +0.0726 alias separation while **phase-scrambled (+0.1131) and shuffled (+0.2664) noise blocks gained more** | Appending columns inflates RMS distance in a higher-D space. A positive gain over the unpadded baseline proves nothing |
| **Noise floor at n=24** | shuffled labels reach \|Spearman ρ\| ≈ **0.52** | **Any correlation below ~0.5 in this project is indistinguishable from chance** |
| **3-soil null band** | per-soil CV SD 22.75 → SD of a 3-soil public mean is 12.28, 95% band **[21.7, 69.9]**; private (10 soils) SD 5.51 | **Public differences below ~5 EMD are unmeasurable, full stop.** Never present an in-domain-vs-external difference without this band beside it |
| **Family-cluster bootstrap** | resampling the 16 families, not the 24 soils, because near-duplicate pairs are not independent | Required on every claimed gain. Also: prove the gate has *power* before trusting it — M8 E1's G4 null was wider (SD 4.82, p95 +8.80) than the +6.09 effect it tested |
| **Reproduction assertion** | rebuilding a predecessor's feature matrix and asserting drift < 1e-6 (measured 2.8e-14, and 0.0 for the anchors) | Caught a real bug in M1 E3 (tile→soil×camera→soil vs tile→image→soil). Keep in every experiment |
| **Bit-exact anchor discipline** | `43.453225201811563` (strictly nested family-honest LOFO) and `41.1475158656982032` (oracle alpha 3) have now been reproduced to `0.000e+00` by three independently written implementations | Never let a numerical coincidence become a reproduction claim. The two protocols differ by 0.4315 EMD because the looser one re-selects alpha on folds containing the scored family |

**Terminology the owner insists on, unchanged everywhere:** `43.0217308796477` = historical
internal anchor; `43.453225201811563` = Model 7+ internal nested-LOFO baseline; any Kaggle number
is external. **Internal CV EMD and external Kaggle EMD are different scales and are never
subtracted.**

---

## 5. Model-by-model record

### Model 1 — classical features, ridge, and the discovery that the ruler was broken
**Code:** `Model 1/Model 1 Experiment {1,2,3,4}/`, generators `scratch/build_m1e{1,2,3,4}.py`,
runners `scratch/run_m1e*.py`. Ran **locally**; only CSVs were submitted.

| exp | what it did | internal | **public** | verdict |
|---|---|---|---|---|
| E1 | 16 features, alpha by LOGO CV | 35.98 | **172.69929** | the model that used images **lost to the one that ignored them** |
| — | constant train-mean curve (metric check, ignores images) | 85.39 | **102.37237** | calibration point |
| E2 | 14 features (dropped 2 defective columns), alpha by CAM+RES — an *instrument-calibration* experiment | 84.31 nested CAM+RES | **71.27346** | established the replacement ruler |
| E3 | 2×2 factorial (core5 / +colour7 / +frequency2); selected cell C = **12 features**, absolute-frequency family removed | 51.87 CAM+RES | **61.23560** | the only clean gate pass |
| E4 | **declared discriminating probe**: C1 = texture-core-only, 5 features | 56.98 CAM+RES | **77.12257** | settled the colour question externally |

**What Model 1 established:** the alpha criterion ranks configurations **backwards**;
`soil_fraction` is a **defect not a weak feature** (constant to 4 dp across all 24 soils, mean
0.9999 / std 0.00027, but reaches 0.829 on test = −63.9 SD → its fitted coefficient is a pure
noise amplifier; `crop_area_fraction` fails the same way, milder) → **rule written in: exclude any
column with near-zero training variance**; the **frequency family costs +26.76 EMD** (CI
[+14.81, +39.07], significant in 4/4 conditions) while the **colour family's transfer effect is
−4.29, indistinguishable from zero**; and finally E4 showed **dropping colour costs 15.89 EMD
externally** — so colour stays. **Model 1 closed** when E4 failed to beat 61.24. `[M]`

**The counter-intuitive lesson carried forward:** colour *moves most* under camera change (1.54 z)
yet removing it changes nothing; the absolute-frequency features *barely move* yet cost ~27 EMD.
**Use the paired removal effect, never the size of the shift, to decide what to keep.** A feature
can shift a lot and not be load-bearing once the model is regularised.

### Model 2 — does a pretrained vision backbone beat 12 hand features? No.
**Code:** `Model 2/Model 2 Experiment {1,2,3}/`, `scratch/build_m2e*.py`, `run_nb_m2e*.py`,
`verify_e2_embeddings.py`. E1/E2 ran on **Kaggle (torch)**; E3 ran locally on cached embeddings.

- **E1** (frozen DINOv2, 384 dims): internal M1 43.0217 / **D 44.3302** / R(random-init ViT)
  39.7769. Submitted arm D as a **declared probe** → **60.56167**, then the project's best.
  **DINOv2 did NOT beat the 12 hand features** — and R "won" in-domain while being the E1
  overfitting trap.
- **E2** (magnification: crops 256/128/64/32): D scored 44.33 / 43.32 / 43.83 / 44.87 — **lost to
  R at all four crops**. No submission. Machinery valid, hypothesis dead.
- **E3** (complementarity: hand + DINOv2 fused, 396 columns): **fuse 43.431 vs M1 43.0217 —
  fusion is 0.41 EMD WORSE**, CI [−6.69, +7.46]. **Model 2 closed at three experiments.**
  Its permutation control `shuf@c256` scored 52.927 (instrument healthy), and the trap arm
  **`fr@c256` (random-init ViT) scored 39.259 — the best in-domain number in project history —
  with the worst transfer (CAM+RES 83.94).**

**Why Model 2 is closed:** replacement, magnification and complementarity are the three ways a
frozen backbone could help, and all three are refuted. **Do not re-propose a DINOv2 feature at
any crop or in any combination without new evidence.** Independently corroborated externally: a
public notebook "DINOv2 + Sqrt-Mass Ridge" scores 61.83, a tie with our 60.56. `[U]`

### Model 5 — camera/domain adaptation. Refuted, and structurally dead.
**Code:** `Model 5/Model 5 Experiment 1/{alignment.py, transfer_eval.py, check_*.py}`,
`scratch/run_m5e1.py`, `verify_m5_result.py`. Pure numpy — **ran locally, zero submissions.**

CORAL alignment of fitting-camera image features onto the 35 unlabelled test images, judged on a
third camera: gain **−8.748 EMD** (pooled, CI [−12.776, −3.967]), **direction-inconsistent**.
Closed at one experiment. **Two sub-claims are now structural, not merely untested:**
(i) **any affine/first-moment feature adaptation** — a constant shift commutes with the soil
median and is then deleted by the head's `StandardScaler`, verified to 6.9e-15, so MEAN and
self-alignment arms are exact zeros *by construction* and can never be informative controls;
(ii) **transductive use of unlabelled test images as an alignment target** — allowed, tested,
did not help.

**What survives:** the **camera transfer term is +5.55 EMD**, measured *inside* training on 45
fold-honest cross-camera rows. That replaced an earlier "+16.2/+18.2 camera penalty" which was
never a camera measurement at all — it was an in-domain-vs-leaderboard difference.

### Model 6 — the evidence pass that killed eight ideas in one night
**Code:** `Model 6/Approach A/` (spec + contract preserved as a **closed research record**, no
code), probes in `scratch/` (`probe_feature_bound.py`, `probe_head_spread.py`,
`spike_curve_ensemble.py`, `reconstruct_m63_rule.py`). **Zero submissions.**

| candidate | decisive measurement | verdict |
|---|---|---|
| M6-A attenuation/expansion of predicted curves | the motivating 50.96→35.42 gain was an **oracle-fitted scalar**; nested `k̂` averaged 0.26 and *degraded* error 45.65→76.41; direction test helps both ways | **dropped** — isotropic scalar correction unsupported |
| M6-2 feature containment (clip into fit-set support) | 41.15 → **41.91** (min/max) → **43.88** (5–95 pctile); intervention demonstrably fires (8.7%/21.9% of cells clipped) | **refuted**, a real negative |
| M6-1 fixed 50/50 curve ensemble | honest-holdout error correlation between the two models **r = 0.745**; Jensen bounds the blend by the average of the parts | **not supported** |
| L1 loss (matches the metric's geometry) | best L1 41.22 vs best L2 41.15, **worse** on both displaced and interior soils | no support |
| Bagging / resampled fits | 41.15 → 43.33 (10 fits) / 42.64 (50) | no support |
| Output-basis rank above 3 | rank-3 floor is 6.25 EMD but end-to-end rank 3→5/7/9/11 buys only **0.38–0.57** | closed (a floor is not a harvest) |
| Fixed bias-vector correction | removing the whole bias vector changes error by **−1%**; estimating it on 12 soils and scoring 12 **loses 3.17** | no systematic curve bias exists to correct |
| **M6-3 new features for the error tail** | forensic diagnostic: top-10 of 24 soils carry **61.2%** of CV error; worst soils sit near a feature-twin whose curve differs by median **65.7 EMD**; **H038 vs G190 is irreducible** (feature distance 1.32, curves **125.4 EMD** apart, **0 of 12** features separates them at \|t\|>3) and that one pair is **10.1% of all CV error, destroyed at pixel level before any model runs** | **closed** |

**Also corrected here:** the "1-NN feature bound" was **leaky** (it let a soil borrow its own
CV-family siblings and fitted the scaler on all 24 rows including the query). Honest local lookup
is **47.66**, i.e. 4.64 EMD *worse* than the shipped ridge — the parametric head genuinely beats
nearest-neighbour. And **alpha-selection cost is 1.68 EMD**, the gap between the best fixed alpha
(3.0 → 41.15) and strictly nested selection (42.83) — the largest single *identified* internal
loss.

### Model 7 — a new feature block, killed by its own pre-gate at zero cost
**Code:** `Model 7/Approach A/{alias_gate.py, check_alias_gate.py, m7a_data.py, m7a_eval.py}`,
`scratch/run_m7a_gate.py`, `check_m7_prereg.py`. **Zero submissions.**

Hypothesis: the alias pairs are separable on within-tile texture the 12 features don't summarise,
so add four new statistics — **A1 structure-tensor anisotropy, A2 0°/90° autocorrelation
e-folding ratio, A3 variogram range-to-sill, A4 directional contrast continuity** — aggregated
tile→image→soil by the same medians.

A **pre-gate was run before any CV arm**: median separation of the frozen 10 alias pairs rose
0.609266 → 0.681902 with the real block (**+0.0726**) while same-dimensionality **noise placebos
rose more** (+0.1131 phase-scrambled, +0.2664 shuffled). `gate_passed = False`, so the model was
never fitted, never scored, and nothing was submitted. **REFUTED-PRE-GATE.** `[M]`

Also registered here: **M7-C (selector-only, `SEL-1SE` inside the frozen alpha grid) was
pre-registered but NEVER RUN — it remains the one unexecuted cheap experiment in the repo.**
And a reusable ruling on scope: Model 5's closure is scoped to a **mechanism** (camera adaptation
/ moment matching, including transductive form), *not* to every use of unlabelled test-image
statistics — but the ambiguity is explicitly **not permission**.

**Process finding worth keeping:** the alias figures M7-A had to reproduce came from a probe that
was never saved to a file — only to the session transcript — and used a *different, looser* rule
than the one registered. It was settled by a **documented amendment (A1)** rather than by tuning
the rule until the number matched. The checker now asserts **both** directions so the two
instruments can never be quietly conflated again.

### Model 8 — the live front: 8 submissions, a new best, and the plateau
**Code:** `Model 8/Experiment {1,2,3,4}/` (`run_e1.py`, `head_d50.py`, `d50.py`, `ruler.py`,
`build_e2.py`, `alpha_probe.py`, `build_e3.py`, `learning_curve.py`). **All ran locally on CPU;
CSVs submitted by hand.**

- **E1 — is the label-side shape map exploitable?** With **true D50 supplied and no features at
  all**, a **quadratic** map from log10 D50 to the curve beats linear by **+6.09 EMD**
  (family-cluster CI [+1.278, +10.769], P(gain<0)=0.0073, better for 19/24 soils, median
  per-family gain +7.4), and **degree 3 does not improve on degree 2** (19.36 vs 19.12) — the
  effect saturates at exactly two parameters. **But wired end-to-end through the noisy D50
  estimate the gain collapses to +0.466 vs the linear control (CI [−3.88, +4.92]).** Verdict
  `NOT-RESOLVED`. **The label-side fix is real; the D50-first route cannot reach it.**
  Independent corroboration: the co-host's own GRAI3 and 2025 paper fit a **two-parameter
  Weibull**, and the dominant public-notebook motif is the same — arrived at from an unrelated
  direction.
- **E2 — colour robustness (H2), 3 arms + 2 alpha probes, all submitted.** **E2-A raw 12-feature
  colour = 55.80591, NEW PROJECT BEST**, beating DINOv2's 60.56 by 4.756. Monotone dose-response
  in the wrong direction: **raw 55.81 < rank-normalised 62.52 < colour-deleted 67.27**. **H2
  REFUTED.** Alpha probes: 0.3 → 59.38746, **3.0 → 55.80591**, 30 → 71.10598, and here the
  internal ruler ranked **correctly** (ρ = +1.00). So: *within* one head the ruler orders things
  right; *across* feature sets it orders them backwards (ρ = −1.00). **α=3 is the external
  optimum and is settled.**
- **E3 — colour *encoding* (H3), 3 arms, all submitted.** CIELAB 56.08671, grey-world-corrected
  57.87360, +2 texture columns 56.43951 — **all worse than control.** Four distinct colour
  encodings span **1.79 EMD**. **H3 REFUTED.** Colour content is worth ~11 EMD; its *encoding* is
  worth under 2. Note E3 also **broke the tidy anti-rank narrative**: its arms gave ρ = **+0.50**,
  so "internal CV anti-ranks arms" is not a stable law — it happened twice at n=3 each.
- **E4 — the data-vs-model question, zero slots spent.** Family-honest learning curve over
  n = 4…21 soils: **`EMD = 35.326 + 139.159/n + 15.851/√n`, R² = 0.999.** The 1/n term is
  **5.80 EMD at n=24**; doubling to 48 buys ~2.9, to 96 ~1.4 more. **Verdict: DATA-LIMITED**,
  total headroom on the Android transfer curve bounded at ~8 EMD. It also closed the curve basis
  (rank 2–11 all sit 42.4–44.1 end-to-end while the oracle floor for rank 2 is 9.75) and closed
  blending mechanistically (**OOF errors at alpha 1/3/10 correlate 0.86–0.95**, so averaging
  cannot help — which also explains why alpha is not worth re-tuning: heads across the range are
  nearly the same model).

**Model 8's own conclusion, which the next design must confront:** *"further tuning of THIS head
is not merely slow, it is uninformative. Twelve handcrafted features and 24 soils appear to be
the binding constraint."* Its ranked recommendations were (1) more labelled soils, (2) a model that
extracts more from each soil, (3) camera augmentation. **Recommendation (2) was Model 10. It
failed.**

### Model 10 — per-image data expansion. Refuted externally, decisively.
**Code:** `Model 10/Experiment 1/{plan.md, image_features.py, build_m10e1.py, ruler.py, d50.py}`
+ Kaggle mirror `Model10_Experiment1.ipynb` and `final_kaggle_upload_m10e1.zip` (0.84 MB).
**Gates:** both anchors reproduced to `0.000e+00`; image→soil rebuild vs `features_soil.csv`
`2.842e-14`; 16 folds / 0 leaks; 9/9 artifacts byte-identical on a cold re-run.

**Hypothesis (H10):** the pipeline collapses each soil to ONE row by median, so fitting on all
**127 image rows** (labels = each soil's curve, `1/n_images` soil-balanced weights) uses
information the median throws away. Arms: **A** incumbent control; **B** image-level fit, mean raw
curves then project once; **C** soil-level fit, project **each** image curve then mean.

| arm | internal | **public** | vs incumbent |
|---|---|---|---|
| A (incumbent, not resubmitted) | 43.4532 | **55.80591** | — |
| B (image-level fit) | **36.8279** ← best internal | **82.87051** | **+27.06 worse** |
| C (per-image projection) | 40.8265 | 60.58798 | +4.78 worse |

**H10 REFUTED.** The internal-best arm scored externally-worst — **Spearman ρ = −1.00 for the
third time**, and the largest swing ever measured here: a 6.63 EMD internal *gain* bought a 27.06
EMD external *loss*. Placebo (permuted labels) 97.05 confirmed the instrument was healthy, and
**every family-cluster bootstrap CI included zero**, so the internal gain was never established
either — it sat below the project's own best-of-N null of 15.29.

**Mechanism, stated before the scores existed:** arm B selected **alpha 0.03–0.3** against A's
0.1–3.0. Image-row scatter inflates residuals, so nested selection shrinks the penalty — B was
effectively a **less-regularised model fitted to noisier inputs**, which is the exact signature
that preceded the project's two worst results (M1 E1, M2 E3 `fr`). And the reason it cannot work
at all: **127 images carry only 24 distinct curves (~8–10 effectively independent). The effective
sample size did not increase; only the noise and the freedom to fit it did.**

**Decision taken under the pre-frozen rule (plan §6b): the incumbent 55.80591 is RETAINED.**
E4's recommendation (2) is now measured and refuted, not merely unresolved.

---

## 6. Complete submission ledger — all 16 public scores

| # | model / arm | what it was | public |
|---|---|---|---|
| 1 | M1 E1 | 16 feats, alpha by LOGO CV | 172.69929 |
| 2 | M1 metric-check | constant train-mean curve, ignores images | 102.37237 |
| 3 | M1 E2 | 14 feats, alpha by CAM+RES | 71.27346 |
| 4 | M1 E3 | **12 feats**, frequency removed | 61.23560 |
| 5 | M1 E4 | 5 feats, colour dropped (probe) | 77.12257 |
| 6 | M2 E1 `D` | 384-dim frozen DINOv2 (probe) | 60.56167 |
| 7 | **M8 E2-A** | **12 feats, colour raw, α=3** | **55.80591** ← best |
| 8 | M8 E2-B | colour deleted (5 feats) | 67.26598 |
| 9 | M8 E2-C | colour rank-normalised | 62.51669 |
| 10 | M8 α-probe | same head, α=0.3 | 59.38746 |
| 11 | M8 α-probe | same head, α=30 | 71.10598 |
| 12 | M8 E3-A | colour → CIELAB | 56.08671 |
| 13 | M8 E3-B | colour ÷ grey-world gains | 57.87360 |
| 14 | M8 E3-C | 12 + spec_centroid + dom_wavelength | 56.43951 |
| 15 | **M10 E1-B** | **fit on 127 image rows** | **82.87051** |
| 16 | M10 E1-C | soil fit, per-image projection | 60.58798 |

**Models 5, 6, 7 produced zero submissions.** Three of the sixteen were explicitly *declared
probes*, not gate passes (M1 E4, M2 E1 D, and M8's alpha probes).

**Counting note:** 16 submission CSVs exist on disk and all 16 map to rows above.
`Model 8/Experiment 2/Submission_Model8_E2_E2A_recheck.csv` is **byte-identical** to
`Submission_Model8_E2_E2A.csv` (verified by sha256) — a plumbing duplicate, **not** a 17th
submission. The 102.37237 entry is `Model 1/Model 1 Experiment 1/MetricCheck_mean_curve_baseline.csv`.

---

## 7. The central finding, restated once

**Every validation instrument this project built measures interpolation on 24 labelled Android
soils, while deployment is extrapolation to 10 out-of-hull iPhone soils.** The rulers are not
broken — they answer their own question correctly and reliably. They are structurally incapable
of answering the question that decides the score.

The evidence is the ledger of internal-vs-external pairs:

| internal | external | note |
|---|---|---|
| 35.98 | **172.70** | best-in-domain, worst-ever |
| 39.26 | 83.94 (CAM+RES) | random-init ViT, best in-domain number in project history |
| 39.78 | 78.55 | random-init ViT, worst arm of its experiment |
| 44.33 | **60.56** | best of its pair externally from the *worse* internal |
| 42.73 | 62.52 | M8 E2-C, internally best of three |
| 43.45 | **55.81** | M8 E2-A, internally **worst** of three → project best |
| 36.83 | **82.87** | M10 B, internally best → externally worst |

**Therefore: any future experiment whose arms differ in feature space, granularity, encoding or
model class CANNOT be selected internally. It must be decided by submission.** The corollary that
took ten models to learn: *internal numbers are still worth computing* — as a health check on the
machinery (placebo, anchors, fold integrity) and never as a ranking.

---

## 8. Closed directions — do not re-propose without new evidence

| direction | what closed it |
|---|---|
| Particle counting / granulometry | ~198 EMD vs ~100 baseline; grains are 0.3 px |
| DINOv2 / pretrained features, any crop, any combination | M2 replacement + magnification + complementarity all refuted; `fr` in-domain-best/transfer-worst |
| Camera adaptation: CORAL, moment matching, **any affine feature transform**, transductive alignment on test images | M5: −8.748 EMD, direction-inconsistent; affine arms are **exact zeros by construction** (verified 6.9e-15) |
| Resolution-matched training | whole 3×3 blur matrix spans only 9.2 EMD; matched training buys ≤2.3 |
| Richer tile→soil aggregation (mean/sd/p10/p90 as extra columns) | 41.15 → 42.13 → 42.22 (worse) |
| Tile-distribution summaries | 47–49 (much worse) |
| Within-tile spatial statistics (structure tensor, variogram, directional contrast) | M7-A pre-gate: real block gained less than **noise** placebos |
| **Per-image data expansion (fit on image rows)** | **M10 E1: internal −6.63, external +27.06** |
| Colour deletion / colour re-encoding / colour-space change | M1 E4 (+15.89 worse), M8 E2 (monotone), M8 E3 (1.79 spread) |
| Curve-basis rank above 3 | end-to-end 0.38–0.57 EMD for ranks 5/7/9/11 |
| Ensembling, bagging, blending, curve averaging | OOF error correlation 0.745–0.95 |
| Isotropic attenuation/expansion of predicted curves | M6-A: oracle-fitted scalar; nested `k̂` degraded 45.65→76.41 |
| Feature containment / clipping to the fit-set support | M6-2: raises error 41.15→41.91→43.88 |
| L1 loss matched to the metric's geometry | −28.84 EMD; per-column regression destroys the shared-basis variance saving |
| Fixed bias-vector correction | −1%; estimating it on 12 soils loses 3.17 |
| Tail-emphasised loss weighting | collides with the measured L1 null and the closed robust/boosted-regressor row |
| Alpha re-tuning | α=3 is the external argmin and the internal argmin; the curve is monotone either side |
| Per-column weighted-L1 head exploiting the metric's separability | measured −28.84 EMD, P(gain<0)=1.000 |

**What is genuinely still OPEN, with solo-feasibility notes:**
1. **More labelled soils.** The only intervention with a measured return (~2.9 EMD per doubling).
   Not achievable unilaterally — the host has said they are "discussing whether to release
   additional training and test data."
2. **M7-C — selector-only (`SEL-1SE` inside the frozen alpha grid).** Pre-registered, never run,
   and it costs zero new data. Cheap. But note the target: the identified alpha-selection cost is
   **1.68 EMD**, which is below the 5 EMD the public board can resolve.
3. **Camera augmentation** — synthesising additional "sensors" from existing images. E4's ranked
   option 3 and **the only idea that addresses the camera axis without new data.** Caution: M5
   closed *affine* adaptation, and the circularity rule applies (a treatment that alters the
   training distribution must be judged on a held-out level of the nuisance it targets), so the
   design must hold out a camera *family*, not just a soil.
4. **A fundamentally different model class or output parameterisation.** E3 explicitly could not
   speak to this; M2 only tested *frozen pretrained features + linear head*. **A parametric
   Weibull/log-normal two-parameter output head — the co-host's own published method and the
   dominant public-notebook motif — has NEVER been tried in this repo.** The closest we came is
   M8 E1's quadratic shape map, which was measured with *true* D50 and is worth +6.09 EMD but
   could not be reached through a predicted-D50 route.
5. **The band-ratio idea** (e4/e8 as blur-robust frequency surrogates) — untested, and the route
   back to the interpretability lost when `dom_wavelength_mm` was deleted.
6. **Reconsidering the label side without a D50 bottleneck.** M8 E1 proved the shape map is curved
   and that the *route* is the problem, not the target. Any design that fits curve shape
   *jointly* rather than in a D50-then-map sequence has not been tested.

**Hard constraints for any next design** (each of these has already cost this project a wasted
experiment):
- Decide arms **externally**, or not at all. Budget: the public board resolves only ≥5 EMD.
- Never let a gain be judged against zero: use the **one-parameter null** and the **best-of-N
  null (15.29 EMD)**.
- Any new feature block needs a **same-dimensionality noise comparator** run *before* CV.
- **Family-honest** folds; the soil is the unit; whole families held out.
- Reproduce the anchors (`43.453225201811563`, `41.1475158656982032`) to 0.0 before believing
  anything else, and prove the gate has **power** before trusting its verdict.
- Freeze arms, gates and thresholds in a written plan **before** implementation; a failed gate
  means stop, not tune.
- Banned inputs: camera, EXIF, ICC, ppm, site, sample_id, cv_group.

---

## 9. Prior art and public solutions (all `[U]` unless we reproduced them)

- **Soranzo, Guardiani & Wu (2025)**, *Soils and Foundations* 65(1) 101575 — MobileNet
  fine-tuned, images in a **dark chamber under constant lighting**, estimates **two Weibull
  parameters**, XAI on driving features. No metrics recoverable (publisher 403).
- **Soranzo (2026)**, *GRAI3*, Computers and Geotechnics 199 108362 — EfficientNetV2-S,
  **two-parameter Weibull** fit, handles cross-device by reading **EXIF** + a pixel-density
  calibration table, centre-crops to fixed resolution, verification model, MC-dropout
  uncertainty, **grouped CV** for generalisation (same principle as our family-honest LOFO).
- **Soranzo (2025)**, *Data in Brief* 60 111631 / Zenodo 14725633 — almost certainly the
  competition's source dataset; schema (`diam` mm, `finer` %) and material range match exactly.
- **St-Cyr et al. (2025)**, arXiv:2506.17469 — photogranulometry dataset, **12,714 images / 321
  soil samples**, finest scale 39.4 µm/px. **13× more samples than this competition.**
- **Manashti et al. (2023)**, arXiv:2303.04265 — 9,600 photos / 15 materials: RMSE 1.8% on a
  random split rising to **9.1% when each material is held out entirely — a ~5× degradation.**
  The external echo of this project's central finding.
- **Lang et al. (2021)**, *GRAINet*, HESS 25 2567 — ~30% of the 1.7 cm RMSE attributed to **human
  annotation noise**. Best-sourced evidence that image-derived PSD error floors are dominated by
  labelling, not model capacity.
- **Ibbeken & Schleyer (1986)**, *Photo-sieving* — the well-validated domain of photographic
  granulometry is particles **>10 mm**, i.e. centimetres, while this competition scores
  millimetre-and-finer boundaries. A strong prior that the fine end is unrecoverable in principle.
- **Countervailing literature we did NOT test:** a body of work holds that surface images are
  unrepresentative of the bulk sample at all (bedform reorganisation; winnowing; surface–subsurface
  divergence in gravel streams). The H374 label anomaly is at least consistent with it. Flagged,
  not asserted.
- **Public notebooks** (titles/scores only; cells are behind sign-in): V4 ridge 40.22;
  DINOv2+Sqrt-Mass Ridge 61.83; MultiTaskElasticNet 57.67; Ridge+ElasticNet+PLS 69.32;
  Physical Scale-Invariant Ridge 70.09; baselines 72.18; **ConvNeXt+Weibull 82.30**;
  Multi-Output LightGBM 90.48; Calibrated CV Ensemble 110.00. Two patterns: **Weibull
  two-parameter regression is the recurring motif**, and **frozen pretrained features + a linear
  head beat end-to-end CNN fine-tuning** (61.8 vs 82.3) — independently corroborating our Model 2.

---

## 10. Environment, code inventory, and how to run anything

**Local machine:** Python 3.14.6, numpy 2.5.2, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.9.0,
PIL 12.3.0. **No torch, no opencv, no scikit-image, no timm, no jupyter/nbformat.** GPU is an RTX
3050 laptop with 4 GB VRAM. Kaggle target is 2× T4 (16 GB each). `[M]`

Landmines that have each cost a run: `np.trapz` removed in numpy 2 (use `np.trapezoid`); local
stdout is **cp1252** so any non-ASCII character in a `print()` kills a run; Python 3.14 rejects
multi-line f-strings and `f"{x:g}"` on Pillow's `IFDRational`; **Kaggle mounts datasets three
levels deep** (`/kaggle/input/datasets/<user>/<slug>`) and resolves attachments
**alphabetically**, so an older dataset holding an older copy of a module silently wins;
`data/`, `*.zip` and `**/.cache/*.npy` are gitignored, so a fresh clone cannot re-run the deep
models without re-harvesting embeddings.

**Shared modules (read-only, reused not retyped):**
- `preprocess/` — the 11-stage pipeline, `config_hash 010f44c36c74`
- `Model 5/Model 5 Experiment 1/transfer_eval.py` — **the frozen primitives**: `FEATURES`,
  `ALPHAS`, `fit_predict` (scale → rank-3 basis → ridge → project), `emd_pair`, `project`,
  `image_features` (tile→image median), `soil_from_images` (image→soil median),
  `labels_matrix`, `nested_predict`, `clustered_bootstrap`
- `Model 8/Experiment 3/ruler.py` — the strictly-nested family-honest protocol; byte-copied into
  each new experiment so it stands alone
- `Model 8/Experiment 3/d50.py` — D50 extraction with the interpolation convention asserted

**Notebook builders and verifiers** live in `scratch/`: `build_m1e*.py`, `build_m2e*.py`,
`build_m5e1.py`, `run_m7a_gate.py`, `make_kaggle_zip.py`, `build_m10_kaggle.py`,
`check_m10_resolvers.py`, `check_m7_prereg.py`, `probe_feature_bound.py`, `probe_head_spread.py`,
`reconstruct_m63_rule.py`. Convention: cell text lives in a Python builder that emits the
`.ipynb`; the flattened `.py` is what gets executed, so the artifact judged is the artifact run.

**Model 10 reproduction, from a clean checkout of this repo:**
```
cd "Model 10/Experiment 1"
python image_features.py          # G0b
python build_m10e1.py gates       # G0a + G0b + G1   (~20 s CPU)
python build_m10e1.py score       # arms A/B/C, placebo, bootstrap   (~51 s)
python build_m10e1.py submit      # fits B and C, asserts the contract, writes 2 CSVs
python build_m10e1.py rerun       # cold re-run + determinism hashes + verdict + report
```
Kaggle mirror: `Model10_Experiment1.ipynb` + `final_kaggle_upload_m10e1.zip` (0.84 MB, 18 files,
**no image pixels**). The zip mirrors the repository layout so `ROOT = HERE.parents[1]` resolves
with zero edits to frozen code; the notebook copies the tree to `/kaggle/working/m10_run` and runs
the same three CLI stages as subprocesses, then asserts every artifact hash against the local run.
Verified against fabricated mounts at 3-deep and 1-deep, an incomplete attachment, a
nothing-attached case, and a stripped `.cache/` directory. **It has not been executed on Kaggle's
own image**, so library-version drift could legitimately change artifact hashes.

**Rules of engagement this project follows** (owner-set, and the reason for the discipline above):
one self-contained runnable artifact per experiment; a written plan frozen before implementation
with numbered tasks approved one at a time; **stop at a failed gate and report raw numbers**;
after a failed gate the freeze is total — no tuning, no reinterpretation; null results are
results; never cite an unmeasured number; internal CV and external Kaggle scores are never
subtracted; commits only on explicit say-so.

---

## 11. Known contradictions inside this repo's own record

Flagged rather than resolved — trust the artifacts over the prose, and note these if a downstream
plan relies on them.

1. **`AGENT_BRIEF.md` §11 discloses** that the recorded "nested" alpha rule selects alpha from
   inner folds computed over all 24 soils, so each fold's inner CV has seen the held-out family —
   it is **not strictly nested**. Reproduced rather than fixed, for comparability with Models
   1/2/5. The strictly-nested value is `43.453225201811563`; the looser historical anchor is
   `43.0217308796477`; they differ by 0.4315 EMD.
2. **`Model 1 Experiment 2/instructions.txt:92`** states E1's CAM prediction as 116.8 (factor
   1.48); **`Experiment2.txt:82,110`** state CAM+RES 205.34 (factor 0.84, CAM 195.95 → 0.88).
   The instructions figure matches nothing else in the record.
3. **`model_plan_exp2.md:164`** says aggregation is a tile→soil arithmetic **mean**; every
   experiment record says two-stage tile→image→soil **median**. The median version is the one that
   reproduces the anchors.
4. **`report.md` §6.3 / `audit.md:73`** carry the **retracted** frame-dependent sigma measurement
   and the "+63–124% texture gap" / "camera explains up to 71%" claims. §7 of `report.md` corrects
   them; the camera share for bands carrying soil signal is ~1.2–1.6%, not 71%.
5. **`AGENT_BRIEF.md:325`** calls the bands "DoG"; `preprocess/verify.py:108` implements a
   **Laplacian pyramid**. Same intent, different operator.
6. **`config.py:69 PROXY_SHORT_SIDE = 480`** is never used — the mask runs at full canonical
   resolution. **`MAX_ACCEPT_REMOVAL`** (the 40% rejection guard) is declared but not wired into
   `soilmask.py`; the guard that actually fires is `MIN_CROP_SOIL_FRACTION = 0.85`.
7. **`target_ppm.json` stores 4.5525162197231825** where other artifacts carry
   4.552516219723183 — a cosmetic float-representation difference, but it is why `config_hash`
   must be read rather than compared by eye.
8. **M6-3's alias figures** (65.7 / 28.1 / 1.65 / 1.58) came from worst-8/best-8 groups with an
   **unconstrained** nearest neighbour that allows same-CV-family siblings, plus **hand-typed**
   pairs for the separability leg. The stricter rule registered for M7-A gives 56.1 instead. Both
   are now asserted by the checker so they cannot be conflated.
9. **`Model 1/instructions.txt:295`** lists `Submission_Model1_E2.csv` twice and omits E4's file.
10. **`project-soil-metric-and-baselines`** (and any older prose) still says the best external is
    60.56167 from six submissions. That was superseded by Model 8 and is now **16 submissions,
    best 55.80591**.

---

## 12. One-paragraph brief for the next model

Twelve handcrafted features (5 Laplacian-pyramid texture bands + 7 colour statistics), aggregated
tile→image→soil by two median stages, fed to a StandardScaler → rank-3 PCA label basis →
closed-form multi-output ridge at α=3 with a monotone projection, validated by strictly-nested
family-honest leave-one-family-out CV over 16 curve-distance families on **24 labelled soils
(~8–10 effectively independent)**. Best Kaggle public score **55.80591**, which is outside the
top 49 of ~400 teams, while frontier image models reportedly score ~10 and no-image approaches
~29. **Ten model-side levers and six feature/representation levers are measured-null or
refuted**, including pretrained features, camera adaptation, richer aggregation, within-tile
statistics, output-basis rank, loss geometry, ensembling, calibration, containment and — most
recently and most decisively — per-image data expansion (internal −6.63 EMD, external **+27.06**).
The project's durable finding is that **every internal ruler measures interpolation while
deployment is extrapolation across a camera split (0 iPhone training images, 35 test images)**, so
arms that differ in feature space, granularity or model class **can only be ranked by
submission**, and the public board resolves only differences ≥ ~5 EMD. The learning curve says the
head is data-limited with ~8 EMD of headroom on the Android transfer curve. **Untried and
specifically open: a parametric two-parameter Weibull/log-normal output head (the co-host's own
published method and the dominant public-notebook motif), joint shape fitting that avoids the
D50-then-map bottleneck that killed a measured +6.09 EMD shape correction, camera augmentation as
the only route at the unmeasurable camera axis, and M7-C, a pre-registered zero-cost selector
experiment that was never run.**
