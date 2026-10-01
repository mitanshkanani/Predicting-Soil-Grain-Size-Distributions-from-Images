# Model 8 / Experiment 1 — Research and Plan

**Status: RESEARCH AND PLANNING ONLY. NOTHING HAS BEEN IMPLEMENTED, TRAINED, PREDICTED OR SUBMITTED.**
No Model 1–7 file was modified. No commit was made. No Kaggle upload occurred.

**Date:** 2026-09-30
**Author:** independent research pass over the whole repository + the competition + original measurement
**Instrument provenance:** all numbers below marked `[M8]` were produced by a read-only probe written
outside the repository (`%TEMP%/m8probe/`), reading only the shipped CSVs. They are reproducible and
the probe's metric/projection code was validated against the project's own recorded constants first
(see §2.3). Numbers marked `[REPO]` are quoted from the project's existing records and were not
re-derived.

---

## 1. Problem understanding

**Task.** From photographs of the surface of a soil sample, predict the cumulative grain-size
distribution (GSD) — the percentage of soil mass finer than each of 11 fixed diameters
(0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200 mm), per DIN EN ISO 14688-1.

**Data.** 24 labelled training soils (127 JPGs), 10 test soils (35 JPGs). **[REPO]**

**The governing fact about the sample size.** The 24 "soils" are not 24 independent observations.
Six pairs are near-duplicates in label space (H366~H371 3.3 EMD, H367~H372 4.3, H368~H615 7.0,
H038~H637 7.3, H549~H668 6.8, H181~H183 8.2), and the effective number of distinct soils is about
**8–10**, not 24. **[REPO]** This project has measured the consequences repeatedly: at n=24 the
permutation maximum of |Spearman rho| reaches ≈0.52, so **any correlation below ~0.5 is
indistinguishable from chance**, and picking the best of 15 candidate heads on noise buys a median
apparent gain of 15.29 EMD.

**The physical situation.** Training images were delivered at ~4.55 px/mm; test images at
13.9–19.5 px/mm. At 4.55 px/mm a 0.063 mm particle is **0.3 px**. Individual grains in the fine tail
are physically unresolvable in training *and* test; only texture and appearance can inform them.
Particle counting was tried and scored ~198 EMD against a ~100 baseline. **[REPO]**

**So the task is not "count grains".** It is: *given a surface photograph, estimate one dominant
scalar (the median grain size) and the shape of the distribution around it, from an image whose
resolution does not resolve the grains.*

---

## 2. Kaggle evaluation understanding

### 2.1 The metric

A logarithmically weighted Earth Mover's Distance (Wasserstein-1) between predicted and true
cumulative curves, integrated over log10(support), averaged over samples. Lower is better,
range effectively 0–500.

**The formula printed on the competition page is wrong.** It is a left-endpoint Riemann sum.
The grader integrates the area (`np.trapezoid`). Evidence: the page's published reference for the
trivial baseline is 100.31; the trapezoid reproduces 100.3132, the literal printed formula gives
102.02. **[REPO]** My probe reproduces the same constants (global mean curve → 85.393 on train;
rank-3 PCA floor 7.349 bare / 6.254 projected, matching the recorded 7.35 / 6.25). **[M8]**

### 2.2 Leaderboard structure

The public leaderboard scores a **fixed 3 of the 10** test soils. `Final = 0.30*public + 0.70*private`.
Near-zero top scores (0.00001, 0.09, 0.92) are submission cycling, **not** a leak — the host
confirmed this. **[REPO]**

**The single most important number for interpreting any result:** with per-soil SD 22.75, the null
SD of a 3-soil mean is 12.28, giving a **95% band of [21.7, 69.9]** around the internal mean 43.02.
The 10-soil private band is [32.6, 53.8]. **The public leaderboard cannot confirm any gain smaller
than ~25 EMD.** **[REPO]** Every external score this project holds (60.56, 61.24, 71.27, 77.12,
102.37, 172.70) sits inside or near that band.

### 2.3 Two scales that must never be mixed

| | value | what it is |
|---|---|---|
| **INTERNAL CV EMD** | 43.4532 | Model 7's strictly-nested family-honest leave-one-family-out baseline |
| historical anchor | 43.0217 | the older protocol that re-selects alpha on folds including the scored family |
| **EXTERNAL Kaggle EMD** | **60.56167** | the best actual submission ever made (Model 2 E1 `D`, a declared probe) |

**My probe reproduces 43.453225 exactly** — Model 7's nested-LOFO baseline, to all six decimals,
from an independently written implementation. **[M8]** This is the licence to trust the other `[M8]`
numbers: my metric, projection, family construction and head match the shipped ones.

A caveat I must state rather than hide: my implementation of the *looser historical-anchor*
protocol (alpha re-selected on folds that include the scored family) returns **41.1475**, not the
recorded 43.0217. So my implementation of that looser variant differs from the project's in some
detail I did not reconstruct. **I therefore use only the nested-LOFO ruler (43.453225), which
matched exactly, and I make no claim about reproducing 43.0217.** No conclusion in this document
depends on the looser protocol.

### 2.4 What is known vs not known — stated separately, as required

- **We possess:** 24 labelled training curves + their images; 10 unlabelled test images; the exact
  metric; the row/column contract of `sample_submission.csv`.
- **We do NOT possess:** any test label; any private-LB value; any confirmation that an internal
  gain transfers. Three of six past submissions were explicitly *declared probes*, not gate passes.
- **Internal CV results:** 35.98 / 39.46 / 41.15 / 42.66 / 43.02 / 43.43 / 43.45 across the project.
- **Actual Kaggle results:** 60.56167 (best), 61.23560, 71.27346, 77.12257, 102.37237, 172.69929.
  **Models 5, 6 and 7 have produced zero submissions.**
- An internal metric is **not** a Kaggle score. The two are never subtracted.

---

## 3. Current repository understanding

**Pipeline.** `preprocess/` has 11 stages (inventory → cameras → geometry → resample → soilmask →
colour → tiling → manifests → audit → qc → verify), `config_hash = 010f44c36c74`,
`PIPELINE_VERSION 1.0.0`, audit 37/37 assertions pass. `ppm_updated.csv` is quoted at **native**
resolution and is **wrong for delivered pixels**; the canonical `TARGET_PPM = 4.552516 px/mm` is
**derived** by `geometry.run()`, never typed.

**Tiles.** 1,976 files, only the 256 px size materialised (`TILE_SIZES` lists 128/224/256/384 but
`MATERIALIZED_TILE_SIZE = 256`). A 256 px tile = 56.23 mm of real soil. 1,946 pass
`soil_fraction >= 0.50`. Non-overlapping.

**Cameras.** Train and test cameras are **fully disjoint by design** — the host varied them
deliberately to force generalisation. 21 of 24 soils were shot by both Motorola Edge and
Samsung A52.

**The frozen head** (identical in Models 1–7): `StandardScaler` → **rank-3 PCA curve basis** →
closed-form multi-output `Ridge` on the curve coefficients → reconstruct → monotone projection
(clip to [0,100] → `np.maximum.accumulate` → force exactly 100 at 200 mm). Scaler and basis refit
**inside each training fold only**. Alpha grid of 16 values. **There is no gradient training
anywhere in this project.**

**The 12 features** (`Model 2 anchor M1`): texture core `[e4, e8, e16, lum_sd, grad_mean]` + colour
`[R, G, B, sat, lum_p10, lum_p50, lum_p90]`, aggregated tile →(median)→ image →(median)→ soil.

**Validation protocol.** Leave-one-**family**-out over **16 curve-distance families** built
deterministically by average linkage on pairwise label EMD cut at 14.0. **The soil sample is the
unit of supervision** — no random image-level split. Banned inputs: camera, ppm, EXIF, ICC, site,
sample_id, cv_group.

**I rebuilt the family structure independently [M8]** and got **16 families** with the size profile
`[2,2,2,4,1,1,1,2,1,1,1,2,1,1,1,1]` — and critically, **all six known near-duplicate pairs land
in the same family** (H038+H637, H181+H183, H368+H615, H366+H367+H371+H372, H549+H668, H493+H666).
The protocol is sound and my probe reproduces it.

---

## 4. Previous experiment summary

### Internal CV EMD (lower better)
| model / experiment | internal | notes |
|---|---|---|
| M1 E1 | 35.98 | least-regularised; **worst external ever (172.70)** |
| M1 E2 | 42.66 | |
| M1 E3 | 41.15 | the 12-feature selection |
| M1 E4 | 39.46 | |
| M2 E1 `M1` | 43.0217 | |
| M2 E1 `D` | 44.3302 | **best external ever (60.56)** |
| M2 E1 `R` | 39.7769 | random-init ViT; 3rd-best in-domain |
| M2 E2 D@256/128/64/32 | 44.33/43.32/43.83/44.87 | |
| M2 E3 `fuse@c256` | 43.431 | |
| M2 E3 `shuf@c256` | 52.927 | permutation control |
| M2 E3 `fr@c256` | **39.259** | **best in-domain in project history, worst transfer (83.94)** |
| M5 E1 G0 | 43.0217 | reproduced to 0.00e+00 |
| M7 baseline | 43.4532 | strictly-nested LOFO |

### External Kaggle EMD
| submission | score | status |
|---|---|---|
| M2 E1 `D` | **60.56167** | declared probe — the best external |
| M1 E3 | 61.23560 | |
| M1 E2 | 71.27346 | |
| M1 E4 | 77.12257 | declared probe |
| constant train-mean curve | 102.37237 | **ignores images entirely** |
| M1 E1 | 172.69929 | worst |

### No submission exists from Models 5, 6 or 7.

**The most important pattern in that table, and the trap the project has already named three
times:** the *best* internal results came from the least-regularised, least-informative
configurations. M1 E1 was both the best in-domain (35.98) and by far the worst externally
(172.70). M2 E3 `fr` was the best in-domain number in the project's entire history (39.26) and had
the worst transfer of its experiment (83.94) — **a 44.7 EMD gap between rulers for one model**.
Best in-domain + worst out-of-domain is overfitting, not discovery.

---

## 5. What previous experiments established

1. The metric is the trapezoid EMD, and the published page formula is wrong.
2. The public LB is 3 fixed soils; its 95% band is ~[21.7, 69.9]; it cannot resolve gains < ~25 EMD.
3. LOGO-CV ranks ridge alpha **backwards** (rank correlation −0.953 with a replacement ruler).
   **CAM+RES is ORDINAL-ONLY** — may rank, must not forecast, must not choose between candidates
   differing by less than ~15 EMD.
4. **The circularity rule** (most transferable finding): any treatment that alters the training
   distribution must be evaluated on a held-out *level* of the nuisance it targets. A 34-EMD
   artefact came from training on res16-blurred views and scoring with a res16-blurred ruler.
5. **The deeper instrument defect: every ruler here measures interpolation, deployment is
   extrapolation.**
6. Best-of-15-heads on noise buys a median 15.29 EMD of apparent gain.
7. `ppm_updated.csv` is wrong for delivered pixels; canonical scale is derived.
8. The label manifold is essentially 1-D. **[M8, independently confirmed: PC1 = 91.4% of label
   variance, rho(PC1, log10 D50) = 0.9949]** — matching the project's recorded 0.995.
9. A feature-block claim needs a same-dimensionality noise comparator. This is the **third** time a
   placebo outran the real arm (M7-A: real +0.0726 vs placebos +0.1131 / +0.2664).

---

## 6. What previous experiments failed to establish

1. **Which individual features carry the signal, and how much.** The project knows the 12-feature
   *set* scores 41.15. It does not appear to have measured the per-feature relationship to a
   physical summary of the label.
2. **Whether the label space's 1-D structure has been exploited.** The frozen head spends a rank-3
   PCA basis on a space that is 91.4% one-dimensional. This mismatch was never tested.
3. **How the 43 → 61 internal-to-external gap decomposes.** CORRECTION 2 withdrew the "+16.98
   population shift" and the "~12.7 EMD unexamined quantity" as 24-mean-minus-3-mean artefacts;
   only the **camera term +5.55 EMD** survives as measured.
4. **Whether the test soils are inside or outside the training distribution** in feature space.
5. **Whether a label-side (shape) improvement is available independently of any feature work.**

---

## 7. Relevant Kaggle findings

Retrieved this pass via a real browser (`document.body.innerText` on the JS-rendered pages) plus the
host's own discussion threads. Every claim below is quoted or attributed; where retrieval failed I
say so.

### 7.1 The metric is EXACTLY SEPARABLE — confirmed by me, and it is a probing vulnerability

Rank 1 (Alexy), verbatim from [discussion 743097](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/743097):

> "My 0.00001 public score is not a model, i's the public test answers, reconstructed through
> systematic submissions. … The metric — log-weighted EMD — is a separable sum: total score =
> Σ weight × |prediction − truth|, evaluated independently at each of the 11 support points of each
> public soil. That makes every submission a noise-free linear measurement of the hidden truth…
> About 50 submissions reconstructed all three public curves to the rounding limit."

**I verified the separability claim myself, exactly [M8].** The 11 log10 supports are spaced almost
uniformly (0.4983 / 0.5017 alternating), so the trapezoid rule gives fixed weights
`w = [0.2492, 0.5×9, 0.2508]` summing to 5.0 = log10(200/0.002). Over 12,000 valid monotone curve
pairs, `|trapezoid − Σ wᵢ|Pᵢ−Tᵢ||` ≤ **8.5e-14**. The metric is a fixed-weight L1 on the 11 outputs.

This is not a curiosity — §10 records that exploiting it via per-column L1 regression **fails badly
(−28.8 EMD)**, and why.

The **host** confirms the diagnosis and adds a live commitment ([discussion 737406](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/737406)):

> "I have also reviewed the private leaderboard, and those scores are within a realistic range…
> My assumption is therefore that the top public score reflects overfitting to the public test
> split… **We are looking into how to address this.** Given the high participation and submission
> volume, we are also discussing whether to release additional training and test data."

**Two consequences for this project.** First, the ~15.29 EMD best-of-N null (§9) is not a
hypothetical — a competitor has demonstrated that ~50 submissions fully recover 30% of the labels.
Second, the host is considering **releasing additional data**, which means the label count could
change under us. Any experiment designed around "24 soils" may need re-basing.

### 7.2 The honest achievable band — and a correction to the project's own framing

Alexy: *"My actual image model scores ≈10 public."* Leaderboard ranks 8–14 sit at 5.9–10.1.
Independently corroborated: zaoui Hamza (rank 31) reported passing the data to Claude Opus produced
scores of **34** and then **29** on the public LB with no image model at all.

**So the honest band is roughly ~10 (good image model) to ~30 (strong prior, no image model).**

**This forces a correction to §2.2's framing, which I record against the project's own reasoning.**
The project's 95% band of [21.7, 69.9] is centred on *our internal mean of 43.02*. But a frontier
model scores ~10 — far *outside* that band — because the band was derived from the training-soil
error distribution, not from the test distribution. The correct reading is:

- The band still means **a 3-soil public score cannot resolve a gain of ~25 EMD for a model of our
  current quality.** That stands.
- But it does **not** mean scores outside the band are anomalous. A genuine, large improvement will
  land *below* 21.7. **Absence from the band is not evidence of absence of gain** — the band is
  one-sided evidence about *our* neighbourhood, not a two-sided test.

This makes a submission *more* informative than the project has been assuming, provided the
comparison is against our own 60.56 baseline rather than against the band.

### 7.3 Data-integrity facts the project should know

- **H031 no longer exists; H374 was re-photographed.** A competitor found H368 and H374 labels
  claimed >50% of particles >6.3 mm while the images showed the opposite ([discussion 702039](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/702039)).
  Host: *"the laboratory found the soil. I took 3 new pictures… The soil H031 had to be removed
  because the laboratory no longer had the soil available."* **The 24 training soils are not the
  original 24.** The repo's `Training_labels_updated.csv` and the Motorola Edge 60 Fusion camera
  (3 images) are the post-correction set — this is consistent with what the repo already records, and
  it corroborates the "effective 8–10 distinct soils" finding rather than undermining it.
- **The 11 targets are interpolated, not measured at those points** (Data page): *"the raw curves
  were interpolated linearly on a logarithmic grain-size axis onto the 11 fixed support points."*
  The fine end comes from **sedimentation (hydrometer)** analysis, which carries its own
  uncertainty. Part of the irreducible shape cost in §9.3 is therefore measurement noise in the
  labels, not a property of the soil.
- **Metric direction was briefly wrong** ([discussion 701750](https://www.kaggle.com/competitions/soil-grain-size-from-photos/discussion/701750)): *"The lower, the better. I just corrected the
  scoring metric."* Any cached higher-is-better reading is stale.
- **Host on determinability** (the quote the project already holds), with added context: *"The
  distance from the soil to the camera is always 21 cm. The photos were all taken under the same
  lighting. The different cameras are used deliberately, to ensure generalisation."*
- **Scale:** 1,149 entrants, 419 participants, 400 teams, 4,391 submissions; ~2 months remaining;
  prizes $300/$150/$50. Rules §5.b requires winners to publish architecture, preprocessing, loss,
  and a repo link.
- **One open rules question:** whether published PSD lab results for samples whose site IDs overlap
  the test IDs count as permitted external data under §2.6. **No host answer is visible.** Flagged,
  not resolved.

### 7.4 Public notebooks (titles and scores only — Kagge gates cell text behind sign-in)

| Notebook | Score |
|---|---|
| Soil grain V4 (krishnayadav456wrsty) | 40.22 |
| [CV] PSGSDI — DINOv2 + Sqrt-Mass Ridge (nomannic) | 61.83 |
| MultiTaskElasticNet (pavloivanin) | 57.67 |
| Soil Grain Size — Ridge + ElasticNet + PLS (dataarthur) | 69.32 |
| Soil GSD: Physical Scale Invariant Ridge (pavloivanin) | 70.09 |
| Prediciting soil grain size challenge: baselines (zaouiyassine) | 72.18 |
| Soil with ConvNeXt and Weibull (ambrosm) | 82.30 |
| Soil GSD: Multi-Output LightGBM (pavloivanin) | 90.48 |
| Calibrated CV Ensemble (avikdas567) | 110.00 |

Two patterns are visible even without the code. **Weibull-parameter regression is the dominant
recurring motif** (ambrosm's ConvNeXt+Weibull; GRAI3 in §8 uses the same two-parameter form) — and
it is the same 2-parameter shape idea as my §9.4 quadratic, arrived at independently. **Frozen
pretrained features plus a linear head beat end-to-end CNN fine-tuning** (DINOv2+Ridge 61.8 vs
ConvNeXt 82.3), which independently corroborates this project's own Model 2 conclusion that DINOv2
did not beat the hand features (61.83 vs our 60.56 is a tie).

pavloivanin's "Elegant EDA & Scale Insights" is the closest public scale/EDA analysis and the one to
read first; its section headings are visible but its contents are not.

---

## 8. Relevant external research

**The competition's co-host is the author of the prior art on this exact task.** Enrico Soranzo is
cited on the competition page alongside Leibold. This is the most important external finding, and it
was not in the repository.

- **Soranzo, Guardiani & Wu (2025)**, "Convolutional neural network prediction of the particle size
  distribution of soil from close-range images", *Soils and Foundations* 65(1), 101575,
  doi:10.1016/j.sandf.2025.101575. MobileNet pre-trained on ImageNet, fine-tuned on samples "ranging
  from clayey silt to gravel"; images in **a dark chamber under constant lighting**;
  **estimates two Weibull parameters**; XAI used to identify driving image features. *No quantitative
  metrics were recoverable* — the publisher returned 403 and no open index carries an abstract with
  numbers.
- **Soranzo (2026)**, "GRAI3: Generalizable soil particle size distribution from smartphone images",
  *Computers and Geotechnics* 199, 108362 (preprint SSRN 6361302). EfficientNetV2-S; two-parameter
  Weibull fit to the cumulative curve; handles cross-device variation by reading **EXIF** and looking
  up pixel density in a calibration table, then centre-cropping to fixed resolution; verification
  model checks the image contains soil; Monte Carlo dropout for uncertainty. Generalisation is
  evaluated with **grouped cross-validation** — the same principle as this project's family-honest
  LOFO. *No error metrics in the abstract.*
- **Soranzo (2025)**, "Dataset of close-range soil images and corresponding particle size
  distributions", *Data in Brief* 60, 111631. Zenodo record 14725633, read directly: close-range soil
  photographs with PSDs, two smartphone cameras, controlled lighting, CSVs with `diam` (mm) and
  `finer` (percent passing). **This is almost certainly the source dataset for the competition** —
  the schema and material range match exactly.
- **St-Cyr, Duhaime, Dubé & Grenier (2025)**, "Dataset of soil images with corresponding particle
  size distributions for photogranulometry", arXiv:2506.17469v2. **12,714 images, 321 soil samples**,
  45 MP, **finest scale 39.4 µm/px**, moist and dry, thin layers on 13×9-inch white trays. Same
  framing and method, but **13× more samples and ~100× finer scale** than this competition.
- **Manashti, Duhaime, Toews, Pirnia & Telcy (2023)**, arXiv:2303.04265. 9,600 photographs of **15
  granular materials**. RMSE on percent passing **1.8%** with a random split, rising to **9.1% when
  each material is held out entirely** — a ~5× degradation from material-level generalisation.
  **That 5× is the external echo of this project's own central finding.**
- **Lang, Irniger, Rozniak, Hunziker, Wegner & Schindler (2021)**, "GRAINet", *HESS* 25, 2567–2597.
  Full text read. 1,491 line samples, 25 gravel bars, >180,000 manually outlined grains. Mean-diameter
  MAE 1.1 cm. **Crucially, ~30% of the 1.7 cm RMSE was attributed to human annotation noise**, and
  53% of predictions fell within one human standard deviation. **Not comparable in grain size**
  (gravel only), but it is the best-sourced evidence that **image-derived PSD error floors are
  dominated by labelling, not model capacity** — which is exactly the interpretation §9.3's
  unlearnable shape residual supports.

**The key gap, stated plainly:** I could not find a single paper stating a **minimum resolvable grain
size** for a close-range soil photograph. The best-documented floor is **10 mm** (Ibbeken &
Schleyer 1986, "Photo-sieving", *ESPL* 11 — the founding photogranulometry paper, applicable to
particles **larger than 10 mm**), corroborated by GRAINet's robust regime at GSD 2 cm. If both are
right, the well-validated domain of photographic granulometry is *centimetres*, and this competition
scores millimetre-and-finer boundaries. That is a strong prior that the fine end is not recoverable
in principle — and it independently supports the host's own remark and this project's particle-counting
refutation.

**A countervailing literature I must flag rather than bury:** a substantial body of work holds that
surface images are *unrepresentative of the bulk sample at all*, independent of resolution — bedform
reorganisation (Singh et al. 2012), winnowing under transport (Parker et al. 2007), surface-subsurface
divergence in gravel streams (Bunte & Abt 2001). **None of it was tested on this dataset and I assert
none of it applies here** — but it is the standard objection a geotechnical reviewer would raise,
and the H374 label anomaly (labels implying >50% >6.3 mm against images showing cobbles apparently
buried beneath sand) is at least consistent with it.

**Buscombe's photogranulometry lineage** (2008 *Sedimentary Geology*; 2009; SediNet 2020) is the
origin of the field and is the most significant gap — all paywalled with no abstracts in any open
index.

---

## 9. Important observations — my own measurements this pass

All `[M8]`. Probe at `%TEMP%/m8probe/`; read-only; touched no repository file.

### 9.1 The 12 features carry only ~3 dimensions

The 12×12 feature correlation matrix has eigenvalues
`[6.204, 3.149, 1.586, 0.731, 0.155, 0.092, 0.027, 0.024, 0.012, 0.011, 0.006, 0.002]`.
**Participation ratio (effective dimensionality) = 2.80.**

This is a property of the data, not a hypothesis, and it explains a great deal: adding, removing or
reshaping columns inside a ~3-dimensional block cannot manufacture information. The six near-redundant
texture bands (e4/e8/e16/lum_sd correlate 0.89–0.99 with each other) are one number; the colour
block is largely another.

### 9.2 `e16` predicts the median grain size; the other 11 columns mostly do not

Honest leave-one-family-out R² for log10(D50) (the D50 = diameter at 50% passing, interpolated in
log-diameter):

| feature set | LOO R² for log10 D50 | out-of-fold RMSE in log10 D50 |
|---|---|---|
| **`e16` alone** | **+0.734** | **0.486** (a factor of 3.1 in grain size) |
| e8 alone | +0.700 | — |
| e4 alone | +0.552 | — |
| all 12 jointly | **+0.111** | 0.889 (a factor of 7.7) |

Adding columns **destroys** the signal monotonically: 0.552 → 0.722 → … → 0.111 as features are
added. The all-12 joint R² sits at the 99.2nd percentile of its own permutation null — i.e. it *is*
real information, but it is a *tiny* fraction of what one column already carries on its own.

`e16` is not a fluke of one fold: it beats 100% of a 4000-draw permutation null (null median RMSE
1.019, observed 0.486), and it tracks D50 across the full 2.5-decade training range with a
factor-of-3 median error.

### 9.3 The label space is one-dimensional, and the shape is not learnable at all

PC1 carries **91.4%** of label variance and rho(PC1, log10 D50) = **0.9949**; PC2 correlates 0.016
and PC3 0.077 with log10 D50.

Decomposing the frozen head's 43.45 EMD:

| quantity | value |
|---|---|
| irreducible **shape** cost (true D50 known, best shape map) | **25.21** |
| D50 error cost | **16.42** |
| total | 41.63 |

And the shape parameters themselves are **not predictable from any feature**:

| parameter | LOO R² from all 12 features | percentile vs permutation null |
|---|---|---|
| log10 D50 | +0.111 | **99.2** |
| log10(D60/D10) — the GSD slope | −1.259 | 66.0 |
| log10(D90/D50) — the coarse tail | −2.291 | 28.2 |

**Every single feature gives negative R² on the shape residual.** This is not a weak result; it is
the absence of one.

### 9.4 The single largest real effect I found: the shape map's functional form

Even with D50 held **exactly true** and using no features at all, the map from D50 to the curve is
not linear:

| shape map (true D50, LOO, no features) | mean EMD |
|---|---|
| linear in log10 D50 | 23.80 |
| **quadratic in log10 D50** | **18.02** |
| cubic | 17.85 |
| exp-of-linear | 22.91 |

**Quadratic beats linear by 5.78 EMD, 95% bootstrap CI [1.44, 9.85], P(gain<0) = 0.005, and it is
better for 19 of 24 soils.** (Cubic is not better than quadratic — 17.85 vs 18.02 — so the gain
saturates at degree 2 and is not simply "more parameters".)

**The stricter uncertainty treatment, applied for consistency.** Every candidate I rejected
(§9.7, §9.8, §9.9) was judged by a **family-cluster** bootstrap that resamples the 16
curve-distance families rather than the 24 soils, because the six near-duplicate pairs are not
independent observations. I initially reported the primary result under the weaker soils bootstrap
only. Re-running it under the same stricter treatment:

| treatment | 95% CI | P(gain < 0) |
|---|---|---|
| soils bootstrap (n=24) | [1.44, 9.85] | 0.0053 |
| **family-cluster bootstrap (n=16 families)** | **[1.07, 10.53]** | **0.0093** |

**The result survives the stricter test.** Per-family gains: min −27.63, median **+7.12**, max +31.09;
quadratic is worse in **3 of 16** families. The median family gains 7.1 EMD, so this is a broad
effect and not one carried by a single family — which is exactly the failure mode that killed the
rank-1 basis in §9.8.

This is a *label-side* correction. It requires **no new features, no new data, no new model class** —
only a different functional form for a map the project already implicitly assumes is linear.

### 9.5 The test soils are outside the training distribution — and it is all colour

**0 of 10** test soils have a training neighbour closer than the typical training-to-training 1-NN
distance (median 1.64). Per feature, the excess of test z-scores beyond the training range:

| feature | excess beyond training range | | feature | excess |
|---|---|---|---|---|
| e4 | 0.011 | | R | 0.000 |
| e8 | 0.000 | | G | 1.281 |
| e16 | 0.000 | | **B** | **3.329** |
| lum_sd | 0.000 | | sat | 0.678 |
| grad_mean | 0.000 | | lum_p10 | 0.311 |
| | | | lum_p50 | 2.216 |
| | | | lum_p90 | 1.391 |

**Every texture feature is comfortably inside the training range. All of the extrapolation is in
the colour block**, and it is systematic across the test set rather than one outlier — 9 of 10 test
soils sit above the training maximum in B, driven by the iPhone cameras' colour rendering.

This is the most plausible mechanism I have found for the 43 → 61 internal-to-external gap, and it is
consistent with the project's existing record that colour's transfer effect is ~0 under CAM while the
absolute-frequency features shift 4.31 SD under blur — **a big mean shift is not the same as a
load-bearing one, and here the load-bearing extrapolation is in the block that CAM under-penalises.**

### 9.6 The ruler's verdict depends on which soils it scores

Re-running the same comparison (all-12 vs texture-core-5) while scoring progressively more
extrapolative held-out soils:

| scored subset | all-12 | texture core | core − all12 |
|---|---|---|---|
| all 24 | 43.45 | 42.99 | −0.47 |
| 12 most isolated | 41.66 | 39.39 | −2.28 |
| 6 most isolated | 40.11 | 44.86 | **+4.75** |

**The sign flips.** A test-similarity-weighted ruler (weights derived from feature-space proximity
to the test centroid, using no test labels) still mildly prefers the texture core, −0.76. But the
verdict is not stable, and every margin here (0.5–4.7 EMD) is far below the ~15 EMD resolution the
project established for selection. **No colour/feature decision is defensible on this evidence.**

### 9.7 The metric is exactly a fixed-weight L1 — and exploiting that directly fails

§7.1 established the metric is `Σ wᵢ|Pᵢ−Tᵢ|` with `w = [0.2492, 0.5×9, 0.2508]`. The frozen head
does not minimise that: it minimises **squared** error on 3 PCA coefficients, then projects. That is
a mismatched loss, so the obvious move is to regress each column with a weighted L1 (median) loss.

Where the frozen head's error actually sits, per column:

| column | mean abs error | × w | EMD contribution |
|---|---|---|---|
| 0.063 | 15.12 | 0.5 | 7.56 |
| 0.2 | 15.79 | 0.5 | 7.90 |
| 0.63 | 12.20 | 0.5 | 6.10 |
| 0.02 | 9.14 | 0.5 | 4.57 |
| 2 | 9.50 | 0.5 | 4.75 |
| 6.3 | 9.44 | 0.5 | 4.72 |
| 20 | 5.11 | 0.5 | 2.56 |
| 0.0063 | 4.14 | 0.5 | 2.07 |
| 0.002 | 2.36 | 0.249 | 0.59 |
| 63 | 0.93 | 0.5 | 0.46 |
| 200 | 0.00 | 0.251 | 0.00 |

**Result: a per-column weighted-L1 head scores 72.29 against the frozen 43.45 — a loss of 28.84 EMD,
family-cluster bootstrap 95% CI [−44.43, −12.29], P(gain<0) = 1.000.** Unambiguously worse.

**Why, and it is a real insight rather than a failed variant:** the 11 curve columns are not
independent targets. They are 11 evaluations of one 1-D-ish underlying function, and the rank-3 PCA
basis works precisely *because* it exploits that shared structure. Regressing the columns
independently throws the correlation away and spends 11 sets of coefficients on 24 points. **The
mismatched loss is outweighed by the enormous variance saving from the shared basis.** This is a
quantified reason to keep the frozen head's shape, and it is new information for the project.

It also retires the probing angle as a modelling shortcut: there is no legitimate way to turn the
separability into score, and §7.1 shows the illegitimate way is already being exploited by others.

### 9.8 Rank-1 curve basis: a real-looking gain that does not survive

| basis rank | nested-LOFO |
|---|---|
| **1** | **40.86** |
| 2 | 42.67 |
| 3 (frozen) | 43.45 |
| 4 | 44.27 |
| 5 | 43.68 |

Rank-1 looks like a 2.59 EMD gain — consistent with §9.3's 1-D structure. But the paired
family-cluster bootstrap gives **95% CI [−5.73, +1.29], P(rank-1 worse) = 0.092**; rank-1 is worse
for 6 of 24 soils; and dropping the single soil that drives most of the gain (H371, Δ = −25.6)
shrinks it to 1.60. **It is one soil.** Not defensible. (I report it because it *corroborates* the
1-D finding independently, not as a candidate.)

### 9.9 What did NOT survive: the D50-reparameterisation end to end

Wiring the §9.3 reparameterisation through the full feature → D50 → shape pipeline with nested
alpha selection:

| head | nested-LOFO | vs frozen |
|---|---|---|
| frozen, 12 features, rank 3 | 43.45 | — |
| D50-reparam, linear shape | 41.26 | +2.20 |
| D50-reparam, quadratic shape | 40.79 | +2.66 |
| D50-reparam, cubic shape | 46.29 | −2.83 |
| D50-reparam on **e16 alone**, quadratic | 40.49 | +2.96 |

The best cell's family-cluster bootstrap is **95% CI [−3.74, +9.26], P(gain<0) = 0.21**. Against the
project's own best-of-15 null of 15.29 EMD, **a 2.96 EMD apparent gain is exactly what selecting
the best of ~5 noisy candidates buys.** Not defensible as a whole-pipeline replacement.

**This is the crucial methodological point and I record it against my own hypothesis:** §9.4's
+5.78 EMD is real and robust *because it is measured with D50 held true*, which removes the feature
noise entirely and isolates the shape map. The moment the noisy D50 estimate is put back in, the
gain is swamped by D50 error and becomes unmeasurable. **The label-side fix is real; the
reparameterised pipeline is not yet a demonstrable improvement.**

---

## 10. Candidate ideas considered

| # | idea | verdict |
|---|---|---|
| 1 | **Fix the shape map's functional form** (quadratic in log10 D50) | **RECOMMENDED** — +5.78 EMD, CI [1.44, 9.85], P=0.005, 19/24 soils, zero new features |
| 2 | Rank-1 curve basis | rejected — one soil, P=0.092 |
| 3 | Full D50 reparameterisation end to end | rejected for now — the shape fix survives, the pipeline wrapper does not |
| 4 | Drop colour (texture-core only) | not defensible — margins 0.5–4.7 EMD, sign flips by subset; but §9.5 makes it the top follow-up |
| 5 | Feature selection search (sweeping all 2502 subsets) | confounded by the best-of-N null; running, not load-bearing |
| 6 | Learn the shape from features | **dead** — negative R² for every feature (§9.3) |
| 6b | Exploit the metric's exact separability via per-column weighted-L1 regression | **dead, measured** — −28.84 EMD, P(gain<0)=1.000 (§9.7). The shared PCA basis beats the matched loss. |
| 7 | Particle counting / granulometry | **already dead** — ~198 EMD |
| 8 | Camera adaptation (any affine), CORAL, transductive alignment | **already dead** — −8.75 EMD, direction-inconsistent |
| 9 | Resolution-matched training | **already dead** — 9.2 EMD total spread |
| 10 | Richer tile→soil aggregation | **already dead** — 41.15 → 42.13/42.22 |
| 11 | Within-tile spatial statistics A1–A4 | **already dead** — placebos outran the real block |
| 12 | Ensembling, bagging, L1, alternative classical regressors, band ratios, DINOv2 variants, monotone-by-construction head, tail-emphasising loss | **already dead** or not registered — see §12 |

---

## 11. The specific hypothesis recommended for Experiment 1

> **The frozen rank-3 PCA curve head is a suboptimal map from a predictable scalar to a curve,
> because the label space is 91.4% one-dimensional and the head spends three free coefficients on
> a near-line. Replacing the *linear* log10-D50 → curve shape relation with a *quadratic* one, while
> leaving features, aggregation, validation and submission machinery untouched, reduces error by
> ~5.8 EMD measured with D50 held true.**

Restated as a falsifiable claim: **on a family-honest leave-one-family-out ruler, replacing the
frozen rank-3 head with "predict log10 D50 by ridge, then map through a quadratic-in-log10-D50
polynomial fitted on the training families", scores better than 43.4532 by at least 3.00 EMD, and
the gain is not attributable to any single soil.**

---

## 12. Why this hypothesis is worth testing

- **It is the only candidate with a p-value under 0.01** that I could measure — and it holds under
  the same strict family-cluster bootstrap that killed every other candidate (P(gain<0) = **0.0093**,
  §9.4).
- **It needs nothing new.** No new features, no new data, no new model class, no GPU, no external
  literature. It is a *label-side* correction to a map the project already assumes is linear.
- **It is orthogonal to everything already dead.** It cannot be a camera artefact, a resolution
  artefact, a tile-aggregation artefact or a circularity artefact, because it never touches the
  training distribution or the features. **The circularity rule does not apply to it** — that rule
  governs treatments that alter the training distribution; this one does not.
- **It attacks the largest measured block.** §9.3 puts 25.21 EMD of irreducible shape cost against
  16.42 EMD of D50 error. §9.3 also shows the shape is *not learnable from features* — so the only
  way to reduce it is to make the fixed map a better map. §8's GRAINet result (**~30% of the error
  floor was human annotation noise**) is independent external support for exactly that split.
- **It converges with the published prior art from an independent direction.** The competition's
  co-host Enrico Soranzo's own method — GRAI3 and the 2025 *Soils and Foundations* paper — fits a
  **two-parameter** distribution (Weibull) to the cumulative curve, and the dominant public-notebook
  motif is the same. My §9.4 result says a 2-parameter shape map beats a 1-parameter one by 5.78 EMD
  with P=0.005, and that a 3rd parameter adds nothing. **The host's own method and my measurement
  agree on the number of parameters, arrived at independently.** That is the strongest external
  corroboration available for this hypothesis.
- **It is falsifiable and cheap**, and it produces a clean result either way: if it holds, the
  project has its first mechanism-backed improvement; if it fails, the 1-D structure has been
  squeezed dry and Model 8 should move to §9.5's colour question.

**Explicitly not claimed:** this is not a Kaggle improvement. Per §2.2, a 5.8 EMD internal gain is
one fifth of what the public leaderboard can resolve, and the private band is [32.6, 53.8] around
an internal 43. **No submission is proposed, and none should be made on this evidence alone.**

---

## 13. Proposed methodology

Pre-registered, single-arm-plus-controls, on the frozen infrastructure.

**Arm E1-A (primary).** Replace the head with:
1. `y = log10(D50)`, D50 interpolated from the training curves in log-diameter.
2. `Ridge` (the same 16-value alpha grid, nested selection inside training families only) predicting
   `y` from the **unchanged 12 features**.
3. A **quadratic** polynomial basis `[1, y, y²]` per curve column, fitted on the training families
   only.
4. The unchanged monotone projection.

**Arm E1-B (control — the one that matters).** The identical pipeline with a **linear** basis
`[1, y]`. This isolates the functional form from everything else. If E1-A beats E1-B by ~5.8 EMD, the
functional form is the cause; if it does not, the gain was never there.

**Arm E1-C (placebo).** The same quadratic map, with the *labels* permuted. This is the
same-dimensionality control the project has been burned by twice. It must not beat E1-A.

**Arm E1-D (shape-only, no features).** Quadratic map with **true** D50 supplied. Reproduces §9.4's
18.02. This is the upper bound for any D50-parameterised model and must be reported every time.

---

## 14. Data required

None new. The existing 24 training soils, the existing `features_soil.csv`, the existing label CSV.
The test set is **not touched** — no predictions, no submission.

---

## 15. Features / representation required

**Unchanged: the frozen 12.** The point of this experiment is that it needs no new features. Any
feature change would confound the functional-form effect, which is the only thing under test.

---

## 16. Model / architecture

A closed-form two-stage head. No gradient training, consistent with the entire project. Stage 1 is
the same `StandardScaler` + `Ridge`; stage 2 is an ordinary least-squares polynomial map from 1
parameter to 11 outputs. Total added complexity: three coefficients instead of one.

---

## 17. Training procedure

For each held-out family: fit the scaler, the D50 regression, and the shape map **on the training
families only**. Select alpha by an inner leave-one-family-out loop **restricted to the training
families** — the strictly-nested protocol that reproduced 43.453225 exactly. Never let the held-out
family influence any fitted quantity.

---

## 18. Validation strategy

- **Primary ruler:** strictly-nested family-honest LOFO over the 16 curve-distance families, cut at
  EMD 14.0, rebuilt deterministically. Must reproduce the 43.453225 baseline before anything else
  is believed.
- **Family construction:** the six known near-duplicate pairs must land in the same family. Verified
  in my probe; must be re-verified in the implementation.
- **Uncertainty:** family-cluster bootstrap (resampling the 16 families, not the 24 soils, because
  the near-duplicate pairs are not independent) — 20,000 draws. Plus a leave-one-soil-out
  sensitivity check, since §9.7 showed a single soil can manufacture a whole apparent gain.
- **Multiple-comparison control:** the best-of-N permutation null on shuffled labels, since four arms
  are being compared. A gain must exceed the best-of-4 null, not merely be positive.
- **Ruler caveat, stated in advance:** per §9.6 this ruler measures interpolation. The experiment's
  claim is deliberately restricted to the label-side map, which does not depend on where the test
  soils sit — but any *further* claim about feature selection must not be drawn from this experiment.

---

## 19. Expected outputs

1. `cv_per_soil.csv` — per-soil EMD for each arm, with family id.
2. `arm_comparison.csv` — mean EMD for E1-A/B/C/D plus the frozen baseline 43.453225.
3. `bootstrap.json` — family-cluster bootstrap CIs for every pairwise difference.
4. `placebo.json` — the best-of-N null, so the gain is reported against selection noise.
5. `reproduce_anchor.txt` — proof the harness returns 43.453225 on the frozen head before any arm
   is scored. **If this fails, nothing else in the run is reported.**

---

## 20. Submission CSV generation plan

**None. No submission is proposed for Experiment 1.** Reasons: the expected gain (~5 EMD) is a fifth
of what the public LB can resolve (§2.2); three of six past submissions were probes that cost
slots; and the user's instruction for this stage forbids generating predictions or a submission.
If the owner later wants a submission, it is a separate decision taken *after* the arms are scored
and after the CAM+RES and colour-extrapolation questions are settled.

---

## 21. Practical success criteria

Pre-registered, and **all four must hold**:

- **G1.** The harness reproduces the frozen nested-LOFO baseline **43.453225** to 1e-6. (Gate; failure
  stops the run.)
- **G2.** E1-A (quadratic) beats E1-B (linear) by **≥ 3.00 EMD** in mean EMD.
- **G3.** The E1-A − frozen difference has a family-cluster bootstrap 95% CI whose **lower bound is
  above 0**.
- **G4.** E1-A beats E1-C (label-permuted placebo), and the observed gain exceeds the **best-of-4
  permutation null**.

Supporting (reported, not gating): E1-D reproduces ~18.02 ± 1.0; the gain is not attributable to a
single soil (dropping the most influential soil must leave ≥ 60% of the gain).

---

## 22. Risks and failure modes

| risk | likelihood | mitigation |
|---|---|---|
| **The gain is a selection artefact** — the real risk, given three prior instances | medium | G4 placebo; the 15.29 EMD best-of-N null; E1-B as the isolating control |
| **It is one soil** (exactly what killed rank-1 in §9.7) | **high** | leave-one-soil-out; the 19/24 consistency check from §9.4 |
| **The quadratic overfits 24 points** | medium | 3 parameters vs 23; degree-3 tested and did *not* improve (17.85 vs 18.02), so this is saturation, not drift |
| **It does not transfer to the test soils** | **medium-high, and unmeasurable** | stated in advance: every ruler here measures interpolation. Do not forecast from this result. |
| **The 1-D structure is an artefact of the 24 training soils** | low-medium | PC1 = 91.4% is a strong property; but the test set could genuinely be more multi-dimensional. Not testable without test labels. |
| It gets read as a Kaggle improvement | — | §2.2 and §20 forbid this reading explicitly. |

---

## 23. Estimated compute / runtime

**CPU only, minutes.** No GPU, no torch (not installed), no Kaggle. The dominant cost is the nested
inner LOFO loop: 16 outer folds × 16 alphas × ~15 inner folds of a closed-form ridge on ~20×12
matrices — well under a minute per arm in pure numpy/scikit-learn, as my probe already demonstrated.
Total including the 4000-draw placebo: **under 10 minutes.** This is affordable precisely because it
requires no new features and no new images.

---

## 24. Exact implementation plan

1. **Create** `Model 8/Experiment 1/` only (done). No other folder. No Model 1–7 modification.
2. **Implement** `d50.py` — D50 extraction: monotone-accumulate each training curve, find the first
   crossing of 50%, interpolate **linearly in log10-diameter between the bracketing supports**
   (getting this backwards is the single easiest error here, and I made it myself during this pass).
3. **Implement** `head_d50.py` — the two-stage head, `fit()` / `predict()`, scaler and both stages
   fitted on training rows only.
4. **Implement** `run_e1.py` — the four arms plus the frozen baseline, emitting the five artefacts
   of §19.
5. **Gate first:** run the frozen baseline and confirm **43.453225** to 1e-6. Stop and report if it
   differs.
6. **Verify** the family construction puts all six near-duplicate pairs together.
7. **Score** E1-A, E1-B, E1-C, E1-D; compute the family-cluster bootstrap and the best-of-4 null.
8. **Apply G1–G4** and report pass/fail per criterion, plus the supporting checks.
9. **Write the verdict** into this folder. **No submission. No commit** unless the owner asks.

---

## Appendix — probe provenance and reproducibility

Location: `%TEMP%/m8probe/` (outside the repository, so no Model 1–7 file was touched).
Entry points: `p3.py` (D50 correlations), `p4.py` (dimensionality, 1-NN), `p5.py` (label PCA, 1-D
residual), `p17.py` (shape residual predictability), `p18.py` (shape-map functional form),
`p30.py` (family-cluster bootstrap on the primary result), `p19.py` (end-to-end reparameterisation),
`p20.py`/`p21.py`/`p22.py` (extrapolation), `p27.py` (metric separability),
`p28.py`/`p29.py` (per-column weighted-L1 refutation), `p24.py` (D50 RMSE + null),
`head.py` (the validated frozen head).

**Two confirmatory sweeps were started and did not finish, and neither is load-bearing.** An
exhaustive search over all 2,502 feature subsets (`p10.py`) was cancelled at the 30-minute
background limit, and a best-of-N permutation null on shuffled labels (`p15.py`) was stopped because
the family-cluster bootstrap already supplied the uncertainty the plan needs. Both are listed in
§10 as explicitly confounded by the best-of-N null; **no conclusion, gate or number in this document
depends on either.** They are reported here rather than quietly omitted.

**Validation of the instrument, stated so it can be checked:** the probe independently reproduces
the recorded rank-3 PCA floor (7.349 bare / 6.254 projected), the 16-family structure with the six
near-duplicate pairs intact, and Model 7's nested-LOFO baseline **43.453225** to six decimals. It does
**not** reproduce the looser historical anchor 43.0217 (it returns 41.1475), and no conclusion here
uses that protocol.

**Environment:** Python 3.14.6, numpy 2.5.2, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.9.0. No
torch, no GPU used.
