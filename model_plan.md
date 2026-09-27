# Model Development Plan — Model 1 / Experiment 1

**Status: PLAN ONLY. Nothing here is implemented. Awaiting approval.**
Companion to `report.md` (preprocessing, complete and verified).

---

## 0. Two facts that shape everything below

**(a) The label manifold is nearly one-dimensional.** PC1 holds 91.4% of label variance and
correlates 0.995 with log₁₀(D50). Best achievable EMD if you predict *k* numbers perfectly:
k=1 → 20.5, k=2 → 11.0, k=3 → 7.4. With 24 samples, the tractable target is a handful of
parameters, not 11 free values.

**(b) Camera is perfectly confounded with split.** Training is Motorola Edge / Samsung A52 /
Edge 60 Fusion; test is iPhone 14 / iPhone 16. **No test camera appears in training at all.** Any
feature that can identify the camera is a leakage channel, and any improvement measured on
within-training folds is an optimistic estimate of what generalises across cameras.

A third, quieter constraint: **at n=24, shuffled labels reach |ρ| ≈ 0.52.** Differences below that
are not evidence.

---

## 1. What Model 1 represents

**Model 1 = the image-signal baseline ladder.**

One fixed validation protocol, one fixed output representation, one fixed loss. The only thing
that changes between experiments is the **capacity and kind of the feature extractor feeding the
head**. This isolates the question every later model depends on: *how much predictive signal is
actually in these photographs, and how much of it survives the camera domain shift?*

It is deliberately not "the DINOv2 model" or "the CNN model". Those are backbone choices, and
comparing backbones is only meaningful once you know what a feature has to beat.

**Why this is Model 1 and not a preliminary:** it produces the reference numbers that every
subsequent model family is scored against, and it is the only tier that can be fully executed and
debugged on CPU before spending GPU quota.

## 2. What Experiment 1 tests

**Experiment 1: hand-designed, camera-independent texture + colour statistics → ridge regression →
low-dimensional curve parameters.**

Hypothesis: a model with **zero learned visual representation** already beats the 82.62 "ignore
the images" floor, and the well-sampled bands (e8, e16) carry most of it.

This is a defensible baseline rather than a toy because those features are already validated:
ρ(e8, log D50) = 0.764 and ρ(e16, log D50) = 0.763 at soil level, both beating their shuffled-label
control, and both **provably signal-neutral through preprocessing** (identical to three decimals
raw vs canonical).

Experiment 1 also establishes, in the same run, the two reference rows every later experiment is
measured against: the constant-median-curve floor and the best-possible-k-parameters ceilings.

## 3. Processed data it uses

All inputs come from the verified preprocessing outputs. **Nothing is recomputed from raw.**

| artifact | use |
|---|---|
| `data/processed_meta/manifest_images.csv` | 162 rows — paths, camera, scale, mask metadata, gray-world gains |
| `data/processed_meta/manifest_tiles.csv` | tile coordinates + `soil_fraction` for filtering |
| `data/tiles/**/*.png` | 1,976 materialised 256 px tiles (56.23 mm square) |
| `data/training_down/`, `data/testing_down/` | 162 canonical PNGs on the 4.552516 px/mm grid |
| `data/processed_meta/manifest_samples.csv` | 24 train labels + 10 test `submission_id` / `submission_row_order` |
| `data/processed_meta/cv_families.csv` | **new**, produced once by the implementation step, not by this plan |

Feature computation **imports `preprocess.verify.feats`** rather than reimplementing it. That is
deliberate: it guarantees the model's features are byte-for-byte the same definition that
`golden_checks.json` was computed against. A silently divergent reimplementation is the single
easiest way to make an experiment unrepeatable.

### Feature set for Experiment 1

Per tile, then aggregated to image, then to soil:

- **Texture bands** `e4, e8, e16` — well-sampled (λ ≥ 10 px), 94–96% soil variance, ~1.5% camera.
- **Saturation** and mean R/G/B — after gray-world gains are applied.
- **Gray-world gains** `gw_gain_r/g/b` — stored, not baked; applying them at load is the experiment.
- **Mask/geometry metadata** `soil_fraction`, `crop_area_fraction` — legitimate, camera-correlated
  but physically meaningful.

**Explicitly excluded as inputs:** `camera`, `native_ppm`, `effective_ppm`, `icc_*`,
`exif_*`, `site`, `sample_id`. These identify the capture device or the sample, and either leak
camera identity or break generalisation. Including `camera` is the trap this project must not fall
into — it would look like a large validation gain and be worthless on the private split.

**Excluded bands:** `e1` (λ = 2.5 px, at Nyquist, ρ = −0.24) and `e2` (fails its permutation
control). They are computed and logged for diagnosis but not fed to the head.

Aggregation: **median across tiles → median across images → soil vector**, with mean and IQR kept
as separate features. Median because 104 tiles are pure tray and would drag a mean.

## 4. Kaggle Input files to upload

One **Kaggle Dataset** (not a Kernel), versioned, containing:

```
soil-gsd-processed/
├── preprocess/                 # the package, imported for feats() and config
├── data/
│   ├── processed_meta/         # manifests, audit.json, golden_checks.json, cv_families.csv
│   ├── training_down/          # 162 canonical PNGs   (~342 MB)
│   ├── testing_down/           #                      (~103 MB)
│   └── tiles/                  # 1,976 tile PNGs      (~269 MB)
└── data/Training_labels_updated.csv, sample_submission.csv, ppm_updated.csv
```

**Deliberately excluded:** `data/qc/` (230 MB of contact sheets) and `_look/`. They are
human-review artifacts and would triple upload time for no modelling benefit.

Upload as a Dataset rather than attaching the competition input, because the competition provides
only raw images and our whole advantage is the audited canonical scale.

## 5. How the notebook accesses them

No hardcoded local paths. One config cell resolves everything:

```python
KAGGLE = Path("/kaggle/input")
INPUT_ROOT = next(p for p in KAGGLE.glob("soil-gsd-processed*") if (p/"data").exists()) \
             if KAGGLE.exists() else Path(".")     # local fallback for CPU smoke tests
sys.path.insert(0, str(INPUT_ROOT))                # makes `import preprocess` work
from preprocess import config, verify
```

Every downstream path is `INPUT_ROOT / "data" / ...`. On the local machine the same notebook runs
against the repo root, which is how a CPU-only experiment gets tested before it costs GPU time.

Kaggle settings: **GPU T4×2**, internet **on** (needed only if a later experiment pulls a
backbone; Exp 1 needs neither), persistence off, and the dataset attached at version pin.

## 6. Proposed model architecture (Experiment 1)

No neural network. A closed-form regressor, chosen so that capacity is not a confound:

```
tile PNG ──verify.feats()──► per-tile vector
             └─ median/mean/IQR over tiles ─► per-image vector
                              └─ median/mean over images ─► per-soil vector x
                                              ridge(x) ─► 4 curve parameters θ
```

- **Ridge regression**, α selected by leave-one-family-out inner CV, features standardised using
  **training-fold statistics only**.
- Two variants reported side by side: ridge on the raw feature vector, and **RANSAC-style robust
  fit**, because with 24 points a single soil can dominate a least-squares fit.
- A gradient-boosted tree variant is logged as Exp 1b — same features, so any difference is
  attributable to linearity alone.

## 7. Input representation

- Tiles at **256 px = 56.23 mm** on the canonical 4.552516 px/mm grid. Physical size is identical
  for every camera by construction — this is the payoff of the preprocessing phase.
- Tiles with `soil_fraction < 0.5` are **excluded from aggregation but retained in the manifest
  join**, so the filter is visible and reversible rather than baked.
- Whole canonical images are also summarised (not tiled) as a second input variant, to test whether
  tile decomposition helps at all.

## 8. Output representation — the main design bet

**Not 11 free values.** Predict **θ = 4 parameters** and reconstruct:

```
curve(θ) = clip( mean_curve + Σ_{k=1..3} θ_k · PC_k , 0, 100 )
then      F = cummax(F);  F[10] := 100
```

PC basis from the 24 training label curves (rank-3 captures 99% of variance; EMD ceiling 7.4).

Reasoning, and its honest cost:
- **Pro:** 4 outputs for 24 samples is a learnable problem; monotonicity is near-guaranteed by
  construction; the competition states labels are log-linear interpolations of smooth lab curves,
  so a smooth low-rank family is the right prior.
- **Con:** rank-3 unconstrained reconstructions **break monotonicity on 9 of 24** samples, so the
  projection step is load-bearing, and the 7.4 ceiling is only reachable if the projection is
  benign.
- **The PC basis is fit on all 24 curves.** For fold-level honesty it must be refit inside each
  training fold. Doing so is cheap; skipping it would leak held-out labels through the basis.
  This is a required implementation detail, not an option.

Experiment 1 also fits a **direct 11-output monotone parameterisation** (softmax over increments)
as Exp 1c, so the output-representation choice is *tested* rather than assumed, per §12 of the
brief.

## 9. Training strategy

Ridge closed form — no epochs, no optimiser, no scheduler. Exp 1's "training" is:
1. Fit feature standardiser + PC basis on the training fold only.
2. Fit ridge with α from inner leave-one-family-out CV.
3. Predict θ, project to a valid curve, score.

For the final submission model, refit on all 24 soils. Recorded hyperparameters are therefore
short: α grid, feature list, aggregation statistic, PC rank, projection rule, seed.

## 10. Validation strategy

**Leave-one-family-out (LOGO) over curve-distance families**, computed by average-linkage
clustering of the pairwise label-curve EMD, cut at 14.

That cut yields **16 groups: 6 multi-member families and 10 singletons** —
`{H366,H367,H371,H372}` (n=4), `{H038,H637}`, `{H181,H183}`, `{H368,H615}`, `{H549,H668}`,
`{H493,H666}`, and 10 soils with no near-twin. Each fold trains on 20–23 soils.

The 10 singletons are *correct*, not a defect: those soils genuinely have no twin, so there is
nothing to leak. The grouping exists specifically to break the dangerous pairs I measured —
H366≈H371 at distance 3.3, H367≈H372 at 4.3, H549≈H668 at 6.8, H368≈H615 at 7.0, H038≈H637 at 7.3,
H181≈H183 at 8.2, H493≈H666 at 8.5 — all of which plain leave-one-soil-out would leak.

**Honest limitation:** only 6 multi-member families means family-level generalisation is estimated
from few events, and the 16 fold scores are strongly correlated. Report the mean, the spread, and
per-family errors — never a single number in isolation.

**Primary metric:** mean EMD over folds, on the official 0–500 scale. Reference rows always
printed alongside: trivial baseline 100.31, no-image median 82.62, best-1-param 20.5,
best-2-param 11.0, best-3-param 7.4.

**A second, harder protocol is reported but not used for selection:** leave-one-*camera*-out
(hold out all Samsung images, or all Motorola). It directly simulates the unseen-camera situation
the test set presents, and it is the most informative single diagnostic available to us — but with
only two cameras it is too high-variance to drive choices.

## 11. Loss and metrics

- **Metric:** the official log-weighted EMD, `np.trapezoid(|F − F̂|, log10(d))`, mean over samples.
  Reimplementation is validated against the published trivial baseline (100.31) and the [0,500]
  range.
- **Exp 1 loss:** squared error on θ (ridge objective). The metric is not the loss here, and that
  is fine for a linear baseline.
- **Later experiments** train the metric directly — weighted-L1 between predicted and true
  cumulative curves is differentiable as written, so loss == metric from Exp 4 onward.
- **Diagnostics per fold:** per-support-point |error| profile (where along the log axis the model
  fails), D50 error, and error split by fine-family vs coarse-family.

## 12. Notebook organisation

~20 cells, five sections, each independently runnable and inspectable. Not a pipeline in one cell.

**A. Setup** — 1 markdown rationale · 2 config/dataclass · 3 imports + `INPUT_ROOT` resolution ·
4 reproducibility (seed, version and `config_hash` capture, assert it matches the manifest).

**B. Data** — 5 load manifests, print schema · 6 label load + sanity (monotonic, 200 mm = 100) ·
7 dataset inspection: tiles per soil/image histograms, `soil_fraction`, camera × split
contingency table, **the camera-confound table printed as a visible warning** · 8 visual QC: a
grid of real tiles per soil with their D50, plus deliberately the pure-tray tiles · 9 build
`cv_families` and display the grouping with within-family diameters.

**C. Features** — 10 compute `verify.feats` per tile, cached to parquet · 11 aggregation to
image then soil · 12 feature audit: correlation matrix, ρ with log D50 **with the permutation
control shown**, and camera-variance share per feature — so a feature that is mostly camera is
visible before it is used.

**D. Model + validation** — 13 PC basis and curve reconstruction, with the monotonicity projection
shown working and failing · 14 reference baselines computed in-run · 15 LOGO loop, per-fold loss
curve and predictions · 16 results table + error analysis plots · 17 variant comparison
(ridge vs robust vs GBM; tiled vs whole-image; 4-param vs 11-param).

**E. Output** — 18 refit on all 24, predict test · 19 **submission validation gate** · 20 write
`Submission_Model_1_E1.csv` and dump `Experiment_1.txt` programmatically.

Cell 20 writes the experiment note from the live config object rather than by hand, so the record
cannot drift from what ran.

## 13. Submission generation

Written directly as `Submission_Model_1_E1.csv` — never renamed from a generic file. The name is a
notebook-level constant derived from `MODEL_ID` and `EXPERIMENT_ID`.

**Hard validation gate before writing** — the cell raises rather than emitting a bad file:
- exactly 10 rows, IDs **equal to and in the order of** `sample_submission.csv` (via
  `submission_row_order`)
- exactly the 11 required columns, exact names, no extras
- all values in [0, 100], no NaN, non-decreasing across columns, `200 == 100.0`
- re-read the written file from disk and re-assert all of the above
- print the predicted curves so a human sees them before uploading

## 14. What `Experiment_1.txt` records

Auto-generated. Model id/name, experiment id/name, **hypothesis and why run**, one-line
**"changed vs previous experiment"**, data version (`config_hash`, `pipeline_version`, dataset
version), input representation (tile size, mm, filter threshold, tiles per soil min/median/max),
feature list **and the explicit exclusions with reasons**, aggregation statistics, output
representation and PC rank, ridge α grid and selection, loss, validation protocol and family
table, seeds, **per-fold EMD and mean ± spread**, the reference rows, Kaggle public/private score
left blank and filled in by hand afterwards, failure analysis, and the next-hypothesis note.

Parameters that don't exist for a closed-form ridge (epochs, batch size, scheduler, patience) are
recorded as `n/a — not applicable to this estimator` rather than invented.

## 15. Files created during implementation

```
models/
├── README.md                          # the model/experiment taxonomy and rules
├── shared/
│   ├── data_io.py                     # manifest joins, tile loading, family assignment
│   ├── features.py                    # thin wrapper importing preprocess.verify.feats
│   ├── curve.py                       # PC basis, reconstruction, monotone projection
│   ├── metrics.py                     # EMD + reference baselines
│   └── validation.py                  # LOGO splitter, camera-out protocol
└── Model_1/
    ├── instructions.txt
    └── Model_1_Experiment_1/
        ├── Model_1_Experiment_1.ipynb
        ├── Experiment_1.txt           # generated by the notebook
        ├── Submission_Model_1_E1.csv  # generated by the notebook
        └── kaggle_setup.md            # upload + attach instructions
data/processed_meta/cv_families.csv    # written once, versioned with the preprocessing
```

No second preprocessing pipeline anywhere under `models/`.

## 16. Assumptions

1. The EMD reimplementation matches Kaggle's. Inferred from the page's three constraints (area
   between curves on log scale, ten intervals, range [0,500]) and consistent with the published
   trivial baseline — **but the formula itself is an unreadable image.** Exp 1's first submission
   tests it: if the reported score is not ≈ the local estimate, the metric is wrong, not the model.
2. Ridge on ~10 features with 24 samples is not capacity-limited relative to the signal. If Exp 1
   lands near the 7.4 rank-3 ceiling, that assumption is false and the ladder needs a different
   second rung.
3. Uploading ~715 MB as a Kaggle Dataset is acceptable to you.
4. `verify.feats` is the right feature definition to reuse. It is validated but was designed for
   auditing, not modelling.
5. Test soils fall inside the training size range. **Unverified** — see risks.

## 17. Risks and limitations

- **n=24 is brutal.** The permutation control reaches |ρ| ≈ 0.52. Most plausible experiment
  deltas will be inside noise, and the honest response is to report spreads and per-family errors,
  not to chase decimal differences.
- **Camera confound is unfixable by validation design.** Every internal estimate is optimistic
  relative to the private split. Leave-one-camera-out is the only proxy, and it is high-variance.
- **The PC basis can leak labels** if fit outside the fold. Guarded by an explicit assertion.
- **Gray-world gains are camera-structured** (Motorola R 0.942/B 1.063 vs iPhone ≈0.98/1.03), so
  feeding them is a soft camera identifier. Mitigation: Exp 1 reports results with and without.
- **Test soils have fewer tiles** (7 of 10 at exactly 36) than training (median ~65), so
  aggregation statistics are estimated from less data on the test side — a quiet bias in tile-count
  sensitive features.
- **The notebook cannot be executed locally** for anything needing torch. Exp 1 is designed to be
  CPU-only precisely so it *can* be run and checked before Kaggle; later experiments will be
  untested until they run remotely. That risk is accepted and should be stated in each note.
- **Whether test soils are in range is unresolved.** Münster and Testfeld Lidl WHV look like
  cobbly gravels coarser than anything training contains, where training tops out at D50 6.1 mm.
  My first automated attempt at measuring this was broken and I discarded it. If the test set is
  genuinely outside the training range, extrapolation risk dominates model choice.
- **Public LB is 3 fixed soils** and final = 0.3·public + 0.7·private. Public scores are nearly
  information-free; do not let them steer anything.

## 18. What turns Model 1 into Model 2

Model 1 changes **only the feature extractor**, holding validation, output representation, loss and
aggregation fixed. A new Model number requires a change of *kind*, not degree:

- **Model 2 — learned representation.** A frozen or partially fine-tuned pretrained backbone
  (DINOv2 / ConvNeXt) replacing hand-designed bands. Same everything else. This is the expected
  next family.
- **Model 3 — changed output parameterisation.** e.g. a physical grading-curve family
  (van Genuchten / Kornhaus) with monotonicity structurally guaranteed instead of projected, or
  direct 11-output with a differentiable EMD loss.
- **Model 4 — changed aggregation or task framing.** Attention-based multiple-instance pooling,
  cross-image reasoning, or self-supervised pretraining on all 162 images including test.
- **Model 5 — explicit domain adaptation.** Camera-invariant learning, or a per-camera
  contrast-response correction estimated with the test images' pixels.

Promotion test: *if I can describe the change as "same model, different features", it is a new
Experiment. If it changes what the model fundamentally is, it is a new Model.*

---

## Proposed execution order once approved

1. `cv_families.csv` + `models/shared/` utilities, unit-checked on CPU.
2. `Model_1/instructions.txt` and `kaggle_setup.md`.
3. Notebook sections A–C, run locally, no GPU.
4. Sections D–E locally; confirm Exp 1 beats 82.62 and the reference rows reproduce.
5. Upload the Kaggle Dataset; run the notebook there; write the submission.
6. `Experiment_1.txt` auto-dumped; record the Kaggle score by hand.
7. Only then decide Exp 2 (pooling) vs Exp 1c (output representation) from what Exp 1 showed.

**Awaiting approval before writing any of it.**
