# Experiment 2 plan — Model 1

**Status: DRAFT FOR REVIEW. Nothing below is implemented.**

Experiment 2 is not a model experiment. It is an instrument experiment. Its output is a
validated rule for choosing configurations, which every later experiment in this project
depends on.

---

## 0. CONFLICT WITH THE PROJECT SPEC — DECISION REQUIRED

`Model 1/instructions.txt` states:

> Every experiment inside Model 1 keeps the validation protocol, the output/curve
> representation, the loss/metric and the sample-level aggregation FIXED. The only thing
> that varies between experiments is the capacity or kind of the FEATURE SET.

Experiment 2 as designed here **changes the validation protocol**, which that sentence
forbids. I am reporting the conflict rather than quietly restructuring.

The evidence that forces it: Experiment 1's protocol selected `alpha = 0.03`. Under that
selection the model scored **172.70** on Kaggle, against **102.37** for a submission that
ignores the images entirely. The internal number that justified the choice was **35.98**.
The instrument does not merely overstate — it ranks configurations in the *opposite*
order to reality (§3.4). Leaving it fixed would mean every subsequent Model 1 conclusion
is measured by a ruler that points the wrong way.

Three ways to resolve it. **I recommend (a).**

| | option | consequence |
|---|---|---|
| **(a)** | Amend `instructions.txt`: the FIXED block becomes "fixed *as of E2*; E2 is the calibration experiment that established it." E2 is labelled a measurement-instrument experiment, not a feature experiment. | Cheapest. Keeps the ladder honest. Requires editing a file that declares itself a rule-set. |
| (b) | Fold the protocol change into E2 as an undeclared second variable. | Violates single-variable discipline. Rejected. |
| (c) | Call it `Model 6 — validation methodology`. | Wrong by the project's own promotion test: nothing about what the model *is* changes. |

Note the existing instructions already list *"whether leave-one-camera-out predicts the
real generalisation gap"* under VARIABLES WE INTEND TO INVESTIGATE, so E2 is inside the
declared research scope. Only the FIXED sentence blocks it.

---

## 1. What Model 1 represents

Unchanged. Classical hand-built tile measurements plus a simple interpretable regressor;
a baseline ladder, not a backbone. See `Model 1/instructions.txt`.

## 2. What Experiment 2 tests

**One question:** *which internal selection criterion predicts the external score?*

E1 answers "how good is this feature set" only if the ruler is right. It is not. E2
compares candidate rulers against the two external ground-truth points we own, picks one,
and re-scores E1's own configuration under it. E2 changes **no model component**: same
rank-3 PCA curve basis, same ridge, same monotone projection, same tile-to-soil mean
aggregation, same LOGO families.

## 3. Evidence gathered for this plan (read-only, local, 2026-09-27)

All of the following was computed from existing artifacts plus `scratch/cal_exp2.py`. It
is *not* Experiment 2 — it is the reconnaissance that justifies its design. Experiment 2
must re-derive every number inside the notebook so the result is reproducible.

### 3.1 The metric is confirmed; the competition page's formula is not

| reading of the metric | trivial baseline scores |
|---|---|
| page formula, left-endpoint sum | 102.02 |
| **ours, trapezoid on log10 d** | **100.3132** |
| **value the host publishes** | **100.31** |

Our implementation reproduces the host's own reference number to four significant figures;
the formula printed on the page does not reproduce it. The page notation is a loose
description of the integral. **No EMD number already computed needs redoing.**

### 3.2 The held-out-camera proxy has real signal

Fitting on one training camera's view of a soil and predicting the other camera's view of
that same soil, with the soil's *label* held out of the fit (21 soils are photographed by
both Motorola Edge and Samsung A52):

| feature set | alpha | in-domain M→M | transfer M→S | transfer S→M |
|---|---|---|---|---|
| 16 (as E1) | 0.03 | 64.5 | **96.9** | **136.6** |
| 16 (as E1) | 10 | 43.4 | 56.3 | 125.7 |
| 16 (as E1) | 1000 | 87.6 | 88.2 | 80.8 |
| 14 (defect-fixed) | 10 | 43.3 | **51.4** | **48.5** |
| 14 (defect-fixed) | 1000 | 88.5 | 88.2 | 88.4 |
| 7 texture only | 10 | 43.9 | 69.5 | 45.3 |
| 7 colour only | 0.03 | 44.8 | **127.2** | 78.4 |

Dropping the two degenerate geometry columns (§3.3) cuts the transfer penalty from
~17 EMD to ~6 EMD at alpha = 10, and cuts test-set saturation from 59% to 13%.

### 3.3 `soil_fraction` is a defect, not a feature

Across all 24 training soils it is constant to four decimals (mean 0.9999, std 0.00027,
range 0.0012). On the test soils it reaches 0.829 — **−63.9 standard deviations**. A
column with no training variance cannot be learned, yet it moves enormously out of
domain, so its fitted coefficient is a pure noise amplifier. `crop_area_fraction` fails
the same way, milder: 8 of 10 test soils outside the training range.

### 3.4 The current criterion ranks configurations backwards

| alpha | LOGO CV (what E1 selected on) | held-out-camera transfer |
|---|---|---|
| 0.03 | **36.03** ← CV's best | 96.9 / 136.6 ← transfer's worst |
| 10 | 47.15 | 56.3 / 125.7 |
| 100 | 65.07 | 65.1 / 92.4 |
| 1000 | 87.32 (worse than no-image) | 88.2 / 80.8 |

CV improves monotonically as alpha falls. Reality does the opposite. This is the single
most important finding of the experiment cycle so far.

### 3.5 The proxy underestimates the real gap — by a factor we can now measure

For E1's exact configuration the proxy predicted `(96.9 + 136.6)/2 = 116.8`. The actual
Kaggle score was **172.70**. Ratio **1.48×**. The no-image baseline moved 86.7 → 102.37,
a ratio of 1.18×. So the proxy is directionally correct and quantitatively pessimistic,
and the model degrades faster than the baseline as the camera shift grows.

### 3.6 Why it underestimates — the hypothesis E2 must test

Motorola and Samsung differ by 0.81 mean `|z|` on the same soil. Training→test differs by
**2.63 sd on B alone**, with `sat` −1.50 sd, `crop_area_fraction` +1.25 sd, `G` +1.09 sd.
Colour carries the shift; texture barely moves (e4 −0.66, e8 −0.58, e16 −0.38 sd).

But texture moves *consistently downward in every band*, and that is not noise. Training
images were delivered at ~4.6 px/mm and resampled to 4.5525 — essentially no change, so
they **alias** high-frequency content into the fine bands. Test images come from 14–20
px/mm and take a genuine 3–4× anti-aliased Lanczos downsample, so they carry no such
folded energy. The two sets are measured at the same nominal scale but with different
information content, and the e4 band (0.44 mm sigma ≈ 1.1 mm wavelength ≈ 5 px) sits
right at the edge of that cliff.

If true, this is a systematic one-directional bias that no amount of camera-holdout
inside the training set can reveal, because every training image shares it.

### 3.7 Secondary

Tiles per soil: train median 64, test median 36. Aggregation means are unbiased by this
but their variance is not, so per-soil feature noise is ~1.3× higher on test.

## 4. Design

A **factorial calibration experiment**. No model component changes.

**Factor A — selection criterion** (the research question):
1. `LOGO-CV` — E1's rule. Leave-one-family-out over the 16 curve-distance families.
2. `CAM-HOLDOUT` — mean of both Motorola↔Samsung directions, 21 dual-camera soils,
   label held out of the fit by family.
3. `RES-MATCH` — *new synthetic camera.* Re-extract tile features from training images
   after an extra anti-aliased downsample-and-resample that reproduces the test
   condition (§3.6), then treat that as a held-out camera. This is the level that tests
   the aliasing hypothesis, and the only candidate proxy whose magnitude can match the
   real shift.

**Factor B — feature set** (defect correction, reported separately from any research claim):
1. `16` — exactly E1's matrix.
2. `14` — E1's matrix minus `soil_fraction` and `crop_area_fraction`.

**Swept, not fixed:** ridge alpha ∈ {0.03, 1, 10, 30, 100, 300, 1000}.

**Held identical to E1:** rank-3 PCA curve basis, `StandardScaler`, monotone projection
(clip → cummax → force 100), tile→soil arithmetic mean, the 16 curve-distance CV
families, the 24/10 split, the EMD implementation.

**Primary result:** a table of every (criterion, feature set, alpha) triple scored under
all three criteria, plus the two external ground-truth points used to check which
criterion ranks correctly. The deliverable is a **selection rule**, written into
`instructions.txt` for E3 onward.

**Secondary result:** `Submission_Model1_E2.csv` from the configuration the winning
criterion selects. This is calibration point #3 for the proxy's pessimism factor — if the
chosen criterion predicts ~55 and Kaggle returns ~80, the factor is stable and we can
trust it. If it returns ~170 again, the proxy is useless and we escalate.

## 5. Processed data used — unchanged from E1

`data/processed_meta/manifest_{images,tiles,samples}.csv`, `data/tiles/` (256 px,
16228 rows, 1976 materialised), `data/Training/*/labels.csv`, `data/sample_submission.csv`,
`data/target_ppm.json`. `config_hash 010f44c36c74`. The notebook re-runs **no**
preprocessing and reads **no** raw `data/Training` images except labels.

Factor A level 3 (`RES-MATCH`) is the one addition: it reads the canonical processed
images and applies a derived resample **at feature-extraction time only**, writing to the
experiment's own `.cache/`. It does not touch `data/processed_meta/` or `data/tiles/`.

## 6. Kaggle inputs — unchanged from E1

Same dataset, same paths, same `kaggle_setup.md`. E2 adds no upload requirement.
E2 runs on **CPU** and needs no GPU; expected runtime ~4 minutes (three feature
extractions instead of one).

## 7. Output representation — unchanged

rank-3 PCA coefficients → 11-point curve → monotone projection. E2 does not touch this.

## 8. Training strategy

Closed-form ridge, no epochs, no optimiser, no augmentation. Alpha is the only knob and it
is swept, never hand-picked.

## 9. Validation strategy

- `LOGO-CV`: 16 families from average-linkage on pairwise label-curve EMD, cut at 14.
- `CAM-HOLDOUT`: fit on camera *c1* rows of training soils outside the held-out family;
  predict camera *c2* rows of the held-out soils. Both directions, averaged.
- `RES-MATCH`: as `CAM-HOLDOUT` but the target is the resolution-matched synthetic view.
- **Leakage rules:** scaler, PCA basis and curve basis refitted on each fold's training
  rows only. For `CAM-HOLDOUT`, the held-out soil's label never appears in the fit even
  though the same soil exists under the other camera — this is the subtlety that makes the
  proxy valid, and the notebook must assert it.
- No random image-level split anywhere.

## 10. Loss and metrics

EMD (§3.1) as the reported score. Secondary diagnostics per configuration: fraction of
predicted columns pinned to 0 or 100, mean EMD of test predictions to the mean training
curve, and the per-feature out-of-range excess normalised by training range.

## 11. Notebook organisation

`Model 1/Model 1 Experiment 2/Model1_Experiment2.ipynb`, sections:

- **A — setup**: config, paths, imports, seed, cache dir.
- **B — data & metric**: manifests, labels, EMD with the §3.1 three-way reconciliation
  printed as an assertion; reference baselines (trivial, mean curve, median curve).
- **C — the E1 result, restated**: reload E1's matrix and print 35.98 / 172.70 / 102.37
  side by side. The motivating table.
- **D — feature-space audit**: per-feature training range vs test position; the
  `soil_fraction` degeneracy check with its own assertion; the band-wise downward texture
  shift plot.
- **E — build the three evaluation criteria** as separate, inspectable cells.
  `CAM-HOLDOUT` prints the 21-soil dual-camera roster and the leak assertion.
- **F — `RES-MATCH` synthetic camera**: the resample, a side-by-side image pair of a real
  vs resolution-matched tile, the re-extracted feature table, and the band-energy ratio
  plot that tests §3.6 directly.
- **G — the factorial sweep**: all (criterion × feature set × alpha) triples. One tidy
  results CSV, printed as a pivot.
- **H — criterion ranking**: which criterion, if any, ranks alpha in the order the two
  external points say is correct. This is the experiment's answer.
- **I — fit the selected configuration, predict the test soils.**
- **J — submission**: build, run the 8 validation gates, write
  `Submission_Model1_E2.csv`, print the row count and the Münster row explicitly.
- **K — error analysis**: predicted vs nearest-training-soil curves, per-row.
- **L — write `Experiment2.txt`** from the accumulated record, same 18-section extractor
  as E1.

No cell runs the whole pipeline. Each cell prints what it did.

## 12. Submission generation

Notebook writes `Submission_Model1_E2.csv` itself — never a renamed `submission.csv`.
IDs from `manifest_samples.csv` (`submission_id`, `submission_row_order`), never built
from folder names. Eight gates before writing: row count 10, exact ID set, exact ID
order, exact column names from `sample_submission.csv`, no NaN, values in [0, 100],
monotone non-decreasing, last column exactly 100.

## 13. What `Experiment2.txt` records

The standard 18 sections, plus three E2-specific ones: the full factorial table; the
criterion-ranking verdict; and an explicit statement that **E2 introduced no model
change**, so any score movement is attributable to the selection rule and the defect fix
alone.

## 14. Files created

```
Model 1/Model 1 Experiment 2/
  Model1_Experiment2.ipynb
  Experiment2.txt            (generated by the notebook)
  instructions.txt           (E2's own scope statement)
  Submission_Model1_E2.csv   (generated on run)
  criterion_comparison.csv   the factorial result — the primary artifact
  feature_range_audit.csv
  cv_families.csv            byte-identical copy of E1's, asserted equal
  features_soil_16.csv / features_soil_14.csv
  tile_features_*.csv        in .cache/, including the RES-MATCH variant
  kaggle_setup.md
```
Plus one edit to `Model 1/instructions.txt` (the FIXED-block amendment, §0 option a).
E1's folder is **not** modified beyond §16, already written.

## 15. Assumptions

- The two external points (172.70, 102.37) are on the same 3 public soils. If the public
  subset changed between submissions, §3.5's 1.48× factor is void.
- Rank-3 remains adequate. E1 measured the rank-3 ceiling at 7.35 EMD, far below every
  number in play, so this is not the binding constraint.
- Motorola Edge and Samsung A52 are the only usable internal camera pair; `Motorola Edge
  60 Fusion` has 3 images and 1 soil (H374, the re-photographed suspect sample) and is
  excluded from `CAM-HOLDOUT`.
- The tile→camera parse from `tile_path` is reliable. Verified tonight: exactly
  `Motorola` 716 / `Samsung` 825 / `iPhone14` 183 / `iPhone16` 240 / `iPhone` 12 tiles.
  The 12 bare-`iPhone` tiles are a filename irregularity the notebook must handle or flag.

## 16. Risks and limits

1. **n = 21 soils, 2 camera pairs.** The `CAM-HOLDOUT` estimate has real sampling error;
   a 6-EMD difference between two configurations is not evidence. Report fold spread, not
   just means, and refuse to rank configurations whose intervals overlap.
2. **Selecting alpha on `CAM-HOLDOUT` and then reporting `CAM-HOLDOUT` as the result is
   circular.** The notebook must nest: choose alpha inside each fold, report the
   held-out-soil score of the *procedure*, not of the best alpha.
3. **`RES-MATCH` may be the wrong model of the gap.** If the real difference is iPhone
   colour pipeline rather than sampling, `RES-MATCH` will look clean while the score stays
   bad. That is a useful negative result and is written up as such.
4. **Public LB is 3 of 10 soils.** One submission is a weak measurement. E2 gets two
   (E1 and E2) and we will not read more into them than they carry.
5. **We cannot rule out that no internal criterion is adequate.** If E2 shows all three
   criteria rank alpha differently from the external evidence, the honest conclusion is
   that Model 1 must be selected on camera-robustness priors (heavy regularisation, no
   colour features) rather than on any measured score. That is an acceptable outcome.

## 17. What this decides for later experiments

- **E3** — feature-family ablation (texture / colour / frequency / gradient), scored by
  whichever criterion E2 validates. This is the experiment `instructions.txt` always
  meant E2 to be; it moves back one slot.
- **E4** — aggregation statistic (mean vs median vs trimmed; tile-level vs whole-image).
- **E5** — regressor kind (Huber, gradient boosting) on the identical matrix.
- **Model 2** — a learned representation becomes justified only once the selected
  criterion says the classical features have plateaued. If camera transfer is the binding
  constraint, a frozen DINOv2 backbone will fail the same way, and E2's verdict is what
  tells us whether to build it.

## 18. Falsifiable predictions — written before the run

| # | prediction | if wrong |
|---|---|---|
| P1 | `LOGO-CV` and `CAM-HOLDOUT` disagree on the best alpha by ≥ 2 orders of magnitude. | The §3.4 reading was an artefact of my ad-hoc script; re-derive before trusting anything. |
| P2 | Dropping `soil_fraction` + `crop_area_fraction` improves `CAM-HOLDOUT` transfer. | The defect is not load-bearing; the colour hypothesis is weakened. |
| P3 | `RES-MATCH` shows a downward shift in *every* texture band, matching the train→test pattern. | The aliasing explanation (§3.6) is wrong; the residual gap is colour, not sampling. |
| P4 | The E2 submission scores **below 102.37** (i.e. beats the no-image baseline). | Camera invariance is harder than the proxy says; escalate to explicit domain adaptation and consider promoting it to Model 5 early. |
| P5 | E2's actual score ≈ 1.5 × its `CAM-HOLDOUT` prediction, matching E1's factor. | The proxy's bias is unstable, so it cannot be used for selection at all. |

P4 and P5 are the ones that decide whether we have a working research method or just a
model that happens to run.
