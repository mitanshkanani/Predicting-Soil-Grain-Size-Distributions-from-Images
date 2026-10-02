# Model 10 / Experiment 1 — Plan

**Status: PLAN ONLY. Nothing implemented, nothing trained, nothing predicted, no submission, no upload.**
**Date:** 2026-10-02
**Owner instruction this plan answers:** build a new proper Model 10 from every mistake and
result recorded in Models 1–8; owner chose **per-image data expansion** as the objective,
"plenty" of submission slots remain, upload a dataset/zip only if genuinely needed (it is not —
see §7), and runs happen locally.

---

## 1. Where this experiment starts

Model 8 closed four experiments and left these facts standing:

| fact | value | provenance |
|---|---|---|
| incumbent external score | **55.80591** (M8 E2-A: 12 handcrafted features, colour raw, alpha=3) | `[repo-measured]` submission artifact |
| incumbent internal, nested LOFO | **43.453225201811563** | `[repo-measured]` reproduced bit-exact in M8 E1/E2/E3 |
| incumbent internal, oracle alpha=3 | **41.1475158656982032** | `[repo-measured]` reproduced bit-exact in M8 E2/E3 |
| head is DATA-limited | `EMD = 35.33 + 139.16/n + 15.85/sqrt(n)`, R2=0.999; 1/n term = 5.80 EMD at n=24 | `[repo-measured]` M8 E4 |
| E4's ranked recommendation (a) | "Per-image features instead of per-soil medians ... the largest untapped source in the project, needs no new sensor data" | `[repo-measured]` M8 E4 report §9 |
| internal ruler anti-ranks FEATURE SETS (Spearman -1.00, n=3) but ranks ALPHA correctly (+1.00) | | `[repo-measured]` M8 E2 §12 |
| public board resolves only differences >= ~5 EMD (3-soil null SD 13.13) | | `[repo-measured]` M8 E3 plan §7 |
| the metric is deterministic (identical file -> identical score) | | `[repo-measured]` M8 E3 plan §7 |

**The gap this experiment attacks.** The pipeline collapses all information to ONE row per soil:
tiles -> (median) -> image -> (median) -> soil. 24 training soils = 24 rows. But the cached
per-tile table holds 127 training images' worth of features. E4 identified this as the largest
untapped source; no experiment has ever tested it.

**Scope boundary (Model 9 is another agent's, status unknown).** This experiment creates only
`Model 10/Experiment 1/`. Model 1–8 files are READ-ONLY inputs. No shared file is modified, so
a Model 9 working anywhere else cannot collide. Nothing in this plan depends on Model 9.

---

## 2. Foundations verified this session (before any design was frozen)

All measured by read-only probes; no repository file was written.

1. **Cached per-tile features exist locally and are complete:**
   `Model 1/Model 1 Experiment 3/.cache/tile_features_256_010f44c36c74.csv` — 1976 tiles x 16
   columns (all 12 frozen features + e1, e2, spec_centroid_cpm, dom_wavelength_mm). Join to
   `manifest_tiles.csv` on `tile_path`: **1976/1976 matched, 0 unmatched.**
2. **The exact frozen-pipeline recipe was recovered:** filter `soil_fraction >= 0.50`
   (1946/1976 tiles), then tile -> image median -> soil median reproduces
   `features_soil.csv` on **all 34 rows with max abs diff 0.0** (train-only without the filter
   was 2.84e-14; test rows require the filter). This is G0b's expectation, measured before the
   plan was written.
3. **Expansion size:** 127 training images across 24 soils (3–8 images/soil), 35 test images
   across 10 soils (3–5). After the soil-fraction filter **zero images lose all their tiles**;
   every surviving train image has >= 4 qualifying tiles; all 1541 qualifying train tiles have
   cached features.
4. **No image pixels are needed.** Image-level features are medians of cached tile rows.

---

## 3. Ledger (standing rule: ledger before experiment)

### SETTLED — safe to assume
- Incumbent scores and both anchors (table in §1). `[repo-measured]`
- Reproduction recipe (§2.2). `[repo-measured]`
- Family-honest nested LOFO over 16 curve-distance families is the ruler; whole FAMILIES hold
  out, never images or soils. `[repo-measured, M8 E1 reproduced 43.453225201811563 bit-exact]`
- Public differences < ~5 EMD are unresolvable; private (10 soils, 70% weight) decides.
  `[repo-measured]`
- Submission contract: 10 rows, `HPC_` ids in `sample_submission.csv` order joined through
  `manifest_samples.submission_id` (NEVER string surgery — the Muenster mojibake trap, M8 E2 §7).
  `[repo-measured]`

### CONTESTED — the reason this is an experiment
- **Does between-image variation carry signal?** E4 recommends per-image expansion; but the two
  nearest measurements are negative — richer soil-level aggregation (median+mean+sd) 41.15 ->
  42.13/42.22, and tile-distribution summaries 47–49. `[repo-measured]` Those tested *summarising
  spread into extra columns*; image-granularity *fitting* is a different estimator and has never
  been run. Both readings are on the record; the experiment decides.
- **Can internal CV rank row-granularity arms?** It ranks within-head knobs (alpha) correctly and
  feature-set changes backwards. Granularity is neither category. Resolution: external decides,
  internal is report-only (§5).

### OPEN — nobody has measured it
- Image-level fit with replicated labels (127 rows, 24 distinct curves) under family-honest CV.
- Test-time predict-per-image-then-aggregate vs aggregate-then-predict (where the monotone
  projection sits). Aggregation DEPTH was measured (+1.14 EMD, M5) but not prediction-space
  aggregation.
- 1/n_images loss weighting vs unweighted image fitting.

### REFUTED — do not re-propose
Richer soil-level aggregation (mean/sd/percentiles); tile-level spread; DINOv2 at any crop or
combination; camera adaptation / CORAL / transductive alignment; resolution-matched training;
per-column weighted-L1 head (-28.84 EMD); rank-1 basis (one soil); within-tile A1–A4 (placebos
outran it); ensembling/bagging/blending; particle counting. Full citations: Model 8 E1 plan §10
and `project-cancelled-hypotheses`.

---

## 4. Hypothesis and arms

> **H10:** Fitting the frozen head at image granularity (127 rows, each image labelled with its
> soil's curve) changes external transfer versus the soil-level incumbent, and the change
> decomposes into a fit-side component and a test-side component.

**Arms — exactly three, frozen now, before any number exists. No grid, no fourth arm, no
post-hoc selection. Both new arms are submitted and both are reported whatever they score.**

| arm | fit side | test side | status |
|---|---|---|---|
| **A (control)** | soil-level (24 rows, two-stage median) — the frozen pipeline | soil-level features -> one prediction -> project | **known scores, NOT resubmitted** (metric deterministic) |
| **B (primary)** | **image-level**: scaler + ridge fit on 127 image rows, each image weighted `1/n_images(soil)` | predict each of the 35 test image curves -> **mean per soil** -> project ONCE | submitted |
| **C (diagnostic)** | soil-level, exactly as A | predict each of the 35 test image curves -> apply the monotone projection to **EACH image curve** -> **mean of the projected curves** per soil | submitted |

**Frozen common machinery (identical in all three arms):** the 12 features; rank-3 PCA label
basis fitted on training-fold labels only (the label set is the same 24 curves in every arm, so
the basis is arm-invariant); the frozen 16-value alpha grid with strictly nested selection
inside training families; StandardScaler fitted inside the fold; the monotone projection
(clip [0,100] -> cumulative max -> force exactly 100 at 200 mm). **B and C differ in WHERE the
projection sits: B projects once (mean first), C projects per image (mean after).**

**Why the 1/n weighting in B:** it gives every soil total loss mass 1 regardless of image count
(3–8), keeping the loss scale at ~24 units so the frozen alpha grid means the same thing as in
A. Unweighted fitting would let 8-image soils dominate 3-image soils 2.7x AND silently weaken
effective regularisation. The unweighted variant is recorded in §3 OPEN and is **not** run —
one primary arm only.

**CV scoring is at SOIL level in all three arms:** every held-out image is scored into one soil
curve before EMD — A via soil-level features, B by averaging raw image curves then projecting
once, C by projecting each image curve then averaging — so all arms are scored on the same 24
curves against the same labels. The arms differ only where stated in the table, never in WHAT is
scored. C's average of projected curves stays contract-valid (a mean of monotone curves is
monotone, and every component equals 100 at 200 mm).

**Decomposition logic:** pre-projection the head is affine, so C vs A isolates the test side
(per-image prediction + projection placement) and B vs A is the total effect. **B vs C is
therefore not a pure fit-side isolation; it is fit side plus projection placement.**

**Placebo (report-only, not a gate):** refit B with soil labels permuted across families,
**seed 90001** (the project's existing placebo-seed convention, frozen here). Expectation pre-declared: it must land far worse than any real arm (E1's equivalent
landed 80.86). If the placebo lands near the real arms, the pipeline is broken — that is a
stop-and-report condition, treated like a failed G0.

---

## 5. Gates

| gate | requirement | decision rule |
|---|---|---|
| **G0a** | `ruler.py` reproduces BOTH anchors bit-exact: `43.453225201811563` (nested) and `41.1475158656982032` (oracle alpha=3) | failure stops the run; nothing else reported |
| **G0b** | image-feature collapse reproduces `features_soil.csv` on all 34 rows, max abs diff <= 1e-10 (measured 0.0 this session) | failure stops the run |
| **G1** | fold integrity: 16 families, 24 train soils scored, 0 family leaks; scaler/basis/ridge/alpha all fitted inside the fold; held-out family images never touch any fitted quantity | failure stops the run |
| **G2 (PRIMARY, external)** | an arm beats **55.80591** | tiers pre-declared in §6 — decided on Kaggle, not internally |
| **G3** | submission contract, asserted in code BEFORE any file is written: 10 rows; ids in `sample_submission.csv` order via `manifest_samples.submission_id`; monotone non-decreasing; exactly 100 at 200 mm; no NaN; values in [0,100] | failure stops that arm's file from being written |
| **D1 (determinism)** | two cold-interpreter runs emit byte-identical artifacts (sha256 comparison) | mismatch = bug; fix and re-run both, report the mismatch openly |

**Reporting rules (binding):**
- Internal CV numbers are **printed, never used to select or justify** anything (M8 E2's
  anti-rank lesson: the arm internal liked most scored worst externally).
- Internal CV EMD and external Kaggle EMD are different scales and are **never subtracted**.
- A public improvement < 5 EMD is reported as "better but unresolved", never as progress.

---

## 6. Outcomes — decided now, not after seeing numbers

- **B beats 55.80591 by >= 5 EMD** → H10 supported at resolved magnitude; image granularity
  becomes the new incumbent; Model 10 continues from B.
- **B beats 55.80591 by 1–5 EMD** → direction supported, magnitude unresolved (public band);
  keep B as candidate incumbent pending the private board; no further arms.
- **B lands within 5 EMD of 55.80591 (better or worse)** → H10 not supported externally;
  **no evidence of benefit at public-board resolution; recommendation (a) is NOT refuted, only
  unresolved.**
- **B is worse by > 5 EMD** → image-granularity fitting is harmful (the errors-in-variables
  risk in §8 materialised); the question is closed with a directional answer.
- **Role of C:** if C ≈ A, the test side contributes nothing and any effect is on the fit side.
  C's own score vs 55.80591 is reported under the same tiers.
- **Both arms land worse** → report plainly: the largest untapped source measured by E4 does
  not convert. That is a result, not a failure.

**No outcome is spun.** `SUBMIT = TRUE` for B and C regardless — the point of an external-first
design (M8 E2's only protocol that ever moved the score) is that the board decides; both files
are produced and both are reported.

## 6b. Final submission rule

**Keep the incumbent (55.80591) as the final selected submission unless B beats it by at least 5
public EMD.** Private score is not visible during the competition and is never used to decide.
This rule is frozen here, before either arm's public score exists.

---

## 7. Data, compute, upload

**Inputs (read-only, all local):**
1. `Model 1/Model 1 Experiment 3/.cache/tile_features_256_010f44c36c74.csv` — tile features
2. `data/processed_meta/manifest_tiles.csv` — tile->image->soil keys, `soil_fraction`, split
   (filter `tile_size_px==256` and `soil_fraction>=0.50`)
3. `data/processed_meta/manifest_samples.csv` — 11-support labels + `submission_id`
4. `Model 1/Model 1 Experiment 3/cv_families.csv` — the 16 fold keys
5. `Model 1/Model 1 Experiment 3/features_soil.csv` — G0b reference
6. `data/sample_submission.csv` — id order for G3
7. `Model 8/Experiment 3/ruler.py` and `d50.py` — copied BYTE-FOR-BYTE into this folder
   (stand-alone experiment; never imported across experiment folders)

**Must NOT be read:** image/tile pixels (no file opens under `data/tiles`, `training_down`,
`testing_down`); camera, ppm, EXIF, ICC, site, sample_id, cv_group as model inputs (banned
features); any Model 9 path; any test label (none exists).

**Compute:** CPU only, minutes. Closed-form ridge on <= 127 x 12; nested LOFO is the dominant
cost and M8's equivalent finished in minutes on this laptop. numpy/pandas/scikit-learn only —
no torch, no GPU, no jupyter.

**Upload: NONE, and no .zip.** A Kaggle dataset exists only for experiments that need tiles or
torch on a GPU. Every input here is a local CSV; generation of both submission files is local
arithmetic. The flow is Model 8 E2/E3's: run locally, owner pastes the CSVs to the submission
page. **If a Kaggle-side mirror is ever wanted for the record, it is a separate owner decision,
not part of this plan.**

---

## 7b. AMENDMENT A1 -- Kaggle mirror added by owner decision (2026-10-02, after Tasks 1-5)

**Old wording (§7, as frozen at approval):** "If a Kaggle-side mirror is ever wanted for the
record, it is a separate owner decision, not part of this plan."

**What changed:** on 2026-10-02, after Tasks 1-5 were approved and executed, the owner asked for
"a .ipynb and data which we can run on Kaggle". That is the separate owner decision §7 reserved
to itself, so it is executed as an addition rather than as a contradiction of the frozen design.

**Scope of the amendment -- delivery only, science untouched:** arms A/B/C, all gates, all
tolerances, the alpha grid, the 1/n weighting, placebo seed 90001, the 20,000-draw family-cluster
bootstrap, the outcome tiers and the §6b final-submission rule are unchanged. No number already
reported is recomputed differently; the Kaggle run must reproduce them.

**How the mirror is built, and why no code is edited:** the dataset mirrors the repository's own
directory layout, so `ROOT = HERE.parents[1]` inside `ruler.py`, `transfer_eval.py` and
`build_m10e1.py` resolves correctly with **zero modifications**. The notebook copies that tree
into a writable location and then runs the same three commands the local run used
(`build_m10e1.py gates|score|submit`) as subprocesses. Byte-fidelity is therefore a property of
the shipped files, not a promise: the notebook prints sha256 of every code file and asserts the
emitted artifacts match the local hashes.

**Measured payload:** 5.5 MB total. The run reads cached tile *features*, never pixels;
`data/tiles` (276 MB), `training_down`/`testing_down` (716 MB) and the DINOv2 embeddings are not
shipped.

---

## 8. Risks and honest limits (pre-stated so they cannot be retrofitted)

| risk | status |
|---|---|
| **Errors-in-variables: image rows carry noisier x than soil medians, which attenuates ridge coefficients — B may be WORSE** | declared up front; a clean negative is a valid, reportable outcome (§6) |
| Between-image information may be worthless (richer aggregation was negative) | CONTESTED in §3; that is what makes this an experiment rather than a build |
| Internal CV may rank B/C wrongly | mitigated by design: external decides, internal report-only (§5) |
| The public board may not resolve a real gain (< 5 EMD) | declared: "better but unresolved" is a first-class outcome (§6) |
| Camera axis (0 iPhone training images) remains invisible to internal CV | unchanged from M8 E4's LIMIT; this experiment does not claim to fix it |
| Effective independent labels remain 24 (six near-duplicate pairs) | image rows are NOT 127 independent samples; CV holds out families precisely for this |
| Arm shopping after scores arrive | prevented: 3 arms frozen here, both new arms submitted, no selection, no tuning after any score is seen |

**What this experiment will NOT claim:** any Kaggle improvement before G2 scores land; that
internal CV predicts transfer; that image rows increase the effective sample size beyond 24
labels; anything about Model 9's territory.

---

## 9. Expected artifacts (all inside `Model 10/Experiment 1/`)

Source:
- `plan.md` (this file)
- `ruler.py`, `d50.py` — byte-copies from `Model 8/Experiment 3/`
- `image_features.py` — tile->image feature construction + G0b check
- `build_m10e1.py` — arms A/B/C, nested CV, placebo, bootstrap, G0–G3, submissions

Evidence:
- `reproduce_anchor.txt` — G0a + G0b evidence
- `arm_comparison.csv` — internal mean EMD per arm (report-only)
- `cv_per_soil.csv` — per-soil EMD, family id, chosen alpha, per arm
- `bootstrap.json` — family-cluster bootstrap (20,000 draws, resample the 16 families) for
  B−A, C−A, B−C
- `placebo.json` — permuted-label arm result
- `verdict.json` — machine-readable gate outcomes and §6 branch taken
- `experiment_report.txt` — written verdict incl. "awaiting Kaggle result" section
- `determinism.sha` — sha256 of all artifacts from both cold runs

Submissions (2):
- `Submission_Model10_E1_B.csv`
- `Submission_Model10_E1_C.csv`

---

## 10. Implementation order (each step is its own approved task — no step runs unapproved)

1. **Task 1 — scaffold:** create folder; copy `ruler.py`/`d50.py` byte-for-byte (sha256
   recorded); write `image_features.py`; G0b must pass; no fitting yet beyond what G0b needs
   (G0b needs no model fit — it is a pure aggregation check).
2. **Task 2 — harness:** `build_m10e1.py` with arms A/B/C and nested CV; run G0a + G1;
   **stop and report** whether they pass (freeze on failure).
3. **Task 3 — score:** internal arms, placebo, bootstrap; emit evidence artifacts; stop and
   report (internal numbers reported, not used to select).
4. **Task 4 — submissions:** fit B and C on all 24 soils / 127 images, predict the 35 test
   images, assert G3, write both CSVs; stop; **owner submits**.
5. **Task 5 — determinism + report:** second cold run, sha256 comparison (D1), write
   `experiment_report.txt` with scores "awaiting"; stop.
6. **Later:** record returned Kaggle scores, close the §6 branch, write verdict.

No git commit at any step unless the owner says so explicitly.

---

## 11. Success criteria for the experiment as a whole

- G0a, G0b, G1, G3 all pass (else the run is void and reported as such).
- Exactly 2 submission files exist, both contract-valid.
- The §6 branch taken is stated with the measured numbers, whatever it is.
- Both arms reported; no arm dropped because it lost.
- Freeze confirmation: no Model 1–8 file modified; arms, gates, thresholds, weighting and
  outcome rules byte-identical to this plan at report time.
