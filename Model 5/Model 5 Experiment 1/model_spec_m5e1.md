# Model 5 / Experiment 1 — SPECIFICATION (for approval, not yet approved)

**Subject:** label-free transductive domain adaptation of the hand-built tile features, tested
for specificity on a held-out camera.
**Status:** DESIGN. No code exists. Nothing has been run. This document is the thing being
approved.
**Date fixed:** 2026-09-29. Every threshold below was written before any Model 5 number existed.

---

## 0. Why this model exists, and what it is allowed to claim

Model 1 closed after four experiments: in-domain error never moved off 39 to 47, and every
external gain came from removing features and adding regularisation. Model 2 closed after three
experiments: DINOv2 features lost to 12 hand numbers (E1), lost to their own random-init twin at
every magnification (E2), and added nothing when concatenated (E3). A throwaway curve-ensemble
spike on 2026-09-29 then showed that averaging the hand model with the DINOv2 model buys 1.45 EMD
in-domain, not significant, with the permuted-noise control buying 0.56 of it.

What has never been the primary variable of any experiment here is the thing that is
demonstrably costing the most: **the camera**. Fitting and scoring inside one camera family, then
predicting the other, costs **+16 to +18 EMD** — the single largest measured loss in the project,
roughly three times the whole 4.9 EMD budget available from changing the output parameterisation,
and more than five times anything the feature work has ever moved.

> **H5:** aligning the fitting camera's image-level feature distribution onto the unlabeled iPhone
> test-image distribution, by a closed-form CORAL transform, reduces mean EMD on a camera that was
> neither fitted on nor used as the alignment target, relative to the identical unaligned pipeline.

**H5 is not assumed true. P1 predicts it is refuted.** The reasons stated in advance: Model 2
showed the informative structure of these images is not expressible as a linear re-mapping of
these features; the camera shift includes resolution and texture changes that a feature-space
affine transform cannot undo once the features have already been aggregated to image medians; and
the one time this project has tried invariance cheaply — Model 1's E7 — it found the easy versions
were either circular or dependent on device identity.

**What a pass would and would not mean.** It would mean the mechanism is real on two measurable
camera pairs. It would **not** forecast a Kaggle score: the production target is iPhone and iPhone
is never scored locally, CAM+RES is ordinal-only by Model 1's E4 ruling, and the public leaderboard
scores a fixed 3 of 10 soils while the final ranking is `0.30*public + 0.70*private`.

---

## 1. Facts the design stands on (measured, not recalled)

| quantity | value | source |
|---|---|---|
| labelled training soils | 24 | `manifest_samples.csv` |
| test soils / images / tiles | 10 / 35 / 435 | manifests, 256 px materialised |
| Motorola-family soils | 23 | `manifest_images.csv`, camera first token |
| Samsung-family soils | 22 | same |
| soils photographed by both | 21 | same |
| direction A (fit Motorola 23 → score Samsung) | 22 evaluation soils | computed |
| direction B (fit Samsung 22 → score Motorola) | 23 evaluation soils | computed |
| test camera families | iPhone 14 (14 images), iPhone 16 (21) | manifests |
| measured camera transfer penalty | +16.2 (D) / +18.2 (M1) EMD | Model 2 E1 record |
| hand features available per tile, all splits | yes, one cache CSV, 1976 rows | `Model 1/Model 1 Experiment 3/.cache/` |

**Consequence: Model 5 needs no new artifacts, no torch, no GPU, no Kaggle run and no dataset
upload.** Everything it consumes already exists on disk.

---

## 2. The one variable, and what is frozen

**Inserted step:** `tiles → image median → ALIGN → soil median → frozen head`. The transform acts
on the 12 hand features at image level. Nothing else anywhere changes.

**Frozen and reused unchanged:** all preprocessing and `data/`; the 12 control features
(`e4, e8, e16, lum_sd, grad_mean, R, G, B, sat, lum_p10, lum_p50, lum_p90`); tile→image→soil median
aggregation; `StandardScaler`; rank-3 curve basis; closed-form ridge; the alpha grid
`0.03 … 1e6`; monotone projection; trapezoid EMD on `log10 d` over the 11 ISO supports; the 16
curve-distance CV families; nested alpha selection; the E4 parsimony rule; `config_hash
010f44c36c74`.

**Prohibited, by rule and by assertion:** camera name/family, device make or model, ppm, EXIF, ICC
profile, site, sample id — as features, as conditioning variables, or as keys the transform looks
up. Test labels do not exist and are never referenced. The alignment function's signature accepts
two feature matrices and nothing else; a static test asserts the produced transform is bit-identical
when the target rows' labels are deleted or permuted.

**Alignment target is the 35 iPhone test images only.** A runtime assertion requires the target
tile/image set to be disjoint from every training-camera image, which is what keeps the evaluation
non-circular by construction.

---

## 3. Arms

| # | arm | transform | role |
|---|---|---|---|
| 1 | `NONE` | identity | baseline; every difference is against it |
| 2 | **`CORAL`** | mean + Ledoit-Wolf-shrunk covariance → iPhone images | **the gated arm** |
| 3 | `MEAN` | mean shift only, covariance untouched | mechanistic decomposition: is the gain a colour/gain offset? report-only |
| 4 | `CORAL_INDEP` | CORAL toward a target whose 12 feature columns are each shuffled **independently**: every per-feature mean, variance and marginal distribution of the real test set is preserved exactly, while the cross-feature covariance is destroyed | attribution control, **inside the gate**. See section 13 for why the row-permutation version first approved here had to be replaced |
| 5 | `CORAL_SELF` | CORAL toward the fitting camera's own image features | implementation canary; mathematically near-identity |
| 6 | `CORAL_TILE` | CORAL estimated at tile level | sensitivity, cannot fire the gate |
| 7 | `CORAL_SOIL` | CORAL estimated at soil level (10 target points, expected to degenerate) | sensitivity, cannot fire the gate |

Each arm runs in both directions, so 7 arms × 2 directions = 14 fitted cells, plus
reproductions. All are CPU and deterministic.

---

## 4. Measurement

For each direction *d* and arm *a*:

1. Select alpha by nested leave-one-family-out CV **within the source camera's labelled soils
   only** — the score of the procedure, never the score of the best alpha.
2. Fit on all source soils; predict the evaluation camera's soils.
3. `err_d_a(s)` = trapezoid EMD for evaluation soil *s*.
4. Paired per-soil difference `Δ_d_a(s) = err_d_NONE(s) − err_d_a(s)`; positive means the arm
   helps.

**Primary quantity:** `Δ_a` = mean of `Δ_d_a` over all 45 evaluation rows (22 + 23), reported per
direction as well.

**Uncertainty.** Paired bootstrap, 4000 resamples, percentile 95% CI, seed pinned at 20260934,
**resampling unit = soil identity, not row.** The 21 soils that appear in both directions move
together as one cluster, because their two rows share one underlying true curve; the 2
Motorola-only and 1 Samsung-only soils are their own clusters. Naive row resampling would treat
correlated errors as independent and narrow every CI by roughly √2 — which is exactly the kind of
error that turns a null into a discovery.

---

## 5. The gate — every clause numeric, fixed before the run

```
G0  SANITY      the full 24-soil, unaligned, no-holdout configuration reproduces Model 2 E3's
                recorded control score 43.0217 within 0.05 EMD.
G1  MAGNITUDE   Δ_CORAL  >=  3.00 EMD                     (pooled over 45 rows)
G2  DIRECTION   Δ_A >= 1.00 EMD AND Δ_B >= 1.00 EMD,
                and neither direction's clustered 95% CI lies entirely below 0
                (no direction may be significantly worse).
G3  SIGNIFICANCE pooled clustered 95% CI of Δ_CORAL has lower bound > 0.
                Per-direction significance is deliberately NOT required (see below).
G4  SPECIFICITY Δ_CORAL - Δ_CORAL_INDEP >= 1.50 EMD AND its clustered CI excludes 0,
                AND Δ_CORAL_INDEP <= Δ_CORAL.
                Because CORAL and CORAL_INDEP share an identical first-moment correction and
                differ ONLY in whether the matched covariance structure is real, this clause
                reads: the gain must come from matching the test set's actual correlation
                structure, not from applying any large same-marginal linear map. A gain that
                CORAL_INDEP also achieves is variance rescaling, i.e. regularisation.
G5  CANARY      |Δ_CORAL_SELF| <= 0.50 EMD pooled AND <= 0.50 in each direction.
                Violation => verdict INVALID-IMPLEMENTATION and NO CONCLUSION AT ALL,
                including no refutation. A broken transform must not be allowed to
                manufacture a null we would then believe.
G6  BOUNDARY    see 5a - reported per DIRECTION, never pooled, never hidden by the other one
G7  LABEL-BLIND static + runtime assertions of section 2. Failure stops the run; it is not a
                result.

clause test: G0 AND G1 AND G2 AND G3 AND G4 AND G5 AND G6 AND G7   (all eight must hold)

SUBMIT = TRUE  <=>  VERDICT == "SUPPORTED"
```

**The conjunction above is what the verdict is derived from; it is not the submission test.** G6
fails only when *both* directions are alpha-clipped, so the conjunction read alone would let the
one-direction caveat case submit — which sections 5a and 7 forbid in as many words. The two
statements are therefore ordered: **clauses → verdict → submit**. Cell G1 computes the conjunction,
folds in the single-direction caveat, and then asserts `SUBMIT == (VERDICT == "SUPPORTED")`, so the
implementation cannot drift from this sentence quietly. Enumerated:

| verdict | submission |
|---|---|
| `SUPPORTED` | **permitted**, provided the eight structural gates pass |
| `SUPPORTED-WITH-BOUNDARY-CAVEAT` | none |
| `REFUTED-WITH-BOUNDARY-CAVEAT` | none, and no permanent closure |
| `INCONCLUSIVE` | none |
| any other verdict | none |

### 5a. The alpha-boundary rule, stated per direction

A boundary hit must never be pooled away, and it must never be able to confer either of two
privileges it has not earned: the right to submit, or the right to declare Model 5 permanently
closed. So the rule is symmetric.

**Reported, always, for every arm and both directions:** the alpha the nested procedure selected,
its status `FLOOR | INTERIOR | CEILING`, the oracle alpha, and the grid spread. Status is computed
per direction; there is no pooled boundary status. The baseline arm's status is reported too, as a
diagnostic flag `symmetric_boundary` when `NONE` and `CORAL` land on the same edge — that flag
changes no verdict, it exists so a reader can see whether the grid constrained both arms equally.

**Exactly one direction at a boundary:** the measured verdict stands but is labelled
`SUPPORTED-WITH-BOUNDARY-CAVEAT` or `REFUTED-WITH-BOUNDARY-CAVEAT` as appropriate, and:

- **no submission is permitted**, even in the `SUPPORTED` case. A gate that passed while the grid
  was clipping one of its two halves is not a gate that passed cleanly, and the whole point of the
  threshold structure is that it must not be argued about after the fact. What it does earn is a
  documented reason to reopen, as a separate owner-approved decision with an amended grid.
- the `REFUTED` case **does not trigger the section 10 closure rule.** A direction that was
  boundary-clipped is a direction where the arm may have been handicapped, so a failure there is
  not evidence of absence. The record must say the closure is withheld pending the caveat.

**Both directions at a boundary:** verdict `INCONCLUSIVE`. Neither `SUPPORTED` nor `REFUTED`, no
submission, and no closure claim of any kind. When the grid constrains both halves, the measured
difference is between a transform and a truncation, and nothing about the camera has been learned.

Neither caveat can ever produce a file: `SUBMIT` is true for `SUPPORTED` alone (section 5), so the
caveat removes the privilege structurally rather than by convention. Where a caveat and a specific
failure diagnosis both describe the same measurement, **section 6a** decides which label is written.

**A grid amendment is never applied inside a run.** Extending `alpha_grid` beyond 0.03 or 1e6 would
change the frozen head definition shared with Model 1 and Model 2, so it is a separate decision
requiring explicit owner approval before any rerun, exactly like the other amendments this project
has kept out of finished experiments.

**Boundary convention, stated so it cannot be argued later:** every threshold above is
*inclusive* - a value exactly equal to it passes. CIs are two-sided percentile intervals from
the same 4000 draws used by every clause, so no clause gets a resample the others did not.

**Where 3.00 and 1.50 come from** — not invented round numbers:

- **3.00 EMD** is (i) about 18% of the +16 to +18 EMD camera penalty that is the explicit target of
  the intervention; (ii) more than three times the 0.9 EMD information-specific effect the
  curve-ensemble spike measured on 2026-09-29 and correctly treated as indistinguishable from
  noise; and (iii) larger than every arm difference this project has ever been able to resolve —
  the E1 probe differing from E3 by 0.674 EMD on 3 public soils was declared a *tie*, and the
  D-versus-M1 gaps E1/E2 measured were 0.3 to 1.9 EMD. It is deliberately far below the 15.0
  `ORDINAL_FLOOR`, which was calibrated for ranking ridge alphas over 24 in-domain soils and does
  not transfer to a paired per-soil difference with 45 rows.
- **1.50 EMD** in G4 is exactly half the G1 floor, chosen so that an arm sitting on the floor
  cannot be explained by a placebo also sitting on the floor.

**Why G3 does not demand per-direction significance.** Two independent 95% tests is a
multiple-comparison tax that mostly raises the false-negative rate; the substantive protection
against a fluke is G2's sign-and-magnitude consistency plus G4's attribution control, both of which
are harder to satisfy by luck than one wide CI. This is a judgement made in advance and written
down, so that it cannot be re-litigated after seeing which side it favours.

**`MEAN` is mechanistic, not a gate clause.** `CORAL` does **not** have to beat `MEAN` for H5 to be
supported. If `Δ_MEAN >= Δ_CORAL`, the honest reading is that the exploitable shift is a
first-moment colour/gain offset and the covariance half of CORAL does nothing; that is recorded as
a finding about the nature of the shift and it changes what a later experiment would try, but it
does not change the verdict. Gating on *which component* carries a real effect would be gating on
mechanism rather than on existence.

---

## 6. Interpretation rules — fixed now, applied later

| outcome | verdict | action |
|---|---|---|
| all clauses pass, no direction clipped | `SUPPORTED` | **submission permitted** (section 7) |
| G1 fails, G4 passes | `SUPPORTED-BELOW-BAND` | **no submission.** Mechanism real but smaller than the floor. Record it; do not dress it as a win |
| G1 passes, G4 fails | `REGULARISATION-ARTEFACT` | **no submission.** The gain is shrinkage, not the measured shift |
| G2 fails | `DIRECTION-INCONSISTENT` | **no submission.** Treated as refuted for decision purposes |
| G3 fails, G1 passes | `NOT-RESOLVED` | **no submission.** Effect size without statistical support |
| G5 fails | `INVALID-IMPLEMENTATION` | **no scientific conclusion at all**, in either direction. Fix the code, rerun; the gate itself stays unchanged |
| CORAL at a boundary in BOTH directions | `INCONCLUSIVE` | no claim either way, no submission, no closure. A grid amendment is a separate owner decision |
| CORAL at a boundary in ONE direction, gate otherwise passes | `SUPPORTED-WITH-BOUNDARY-CAVEAT` | **no submission** despite the pass (section 5a). Documents why a rerun is warranted |
| CORAL at a boundary in ONE direction, gate otherwise fails | `REFUTED-WITH-BOUNDARY-CAVEAT` | no submission, and **section 10's permanent closure does not fire** |
| G6 triggers | `INCONCLUSIVE` | no claim either way; a grid amendment becomes a separate owner decision |
| G1 fails outright | `REFUTED` | **Model 5 closes at one experiment**, per the closure rule below |

### 6a. Verdict precedence — the table above overlaps, so the order is part of the contract

Two rows can describe the same measurement. A run in which exactly one direction is alpha-clipped
**and** G2 fails matches both "`G2 fails` → `DIRECTION-INCONSISTENT`" and "`CORAL` at a boundary in
ONE direction, gate otherwise fails → `REFUTED-WITH-BOUNDARY-CAVEAT`". A tie-break is therefore
required rather than a matter of taste, and it is fixed here, before any score exists, so that it
cannot be chosen by whichever label is more convenient afterwards. Cell G1 evaluates the ladder top
to bottom and stops at the first match:

| # | condition (first match wins) | verdict |
|---|---|---|
| 1 | `NOT (G0 AND G5 AND G7)` | `INVALID-IMPLEMENTATION` |
| 2 | boundary in **both** directions | `INCONCLUSIVE` |
| 3 | `G1 AND G2 AND G3 AND G4` | `SUPPORTED`, or `SUPPORTED-WITH-BOUNDARY-CAVEAT` when exactly one direction is clipped |
| 4 | `G1 AND G2 AND G3 AND NOT G4` | `REGULARISATION-ARTEFACT` |
| 5 | `NOT G2` | `DIRECTION-INCONSISTENT` |
| 6 | `NOT G1 AND G4` | `SUPPORTED-BELOW-BAND` |
| 7 | `G1 AND NOT G3` | `NOT-RESOLVED` |
| 8 | anything else | `REFUTED`, or `REFUTED-WITH-BOUNDARY-CAVEAT` when exactly one direction is clipped |

**Why this order.** The boundary caveat is attached by this spec to exactly two outcomes — a pass
with a caveat (row 3) and a plain refutation with a caveat (row 8) — so it is applied only there,
and a *specific diagnosis* (rows 4 to 7) is named in preference to a caveat, because it carries more
information about what failed. Validity precedes every scientific reading: a broken transform
supports no conclusion in either direction. Both-directions clipping precedes all of them, because
a grid truncated at both ends makes the direction and significance tests themselves uninterpretable.

**What this does not do.** The precedence assigns *labels* only. It moves no threshold, no arm, no
clause test, no CI, and no submission outcome. `SUBMIT` is true for `SUPPORTED` alone (section 5),
and row 3 can produce `SUPPORTED` only when nothing is clipped, so no ordering choice here can ever
license a file that section 5a would forbid.

**Pre-registered predictions** (`blind` = written before the run and never revised afterwards):

- **P1 [BLIND, the main one]** — H5 is refuted: `Δ_CORAL < 3.00` EMD, or its CI includes 0.
- **P2 [BLIND]** — `Δ_CORAL_INDEP` is within ±1.0 EMD of `Δ_CORAL`; most of any gain is variance
  rescaling rather than matching the real joint structure.
- **P3 [BLIND]** — `Δ_MEAN >= Δ_CORAL - 1.0`: if anything works, the mean shift carries most of it,
  because a colour/gain offset is the largest and most linear part of the measured shift.
- **P4 [BLIND]** — `|Δ_CORAL_SELF| < 0.10` EMD: self-alignment is identity up to shrinkage. This is a
  prediction about *magnitude*; it is not the same thing as G5, which is a *validity bound*. A
  self-alignment drift between 0.10 and 0.50 refutes P4 while leaving the run valid, and that is
  the intended reading, not a contradiction.
- **P5 [BLIND]** — the camera penalty measured here (the `NONE` arm's loss against same-camera
  in-domain error) is **not reduced by more than 25%** by any arm; a full fix would need a
  representation change, not a re-mapping.
- **P6 [BLIND]** — `CORAL_TILE` and `CORAL_SOIL` disagree with the gated arm in *magnitude* but not
  in *sign*; if the sign flips on the estimation unit, the result is an artefact of that unit and
  must be reported as uninterpretable.

---

## 7. Submission rule — exact and conditional

`SUBMIT = TRUE` **if and only if** `VERDICT == "SUPPORTED"`. This is the single submission test: the
clause conjunction gates the *verdict*, this rule gates the *file*. A
`SUPPORTED-WITH-BOUNDARY-CAVEAT` (section 5a) does **not** qualify, and neither does
`REFUTED-WITH-BOUNDARY-CAVEAT`, `INCONCLUSIVE`, or any other verdict. Cell G1 asserts the identity
`SUBMIT == (VERDICT == "SUPPORTED")`, so an edit that let `SUBMIT` be true beside any other verdict
fails the run rather than shipping a file.

- Arm: `CORAL`. File: `Submission_Model5_E1.csv`, written by the notebook itself under that name,
  never renamed.
- Configuration is the **production recipe**, which is deliberately *not* the tested
  configuration: fit on **all 24 labelled training soils from both cameras**, align those 24 soils'
  image features onto the 35 iPhone test-image features with the same CORAL transform, predict the
  10 test soils.
- Alpha comes from the same nested procedure on the full 24 soils; no alpha may be chosen by
  looking at the submission.
- The eight existing structural validation gates run before writing and must pass.
- **Declared difference:** the submitted model was never itself evaluated by the gate — it uses
  two more fitting soils than either direction, and its alignment target is the same set it
  predicts. That is the legitimate application of a validated recipe, not a measurement, and the
  record must state it in those words rather than implying the gate scored this arm.
- **No probe.** If the verdict is anything other than `SUPPORTED`, no CSV is written and the
  notebook says which clause failed.

---

## 8. Error handling and how the run stops

- Missing feature cache, non-disjoint alignment target, transform not reproducible across two
  identical runs, or any label reaching the alignment function → raise and name the check. No
  degraded mode, no silent fallback. Model 2 E3 ran with no mock path for exactly this reason and
  it paid for itself twice.
- Every cell must print something. A cell that prints nothing is treated as a bug, because that is
  how E1's `ndarray.median()` crash and E3's silent figure cell were caught.
- Determinism: run twice, byte-compare every emitted data file.
- Nothing in `Model 1/`, `Model 2/`, `data/` or any previous experiment folder is written to. The
  run outputs into `Model 5/Model 5 Experiment 1/` only.

## 9. Expected artifacts

`Experiment1.txt` (record, generated from live objects, including this gate verbatim and the
P1–P6 table); `transfer_results.csv` (per arm × direction: mean EMD, Δ, CI, alpha, wins);
`per_soil_transfer.csv` (all 45 rows with their cluster ids); `arm_config.csv`;
`alignment_report.csv` (per feature: source mean/sd, target mean/sd, post-transform mean/sd, and
the shrinkage weight actually used); `placebo_report.csv`; `canary.json` (G0 and G5 outcomes);
`predictions.csv` (P1–P6 mechanically evaluated); `gate.json` (clause, measured value, threshold,
pass, verdict); `Submission_Model5_E1.csv` **only if** `SUPPORTED`; `Model5_Experiment1.ipynb`.

## 10. Closure rule, fixed in advance

If `Δ_CORAL` misses G1 **with no boundary caveat** (section 5a), **Model 5 closes at one
experiment** and no follow-up alignment variant is scheduled: the cheap, closed-form, label-free form of domain adaptation will have been tested on
the only two measurable camera directions that exist, against a permuted-target placebo, and found
insufficient. Remaining competition time then goes to making the current best submission
(60.56, Model 2 E1's probe) robust rather than to a fourth way of asking the same question.

---

## 11. Explicit statement on pre-registration

**Yes — every number in section 5, every verdict name in section 6, every prediction in P1–P6, the
submission rule in section 7, the resampling unit in section 4, the estimation unit, arms, seeds,
and the closure rule in section 10 are fixed before the experiment runs.** After the run they may
change for exactly two reasons, and no others: (a) correcting code that disagrees with this
document, which must be disclosed in the record the way Model 2 E3 disclosed its two gate bugs; or
(b) the owner explicitly re-approving a change *before* a rerun. Nothing may be revised to convert
a failure into a pass, and no threshold may be justified by the data it is about to judge.

## 12. Open limitation I will not paper over

Two camera directions and 45 evaluation rows is the entire measurable universe of this problem. A
`SUPPORTED` verdict rests on two domain pairs; a `REFUTED` verdict is weaker than it looks for the
same reason — absence of evidence across two pairs is not evidence of absence on iPhone. The
specificity design is the strongest thing available with this data, not a guarantee, and the record
will say so in those words.

## 13. Design correction made before any run, with owner approval

The placebo arm first approved for this specification was `CORAL_PERM`: CORAL toward the test
features with their **rows** permuted. That construction is **provably inert**. CORAL's transform
depends on the target only through its mean vector and covariance matrix, and both are invariant to
row order — measured on a correlated 12-column table, permuting rows changed the mean by 1.1e-16 and
the covariance by 4.4e-16. `Δ_CORAL − Δ_CORAL_PERM` would therefore have been exactly 0.00 in every
run, G4 could never have passed, and the verdict would have come out `REGULARISATION-ARTEFACT`
regardless of the data.

Why the identical-looking control was valid in Model 2 E3 and is not here: E3's `SHUF` block fed a
*predictive* model, where the row-to-label pairing is the signal being destroyed. A domain-adaptation
target is used only for its moments, and moments have no pairing to destroy.

Replacement, approved by the owner on 2026-09-29 **before any code was written**: independent
per-column shuffling of the target matrix (`CORAL_INDEP`), which preserves every marginal exactly and
destroys the joint structure, so the arm differs from `CORAL` in one factor only. No threshold, no
other arm, no interpretation rule and no submission rule was changed by this correction.

## 14. Fold-level leakage rule - found at implementation, approved before any score existed

**14.1 Why the original cross-camera evaluation leaked soil information.** The specification as
first written said "fit on Motorola's 23 labelled soils, predict Samsung's 22". Taken literally,
the fit set and the scored set are two views of largely the same soils: 21 of the 22 soils scored
in direction A had their true grain-size curve in the training data, reached through Motorola
images, before being scored through Samsung images. A ridge fitted on 23 soils with a rank-3 curve
basis has enough capacity to lean on soil identity, so a held-out-camera error computed this way is
not a held-out error at all. It also simulates the wrong task: in production the 10 test soils
appear nowhere in training, so there is no analogue of "this soil is in the fit set" for the model
to exploit. The consequence is unpredictable rather than merely optimistic - memorisation shrinks
every arm's error toward the soil's own curve, which compresses the differences the gate is built
to detect, and it does so by different amounts for arms whose feature geometries differ.

**14.2 How the fold-based evaluation prevents it.** Each direction is divided into CV folds over
the fitting camera's soils. For a fold holding out family F: the fit set is the fitting camera's
soils excluding F; the alignment transform is estimated from the surviving fitting-camera images
plus the 35 test images; the model is fitted and alpha selected inside that fit set; and the soils
scored in that fold are the evaluation camera's images of the held-out F soils. Therefore every
scored soil is unseen by the model that scores it - unseen in both cameras, not just the fitting
one. A soil with no image in the fitting camera at all (H183 in direction A, 2 soils in B) can
never enter a fit set, so it is scored once against the full fitting pool rather than dropped.
`check_transfer_eval.test_no_soil_is_scored_by_a_model_that_saw_it` asserts the two properties that
make this true - every scored soil appears exactly once, and no fold's fit and score sets intersect
- and a mutation test confirms the check fails if a fold is quietly given back its held-out soils.

**14.3 Why the 45 evaluation rows and 24 bootstrap clusters are unchanged.** The scored set is
still "every soil with an image in the evaluation camera": 22 in direction A, 23 in direction B, 45
rows. Folding changes which model produces each row, not which rows exist, so the statistical
universe the thresholds were calibrated against is intact - still 45 rows, still 24 soil identities,
still the same 21 soils contributing two rows each. Nothing was discarded to gain cleanliness, which
is why no threshold needed revisiting and why the contract's counts section did not move.

**14.4 This is validation structure, not a scientific threshold.** Unchanged and explicitly restated:
the arms (`NONE`, `CORAL`, `MEAN`, `CORAL_INDEP`, `CORAL_SELF`, `CORAL_TILE`, `CORAL_SOIL`), every
gate clause and its number (G0 0.05, G1 3.00, G2 1.00, G4 1.50, G5 0.50, P4 0.10), the verdict
names and their table, the per-direction alpha-boundary rule in section 5, the submission rule in
section 7 and the closure rule in section 10. What changed is only how a score is produced. The
change is asymmetric in one direction only: it can make `CORAL` look worse, never better, because
it removes an advantage the arm previously had. Had it been introduced after seeing results it would
have been suspect; it was introduced in Task 3, before a single Model 5 number existed, and
`GATE G0` still reports the same unaligned control score it was verified against.
