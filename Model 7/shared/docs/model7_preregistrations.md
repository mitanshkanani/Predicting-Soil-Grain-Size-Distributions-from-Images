# Model 7 candidate boundaries and preregistrations

Date fixed: 2026-09-29. Status: **boundary pass and pre-registrations. No M7 candidate has been
implemented, fitted, tuned, or scored, and no submission has been generated.** Shortlist frozen as
M7-A, M7-C, M7-B, M7-E, M7-T; this document resolves their boundaries and pre-registers M7-A and M7-C.

**Owner decision recorded 2026-09-29, after this document was approved:** run M7-A first. It is
registered as Model 7 / Experiment 1, and its pre-registration was relocated verbatim to
`Model 7/Approach A/model_spec_m7a.md` with the frozen K = 4 definition, the alias-separability gate,
the controls, the family-honest LOFO protocol, the G0 anchor 43.0217308796477 and the G1 >= 3.00 EMD
threshold unchanged. M7-C stays pre-registered below and is **not run** unless M7-A is rejected by its
pre-gates or fails G1. M7-T stays suspended: the ambiguous M5 closure is not read as permission. M7-B
and M7-E are not revived. No tuning, feature expansion, or post-hoc selection after the CV result, and
no submission unless the pre-registered criteria are satisfied.

`research_map_1_2_5_6.md` is the evidence baseline. Correction records (`AGENT_BRIEF.md` section 11
CORRECTION 2 and sections 12 and 13; `Model 5/instructions.txt` CORRECTION block;
`Model 5/Model 5 Experiment 1/Experiment1.txt` sections 16-19) are preserved exactly as written and
are not touched here. No withdrawn claim is revived, including the leaky 1-NN bound.

Two scales stay separate throughout. **Internal CV EMD** = leave-one-CV-family-out on the 24 labelled
soils; the shipped anchor is **43.0217308796477**. **External Kaggle EMD** = the host's score on real
hidden soils; the current baseline is **60.56167** (Model 2 E1). They are different evaluation scales
and are never compared arithmetically. Every "headroom" number below is an internal-scale **estimate
unless it cites a measurement**, and is labelled.

---

## 1. Boundary rulings

### 1a. M7-T (transductive geometry from the 35 unlabeled test images) - **SUSPENDED, not killed by rule**

The question was whether M5's permanent closure explicitly covers **all** use of unlabeled
test-image statistics. **Answer: no, it does not.** The three records that state the closure scope
it to a mechanism, not to a data source:

> `AGENT_BRIEF.md` section 12, closed row: "**Label-free camera adaptation (CORAL / moment matching, incl.
> transductive use of the unlabeled test images)** | **REFUTED, CLOSED (M5 E1)**"

> `Model 5/Model 5 Experiment 1/model_spec_m5e1.md` section 10: "**Model 5 closes at one experiment**
> and no follow-up **alignment variant** is scheduled"

> `Model 5/instructions.txt`, WHAT IS FORBIDDEN NOW: re-running with "a different alignment target
> because the answer was unwelcome" is "a NEW pre-registered experiment requiring an explicit owner
> decision, not an edit"

The parenthetical closes the transductive **variant of camera adaptation**. It does not close every
statistic ever computed from a test image. The exact distinction, so it can be audited rather than
asserted:

| | closed by M5 | M7-T as proposed |
|---|---|---|
| what is estimated | a map taking the source feature distribution onto the target's | the subspace and penalty geometry the head fits in |
| operates by | matching means and covariances | fixed-rank geometry; no distributional matching |
| reads camera identity | yes, its directions are camera-defined | no; test rows enter only as unlabeled feature rows |
| touches the inputs | yes, re-maps them | no, changes the estimator's geometry |
| is an "alignment variant" | - | no |

**The spirit sentence cuts the other way, and I will not resolve that ambiguity in my own favour.**
The same section 12 block that lists what is NOT dead says the in-domain headroom is
"**Not reachable by re-summarising the same hand-built features (Model 2 tested that three ways and closed), nor by adapting the features to the test distribution (Model 5 tested that and closed)**".
M7-T's motivation was to make the estimator fit the deployment distribution's geometry. Under that
sentence the idea is closed **in intent** even though no closure row names it.

**Ruling recorded, not assumed: M7-T is suspended and is not preregistered here.** It is not killed by
rule, and ambiguity is not permission. It returns only if the owner rules that "adapting the features
to the test distribution" meant distribution matching specifically rather than any test-informed
estimator. If it returns it returns as a fresh pre-registered experiment whose first clause is an
inertness check, because the honest risk is structural rather than statistical: at 12 dimensions and
rank 3, the train+test pooled subspace may be nearly identical to the train-only one, which would make
the arm inert rather than merely null. Its prize is also small - the camera target the closure itself
cites is only ~5.55 EMD.

**Self-correction.** My first draft of this section answered "yes, the closure covers it" and killed
M7-T on the parenthetical alone. That over-read all three records. The distinction above is the answer
to the question as it was actually asked.

### 1b. M7-E (tail-emphasising loss weighting) - **COLLIDES, recommend kill**

`AGENT_BRIEF.md` section 12 already cancels "**Alternative classical regressors (robust, boosted trees)**"
with the reason "Huber is second-order on a model dominated by the curve basis", and the map measures
the neighbouring case: L1 fitting 41.22 vs 41.15, i.e. the loss geometry is *not* the lever. M7-E is
the opposite-signed version of a measured null. It is not revived here; it is flagged. Its 14.40 EMD
tail ceiling stays in the map as arithmetic, not as a candidate.

### 1c. M7-A boundary correction found while checking its premise

The map's B1 pitch named three candidate statistics. Checking each against the records:

| M7-A ingredient | status after checking |
|---|---|
| Angle-averaged (radial) power spectrum position, as an **absolute** value | **already computed and excluded.** `spec_centroid_cpm`, `dom_wavelength_mm` exist in the tile cache and were left out of the 12 (`Model 1 E3 selection_rule.json`) |
| Band **ratios** as scale-free frequency surrogates | **cancelled, weakly** (`AGENT_BRIEF.md` section 12) - the argument was about the absolute family's failure, so the scale-free form was never actually tested |
| **Angular / directional** structure | **never computed.** `preprocess/verify.py:163-182` radially averages, discarding angle; `grad_mean` is magnitude only (`build_m1e3.py:337-338`) |
| Connected-component / blob size | **REFUTED at grain scale** (granulometry ~ 198 EMD); untested at clump scale |
| **Cross-tile** arrangement at image level | **dead from geometry, measured today.** A 256 tile is 56.23 mm and an image holds a 4-5 x 2-3 grid: 10-15 points per image. No two-point statistic is estimable on that |

Consequence: M7-A is **narrowed to within-tile directional and scale-free organisation statistics**.
The cross-tile variant is dropped before any code is written, because the grid cannot support it.

### 1d. M7-C boundary correction found while verifying its mechanism

The owner asked me to verify the 1.68 EMD alpha-selection cost. `scratch/check_m7c_boundary.py`
(11 checks, exit 0) established:

- Fold honesty **asserted, not assumed**: every soil scored exactly once; **0** leaks of a scored
  soil or any of its CV-family siblings into the fitting set; deterministic across a second full pass.
- The comparison is honest on one side only. Nested (honest) = **42.83**; best fixed alpha = **41.15**
  at alpha 3.0; **1.68 is ORACLE REGRET**, because the oracle alpha is the argmin of the mean error
  over all 24 soils including the ones being scored. Only **7 of 16** folds independently choose 3.0,
  so no fold-local procedure can reach it. **The maximum a better selector can win is 1.68 EMD**, and
  a realistic claim is less.
- Selection instability is the measured mechanism: the nested rule chooses {3.0: 7, 1.0: 7, 0.3: 1,
  0.1: 1} across folds, i.e. it flips between adjacent grid points on half the folds.
- **The 40.20 arm is not a selector.** `BayesianRidge` learns its own penalty per column; its fitted
  values (median 0.0032, max 0.025) fall **below the frozen grid's floor of 0.03**, and only **2%**
  land within a factor of two of any grid value. So its gain cannot be attributed to better
  *selection* - it adds model capacity (per-column precision plus a learned noise precision).

Therefore M7-C is **restructured**: the primary registered intervention is a *selector-only* rule
inside the frozen alpha family, which changes one thing and nothing else. The adaptive estimator is
demoted to a separately-labelled arm that cannot authorise a submission, because attributing its gain
to the selector would repeat exactly the error the map's bucket A warns about.

---

## 2. M7-A - PRE-REGISTRATION

**Moved on 2026-09-29 to `Model 7/Approach A/model_spec_m7a.md` when the owner registered M7-A as
Model 7 / Experiment 1.** The text there is byte-identical to the approved version and is the
authoritative copy. This register keeps section 1 (boundary rulings) and section 3 (the unrun M7-C
pre-registration) together. Nothing in the M7-A definition changed: K = 4, the alias-separability
gate, the four controls, the family-honest LOFO protocol, the G0 anchor 43.0217308796477 and the
G1 >= 3.00 EMD threshold are exactly as approved.

---

## 3. M7-C - PRE-REGISTRATION (selector-only replacement inside the frozen alpha family)

**Hypothesis H7-C.** A measurable slice of internal CV error is created by *how alpha is chosen*, not
by what is fitted: the frozen nested rule flips between adjacent grid points on half the folds, and
the gap to the best-fixed-alpha oracle is 1.68 EMD. A deterministic, more parsimonious selection rule
inside the same frozen grid recovers part of that regret with no change to the model class.

**Mechanism, measured (this is the one candidate whose mechanism is already quantified).**
Nested/honest 42.83 versus best fixed alpha 41.15 at 3.0 -> **oracle regret 1.68 EMD**; per-fold
choices {3.0: 7, 1.0: 7, 0.3: 1, 0.1: 1}. Both endpoints rest on the asserted fold honesty in
section 1d. **1.68 is an upper bound on the prize, not an expected gain**, and it is a regret against an
oracle no procedure can reach.

**Exact intervention - primary arm.** Replace only the alpha selection step.
`SEL-1SE`: among the frozen 16-value grid, take the inner-fold errors; compute their standard error
across inner folds; choose the **largest** (most regularised) alpha whose inner error is within 1 SE
of the inner minimum; ties broken by the larger alpha. Nothing else changes - features, standardiser,
rank-3 basis, ridge, projection, EMD, fold structure and the 16-value grid stay byte-identical.
One parameter of the rule (1 SE) is fixed by convention, not tuned; no alternative band is tried, and
if the rule degenerates (band covers the whole grid) that is reported as an inert arm, not repaired
mid-run.

**Explicitly out of scope for this experiment.** The adaptive-penalty estimator that topped the
15-head sweep is **not** the primary arm: section 1d shows it fits its own penalty below the grid floor
(median 0.0032, only 2% within a factor of 2 of a grid value) and therefore changes model class, not
selection. It may be registered later as a separate experiment with its own hypothesis; it cannot
authorise a submission here, and its 40.20 sweep value is recorded as a *prior expectation*, not as
evidence for H7-C. Choosing the winner of the 15-head sweep would collide with a measured best-of-15
selection null of 15.29 EMD.

**Validation protocol.** Identical family-honest CV as 2 (16 families, no scored soil or sibling in
the fitting set, basis and standardiser fitted per fold, anchor G0 = 43.0217308796477 to 0.00e+00,
family-clustered paired bootstrap 4000 draws), plus the two-camera M5 transfer directions run
**report-only** - CAM+RES stays ordinal-only and may not select anything.

**Pre-registered nulls and controls.**
1. **Identity control:** the existing frozen nested rule, expected to reproduce 43.02 exactly.
2. **Degenerate-parameter check:** with the SE multiplier set to 0, SEL-1SE must reduce to the plain
   argmin and reproduce the control bit-for-bit; if it does not, the implementation is invalid.
3. **One-parameter null:** the apparent gain of tuning *one* scalar on this data, measured on permuted
   labels before the run - the instrument that closed M6-A (median +3.82, p90 +11.71 EMD), remeasured
   for this protocol rather than inherited.
4. **Capacity-neutral control (P-COLS equivalent):** confirm no feature, basis or row is touched, by
   asserting the design matrices are byte-identical apart from the chosen alpha sequence.
5. **Regularisation-artefact test (carried over from M5's G4 logic):** if the gain appears but the
   direction-consistency check reverses across the two camera directions, the gain is shrinkage, not
   recovered selection cost.

**Pass criteria (all must hold) -> SUPPORTED.**
- G1 gain >= **1.00 EMD** mean internal CV reduction, i.e. at least 59% of the 1.68 EMD regret.
  A candidate that cannot capture most of the very regret that motivated it has not established the
  mechanism.
- G2 family-clustered bootstrap CI of the gain excludes 0.
- G3 the gain clears the one-parameter permutation null.
- G4 direction consistency: the gain does not reverse in either camera direction (report-only ruler,
  so this is a consistency clause, not a ranking).
- G5 validity: anchor reproduced, determinism, finite, empty-cell checks.
- G6 no alpha clipping: the chosen alpha is interior to the frozen grid; a boundary choice in both
  directions gives INCONCLUSIVE, in one direction gives a caveat with no submission - M5E1's rule,
  reused verbatim rather than re-invented.

**Verdicts -> submission rule.** SUBMIT = SUPPORTED only, and precedence follows M5E1 6a unchanged
(validity -> inconclusive -> full pass -> artefact -> direction -> below-band -> not-resolved -> refuted).
SUPPORTED-BELOW-BAND, artefact, not-resolved and any caveat all mean **no submission**.

**Estimated headroom (estimate, and bounded by a measurement).** By construction <= 1.68 EMD internal;
realistic expectation **0.5-1.5 EMD**, with a substantial chance of a clean null.

**Path to the external baseline 60.56167.** Effectively none: a sub-1.7 EMD internal change is far
inside a 3-soil public band of [21.7, 69.9]. This candidate should be judged as a **defensibility and
robustness** experiment - the cheapest possible test of whether the pipeline is leaving money on the
table in its own selection rule - not as a leaderboard move. If the owner's priority is a legitimate
submission path, M7-A has the ceiling and M7-C has the cost advantage; that trade-off is the reason
these two are the finalists.

**Submission:** technically yes if SUPPORTED, but a SUPPORTED here is a below-band-grade result by
construction; I would not recommend spending the single authorised submission on it.

**Expected failure mode.** 1-SE parsimony selects a larger alpha and lands in the flat over-shrunk
region (grid upper values reach 93.02, essentially a constant curve), so the "regret" is not
recoverable by parsimony; or the rule is inert because inner errors are too flat to distinguish, in
which case the arm cannot move the metric and must be reported as structural, not as a measured null.

---

## 4. Side by side

| | **M7-A** within-tile spatial organisation | **M7-C** selector-only replacement |
|---|---|---|
| Bucket | B - genuinely new information | A - modifies the existing pipeline |
| Variable changed | **inputs** (4 new columns) | **alpha selection rule** only |
| Mechanism status | **unverified hypothesis**; premise (position-freeness) *verified* in code | **mechanism measured**: 1.68 EMD oracle regret + fold instability |
| Statistical support for the *idea* | none yet | the regret is a measurement, not an argument |
| Pre-registered gain bar | >= **3.00 EMD** internal | >= **1.00 EMD** internal |
| Ceiling | estimate 2-6, arithmetic tail ceiling 14.40 | **bounded** by 1.68, realistic 0.5-1.5 |
| Placebos | phase-scrambled, column-shuffled, best-of-4 null, linear-combination control | identity, SE=0 degenerate, one-parameter null, byte-identical-matrix check |
| Kills itself cheaply | yes - alias-separability gate before any fit; also killed by geometry for the cross-tile variant | yes - SE=0 degenerate check; anchor G0 |
| Build cost | higher: new feature module + contract test + gate | lower: one selection function, head untouched |
| Main risk | in-domain gain / external loss (`fr` trap); variance at n=24 | parsimony lands in the over-shrunk flat region; or inert |
| External route to 60.56167 | the only credible one left, still unconfirmable | effectively none |
| Submission if SUPPORTED | yes | yes, but not recommended as the one authorised run |
| Boundary conflicts | none, after narrowing (section 1c) | none; the adaptive estimator is *excluded* from the primary arm (section 1d) |

**Both candidates share these unchanged constraints:** G0 anchor 43.0217308796477 reproduced to
0.00e+00 before any arm is scored; family-honest folds only; no camera / ppm / EXIF / ICC / site /
sample-id as input or dispatch key; no random image-level split; CAM+RES ordinal-only and never a
selector; no model or threshold chosen by leaderboard position; external band written down before any
score exists; one submission; no re-run with relaxed thresholds after seeing results; verdict,
precedence and interpretation fixed before the run; historical results and correction blocks
untouched.
