# Model 7 / Approach A - SPEC (M7-A, within-tile spatial organisation features)

**Status: APPROVED AND FROZEN 2026-09-29 by the owner. Registered as Model 7 / Experiment 1.**

**Nothing below was reworded, re-thresholded or re-parameterised when it moved here.** It is the
approved pre-registration text, relocated verbatim so this folder is the single source of truth for
M7-A. Section 1 (boundary rulings) and section 3 (the unrun M7-C pre-registration) stay in
`model7_preregistrations.md` at the repository root, unchanged.

**Owner instruction on registration:** register M7-A first; keep the exact frozen K = 4 feature
definition, the alias-separability gate, the controls, the family-honest LOFO protocol, the G0 anchor
43.0217308796477 and the G1 >= 3.00 EMD threshold unchanged; do not submit unless the pre-registered
criteria are satisfied; no tuning, feature expansion, or post-hoc selection after seeing the CV
result. M7-T stays suspended; M7-B and M7-E are not revived; M7-C stays preregistered and unrun
unless M7-A is rejected by its pre-gates or fails G1.

---


**Hypothesis H7-A.** Some of the remaining internal CV error is carried by spatial *organisation*
of texture inside a tile - directionality and scale-free continuity - which the 12 features cannot
represent because each is a permutation-invariant summary of a tile's masked pixel multiset
(`preprocess/verify.py:105-128`; tile->image and image->soil aggregation are two plain per-column
medians). Adding four such statistics as frozen-pipeline inputs lowers family-honest internal CV EMD.

**Data source.** Existing materialised 256 px tiles and `manifest_tiles.csv`
(`tile_x, tile_y, grid_col, grid_row, tile_w_mm`). Tiles are *read*, never regenerated;
`data/Training`, `data/Test`, `data/processed_meta` are not modified; no preprocessing is re-run.
1946 qualifying tiles (soil_fraction >= 0.50, materialized), 1541 train / 405 test split.

**The four statistics, fixed now, K = 4 columns.** Computed per tile on the mask-interior grey field
`g` and the band-pass residual `e` already defined in preprocessing; then aggregated by the same
median rules as the existing 12.

**AMENDMENT 1 - owner-authorized, 2026-09-29, before Task 2.** The three cells below were amended from
their approved wording to state one measurement each, unambiguously. Nothing else in this
pre-registration changed: the hypothesis, K = 4, the four statistic *ids and roles*, the controls, the
protocol, gates G0-G7, the 3.00 EMD bar and the submission rule are exactly as approved. The owner
ruled on four divergences that the Task 1 review found between this text and the plan's code; each
ruling is recorded with the original wording preserved verbatim underneath it, and the pre-amendment
text is also preserved in the repository history. The A1 cell is unchanged.

| id | definition | why the 12 cannot express it |
|---|---|---|
| A1 | **structure-tensor anisotropy** `log((lambda2 + eps) / (lambda1 + eps))` of `(gx, gy)` over masked pixels, lambda1 >= lambda2 | `grad_mean` is `hypot(gx,gy)` magnitude only; orientation is never used |
| A2 | **autocorrelation e-folding length ratio** `L(0 deg) / L(90 deg)` of the residual `e`, each length measured in mm along its own profile. A **plain ratio, not a log-ratio.** A profile line lying wholly outside the mask is **skipped**, never substituted by 0.0; if every line in a direction is masked out, A2 is **undefined** for that tile. The value is scale-free because both lengths share a unit, so mm is a reporting statement, not a divisor | the existing radial spectrum is **angle-averaged** by construction |
| A3 | **normalised variogram range** `range / (0.5 * tile extent in mm)`, dimensionless. The sill is used **only to locate the range** - the first lag at which the variogram reaches `0.9 * sill`; the sill is not the denominator. Owner's chosen interpretation of "range-to-sill ratio", kept scale-free as the original cell intended | scale-free; the excluded `dom_wavelength_mm` / `spec_centroid_cpm` are absolute values |
| A4 | **directional contrast continuity** `omni - C(45 deg)`, where `C(d)` is the fraction of masked pixel pairs at the fixed ~1 mm lag in direction `d` whose grey difference exceeds the tile's own MAD, and `omni` is the **mean of C over 0, 45, 90 and 135 deg**. Diagonal directions use a **3 px per-axis** offset, the owner's chosen physically matched diagonal lag: at 4.5525 px/mm the axis-aligned pair distance is 5 px = 1.098 mm and the diagonal one is 3*sqrt(2) px = 0.932 mm. A 4 px offset would give 1.242 mm, so 3 px is nearer the 1.0 mm target than 4 px; the earlier claim that 3 px was the closest the integer grid allowed was wrong and is corrected here - the VALUE is unchanged by this correction | a two-point property at the coarse end where 12 of 24 soils are resolvable |

Original wording, preserved verbatim for the record:

- A2: "**autocorrelation e-folding length ratio** between the 0 deg and 90 deg directions of the
  residual `e`, in mm"
- A3: "**variogram range-to-sill ratio** of `g` (dimensionless)"
- A4: "**directional contrast continuity**: fraction of pixel pairs at fixed lag ~ 1 mm whose grey
  difference exceeds the tile's own MAD, omni minus 45 deg"

Two facts the owner was told before ruling, because they bear on interpretation and are not hidden
here: A3's normalisation makes `ppm` cancel exactly (measured 0.3984375000 at 2.276, 4.552516 and
9.105 px/mm), and A2's value does move with `ppm` through the illumination-base sigma (measured
-0.327 / -0.912 / +0.390 on a synthetic blob tile at 2.28 / 4.55 / 9.11). The owner's instruction on
the second point is explicit: the statistic is **not** to be altered because its value changes with
scale. With a 5 px axis lag (1.098 mm) the closest integer diagonal offset is 3 px per axis
(0.932 mm); 4 px would give 1.242 mm, so 3 px is the ruling and the residual mismatch is stated
rather than smoothed over.

**Frozen numeric parameters, named here once so "no parameter per statistic" is checkable.** Any
change to a value in this table after the run is a methodology change and must be ruled on again.

| constant | value | role |
|---|---|---|
| `A4_LAG_MM` | 1.0 | the fixed pair lag |
| `A3_REFERENCE_LAG` | 0.5 | A3's denominator is this times the tile extent in mm |
| `A3_SILL_FRACTION` | 0.9 | variogram crossing that defines the range |
| `A3_SILL_WINDOW` | 0.10 | trailing fraction of lags averaged to form the sill |
| `MIN_MASK_PIXELS` | 16 | below this the mask is degenerate |
| `MIN_PAIRS_PER_LAG` | 8 | below this a variogram lag is undefined |
| `MIN_PAIRS_PER_DIRECTION` | 8 | below this an A4 direction is undefined |
| `MIN_PROFILE_LINES` | 4 | below this an A2 direction is undefined |
| `A4_DIAG_PX` | 3 | per-axis offset of both diagonal pair sets, the owner's ruled physically matched diagonal lag (see AMENDMENT 2) |
| `EPS` | 1e-12 | floor inside every log and ratio |
| `ACF_MAX_LAG_FRACTION` | 0.5 | ACF is evaluated over this fraction of the profile length |

No fifth statistic, no scale sweep, no parameter per statistic. Lags are fixed to the tile geometry
before the run. If a statistic is undefined for a tile (degenerate mask, too few pairs, or too few
valid profile lines for a direction), **that statistic is NaN** - it is never answered by a fallback
value, a carried-forward lag, a maximum-lag range or a 0.0 exceedance fraction. The NaN rate is
reported, not hidden. A degenerate tile and an invalid call are distinguished by separate exception
types, so a systemic bug can never be recorded as tile degeneracy.

**Primary validation protocol - family-honest CV.** Leave-one-CV-family-out over the 16 families
(24 soils). Per fold: standardise on training soils only; rank-3 curve basis fitted on training soils
only; alpha chosen by the **existing frozen nested rule**, unchanged; monotone projection unchanged;
EMD the recorded trapezoid form. **G0 anchor: the pipeline without the new block must reproduce
43.0217308796477 to 0.00e+00 before any arm is scored** - an arm that cannot reproduce the anchor is
`INVALID-IMPLEMENTATION` and yields no conclusion in either direction. Paired differences use a
family-clustered bootstrap (resampling unit = CV family, the honest holdout unit), 4000 draws.

**Pre-registered nulls and controls, all four required to pass.**

1. **Random-feature placebo (P-RAND).** A1-A4 recomputed from phase-scrambled tiles: FFT magnitude
   preserved, phase randomised. This keeps every pixel-value marginal and the entire power spectrum
   while destroying higher-order spatial organisation. If the real block's gain is not exceeded by
   P-RAND's, the gain is not about spatial organisation.
2. **Column-shuffled placebo (P-SHUF).** The real A1-A4 block with each column independently
   permuted across images - exact per-column marginals, joint structure destroyed. This is M5's
   `CORAL_INDEP` lesson applied here: a *row* permutation is inert for marginals, so it must be a
   per-column shuffle, and the permutation vectors are written out so the control is reconstructable.
3. **Best-of-K selection null (P-SEL).** The whole 4-block selection is replayed on permuted soil->curve
   labels; the recorded gain is compared against that distribution's best-of-4, **not against zero**.
   This null is measured for K=4 and is not inherited from the map's best-of-15 figure of 15.29 EMD.
4. **Feature-count control (P-COLS).** Four extra columns that are linear combinations of the
   existing 12, so any "more columns help" effect is separated from "new information helps".

**Alias-separability gate (runs first, and can close the candidate at zero modelling cost).**
The pair set is defined by a deterministic rule over recorded artifacts, so it is reconstructible
and cannot be re-cut after the fact: take the **10 highest-error soils** of the shipped model from
`Model 2/Model 2 Experiment 3/cv_per_soil.csv` (the set Model 6-3 recorded as carrying 61.2% of
summed CV error), and for each, its **nearest neighbour in the existing 12-feature standardised
space**, with the neighbour drawn from a different CV family than either soil. That yields the alias
pairs. **Control pairs** are the same rule applied to the 14 lowest-error soils, one per soil,
matched by neighbour distance as closely as the 24-soil corpus allows. No new threshold, no
hand-picked pair.

For each pair, compute separation between the two soils in the A1-A4 space versus the existing
12-feature space. **Gate: the new block must increase median separation on alias pairs by a margin
that P-RAND and P-SHUF do not also achieve.** If it does not, M7-A closes with no experiment run, per
the standing rule that closing a candidate is better than inventing a transform.

Honesty note on this gate, because it is the subtle kind: the statistics themselves are label-free by
construction (fixed definitions, nothing fitted), so no fold logic is needed to keep them clean. The
*pair list* is chosen using recorded CV errors, which are labels. That is acceptable only because
A1-A4 are frozen before the gate runs and the gate cannot be used to pick among statistics - it can
only pass or fail a fixed block. If the gate is ever allowed to choose which statistics survive, it
becomes selection on scored data and the whole candidate inherits the best-of-K problem instead of
controlling it.

Note on provenance, stated rather than glossed: Model 6-3's diagnostic was computed read-only and
was **not** persisted as a pair-list artifact, so this gate re-derives the pair set from the two
recorded files above (`cv_per_soil.csv` plus the frozen soil-feature assembly in
`transfer_eval.py`). That re-derivation is checked, before any arm is scored, against the Model 6-3
figures that are properties of the registered rule itself: the top-10 share of summed CV error
(61.2%) and the worst-soil share (10.1%). A mismatch in either is `INVALID-IMPLEMENTATION`, not a
nuance.

**AMENDMENT A1 (2026-09-30, owner-approved wording before the edit).** The median alias curve gap of
65.7 EMD is REMOVED as a reproduction precondition for this gate. Reason, established forensically
and not by adjustment: 65.7 was printed by Model 6-3's own procedure, which groups the **worst 8 and
best 8** soils and takes each soil's Euclidean nearest neighbour over **all 23 other soils with no
CV-family constraint**. That is a different instrument from the registered rule above, which groups
**top 10 / remaining 14** and requires the neighbour to come from a **different CV family**. Under
Model 6-3's own rule all of its figures reproduce exactly (65.7135 and 28.0654 EMD, neighbour
distances 1.6460 and 1.5763, and the separability contrasts 1.9542 and 1.3697; reconstruction in
`scratch/reconstruct_m63_rule.py`, 21 checks, 0 failures, source recovered verbatim from the session
transcript into `scratch/_m63_commands_verbatim.txt`). Under the registered rule the median alias gap
is 56.0966, and it cannot be 65.7, because the two rules select different partners for the same
soils. Requiring the registered rule to reproduce 65.7 therefore asked one rule to match another
rule's output, which is internally inconsistent and can never be satisfied by any implementation.
This is a provenance mismatch, not evidence that the M7-A implementation is broken.

What the amendment does NOT do. The pair rule, its groups, its cross-family requirement, its
deterministic argmin, the gate's separation statistic, P-RAND/P-SHUF/P-SEL/P-COLS, the thresholds
G1-G7 and the verdict precedence are untouched. The historical worst-8/best-8 rule is NOT adopted,
same-family neighbours are NOT allowed, no pair is hand-picked, and no further neighbour definition
may be searched for in order to print 65.7. `data/alias_pairs.csv` is NOT regenerated or replaced.
65.7 EMD is retained as a recorded historical figure with its own provenance label, and
`check_alias_gate.py` must keep asserting both directions of the finding: that Model 6-3's rule still
yields 65.7, and that the registered rule does NOT. If either assertion ever breaks, the two
instruments are being confused and Task 6 must not run.

**Pass criteria (all must hold) -> SUPPORTED.**
- G1 gain: mean family-honest internal CV EMD reduction **>= 3.00 EMD** versus the anchor.
  Chosen deliberately: 3.00 is more than twice the 1.68 EMD ceiling that a selector-only fix could
  ever deliver, so a candidate introducing new inputs must beat the knob-fix ceiling or it is not
  worth the added variance.
- G2 CI: the paired family-clustered bootstrap CI of the gain excludes 0.
- G3 placebo: gain > P-RAND gain **and** > P-SHUF gain, by more than the bootstrap noise partner.
- G4 null: gain clears the best-of-4 permutation null P-SEL.
- G5 validity: G0 anchor reproduced to 0.00e+00; determinism across two runs; all cells finite;
  NaN rate reported and identical between arm and control.
- G6 inertness: with the new columns multiplied by 0, the arm must reproduce the anchor exactly -
  proving the plumbing is capable of moving the metric before its null is trusted.
- G7 honesty: no test-split statistic, no camera label, no ppm, no sample id enters any feature,
  selection, or fold assignment.

**Interpretation table, fixed now** (precedence is top-down, one deterministic rule, reused from
M5E1 section 6a rather than re-invented):

| outcome | verdict | action |
|---|---|---|
| G5 or G6 fails | INVALID-IMPLEMENTATION | no conclusion in either direction; fix code, rerun, gate unchanged |
| all clauses pass | SUPPORTED | submission permitted, one CSV |
| G1 passes, G3 or G4 fails | SELECTION-OR-CAPACITY-ARTEFACT | no submission; the gain is the block's own degrees of freedom, not organisation |
| G2 fails, G1 passes | NOT-RESOLVED | no submission; effect size without statistical support |
| G1 fails, G3 and G4 pass | SUPPORTED-BELOW-BAND | no submission; mechanism real under 3.00 EMD, recorded, not dressed as a win |
| G1 and G3 fail | REFUTED | M7-A closes at one experiment |

**Estimated headroom (an estimate, explicitly not a measurement).** The only large internal-scale
target on the board: if the 10 worst soils behaved like the other 14, internal CV mean would move
43.02 -> 28.62, a 14.40 EMD arithmetic ceiling. Realistic expectation for four added columns at n=24
is **2-6 EMD with material probability of exactly 0**, and the known variance wall (rank 3->5 bought
0.38-0.57 EMD despite a 3.87 EMD floor) argues the low end.

**Path to the external baseline 60.56167.** This is the only surviving shortlist candidate whose
mechanism is not a re-knobbing of the existing fit, so it is the only one with any credible route to
a resolvable external change. The public set scores 3 of 10 soils with a 95% band of [21.7, 69.9],
so **no** internal gain here is externally confirmable; the external band must be written down before
any score exists.

**Submission:** yes, if and only if verdict = SUPPORTED. The notebook under its final name emits the
CSV against the existing submission schema; otherwise no submission is manufactured. One submission.

**Expected failure mode.** The project's signature trap: an in-domain gain that reverses on transfer
(Model 2's untrained `fr` arm: best in-domain, worst transfer). Second mode: four extra columns at
n=24 spend the gain as variance, the same wall that ate the rank-5 floor. Third mode: A1-A4 are
camera-sensitive, and the camera term is only ~5.55 EMD to begin with.

---


**AMENDMENT 2 - owner-authorized, 2026-09-29, wording and disclosure only.** No definition, no control, no gate, no threshold and no protocol changed. Three things are recorded.

1. **A4's diagonal offset justification was corrected.** The value stays 3 px per-axis, as the owner ruled. The sentence claiming 3 px was the closest the integer grid allowed to the 5 px axis lag was arithmetically false - 4 px (1.242 mm) is closer to 1.098 mm than 3 px (0.932 mm) is, though 3 px is nearer the registered 1.0 mm target. The cell above now states both actual distances instead of a justification.
2. **A2's sensitivity to illumination scale is retained as a diagnostic, not repaired.** Measured on a synthetic tile: A2 = -0.327 / -0.912 / +0.390 at 2.28 / 4.55 / 9.11 px/mm. The owner's ruling is explicit: the feature definition is not to be altered because its value moves with scale, and the movement is reported at run time so a reader can see it.
3. **P-RAND's registered claim is measured, and it does not fully hold.** The placebo preserves the pixel multiset EXACTLY (verified with array_equal per channel on five tiles) and the DC/total power to within 0.0-6.0%, but the non-DC radial power spectrum is perturbed by relative L1 0.152 (noise) / 0.656 (fine) / 0.675 (coarse) / 0.747 (blobs); a row-constant synthetic tile is the one case where it is preserved. Both properties cannot hold at once: the re-ranking that restores the marginal is what moves the spectrum, and dropping the re-ranking would instead force the autocorrelation to match by Wiener-Khinchin, making A2/A3 unable to differ from the real block at all. The owner's instruction is to keep the controls unchanged, so P-RAND stands exactly as pre-registered and the gate is untouched. What changes is only the strength of the inference G3 supports: because P-RAND differs from the real block in arrangement AND spectrum, a gain that P-RAND does not match is evidence of specificity, but not the clean arrangement-only isolation the original wording implied. This limitation is a result to report, not a defect to fix, and it does not weaken any threshold.

**AMENDMENT 3 - owner-authorized, 2026-09-30, A2's word "undefined" made operational.** Ruling (a): A2 requires a **measurable decay in both directions** and is NaN when, for either direction, the autocorrelation never falls below 1/e within the evaluated lags, or the 1/e crossing falls below one full lag step (lag 0). Both tests are structural - they read the existing 1/e definition and the integer lag grid. **No new numeric energy or magnitude floor was introduced and none may be.** The plain-ratio form, the masked-line skipping, the energy-equals-zero guard and the warnings-as-errors policy are unchanged, as are A1, A3, A4, all controls, all gates and all thresholds.
Why it was needed: three synthetic tiles answered A2 with a number when no length was measurable - 1.2166e+13, 8.2194e-14 and 1.0000000000036 (the last while A3 and A4 on the same tile were already NaN). All three now return NaN. Cost on the real corpus, measured on 150 random training tiles: 0 NaN for A1-A4, so this is a correctness fix, not a feature-eviscerating one. Finite ranges on that sample: A1 p5..p95 -0.5078..-0.0758, A2 0.4672..2.8098, A3 0.0156..0.3324, A4 -0.0041..0.0231 - no order-of-magnitude outliers reach the standardiser.
