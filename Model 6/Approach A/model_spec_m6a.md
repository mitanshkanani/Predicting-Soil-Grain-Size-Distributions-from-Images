# Model 6 / Approach A — Attenuation correction of predicted curves

**Status: SPECIFICATION, awaiting owner review. No code, no plan, no run.**
Contract summary: `instructions.txt`. This file is the detailed version it argues from.

> **Headline, so the owner does not have to find it in section 8:** the pre-registration
> probes written up in section 8 came back **against** this approach. The result that motivated
> it was an oracle-fitted scalar inside a chance band, and the direction-asymmetry test — the one
> that distinguishes attenuation from metric geometry — fails. The recommended uses of this
> document are therefore either (a) register it as a cheap, null-anchored closure of attenuation,
> or (b) drop Approach A before implementation. It is not written as a prediction of success.

---

## 1. Hypothesis

**H6.** A ridge head fitted on 24 labelled soils systematically **under-predicts the magnitude**
of a soil's deviation from the training mean curve, and correcting that by a single multiplicative
expansion about the mean curve — with the expansion factor estimated **only** from held-out
training folds — reduces EMD on feature-displaced soils of the kind the Kaggle set contains.

Formally, with `ŷ` the frozen pipeline's predicted cumulative curve, `ȳ` the fitting set's mean
curve, and `k ≥ 1` a scalar:

```
ŷ' = project( ȳ + k · (ŷ − ȳ) )
```

`project` is the existing monotone CDF projection, unchanged, and applied **after** expansion. The
expansion therefore acts in curve space on the deviation from the centroid, not in rank-3 score
space, and any part of it that would leave `[0, 100]` is clipped exactly as the current pipeline
clips it. Section 4.3 records why this ordering choice matters and is frozen now.

**Null hypothesis:** `k = 1`, i.e. the pipeline as shipped by Models 1 and 2 E3.

## 2. Why this is a different thing from Models 1, 2 and 5

| | what it changed | where | outcome |
|---|---|---|---|
| M1 | which hand features feed the head | inputs | closed, 61.24 |
| M2 | which backbone produces the features | inputs | closed, 60.56 probe |
| M5 | the input **distribution**, by an affine map | inputs | refuted, −8.75 EMD |
| **M6-A** | **the output scale, by one scalar** | **outputs** | this spec |

M5's structural finding is what makes an output-side intervention the remaining option: any affine
map applied to the features is annihilated by the head's own `StandardScaler` (verified to 6.9e-15),
so input-side moment matching **cannot** work in this pipeline. That closes a class, not a
direction. Approach A operates on a quantity no previous model touched.

## 3. What is frozen

Inherited byte-for-byte from Model 2 E3 / Model 1 E3: preprocessing and `config_hash
010f44c36c74`; the 12 control features; tile→image median→soil median aggregation; `StandardScaler`;
rank-3 curve basis; closed-form ridge; the 16-value alpha grid `0.03 … 1e6`; monotone projection
with 100 forced at 200 mm; trapezoid EMD on `log10 d`; the CV family table; the parsimony rule
shape. Camera/ppm/EXIF/ICC/sample-id may define a **split** and may never transform a **prediction**.

The one addition is the expansion step and the `k` estimator.

## 4. Method

### 4.1 Displacement, measured label-free

For soil *s*, standardise the 12 soil-level features on the 24 labelled soils and set

```
d(s) = || z(s) − mean(z) ||₂
```

`d` uses **no labels and no camera identity**. It is the quantity used to rank soils for the
rulers in section 5. The Euclidean measure is used deliberately in preference to Mahalanobis:
with n=24 in 12 dimensions the sample covariance is near-singular and inflates outside distances
by a factor the training geometry cannot support — measured as 2.80× by Mahalanobis against 1.46×
here, and the two cannot both be trusted (section 7, fact F3).

### 4.2 The estimator for k

`k̂` is **not** fitted on scored data. On the inner folds of a fitting set, for each candidate `k`
in the frozen grid, compute mean inner-fold EMD; take the smallest `k` whose value is within 2% of
the inner minimum. This mirrors the project's ridge-alpha parsimony rule so a reviewer can audit
one against the other.

`k̂` is estimated per configuration, never globally tuned, and the same code path is used for
validation and for the production recipe.

### 4.3 Ordering choice, frozen before any number

Expansion is applied **before** projection, not after. The reason is substantive: after clipping,
a curve already pinned at 100 across its upper supports is invariant to further expansion, so
expanding post-projection silently does nothing exactly where the correction is supposed to bite.
Measured saturation: 40% of test soils already have a saturated cell versus 12% of training soils.
This ordering is fixed now so it cannot be flipped later to rescue a result.

## 5. Validation: three rulers plus a null

### N1 Feature-displacement holdout — primary

Rank the 24 soils by `d(s)`. For `h ∈ {6, 8, 10}`: score the `h` most-displaced, fit on the
remaining `24−h`, estimate `k̂` from inner folds drawn **only** from the fit set, scored the same
way. Repeat over several interior split definitions so a single lucky partition cannot carry the
result. This is the ruler that resembles deployment: the scored soils are genuinely outside the fit
set's feature support, which is what the Kaggle set is.

*Why this replaced the label-displacement split.* The probe that started this used extreme-D50
soils as "extrapolation". Measured, those soils are **not** displaced in feature space: mean `d`
of 3.35 against a corpus mean of 3.35. Extrapolating in label space therefore tests a condition the
test set does not exhibit, and it is retired from the primary role.

### N2 Family holdout — secondary, continuity

The existing 16-fold CV-family leave-group-out, unchanged, so results are quotable next to
Models 1, 2 and 5. Reported because it is the ruler the project has historically over-trusted; it
is interpolation and **cannot alone authorise a submission**.

### N3 Reverse-displacement holdout — falsification control

N1 inverted: fit on the most-displaced, score the least-displaced. Attenuation predicts expansion
helps in N1 and **hurts** in N3. Equal gains in both mean the effect is not displacement-specific,
and that outcome must be written up as evidence against Approach A whatever N1 shows. This clause
is a pass/fail gate condition (G4), not a diagnostic.

### N4 One-parameter null — the anchor, new to this project

Empirical distribution of the gain obtainable by tuning **any** single scalar on `h` held-out soils,
over ≥2000 random draws for `h ∈ {6, 8, 10}`. Every claimed gain is compared to this distribution,
not to zero. This instrument did not exist when Models 1–5 were gated, and it is the reason the
motivating result in section 8 dissolved.

### Uncertainty

Paired per-soil differences against `k = 1`; soil-clustered bootstrap, 4000 draws, seed 20260941;
resampling unit is the soil. Never an unpaired comparison of arm means.

## 6. Arms

| arm | definition | role |
|---|---|---|
| **A0** | `k = 1.000` | identity; the exact frozen pipeline |
| **A1** | `k = k̂`, nested | **the candidate** |
| **A2** | `k = argmin` on the scored fold | diagnostic ceiling only, never submittable, never gate input |
| **A3** | best alpha over the grid at `k = 1` | "it is just regularisation" control |
| **A4** | same magnitude, random rank-3 direction | direction placebo |

`k` grid: `1.00 1.05 1.10 1.15 1.20 1.30 1.40 1.50 1.60 1.80 2.00`.

A2 exists precisely because confusing it with A1 is the error this spec was written to make
impossible. Any report that cites A2 as support for H6 is invalid by construction.

## 7. Measured facts this spec is built on

All reproducible from disk; none is a Model 6 result.

- **F1 — the test set is displaced in feature space.** 35 of 90 colour+geometry cells outside the
  training envelope (1 of 70 for texture); 60% of test soils have at least one cell beyond ±3 SD.
- **F2 — but its predicted curves are contracted, not expanded.** Deviation from the training mean
  curve in the head's own rank-3 view: test mean 45.34 vs training 55.56, ratio **0.81**, while
  input displacement is 1.46× **larger**. The displacement lies in a low-gain direction.
- **F3 — two displacement measures disagree in magnitude.** Euclidean 1.46×, Mahalanobis 2.80×,
  from a 12×12 covariance estimated on 24 points. Only the Euclidean and the per-column envelope
  counts are treated as trustworthy.
- **F4 — saturation is more than 3× as common on test soils** (40% vs 12% with a clipped cell),
  which bounds how much expansion can do at all.
- **F5 — the external gap decomposition stands but does not imply displacement.** Zero-skill
  (train-mean curve) 85.39 → 102.37 = **+16.98**; model-specific residual +1.23 (M1 E3 nested) and
  −0.75 (M2 E1 D). See section 9 for why the first term is not evidence about the test soils.
- **F6 — error correlation between the two independently-built models is 0.745**, which also caps
  what any two-model averaging (Approach C) could buy.

## 8. Pre-registration probes — EXPLORATORY, not experiment evidence

Run 2026-09-29 while writing this spec, seed 20260939, in `Model 6/Approach A/scratch/` territory,
labelled here so the registered run never has to share a number with them.

**P-1 the motivating result was oracle-fitted.** Expanding predictions in a label-displacement
split moved 50.96 → 35.42 EMD. But that is `k` chosen with hindsight on the six soils being
scored. Against N4-style chance — tuning one scalar on six random soils gives median +3.82, p90
+11.71, p97.5 +22.61, max +40.18 EMD — a +15.5 gain is roughly p ≈ 0.03–0.09 on a *random* set,
and the set was selected for extremeness, which inflates it further. Not evidence.

**P-2 the direction test fails.** Feature-displacement holdout, tuning `k` with hindsight:
most-displaced holdouts gain +4.20 / +11.15 / +4.52 (h = 6/8/10); least-displaced holdouts gain
**+13.74 / +10.54 / +6.75**. The reverse case gains as much or more. Expansion is not correcting
displacement.

**P-3 nested `k̂` is worse, not better.** Applying the honest rule rather than oracle-`k`: on the
family-holdout ruler `k̂` averaged 0.26 with SD 0.34 and degraded error from 45.65 to **76.41**. On
the two displacement rulers it returned 1.32 and 1.31 — statistically the same number for opposite
conditions — with gains +6.61 and +7.43, both inside the P-1 chance band. A rule that returns the
same answer for a condition and its inverse is not measuring the condition.

**Conclusion drawn in advance of any registered run:** G4 (specificity) and G1 (gain beyond the
null) both fail on this evidence, and G5 (stability) fails on the family-holdout fold spread. The
expected registered verdict is **REFUTED-DIRECTION-NONSPECIFIC** or **REFUTED**, with no submission.

## 9. The zero-skill gap, and why it is not the premise

`instructions.txt` states this as the central falsifiable premise, so it is worth being exact about
what +16.98 licenses. The constant train-mean curve is computed *from* the 24 training soils, so it
is by construction nearer to them than to soils it was not fitted on. Its 85.39 → 102.37 worsening
is close to a tautology about centroids, and **it does not establish that the test soils are far
from the training range in D50.** What it does establish, and usefully: the test population is not
easier, and no model-specific transport defect above a few EMD is visible. Approach A's premise of
range displacement therefore rests on F1 and F2 — feature-space measures — and F2 actively points
the other way.

## 10. Gate

| clause | test | threshold and its anchor |
|---|---|---|
| **G0** identity | A0 reproduces M2 E3's 43.0217308796477 on N2 | ±0.05, the standing M5 tolerance, read from E3's artifact at run time |
| **G1** nested gain | A1 over A0 on N1 | paired gain > N4 **97.5th percentile** at the same h **and** soil-clustered CI excludes 0. No fixed EMD floor: the honest comparison is the null, not zero |
| **G2** not-regularisation | A1 gain > A3 gain | by ≥ 1.00 EMD — the project's smallest-signal unit (M5 G2's margin; well above the 0.674 EMD public difference declared a tie) |
| **G3** placebo | A1 over A4 on N1 | ≥ 1.00 EMD and CI excludes 0 |
| **G4** specificity | N1 gain strictly > N3 gain | hard pass/fail. Fails on section 8 evidence |
| **G5** stability | ≥ 75% of N1 folds select `k̂ ≥ 1.10` | a correction half the folds refuse to apply is not a correction |
| **G6** boundary | `k̂` strictly inside the grid | `k̂ = 1.00` is **no effect → REFUTED**, not a boundary hit; `k̂ = 2.00` is **INCONCLUSIVE**, grid truncated. Reported per ruler, never pooled (M5 5a rule) |
| **G7** label-blind | assertions of sections 3 and 4 | failure stops the run and is not a result |

```
submit = G0 ∧ G1 ∧ G2 ∧ G3 ∧ G4 ∧ G5 ∧ G6 ∧ G7
```

**Verdict precedence**, fixed now because rows would otherwise overlap:
1. ¬(G0 ∧ G7) → `INVALID-IMPLEMENTATION`
2. G6 upper-edge on any ruler → `INCONCLUSIVE`
3. G1∧G2∧G3∧G4∧G5 → `SUPPORTED`
4. G1∧G2∧G3∧¬G4 → `REFUTED-DIRECTION-NONSPECIFIC`
5. G1∧¬G2 → `EQUIVALENT-TO-REGULARISATION`
6. ¬G1∧G3∧G5 → `BELOW-BAND`
7. ¬G5 → `UNSTABLE-k`
8. otherwise → `REFUTED`

## 11. Pre-registered predictions

Tagged honestly: after section 8 these are **informed**, not blind. Recording them anyway is the
point — a prediction made with a probe in hand and then falsified is worth more than a coin flip.

- **P1 [informed]** H6 is refuted. No nested gain clears the N4 null on N1.
- **P2 [informed]** `k̂` will **not** differ systematically between the most- and least-displaced
  rulers (probe values 1.32 vs 1.31). A real attenuation correction must separate them.
- **P3 [informed]** A3 (alpha-only) recovers most of whatever A2 (oracle) achieves, because both
  are free scalars on tiny holdouts.
- **P4 [blind]** A0 reproduces 43.0217308796477 within 0.05. This one is genuinely blind: it tests
  the harness, not the world, and if it fails everything above is void.
- **P5 [informed]** the N3 gain will be ≥ the N1 gain. Confirming P5 is the cleanest single way to
  kill this approach.

## 12. Submission rule and the external band

Only on `SUPPORTED`, at most one file, `Submission_Model6_A.csv`, written by the notebook, never
renamed. Recipe in `instructions.txt`. The band `55.5 .. 66.0` and its five readings are fixed now,
before any score, and will be quoted unchanged. If the gate does not pass there is no submission and
therefore no band to test, and this section simply records that no score was sought.

## 13. Closure rule

`REFUTED`, `REFUTED-DIRECTION-NONSPECIFIC`, `EQUIVALENT-TO-REGULARISATION`, `BELOW-BAND` or
`UNSTABLE-k` closes Approach A permanently. A retry with a per-soil `k`, a per-direction `k`, a
gain matrix, or a re-tuned grid is **a different hypothesis** and requires its own brainstorming
pass and owner approval, not a continuation of this one. `INCONCLUSIVE` does not close: it licenses
exactly one decision, by the owner, about widening the grid, and nothing else.

## 14. Data sufficiency

Sufficient, and verified by reading during the probes: cached tile features
(`Model 1/Model 1 Experiment 3/.cache/tile_features_256_010f44c36c74.csv`), the three manifests,
`Model 2/Model 2 Experiment 3/cv_families.csv`, and the frozen head. **No new dataset, no Kaggle
dataset, no zip, no GPU, no torch.** CPU arithmetic on 24×12. If the run finds anything missing it
stops and names it rather than substituting.

## 15. Honest limitations

1. **n = 24 labelled soils.** Rulers that hold out 6 leave 18 to fit and 6 to judge. N4 exists
   because that is small enough for one free parameter to look like a discovery.
2. **A single global scalar is a crude model of attenuation**, and F2 says the test deviation lies
   in a low-gain direction, which no isotropic scalar can address. If Approach A fails, the failure
   is partly the parameterisation's, and that is a legitimate finding rather than a death of the
   idea — but re-parameterising is section 13's "different hypothesis".
3. **Projection clipping bounds the intervention** (F4).
4. **A pass would still rest on 3 public soils**, and a Kaggle score cannot distinguish
   "correction worked" from "correction happened to match this particular subset". The band in
   section 12 is written to make that ambiguity explicit rather than deniable.
5. **The direction test can be passed by chance.** With one binary clause (G4) at n=6–10 per
   ruler, a coin flip passes half the time. That is why G1 additionally has to clear the null, and
   why P2 and P5 exist as independent reads.

## 16. What I need from the owner

1. **Choose (a), (b) or (c)** from the box near the top of `instructions.txt`. The evidence in
   section 8 is the reason this is a real fork and not a formality.
2. If (a): confirm the k grid, the 2% parsimony band, and the `55.5 .. 66.0` external band are
   acceptable as frozen, since all three are judgement calls made here rather than measurements.
3. If (c): the direction-dependent-gain idea is genuinely different and I would take it back
   through brainstorming from the start rather than bolt it onto this spec.

No plan document and no code until those answers exist.
