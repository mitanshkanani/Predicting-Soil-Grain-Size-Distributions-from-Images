# Research map: Models 1, 2, 5, 6

Compiled 2026-09-29, after Model 5 E1 ran to a registered verdict and Model 6 approaches A, 1, 2
and 3 were closed at the feasibility stage.

**Scope of this file: consolidation and analysis only.** Nothing here was implemented. No experiment
was run, no threshold, arm, gate or verdict was changed, and no Model 7 is proposed. The only new
computation is read-only measurement on artifacts Models 1, 2 and 5 already validated
(`scratch/probe_feature_bound.py`, `scratch/probe_head_spread.py`), and those numbers are labelled
as probe output everywhere they appear, never as experiment evidence.

Correction records written earlier stay exactly as they are: `AGENT_BRIEF.md` section 11 CORRECTION 2,
`AGENT_BRIEF.md` section 12, `Model 5/instructions.txt` CORRECTION block, and
`Model 5/Model 5 Experiment 1/Experiment1.txt` sections 16-19. This file adds no new revision to any
historical result. Where a claim *I* made in the first draft of this file turns out to be wrong, the
withdrawal is stated in section 1 rather than quietly replaced.

---

## 0. How to read the numbers in this file

Two scales are used constantly and they are not interchangeable.

| | what it is | values |
|---|---|---|
| **EXTERNAL (Kaggle EMD)** | the host's score on real hidden soils. Public set is a fixed **3 of 10** soils | **60.56167** = current best, Model 2 E1. 61.23560 = Model 1 E3. 71.27346, 77.12257, 172.69929 = earlier Model 1 runs |
| **INTERNAL (CV EMD)** | leave-one-family-out / LOGO cross-validation on the 24 labelled training soils | 43.0217308796477 = recorded **nested** anchor. 41.15 = Model 1's recorded in-domain figure, whose alpha was chosen by CAM+RES and which is *not* the nested procedure (see section 4 A). 40.20-42.83 = head-sweep range in section 4 |

Rules used throughout:
- The internal 43.02 is **not** an external score and is never compared against 60.56167.
- Figures quoted from Model 1 records or from the Model 6 probes keep whatever alpha/aggregation
  convention those artifacts used. This map reports them as recorded and does not re-derive them, so
  comparisons *within* a quoted series are valid and comparisons *across* series need the provenance
  note in section 4 A before they carry any weight.
- The 0.674 EMD external gap between 60.56167 and 61.23560 was measured, and section 2 shows the
  public band is far wider than it, so it is a tie, not a result.
- Any headroom quoted in EMD below is **internal** headroom. Section 2 explains why internal
  headroom does not convert into a demonstrable external gain at this n.

---

## 1. Two measured bounds, corrected

**Bound 1 - can anything be squeezed out of the 12 features without a model?**
Test: leave-one-out, average the true curves of the k nearest soils in standardised 12-feature
space. No head, no labels beyond the neighbours' own curves.

Two independent leaks were found in the first draft's version of this test, so all four conventions
are reported rather than two. The candidate pool either allows or excludes a soil's **same-CV-family
siblings** (nested CV holds out a whole family, so a scored soil's siblings must not be available),
and the feature standardisation either uses **only the candidate pool** or the **all 24 rows
including the query**. Artifact: `scratch/_feature_bound.csv`.

| k | siblings allowed, scaler on all 24 (the withdrawn figure) | siblings excluded, scaler on all 24 | siblings allowed, honest scaler | **fully honest** |
|---|---|---|---|---|
| 1 | 44.23 | 50.66 | 52.84 | **59.27** |
| **2** | **41.95** | 47.66 | 41.95 | **47.66** |
| 3 | 45.73 | 50.27 | 45.73 | **50.27** |
| 4 | 44.96 | 51.12 | 44.88 | **51.04** |
| 5 | 48.26 | 51.55 | 48.26 | **51.55** |
| 6 | 47.12 | 54.25 | 46.51 | **55.32** |
| 8 | 50.99 | 57.21 | 50.83 | **61.59** |

The two leaks behave differently, and both need stating because they invalidate different numbers.
The **sibling** leak costs 5.71 EMD at k=2 and does not grow monotonically (its worst is +10.76 at
k=8 under the honest scaler). The **scaler** leak is what makes single-neighbour numbers untrustable:
it costs 8.61 EMD at k=1 (44.23 -> 52.84) and *nothing at all* at k=2, k=3 and k=5, where the two
scaler conventions print identical values, because averaging two or more neighbours is insensitive to
which rows set the standardisation. Total leak, honest versus the withdrawn column, is +15.04 at k=1,
+5.71 at k=2, +4.55 at k=3, +6.09 at k=4, +3.30 at k=5, +8.20 at k=6 and +10.60 at k=8.

References on the same honest footing: constant train-median curve 82.62, constant train-mean curve
85.39, trivial equal-mass baseline 100.31, shipped nested ridge **43.02**, and 41.15 for the same
ridge at the best fixed alpha on the frozen grid (probe, section 4 A).

CV families here are Model 2 E3's: 16 families over 24 soils, sizes 4 + five pairs + ten
singletons, so 14 of the 24 soils have at least one sibling. Nested CV holds out a whole family,
which is why "honest" is the only pool consistent with every number the project has ever reported.

> **WITHDRAWAL.** The first draft of this file stated that a two-neighbour lookup beats the shipped
> model by 1.07 EMD, and used that to cap all feature-only work at 1 to 1.5 EMD. That 41.95 came from
> a pool that allowed same-CV-family siblings, worth 5.71 EMD of the error; at k=2 the scaler
> convention contributed nothing. The separate 44.23 one-neighbour figure additionally carried the
> scaler leak, worth another 8.61 EMD. Both columns are quoted above rather than reconstructed here.
> The honest result runs the other way: **best honest local lookup (47.66) is 4.64 EMD WORSE than
> the shipped model (43.02)**. The parametric head is not failing to use information that a
> neighbour average would have found; it is substantially better than local lookup.
> The 1.07-EMD cap is withdrawn, and every downstream claim that used it is corrected below.
> Side finding worth keeping: sibling exclusion costs a neighbour estimator 3.29-10.76 EMD depending
> on k, which independently validates CV-family holdout as the correct unit; and a k=1 estimator moves
> 8.61 EMD on a scaler convention alone, which is why no single-point bound in this project should be
> quoted without stating its pool and its standardisation.

**Bound 2 - is the output basis the constraint?** Rank-k representation floor, i.e. the EMD left
after projecting the *true* curves onto the best rank-k subspace. Reported in both conventions
because the project has quoted both:

| rank | after the monotone/clip projection | bare linear projection |
|---|---|---|
| 2 | 9.75 | 10.98 |
| 3 (shipped) | 6.25 | **7.35** |
| 4 | 3.84 | 4.35 |
| 5 | 2.38 | 2.65 |
| 7 | 1.15 | 1.28 |

Convention note: the recorded "rank-3 representation ceiling 7.35" is the bare-projection column,
and the "6.25" is post-projection. They are the same floor, not two measurements. The older 6.85 /
1.92 pair in early notes used a third convention and should not be cited.

These floors are fitted on all 24 curves, so they are optimistic in principle. Even so: rank 3 -> 5
should free 3.87 EMD of floor on the projected convention and 4.70 on the bare one, and measured
end-to-end it freed **0.38-0.57 EMD** (41.15 -> 40.77 -> 40.58 at ranks 3/5/9, quoted as recorded
from the Model 1 and Model 2 artifacts, with their own alpha convention). The extra basis vectors are
estimated from 24 curves and spend the gain as variance. **The floor is real and unreachable at this
n.** That is a data limit, not a modelling failure.

---

## 2. The evaluation constraint, which no model has beaten

Per-soil honest CV errors of the shipped model: mean 43.02, **SD 22.75**, range 7.27 (G190) to
104.73 (H038). Resampling that distribution:

| quantity | value |
|---|---|
| 95% band for a **3-soil** public mean | **[21.7, 69.9]** |
| 95% band for a **10-soil** private mean | [32.6, 53.8] |
| Both observed external scores | 61.23560 and 60.56167, both **inside** the 3-soil band |
| External difference previously called a tie | 0.674 EMD |

Consequence: the public leaderboard cannot confirm a gain smaller than roughly 25 EMD, and no
mechanism left in this project has a plausible effect size anywhere near that. This is also why
`AGENT_BRIEF.md` section 11 CORRECTION 2 exists: the earlier "+16.98 measured population shift" was
a 24-mean minus 3-mean difference, not a measurement, and is withdrawn there.

---

## 3. Model-by-model ledger

### Model 1 - which hand-built features to feed a fixed head

| field | record |
|---|---|
| **Hypothesis** | E1: a wider feature set lowers EMD. E2: a camera-based ruler can rank configurations. E3: the absolute-frequency family costs accuracy and colour carries transfer error. E4: the colour representation chosen at preprocessing matters |
| **Exact intervention** | E1 16 features, alpha by LOGO-CV. E2 14 features, alpha by CAM+RES, two degenerate columns dropped. E3 2x2 factorial {colour?} x {absolute-frequency?} on a fixed 5-feature texture core, giving 12 features. E4 six colour representations on a fixed texture core, 5 features |
| **Validation protocol** | in-domain LOGO-CV over CV families; CAM+RES as a transfer ruler; soil-clustered paired bootstrap; submissions generated by the notebook |
| **Result** | in-domain / external: E1 35.98 / **172.69929**. E2 42.66 / 71.27346. E3 41.15 / **61.23560**. E4 39.46 / 77.12257 (declared a probe, not a gate pass). E3 factorial: frequency MAIN **+26.76** CI [+14.81, +39.07]; colour MAIN **-4.29** CI [-14.64, +5.27]; interaction -1.64, additive. E4: removing colour costs +5.11 CI [-6.23, +17.31], not resolvable |
| **Closure reason** | Four experiments reached an in-domain floor near 39-41 while the best external number stayed 61.24. The gains that did appear came from *removing* features and raising alpha - variance reduction, not better extraction. Further progress needed a change of kind, not more selection |
| **Estimated EMD headroom** | ~34 internal EMD against a rank-3 floor of 6.25-7.35, and that headroom is bounded by labels rather than by method (section 1, Bound 2) |
| **Limitation class** | **representation** (the features), plus **evaluation uncertainty**: E1 had the best in-domain number of its day and the worst external score in the project, which is what demoted CAM+RES to ordinal-only |

### Model 2 - can a learned representation beat the hand features

| field | record |
|---|---|
| **Hypothesis** | E1: frozen DINOv2 tile embeddings replace the 12 features. E2: if it fails, the patch was too coarse - magnify. E3: if replacement fails, the two are complementary - concatenate |
| **Exact intervention** | E1 DINOv2 ViT-S/14 vs hand features vs a random-weights twin, same head. E2 centre crops at 256/128/64/32 px. E3 12 + 384 = 396 columns, with `shuf` (row-permuted D block) and `fr` (random-init block) as controls |
| **Validation protocol** | nested in-domain LOGO-CV under the frozen alpha rule; CAM+RES report-only; byte-exact reproduction of the M1 anchor 43.0217308796477 |
| **Result** | E1 M1 43.02 / D 44.33 / R 39.78; D submitted, external **60.56167**, a 0.674 EMD tie against 61.24. E2: D lost to its own random-init twin at all four crops, no trend, and only 2 of 24 soils became newly resolvable. E3: `fuse` 43.43 vs 43.02, gain **-0.41** CI [-6.69, +7.46]; `shuf` 52.93 (9.90 worse); `fr` 39.26 - best in-domain, **worst transfer at 83.94** |
| **Closure reason** | All three hypotheses refuted. `fr` is the project's cleanest demonstration of the capacity trap: an untrained random projection wins in-domain CV and collapses on transfer, so a good in-domain number is not evidence of anything |
| **Estimated EMD headroom** | none identified. D's greater invariance (absolute shift 0.62 vs M1's 1.21) did not convert into accuracy |
| **Limitation class** | **representation** for the hand features; **data** for the learned ones - 384 columns at n=24 is the wrong regime. The `shuf`/`fr` controls prove the instrument was healthy while the null was real |

### Model 5 - can the camera shift be removed without labels

| field | record |
|---|---|
| **Hypothesis** | H5: aligning the fitting camera's image-level features onto the 35 unlabeled test images with closed-form CORAL lowers EMD on a camera that was neither fitted on nor the alignment target |
| **Exact intervention** | `tiles -> image median -> ALIGN -> soil median -> frozen head`, 7 arms, with `CORAL_INDEP` (independent per-column shuffle) as a same-marginal placebo, `MEAN` and `CORAL_SELF` as structural arms, and tile/soil estimation units as sensitivity |
| **Validation protocol** | two camera directions, fold-honest (each fold drops a CV family from the fitting camera), 45 rows over 24 soil clusters, soil-clustered bootstrap, G0 anchor to 43.0217308796477, gates G0-G7 and a per-direction alpha-boundary rule fixed before the run |
| **Result** | Delta_CORAL pooled **-8.748** CI [-12.776, -3.967]; direction A **-18.464**, direction B **+0.546**. Placebo worse still (-13.446). G1-G4 fail, G5-G7 pass, no alpha clipping |
| **Closure reason** | `DIRECTION-INCONSISTENT`, and REFUTED on the contract's own terms (missed G1 with no boundary caveat). Model 5 closed at one experiment. Two implementation defects were found and fixed *before* any gate ran and are disclosed in `Experiment1.txt` section 16 |
| **Estimated EMD headroom** | camera transfer measured at **+5.55 EMD**, the smallest of the three figures previously conflated, and unreachable by affine re-mapping |
| **Limitation class** | **representation**. A constant feature shift commutes with the soil median and is annihilated by the head's `StandardScaler`, so MEAN and CORAL_SELF are zeros to 6.9e-15 and 3.7e-15. First-moment adaptation *cannot* act in this pipeline. Any future adaptation must be non-affine or must act before standardisation |

### Model 6 - four candidates, all closed before implementation

| | **Hypothesis** | **Exact intervention** | **Validation protocol** | **Result** | **Closure reason** | **Estimated EMD headroom** | **Limitation class** |
|---|---|---|---|---|---|---|---|
| **6-A** | predictions are attenuated toward the training mean; expand by `k` about it | isotropic post-hoc expansion, `k` chosen on inner extrapolation folds | `k` from inner extrapolation folds only; one-parameter permutation null; reverse-displacement control | oracle-`k` looked large (50.96 -> 35.42) but sits inside the null for tuning one scalar on 6 points (median +3.82, p90 +11.71); honest nested `k` averaged 0.26 and made error **worse** (45.65 -> 76.41); the reverse-displacement asymmetry never appeared | fails its own one-parameter null and its direction test | 0 | **evaluation uncertainty** (an oracle-selected scalar looked like a mechanism) |
| **6-2** | out-of-support test features cause the error; contain them | winsorise/clip scored features to the fitting-set envelope | sign test on the displaced soils rather than the in-domain mean; clip-rate census as the mechanism check | mechanism fires (22.5% of test cells clip; 8.7-21.9% of held-out cells clip in-domain) but moves the wrong way: 41.15 -> 41.91 -> 43.88, and 32.81 -> 38.00 -> 42.46 on displaced soils | refuted by a sign test | 0 | **representation** - out-of-support is not where the error lives |
| **6-1** | two decorrelated models average better than either | fixed 50/50 curve ensemble of the two validated submissions, no tuned weight | no training at all; per-soil error correlation plus a Jensen bound on the blend | per-soil error correlation **r = 0.745**; Jensen bounds the blend by the average of the parts (60.899 external), which is worse than the incumbent 60.56167; an earlier spike's +1.45 CI [-6.69, +3.19] was about 0.9 net of its noise partner | guarantee too weak and nothing uncorrelated to pair with | ~0 external | **data** - only two independent submission-grade models exist |
| **6-3** | the error tail is fixable with better features | strictly read-only forensic on per-soil errors | admitted only after an exact reproduction of 43.0217308796477; stage attribution across aggregation / soil / head rank / alpha | see the findings block below | no surviving systematic mechanism | 0 for a corrective transform | **data** (one pair is irreducible) plus **representation** (the rest) |

None of the four was registered as an experiment, none produced a submission, and none reused
another candidate's null: the shared word "probe" hides four different instruments, described in the
protocol column and in the detail below.

**Model 6-3 findings, restated with the corrected bound.** The top 10 of 24 soils carry **61.2%** of
summed CV error; the single worst soil carries 10.1%. The rank-3 floor explains only 3-20% of the
worst soils' error versus 22-66% of the best, so it is not basis capacity. Worst soils are *less*
feature-displaced than best (median neighbour distance 1.65 vs 1.58), so it is not extrapolation.
Within-soil image scatter is *lower* for the worst soils (3.00 vs 3.73), so it is not aggregation
noise. The worst soils each have a near feature neighbour whose curve is 65.7 EMD away (best soils
28.1) - **aliasing**. But the alias pairs are *more* separable on per-image features than pairs the
model already handles (median max d 1.95 vs 1.37; 3-6 of 12 features at |t| > 3), so the information
is present and unused. Corrected consequence: since honest local lookup is 4.64 EMD *worse* than the
model (section 1), "present and unused" does **not** mean a local or neighbour-based fix is
available; it means the current *functional form* fails to use what is there, which is a different
and harder claim.

**One genuinely irreducible case: H038 vs G190.** Feature distance 1.32, curves **125.4 EMD** apart,
**0 of 12** features separate them at |t| > 3 (max d 1.08). The images are statistically the same
soil. H038 is the project's worst soil (104.73) and G190 among its best (7.27); the model can be
right about exactly one of them. That pair is 10.1% of all CV error and was destroyed at pixel level
before any model ran. Third independent confirmation of the delivered-resolution floor
(0.063 mm is about 0.3 px at 4.55 px/mm), alongside the cancelled granulometry and the bounded
magnification sweep.

**Shared structure, honestly noted.** Per-soil error ranks agree across model families (Spearman vs
D +0.594, vs `fuse` +0.668, vs `fr` +0.570) while the `shuf` noise control gives +0.119, so the
ranking carries real signal. Cross-camera agreement +0.481 with 5 of 8 worst soils shared. Some
confusion is soil-intrinsic; some is feature-set-specific (F827 is 20th-worst under M1 and best
under `fr`).

---

## 4. Remaining opportunities, in four buckets

Each item is tagged **[measured]** (a number in a validated artifact supports it) or
**[unverified]** (a mechanism that is consistent with the evidence but has not been tested). No item
below is a Model 7 proposal.

### A. Modifications to the existing pipeline (same features, same head family)

The probe in `scratch/probe_head_spread.py` is the first honest look at this whole class, because
earlier comparisons were made against the leaky Bound 1. Protocol: leave-one-CV-family-out over 24
soils, every head mapping the 12 standardised features onto the same rank-3 curve basis with the
same monotone projection, **no hyperparameter tuned anywhere** (ridge uses the frozen nested alpha
rule, everything else runs at library defaults), and the identical sweep replayed on 12 random
label permutations as a null. All figures internal CV EMD.

| head (family-honest LOFO CV) | internal EMD | same head on permuted labels (median of 12 nulls) |
|---|---|---|
| BayesianRidge | **40.20** | 89.41 |
| KernelRidge (default rbf) | 41.28 | 82.30 |
| shipped ridge, strictly nested alpha | 42.83 | 89.91 |
| ExtraTrees | 43.96 | 88.09 |
| PLS, 2 comps | 46.42 | 91.86 |
| PLS, 3 comps | 47.04 | 91.76 |
| k-NN k=2, inverse-distance weighted | 47.46 | 96.06 |
| k-NN k=2 | 47.58 | 94.95 |
| GradientBoosting | 49.98 | 92.10 |
| k-NN k=4, weighted | 50.17 | 90.52 |
| k-NN k=3 | 50.24 | 93.37 |
| RandomForest | 50.30 | 86.45 |
| k-NN k=4 | 51.12 | 89.85 |
| 1-NN | 57.70 | 104.87 |
| SVR (defaults) | 113.93 | 98.35 |
| recorded shipped nested anchor | 43.02 | - |

Two things the null column buys. It shows the instrument **can** fail (every head lands at 82-105
under permuted labels), and it shows the sweep is not uniformly credited: SVR at its library defaults
scores 113.93 on real labels, *worse than its own null*, which is a head that is simply badly posed
for unscaled curve coefficients rather than a finding.

- **[measured] Best fixed alpha on the frozen grid is 3.0, scoring 41.15.** The whole curve under
  this family-honest protocol (probe artifact `scratch/_alpha_curve.csv`), which is what the nested
  rule competes against:

  | alpha | 0.03 | 0.1 | 0.3 | 1 | **3** | 10 | 30 | 100 | 300 | 1e3 | 3e3 | 1e4 | 3e4 | 1e5 | 3e5 | 1e6 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | LOFO EMD | 53.80 | 48.35 | 43.92 | 41.28 | **41.15** | 45.65 | 51.70 | 64.87 | 78.19 | 87.41 | 91.00 | 92.40 | 92.82 | 92.96 | 93.00 | 93.02 |

  The strictly nested rule in the same probe scores 42.83, and the recorded nested anchor is
  43.02. **The cost of choosing alpha honestly at n=24 is therefore 1.68 internal EMD** (42.83 -
  41.15). This is the quantitative form of the alpha/regularisation diagnostic raised after M5E1:
  it is real, and it is the largest single *identified* loss inside the pipeline. The curve is
  strongly asymmetric: at the two ends of the grid, under-regularising costs 53.80 and
  over-regularising costs 93.02, so a mis-selected alpha in the large direction is far more damaging
  than one in the small direction. From 3e3 upward the values are flat because the head has collapsed
  to essentially one prediction for every soil (that fold's mean curve), which is why it sits near,
  but not on, the constant-curve references of section 1.
  Provenance discipline, because this is where a number nearly got miscoded twice: `43.02` is the
  recorded **nested** in-domain LOGO-CV of the 12-feature control
  (`Model 2/Model 2 Experiment 1/instructions.txt` lines 181, 260). The figure `41.15` is Model 1's
  recorded in-domain number produced with **alpha chosen by CAM+RES**, and `Model 2/Model 2
  Experiment 1/instructions.txt` line 18 explicitly warns that this is not the nested procedure. The
  oracle *alpha value* is recorded as 3 (`Model 2/Model 2 Experiment 3/Experiment3.txt` line 42) but
  the oracle *score* was never recorded, so 41.15 as an oracle-alpha figure comes from this probe
  only. That the probe's best fixed alpha (3.0) equals the recorded oracle alpha, and that the probe
  score lands exactly on Model 1's CAM+RES-alpha number, is unexplained and is **not** treated as a
  reproduction of it.
- **[measured] Local lookup is worse than the model** (47.46-51.12 in the sweep, 47.66 in the
  independent curve-space probe). Bucket-D-by-neighbourhood is closed.
- **[unverified] An adaptively-fitted penalty beats a grid-selected one**: BayesianRidge is 2.63
  EMD better than the nested ridge and 0.95 better than even the best *fixed* alpha on the frozen
  grid (41.15, probe). Family-clustered paired bootstrap gives **+2.63 CI95 [-0.62, +5.28], P(no
  gain) 0.053** - the CI includes zero. And the null says picking the winner of 15 heads on noise
  buys a median apparent gain of **15.29 EMD** (p90 18.29, max 26.72). So a 2.63 EMD best-of-15 edge
  is roughly six times smaller than what selection alone manufactures. **This is a candidate
  hypothesis with a measured effect size, not a demonstrated improvement.** Any future test must
  pre-register *one* head.
- **[measured] Everything else in this bucket is already null or negative**: L1 fitting 41.22 vs
  41.15; bagging 41.15 -> 43.33 / 42.64; rank 3 -> 9 buys 0.57; removing the entire fixed bias vector
  changes error by -1% while estimating it on 12 soils to score 12 loses 3.17 EMD; richer tile
  aggregation 42.13 / 42.22 vs 41.15; feature selection was Model 1's whole subject.
- **Cap: unknown, and no longer upper-bounded by Bound 1.** The honest statement is that the largest
  identified internal loss in this bucket is the 1.68 EMD alpha-selection cost, and the largest
  observed candidate gain is 2.63 EMD with a CI that crosses zero. Neither is demonstrable on the
  public set.

### B. Methods that introduce genuinely new information

1. **Spatial arrangement of texture.** **[unverified mechanism, measured premise.]** The pipeline
   reduces each tile to a 12-vector and each soil to a median, so tile *position* is discarded;
   that loss is a property of the code, not a hypothesis. Grain-size distributions have spatial
   organisation, and a two-point statistic (variogram, autocorrelation anisotropy, blob-size
   distribution of the visible component) is information the 12 features provably do not contain.
   Unlike granulometry it need not resolve sub-pixel grains: at 4.55 px/mm, organisation is
   measurable at the coarse end, where the magnification sweep already measured that 12 of 24 soils
   *are* resolvable. Not covered by any closed hypothesis: Model 2 tested *learned* embeddings
   (which are position-free by construction) and Model 5 tested *affine re-mapping*; neither touched
   spatial statistics. Cost: requires re-reading tiles, so it is the one B item that is not free.
2. **The 35 unlabeled test images, used non-affinely.** **[measured limit, unverified remedy.]**
   Model 5 measured that affine moment matching is inert after standardisation (zeros to 6.9e-15)
   and that CORAL is actively harmful (-8.75). Transductive *geometry* - defining the feature
   subspace or the distance metric from train+test features at fixed rank, before standardisation -
   is a different object that M5 did not test. Supporting evidence: the alias diagnosis says the
   current metric places a 2.4 mm soil next to a 0.06 mm one. Opposing evidence: Bound 1 now shows
   metric-driven local methods score badly here, and Model 5's closure rule fired, so the prior on
   "camera-derived re-mapping" is poor.
3. **More labelled soils.** **[measured.]** Bound 2 measures 4.70 bare / 3.87 projected EMD of basis
   headroom between rank 3 and rank 5 that this dataset cannot pay for. Not available inside the
   competition.

### C. Methods that require additional labelled soils

**[measured, blocked.]** Everything in B3 plus: CV at a usable n (per-soil SD 22.75 is what makes the
public band 48 EMD wide), per-band specialists, learning an anisotropic metric (the alias diagnosis
says the metric is wrong; fitting a metric needs labelled pairs), and harvesting the rank-4/5 floor.
This bucket is why the honest read is that the ceiling of this dataset is close to where the project
already stands.

### D. Information present in the 12 features but currently unused

- **[measured]** The alias analysis of M6-3 established that the worst soils' confusion pairs are
  separable on per-image features (3-6 of 12 features at |t| > 3), so some information is unused.
- **[withdrawn]** Its *size*. The previous estimate - about 1 EMD, from 2-NN 41.95 against the model
  at 43.02 - was leaky. Corrected: honest local lookup lands at 47.66, so **there is no valid upper
  bound on unused feature information at n=24**, and equally no valid lower bound. What replaced it
  is weaker but real: the head-sweep in bucket A shows the *best* alternative functional form beats
  the shipped one by 2.63 EMD with a CI crossing zero, which is the current honest measurement of
  this bucket's size.
- **[measured]** Aggregation depth/order was tested and is negative (42.13, 42.22 vs 41.15).
- **Conclusion:** bucket D is neither empty nor demonstrably profitable. It is the bucket where the
  measurement instrument, not the idea, is the binding constraint.

---

## 5. Does Model 7 need a genuinely different information source?

**Not settled, and the reason has changed.**

The first draft of this section answered "yes" on the strength of Bound 1's 1-1.5 EMD cap. That cap
is withdrawn, so the argument collapses: **there is no longer any measured ceiling on what the
existing 12 features can give.** The strongest position the evidence supports is:

- **[measured]** Local, neighbour-based, ensemble, post-hoc-scaling, containment, basis-rank and
  affine-alignment routes are all closed - six families of transform, each with a number.
- **[measured]** The largest identified *internal* loss inside the current pipeline is the 1.68 EMD
  paid to choose alpha honestly, and the model's own error is concentrated (61.2% in the top 10
  soils) rather than diffuse.
- **[unverified]** One head family (adaptive penalty / kernel) beat the shipped head by 2.63 EMD
  under a fold-honest protocol with no tuning. Effect size is plausible; statistical support is not
  there; selection bias of 15.29 EMD dwarfs it.
- **[unverified]** Spatial texture organisation is information the features demonstrably do not
  contain, and no closed hypothesis covers it.

So the choice for Model 7 is between the one **cheap, single-parameter, in-pipeline candidate**
bucket A now has (an adaptive penalty instead of a grid-selected one: measured effect 2.63 EMD,
ceiling unknown) and the **new-information route** B1, whose ceiling is unmeasured and whose build
cost is real. Both are defensible. What is not defensible is deciding between them on in-domain CV
alone: Model 2's `fr` arm posted the *best* in-domain number of any Model 2 arm (39.26) and the worst
transfer number (83.94), a 44.7 EMD gap.

---

## 6. Decision framework for selecting Model 7

No candidate may be implemented until it can answer all five rows. Partial answers are how M6-A and
M6-2 produced impressive numbers that reversed under their own controls.

1. **A falsifiable mechanism.** A sentence naming the variable that is changed, the quantity it
   moves, and the observation that would prove it wrong. "Better features" is not a mechanism;
   "tile positions carry a two-point statistic that the 12 median features discard" is.
2. **A pre-registered null, computed before the run, that the arm must clear.** Sized by the thing
   being selected: one scalar -> the one-parameter permutation null (M6-A's +3.82 / p90 +11.71);
   a head or subset chosen among N -> the best-of-N null (this file measures it at median **15.29**
   EMD for N=15). A gain smaller than its own null is a non-result and must be recorded as such.
3. **A validation protocol that matches deployment.** Fold-honest at the CV-family unit; the
   extrapolation direction primary with the reverse direction as a falsification control; a
   structural-inertness check confirming the arm *can* move the metric at all (MEAN and CORAL_SELF
   were zeros to 6.9e-15); the anchor 43.0217308796477 reproduced exactly; exit-code-level
   verification, not a tail of printed lines.
4. **A defined success criterion, frozen before the run.** Threshold, confidence interval, direction
   consistency, and the boundary rule for when an optimum sits at a grid edge - all written down
   while the outcome is unknown, with an explicit interpretation table and a single deterministic
   precedence rule.
5. **A path to a Kaggle submission.** If the gate passes, the notebook under its final name emits
   the CSV against the existing submission schema; if it fails, no submission is manufactured. The
   external band must be written down before the score exists, one submission only, and the score
   is never used to choose a model, a scalar or a threshold. Internal CV values and external EMD are
   reported on separate lines and never compared to each other.

**Baseline for any future external claim: 60.56167 Kaggle EMD (Model 2 Experiment 1).**
The 3-soil public band around it is [21.7, 69.9], so nothing under roughly 25 EMD of true improvement
is confirmable externally. That cuts both ways, and it is the reason not to let it drive the choice:
a real 2.6 EMD in-pipeline gain would be invisible on the public set, and an expensive
new-information route cannot claim credit externally either unless its effect is enormous. Select on
the strength of the mechanism and the honesty of the test, not on which one looks like a leaderboard
move.

**Selection sequence, so a candidate cannot skip a step:**
1. State the mechanism and its null in writing, before any measurement for it exists.
2. Run the cheapest label-free falsification on artifacts already validated. Four Model 6 candidates
   died here for a total cost of minutes, which is the whole argument for this sequence.
3. Only if it survives, register the experiment: folder, instructions contract, spec, plan, owner
   approval at each step.
4. Gate on a local rule, emit the submission only on a clean pass, and record the verdict table
   including the boundary and precedence rules before the run.

This map deliberately stops before step 1 for Model 7 and proposes no candidate.

---

## 7. Settled, and not to be re-litigated

**[measured]** Camera shift is worth about 5.55 EMD and affine alignment cannot recover it.
Learned tile representations do not beat the 12 features, alone or fused. Magnification cannot
resolve the fines. Post-hoc isotropic output scaling, feature containment, ensembling the two
existing submissions, richer aggregation, particle counting, CAM+RES as a ranker, and L1 or bagged
variants of the current head are measured-null or measured-negative. The public leaderboard cannot
confirm any gain below about 25 EMD. Honest local lookup is worse than the shipped model.

**[withdrawn, listed so it does not return]** the "+16.98 measured population shift" (see
`AGENT_BRIEF.md` CORRECTION 2), and this file's first-draft "1-1.5 EMD cap on all feature-only work"
(see section 1). Both were measurement artifacts, not results.

## 8. Provenance

- Experiment evidence: `Model 1/Model 1 Experiment {1,2,3,4}/`, `Model 2/Model 2 Experiment {1,2,3}/`,
  `Model 5/Model 5 Experiment 1/` (`Experiment1.txt`, `gate.json`, `per_soil_transfer.csv`,
  `predictions.csv`), `Model 6/Approach A/` (spec and instructions, kept as a closed record).
- Correction records: `AGENT_BRIEF.md` sections 11, 12, 13, 18; `Model 5/instructions.txt`;
  `Model 5/Model 5 Experiment 1/Experiment1.txt` sections 16-19.
- Read-only probes (this file only, not experiment evidence): `scratch/probe_feature_bound.py`,
  `scratch/probe_head_spread.py`, `scratch/_head_spread.csv`.
- External scores from the competition host; internal CV values from nested CV in the artifacts above.
