# Experiment 3 plan — Model 1

**Status: DRAFT FOR REVIEW. Nothing below is implemented.**

E3 is the first *ordinary* Model 1 experiment: one variable (the feature set), judged by
the ruler E2 validated. It is also the first genuinely out-of-sample test of that ruler.

---

## 0. HONESTY NOTE — read before the predictions

I ran reconnaissance before writing this plan (`scratch/cal_exp3.py`, read-only, nothing
outside `scratch/` was written). It already answers most of E3's internal question. So:

- **P1–P3 below are confirmations, not blind predictions.** They will very likely pass,
  and passing them is worth little.
- **P4–P6 are genuinely blind.** They test something recon cannot: whether the ruler
  extrapolates to a *different kind* of change. E2 calibrated it on a regularisation and
  column-defect change; E3 changes the feature families. Those are different axes and the
  ruler may not transfer.

I am stating this rather than presenting recon's answer as a prediction that came true.
The distinction matters: it is the difference between a test and a demonstration.

## 1. What Experiment 3 tests

**One question:** which feature families survive the move to unseen capture devices, and
does removing the ones that do not improve the external score?

Single variable: **the feature set.** The ruler (CAM+RES), the rank-3 curve basis, the
ridge, the monotone projection, the two-stage tile→image→soil median aggregation, the 16
CV families and the EMD implementation are all frozen at their E2 values.

## 2. What the reconnaissance found

### 2.1 A 2×2 factorial over {colour, frequency} on a fixed texture core

Texture core (5): `e4, e8, e16, lum_sd, grad_mean`.
Colour (7): `R, G, B, sat, lum_p10, lum_p50, lum_p90`.
Frequency (2): `spec_centroid_cpm, dom_wavelength_mm`.

| cell | n | LOGO-CV | CAM | RES | **CAM+RES** | best alpha |
|---|---|---|---|---|---|---|
| CORE5 (neither) | 5 | 42.47 | 47.39 | **49.05** | **56.30** | 10 |
| CORE+frequency | 7 | 44.05 | **43.93** | 77.23 | 76.24 | 30 |
| CORE+colour | 12 | **41.15** | 57.39 | 55.30 | **54.12** | 10 |
| CORE+both (= E2) | 14 | 42.66 | 45.82 | 83.32 | 75.73 | 30 |

The 14-feature row reproduces E2's stored `alpha_sweep.csv` exactly (42.66 / 45.82 /
83.32 / 75.73), which confirms the recon is on the same footing as the experiment.

### 2.2 Paired simple effects, CAM+RES ruler, 21 shared held-out soils

| contrast | mean paired Δ | 95% CI | verdict |
|---|---|---|---|
| frequency, no colour | **+19.94** | [+0.51, +38.93] | significant |
| frequency, with colour | **+21.60** | [+8.66, +35.64] | significant |
| colour, no frequency | −2.18 | [−22.53, +18.02] | not distinguishable from 0 |
| colour, with frequency | −0.52 | [−14.63, +13.73] | not distinguishable from 0 |

Under the RES ruler alone the frequency effect grows to **+24.94 / +28.02**, both
significant. Under the CAM ruler alone it is inconsistent (+7.52 significant, −6.40 ns).

### 2.3 What this means, and how it contradicts my prior

**My stated E2 hypothesis was wrong.** I predicted colour was the lever, because blue
shifts 2.63 SD between train and test and E1 flagged `B` as 96% camera-explained. On this
data the colour main effect is indistinguishable from zero under CAM+RES.

The actual lever is the **two absolute-frequency features**. They cost ~20 EMD under
CAM+RES and ~25–28 under pure sharpness change, consistently across both colour levels.
That is mechanistically obvious in hindsight: `spec_centroid_cpm` and `dom_wavelength_mm`
measure *absolute spatial frequency*, which is exactly what a low-pass filter destroys.
The test images came through 3.1× and 4.3× anti-aliased downsamples; the training images
came through ~1.0×. Any feature whose value is a frequency reading is reading the
resampling history, not the soil.

The two rulers separate two distinct nuisances cleanly, and this is the explanatory
result of the cycle:

| family | fails under | survives |
|---|---|---|
| colour | camera change (COL7 CAM = 85.47 vs CORE5 47.39) | sharpness change |
| absolute frequency | sharpness change (RES 77 vs 49) | camera change |
| DoG bands, gradient, luminance spread | neither | both |

E2's P3 already established that blur does **not** explain the observed mean band shift,
so this is not a contradiction: the frequency features are *hypersensitive* to a small
mean shift. A feature can matter more than its average movement suggests.

### 2.4 An unresolved tension in the selection

`CAM` prefers CORE5 (47.39) over CORE+colour (57.39). `CAM+RES` prefers CORE+colour
(54.12) over CORE5 (56.30). The two are within noise of each other on both rulers.
E2's rule says CAM+RES wins on deployment fidelity, which selects CORE+colour — but a
minimax criterion (minimise the worst ruler) selects CORE5. This is a real decision
point, handled explicitly in §5 rather than quietly.

## 3. Design

**Factorial, 2 × 2, orthogonal:** colour present {no, yes} × frequency present {no, yes},
on the fixed 5-feature texture core. Four cells. Each neighbour differs by exactly one
family, so every simple effect is a single-variable comparison.

**Plus two anchor cells** carried forward for continuity: E2's 14-feature configuration
(identical to CORE+both, so this is a consistency check, not a fifth arm) and the
no-image baseline measured under each ruler.

**Alpha swept, never hand-picked:** the E2 grid, 0.03 … 1000.

**Not in E3** (each is its own experiment, to keep attribution clean):
- new derived features such as band *ratios* that might be blur-robust → E5
- training on resolution-matched views → E4
- aggregation statistics, regressor kind → later

## 4. Statistical protocol — the part that must be done better than recon

Recon used one evaluation condition (Motorola→Samsung, res14) and got CIs of ±20 EMD.
E3 increases power in three ways, all pre-specified:

1. **Four evaluation conditions** instead of one: {Motorola→Samsung, Samsung→Motorola}
   × {res14, res16}. That is 84 paired observations rather than 21. They remain
   correlated within a soil, so the bootstrap **resamples soils, not observations** —
   otherwise the CIs would be falsely tight. This is the single most important
   implementation detail in the notebook.
2. **Paired differences are the primary statistic.** Every comparison is reported as the
   mean per-soil difference with a soil-level bootstrap CI and a sign count
   (e.g. "wins 7/21"). Unpaired means are secondary and explicitly labelled as such.
3. **A null must be reported as a null.** Where a CI spans zero the record says
   "not distinguishable", never "no effect". The colour null in §2.2 is an *absence of
   evidence*; colour could still matter at an effect size this design cannot resolve.

**Multiplicity:** 4 cells × 4 conditions = 16 simple-effect tests. Report them all, and
base the conclusion on the two *main effects* (frequency, colour), each averaged over its
two levels, rather than on the minimum of 16 numbers.

**Leakage:** unchanged from E2. Scaler, PCA basis and curve basis refit inside each
training fold. Under any camera ruler the held-out soil's label never enters the fit
even though that soil exists under the other camera — asserted in-cell, as E2 does.

**Circularity:** unchanged from E2. Alpha is chosen inside each outer fold; the headline
number is the score of the *procedure*, never of the best alpha.

## 5. Selection rule — stated before the run

1. Compute all four cells under CAM+RES with nested procedure scores.
2. **Primary pick:** the cell with the best nested CAM+RES score.
3. **Robustness check, reported but not used to choose unless it disagrees by more than
   the paired CI:** the minimax pick, i.e. the cell minimising its worst performance
   across CAM and CAM+RES. If primary and minimax disagree by more than the paired CI,
   the record says so and the *more conservative* (fewer features) cell is submitted.
4. If the winning cell's advantage over E2's 14-feature configuration has a paired CI
   spanning zero, the submission is explicitly labelled **a bet, not a finding**, and
   that phrase goes in `Experiment3.txt`.

I expect rule 3 to fire. Recon has CORE+colour and CORE5 within 2 EMD on CAM+RES and
10 EMD apart on CAM, which is exactly the disagreement this rule exists to handle.

## 6. Pre-registered predictions

Written before the run. The run may not edit this list.

| # | prediction | blind? | if wrong |
|---|---|---|---|
| P1 | The frequency main effect under CAM+RES is positive with a soil-bootstrap CI excluding zero in ≥3 of 4 conditions | no — confirmation | Recon's paired result was an artefact of the single condition; the whole §2 story collapses |
| P2 | The colour main effect under CAM+RES has a CI spanning zero | no — confirmation | Colour matters after all and E4 must revisit it |
| P3 | No colour × frequency interaction (main effects approximately additive: 19.94 vs 21.60) | no — confirmation | The families are entangled and a 2×2 was the wrong design |
| **P4** | **The E3 submission scores below E2's 71.27** | **YES** | The ruler does not extrapolate across kinds of change; stop trusting it for Model 2 decisions |
| **P5** | **Actual score lands in 45–52** (CAM+RES 54.1 × the 0.84–0.94 factors measured on E1 and E2) | **YES** | The calibration band is too narrow; widen it and re-derive before selecting on it |
| **P6** | **The implied factor stays within 0.7–1.1** | **YES** | The ruler's bias is configuration-dependent, so it can rank but not predict |

P4–P6 are the experiment. P1–P3 are its plumbing.

## 7. Processed data used — unchanged from E2

`data/processed_meta/manifest_{images,tiles,samples}.csv`, `data/tiles/` (256 px),
`data/sample_submission.csv`, `data/target_ppm.json`, `config_hash 010f44c36c74`.
Labels from `manifest_samples.csv` only — the notebook never reads `data/Training` or
`data/Test`. Tile caches from E2 are reused when the `config_hash` matches, so no
re-extraction is needed.

No new Kaggle uploads beyond the three E1 CSVs already required by E2. E3 additionally
needs E2's `features_soil.csv`, `cv_families.csv` and `Submission_Model1_E2.csv` under
`experiment_history/Model1_E2/` for the same reproduction-assertion reason E2 had for
needing E1's.

## 8. Notebook organisation

`Model 1/Model 1 Experiment 3/Model1_Experiment3.ipynb` — sections:

- **A setup** — config, imports, input resolution, `config_hash` pin, seed.
- **B data & metric** — manifests, labels from the sample manifest, EMD with the
  three-way reconciliation assertion, reference baselines, and all three external
  ground-truth scores (E1 172.70, baseline 102.37, E2 71.27) as constants.
- **C prior results restated** — load E2's artifacts, assert the rebuilt feature matrix
  matches E2's to <1e-6, print the 172.70 → 71.27 trajectory.
- **D family definitions** — the three families declared once, with an assertion that
  their union is E2's 14 and that they are pairwise disjoint.
- **E the four evaluation conditions** — reuse E2's ruler machinery; add both camera
  directions and both resolution targets; assert the leak rule in all four.
- **F the 2×2 grid** — every cell × every alpha × every ruler, flat.
- **G paired simple effects** — main effects and interaction, soil-level bootstrap,
  sign counts. **This is the primary result table.**
- **H mechanism table** — per family, the mean shift under camera change and under
  sharpness change separately, so §2.3's split is derived in-notebook rather than
  asserted from recon.
- **I permutation control** — each family against shuffled labels under CAM+RES, so a
  family that carries no signal is not credited with carrying robustness.
- **J selection** — apply §5's rule, write `selection_rule.json`, report whether the
  primary and minimax picks agreed.
- **K predict & submission** — fit the selected cell on all 24 soils, 8 gates, write
  `Submission_Model1_E3.csv`.
- **L expected-score band** — CAM+RES reading × the two measured calibration factors,
  with P4/P5/P6 printed as explicit pass/fail lines to be checked after submitting.
- **M error analysis** — per-soil paired-difference plot, saturation readout against
  E1's 59.1% and E2's 10.0%, and the Münster row.
- **N record** — write `Experiment3.txt`.

## 9. Submission

`Submission_Model1_E3.csv`, generated by the notebook under that exact name, IDs from
`manifest_samples.csv` (`submission_id`, `submission_row_order`), validated by the same
8 gates before writing and re-read from disk afterwards.

## 10. Files created

```
Model 1/Model 1 Experiment 3/
  Model1_Experiment3.ipynb
  Experiment3.txt
  instructions.txt
  Submission_Model1_E3.csv
  factorial_grid.csv          every cell x alpha x ruler
  paired_effects.csv          the primary result
  mechanism_table.csv         per-family camera vs sharpness shift
  family_controls.csv         permutation controls per family
  selection_rule.json
  features_soil.csv  cv_families.csv
  kaggle_setup.md
```
`scratch/build_m1e3.py` and `scratch/mk_run3.py` as the generator and headless runner.
E1's and E2's folders are not modified.

## 11. Assumptions

- The public leaderboard subset has not changed between E1, E2 and E3. All three
  calibration points depend on it. If the host rotates the public soils the factors are
  void and P4–P6 become meaningless — this is the largest single assumption in the plan.
- The rank-3 curve basis remains adequate (ceiling 7.35, far below everything in play).
- 21 dual-camera soils remain the evaluation population, so all camera rulers share one
  correlated sample. Gains in §4 raise the *number of conditions*, not the number of
  independent soils; the real ceiling on this design's power is the 21.

## 12. Risks

1. **Power.** Even with four conditions the CIs may span zero for the winner-vs-runner-up
   comparison. The design can establish the ~20 EMD frequency effect; it probably cannot
   resolve a 5 EMD difference between CORE5 and CORE+colour. §5 rule 5 handles this
   honestly rather than pretending to a decision.
2. **Selecting the best of four cells on a noisy ruler is itself a multiplicity
   problem.** The nested procedure score partly controls it, but the *choice of cell* is
   made on a statistic with a ±20 CI. Report the winner's CI, not just its rank.
3. **The ruler may not extrapolate.** Its two calibration points both involve changing
   regularisation. E3 changes which features exist. If P4 fails, that is the finding, and
   it is more informative than a score gain would have been.
4. **Removing the frequency features removes the only physically interpretable proxy for
   grain size.** `dom_wavelength_mm` is the one feature whose units are millimetres of
   soil texture. Losing it makes the model less explainable even if it scores better, and
   the record should say so rather than celebrate.
5. **Colour may be doing work the ruler cannot see.** Colour is robust to blur and the
   ruler's blur model is a Gaussian approximation of a Lanczos prefilter. If the real
   iPhone pipeline differs more sharply than that, colour's apparent harmlessness is an
   artefact of the synthetic camera.

## 13. What this decides for later experiments

- **E4** train on resolution-matched views. If E3 confirms that sharpness sensitivity is
  the binding constraint, making the *training* distribution match the test's is the
  complementary fix to removing sharpness-sensitive features, and the two should be
  tested separately so their contributions stay attributable.
- **E5** blur-robust frequency surrogates: band *ratios* (e4/e8) rather than absolute
  spectral position. This re-introduces the physical interpretability lost in risk 4,
  without re-introducing the sensitivity.
- **Model 2** stays on hold until the ruler has at least three calibration points and
  P4–P6 have been resolved.

## 14. Cost

Roughly 3–4 minutes on CPU. No new tile extraction — E3 reuses E2's three caches. The
grid is 4 cells × 10 alphas × 4 rulers × 4 conditions, all closed-form ridge on ≤24 rows.
