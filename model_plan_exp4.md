# Experiment 4 plan — Model 1

**Status: DRAFT FOR REVIEW. Nothing below is implemented.**

This plan **cancels the experiment E3 scheduled** and replaces it. The cancellation is
the main result of the reconnaissance, so it is argued from measurements first.

---

## 0. What E3's leaderboard score did to the ruler

E3 scored **61.23560** against a predicted band of 44–49.

| experiment | CAM+RES reading | actual Kaggle | factor |
|---|---|---|---|
| E1 | 205.34 | 172.70 | 0.84 |
| E2 | 75.73 | 71.27 | 0.94 |
| E3 | 51.87 | 61.24 | **1.18** |

- **P4 CONFIRMED** — 61.24 beat 71.27. The ruler's *ranking* held for the third time.
- **P5 REFUTED** — 61.24 is 12.2 above the top of the band.
- **P6 REFUTED** — the factor left the 0.7–1.1 window and has moved monotonically.

The deltas are the real story. The ruler predicted E2→E3 would gain 23.9 EMD; it gained
10.0. It predicted E1→E2 would gain 129.6; it gained 101.4. **The ruler's optimism grows
as the configurations improve** — the classic signature of a proxy that models the
nuisances it was built from and goes blind once those nuisances are removed.

**Consequence, and it is binding for E4:** CAM+RES is an **ordinal** instrument. It may
be used to rank candidates. It must NOT be used to forecast a score, and it must not be
used to choose between two candidates whose readings differ by less than roughly 15 EMD.
Every expectation in this plan is written as "better or worse than 61.24", never as a
number.

## 1. The planned E4 is cancelled, with evidence

E3's record scheduled E4 as *train on resolution-matched views* — fit the model on
training tiles after applying the anti-alias blur the test images actually received. Reconnaissance
(`scratch/cal_exp4.py`, read-only) killed it.

### 1.1 Once the frequency features are gone, the model barely cares about sharpness

E3's selected 12-feature set. Each row trains on the Motorola view at one blur level and
predicts the Samsung view at the column's level. σ: real ≈ 0.04 px, res14 = 0.84, res16 = 1.20.

| train ↓ / predict → | real | res14 | res16 |
|---|---|---|---|
| real | 57.39 | 54.12 | 53.81 |
| res14 | 59.49 | 52.88 | **51.54** |
| res16 | 60.70 | 54.67 | 53.51 |

Total spread across all nine cells: **9.2 EMD**. Training on matched views buys at most
2.3 EMD on the deployment column (53.81 → 51.54), and the fully-matched row is *worse*
than the partially-matched one (53.51 vs 51.54). That is noise, not signal. **E3's feature
selection already absorbed the sharpness problem.**

### 1.2 For the 14-feature set the same table shows the trap

| train ↓ / predict → | real | res14 | res16 |
|---|---|---|---|
| real | 50.99 | 75.73 | **105.18** |
| res14 | 77.76 | 48.74 | 80.20 |
| res16 | **115.91** | 84.24 | **52.32** |

Textbook diagonal dominance: train-at-level-L, predict-at-level-L always scores ~50;
every off-diagonal explodes. The frequency features turn the model into a **sharpness
meter**.

And here is the trap. If we train on res16-blurred views and evaluate with a ruler whose
target is res16-blurred views, frequency looks *harmless*:

| held-out sharpness? | 12 features | 14 features (+freq) | frequency costs |
|---|---|---|---|
| train real → test res16 (**deployment**) | 53.81 | 111.41 | **+57.60** |
| train res14 → test res16 (held out) | 51.54 | 86.79 | **+35.24** |
| train res16 → test res16 (matched) | 53.51 | 50.44 | **−3.08** ← looks *helpful* |

The bottom row would have us conclude that E3 was wrong and put the frequency features
back. That conclusion is a **34-EMD artefact** of a ruler measuring the exact thing the
model was trained on.

### 1.3 The general rule this establishes

> Any treatment that alters the training distribution must be evaluated on a held-out
> level of the nuisance it targets. A ruler whose evaluation condition equals the
> treatment condition is circular and will report the treatment as free.

This is the sharpest methodological finding of the project and it was not anticipated in
any plan. It also retroactively flags E2's RES-MATCH construction as safe only because
E2 never *trained* on the res-matched views.

## 2. The other candidate is also dead

E6 on the ladder was "richer aggregation". Reconnaissance (`scratch/cal_exp4b.py`) tested
it directly, on E3's own 12 features and identical protocol:

| tile → soil summary | columns | LOGO-CV |
|---|---|---|
| median only (current protocol) | 12 | **41.15** |
| median + mean + sd | 36 | 42.13 |
| + p10 / p90 spread | 60 | 42.22 |

Quintupling the feature space by adding distributional summaries makes it **slightly
worse**. The intuition that a grain-size *distribution* deserves a distributional summary
is right in physics and wrong at n = 24 labelled soils. **E6 is cancelled too.**

## 3. What is actually left

Error budget after E3:

| quantity | EMD |
|---|---|
| E3 actual Kaggle | 61.24 |
| E3 in-domain, oracle alpha | 41.15 |
| **unmodelled domain shift** | **~20** |
| rank-3 representation ceiling | 7.35 |
| in-domain headroom | ~34 |
| rank 49 on the leaderboard | 40.22 |

Sharpness is handled (§1.1). Aggregation has nothing to give (§2). The ~20 EMD of
remaining domain shift is therefore **the camera's colour pipeline** — the one measured
shift that survives: colour moves 1.54 z between Motorola and Samsung, the largest of any
family, and E3 showed the model still keeps seven colour features.

But E3 also showed something awkward: **the ruler cannot decide whether to keep them.**

| cell | features | nested CAM+RES | nested CAM |
|---|---|---|---|
| A core | 5 (no colour) | 56.98 | **49.27** |
| C core+colour | 12 | **51.87** | 52.26 |

CAM+RES prefers colour. CAM prefers no colour. The difference is 5 EMD on CAM+RES — well
inside the ±15 EMD band §0 just declared unresolvable. **This is exactly the decision the
ordinal ruler is not allowed to make, and it is the decision E4 must settle.**

## 4. What Experiment 4 tests

**One question:** does the colour block earn its place, and if so in which form?

Single variable: **how colour is represented.** Everything else — texture core, rank-3
basis, ridge, monotone projection, two-stage median aggregation, CV families, split,
metric, and the CAM+RES ruler's *construction* — is frozen at E3's values.

### Arms

| arm | colour treatment | features | why it is in the experiment |
|---|---|---|---|
| **C0** | E3's raw channel means + saturation | 12 | control; external score already known (61.24) |
| **C1** | none | 5 | the camera-preferred option; maximal robustness and parsimony |
| **C2** | per-camera robust z-score of the same 7 | 12 | removes the device offset while keeping within-camera colour variation — the classic label-free domain-adaptation move |
| **C3** | chromaticity ratios R/Σ, G/Σ, B/Σ + saturation | 4 | scale-invariant colour that cannot carry a device's brightness or gain |
| **C4** | raw colour only, core removed | 7 | negative control; E3 measured colour-only transfer as catastrophic and C4 must reproduce that or the pipeline has changed |

C2 and C3 are new *representations* of an existing family, not new measurements — no
re-extraction, no new image processing. Everything is derived from the tile caches E3
already has.

### The circularity disclosure, applied to our own experiment

**C2 cannot be scored by the CAM ruler.** Per-camera standardization removes precisely
the between-camera mean and scale difference that CAM measures, so CAM would rate it near
perfect by construction. Same failure mode as §1.2. Therefore:

- C2 is ranked on **CAM+RES** and on **CAM+RES at a held-out blur level** (train on real,
  evaluate at res16 — the deployment condition), never on CAM alone.
- C2's true value is settled by the **leaderboard**, not internally. It is the arm the
  ruler is structurally incapable of judging.
- The notebook prints this disclosure next to C2's numbers rather than burying it.

### A second, unavoidable problem with C2

Per-camera standardization needs to know which camera produced which test image. The test
filenames do carry the camera, so it is technically available — but the host varied
cameras deliberately *to force generalisation*, and keying a normalisation to device
identity is close to the metadata line the project rules already draw for inputs.

E4 should therefore report C2 twice: **C2a** using true camera identity, and **C2b** using
unsupervised clustering of the colour features to assign each image to a device group
without ever reading the camera label. If C2b ≈ C2a, the trick is legitimate and portable.
If C2a ≫ C2b, C2 only works by exploiting metadata and should be rejected on principle
even if it scores.

## 5. Statistical protocol

Unchanged from E3 and deliberately so — this is the same design that produced a FINDING
last time:

1. **Nested procedure scores.** Alpha chosen inside each outer fold on CAM+RES; the
   headline number is the score of the procedure, never of the best alpha.
2. **Paired per-soil differences** as the primary statistic, with a **soil-level**
   bootstrap (the conditions share 21 soils; resampling observations would give falsely
   tight intervals — asserted in-cell).
3. **Four CAM+RES conditions** (both camera directions × both resolution targets) plus
   two CAM conditions, plus a held-out-blur condition for the arms that need it.
4. **A null is reported as a null.**
5. **The ordinal rule, stated before the run:** if two arms' paired CI spans zero, the
   notebook must print "the ruler cannot separate these" and must not pick between them on
   score. The tie is broken by parsimony (fewer features) and the choice is flagged as a
   bet.

## 6. Pre-registered predictions

Reconnaissance has NOT tested arms C2, C3 or C4, so unlike E3 most of these are genuinely
blind. That is stated per prediction.

| # | prediction | blind? |
|---|---|---|
| P1 | C4 (colour only) reproduces E3's catastrophic colour-only transfer, CAM > 75 | no — it is a pipeline consistency check |
| P2 | C1 (no colour) and C0 (raw colour) are **not** separable by CAM+RES: the paired CI spans zero | **YES** — and this is the plan's central claim, so refuting it would be the better news |
| P3 | C3 (chromaticity) transfers better than C0 under CAM+RES while retaining most of C0's in-domain score | **YES** |
| P4 | C2a beats C2b on CAM+RES, i.e. identity-based normalisation gains something clustering cannot recover | **YES** |
| **P5** | **The submitted arm scores below 61.24** | **YES** — the only real test |
| **P6** | **No point estimate is offered for the score.** Any forecast outside a ±15 EMD band is declared invalid by §0 before the run | n/a — this is a commitment, not a prediction |

P2 is the important one. If the ruler *can* separate C0 from C1, then E3's ambiguous
result was a measurement problem rather than a power problem, and the ordinal-only
verdict in §0 was too hasty.

## 7. Selection and submission

1. Rank the arms by nested CAM+RES.
2. Apply the ordinal rule: any two arms within an unresolved CI are tied, and the tie
   goes to the **fewer features**.
3. C2a is excluded from internal ranking entirely (it is the circular arm) and is
   submitted only if it wins on the held-out-blur condition, with that basis written into
   the record.
4. The submission is the selected arm. **The record must state which of the two possible
   reasons applies**: either the ruler separated the arms, or it did not and parsimony
   decided.

## 8. Notebook organisation

`Model 1/Model 1 Experiment 4/Model1_Experiment4.ipynb`:

- **A setup** — config, imports, input resolution, `config_hash` pin, seed.
- **B data & metric** — manifests, labels from the sample manifest, metric reconciliation,
  the four external ground-truth scores (E1 172.70, baseline 102.37, E2 71.27, E3 61.24).
- **C ruler status, restated** — the 0.84 / 0.94 / 1.18 calibration table, the shrinking
  delta ratios, and the ordinal-only rule derived from them. Printed before any arm is
  scored so the constraint is visible, not retrofitted.
- **D the two cancelled hypotheses** — reproduce §1.1, §1.2 and §2's tables in-notebook
  from the caches, so the cancellations are auditable rather than asserted from a scratch
  file. Includes the matched-training × matched-ruler circularity table.
- **E colour arms** — build C0–C4 from E3's tile features; assert each arm's feature
  count and that C0 is byte-identical to E3's cell C matrix.
- **F evaluation conditions** — E3's six plus the held-out-blur condition; leak assertion
  across all of them.
- **G nested scoring** of every arm.
- **H paired effects vs C0**, with CIs, sign counts, and the explicit "cannot separate"
  verdict where applicable.
- **I circularity audit of C2** — C2's CAM score next to its CAM+RES and held-out-blur
  scores, demonstrating the inflation numerically.
- **J C2a vs C2b** — identity-based vs cluster-based device grouping.
- **K selection, submission, 8 gates.**
- **L error analysis** — per-soil paired differences vs C0, blur-sensitivity matrix for
  the selected arm (the §1.1 table re-run on it), saturation.
- **M record** — write `Experiment4.txt`.

## 9. Data, paths, Kaggle

Unchanged from E3. No new tile extraction — every arm is derived from E3's three caches.
Adds `experiment_history/Model1_E3/{features_soil.csv, cv_families.csv,
Submission_Model1_E3.csv, paired_effects.csv}` to the dataset, for the same
reproduction-assertion reason E2 and E3 had. Runtime ~3 minutes first, ~1 minute cached.

## 10. Files created

```
Model 1/Model 1 Experiment 4/
  Model1_Experiment4.ipynb   Experiment4.txt   instructions.txt
  Submission_Model1_E4.csv   kaggle_setup.md
  colour_arms.csv            nested_scores.csv
  paired_effects.csv         blur_sensitivity_matrix.csv
  c2_circularity_audit.csv   device_grouping_check.csv   selection_rule.json
```
Plus an amendment to `Model 1/instructions.txt` recording that the planned E4 (resolution
matching) and E6 (aggregation) are **cancelled with evidence**, and the ordinal-only
ruling.

## 11. Assumptions

- The public leaderboard subset is unchanged across E1–E4. All four calibration points
  depend on it. This is the largest single assumption in the project and it is
  untestable from our side.
- 21 dual-camera soils remain the evaluation population. Power cannot improve without
  more labelled soils, so E4 will not resolve differences below ~15 EMD. That is a
  property of the dataset, not of the design.
- E3's selected arm remains the correct control; if E3's 61.24 turns out to be partly
  luck on 3 soils, C0's anchor is misleading by that much.

## 12. Risks

1. **We may be near the ceiling of what Model 1 can do.** Sharpness handled, aggregation
   exhausted, colour ambiguous within noise. If no arm separates from C0, the honest
   conclusion is that classical features have plateaued and Model 2 is justified on
   evidence rather than on restlessness.
2. **C2 may be a metadata exploit wearing a normalisation costume.** §4's C2b test exists
   specifically to catch this. If C2 only works with true camera identity, reject it even
   if it scores — the private split may not reward it and the project rules already ban
   camera as an input.
3. **Five arms is more multiple-comparison surface than E3's two factors.** The ordinal
   rule in §5.5 is the guard; it must be applied, not admired.
4. **A null result here is genuinely likely** and is a legitimate outcome. P2 *predicts* a
   null. If it comes true, the deliverable is "colour is not worth arguing about; move on"
   — which is worth more than a 3 EMD win we could not believe.

## 13. What this decides

- If an arm separates from C0 and scores better externally → Model 1 still has room; run
  the remaining ladder.
- If nothing separates and P5 fails → **Model 1 is finished.** The in-domain headroom
  (~34 EMD) is not reachable by re-summarising the same hand-built features, and Model 2
  (learned representation) becomes the evidence-backed next step rather than a hunch.
- Either way E4 is the last cheap Model 1 experiment worth running before that decision.
