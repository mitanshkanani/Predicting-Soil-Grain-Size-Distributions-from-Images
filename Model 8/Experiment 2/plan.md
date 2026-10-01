# Model 8 / Experiment 2 — Plan

**Status: PLAN ONLY. Nothing trained, nothing predicted, no submission, no upload.**
**Date:** 2026-09-30
**Owner instruction this plan answers:** we are a month behind, gates keep failing, stop
shopping for gains and find out *seriously* whether we are good or bad. Run like someone
who intends to train, test and submit.

---

## 1. The question this experiment exists to answer

Not "can I find another 2 EMD internally." That is the question that has produced 24
internal numbers and zero transfer, and it is unanswerable with the data we have.

**The real question: is our ~60 external score the ceiling, or is the model simply
mis-specified for the test soils?**

There is no way to answer that from internal CV. Every ruler in this project measures
interpolation on 24 labelled soils, and all of them will keep saying 43 while Kaggle
keeps saying 60. So this experiment is deliberately the first one that ends at a
submission.

---

## 2. What the evidence actually says about the leak

The owner is right that 39.26 → 83.94 is the key evidence, and it is worth stating
properly. The project has three instances:

| internal | external | transfer |
|---|---|---|
| 35.98 (M1 E1, 16 features) | 172.70 | worst ever |
| 39.26 (M2 E3 `fr`) | 83.94 | worst of its experiment |
| 39.78 (M2 E1 `R`, random-init ViT) | 78.55 | worst arm |
| **44.33 (M2 E1 `D`)** | **60.56** | **best — from the WORSE internal of the pair** |

That last row is the informative one. `D` (pretrained DINOv2) and the M1 control differ by
3.18 EMD internally, and `D` wins externally by 0.674. Internal CV ranked them the wrong
way round.

**Three candidate mechanisms, and the evidence is genuinely ambiguous:**

1. **Feature shift.** Measured in Model 8 E1 §9.5: 0 of 10 test soils have a training
   neighbour closer than the typical training 1-NN distance. All the extrapolation is in
   the colour block (B exceeds the training range by 3.33 SD, 9/10 soils above training
   max). Texture features are all inside range.
2. **The head is wrong for the test distribution.** Our internal CV never scores a soil
   outside the training hull, so a head that extrapolates badly is invisible to it.
3. **n=8–10 effective soils.** The 3-soil public mean has a 95% band of [21.7, 69.9].
   Most of "transfer penalty" may be *irreducible sample noise*, not a fixable defect.

Mechanism 3 is genuinely plausible and nobody in this project has ever priced it. **That
is the gap this experiment closes.**

---

## 3. Why internal CV literally cannot answer it

The 10 test soils are all outside the training feature hull. So there is no honest way to
make our own validation look like the test set, because **we do not have 10 soils outside
our own hull with known labels.**

What we do have: 24 labelled soils, ~8–10 effectively independent. So:

- We cannot measure our own transfer penalty.
- We cannot estimate the test-set error to better than ±25 EMD on the public board.
- We can only *submit* and observe.

That is the honest reason the project produced six submissions and no understanding.

---

## 4. Hypothesis under test

> **H2: the frozen head is mis-specified for the test soils because it was fitted on a
> feature distribution the test soils do not occupy, and a head constrained to be
> invariant to the direction the test data actually moved (colour) will transfer better
> than one that is not — regardless of what it scores internally.**

Note what this hypothesis does *not* say. It does not say "drop colour" because that
scored better internally. §9.6 showed the internal verdict flips sign depending on which
soils are scored (all 24: −0.47; 12 most isolated: −2.28; 6 most isolated: **+4.75**), so
the colour question is undecidable internally and has been for two experiments.

**This is the one question in the project that is decidable only by submitting.** So it
gets the submission.

---

## 5. Pre-registered arms — exactly three, fixed now, before any number is seen

No arm-shopping. This is the discipline that produced 39.26→83.94: every previous
experiment tested 3–5 arms and reported the winner, and the project's own measurement
shows picking the best of 15 heads on *shuffled labels* buys a median **15.29 EMD**
apparent gain. A 2.96 EMD "improvement" is exactly what picking the luckiest of five
buys. Three arms, decided in advance:

- **E2-A — the incumbent.** The frozen head, refit on all 24 soils, 12 features. This is
  the number we must beat and it is *not* a new arm; it is the reference.
- **E2-B — colour-dropped.** The same head on the 5 texture-core features
  (`e4, e8, e16, lum_sd, grad_mean`). Tests whether removing the block the test data
  actually moved in improves transfer.
- **E2-C — colour-invariant.** The same 5 features, plus per-soil colour *rank*
  normalisation (each colour channel converted to its within-training-set percentile)
  rather than dropping it. Keeps the information, removes the scale shift.

**E2-C exists because "drop colour" and "make colour invariant" are different claims,
and the project's best external result (60.56) came from the arm with the smallest
transfer penalty — an invariance argument, not a deletion argument.**

No third arm. No grid. If all three lose, we report that and stop.

---

## 6. Gates — designed for *power*, not just rigour

Experiment 1's lesson, recorded honestly in its report: G4 used a permutation null with SD
4.82 and p95 +8.80, **wider than the effect it was meant to test**. A gate that cannot
resolve an effect will always fail, and I built one without checking its power first.

So each gate here is paired with a **power check run first**:

| gate | requirement | power check run BEFORE the gate is trusted |
|---|---|---|
| **G0** | frozen head reproduces 43.453225201811563 | already proven bit-exact in E1; re-asserted |
| **G1** | each arm's internal LOFO is finite and its fold integrity is clean | fold report before scoring |
| **G2** | **PRIMARY, and it is external:** the arm's public score beats the incumbent's | *by construction* — a submission is the measurement |
| **G3** | submission is contract-valid: 10 rows, IDs in `sample_submission.csv` order, monotone, exactly 100 at 200 mm, no NaN | asserted before any upload |
| **G4** | a gain is only claimed if it exceeds the **tie threshold** (~1 EMD, since public scores 60.5–61.2 are inside the project's own tie band) | pre-registered as a *reporting* rule, not a selection step |

**Deliberate change from Experiment 1: the primary gate is EXTERNAL.** An internal gate
cannot answer this project's only open question, and pretending otherwise is what produced
24 internal numbers and zero understanding.

---

## 7. Data — no new measurements, and this is why it is fast

- `Model 1/Model 1 Experiment 3/features_soil.csv` — 34 rows (24 train + 10 test), cached,
  read directly. No image tiles are opened. **No `data/tiles`, no zip, no GPU, no Kaggle
  notebook.**
- `Model 2/Model 2 Experiment 3/cv_families.csv` — the fold key.
- `data/processed_meta/manifest_samples.csv` — the 11-support labels.

Verified present on disk this session. The whole run is CPU arithmetic on 34 rows.

---

## 8. Compute

**CPU only. Minutes.** Three arms, each a closed-form ridge on 24×5 or 24×12. Generating
the submission is one matrix multiply plus the frozen projection. There is no deep model,
no embedding extraction and no image read anywhere in this experiment.

The Kaggle side is different and worth stating plainly: **you do not need to upload
anything.** Generation happens here, on this machine, in seconds. You paste the CSV to the
competition submission page exactly as Model 1 E1–E4 did — those all ran locally too.

---

## 9. Submission contract — verified this session, not assumed

Checked against `Model 1/Model 1 Experiment 3/Submission_Model1_E3.csv`:

- 10 rows, 12 columns: `sample_id` + the 11 supports
- `sample_id` values are the **`HPC_`-prefixed competition ids**, in the exact order of
  `data/sample_submission.csv` — **not** the internal sample ids
- monotone non-decreasing across the supports
- exactly **100** at the 200 mm column
- no NaN

All five assertions are enforced in code before any file is written for upload.

---

## 10. What each outcome means — decided now, not after seeing the number

- **An arm beats 60.56167 by >1 EMD** → we were mis-specified, the colour hypothesis is
  supported, Model 8 continues in that direction.
- **An arm lands inside 60.5–61.2** → **inconclusive, not a win.** The public band cannot
  resolve this. Do not report it as progress.
- **All three arms exceed 61.24** → the colour hypothesis is refuted externally *and*
  internally was undecidable; the ~60 ceiling is probably mechanism 3 (sample noise), and
  the honest conclusion is that this dataset cannot resolve differences of this size.
- **All three arms land near 60.5–61.2 and we submit anyway** → we have spent a slot to
  learn the band width, which is itself worth having once.

**No outcome is spun.** The project's own rule stands: `SUBMIT = TRUE` iff
`VERDICT == SUPPORTED`, and an external tie is not support.

---

## 11. Risks

| risk | likelihood | mitigation |
|---|---|---|
| We burn slots on noise | **high** | three arms, one submission each, pre-registered; no grid |
| The public board cannot resolve anything | **very high** | stated in advance; the tie band is reported as a first-class result |
| Internal numbers mislead us again | high | E2-A's internal is *reported* but never used to select the submission |
| Colour-robustness overfits 24 soils | medium | E2-C's percentile transform is a monotone per-channel map with no fitted parameter beyond the training quantiles |
| We conclude "we are bad" from a noisy board | medium | mechanism 3 is priced explicitly in section 2, before any score is seen |

---

## 12. Exact implementation plan

1. Reuse `Model 8/Experiment 1/ruler.py` and `d50.py` **by copy into this folder**, not by
   import — this experiment must stand alone and not depend on another experiment's folder.
2. `build_e2.py` — assemble the three designs, score each under the registered nested LOFO,
   emit `arm_comparison.csv` and `cv_per_soil.csv`.
3. **G0 first**: reproduce 43.453225201811563 on the incumbent. Stop and report if not.
4. Fit all three arms on all 24 training soils.
5. Predict the 10 test soils; apply the frozen projection.
6. Assert the five submission contract rules. Write `Submission_Model8_E2_<arm>.csv` per arm.
7. Write `experiment_report.txt` with the internal numbers, the contract evidence, and an
   explicit "awaiting Kaggle result" section.
8. **Stop.** The user submits. Record the returned score in the report when it arrives.

---

## 13. Honest statement of what this experiment can and cannot deliver

**Can:** the first real external measurement of three distinct heads under one protocol,
on one submission slot each, with the contract verified. And a definitive internal
statement of the colour question, which two experiments have now failed to resolve.

**Cannot:** attribute a score difference to a mechanism at n=8–10 effective soils. The
public board has a ±25 EMD band. **If the three arms differ by less than that, we will not
know which is better, and no amount of internal work will tell us.**

That is the honest limit of this dataset, and it is worth stating now rather than after
we have spent three slots discovering it.

**The one thing that is worth doing regardless of the scores:** if all three arms land in
the same band, that is the finding — it establishes that the project has reached the
information ceiling of a 24-soil dataset, and that further internal tuning is not merely
slow but *uninformative*. That is a result, and it is one nobody here has established yet.