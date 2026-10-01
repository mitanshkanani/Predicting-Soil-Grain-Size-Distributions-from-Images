# Model 8 / Experiment 3 — Plan

**Status: PLAN ONLY. Nothing trained, nothing predicted, no submission, no upload.**
**Date:** 2026-10-01
**Owner instruction this plan answers:** "write the experiment 3 plan buddy"

---

## 1. Where Experiment 2 left us

Five Kaggle scores, all from one feature family, all measured under the registered nested protocol:

| arm / probe | treatment | internal | **external** |
|---|---|---|---|
| **E2-A** | 12 features, colour raw, α=3 | 43.4532 | **55.80591** ← project best |
| E2-C | colour rank-normalised | 42.7268 | 62.51669 |
| E2-B | colour dropped | 42.9885 | 67.26598 |
| α-probe | α=0.3 | 43.9182 | 59.38746 |
| α-probe | α=30 | 51.7009 | 71.10598 |

Two results, and they point in opposite directions:

- **The α curve is ranked CORRECTLY.** Internal argmin α=3, external argmin α=3. Spearman ρ = +1.000. We are already at the regularisation optimum; α is not a live lever.
- **The arms are ranked EXACTLY BACKWARDS.** Spearman ρ = −1.000. n=3, not significant alone — but noise cannot produce a perfect anti-rank under a deterministic metric, so this is real.

**The diagnosis that reconciles them:** the internal ruler is not broken. It measures *interpolation on 24 labelled soils* and answers that correctly. Experiment 2 needed *transfer to 10 out-of-hull soils*, and no amount of tuning inside the first question moves the second. The α curve confirms the ruler works; the arm ranking shows what it cannot see.

**Consequence for this plan: the question is not "which head is better". It is "what property of the head governs transfer".** That question cannot be answered internally at all, so this experiment, like E2, ends at submissions.

---

## 2. The actual cause: the split is a CAMERA split

This was found in E2's write-up and verified this session. It is not a modelling detail — it is the whole explanation.

`data/processed_meta/manifest_images.csv`, 162 images:

| camera | images | split |
|---|---|---|
| Motorola Edge | 69 | train |
| Samsung A52 | 55 | train |
| **iPhone 16** | **21** | **test only** |
| **iPhone 14** | **14** | **test only** |
| Motorola Edge 60 Fusion | 3 | train |

**Zero iPhone images in training. All 35 test images are iPhone.** The train/test split is by camera, not at random.

So every "extrapolation" this project has measured is a **phone-manufacturer colour-rendering shift**, not a soil-property shift:

- Blue exceeds training range by 3.33 SD, on 9/10 test soils
- lum_p50 by 2.22 SD, lum_p90 by 1.39 SD, G by 1.28 SD
- texture features: **all ≈ 0.000 excess**

Texture is camera-stable. Colour is not. That is why the frozen head survives, and it is also why E2-C (rank normalisation) failed — it destroys exactly the absolute colour scale that still carries soil identity when the sensor changes.

**And the preprocessing pipeline already applies a per-image grey-world correction whose gains differ systematically by camera:**

| camera | gw_gain_r | gw_gain_g | gw_gain_b |
|---|---|---|---|
| Motorola Edge | 0.942 | 0.999 | 1.063 |
| Samsung A52 | 0.905 | 0.977 | 1.132 |
| Motorola Edge 60 Fusion | 0.898 | 0.882 | 1.264 |
| iPhone 16 | 0.974 | 0.994 | 1.033 |
| iPhone 14 | 0.980 | 0.988 | 1.033 |

Motorola/Samsung get their blue pushed **up 6–26%**; iPhones get it up **3%**. After correction, blue-heavy iPhone soils are being compared against blue-pushed Samsung soils. That is a *direction* mismatch, not just a scale one, and it is the concrete mechanism behind the 3.33 SD blue excess.

---

## 3. Hypothesis under test

> **H3: transfer is governed by how camera-invariant the feature space is. Converting the colour block from a sensor-relative (RGB) space to a perceptually uniform space that separates luminance from chroma will improve transfer to an unseen camera, holding the head fixed.**

Note what H3 does **not** say. It does not say "add features" or "use a bigger backbone". DINOv2 already lost to the handcrafted head here by 4.76 EMD. It says the *colour basis* is the defect, and it is the same colour block E2 already proved is load-bearing — dropping it cost 11.46 EMD.

**Why internal CV cannot answer this.** Every ruler in this project scores interpolation on 24 soils, none of which were shot on an iPhone. A colour space cannot be evaluated on training data that contains no test-sensor data. So this is decidable only by submission, exactly as E2 was.

---

## 4. Arms — exactly three, declared now, before any number is seen

Same discipline as E2, which is the only reason E2 produced a result. No grid, no selection, all three submitted.

- **E3-A — perceptual colour.** E2-A unchanged except the colour block is recomputed in a luminance/chroma-decorrelated space. All 12 feature roles preserved (5 texture + 7 colour), so this is a one-variable change against the 55.80591 incumbent. *Implementation note:* `scikit-image` and `OpenCV` are **not installed**; PIL 12.3.0 is. The sRGB→CIELAB matrix transform is ~10 lines of numpy and is transcribed explicitly rather than imported, so no install is needed. Alpha stays 3.0 — the α curve proved it is at the optimum, so changing it here would confound the experiment.
- **E3-B — grey-world–matched colour.** The colour block is divided by the per-image grey-world gains **already recorded in `manifest_images.csv`** (`gw_gain_r/g/b`), making the features invariant to exactly the correction the pipeline already applies. This is a *measured* correction taken from the project's own metadata, not a guessed one. Directly tests whether undoing the pipeline's own colour handling recovers the missing invariance.
- **E3-C — E2-A plus the two unused texture columns** (`spec_centroid_cpm`, `dom_wavelength_mm`), 14 features. The only unused signal left in `features_soil.csv`. If texture is the camera-stable block, adding more of it should help; if it does nothing, texture is saturated and colour is the only lever. Either way it is informative, and it costs one slot.

**Positive control, free of charge.** E2-A's own score *is* the control at 55.80591 — no re-submission needed, because the metric is deterministic. Every arm is therefore measured against a known reference point.

**Rejected arms, and why** — recorded so this is not mistaken for arm-shopping:
- *DINOv2 + handcrafted fusion* — DINOv2 lost by 4.76 EMD; fusing a demonstrably worse feature space is not motivated by any result we hold.
- *α re-tuning* — α=3 is externally optimal. Changing it is the 39.26→83.94 pattern.
- *More arms* — three is the cap. The project's own measurement: best-of-15 on shuffled labels buys 15.29 EMD of fake gain.

---

## 5. Gates

| gate | requirement |
|---|---|
| **G0** | `ruler.py` reproduces both registered values — 43.453225201811563 (nested) **and** 41.1475158656982032 (oracle α=3, verified bit-exact in E2) |
| **G1** | each arm's design is finite, fold integrity clean (16 folds, 0 family leaks) |
| **G2** | submission contract valid — 10 rows, IDs in `sample_submission.csv` order, monotone, exactly 100 at 200 mm, no NaN |
| **G3** | **PRIMARY, external:** arm beats 55.80591 by >1 EMD |
| **G4** | reporting rule only — a win inside the public band (±~25 EMD) is reported as INCONCLUSIVE, never as progress |

G0 is deliberately strengthened to check **two** recorded values this time. In E2 the 41.1475158656982032 match proved the transcription reproduces Model 2 E3's oracle EMD to the last bit. Both anchors, from now on.

---

## 6. Outcomes — decided now, not after

- **E3-A or E3-B beats 55.80591 by >1 EMD** → H3 supported; camera-invariance is the lever; Model 8 continues in that direction.
- **All three land inside 55–57** → H3 refuted. Combined with E2's α result, this says the handcrafted head is at its ceiling and the ~24 labelled soils are the binding constraint. **Stop tuning the head** — that would itself be the most valuable thing this project has established.
- **All three land worse than 55.80591** → the raw RGB colour block was better *despite* the camera shift, which would mean the grey-world gains carry real soil information rather than being pure sensor error. Surprising, and worth knowing.
- **Nothing lands inside the public band** → the dataset cannot resolve differences of this size and further submission-based search is uninformative. E2 §10 already predicted this as the likely end state; this would confirm it.

**No outcome is spun.** `SUBMIT = TRUE` iff H3 is supported. An external tie is not support.

---

## 7. Data, compute, and the honest limits

- **No new measurements.** `features_soil.csv` (34 rows) plus `manifest_images.csv` grey-world gains. No image tiles opened, no embedding extraction, no GPU, **no Kaggle notebook, nothing to upload.** Generation is CPU arithmetic on 34 rows, in seconds — same as E2. You paste the CSV to the submission page.
- **Local environment:** Python 3.14.6, numpy 2.5.2, pandas 2.3.3, scipy 1.18.0, scikit-learn 1.9.0, PIL 12.3.0. **No torch, no timm, no skimage, no cv2.** Landmines: `np.trapz` removed in numpy 2 (use `np.trapezoid`); local stdout is cp1252, so all printed text must be ASCII.
- **Compute:** minutes, CPU only.

**Limits stated before any score is seen, so they cannot be retrofitted:**
- Public board is 3 fixed test soils; per-soil SD 22.75 → 3-soil null SD 13.13. A 4.76 EMD margin is **0.36 SD**. Differences below ~5 EMD are not resolvable on the public board. Private carries 70% weight.
- n = 24 labelled soils, **~8–10 effectively independent** (six near-duplicate pairs). Any conclusion is about that many units, not 24.
- Three arms cannot separate "CIELAB is better" from "CIELAB decorrelates luminance from chroma", nor "CIELAB better" from "any monotone recolouring would do".
- The camera split is **confounded with the split itself**. With 0 iPhone training images, we cannot separate "unseen camera" from "unseen anything" — no amount of data from *this* dataset fixes that, only more cameras would.
- **The metric is deterministic.** Resubmitting an identical file returns an identical score. There is no noise-measurement experiment available, so the board's resolution is a fixed property we work around, not something we can calibrate away.

---

## 8. Implementation plan

1. `ruler.py` and `d50.py` already copied into this folder from E2, per standing instruction — E3 stands alone and imports nothing from another experiment.
2. `build_e3.py` — load features + grey-world gains; assert G0's two anchors; build the three designs; score each under the registered nested LOFO (**reported, never used to select**); fit on all 24; assert G2 before writing; emit `Submission_Model8_E3_<arm>.csv` ×3.
3. `arm_comparison.csv`, `run_manifest.json`, `reproduce_anchor.txt`.
4. `experiment_report.txt` with the pre-declared outcome branch and an explicit "awaiting Kaggle result" section.
5. **Stop.** You submit; I record the returned scores and close the verdict.

**Nothing is built until this plan is approved.**

---

## 9. What this experiment is really for

Not a better score. **A decision.**

E2 left the project with a measured, uncomfortable fact: our selection ruler anti-ranks arms, so internal work cannot tell us which head to submit, and we are one month behind. E3 is designed to answer a single question — *is the camera split the reason?* — with three pre-registered arms and an external verdict.

Either it is, and the project finally has a mechanism to fix. Or it isn't, and we learn that a 24-soil, single-camera-family dataset cannot resolve differences of this size, which is worth knowing precisely because it tells us to stop spending slots. Those are very different places to be standing, and right now we are in neither.