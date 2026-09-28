# Model 2 — Experiment 2 implementation plan

> **For agentic workers:** implement task-by-task, checkbox by checkbox. Commit only when
> the project owner asks — this repository's convention is that the owner creates commits.

**Format note:** this keeps the project's established review format (the plan-plus-
pre-registered-predictions structure used for E2/E3/E4 and Model 2 E1, specified and
approved four times) and adds the task decomposition, file-responsibility map and
verification steps from the plan-writing convention. Where the two disagree, the project's
format governs. Plans live in the repository root, not `docs/superpowers/plans/`, for the
same reason.

**Goal:** Determine whether Model 2 E1's refutation of the pretrained backbone survives correcting the one confound it declared in advance — patch magnification — and establish, per soil-size group, whether making a DINOv2 patch smaller than the grain buys anything.

**Architecture:** One new variable: the physical soil area a tile contributes to the backbone. A centred sub-window of the existing 256 px tile is cropped and resized to 224 px, sweeping four scales (crop 256 = E1's own configuration, 128, 64, 32). Both embedding arms ride the whole sweep — `D` pretrained and `R` random-init — because the pretraining/capacity ablation has to survive the new variable or E2 is as uninterpretable as E1 would have been without R. `M1`, Model 1 E3's 12 hand features, stays fixed at full tile as the external anchor. Everything else is frozen at E1's values.

**Tech stack:** unchanged — numpy / scipy / pandas / scikit-learn / matplotlib / Pillow, plus `torch` 2.10.0+cu128 and `timm` 1.0.26 **only on Kaggle**. `backbones.py` is **not modified**; the crop happens in the notebook before `embed_tiles` is called, so E1's frozen interface stays byte-identical and E1's record stays reproducible.

**Spec:** this file is the spec. The governing prior findings are `Model 2/Model 2 Experiment 1/instructions.txt` (the refutation, the probe result, and the R-artefact analysis) and `Model 1/instructions.txt` (the instrument demotions).

---

## Global constraints

- **Metric:** trapezoid EMD on log10 d, `np.trapezoid(|F − F̂|, log10 d)`, mean over soils. Reconciled against the host's published trivial baseline 100.31 in every notebook.
- **Primary selection metric is in-domain LOGO-CV**, nested, as in E1. CAM+RES is **reported, never used to select** (E4's demotion).
- **`backbones.py` and `check_backbones.py` are not edited.** The crop is a data-view operation owned by the notebook. If either file must change, stop and report — that breaks E1's reproducibility.
- **Cache keys must include the crop.** `embed_cache_name` currently omits it; reusing E1's key would silently load the 256 px embedding for every crop. This is a correctness gate, not a nicety.
- **crop 256 is a regression check, not an arm.** It must reproduce E1's numbers exactly (M1 43.02, D 44.33, R 39.78) or E2 is void.
- **Validation:** leave-one-family-out over the 16 curve-distance families, copied unchanged from E1/E2/E3. The soil is the unit of supervision.
- **Leakage:** scaler, curve basis and ridge fitted on training folds only. Asserted in-cell.
- **Banned inputs:** camera, device, ppm, EXIF, ICC, site, sample-id. **Crop size may never be chosen per soil from its label** — D50 is a label; test labels are unknown. A label-free adaptive crop is a different experiment.
- **Original data:** `data/Training/` and `data/Test/` are never read or written. Labels come from `manifest_samples.csv`.
- **Preprocessing:** never re-run. `config_hash 010f44c36c74` pinned and asserted. No re-tiling — only 256 px tiles are materialised, and creating new tile sizes would be a preprocessing change.
- **A local mock run writes only into `local mock dry-run/`**, never the experiment root. Enforced in cell A1 since a mock run destroyed E1's downloaded outputs once.
- **Never** overwrite or delete a previous experiment's results.
- Local environment has **no torch, no timm, no pytest**. Verification scripts are plain `python` modules with fail-loud `assert`.

---

## 0. Why this experiment exists at all

E1 refuted its premise: a frozen DINOv2 ViT-S/14 does not beat Model 1's 12 hand-built features in-domain (D 44.33 vs M1 43.02), and it did not beat its own random-weights control (D − R = +4.55, CI [−7.02, +16.78]).

But E1 declared its own hole before it ran, in its §3 and in cell B1: **a patch spans 3.51 mm and 12 of 24 training soils have D50 below 1.03 mm**, so for those soils the backbone sees fines as texture and never as particles. Testing a representation at a magnification that cannot resolve the thing being measured is a weak test. Until that is separated, "learned features don't help" is provisional — and any later work built on arm D would be built on a possibly-handicapped measurement.

E2 closes that hole. It is cheap: one Kaggle session, no training, no gradients.

## 1. The measurement that bounds E2 before it runs

This was measured on the real tiles and real labels on 2026-09-28, not assumed. It changes what E2 can honestly predict.

Patch footprint for a crop of `c` canonical px resized to 224, and how many of the 24 training soils have a median grain at least one third of a patch:

| crop px | patch mm | window mm | soils resolvable | tile area used |
|---|---|---|---|---|
| 256 (E1) | 3.5145 | 56.23 | **12** | 100% |
| 128 | 1.7573 | 28.12 | 13 | 25% |
| 64 | 0.8786 | 14.06 | 13 | 6.2% |
| 32 | 0.4393 | 7.03 | **14** | 1.6% |
| 16 | 0.2197 | 3.51 | 16 | 0.4% |
| 8 | 0.1098 | 1.76 | 19 | 0.1% |

**Two things follow, and both belong in the record.**

1. **Magnification barely helps.** Going from E1's scale to a 32 px crop raises resolvability from 12 soils to 14 — while discarding 98.4% of the tile area. The relationship is nearly flat because the D50 distribution is bimodal: ten soils sit at 0.02–0.10 mm and twelve at 1.8–6.1 mm, with only two in between (0.211, 0.814). There is no crop that resolves the fines without throwing away the context the coarse soils need.
2. **The fines are physically unreachable, not under-sampled.** To give the largest fine soil (D50 = 0.1035 mm) a one-third-patch grain you need a ~23×23 canonical px window upsampled to 224×224 — 511 real pixels carrying no grain-level information. Upsampling cannot manufacture resolution; it can only re-space what is already there.

So E2's realistic scope is **the coarse twelve and the two mid soils**, not the fines. The prediction that follows is *specific and falsifiable*: any magnification gain must appear in the patch-limited D50 band and be absent in the fines. A uniform gain across all soils would be suspicious, and a gain concentrated in the fines would be impossible.

Also measured, because a small crop could in principle land off the soil: centre crops are representative. Soil density inside the crop versus the whole tile is 1.005 / 1.009 / 1.007 for crops of 128 / 64 / 32 px (p05 0.93, p95 1.12), and **zero of 240 sampled tiles had a centre crop below 0.5 soil fraction**. The tiles are cut from the inscribed soil rectangle, so soil fills them uniformly.

## 2. A correction to E1's record

E1's cell B1 and `backbones.PATCH_MM` state the patch as `14 / 4.552516 = 3.0752 mm`. That omits the 256→224 resize factor. The true footprint is **3.5145 mm** — 14.3% larger.

It does not change any E1 conclusion (the resolvability count is 12 either way, and the refutation rests on in-domain error, not on scale), but the number in the record is wrong and E2 must use the measured one. Task 1 fixes the constant in the *notebook's* derived values and records the correction; it does **not** edit `backbones.py`, whose `PATCH_MM` is left as authored so E1 stays byte-reproducible. The notebook computes the correct value and says why it differs.

## 3. Why crop 256 is a regression check, not an arm

E1's arm D at crop 256 is *the same computation* as E2's arm D at crop 256. So E2 must reproduce E1's 44.33 / 39.78 / 43.02 to floating-point noise, from a different notebook, in a different session. If it doesn't, something in E2's pipeline differs from E1's and no E2 number means anything. This is the same class of gate that caught E3's stage-ordering error worth 21.8 EMD.

It costs nothing: the crop-256 pass is needed anyway as the sweep's baseline.

## 4. Design — arms, and why R rides along

| arm | features/tile | crop sweep | question it holds fixed |
|---|---|---|---|
| `M1` | 12 hand-built | none (full tile) | the Model 1 anchor |
| `R` | 384, random ViT | 256, 128, 64, 32 | capacity, architecture, pipeline position |
| `D` | 384, frozen DINOv2 | 256, 128, 64, 32 | **nothing but magnification** |

R must ride the whole sweep for two independent reasons. First, if D improves at fine crops, that is only evidence about *pretraining* if R does not improve equally — R at 256 px already beat everything in-domain by exploiting a collapsed feature space, and a finer crop plausibly un-collapses it, which would be a scale effect on the random projection, not on learning. Second, E1's single most useful control becomes useless if it exists at only one scale.

**What is deliberately NOT in E2:** multi-crop per tile (changes the aggregation input count — a second variable); patch-token mean pooling instead of CLS (a second variable); any other backbone; fine-tuning; a label-adaptive crop (label leakage); re-tiling at a new physical size (preprocessing change). Each is a legitimate later experiment and each is out of scope here.

---

## 5. File structure and responsibilities

```
Model 2/
└── Model 2 Experiment 2/
    ├── instructions.txt                E2 scope, the bounded-sweep finding, arms,
    │                                   predictions, gate. Written before code.
    ├── Model2_Experiment2.ipynb        generated, not hand-edited
    ├── Experiment2.txt                 generated by the notebook from live objects
    ├── Submission_Model2_E2.csv        generated only if the gate fires
    ├── crop_geometry.py                THE CROP. Pure numpy, no torch. Owns the
    │                                   centred-window extraction and the patch-mm
    │                                   arithmetic, so both are testable locally.
    ├── check_crop_geometry.py          contract test for crop_geometry.py, runs
    │                                   locally with no torch
    └── kaggle_setup.md                 what to attach, what to run

scratch/
├── build_m2e2.py                       notebook generator — single source of cell text
└── mk_run3.py                          regenerate + flatten to run_m2e2.py + compile-check
```

`crop_geometry.py` exists for the same reason `backbones.py` did: it is the one new piece of logic that must be right, and putting it in a module makes it testable without torch and impossible to silently diverge between the notebook and the checks. It is small on purpose.

`backbones.py` is **imported from E1's folder, unmodified.** The notebook resolves it by search, exactly as E1 does.

---

## Task 1: `crop_geometry.py` — the new variable, isolated and testable

**Files:**
- Create: `Model 2/Model 2 Experiment 2/crop_geometry.py`

**Interfaces:**
- Produces:
  - `CROPS: Tuple[int, ...] = (256, 128, 64, 32)`
  - `INPUT_SIZE: int = 224`, `PATCH_PX: int = 14` — the backbone's input geometry, mirrored
    here rather than imported from `backbones` so this module has no torch-side dependency
    and stays testable on the dev machine.
  - `def crop_tile(rgb: np.ndarray, c: int) -> np.ndarray` — centred square crop of side `c` from an `(N,H,W,3)` batch; returns `(N,c,c,3)`, dtype preserved, no copy when `c == H`.
  - `def patch_mm(c: int, ppm: float, input_size: int = INPUT_SIZE, patch: int = PATCH_PX) -> float` — physical footprint of one patch after `c → input_size` resize.
  - `def window_mm(c: int, ppm: float) -> float`
  - `def resolvability(d50_mm: np.ndarray, c: int, ppm: float, divisor: float = 3.0) -> int` — count of soils whose median grain is at least `1/divisor` of a patch.

- [ ] **Step 1: Write the module** with the docstring stating why it exists separately (testable without torch; the notebook and the checks cannot diverge) and the E1 correction from §2 written out in full.

```python
"""The one new variable in Model 2 E2: how much physical soil a tile contributes.

E1 tested a frozen ViT whose patch spans 3.51 mm against soils whose median grain is
often under 0.1 mm, and declared that confound in advance. E2 varies it.

This lives in a module rather than in notebook cells for the same reason backbones.py
does: it is the single piece of new logic that has to be right, and a module can be
tested without torch. The development machine has none.

A CORRECTION, carried here so it is not silently inherited. E1's backbones.PATCH_MM is
14 / 4.552516 = 3.0752 mm. That omits the 256 -> 224 resize: a 256 px tile becomes a
56.23 mm window squeezed into 224 px, so one 14 px patch covers 3.5145 mm, 14.3% more
than recorded. backbones.PATCH_MM is left untouched - E1's record must stay byte-
reproducible - and every E2 number is computed from patch_mm() below.
"""
```

- [ ] **Step 2: Implement `crop_tile`** with the centred-window arithmetic and the no-op fast path.

```python
def crop_tile(rgb, c):
    a = np.asarray(rgb)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {a.shape}")
    h, w = a.shape[1], a.shape[2]
    if c <= 0 or c > min(h, w):
        raise ValueError(f"crop {c} does not fit a {h}x{w} tile")
    if c % 4:
        raise ValueError(f"crop {c} is not a multiple of 4; the mock backend box-downsamples by 4")
    if c == h == w:
        return a
    o_y, o_x = (h - c) // 2, (w - c) // 2
    return a[:, o_y:o_y + c, o_x:o_x + c, :]
```

- [ ] **Step 3: Implement the geometry helpers.**

```python
def patch_mm(c, ppm, input_size=INPUT_SIZE, patch=PATCH_PX):
    """One patch's footprint in mm: (window mm / input px) * patch px."""
    return (c / ppm) / input_size * patch


def window_mm(c, ppm):
    return c / ppm


def resolvability(d50_mm, c, ppm, divisor=3.0):
    return int((np.asarray(d50_mm, float) >= patch_mm(c, ppm) / divisor).sum())
```

- [ ] **Step 4: Confirm the module imports with no torch present.** Run `python -c "import crop_geometry"` from the experiment folder; it must succeed on this machine. The test in Task 2 asserts the import surface contains no `torch` reference by scanning the source, rather than adding a runtime guard that would itself be untested.

## Task 2: `check_crop_geometry.py` — the contract test that runs locally

**Files:**
- Create: `Model 2/Model 2 Experiment 2/check_crop_geometry.py`

**Interfaces:**
- Consumes: everything Task 1 produces.
- Produces: a pass/fail script the notebook also shells out to before extracting anything, matching E1's `check_backbones.py` pattern.

- [ ] **Step 1: Write the assertions.** Shape and centring, dtype preservation, the `c == 256` identity, the multiple-of-4 guard, and the numeric facts §1 measured — so a future edit that changes the sweep cannot pass unnoticed. The file uses the same plain-`assert`, printed-PASS/FAIL, non-zero-exit-on-failure style as `Model 2/Model 2 Experiment 1/check_backbones.py`, because pytest is not installed locally. It also asserts `"torch" not in open("crop_geometry.py").read()`.

```python
def test_centred_and_exact():
    a = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(1, 8, 8, 3)
    c4 = cg.crop_tile(a, 4)
    assert c4.shape == (1, 4, 4, 3)
    assert np.array_equal(c4[0, 0, 0], a[0, 2, 2])          # centred: offset (8-4)//2 = 2
    assert c4.dtype == a.dtype
    assert np.array_equal(cg.crop_tile(a, 8), a), "full-size crop must be the identity"


def test_geometry_matches_the_measured_sweep():
    """These are the numbers §1 of the plan measured on real tiles. If they move, the
    sweep changed and the experiment is a different one."""
    ppm = 4.552516
    want = {256: 3.5145, 128: 1.7573, 64: 0.8786, 32: 0.4393}
    for c, mm in want.items():
        assert abs(cg.patch_mm(c, ppm) - mm) < 5e-4, (c, cg.patch_mm(c, ppm), mm)


def test_e1_recorded_number_is_the_wrong_one():
    """Pinned so nobody 'fixes' it back. E1 recorded 3.0752 mm by omitting the resize."""
    ppm = 4.552516
    assert abs(14 / ppm - 3.0752) < 1e-3
    assert abs(cg.patch_mm(256, ppm) - 14 / ppm) > 0.4      # the gap is real and material


def test_reachability_is_bounded():
    """Magnification cannot resolve the fines. Assert the flatness that makes E2's
    expected effect small, so a future reader cannot be surprised by it."""
    d50 = np.array([0.021, 0.022, 0.030, 0.030, 0.035, 0.059, 0.062, 0.068, 0.099,
                    0.103, 0.211, 0.814, 1.795, 2.434, 2.535, 3.066, 3.674, 4.337,
                    4.585, 4.791, 4.792, 5.339, 5.911, 6.126])
    ppm = 4.552516
    got = {c: cg.resolvability(d50, c, ppm) for c in cg.CROPS}
    assert got[256] == 12 and got[32] == 14, got
    assert got[32] - got[256] <= 3, f"the sweep resolves at most 3 more soils: {got}"
```

- [ ] **Step 2: Run it.** `python check_crop_geometry.py` from the experiment folder. Expected: `CHECK CROP GEOMETRY: PASSED`.
- [ ] **Step 3: Confirm it fails loudly on a wrong crop.** Temporarily change `CROPS` to include 200 and re-run; expect a failure on the geometry table. Revert.

## Task 3: The notebook generator

**Files:**
- Create: `scratch/build_m2e2.py`
- Create: `scratch/mk_run3.py`

**Interfaces:**
- Consumes: `crop_geometry` (this experiment), `backbones` (E1's folder, unmodified), E1's `Submission_Model2_E1.csv` and `cv_per_soil.csv` for the regression check, E3's `features_soil.csv` / `cv_families.csv` / `Submission_Model1_E3.csv`.
- Produces: `Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb` and `scratch/run_m2e2.py`.

Sections, in order. Cells are numbered in the generator, not here.

- **A setup** — config (`CROPS`, arms, the E1 alpha grid unchanged, `probe_arm = None`), imports, `resolve_input_root`, `resolve_sibling("Model 2 Experiment 1")` for `backbones.py`, `crop_geometry` import, `config_hash` pin, seed. **Local mock runs route to `local mock dry-run/`.**
- **B data & metric** — manifests, labels, metric reconciliation, the six external ground-truth scores (E1 172.70, baseline 102.37, E2 71.27, E3 61.24, E4 77.12, **M2E1 probe 60.56167**).
- **C prior results restated** — load E1's artifacts, print E1's verdict table, print the §1 reachability table computed live, print the PATCH_MM correction with both numbers.
- **D crop geometry checks** — shell out to `check_crop_geometry.py`; then verify on real tiles that centre crops are soil-representative (density ratio vs the whole tile, count below 0.5 soil) and abort if that assumption fails.
- **E environment probe** — torch/timm/CUDA, resolved weights, resolved backbones path.
- **F embeddings** — for each crop × each backbone arm, crop then `embed_tiles`. Cache key **must include the crop**: `embed_<backend>_c<crop>_<tile_size>_<config_hash>_<seed>_<view>.npy`. Assert crop 256's cache equals E1's file byte-for-byte when present — that is the cheapest possible proof the crop path is a no-op at full size.
- **G aggregation** — tile→image→soil median per (arm, crop), identical code path to E1; assert the M1 control reproduces E3 to < 1e-6.
- **H regression check** — crop 256 must reproduce E1's in-domain table (M1 43.02, D 44.33, R 39.78) within 0.05 EMD. **This gate stops the notebook.**
- **I in-domain scoring** — nested LOGO-CV per (arm, crop) per alpha; the E4 parsimony rule for final alphas; `cv_per_fold.csv`, `cv_per_soil.csv`, `arms_config.csv`.
- **J the sweep as a trend** — Spearman ρ between crop px and D's in-domain error, and between crop px and R's. A trend is the pre-registered primary read; picking the best of four crops is a multiple-comparison trap.
- **K per-size-group analysis** — split the 24 soils by D50 into fines (<0.11 mm), mid (0.11–1.5 mm) and coarse (>1.5 mm) and report the paired crop-256-vs-crop-32 difference per group. **This is where E2's answer actually lives.** A gain that is uniform across groups, or concentrated in the fines, contradicts §1 and must be reported as suspicious.
- **L pretraining ablation at each crop** — paired D−R with soil bootstrap, per crop.
- **M mechanism** — camera shift z and blur/soil distance per (arm, crop); does magnification cost D its camera stability?
- **N report-only ruler** — CAM+RES per (arm, crop), with the standing header that it selects nothing, plus the ruler's calibration points.
- **O gate** — §6 executed as code.
- **P submission** — only if the gate fires; the eight gates always run as a labelled plumbing check, as in E1.
- **Q error analysis** — sweep curve plot, per-group strip, the D-vs-M1 per-soil difference at each crop.
- **R record** — `Experiment2.txt` from live objects, including the reachability table and the regression-check outcome.

## Task 4: Generate, dry-run locally under mock, and verify

- [ ] **Step 1:** `python scratch/mk_run3.py` — regenerates the notebook, flattens to `scratch/run_m2e2.py`, `compile()`-checks it.
- [ ] **Step 2:** `BACKEND=mock python scratch/run_m2e2.py`. Expected: completes; writes into `local mock dry-run/`; mock in-domain at or worse than 82.62 at **every** crop; no submission; the crop-256 regression check passes for M1.
- [ ] **Step 3:** Determinism — run twice, compare every emitted file byte for byte.
- [ ] **Step 4:** Confirm the submission path executes even though nothing is written (E1's lesson: a gate-conditional section is untested code).
- [ ] **Step 5:** `python -m preprocess.final_check` PASSED; nothing under `data/Training`, `data/Test`, `data/processed_meta` modified.
- [ ] **Step 6:** Read the mock log for any cell whose output is empty — an empty section means a loop that never ran, which is exactly how E1's `ndarray.median()` bug survived.

## Task 5: Kaggle run and readout

- [ ] **Step 1:** Reuse `final_kaggle_upload_m2e1.zip` **only if** nothing packaged changed; otherwise rebuild via `scratch/make_kaggle_zip.py` and hash-verify the members. E2 adds no packaged file, so the existing dataset should be reusable — verify, don't assume.
- [ ] **Step 2:** Upload `Model2_Experiment2.ipynb`, attach the one data-bearing dataset, GPU T4 x2, internet ON for the weights download, no `BACKEND` variable (defaults to `both`).
- [ ] **Step 3:** Save Version → Save & Run. Read cells H, I, J, K in that order.
- [ ] **Step 4:** Download every output into `Model 2/Model 2 Experiment 2/`, plus the executed notebook as evidence.

## Task 6: Record the verdict

- [ ] **Step 1:** Append E2's outcome to `Model 2/instructions.txt`, including whether Model 2 stays open.
- [ ] **Step 2:** Add any newly dead hypothesis to `Model 1/instructions.txt`'s cancelled list if it kills a direction.
- [ ] **Step 3:** Copy the public score into `Experiment2.txt` §P6 only if a submission was made.

---

## 6. The decision gate — fixed before the run, executed as code

```
submit = exists a crop c in {128, 64, 32} such that
           in_domain(D, c) < in_domain(M1)
         AND in_domain(D, c) < in_domain(R, c)
         AND the paired soil-bootstrap CI of (D - R) at c excludes zero with D better
         AND the gain at c is concentrated in the patch-limited D50 groups
```

The fourth clause is new and it is the point of the experiment. A gain that is uniform across soil sizes cannot be explained by magnification — magnification only changes anything for soils whose grain was below the patch scale. Requiring the gain to be *located* is what stops a lucky mean from reviving a dead premise.

| outcome | meaning | action |
|---|---|---|
| a crop satisfies all four clauses | E1's refutation was partly a scale artefact; learned features do help when the patch resolves the grain | submit; Model 2 continues with the corrected scale as the frozen default |
| D improves with magnification but not past R | the scale effect is real but it is not about *pretraining* | do not submit; Model 2 closes, and the crop finding is reported as a property of the tiles |
| no crop helps D at all | E1's refutation survives its declared confound | do not submit; **Model 2 is closed on evidence**, and the remaining time goes to the transfer penalty of Model 1's configuration |
| crop 256 fails the regression check | E2 is not comparable to E1 | void; fix the pipeline before reading any number |

Three of four outcomes close Model 2. That is the honest expectation given §1, and it is still worth the session: the fourth outcome would be important, and E1's hole stays open until it is tested.

## 7. Pre-registered predictions

Written before implementation. The blind/confirmation label is per prediction and may not be edited afterwards.

| # | prediction | blind? |
|---|---|---|
| P1 | mock in-domain LOGO-CV is at or worse than the no-image floor 82.62 at **every** crop | no — plumbing correctness |
| P2 | crop 256 reproduces E1's in-domain table within 0.05 EMD | no — regression check; if it fails E2 is void |
| P3 | D's in-domain error **decreases** monotonically as crop shrinks 256→32 (Spearman ρ > 0 with crop px) | **YES** |
| P4 | the improvement from crop 256 to crop 32 is **larger for the coarse and mid groups than for the fines** | **YES** — the mechanism prediction, and the one that distinguishes a real scale effect from noise |
| P5 | R's in-domain error also decreases with magnification, because its feature collapse at 256 px is itself a scale artefact | **YES** — if this holds and P3 holds equally, the "gain" is not about pretraining |
| P6 | D's camera/soil distance ratio stays below 0.35 at every crop (magnification does not cost it the stability E1 measured) | **YES** — diagnostic |
| P7 | no crop lets D beat both M1 and R with a significant CI | **YES** — this is the expected result given §1, and it closes Model 2 |

P7 is the prediction I actually expect to hold. Stating it plainly is the point: this experiment is designed to be refutable in the direction that would matter, not to be vindicated.

## 8. Assumptions

- Kaggle's image still has torch 2.10.0 and timm 1.0.26, and `{'img_size': 224, 'dynamic_img_size': True}` is still the accepted construction. E1 verified both; a platform update is the risk. Cell E re-probes and cell F re-runs `check_backbones.py` against the real arms before any tile is processed.
- The 256 px tiles are the only materialised size. Confirmed from `manifest_tiles.csv`: 16,228 rows across four sizes, 1,976 materialised, all at 256.
- A centred crop is representative of its tile. Measured in §1 (density ratio 1.005–1.009, no crop below 0.5 soil in 240 sampled tiles) and re-asserted in cell D on every tile before extraction.
- The public 3-soil subset is unchanged across all six submissions. Untestable from our side; the largest standing assumption in the project.
- DINOv2 weights remain reachable. E1 fetched them from the hub with internet on.

## 9. Risks

1. **The effect may be physically capped at ~2 soils.** §1 is not a guess — the sweep resolves 12→14 soils. A null is the most likely outcome and must be reported as a clean answer, not a failed experiment.
2. **Sample area collapses.** A 32 px crop uses 1.6% of the tile. Test soils already average 36 tiles against training's 64, so the effective sample per test soil shrinks hard. Expect wider intervals at fine crops; report them rather than normalising them away.
3. **Four crops × two contrasts is multiple-comparison surface.** Mitigated by making the *trend* (P3/P5) and the *per-group location* (P4) primary and the best-of-four a secondary read. The gate's fourth clause exists specifically to stop a cherry-picked crop.
4. **R may improve for the wrong reason.** If R's collapse at 256 px un-collapses at 64 px, R could gain as much as D and the pretraining claim dies cleanly. P5 is written to catch exactly that.
5. **Upsampling is not resolution.** A 32 px window stretched to 224 is interpolation. It changes what the patch grid sees, which is the legitimate mechanism under test, but it does not add information and the record must not imply otherwise.
6. **No local verification of the backbone path**, unchanged from E1. Mitigated by the mock arm covering every crop and by cell H's regression check.

## 10. What this decides

- **Gate fires** → Model 2 continues, the corrected crop becomes frozen, and E3 becomes the M1 + D ensemble at the right scale.
- **D gains but not past R** → the pretrained line closes; the magnification finding is still worth reporting because it applies to Model 1's texture features too.
- **Nothing gains** → E1's refutation stands after its one declared confound has been tested. **Model 2 closes**, and the project's remaining time goes where the measured leverage is: the +16 to +18 EMD transfer penalty on the 60.56/61.24 configuration, which is larger than every in-domain gain this project has ever achieved.

## 11. Cost

Local mock dry run: ~4 minutes, CPU, free.
Kaggle: 8 embedding passes over 1,976 real tiles plus 2 ruler views over 1,541 training tiles each ≈ 20,000 frozen forward passes of a 21M-parameter ViT-S/14 at batch 64. A few minutes on one T4. No training, no gradients, no checkpoints. Whole notebook inside 15 minutes including the nested sweeps. One session, no submission spent unless the gate fires.

---

## Sources

- Model 2 E1 record: `Model 2/Model 2 Experiment 1/Experiment1.txt`, `instructions.txt` (refutation, probe result, R-artefact analysis, mechanism table)
- Model 1 instrument rulings and cancelled hypotheses: `Model 1/instructions.txt`
- Measured geometry and reachability: computed 2026-09-28 from `data/processed_meta/manifest_samples.csv` labels and `data/tiles/` at `config_hash 010f44c36c74`; reproduced by `check_crop_geometry.py`
- [timm/vit_small_patch14_dinov2.lvd142m](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
- [DINOv2 — facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
