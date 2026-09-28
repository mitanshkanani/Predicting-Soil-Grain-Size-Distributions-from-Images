# Kaggle setup — Model 2 / Experiment 1

This experiment **needs a GPU only optionally** but it **needs torch**, which the
development machine does not have. Locally only the `mock` arm can run; the two backbone
arms (`vit_random`, `dinov2`) run here.

---

## 1. What to upload — one zip, one dataset

Model 1 never needed Kaggle: E1–E4 ran locally and only the finished CSV went to the
competition page. Model 2 is the first experiment that needs the data **on** Kaggle, because
torch isn't installed locally. So there is no pre-existing Kaggle dataset to reuse — this zip
is the whole thing.

> **`final_kaggle_upload_m2e1.zip`** in the repository root — **278 MB**. Built by
> `python scratch/make_kaggle_zip.py`, which also round-trips the archive and proves all
> 1,976 tiles come out byte-identical (including the 75 Münster filenames carrying a
> non-ASCII u-umlaut, which is the one thing that could silently break cell B1).

### 1.1 The steps

1. Kaggle → **Datasets → New Dataset**
2. Title: `soil-gsd-m2-e1` · Visibility: **Private**
3. Drag `final_kaggle_upload_m2e1.zip` into the upload area
4. **Create**. Kaggle unzips it, so the dataset's file list should show `data/`,
   `preprocess/`, `backbones.py`, `check_backbones.py`, `Model 1 Experiment 3/` —
   **not** the zip itself
5. Open the notebook → Input panel → **Add Input → Your Datasets → `soil-gsd-m2-e1`**

That is the only attachment. Nothing else to pin, version or match.

### 1.2 What is in the zip, and what is deliberately left out

```
data/tiles/                 1,976 PNGs, 276 MB   <- every pixel the notebook reads
data/processed_meta/        manifest_images.csv, manifest_tiles.csv,
                            manifest_samples.csv, audit.json,
                            golden_checks.json, target_ppm.json
data/sample_submission.csv                         <- authoritative column names
preprocess/                 the package; `from preprocess import verify` is the
                            validated feature definition, reused not reimplemented
backbones.py                the torch surface
check_backbones.py          the interface contract test
Model 1 Experiment 3/       features_soil.csv, cv_families.csv,
                            Submission_Model1_E3.csv, .cache/tile_features_*.csv
```

Left out, with the reason measured rather than guessed — the notebook opens exactly two
kinds of file, tile images and the five manifest artifacts:

| excluded | MB | why |
|---|---|---|
| `data/training_down`, `data/testing_down` | 445 | never opened. Their paths are join keys in `manifest_images.csv` only |
| `data/Training`, `data/Test` | — | raw sources. Model notebooks are forbidden to read them; their absence is the guarantee |
| `data/qc` | 230 | contact sheets, nothing reads them |
| `data/processed_meta/masks` | — | E1 never reads masks; per-tile `soil_fraction` is already in the tile manifest |
| `Training_labels_updated.csv`, `ppm_updated.csv` | — | labels come from `manifest_samples.csv` only. Shipping a second label file is how two experiments end up disagreeing |

The `.cache` tile features matter: they let cell F3 prove the control arm reproduces E3
against **E3's own saved extraction** rather than a fresh one.

### 1.3 DINOv2 weights — pick ONE

| option | how | internet |
|---|---|---|
| **A. hub download** (simplest, start here) | leave `DINO_WEIGHTS` unset; timm fetches `vit_small_patch14_dinov2.lvd142m`, about 90 MB, once | **ON** |
| **B. attached weights** (most reproducible) | add the timm weight file to the same dataset and set `DINO_WEIGHTS` to its filename | OFF |

If you set `DINO_WEIGHTS` to a bare filename, cell D1 finds it by searching `/kaggle/input`
and asserts it exists **before** any extraction starts. Option B is better long-term —
pinned, cannot fail mid-run, keeps the notebook internet-free — but A costs one switch and
no extra upload, so use A first.

---

## 2. Notebook settings

| setting | value |
|---|---|
| Accelerator | **None** works, **T4 x2** is faster. About 8,500 frozen forward passes of a 21M-param ViT-S/14 at batch 64. On CPU expect a few minutes; on a T4 well under one. |
| Internet | **ON only for weights option A**, otherwise Off |
| Language / kernel | Python 3 |
| Persistence | **On** - the `.npy` embedding caches are the expensive artifact |
| Datasets | `soil-gsd-m2-e1` (section 1.1) — the one attachment, everything in it |
| Environment variables | see section 3 |

### 2.1 Getting the notebook itself onto Kaggle

Model 1 never went to Kaggle as a notebook — it ran locally and only the CSV was submitted.
This is the first time the notebook itself has to run there, so:

1. **Kaggle → Your Work → New Notebook → upload notebook** (or the up-arrow in the notebook
   toolbar) and pick `Model 2/Model 2 Experiment 1/Model2_Experiment1.ipynb`.
2. Right-hand panel: **Input → Add Input → Your Datasets → `soil-gsd-m2-e1`**.
3. **Settings** panel: Accelerator, Internet, and under **Environment variables** add
   `BACKEND` (section 3).
4. **Save Version → Save & Run**, with the time limit set to 2-3 hours.

Do not paste `backbones.py` into a cell to avoid the attachment step. If the backbone code
exists in two places, the notebook and the module can disagree and the result is no longer
reproducible - that is the whole reason `backbones.py` is a separate file.

If `backbones.py` is missing from the attachment, cell A3 stops with a clear message rather
than substituting anything. There is no in-cell fallback, on purpose.

---

## 3. Environment variables

Set these in the notebook's **Environment** panel so they survive re-runs, rather than
editing cell A1.

| var | value | meaning |
|---|---|---|
| `BACKEND` | `vit_random` for run 1, `both` for run 2 | which backbone arms to execute. `mock` / `all` / a comma list also accepted. Unset defaults to `mock` locally and `both` on Kaggle. |
| `DINO_WEIGHTS` | filename or path, optional | offline weights. Unset = download from the hub. |

`BACKEND=both` matters: the gate needs **R and D in the same run**. With one arm only, cell
L1 reports `evaluable=False` and writes no submission — correct behaviour, wasted session.

---

## 4. Run order

**One run: `BACKEND=both`** (or leave it unset — that is the Kaggle default). It needs no
`BACKEND` value typed at all. The gate requires R and D in the same pass, and cell E1
constructs and checks both backbones **before** any tile is extracted, so a weights download
failure costs about 30 seconds, not a session.

If that single run dies in cell E1 or D1, fall back to `BACKEND=vit_random` — random weights
need no download, so it isolates "is the torch path sound" from "can we reach the weights".
Arm R from that run is still a real result; only the gate stays unevaluated.

If it dies on a missing `timm`: `pip install timm` in a cell added *above* cell A2, with
internet on. If `timm` cannot be installed at all, stop and report it — do **not** swap in a
different backbone. The plan's logic (pretrained vs same-architecture-random) survives a
different network, but not silently.

Cell O2 writes `Experiment1.txt` with the scope statement in §13 naming exactly which arms
ran, so a partial run is never mistaken for the result run.

---

## 5. What the first cells will tell you

| cell | what it proves | if it fails |
|---|---|---|
| A4 | `config_hash 010f44c36c74` matches the pinned dataset | wrong dataset version — stop, do not proceed |
| D1 | torch/timm versions, CUDA availability, the resolved weights path | `DINO_WEIGHTS` points at nothing, or timm absent |
| E1 | `check_backbones.py` run against the **real** backend: shape `(N,384)`, finiteness, L2 unit norms, determinism, batching | the interface contract is broken; fix `backbones.py`, do not continue |
| F3 | the control arm equals E3's `features_soil.csv` to < 1e-6 | drift here means Model 2 is not comparable to Model 1 and nothing later means anything |
| H4 | nested in-domain LOGO-CV per arm — **the primary table** | — |
| L1 | the gate, printed as clauses | `evaluable=False` means an arm is missing, not that the experiment failed |
| M2 | eight submission gates, then write, re-read, re-assert | refuses to write a malformed file |

Expected wall time for run 2: **8–12 minutes** on a T4, most of it tile I/O rather than the
network.

---

## 6. Reading the outcome

The gate was fixed before the run and is executed as code in cell L1:

```
submit = in_domain(D) < in_domain(M1)
         AND in_domain(D) < in_domain(R)
         AND the paired soil-bootstrap CI of (D - R) excludes zero with D better
```

| gate outcome | meaning | do this |
|---|---|---|
| **SUBMIT** | pretraining, not capacity, lowered in-domain error | submit `Submission_Model2_E1.csv`. Record the score in `Experiment1.txt` §10. P6 requires it to beat 61.24. Then Model 2 E2 attacks the physical scale of the patch. |
| **D beats M1 but not R** | it is dimensionality | do not submit. The finding is real and cheap: report it, and go to E2 (magnification) rather than deeper pretrained models. |
| **D does not beat M1** | representation is not the bottleneck | do not submit. The ~34 EMD in-domain headroom is a data limit at n=24; spend the remaining time hardening the 61.24 configuration. |
| **not evaluable** | an arm was missing | re-run with `BACKEND=both`. Not a result either way. |

`CAM+RES` appears in section K of the output. It is **reported and selects nothing** — E4
demoted it after it ranked E4 above E2 by 24.6 EMD and the leaderboard reversed them.

---

## 7. What to bring back and where it goes

Download the notebook's Output files into `Model 2/Model 2 Experiment 1/`, overwriting
nothing outside that folder:

`Experiment1.txt`, `cv_per_fold.csv`, `cv_per_soil.csv`, `alpha_selection_in_domain.csv`,
`paired_effects.csv`, `mechanism_table.csv`, `embedding_separation.csv`,
`camres_report_only.csv`, `predictions.csv`, `gate.json`, `features_soil_M1_control.csv`,
`cv_families.csv`, and `Submission_Model2_E1.csv` if it exists.

Also worth saving from `.cache/`: the `embed_*.npy` + `embed_*.paths.npy` pairs. They are
~6 MB each at float64 and re-extracting them costs a session.

Then: **do not tune anything against the public leaderboard.** It scores a fixed 3 of the 10
test soils and the final ranking is `0.30·public + 0.70·private`.

---

## 8. Verified locally, and what was not

Verified here (`BACKEND=mock`, two runs, all 18 emitted files byte-identical):

- the control arm reproduces E3's soil matrix to **2.84e-14** (gate < 1e-6);
- the texture/colour mechanism numbers reproduce E4's published values (colour camera shift
  **1.541 z** against E4's reported 1.54), so the mechanism table is on the same scale as
  Model 1's;
- the metric self-check reproduces the host's 100.31 trivial baseline;
- mock in-domain LOGO-CV **87.82**, worse than the no-image floor 82.62, and mock − M1
  **+44.80** EMD, CI [+31.81, +57.68], wins 3/24 — the pipeline does not manufacture signal
  out of nothing;
- the gate refuses to fire and **no submission is written** under mock, in both the mock run
  and a `BACKEND=vit_random` run on this machine (arm skipped as non-executable, exit 0);
- `python -m preprocess.final_check` PASSED afterwards; nothing under `data/Training`,
  `data/Test` or `data/processed_meta` was modified.

**The upload zip itself was tested, not assumed.** `scratch/make_kaggle_zip.py` round-trips
the archive and checks all 1,976 tiles extract byte-identical, including the 75 Münster
filenames carrying U+00FC. Then the notebook was run with that extracted directory as its
**only** input — no repo data, no other attachment — and it completed (exit 0), resolved
`INPUT_ROOT`, `backbones.py` and E3's folder from the extracted tree alone, reproduced E3 to
the same 2.84e-14, and produced tables **bit-identical (max abs difference 0.00e+00)** to the
run from the repository. That is the closest local stand-in for the Kaggle filesystem this
machine can give.

**Not verified anywhere yet:** the real embedding geometry, the timm CLS readout shape, and
every number involving arms R and D. Cell E1 exists specifically so those are checked before
1,976 tiles go through the backbone rather than after.
