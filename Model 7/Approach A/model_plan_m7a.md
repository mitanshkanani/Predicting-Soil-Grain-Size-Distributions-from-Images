# M7-A (Model 7 / Approach A / Experiment 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute four frozen within-tile spatial-organisation statistics (A1-A4), add them to the
frozen 12-feature pipeline, and decide M7-A by the pre-registered gates without touching any other
model's artifacts.

**Architecture:** One new module computes the statistics per tile from the materialised tile PNGs and
aggregates them by the pipeline's existing two median steps; one contract test pins their behaviour; a
separate gate module re-derives the alias pair list and proves it against Model 6-3's recorded
summary before any CV arm runs; a notebook generator emits the experiment notebook, a runner executes
it headlessly, and static self-checks verify the emitted notebook matches this contract. The frozen
head is imported, never reimplemented.

**Tech Stack:** Python 3.14, numpy, pandas, scipy.ndimage, Pillow. No torch, no nbformat locally -
notebooks are generated as JSON by a builder script and executed by a plain runner, exactly as in
Models 1, 2 and 5.

**Spec:** `Model 7/Approach A/model_spec_m7a.md` (approved, frozen)
**Contract:** `Model 7/Approach A/instructions.txt`
**Evidence baseline:** `research_map_1_2_5_6.md`

## Global Constraints

Every task implicitly includes all of these. They are copied from the contract and the spec.

- K = 4 columns, ids A1-A4, exactly as defined in the spec. No fifth statistic, no scale sweep, no
  free parameter per statistic, no change to a definition after the run.
- G0 anchor: the pipeline **without** the new block reproduces **43.0217308796477** with maximum
  deviation **0.00e+00**, before any arm is scored.
- G1 success bar: mean family-honest internal CV EMD reduction **>= 3.00 EMD**.
- Primary validation is **leave-one-CV-family-out** over the 16 families and 24 soils. A scored
  soil's own family is never in its fitting set, and alpha is re-selected inside the remaining
  families for every fold. Paired differences use a family-clustered paired bootstrap, 4000 draws.
- Controls, all four required: P-RAND (phase-scrambled tiles), P-SHUF (per-column shuffle), P-COLS
  (linear combinations of the existing 12), P-SEL (best-of-4 permutation null, measured for K=4).
- Gates G0-G7 and the verdict precedence are those in the contract. SUBMIT = TRUE if and only if
  VERDICT == SUPPORTED.
- Never modify or rename `data/Training/`, `data/Test/`, `data/processed_meta/`, or any previous
  experiment's results or notebook. Tiles are read, never regenerated. No preprocessing re-run.
- No camera, device, ppm, EXIF, ICC, site or sample-id as a feature, a selection key or a dispatch
  key. `ppm` may be used only as the tile's physical scale inside a statistic definition, never as an
  input column or a grouping key.
- Internal CV EMD and external Kaggle EMD (baseline **60.56167**) are different scales. Never compare
  them arithmetically; never put both on one line of a printout.
- CAM+RES is ordinal-only and may not select anything. The 15 EMD ordinal floor may not be used to
  resolve a smaller difference.
- No tuning, feature expansion or post-hoc selection after seeing the CV result. No submission unless
  the pre-registered criteria are satisfied.
- Printed output is ASCII-only. `->` not an arrow glyph, `>=` not a math glyph.
- Run everything on local CPU. No Kaggle run is assumed.
- Do not revive the withdrawn leaky 1-NN bound (41.95 / 44.23) or the withdrawn 1-1.5 EMD cap.
- M7-C stays unrun. M7-T stays suspended. M7-B and M7-E are not revived.

---

## File structure

| file | responsibility |
|---|---|
| `Model 7/Approach A/spatial_features.py` | compute A1-A4 on one tile; the phase-scrambled and shuffled placebos; aggregation to image and soil. Knows nothing about labels, cameras or the run |
| `Model 7/Approach A/check_spatial_features.py` | contract test for the module above: behaviour, invariances, placebo sanity, determinism |
| `Model 7/Approach A/alias_gate.py` | deterministic pair-set derivation plus the separability gate. Refuses to run if the derivation disagrees with Model 6-3's recorded summary |
| `Model 7/Approach A/m7a_eval.py` | the family-honest LOFO arm runner built on the frozen head, the four controls, the best-of-4 null, the bootstrap and the verdict ladder |
| `scratch/build_m7a1.py` | generates `Model7_ApproachA_Experiment1.ipynb` |
| `scratch/run_nb_m7a1.py` | executes the shipped notebook cell by cell and writes outputs to `Model 7/Approach A/results/run1/` |
| `scratch/mk_run7.py` | static self-checks on the built notebook: compile per cell, cell-list pinning, thresholds-appear-once, contract-agreement markers, banned patterns, free-name scan |
| `scratch/check_m7a_result.py` | independent re-verification of the finished run's artifacts against the contract |

The frozen pipeline is **imported**, never retyped: `Model 5/Model 5 Experiment 1/transfer_eval.py`
supplies `FEATURES`, `ALPHAS`, `PC_RANK`, `image_features`, `soil_from_images`, `labels_matrix`,
`fit_predict`, `nested_predict`, `emd_pair`, `project`, `no_holdout_control_score`,
`clustered_bootstrap`.

---

### Task 1: A1-A4 per-tile statistics

**Files:**
- Create: `Model 7/Approach A/spatial_features.py`
- Test: `Model 7/Approach A/check_spatial_features.py`

**Interfaces:**
- Consumes: `preprocess.verify._illumination_base(g, sigma_px)`, `preprocess.verify.BASE_SIGMA_MM`,
  `preprocess.verify.BAND_SIGMA_MM` (already used by the frozen 12).
- Produces: `SPATIAL = ("A1", "A2", "A3", "A4")`;
  `tile_spatial(a: np.ndarray, ppm: float) -> dict[str, float]` taking an HxWx3 float RGB tile and
  returning exactly the four keys; `mask_and_field(a, ppm) -> (g, m, e)` for reuse by the placebo.
  There is no scale re-derivation anywhere in this module: the tile's ppm is always the caller's.

- [ ] **Step 1: Write the failing test**

Add to `Model 7/Approach A/check_spatial_features.py`:

```python
"""Contract test for spatial_features.py. Run: python check_spatial_features.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_features as sf  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def striped(width=256, height=256, period=8):
    """A tile with stripes along x: strongly anisotropic, and orientation-flippable.

    The broadcast_to is not decoration: `128.0 + 40.0 * cos(...)` has shape (H, 1), and dstack of
    three (H, 1) arrays gives (H, 1, 3), on which np.gradient raises "Shape of array too small".
    Verified 2026-09-29 while checking this task before dispatch.
    """
    y = np.arange(height)[:, None]
    row = 128.0 + 40.0 * np.cos(2 * np.pi * (y % period) / period)
    g = np.broadcast_to(row, (height, width)).copy()
    return np.dstack([g, g, g])


def blobs(seed=0, size=24):
    rng = np.random.default_rng(seed)
    a = np.full((256, 256, 3), 120.0)
    for _ in range(30):
        cy, cx = rng.integers(0, 256, 2)
        yy, xx = np.ogrid[:256, :256]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (size / 2) ** 2
        a[m] += 60.0
    return a


def test_returns_four_keys_in_order():
    out = sf.tile_spatial(striped(), 4.552516)
    check("returns exactly A1-A4", tuple(out) == sf.SPATIAL, str(tuple(out)))
    check("all four are finite floats",
          all(isinstance(v, float) and np.isfinite(v) for v in out.values()))


def test_orientation_is_detected():
    """A1 is an anisotropy MAGNITUDE, so it is orientation-INVARIANT by construction.

    The plan's first draft asserted that a striped tile and its transpose differ on A1. Measured
    2026-09-29 before dispatch: both give exactly -33.761250, difference 0.000e+00, so that
    assertion could never have passed. What the statistic must actually do is (a) ignore WHICH
    axis is dominant and (b) separate a striped tile from a near-isotropic one (measured
    difference 33.74). A2 is the directional one, so it must flip sign under transpose.
    """
    horiz = sf.tile_spatial(striped(), 4.552516)
    vert = sf.tile_spatial(np.transpose(striped(), (1, 0, 2)).copy(), 4.552516)
    iso = sf.tile_spatial(blobs(), 4.552516)
    check("A1 is invariant under transposing the tile",
          abs(horiz["A1"] - vert["A1"]) < 1e-9, "%.6f vs %.6f" % (horiz["A1"], vert["A1"]))
    check("A1 separates a striped tile from a near-isotropic one",
          abs(horiz["A1"] - iso["A1"]) > 0.5, "%.4f vs %.4f" % (horiz["A1"], iso["A1"]))
    check("A2 flips sign under transpose, because it is directional",
          abs(horiz["A2"] + vert["A2"]) < 1e-6 and abs(horiz["A2"] - vert["A2"]) > 1e-6,
          "%.4f vs %.4f" % (horiz["A2"], vert["A2"]))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_spatial_features.py`
Expected: `ModuleNotFoundError: No module named 'spatial_features'`

- [ ] **Step 3: Write the module**

Create `Model 7/Approach A/spatial_features.py`:

```python
"""spatial_features.py - the four within-tile statistics M7-A adds, and nothing else.

This module deliberately knows no labels, no camera, no file paths and no experiment. It takes a tile
as a float array and returns four numbers. That narrow surface is what makes the "no leakage" claim
checkable rather than aspirational, and it is the same reason Model 5's alignment.py was isolated.

The mask and the grey field are taken from the FROZEN preprocessing definitions
(preprocess/verify.py:105-128) so that A1-A4 are computed on exactly the pixel set the existing 12
features are computed on:

    g = mean over channels
    m = g > percentile(g, 12)
    e = residual after removing an illumination base of sigma 12 mm

Every statistic is a function of a tile's own pixels. Nothing here reads another tile, a position, or
a label.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "preprocess") not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess.verify import BASE_SIGMA_MM, _illumination_base  # noqa: E402

SPATIAL: Tuple[str, ...] = ("A1", "A2", "A3", "A4")
A3_REFERENCE_LAG = 0.5      # range expressed as a fraction of the tile's own extent: dimensionless
A4_LAG_MM = 1.0             # the spec's fixed ~1 mm lag
EPS = 1e-12


def mask_and_field(a: np.ndarray, ppm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grey field, mask and illumination residual, using the tile's own recorded scale.

    ppm comes from manifest_tiles.normalized_ppm, never from a camera or device name. Deriving it
    from the tile extent instead would silently assume every tile is 256 px at 4.5525 px/mm, and the
    statistics are all in mm, so that assumption would be load-bearing.
    """
    a = np.asarray(a, float)
    ppm = float(ppm)
    if ppm <= 0.0:
        raise AssertionError(f"ppm must be positive, got {ppm}")
    if a.ndim != 3 or a.shape[2] != 3:
        raise AssertionError(f"expected HxWx3 tile, got {a.shape}")
    g = a.mean(axis=2)
    m = g > np.percentile(g, 12)
    if m.sum() < 16:
        raise AssertionError(f"degenerate mask: only {int(m.sum())} pixels survive the 12th pct")
    base = _illumination_base(g, BASE_SIGMA_MM * ppm)
    e = np.where(m, g - base, 0.0)
    return g, m, e


def _acf_along(field: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Normalised autocorrelation of the masked field along one axis, lag 0..N/2."""
    f = np.where(mask, field, np.nan)
    prof = np.nanmean(f, axis=1 - axis)
    prof = np.where(np.isfinite(prof), prof, 0.0)
    prof = prof - prof.mean()
    n = len(prof)
    ac = np.array([float(np.dot(prof[:n - k], prof[k:]) / (np.dot(prof, prof) + EPS))
                   if k < n else 0.0 for k in range(n // 2)])
    return ac


def _efold(ac: np.ndarray) -> float:
    """First lag at which the autocorrelation falls below 1/e, in samples (float, interpolated)."""
    thr = 1.0 / np.e
    for k in range(1, len(ac)):
        if ac[k] < thr:
            span = ac[k - 1] - ac[k]
            frac = (ac[k - 1] - thr) / span if span > EPS else 0.0
            return (k - 1) + float(frac)
    return float(len(ac))


def tile_spatial(a: np.ndarray, ppm: float) -> Dict[str, float]:
    g, m, e = mask_and_field(a, ppm)
    gy, gx = np.gradient(g)
    jxx = float(np.mean(gx[m] ** 2))
    jyy = float(np.mean(gy[m] ** 2))
    jxy = float(np.mean(gx[m] * gy[m]))
    tr, det = jxx + jyy, jxx * jyy - jxy ** 2
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = max(tr / 2.0 - np.sqrt(disc), 0.0)
    a1 = float(np.log((l2 + EPS) / (l1 + EPS)))

    ac0 = _acf_along(e, m, axis=1)
    ac90 = _acf_along(e, m, axis=0)
    a2 = float(np.log((_efold(ac0) + EPS) / (_efold(ac90) + EPS)))

    range_mm = _variogram_range(g, m, ppm)
    extent_mm = g.shape[0] / ppm          # px -> mm at the tile's own recorded scale
    a3 = float(range_mm / (A3_REFERENCE_LAG * extent_mm))

    lag_px = int(round(A4_LAG_MM * ppm))
    a4 = float(_continuity(g, m, lag_px) - _continuity(g, m, lag_px, diag=True))
    out = {"A1": a1, "A2": a2, "A3": a3, "A4": a4}
    if not all(np.isfinite(v) for v in out.values()):
        raise AssertionError(f"non-finite statistic in {out}")
    return out


def _variogram_range(g: np.ndarray, m: np.ndarray, ppm: float) -> float:
    """Lag where the masked variogram first reaches 90% of its own sill, in mm.

    Averaged over the horizontal and vertical directions only. No diagonal, no free lag count: the
    two directions are what an omnidirectional range needs and adding more would be a scale sweep.
    """
    h = np.arange(1, max(2, g.shape[0] // 2))
    gam = []
    for k in h:
        vals = []
        for ax in (0, 1):
            sl_a = [slice(None)] * 2; sl_a[ax] = slice(None, -k)
            sl_b = [slice(None)] * 2; sl_b[ax] = slice(k, None)
            keep = m[tuple(sl_a)] & m[tuple(sl_b)]
            d = g[tuple(sl_a)][keep] - g[tuple(sl_b)][keep]
            if d.size >= 8:
                vals.append(0.5 * float(np.mean(d ** 2)))
        gam.append(float(np.mean(vals)) if vals else (gam[-1] if gam else 0.0))
    gam = np.asarray(gam)
    sill = float(np.mean(gam[-max(1, len(gam) // 10):]))
    hit = np.argmax(gam >= 0.9 * sill) if sill > EPS and (gam >= 0.9 * sill).any() else len(gam) - 1
    return float(h[hit]) / ppm


def _continuity(g: np.ndarray, m: np.ndarray, lag_px: int, diag: bool = False) -> float:
    """Fraction of masked pairs at the fixed lag whose grey difference exceeds the tile's MAD."""
    if diag:
        sl_a, sl_b = (slice(None, -lag_px), slice(lag_px, None))
        pair = (g[sl_a, sl_a][m[sl_a, sl_a] & m[sl_b, sl_b]],
                g[sl_b, sl_b][m[sl_a, sl_a] & m[sl_b, sl_b]])
    else:
        keep = m[:, :-lag_px] & m[:, lag_px:]
        pair = (g[:, :-lag_px][keep], g[:, lag_px:][keep])
    d = np.abs(pair[0] - pair[1])
    if d.size == 0:
        return 0.0
    mad = float(np.median(np.abs(g[m] - np.median(g[m])))) + EPS
    return float(np.mean(d > mad))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "Model 7/Approach A" && python check_spatial_features.py`
Expected: both new checks PASS, exit 0.

- [ ] **Step 5: Commit**

```bash
git add "Model 7/Approach A/spatial_features.py" "Model 7/Approach A/check_spatial_features.py"
git commit -m "M7-A task 1: the four frozen within-tile statistics and their contract test"
```

---

### Task 2: placebo constructors and aggregation, with the invariances the controls depend on

**Files:**
- Modify: `Model 7/Approach A/spatial_features.py` (add functions)
- Modify: `Model 7/Approach A/check_spatial_features.py` (add tests)

**Interfaces:**
- Consumes: `tile_spatial`, `mask_and_field`.
- Produces: `phase_scrambled(a, seed) -> np.ndarray`, `shuffle_columns(mat: np.ndarray, seed) ->
  Tuple[np.ndarray, np.ndarray]`, `image_block(tile_rows: list[dict]) -> dict[str, float]`,
  `soil_block(image_rows: list[dict]) -> dict[str, float]`, `linear_combinations(X12, seed) ->
  np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
def test_phase_scramble_preserves_power_spectrum_and_marginals():
    a = blobs()
    s = sf.phase_scrambled(a, seed=7)
    def rp(img):
        f = np.abs(np.fft.fft2(img.mean(axis=2))) ** 2
        return np.sort(f.ravel())[:100]
    check("scramble keeps the pixel multiset (marginals exact)",
          np.allclose(np.sort(a.ravel()), np.sort(s.ravel()), rtol=1e-9))
    check("scramble keeps the power spectrum",
          np.allclose(rp(a), rp(s), rtol=1e-6))
    check("scramble is not a no-op", not np.allclose(a, s))


def test_column_shuffle_keeps_marginals_and_breaks_joint():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(20, 4))
    M[:, 1] = M[:, 0] * 2.0
    S, vecs = sf.shuffle_columns(M, seed=11)
    check("shuffled columns have identical per-column values",
          all(np.allclose(np.sort(M[:, j]), np.sort(S[:, j])) for j in range(4)))
    check("the planted correlation is destroyed",
          abs(np.corrcoef(S[:, 0], S[:, 1])[0, 1]) < 0.5)
    check("the permutation vectors are returned and reconstruct the shuffle",
          vecs.shape == M.shape and np.allclose(M[vecs[0], 0], S[vec_argsort(vecs[0]), 0])
          if False else vecs.shape == M.shape)


def test_aggregation_is_two_plain_medians_like_the_frozen_pipeline():
    rows = [{"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0},
            {"A1": 5.0, "A2": 6.0, "A3": 7.0, "A4": 8.0}]
    check("image_block medians over tiles", sf.image_block(rows)["A1"] == 3.0)
    check("soil_block medians over images", sf.soil_block([rows[0], rows[1]])["A2"] == 4.0)


def test_linear_combination_control_has_no_new_information():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 12))
    C = sf.linear_combinations(X, seed=13)
    check("P-COLS returns 4 columns", C.shape == (24, 4))
    check("every P-COLS column lies in the span of the original 12",
          all(np.linalg.matrix_rank(np.column_stack([X, C[:, j]])) == 12 for j in range(4)))
```

Delete the unused helper `vec_argsort` from the shuffle test above before running: the reconstruction
assertion is the one that reads `np.allclose(M[vecs[j], j], S[:, j])`, which is what the next step
implements.

- [ ] **Step 2: Run them to verify they fail**

Run: `cd "Model 7/Approach A" && python check_spatial_features.py`
Expected: FAIL with `AttributeError: module 'spatial_features' has no attribute 'phase_scrambled'`

- [ ] **Step 3: Implement the placebo and aggregation functions**

```python
def phase_scrambled(a: np.ndarray, seed: int) -> np.ndarray:
    """P-RAND: randomise FFT phase, then re-rank the field back onto the tile's own pixel values.

    Keeping the magnitude spectrum exactly, and the pixel multiset exactly, is the point: the only
    thing this placebo is missing is higher-order spatial organisation. Without the re-ranking step a
    plain phase scramble also Gaussianises the marginal, so a gain the placebo reproduces could be
    about the histogram rather than about arrangement.
    """
    a = np.asarray(a, float)
    out = np.empty_like(a)
    rng = np.random.default_rng(int(seed))
    for ch in range(a.shape[2]):
        g = a[:, :, ch]
        F = np.fft.fft2(g)
        phase = np.exp(2j * np.pi * rng.random(g.shape))
        phase[0, 0] = 1.0
        rec = np.real(np.fft.ifft2(np.abs(F) * phase))
        order = np.argsort(np.argsort(rec.ravel()))
        out[:, :, ch] = np.sort(g.ravel())[order].reshape(g.shape)
    return out


def shuffle_columns(mat: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """P-SHUF: permute every column independently. Marginals exact, joint structure destroyed.

    A ROW permutation would be inert here for the same reason it was inert in Model 5: a per-column
    median and a per-column distribution do not care about row order. The stacked permutation vectors
    are returned because a control nobody can reconstruct is not a control.
    """
    mat = np.asarray(mat, float)
    rng = np.random.default_rng(int(seed))
    vecs = np.array([rng.permutation(mat.shape[0]) for _ in range(mat.shape[1])])
    out = np.empty_like(mat)
    for j in range(mat.shape[1]):
        out[:, j] = mat[vecs[j], j]
    return out, vecs


def image_block(tile_rows) -> Dict[str, float]:
    """Median of a tile list -> one image row. The frozen first aggregation, same column set."""
    arr = np.array([[r[k] for k in SPATIAL] for r in tile_rows], float)
    med = np.nanmedian(arr, axis=0)
    return {k: float(med[i]) for i, k in enumerate(SPATIAL)}


def soil_block(image_rows) -> Dict[str, float]:
    """Median of image rows -> one soil row. The frozen second aggregation, same column set."""
    arr = np.array([[r[k] for k in SPATIAL] for r in image_rows], float)
    med = np.nanmedian(arr, axis=0)
    return {k: float(med[i]) for i, k in enumerate(SPATIAL)}


def linear_combinations(X12: np.ndarray, seed: int) -> np.ndarray:
    """P-COLS: four fixed random mixtures of the existing 12 columns. No new information by
    construction, so any gain this control also achieves is a feature-count effect."""
    X = np.asarray(X12, float)
    rng = np.random.default_rng(int(seed))
    W = rng.normal(size=(X.shape[1], 4))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return X @ W
```

Also replace the shuffle test's reconstruction assertion with the complete function below, so the test
file stays syntactically valid at every step of this task rather than mid-edit:

```python
def test_column_shuffle_keeps_marginals_and_breaks_joint():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(20, 4))
    M[:, 1] = M[:, 0] * 2.0
    S, vecs = sf.shuffle_columns(M, seed=11)
    check("shuffled columns have identical per-column values",
          all(np.allclose(np.sort(M[:, j]), np.sort(S[:, j])) for j in range(4)))
    check("the planted correlation is destroyed",
          abs(np.corrcoef(S[:, 0], S[:, 1])[0, 1]) < 0.5)
    ok = all(np.allclose(M[vecs[j], j], S[:, j]) for j in range(M.shape[1]))
    check("the permutation vectors reconstruct the shuffle", bool(ok))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "Model 7/Approach A" && python check_spatial_features.py`
Expected: all checks PASS, exit 0.

- [ ] **Step 5: Prove the placebos are capable of differing, then commit**

Run the two placebo sanity probes and record the numbers in the test file's docstring, because a
control that cannot move the statistic is worthless:

```bash
cd "Model 7/Approach A" && python -c "
import numpy as np, spatial_features as sf
from check_spatial_features import blobs
a=blobs()
print('real    ', sf.tile_spatial(a,4.552516))
print('scrambled', sf.tile_spatial(sf.phase_scrambled(a,seed=7),4.552516))"
```

Expected: the scrambled vector differs from the real one on A1/A2/A4 while A3 stays close, which is
what a phase-only scramble should do.

```bash
git add "Model 7/Approach A/spatial_features.py" "Model 7/Approach A/check_spatial_features.py"
git commit -m "M7-A task 2: placebo constructors, two-step median aggregation, invariance tests"
```

---

### Task 3: tile and image assembly, with the scale and mask checks the statistics depend on

**Files:**
- Create: `Model 7/Approach A/m7a_data.py`
- Test: `Model 7/Approach A/check_m7a_data.py`

**Interfaces:**
- Consumes: `Model 5/Model 5 Experiment 1/transfer_eval.py` -> `image_features`, `FEATURES`,
  `TILE_PX`, `MIN_SOIL_FRACTION`; `data/processed_meta/manifest_tiles.csv`; tile PNGs under the
  discovered input root.
- Produces: `find_input_root() -> Path`; `tile_table(root) -> pd.DataFrame` of qualifying tiles with
  **no camera column derived or carried**; `spatial_image_table(root, scrambled=False, seed) ->
  pd.DataFrame` indexed by `parent_image_path` with columns `A1-A4` plus `sample_id`;
  `spatial_soil_table(root, scrambled=False, seed) -> pd.DataFrame`; `NAN_REPORT: dict`.

- [ ] **Step 1: Write the failing test**

```python
"""Contract test for m7a_data.py. Run: python check_m7a_data.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Model 5" / "Model 5 Experiment 1"))
import m7a_data as d  # noqa: E402
import spatial_features as sf  # noqa: E402
import transfer_eval as tv  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def test_tile_selection_is_the_frozen_rule():
    t = d.tile_table(HERE.parents[1])
    check("only 256 px tiles", bool((t.tile_size_px == 256).all()))
    check("only materialized tiles", bool(t.materialized.astype(bool).all()))
    check("only soil_fraction >= 0.50 tiles", bool((t.soil_fraction >= 0.50).all()))
    check("1946 qualifying tiles, as measured for the pre-registration",
          len(t) == 1946, "%d" % len(t))


def test_image_table_row_count_matches_the_frozen_pipeline():
    img = d.spatial_image_table(HERE.parents[1])
    ref = tv.image_features(HERE.parents[1])
    check("same images as the frozen 12-feature assembly",
          sorted(img.index.astype(str)) == sorted(ref.index.astype(str)),
          "%d vs %d" % (len(img), len(ref)))
    check("columns are A1-A4 plus sample_id only",
          list(img.columns) == list(sf.SPATIAL) + ["sample_id"])
    check("no camera, ppm or sample-derived column is carried into the design",
          not any(c in img.columns for c in ("camera", "camera_fam", "normalized_ppm", "tile_x",
                                             "tile_y", "grid_col", "grid_row")))
    check("no NaN in any image-level statistic", not img[sf.SPATIAL].isna().to_numpy().any())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_m7a_data.py`
Expected: `ModuleNotFoundError: No module named 'm7a_data'`

- [ ] **Step 3: Implement the assembly**

```python
"""m7a_data.py - read the materialised tiles, compute A1-A4, aggregate exactly as the pipeline does.

Tiles are READ. Nothing here regenerates a tile, writes into data/, or re-runs preprocessing; the
qualifying-tile filter is copied from the frozen Model 5 loader so the pixel set is identical. The
only new computation is the four statistics in spatial_features.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "Model 5" / "Model 5 Experiment 1", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import spatial_features as sf  # noqa: E402
import transfer_eval as tv  # noqa: E402

NAN_REPORT: Dict[str, object] = {}


def find_input_root() -> Path:
    """The repository root, which is where data/ and the model folders live."""
    for cand in (ROOT, Path.cwd(), Path("/kaggle/input")):
        if (cand / "data" / "processed_meta" / "manifest_images.csv").exists():
            return cand
    raise AssertionError("no data/processed_meta/manifest_images.csv found from any known root")


def tile_table(root) -> pd.DataFrame:
    t = pd.read_csv(Path(root) / "data" / "processed_meta" / "manifest_tiles.csv")
    q = t[(t.tile_size_px == tv.TILE_PX) & t.materialized
          & (t.soil_fraction >= tv.MIN_SOIL_FRACTION)].copy()
    # dropped, not merely unused: the manifest carries camera and cv_group, and a design that cannot
    # see them cannot accidentally be selected on them later
    q = q.drop(columns=[c for c in ("camera", "camera_fam", "cv_group", "source_effective_ppm",
                                    "source_image_width", "source_image_height")
                        if c in q.columns])
    if "normalized_ppm" not in q.columns:
        raise AssertionError("normalized_ppm missing; the statistics have no physical scale")
    if q.empty:
        raise AssertionError("no qualifying tiles; the frozen filter did not match")
    return q


def _compute(root, q: pd.DataFrame, scrambled: bool = False,
             seed: int = 90001) -> List[Dict[str, float]]:
    rows = []
    nan = {"degenerate_mask": 0, "non_finite": 0}
    for r in q.itertuples():
        path = Path(root) / r.tile_path
        if not path.exists():
            raise AssertionError(f"manifest references a missing tile: {r.tile_path}")
        a = np.asarray(Image.open(path).convert("RGB"), float)
        if scrambled:
            a = sf.phase_scrambled(a, seed=seed + int(r.Index))
        try:
            out = sf.tile_spatial(a, float(r.normalized_ppm))
        except sf.TileDegenerate:
            # TileDegenerate is the ONLY thing a bad tile may raise. It is not an AssertionError and
            # not a ValueError, so a systemic error can never be logged here as degeneracy.
            nan["degenerate_mask"] += 1
            out = {k: float("nan") for k in sf.SPATIAL}
            rows.append(out)
            continue
        if not all(np.isfinite(v) for v in out.values()):
            nan["non_finite"] += 1
        rows.append(out)
    NAN_REPORT.update({"n_tiles": int(len(q)), **{k: int(v) for k, v in nan.items()},
                       "nan_rate": float(np.mean([not all(np.isfinite(x) for x in r.values())
                                                  for r in rows]))})
    return rows


def spatial_image_table(root, scrambled: bool = False, seed: int = 90001) -> pd.DataFrame:
    q = tile_table(root)
    q = q.assign(_row=_compute(root, q, scrambled=scrambled, seed=seed))
    out = {}
    # grouped by image and soil only, and camera is not in q at all by the time we get here
    for (img, sid), grp in q.groupby(["parent_image_path", "sample_id"]):
        out[img] = dict(sf.image_block(list(grp._row)), sample_id=str(sid))
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = "parent_image_path"
    df["sample_id"] = df["sample_id"].astype(str)
    return df[list(sf.SPATIAL) + ["sample_id"]]


def spatial_soil_table(root, scrambled: bool = False, seed: int = 90001) -> pd.DataFrame:
    """Soil rows via sf.soil_block, NOT a second pandas median.

    The tested aggregation is the only one allowed, otherwise the spec's claim that the block is
    "aggregated by the same median rules as the existing 12" would be checked by one function and
    executed by another.
    """
    img = spatial_image_table(root, scrambled=scrambled, seed=seed)
    out = {}
    for sid, grp in img.groupby("sample_id"):
        out[sid] = sf.soil_block([r for r in grp.to_dict("records")])
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = "sample_id"
    return df[list(sf.SPATIAL)]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "Model 7/Approach A" && python check_m7a_data.py`
Expected: both test functions PASS. The 1946 count and the image-set equality against
`tv.image_features` are the two facts that prove the pixel set is the frozen one.

- [ ] **Step 5: Record scale sanity, then commit**

```bash
cd "Model 7/Approach A" && python -c "
import m7a_data as d, numpy as np, pandas as pd
root=d.find_input_root()
q=d.tile_table(root)
print('ppm used:', sorted(q.normalized_ppm.round(4).unique())[:4])
print('tile_w_mm:', round(float(q.tile_w_mm.iloc[0]), 5))
s=d.spatial_soil_table(root)
print(s.describe().loc[['min','50%','max']].round(3).to_string())
print('nan report:', d.NAN_REPORT)"
```

Expected: `tile_w_mm 56.23264`, one unique ppm for the train split, and a NaN report whose
`degenerate_mask` count is 0 or is explicitly listed. Paste both outputs into the test file's
docstring, then:

```bash
git add "Model 7/Approach A/m7a_data.py" "Model 7/Approach A/check_m7a_data.py"
git commit -m "M7-A task 3: tile assembly on the frozen pixel set, image and soil aggregation"
```

---

### Task 4: G0 anchor gate, and the with-block arm that must reproduce it when the block is zeroed

**Files:**
- Create: `Model 7/Approach A/m7a_eval.py`
- Test: `Model 7/Approach A/check_m7a_g0.py`

**Interfaces:**
- Consumes: `transfer_eval.FEATURES`, `ALPHAS`, `fit_predict`, `emd_pair`, `clustered_bootstrap`,
  `no_holdout_control_score`; `m7a_data.spatial_soil_table`.
- Produces: `soil_design(root, block: str) -> (X: np.ndarray, Y: np.ndarray, fams: list[str],
  ids: list[str])` where `block` is one of `"none"`, `"real"`, `"rand"`, `"shuf"`, `"cols"`, `"zero"`;
  `lofo_errors(X, Y, fams) -> np.ndarray` (one honest error per soil, alpha chosen inside the training
  folds only); `ANCHOR = 43.0217308796477`; `g0_gap(root) -> float`.

- [ ] **Step 1: Write the failing test**

```python
"""G0 and G6 checks. Run: python check_m7a_g0.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Model 5" / "Model 5 Experiment 1"))
import m7a_eval as ev  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def test_anchor_reproduces_exactly():
    gap = ev.g0_gap(HERE.parents[1])
    check("G0 anchor 43.0217308796477 reproduced to 0.00e+00", gap == 0.0, "gap %.2e" % gap)


def test_design_shapes_and_leakage():
    X, Y, fams, ids = ev.soil_design(HERE.parents[1], "real")
    check("16 columns with the block", X.shape == (24, 16), str(X.shape))
    check("12 columns without it", ev.soil_design(HERE.parents[1], "none")[0].shape == (24, 12))
    check("labels are 24 x 11", Y.shape == (24, 11))
    check("24 distinct soils, 16 families", len(set(ids)) == 24 and len(set(fams)) == 16)
    check("finite everywhere", bool(np.isfinite(X).all() and np.isfinite(Y).all()))


def test_g6_inertness_zero_block_reproduces_anchor():
    X0, Y, fams, _ = ev.soil_design(HERE.parents[1], "zero")
    e0 = ev.lofo_errors(ev.soil_design(HERE.parents[1], "none")[0], Y, fams)
    ez = ev.lofo_errors(X0, Y, fams)
    check("zeroing the block reproduces the no-block arm exactly",
          np.array_equal(e0, ez), "max diff %.2e" % float(np.abs(e0 - ez).max()))
    check("and therefore the anchor", abs(float(ez.mean()) - 43.0217308796477) < 1e-9)


def test_every_soil_scored_once_with_no_family_in_its_fit():
    X, Y, fams, _ = ev.soil_design(HERE.parents[1], "real")
    per, seen = ev.fold_report(X, Y, fams)
    check("each soil appears in exactly one held-out fold",
          sorted(seen) == sorted(range(len(fams))))
    check("no fold ever fits on the family it scores", per["family_leaks"] == 0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_m7a_g0.py`
Expected: `ModuleNotFoundError: No module named 'm7a_eval'`

- [ ] **Step 3: Implement the design and the honest arm**

```python
"""m7a_eval.py - the family-honest arm, the four controls and the verdict ladder.

The head is imported, never reimplemented, so the only difference between the anchor and the gated arm
is the presence of four columns. That is what makes SUPPORTED-BELOW-BAND and
SELECTION-OR-CAPACITY-ARTEFACT distinguishable at all.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import m7a_data as d  # noqa: E402
import spatial_features as sf  # noqa: E402
import transfer_eval as tv  # noqa: E402

ANCHOR = 43.0217308796477
SEED_RAND, SEED_SHUF, SEED_COLS, SEED_PERM = 90001, 90002, 90003, 90004


def families() -> Dict[str, str]:
    t = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    return dict(zip(t.sample_id.astype(str), t.cv_family.astype(str)))


def soil_design(root, block: str = "none") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Returns (X, Y, fam_list, ids). fam_list is a per-soil list of CV-family labels used ONLY as a
    fold key. It is never concatenated into X, and no family label is derived from a device name."""
    ids, Y = tv.labels_matrix(root)
    ids = [str(i) for i in ids]
    X12 = tv.soil_from_images(tv.image_features(root)).loc[ids, list(tv.FEATURES)].to_numpy(float)
    if block == "none":
        X = X12
    elif block == "zero":
        X = np.hstack([X12, np.zeros((len(ids), len(sf.SPATIAL)))])
    elif block == "real":
        B = d.spatial_soil_table(root).loc[ids, list(sf.SPATIAL)].to_numpy(float)
        X = np.hstack([X12, B])
    elif block == "rand":
        Br = d.spatial_soil_table(root, scrambled=True, seed=SEED_RAND)
        X = np.hstack([X12, Br.loc[ids, list(sf.SPATIAL)].to_numpy(float)])
    elif block == "shuf":
        B = d.spatial_soil_table(root).loc[ids, list(sf.SPATIAL)].to_numpy(float)
        X = np.hstack([X12, sf.shuffle_columns(B, SEED_SHUF)[0]])
    elif block == "cols":
        X = np.hstack([X12, sf.linear_combinations(X12, SEED_COLS)])
    else:
        raise AssertionError(f"unknown block {block!r}")
    if not np.isfinite(X).all():
        raise AssertionError("non-finite design matrix")
    fam_of = families()
    fam_list = [fam_of[i] for i in ids]
    return X, Y, fam_list, ids


def lofo_errors(X, Y, fams, return_alphas=False):
    """Leave-one-CV-family-out: one honest error per soil, with the soil's whole CV family absent
    from its fitting set and alpha re-selected inside the remaining families only."""
    F = np.asarray(fams)
    out = np.full(len(X), np.nan)
    alphas = {}
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        alphas[g] = _alpha_train(X[tr], Y[tr], list(F[tr]))
        P = tv.fit_predict(X[tr], Y[tr], X[te], alphas[g])
        for k, i in enumerate(te):
            out[i] = tv.emd_pair(P[k], Y[i])
    if np.isnan(out).any():
        raise AssertionError("a soil was never scored")
    return (out, alphas) if return_alphas else out


def _alpha_train(Xtr, Ytr, Ftr) -> float:
    inner = sorted(set(Ftr))
    curve = {}
    for al in tv.ALPHAS:
        errs = []
        for F in inner:
            hold = np.array([i for i, f in enumerate(Ftr) if f == F])
            keep = np.array([i for i, f in enumerate(Ftr) if f != F])
            if not hold.size or not keep.size:
                continue
            P = tv.fit_predict(Xtr[keep], Ytr[keep], Xtr[hold], al)
            errs.extend(tv.emd_pair(P[k], Ytr[i]) for k, i in enumerate(hold))
        curve[al] = float(np.mean(errs)) if errs else np.inf
    return min(curve, key=curve.get)


def fold_report(X, Y, fams) -> Tuple[Dict[str, object], List[int]]:
    F = np.asarray(fams)
    seen, leaks = [], 0
    for g in sorted(set(F)):
        te = list(np.where(F == g)[0])
        tr = list(np.where(F != g)[0])
        seen += te
        if set(np.asarray(F)[te]) & set(np.asarray(F)[tr]):
            leaks += 1
        if set(te) & set(tr):
            leaks += 1
    return {"family_leaks": leaks, "n_folds": len(set(F)), "n_scored": len(seen)}, sorted(seen)


def g0_gap(root) -> float:
    return abs(float(tv.no_holdout_control_score(root)) - ANCHOR)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "Model 7/Approach A" && python check_m7a_g0.py`
Expected: all four test functions PASS. If G0 is not exactly 0.0, **stop**: the head was transcribed
rather than imported, and no arm may be scored until it is.

- [ ] **Step 5: Commit**

```bash
git add "Model 7/Approach A/m7a_eval.py" "Model 7/Approach A/check_m7a_g0.py"
git commit -m "M7-A task 4: G0 anchor to 0.00e+00, G6 inertness, family-honest arm"
```

---

### Task 5: the alias pair set, which must reproduce the Model 6-3 figures owned by its own rule

**Files:**
- Create: `Model 7/Approach A/alias_gate.py`
- Test: `Model 7/Approach A/check_alias_gate.py`
- Create: `Model 7/Approach A/data/alias_pairs.csv`

**Interfaces:**
- Consumes: `Model 2/Model 2 Experiment 3/cv_per_soil.csv` (column `M1`), `transfer_eval.FEATURES`,
  `soil_from_images`, `image_features`, `labels_matrix`, `emd_pair`, `cv_families.csv`.
- Produces: `derive_pairs(root) -> pd.DataFrame` with columns `kind` (`"alias"` or `"control"`),
  `soil_a, soil_b, feature_distance, curve_emd`, and `SUMMARY: dict` carrying `top10_share`,
  `worst_share`, `median_alias_curve_gap`, `n_alias`, `n_control`.

- [ ] **Step 1: Write the failing test**

```python
"""The alias set is only usable if it reproduces the Model 6-3 record. Run: python
check_alias_gate.py"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "Model 5" / "Model 5 Experiment 1"))
import alias_gate as ag  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def test_reproduces_model_6_3_summary():
    s = ag.summary(HERE.parents[1])
    check("top-10 share of summed CV error is 61.2%", abs(s["top10_share"] - 61.2) < 0.15,
          "%.2f" % s["top10_share"])
    check("worst soil share is 10.1%", abs(s["worst_share"] - 10.1) < 0.05, "%.2f" % s["worst_share"])
    check("median alias curve gap is REPORTED, not target-matched (amendment A1)",
          s["median_alias_curve_gap"] > 0.0, "%.4f" % s["median_alias_curve_gap"])
    h = ag.historical_gaps(HERE.parents[1])
    check("65.7 is NOT reproduced by the registered rule and must not be (amendment A1 teeth)",
          abs(s["median_alias_curve_gap"] - 65.7) >= 0.6, "%.4f" % s["median_alias_curve_gap"])
    check("Model 6-3's own worst-8/best-8 rule still reproduces its 65.7 and 28.1",
          abs(h["worst8_median_curve_emd"] - 65.7) < 0.6
          and abs(h["best8_median_curve_emd"] - 28.1) < 0.6,
          "%.4f / %.4f" % (h["worst8_median_curve_emd"], h["best8_median_curve_emd"]))
    check("10 alias pairs and 14 control pairs", s["n_alias"] == 10 and s["n_control"] == 14,
          "%s/%s" % (s["n_alias"], s["n_control"]))
    check("H038-G190 is in the alias set", ("H038", "G190") in [tuple(sorted(p)) for p in
          s["alias_pair_ids"]])


def test_pairs_are_cross_family_by_construction():
    df = ag.derive_pairs(HERE.parents[1])
    bad = [r for r in df.itertuples() if r.soil_a_family == r.soil_b_family]
    check("no pair joins two soils of the same CV family", not bad, "%d bad" % len(bad))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_alias_gate.py`
Expected: `ModuleNotFoundError: No module named 'alias_gate'`

- [ ] **Step 3: Implement the deterministic derivation**

```python
"""alias_gate.py - the pair set, derived by one deterministic rule over recorded artifacts.

Model 6-3 ran this diagnostic read-only and never persisted the pair list, so it is re-derived here
from two files that do exist. It must reproduce the Model 6-3 figures that are properties of this
rule (the 61.2% and 10.1% shares); the 65.7 EMD figure belongs to Model 6-3's own worst-8/best-8
unconstrained procedure and is not a target for this rule - amendment A1 in the spec.

Rule, fixed in the pre-registration: for each of the 10 highest-error soils, its nearest neighbour in
the standardised 12-feature space, drawn from a DIFFERENT CV family than either endpoint, is its alias
partner. The control set is the same rule on the 14 lowest-error soils.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import transfer_eval as tv  # noqa: E402

SUMMARY: Dict[str, object] = {}


def _base(root):
    ft = pd.read_csv(root / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    fam = dict(zip(ft.sample_id.astype(str), ft.cv_family.astype(str)))
    ids, Y = tv.labels_matrix(root)
    ids = [str(i) for i in ids]
    X = tv.soil_from_images(tv.image_features(root)).loc[ids, list(tv.FEATURES)].to_numpy(float)
    err = pd.read_csv(root / "Model 2" / "Model 2 Experiment 3" / "cv_per_soil.csv")
    err = err.set_index(err.sample_id.astype(str))["M1"].loc[ids].to_numpy(float)
    return ids, X, Y, fam, err


def derive_pairs(root) -> pd.DataFrame:
    ids, X, Y, fam, err = _base(root)
    order = np.argsort(err)[::-1]
    alias, control = list(order[:10]), list(order[10:])
    rows = []
    for kind, group in (("alias", alias), ("control", control)):
        for q in group:
            # one rule for both sets, exactly as pre-registered: the neighbour is from a different CV
            # family than the query. No extra exclusion, no tuned radius, no hand-picked pair.
            ok = [j for j in range(len(ids)) if j != q and fam[ids[q]] != fam[ids[j]]]
            if not ok:
                continue
            sd = np.where(X.std(0) > 1e-12, X.std(0), 1.0)
            z = (X - X.mean(0)) / sd
            dist_all = np.sqrt(((z[ok] - z[q]) ** 2).sum(1))
            j = int(ok[int(np.argmin(dist_all))])
            rows.append({"kind": kind, "soil_a": ids[q], "soil_b": ids[j],
                         "soil_a_family": fam[ids[q]], "soil_b_family": fam[ids[j]],
                         "feature_distance": float(dist_all.min()),
                         "curve_emd": tv.emd_pair(Y[q], Y[j])})
    df = pd.DataFrame(rows)
    HERE.joinpath("data").mkdir(exist_ok=True)
    df.to_csv(HERE / "data" / "alias_pairs.csv", index=False)
    return df


def summary(root) -> Dict[str, object]:
    ids, X, Y, fam, err = _base(root)
    tot = float(err.sum())
    order = np.argsort(err)[::-1]
    df = derive_pairs(root)
    SUMMARY.update({
        "top10_share": 100.0 * float(err[order[:10]].sum()) / tot,
        "worst_share": 100.0 * float(err[order[0]]) / tot,
        "median_alias_curve_gap": float(df[df.kind == "alias"].curve_emd.median()),
        "n_alias": int((df.kind == "alias").sum()),
        "n_control": int((df.kind == "control").sum()),
        "alias_pair_ids": [sorted([r.soil_a, r.soil_b]) for r in
                           df[df.kind == "alias"].itertuples()],
    })
    return SUMMARY
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd "Model 7/Approach A" && python check_alias_gate.py`
Expected: both tests PASS. If a share differs from the recorded value, the derivation is wrong and the
gate must not run - fix the rule, not the tolerance. The 65.7 EMD figure is not a property of this
rule's group and is checked in both directions instead: reproduced by Model 6-3's own rule, NOT
reproduced by this one (amendment A1).

- [ ] **Step 5: Commit**

```bash
git add "Model 7/Approach A/alias_gate.py" "Model 7/Approach A/check_alias_gate.py" \
        "Model 7/Approach A/data/alias_pairs.csv"
git commit -m "M7-A task 5: deterministic alias pair re-derivation, matches Model 6-3 record"
```

---

### Task 6: the separability pre-gate, which can close the experiment before any CV arm

**Files:**
- Modify: `Model 7/Approach A/alias_gate.py` (add `separability`)
- Modify: `Model 7/Approach A/check_alias_gate.py` (add the gate test)

**Interfaces:**
- Consumes: `derive_pairs`, `spatial_features.tile_spatial` via `m7a_data`, the real block, and the
  P-RAND / P-SHUF blocks.
- Produces: `separability(root) -> dict` with keys `base12`, `block16`, `rand16`, `shuf16` (each a
  median alias-pair separation in z units), `gain_vs_base`, `gate_passed: bool`.

- [ ] **Step 1: Write the failing test**

```python
def test_separability_gate_reports_all_three_blocks():
    s = ag.separability(HERE.parents[1])
    for key in ("base12", "block16", "rand16", "shuf16", "gain_vs_base", "gate_passed"):
        check("separability reports %s" % key, key in s)
    check("the block is compared against the 12 on the same pairs",
          s["base12"] != s["block16"])
    check("placebos are computed, not assumed equal", s["rand16"] != s["shuf16"]
          or abs(s["rand16"] - s["shuf16"]) < 1e-9)
    check("gate_passed is a bool, never a float close to one",
          isinstance(s["gate_passed"], bool))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_alias_gate.py`
Expected: `AttributeError: module 'alias_gate' has no attribute 'separability'`

- [ ] **Step 3: Implement it**

```python
def _median_separation(mat: np.ndarray, df: pd.DataFrame, ids) -> float:
    """Median |z difference| between the two soils of each alias pair, in the given design space.
    Standardisation uses the full 24-soil corpus, which is legitimate here because nothing is fitted
    and no label is read: this is a property of the feature space, not a model score."""
    pos = {s: i for i, s in enumerate(ids)}
    mu, sd = mat.mean(0), np.where(mat.std(0) > 1e-12, mat.std(0), 1.0)
    z = (mat - mu) / sd
    out = []
    for r in df[df.kind == "alias"].itertuples():
        out.append(float(np.sqrt(((z[pos[r.soil_a]] - z[pos[r.soil_b]]) ** 2).mean())))
    return float(np.median(out))


def separability(root) -> Dict[str, object]:
    import m7a_data as d
    import spatial_features as sf
    ids, X, Y, fam, err = _base(root)
    df = derive_pairs(root)
    B = d.spatial_soil_table(root).loc[ids, list(sf.SPATIAL)].to_numpy(float)
    Br = d.spatial_soil_table(root, scrambled=True).loc[ids, list(sf.SPATIAL)].to_numpy(float)
    Bs = sf.shuffle_columns(B, 90002)[0]
    base = _median_separation(X, df, ids)
    block = _median_separation(np.hstack([X, B]), df, ids)
    rand = _median_separation(np.hstack([X, Br]), df, ids)
    shuf = _median_separation(np.hstack([X, Bs]), df, ids)
    gain = block - base
    return {"base12": base, "block16": block, "rand16": rand, "shuf16": shuf,
            "gain_vs_base": gain,
            "gate_passed": bool(gain > 0.0 and gain > (rand - base) and gain > (shuf - base))}
```

- [ ] **Step 4: Run it, then record the outcome as a pre-gate decision**

Run: `cd "Model 7/Approach A" && python check_alias_gate.py`
Expected: all tests PASS.

Then run the gate itself and write the four numbers into the run log before anything else is scored:

```bash
cd "Model 7/Approach A" && python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0, str(Path('.') .resolve().parents[1] / 'Model 5' / 'Model 5 Experiment 1'))
import alias_gate as ag, json
s = ag.separability(Path('.').resolve().parents[1])
print(json.dumps({k: (v if not isinstance(v, float) else round(v, 4)) for k, v in s.items()}))"
```

If `gate_passed` is false, **M7-A closes here**: record the four numbers, write the verdict
`REFUTED-PRE-GATE` into `Model 7/instructions.txt`, skip Tasks 7-9, and move to the M7-C decision.
That is the branch the pre-registration was built to allow.

- [ ] **Step 5: Commit**

```bash
git add "Model 7/Approach A/alias_gate.py" "Model 7/Approach A/check_alias_gate.py"
git commit -m "M7-A task 6: separability pre-gate with real, phase-scrambled and shuffled blocks"
```

---

### Task 7: the four controls and the best-of-4 null

**Files:**
- Modify: `Model 7/Approach A/m7a_eval.py` (add `controls`, `best_of_k_null`)
- Modify: `Model 7/Approach A/check_m7a_g0.py` (add control tests)
- Create: `Model 7/Approach A/results/controls.csv`

**Interfaces:**
- Consumes: `soil_design`, `lofo_errors`, `clustered_bootstrap`.
- Produces: `controls(root) -> pd.DataFrame` with one row per block in
  `("none", "real", "rand", "shuf", "cols")` and columns `block, mean_error, gain_vs_none`;
  `best_of_k_null(root, n_perm=40) -> dict` with keys `median, p90, p95, max, n_perm,
  seconds_per_draw` (the count is cost-derived; see Task 7).

- [ ] **Step 1: Write the failing test**

```python
def test_controls_are_able_to_differ():
    c = ev.controls(HERE.parents[1]).set_index("block")
    check("none and cols are distinguishable", abs(c.loc["none", "gain_vs_none"]) < 1e-12)
    check("cols has a gain of its own or is exactly zero, both are informative",
          abs(c.loc["cols", "gain_vs_none"]) >= 0.0)
    check("rand, shuf and real are three different numbers",
          len({round(c.loc[b, "mean_error"], 6) for b in ("real", "rand", "shuf")}) == 3)
    check("the placebo blocks are not secretly identical to the real one",
          c.loc["real", "mean_error"] != c.loc["rand", "mean_error"])


def test_best_of_k_null_is_measured_not_inherited():
    n = ev.best_of_k_null(HERE.parents[1], n_perm=4)      # 4 draws here; the run uses 40
    for key in ("median", "p90", "p95", "max", "n_perm", "seconds_per_draw"):
        check("null reports %s" % key, key in n)
    check("the test itself does not pretend to be the null", n["n_perm"] == 4)
    check("null is a positive quantity that a real gain must clear", n["median"] > 0.0,
          "median %.2f" % n["median"])
    check("the cost is timed, because n_perm was set from a measurement",
          n["seconds_per_draw"] > 0.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_m7a_g0.py`
Expected: `AttributeError: module 'm7a_eval' has no attribute 'controls'`

- [ ] **Step 3: Implement**

```python
def controls(root) -> pd.DataFrame:
    base = None
    rows = []
    for block in ("none", "real", "rand", "shuf", "cols"):
        X, Y, F, ids = soil_design(root, block)
        e = lofo_errors(X, Y, F)
        if base is None:
            base = e
        rows.append({"block": block, "mean_error": float(e.mean()),
                     "gain_vs_none": float((base - e).mean())})
    df = pd.DataFrame(rows)
    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "controls.csv", index=False)
    return df


def best_of_k_null(root, n_perm: int = 40, seed: int = SEED_PERM) -> Dict[str, float]:
    """Replay the best-of-4 selection on permuted soil-to-curve pairings.

    The null must be measured for THIS K. The map's 15.29 EMD figure is a best-of-15 null on a
    different design matrix; inheriting a number across designs is how a null stops being a null.

    n_perm is 40, not the project's usual 200, and the reason is measured rather than assumed: one
    lofo_errors call costs 4.15 s on this 24 x 16 design (timed 2026-09-29 on local CPU), and each
    permutation needs 6 of them (the no-block base plus the four single columns plus the block), so
    200 permutations would cost about 83 minutes inside a run that is already doing two full passes.
    40 draws keep the p95 conservative and the whole null near 17 minutes. The per-call cost and the
    resulting budget are printed by the notebook so the number is auditable, and this deviation from
    the usual 200 is disclosed in Experiment1.txt rather than buried.
    """
    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    X0, Y0, F, _ = soil_design(root, "none")
    B = soil_design(root, "real")[0][:, 12:]
    blocks = {"A1": [0], "A2": [1], "A3": [2], "A4": [3], "ALL4": [0, 1, 2, 3]}
    draws = []
    for _ in range(int(n_perm)):
        perm = rng.permutation(len(Y0))
        Yp = Y0[perm]
        gains = []
        base = float(lofo_errors(X0, Yp, F).mean())
        for cols in blocks.values():
            Xb = np.hstack([X0, B[:, cols]])
            gains.append(base - float(lofo_errors(Xb, Yp, F).mean()))
        draws.append(max(gains))
    draws = np.asarray(draws)
    return {"median": float(np.median(draws)), "p90": float(np.percentile(draws, 90)),
            "p95": float(np.percentile(draws, 95)), "max": float(draws.max()),
            "n_perm": int(n_perm),
            "seconds_per_draw": float((time.time() - t0) / max(1, int(n_perm)))}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `cd "Model 7/Approach A" && python check_m7a_g0.py`
Expected: all tests PASS, including G0 still exact.

- [ ] **Step 4b: Confirm the runtime budget before committing**

```bash
cd "Model 7/Approach A" && python -c "
import time, numpy as np, sys
from pathlib import Path
sys.path.insert(0,'.'); sys.path.insert(0,'../../Model 5/Model 5 Experiment 1')
import m7a_eval as ev
root=Path('.').resolve().parents[1]
X,Y,F,_=ev.soil_design(root,'real'); t=time.time(); ev.lofo_errors(X,Y,F); c=time.time()-t
print('one lofo_errors call: %.2f s -> null (40 x 6 calls): %.1f min' % (c, 40*6*c/60))"
```

If the measured per-call cost is more than twice the 4.15 s this plan assumed, stop and report the
budget rather than silently dropping permutations.

- [ ] **Step 5: Commit**

```bash
git add "Model 7/Approach A/m7a_eval.py" "Model 7/Approach A/check_m7a_g0.py"
git commit -m "M7-A task 7: four controls plus a measured best-of-4 permutation null"
```

---

### Task 8: the verdict ladder

**Files:**
- Modify: `Model 7/Approach A/m7a_eval.py` (add `paired_gain`, `verdict`)
- Modify: `Model 7/Approach A/check_m7a_g0.py` (add ladder tests)

**Interfaces:**
- Consumes: `lofo_errors`, `clustered_bootstrap`, `controls`, `best_of_k_null`.
- Produces: `paired_gain(e_base, e_arm, F) -> dict(mean, lo, hi, p_no_gain)`;
  `specificity(e_base, e_real, e_placebo, F) -> dict`, which is G3's "by more than the paired noise
  partner" clause made concrete as the family-clustered bootstrap of `e_placebo - e_real`;
  `verdict(**clauses) -> dict(G1..G7, verdict, submit)` with the contract's precedence.

- [ ] **Step 1: Write the failing test**

```python
def test_paired_gain_direction_is_base_minus_arm():
    F = [f"F{i % 6}" for i in range(24)]
    e_base = np.linspace(60.0, 20.0, 24)
    e_arm = e_base - 3.0
    g = ev.paired_gain(e_base, e_arm, F)
    check("a uniform 3 EMD improvement reads as +3, not -3", abs(g["mean"] - 3.0) < 1e-9,
          "%.3f" % g["mean"])
    check("the CI brackets it tightly", g["lo"] > 2.9 and g["hi"] < 3.1)


def test_verdict_precedence_is_deterministic():
    v = ev.verdict(G0=True, G1=False, G2=True, G3=True, G4=True, G5=False, G6=True, G7=True)
    check("invalid implementation outranks everything", v["verdict"] == "INVALID-IMPLEMENTATION")
    v = ev.verdict(G0=True, G1=True, G2=True, G3=True, G4=True, G5=True, G6=True, G7=True)
    check("all clauses pass is SUPPORTED", v["verdict"] == "SUPPORTED")
    check("SUBMIT is true only for SUPPORTED", v["submit"] is True)
    v = ev.verdict(G0=True, G1=True, G2=True, G3=False, G4=True, G5=True, G6=True, G7=True)
    check("gain without specificity is an artefact verdict",
          v["verdict"] == "SELECTION-OR-CAPACITY-ARTEFACT" and v["submit"] is False)
    v = ev.verdict(G0=True, G1=False, G2=True, G3=True, G4=True, G5=True, G6=True, G7=True)
    check("below band is recorded, not dressed up",
          v["verdict"] == "SUPPORTED-BELOW-BAND" and v["submit"] is False)
    v = ev.verdict(G0=True, G1=False, G2=False, G3=False, G4=False, G5=True, G6=True, G7=True)
    check("G1 and G3 both failing is REFUTED", v["verdict"] == "REFUTED")
    v = ev.verdict(G0=True, G1=True, G2=False, G3=True, G4=True, G5=True, G6=True, G7=True)
    check("effect size without support is NOT-RESOLVED", v["verdict"] == "NOT-RESOLVED")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd "Model 7/Approach A" && python check_m7a_g0.py`
Expected: `AttributeError: module 'm7a_eval' has no attribute 'paired_gain'`

- [ ] **Step 3: Implement**

```python
def paired_gain(e_base, e_arm, fams, n_boot: int = 4000, seed: int = 20260934) -> Dict[str, float]:
    """d = base - arm, so a POSITIVE mean is an improvement. Clusters are CV families: siblings share
    a fold, so their errors are not independent and row resampling would narrow every interval."""
    d = np.asarray(e_base, float) - np.asarray(e_arm, float)
    mean, lo, hi = tv.clustered_bootstrap(d, list(fams), n=n_boot, seed=seed)
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi),
            "p_no_gain": float(np.mean(np.asarray(d) <= 0.0))}


def specificity(e_base, e_real, e_placebo, fams, n_boot: int = 4000) -> Dict[str, float]:
    """G3's 'by more than the paired noise partner', made concrete.

    The comparison is not real gain minus placebo gain, which is the difference of two correlated
    quantities; it is the bootstrap of the per-soil difference (e_base - e_real) - (e_base - e_placebo)
    = e_placebo - e_real, clustered by CV family. G3 needs that CI to exclude zero, i.e. the real block
    must beat the placebo as a paired quantity, not merely look larger on average.
    """
    return paired_gain(e_placebo, e_real, fams, n_boot=n_boot)


def verdict(**c) -> Dict[str, object]:
    required = {"G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"}
    if set(c) != required:
        raise AssertionError(f"verdict needs exactly {sorted(required)}, got {sorted(c)}")
    if not (c["G5"] and c["G6"] and c["G0"] and c["G7"]):
        v = "INVALID-IMPLEMENTATION"
    elif all(c[k] for k in ("G1", "G2", "G3", "G4")):
        v = "SUPPORTED"
    elif c["G1"] and not (c["G3"] and c["G4"]):
        v = "SELECTION-OR-CAPACITY-ARTEFACT"
    elif c["G1"] and not c["G2"]:
        v = "NOT-RESOLVED"
    elif not c["G1"] and c["G3"] and c["G4"]:
        v = "SUPPORTED-BELOW-BAND"
    else:
        v = "REFUTED"
    return {**c, "verdict": v, "submit": bool(v == "SUPPORTED")}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "Model 7/Approach A" && python check_m7a_g0.py`
Expected: all tests PASS, exit 0. Every verdict in the contract's table is executed, so a precedence
change cannot slip through unnoticed.

- [ ] **Step 5: Commit**

```bash
git add "Model 7/Approach A/m7a_eval.py" "Model 7/Approach A/check_m7a_g0.py"
git commit -m "M7-A task 8: family-clustered paired bootstrap and deterministic verdict precedence"
```

---

### Task 9: notebook generator and static self-checks

**Files:**
- Create: `scratch/build_m7a1.py`
- Create: `scratch/mk_run7.py`
- Produce: `Model 7/Approach A/Model7_ApproachA_Experiment1.ipynb`

**Interfaces:**
- Consumes: every module from Tasks 1-8; the builder idiom in `scratch/build_m5e1.py`
  (`md()`, `code()`, the `EXPECTED_CELLS` pin, the triple-quote cell-terminator guard).
- Produces: a notebook whose cells are `["A1","A2","B1","C1","D1","E1","E2","F1","G1","H1","P1",
  "Q1","R1"]`, and `scratch/mk_run7.py` exit 0.

- [ ] **Step 1: Write the generator's cell list and the guard first**

Create `scratch/build_m7a1.py` starting from the builder skeleton in `scratch/build_m5e1.py`, then
change only the pins:

```python
EXPECTED_CELLS = ["A1", "A2", "B1", "C1", "D1", "E1", "E2", "F1", "G1", "H1", "P1", "Q1", "R1"]
REUSED = ["A1"]                      # A1 imports transfer_eval unchanged; nothing else is retyped
MODEL_ID = "Model 7 / Approach A"
EXPERIMENT_ID = "Experiment 1"
OUT = ROOT / "Model 7" / "Approach A" / "Model7_ApproachA_Experiment1.ipynb"
```

Keep the terminator guard verbatim - it is the check that caught seven swallowed cells in Model 5:

```python
def code(src: str) -> None:
    last = src.rstrip().split("\n")[-1]
    if last.rstrip().endswith('"""'):
        raise AssertionError('a code cell ends with a triple-double-quote, which means a cell block '
                             'was closed with """ instead of \'\'\' - the next cell has been swallowed')
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": src})
```

- [ ] **Step 2: Add the frozen thresholds to the notebook and to the checker's "appears once" list**

`scratch/mk_run7.py` must fail if a threshold appears in more than one cell, so list them exactly:

```python
THRESHOLDS = {
    "43.0217308796477": 1,      # G0 anchor
    "3.00": 1,                  # G1 bar
    "0.00e+00": 1,              # G0 tolerance
    "4000": 1,                  # bootstrap draws
    "n_perm=40": 1,             # best-of-4 null draws, cost-derived (see Task 7)
    "4.15": 2,                  # the measured lofo_errors cost justifying n_perm=40
}
CONTRACT_MARKERS = [
    "SUBMIT = TRUE if and only if VERDICT == SUPPORTED",
    "SELECTION-OR-CAPACITY-ARTEFACT",
    "SUPPORTED-BELOW-BAND",
    "INVALID-IMPLEMENTATION",
    "Alias-separability gate",
]
BANNED = ["camera_fam", "cv_group", "EXIF", "exif", "icc_profile", "config_hash",
          "Image.open", "data/Training", "data/Test", 'to_csv("data/', "TARGET_PPM"]
# 60.56167 is deliberately NOT in this list: cell P1 must write the pre-declared external band before
# any score exists. Scale discipline is enforced per cell instead - see Step 3.
```

- [ ] **Step 3: Write the cells, in gate order, each importing rather than restating**

`A1` imports `transfer_eval`, `m7a_data`, `alias_gate`, `m7a_eval` and asserts
`tv.FEATURES` is the 12-tuple. `B1` runs `ag.summary` and asserts the two Model 6-3 shares that this rule owns (61.2%, 10.1%) with
the same tolerances as Task 5, records the registered-rule median alias gap without comparing it to
65.7, and asserts the amendment A1 direction check (Model 6-3's own rule reproduces 65.7; the
registered rule does not); it stops the notebook if any of those fails. `C1` runs
`ev.g0_gap` and asserts it is exactly `0.0`. `D1` runs `ag.separability` and records the four numbers
plus `gate_passed`; if false it writes the verdict file and raises `SystemExit(0)` - a clean close,
not an error. `E1` runs `ev.controls` for all five blocks. `E2` runs `ev.best_of_k_null(n_perm=40)`.
`F1` computes `ev.paired_gain` for base versus the real block. `G1` evaluates each clause G1-G7 from
the numbers already printed, including the G6 zero-block equality. `H1` calls `ev.verdict(...)`,
asserts `submit == (verdict == "SUPPORTED")`, and writes `gate.json`. `P1` writes
`Experiment1.txt`. `Q1` writes the per-soil CSV and the control table. `R1` prints the go/no-go
decision for a submission and, only when `submit` is true, emits
`submission.csv` from the frozen full-fit recipe using the 16-column design.

- [ ] **Step 4: Build, compile and statically check**

Run: `python scratch/build_m7a1.py && python scratch/mk_run7.py; echo "RC=$?"`
Expected: builder prints the cell count equal to `len(EXPECTED_CELLS)`, mk_run7 exits `RC=0` with
every cell compiling, the contract markers found, thresholds appearing the expected number of times,
and the free-name scan clean. **Read the exit code, not the last printed line.**

`BANNED` is matched against the notebook's code cells, with the reasoning recorded rather than
assumed: `Image.open` is banned in cells because tiles are read only inside `m7a_data.py`, which is
imported, so an inline tile read would be an unaudited second data path; `camera_fam`, `cv_group`,
`EXIF`, `icc_profile` and `config_hash` are banned because the fold key is the CV family from
`cv_families.csv` and no device grouping may select anything; `to_csv("data/` is banned because the
run may not write into `data/`. The external baseline `60.56167` is **not** in `BANNED`, because cell
P1 must write the pre-declared external band before any score exists; instead `mk_run7.py` enforces
scale discipline by cell id, failing if `60.56167` appears in any cell except P1, since mixing the
external scale into an internal gate decision is the error the contract forbids. A banned string that
appears in a printed sentence about the prohibition is a false positive, so `mk_run7.py` scans code
cells only, never markdown.

- [ ] **Step 5: Commit**

```bash
git add scratch/build_m7a1.py scratch/mk_run7.py "Model 7/Approach A/Model7_ApproachA_Experiment1.ipynb"
git commit -m "M7-A task 9: notebook generator, cell pinning and static contract checks"
```

---

### Task 10: the real run, determinism, and verdict recording

**Files:**
- Create: `scratch/run_nb_m7a1.py`
- Create: `Model 7/Approach A/results/run1/` containing `Experiment1.txt`, `gate.json`,
  `controls.csv`, `per_soil_m7a.csv`, `separability.json`, `alias_pairs.csv`
- Modify: `Model 7/instructions.txt`, `AGENT_BRIEF.md`, `model7_preregistrations.md`
- Modify: memory `MEMORY.md` and `project-model7-registration.md`

**Interfaces:**
- Consumes: the built notebook.
- Produces: the run artifacts and a recorded verdict. No submission file unless
  `gate.json["submit"]` is true.

- [ ] **Step 1: Write the runner from the Model 5 runner unchanged except for paths**

```python
RUNS = [("run1", HERE / "Model 7" / "Approach A" / "results" / "run1")]
NOTEBOOK = HERE / "Model 7" / "Approach A" / "Model7_ApproachA_Experiment1.ipynb"
```

Execute cell by cell, capture stdout per cell, and write `run1/executed.ipynb` so the artifact is the
one that ran.

- [ ] **Step 2: Run it for real**

Run: `python scratch/run_nb_m7a1.py; echo "RC=$?"`
Expected: `RC=0`, and the printed gate line shows G0 gap `0.00e+00`.

- [ ] **Step 3: Determinism - second run must be byte-identical**

```bash
M7A_OUT=run2 python scratch/run_nb_m7a1.py; echo "RC=$?"
diff "Model 7/Approach A/results/run1/per_soil_m7a.csv" \
     "Model 7/Approach A/results/run2/per_soil_m7a.csv"
diff "Model 7/Approach A/results/run1/controls.csv" \
     "Model 7/Approach A/results/run2/controls.csv"
diff "Model 7/Approach A/results/run1/gate.json" \
     "Model 7/Approach A/results/run2/gate.json"
echo "DIFF_RC=$?"
```

`run_nb_m7a1.py` reads the output directory from `M7A_OUT` and defaults to `run1`, so the two passes
are the same notebook executed twice rather than two scripts that could drift. Expected: `RC=0` and
`DIFF_RC=0` with no diff output. Any difference means a seed or an ordering leak, and the result is
not usable.

- [ ] **Step 4: Independent re-verification, not a re-read of the notebook's own print**

Create `scratch/check_m7a_result.py` asserting, from the written artifacts alone: the G0 gap is
exactly 0.0; the per-soil array has 24 finite values with no soil scored twice; `controls.csv`
contains all five blocks and three distinct placebo means; the best-of-4 null has `n_perm == 40`, a
positive median, and a recorded `seconds_per_draw`; `gate.json["submit"] is (gate.json["verdict"] == "SUPPORTED")`; the recorded
verdict is one of the six contract verdicts; and `submission.csv` exists **only** when submit is true.

Run: `python scratch/check_m7a_result.py; echo "RC=$?"`
Expected: `RC=0`.

- [ ] **Step 5: Write the external band down before any score exists**

If and only if `gate.json["submit"]` is true, record in `Experiment1.txt` **before** submitting: the
expected external band is the 3-soil public band **[21.7, 69.9]** around the shipped internal mean,
the current baseline is **60.56167** external Kaggle EMD, and the internal gain just measured is
**not comparable** to either number. State plainly that a gain of this size is not externally
confirmable, and that the single submission is authorised by the local gate, not by a forecast. This
paragraph must exist before the score exists; writing it afterwards is the error the contract forbids.

- [ ] **Step 6: Record the verdict in the same run it was produced**

Append to `Model 7/Approach A/results/run1/Experiment1.txt`: the hypothesis, every gate value, the
verdict, the P-RAND / P-SHUF / P-COLS / P-SEL numbers next to the gated gain, and an explicit
"what this does not show" paragraph. Then update:

- `Model 7/instructions.txt` - M7-A status line and, if the experiment closes, the closure reason in
  the same words the contract uses.
- `AGENT_BRIEF.md` section 19 - the outcome line.
- `model7_preregistrations.md` - a dated note recording whether the fallback condition to M7-C fired.
- memory - the result and the corrected camera/feature picture if it changes.

```bash
git add "Model 7/Approach A/results" "Model 7/instructions.txt" AGENT_BRIEF.md \
        model7_preregistrations.md scratch/check_m7a_result.py scratch/run_nb_m7a1.py
git commit -m "M7-A task 10: real run, determinism, independent re-verification, verdict recorded"
```

---

## What this plan does not do

No threshold, gate, statistic or precedence changes anything in the approved spec. No M7-C, M7-T, M7-B
or M7-E code is written. No Kaggle notebook is built and no dataset is uploaded: all inputs are
materialised tiles and existing caches. No submission CSV is produced unless
`gate.json["submit"]` is true, and if the separability pre-gate fails the experiment closes at Task 6
with four numbers and no CV arm scored.

---

## Owner rulings, 2026-09-29  -  this section supersedes any conflicting code above

Issued after the Task 1 review found that this plan's code and the frozen pre-registration disagreed
in three statistic definitions and in the NaN rule. The spec cell text was amended for A2, A3 and A4
under the same authority; the A1 cell and everything else is unchanged. **Where code in a task above
contradicts this section, this section wins.**

1. **A1** - implement exactly as registered. No change to definition, scaling or interpretation.
2. **A2** - plain ratio `L(0 deg) / L(90 deg)`, not a log-ratio. A profile line wholly outside the
   mask is **skipped**, never written as 0.0. If a direction has too few valid lines, A2 is **NaN**.
   Any unexpected warning during a test is a **test failure**. Do not alter the statistic because its
   value changes with ppm - that sensitivity is known and accepted.
3. **A3** - `range / (0.5 * tile extent in mm)`. The sill locates the range; it is not the
   denominator. Do **not** switch to range / sill. The spec cell was amended to say this explicitly.
4. **A4** - `omni - C(45 deg)`, `omni` = mean of `C` over 0, 45, 90, 135 deg. Diagonal directions use
   a **3 px per-axis** offset; axis-aligned directions use the 5 px offset. Implement the registered
   intent; do not keep the previous implementation because it already exists.
5. **Degeneracy** - every undefined statistic is **NaN**, never a fallback number. Two distinct
   exception types: `TileDegenerate` for a genuinely degenerate tile, and a plain `ValueError` /
   assertion for an invalid or systemic call. A systemic error must never be caught and recorded as
   tile degeneracy. The minimum-pair thresholds are named and frozen in the spec's parameter table.
6. **Tests** - the checks that pinned the old divergent behaviour must be re-derived, not kept green
   by relabelling. A test that asserts a spec violation is worse than no test.

Scope guard: no change to the hypothesis, K = 4, the four statistic ids and roles, the controls, the
protocol, gates G0-G7, the 3.00 EMD bar, or the submission rule, beyond the three cell wordings the
owner authorised. The Kaggle score is not used for any tuning decision and external test labels are
not inspected.
