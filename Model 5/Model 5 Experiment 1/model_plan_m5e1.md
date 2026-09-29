# Model 5 / Experiment 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether a closed-form, label-free CORAL alignment of hand-built tile features onto the unlabeled iPhone test-image distribution reduces EMD on a camera that was neither fitted on nor used as the alignment target.

**Architecture:** Three new numpy-only modules — `alignment.py` (the transform), `transfer_eval.py` (image-level data assembly, two camera directions, nested alpha selection, soil-cluster bootstrap) — are each pinned by a runnable contract test before any experiment runs. A generated notebook wires them into the frozen Model 1/Model 2 head, runs the seven arms in both directions, evaluates the pre-registered gate G0–G7 clause by clause, and writes the verdict record. Nothing touches `data/`, and nothing runs on Kaggle.

**Tech Stack:** Python 3.14 local, numpy 2.5.2, pandas 2.3.3, scipy 1.18, scikit-learn 1.9 (`LedoitWolf`, `Ridge`, `StandardScaler`), matplotlib, Pillow. No torch, no timm, no GPU, no Kaggle, no dataset upload, no zip. `pytest` is not installed — contract tests are plain asserts with printed PASS/FAIL and a non-zero exit, matching `check_fusion.py` and `check_crop_geometry.py`.

**Spec:** `Model 5/Model 5 Experiment 1/model_spec_m5e1.md`. It is the scientific authority. This plan implements it and may not change it. If a step here appears to require changing a threshold, arm, verdict name, submission rule or closure rule, **stop and report** — that is a spec decision for the owner, not an implementation detail.

## Global constraints

- Every gate clause, threshold, verdict name and interpretation rule is in `model_spec_m5e1.md` sections 5, 5a, 6, 7, 10 and is **fixed before the run**. Allowed post-run changes are exactly two: correcting code that disagrees with the spec (disclosed in the record the way Model 2 E3 disclosed its two gate bugs), or an owner-approved revision before a rerun.
- Label-free: the alignment transform function's only arguments are feature matrices. A static test and a runtime test both assert this. Test labels do not exist and are never referenced. Camera identity, device make/model, ppm, EXIF, ICC, site and sample id are never features, conditioning variables or lookup keys.
- The alignment target is the 35 iPhone test images only; a runtime assertion requires it disjoint from every training-camera image.
- Frozen from Model 1/Model 2 and not edited: preprocessing, the 12 control features, tile→image→soil median aggregation, `StandardScaler`, rank-3 curve basis, closed-form ridge, alpha grid `0.03 … 1e6`, monotone projection, trapezoid EMD on `log10 d`, the 16 curve-distance CV families, nested alpha selection, the E4 parsimony rule, `config_hash 010f44c36c74`.
- Reuse-by-import: shared math is imported byte-for-byte from `Model 2/Model 2 Experiment 3/Model2_Experiment3.ipynb`, never retyped, and the notebook records which cells were reused.
- Never modify or delete `data/`, `Model 1/`, `Model 2/`, or any previous experiment's outputs. Model 5 writes only into `Model 5/Model 5 Experiment 1/`.
- No mock mode. Missing inputs raise and list every missing item. There is no branch that runs locally and a different branch elsewhere.
- All printed text ASCII-only (Windows console is cp1252).
- Seeds, pinned now: `SEED = 20260935`, bootstrap seed `20260934`, `CORAL_INDEP` shuffle seed `20260936`. `embed_seed` is irrelevant here and must not appear.
- Do not run `git commit` unless the owner asks. Each task ends with a checkpoint the owner approves.

---

## File structure and responsibilities

```
Model 5/Model 5 Experiment 1/
  ├── instructions.txt        the contract: H5, arms, gate with numbers, verdict table,
  │                           submission rule, closure rule. Written first.
  ├── alignment.py            THE new idea. Pure numpy. CORAL / mean-only / independent-column
  │                           placebo / self. Knows nothing about cameras, labels or files.
  ├── check_alignment.py      contract test for alignment.py, incl. the row-permutation
  │                           invariance that made the first placebo design dead (pinned so it
  │                           is never silently reintroduced).
  ├── transfer_eval.py        data assembly at IMAGE level, the two directions, nested alpha
  │                           selection with boundary status, soil-clustered bootstrap.
  ├── check_transfer_eval.py  contract test: exact soil counts, disjointness, G0 reproduction
  │                           of Model 2 E3's 43.0217, clustered CI wider than naive CI.
  ├── Model5_Experiment1.ipynb generated notebook
  ├── Experiment1.txt          written by the run, from live objects
  └── (outputs listed in spec section 9)

scratch/
  ├── build_m5e1.py            notebook generator; imports shared cells verbatim from E3
  ├── mk_run6.py               regenerate + compile each cell independently + flatten
  └── run_nb_m5e1.py           execute the SHIPPED notebook, not the builder's output
```

---

## Task 1: Folder and the written contract

**Files:**
- Create: `Model 5/Model 5 Experiment 1/instructions.txt`

**Interfaces:**
- Consumes: `model_spec_m5e1.md` sections 5, 5a, 6, 7, 10
- Produces: the text the notebook's gate cell is checked against, in prose

- [ ] **Step 1: Create the folder** `Model 5/Model 5 Experiment 1/`.

- [ ] **Step 2: Write `instructions.txt`** containing, verbatim in intent and numerically exact:
      H5 as stated in spec section 0; the measured facts table from spec section 1 (24 train soils,
      23 Motorola, 22 Samsung, 21 both, 35 test images, 435 test tiles, 45 evaluation rows, 24
      bootstrap clusters); the single insertion point `tiles → image median → ALIGN → soil median →
      frozen head`; the seven-arm table with each arm's purpose; the gate block G0–G7 with every
      number (3.00 / 1.00 / 1.50 / 0.50 / 0.05 / ±0.10) and the boundary-inclusive convention; spec
      section 5a's per-direction boundary rule in full; the interpretation table including
      `SUPPORTED-WITH-BOUNDARY-CAVEAT`, `REFUTED-WITH-BOUNDARY-CAVEAT` and `INCONCLUSIVE`; the
      submission rule requiring the verdict to be exactly `SUPPORTED`; P1–P6 with blind flags; the
      closure rule; and spec section 13's correction of the row-permutation placebo.

- [ ] **Step 3: Verify no placeholder language.** Run
      `grep -n -i -E "TBD|TODO|fill in|appropriate error handling|handle edge cases" "Model 5/Model 5 Experiment 1/instructions.txt"`.
      Expected: no matches.

- [ ] **Step 4: Checkpoint.** Report the file to the owner. Do not commit unasked.

---

## Task 2: `alignment.py` — the transform, isolated from everything

**Files:**
- Create: `Model 5/Model 5 Experiment 1/alignment.py`
- Test: `Model 5/Model 5 Experiment 1/check_alignment.py`

**Interfaces:**
- Consumes: numpy only
- Produces:
  - `covariance(X: np.ndarray) -> tuple[np.ndarray, float]` — shrunk covariance, shrinkage weight
  - `coral_transform(source: np.ndarray, target: np.ndarray, match_cov: bool = True) -> dict` — keys `A (d,d)`, `b (d,)`, `shrink_source`, `shrink_target`, `match_cov`
  - `apply_transform(X: np.ndarray, T: dict) -> np.ndarray` — `X @ T["A"] + T["b"]`
  - `independent_column_shuffle(target: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]` — shuffled table and the stacked permutation vectors
  - `transform_distance(T1: dict, T2: dict) -> float` — max abs elementwise difference of `A`, for canary and placebo checks

- [ ] **Step 1: Write the failing test file first** with these assertions, in the house
      plain-assert style (`check(label, ok, detail)`, printed PASS/FAIL, non-zero exit):

```python
def test_coral_matches_target_moments():
    rng = np.random.default_rng(1)
    S = rng.normal(size=(23, 12)); S[:, 0] *= 4.0; S[:, 1] += 0.9 * S[:, 2]
    T = rng.normal(size=(35, 12)) + np.array([2.0] + [0.0] * 11)
    tr = al.coral_transform(S, T)
    Sm = al.apply_transform(S, tr)
    assert abs(Sm.mean(0) - T.mean(0)).max() < 1e-9
    assert abs(np.cov(Sm.T) - np.cov(T.T)).max() < 0.35      # shrunk, so not exact

def test_self_alignment_is_identity():
    X = np.random.default_rng(2).normal(size=(22, 12))
    tr = al.coral_transform(X, X)
    assert np.abs(tr["A"] - np.eye(12)).max() < 1e-8
    assert np.abs(tr["b"]).max() < 1e-8

def test_mean_arm_leaves_covariance_alone():
    X = np.random.default_rng(3).normal(size=(30, 12)); X[:, 0] += 0.7 * X[:, 1]
    tr = al.coral_transform(X, np.zeros((10, 12)), match_cov=False)
    Y = al.apply_transform(X, tr)
    assert np.allclose(np.cov(Y.T), np.cov(X.T), atol=1e-10)
    assert np.allclose(Y.mean(0), X.mean(0) + tr["b"], atol=1e-12)

def test_row_order_of_target_is_irrelevant():
    """Pins WHY the first placebo design was dead. Row order cannot change mean or covariance,
    so any placebo built on it computes an identical transform."""
    S = np.random.default_rng(4).normal(size=(20, 12))
    T = np.random.default_rng(5).normal(size=(35, 12))
    Tp = T[np.random.default_rng(6).permutation(len(T))]
    a1 = al.coral_transform(S, T)["A"]; a2 = al.coral_transform(S, Tp)["A"]
    assert al.transform_distance({"A": a1, "b": np.zeros(12)},
                                 {"A": a2, "b": np.zeros(12)}) < 1e-12

def test_independent_column_shuffle_preserves_marginals_and_kills_joint():
    T = np.random.default_rng(7).normal(size=(35, 12)); T[:, 1] += 0.8 * T[:, 2]
    S, _vecs = al.independent_column_shuffle(T, 20260936)
    assert np.allclose(S.mean(0), T.mean(0), atol=1e-12)
    assert np.allclose(S.var(0), T.var(0), atol=1e-12)
    off_T = np.abs(np.corrcoef(T.T) - np.eye(12)).max()
    off_S = np.abs(np.corrcoef(S.T) - np.eye(12)).max()
    assert off_S < off_T
    src = np.random.default_rng(8).normal(size=(20, 12))
    d = al.transform_distance(al.coral_transform(src, T), al.coral_transform(src, S))
    assert d > 1e-6                                   # the placebo must actually differ

def test_shuffle_is_deterministic_and_reversible_per_column():
    T = np.random.default_rng(9).normal(size=(35, 12))
    A, v1 = al.independent_column_shuffle(T, 20260936)
    B, v2 = al.independent_column_shuffle(T, 20260936)
    assert np.array_equal(A, B) and np.array_equal(v1, v2)
    for j in range(12):
        assert sorted(A[:, j].tolist()) == sorted(T[:, j].tolist())   # bijection per column

def test_no_labels_and_no_torch_in_the_module():
    src = pathlib.Path("alignment.py").read_text(encoding="utf-8")
    import inspect
    for fn in (al.coral_transform, al.apply_transform, al.independent_column_shuffle):
        assert not {"y", "labels", "target_y", "Y", "camera"} & set(
            inspect.signature(fn).parameters)
    assert not re.search(r"^\s*(import torch|from torch|import timm)", src, re.M)

def test_rank_deficient_target_does_not_produce_nan():
    T = np.tile(np.random.default_rng(10).normal(size=(1, 12)), (35, 1))   # all rows equal
    tr = al.coral_transform(np.random.default_rng(11).normal(size=(20, 12)), T)
    assert np.isfinite(tr["A"]).all() and np.isfinite(tr["b"]).all()
```

- [ ] **Step 2: Run it and confirm it fails.** `python check_alignment.py` →
      expected `ModuleNotFoundError: No module named 'alignment'` or FAILs on every check.

- [ ] **Step 3: Implement `alignment.py`.** Docstring states the one responsibility and that it
      knows nothing about cameras, labels or files. Core:

```python
def covariance(X):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(X)
    return lw.covariance_, float(lw.shrinkage_)


def _sym_pow(C, power, tol=1e-10):
    w, V = np.linalg.eigh((C + C.T) / 2.0)
    w = np.where(w > tol, w, 0.0)
    f = np.where(w > 0.0, w ** power, 0.0)
    return (V * f) @ V.T


def coral_transform(source, target, match_cov=True):
    source = np.asarray(source, float); target = np.asarray(target, float)
    if source.shape[1] != target.shape[1]:
        raise AssertionError("dimension mismatch")
    ms, mt = source.mean(0), target.mean(0)
    ws = wt = 0.0
    if not match_cov:
        A, b = np.eye(source.shape[1]), mt - ms
    else:
        Cs, ws = covariance(source); Ct, wt = covariance(target)
        A = _sym_pow(Cs, -0.5) @ _sym_pow(Ct, 0.5)
        b = mt - ms @ A
    for v, nm in ((A, "A"), (b, "b")):
        if not np.isfinite(v).all():
            raise AssertionError(f"non-finite {nm} in CORAL transform")
    return {"A": A, "b": b, "match_cov": bool(match_cov),
            "shrink_source": float(ws), "shrink_target": float(wt)}


def apply_transform(X, T):
    return np.asarray(X, float) @ T["A"] + T["b"]


def independent_column_shuffle(target, seed):
    rng = np.random.default_rng(int(seed)); n = len(target)
    vecs = np.array([rng.permutation(n) for _ in range(target.shape[1])])
    out = np.empty_like(target)
    for j in range(target.shape[1]):
        out[:, j] = target[vecs[j], j]
    return out, vecs


def transform_distance(T1, T2):
    return float(max(np.abs(T1["A"] - T2["A"]).max(), np.abs(T1["b"] - T2["b"]).max()))
```

- [ ] **Step 4: Run the test.** `python check_alignment.py` → expected
      `CHECK ALIGNMENT: PASSED`.

- [ ] **Step 5: Prove the test bites.** Temporarily make `independent_column_shuffle` shuffle rows
      instead of columns; expect `test_independent_column_shuffle_preserves_marginals_and_kills_joint`
      to FAIL because the placebo then equals the real arm. Revert and re-run to PASSED.

- [ ] **Step 6: Checkpoint** with the owner.

---

## Task 3: `transfer_eval.py` — data assembly, directions, nested alpha, clustered bootstrap

**Files:**
- Create: `Model 5/Model 5 Experiment 1/transfer_eval.py`
- Test: `Model 5/Model 5 Experiment 1/check_transfer_eval.py`

**Interfaces:**
- Consumes: `data/processed_meta/manifest_images.csv`, `manifest_tiles.csv`, `manifest_samples.csv`,
  `data/sample_submission.csv`, `Model 1/Model 1 Experiment 3/.cache/tile_features_256_<hash>.csv`,
  `Model 2/Model 2 Experiment 3/cv_families.csv`, `alignment.coral_transform`
- Produces:
  - `FEATURES: tuple` — the 12 control feature names, ordered
  - `ALPHAS: tuple[float, ...]` — the frozen grid, floor `0.03`, ceiling `1000000.0`
  - `image_features(root) -> pd.DataFrame` — index `image_id`; cols `FEATURES + ["sample_id", "camera_fam", "split"]`
  - `soil_from_images(img) -> pd.DataFrame` — index `sample_id`, cols `FEATURES`
  - `alignment_target(img) -> np.ndarray` — the 35 test images' feature rows
  - `direction(name, img, labels, fams) -> dict` with keys `name`, `fit_ids`, `eval_ids`,
    `fit_families`, `n_fit`, `n_eval`
  - `alpha_status(a) -> str` in `{"FLOOR", "INTERIOR", "CEILING"}`
  - `nested_predict(train_X, train_Y, train_fam, test_X, alphas) -> dict`
    with `pred`, `alpha`, `status`, `selection_emd`, `best_emd`, `grid_spread`, `sweep`
  - `clustered_bootstrap(rows, clusters, n=4000, seed=20260934) -> tuple[float, float, float]`
  - `naive_bootstrap(rows, n, seed) -> tuple[float, float, float]`

- [ ] **Step 1: Write the failing test first.**

```python
def test_counts_match_the_spec():
    """24 train soils, 23 Motorola, 22 Samsung, 21 both, 45 eval rows, 24 clusters.
    These numbers are the experiment's whole statistical universe; a silent change here
    changes what the thresholds mean."""
    img = te.image_features(ROOT)
    fams = pd.read_csv(E3_DIR / "cv_families.csv").set_index("sample_id").cv_family
    labels = pd.read_csv(META / "manifest_samples.csv")
    dA = te.direction("A", img, labels, fams)
    dB = te.direction("B", img, labels, fams)
    assert dA["n_fit"] == 23 and dA["n_eval"] == 22
    assert dB["n_fit"] == 22 and dB["n_eval"] == 23
    assert dA["n_eval"] + dB["n_eval"] == 45
    assert len(set(dA["eval_ids"]) & set(dB["eval_ids"])) == 21

def test_alignment_target_is_test_only():
    img = te.image_features(ROOT)
    tgt = te.alignment_target(img)
    assert tgt.shape == (35, 12)
    assert set(img.loc[img.split == "test"].camera_fam) == {"iPhone"}
    assert not (set(img.loc[img.split == "train", "sample_id"])
                & set(img.loc[img.split == "test", "sample_id"]))

def test_g0_reproduces_model_2_e3_control():
    """The sanity gate from spec G0: the no-holdout, unaligned configuration on all 24 soils
    must reproduce Model 2 E3's recorded 43.0217 within 0.05 EMD. This is what proves this
    module's head is the frozen head and not a lookalike."""
    ref = float(pd.read_csv(E3_DIR / "sweep_results.csv")
                .set_index("arm").loc["M1", "nested_in_domain"])
    got = te.no_holdout_control_score(ROOT)
    assert abs(got - ref) <= 0.05, f"spike {got:.4f} vs E3 {ref:.4f}"

def test_clustered_ci_is_wider_than_naive():
    """A soil appearing in both directions contributes two correlated rows. Resampling rows
    would treat them as independent and narrow every CI. Clusters must be strictly wider."""
    rows = np.concatenate([np.full(21, 3.0), np.full(21, 3.0), [1.0], [1.0], [-2.0]])
    clusters = ([f"S{i}" for i in range(21)] * 2) + ["X1", "X2", "X3"]
    m_naive, n_lo, n_hi = te.naive_bootstrap(rows, 4000, 20260934)
    m_clus, c_lo, c_hi = te.clustered_bootstrap(rows, clusters, 4000, 20260934)
    assert (c_hi - c_lo) > (n_hi - n_lo),         f"clustered CI {(c_lo, c_hi)} is not wider than naive {(n_lo, n_hi)}"
    assert abs(m_naive - m_clus) < 1e-9                 # same point estimate, wider interval

def test_alpha_status_classification():
    assert te.alpha_status(0.03) == "FLOOR"
    assert te.alpha_status(1000000.0) == "CEILING"
    assert te.alpha_status(300.0) == "INTERIOR"

def test_nested_predict_contract():
    """Alpha comes from the inner LOGO curve only, so what is reported is the score of the
    procedure. The selection is the argmin of that curve, therefore there is no separate
    oracle alpha to report and the boundary status of the SELECTED alpha is the fact
    that matters (spec 5a). Predictions must be legal CDFs."""
    rng = np.random.default_rng(12)
    X = rng.normal(size=(23, 12))
    Y = np.clip(np.tile(np.linspace(0, 100, 11), (23, 1)) + rng.normal(scale=6.0, size=(23, 11)),
                0, 100)
    fam = [f"F{i % 8}" for i in range(23)]
    r = te.nested_predict(X, Y, fam, X[:5], te.ALPHAS)
    assert r["alpha"] in te.ALPHAS and r["status"] in ("FLOOR", "INTERIOR", "CEILING")
    assert r["pred"].shape == (5, 11) and np.isfinite(r["pred"]).all()
    assert (np.diff(r["pred"], axis=1) >= -1e-9).all(), "prediction is not monotone"
    assert abs(r["pred"][:, -1] - 100.0).max() < 1e-9, "200mm support must be forced to 100"
    assert r["selection_emd"] <= r["best_emd"] + 1e-12
    with np.errstate(all="ignore"):
        assert np.all(np.isfinite(r["sweep"][a]) for a in te.ALPHAS)

def test_no_label_argument_reaches_alignment():
    import inspect
    src = inspect.getsource(te.alignment_target)
    assert "labels" not in src and "Y" not in src
```

- [ ] **Step 2: Run it; expect failure** (module absent).

- [ ] **Step 3: Implement `transfer_eval.py`.** Key points, each of which the test above pins:
  aggregate tile→image by **median** over `parent_image_path`; image→soil by **median** over
  `sample_id`; direction A is `camera_fam == "Motorola"` as fit and `"Samsung"` as eval, B mirrored;
  families come from E3's `cv_families.csv` unchanged and are restricted to the fit soils;
  `nested_predict` selects alpha by plain argmin over inner-fold mean error, then fits on all
  training rows and predicts; `no_holdout_control_score` runs the full 24-soil nested LOGO-CV with
  identity alignment — the number G0 compares.

```python
def nested_predict(train_X, train_Y, train_fam, test_X, alphas):
    fams = sorted(set(train_fam))
    if len(fams) < 2:
        raise AssertionError("need at least two families to select alpha")
    err = {}
    for al in alphas:
        per = []
        for F in fams:
            hold = [i for i, f in enumerate(train_fam) if f == F]
            keep = [i for i in range(len(train_fam)) if i not in set(hold)]
            P = fit_predict(train_X[keep], train_Y[keep], train_X[hold], al)
            per += [emd_pair(P[k], train_Y[i]) for k, i in enumerate(hold)]
        err[al] = float(np.mean(per))
    alpha = min(err, key=err.get)          # argmin of the inner LOGO curve
    pred = fit_predict(train_X, train_Y, test_X, alpha)
    return {"pred": pred, "alpha": alpha, "status": alpha_status(alpha),
            "selection_emd": err[alpha], "best_emd": min(err.values()),
            "grid_spread": max(err.values()) - min(err.values()), "sweep": err}
    # `alpha` IS the argmin of the sweep here, so there is no separate "oracle alpha" to
    # report; the boundary status of the selected alpha is the fact that matters (spec 5a).
```

- [ ] **Step 4: Run it** → `CHECK TRANSFER EVAL: PASSED`, including that G0 prints a gap ≤ 0.05.

- [ ] **Step 5: Prove it bites.** Temporarily change the direction A fit filter to include Samsung;
      expect `test_counts_match_the_spec` to FAIL on `n_fit`. Revert.

- [ ] **Step 6: Checkpoint** with the owner, quoting the G0 number actually obtained.

---

## Task 4: Notebook generator and the gate implementation

**Files:**
- Create: `scratch/build_m5e1.py`, `scratch/mk_run6.py`, `scratch/run_nb_m5e1.py`
- Create (generated): `Model 5/Model 5 Experiment 1/Model5_Experiment1.ipynb`

**Interfaces:**
- Consumes: `alignment.py`, `transfer_eval.py`, verbatim cells from E3's notebook
- Produces: `gate.json`, `transfer_results.csv`, `per_soil_transfer.csv`, `alignment_report.csv`,
  `placebo_report.csv`, `canary.json`, `predictions.csv`, `Experiment1.txt`, and
  `Submission_Model5_E1.csv` **only** on verdict `SUPPORTED`

- [ ] **Step 1: Write `build_m5e1.py`** on the E3 pattern: a `reuse(name)` helper that pulls cell
      text byte-for-byte out of `Model2_Experiment3.ipynb`, plus authored cells. Import verbatim at
      minimum `A4` (config-hash and submission schema), `B2` (EMD and its reconciliation), `B3`
      (no-image floor and the six external leaderboard scores). Print a provenance table and inject
      it into the record, as E3's builder does.

- [ ] **Step 2: Author the config cell.**

```python
MODEL_ID = "Model5"; EXPERIMENT_ID = "E1"; SEED = 20260935
BOOT_SEED, SHUF_SEED = 20260934, 20260936
ARMS = ("NONE", "CORAL", "MEAN", "CORAL_INDEP", "CORAL_SELF", "CORAL_TILE", "CORAL_SOIL")
GATED = "CORAL"; DIRECTIONS = ("A", "B")
G1_MIN, G2_MIN, G4_MIN, G5_MAX, G0_TOL = 3.00, 1.00, 1.50, 0.50, 0.05
VERDICTS = ("SUPPORTED", "SUPPORTED-BELOW-BAND", "SUPPORTED-WITH-BOUNDARY-CAVEAT",
            "REFUTED", "REFUTED-WITH-BOUNDARY-CAVEAT", "REGULARISATION-ARTEFACT",
            "DIRECTION-INCONSISTENT", "NOT-RESOLVED", "INCONCLUSIVE",
            "INVALID-IMPLEMENTATION")
```

- [ ] **Step 3: Author the arm-construction cell.** For each direction, build the image-level source
      and eval tables, compute the transform per arm, and apply **the same transform to both sides**:

```python
tgt_real = te.alignment_target(img)
tgt_indep, _vecs = al.independent_column_shuffle(tgt_real, SHUF_SEED)
UNIT = np.eye(12), np.zeros(12)
T = {
  "NONE":        {"A": np.eye(12), "b": np.zeros(12)},
  "CORAL":       al.coral_transform(src_img, tgt_real),
  "MEAN":        al.coral_transform(src_img, tgt_real, match_cov=False),
  "CORAL_INDEP": al.coral_transform(src_img, tgt_indep),
  "CORAL_SELF":  al.coral_transform(src_img, src_img),
  "CORAL_TILE":  al.coral_transform(src_tile, tgt_tile),
  "CORAL_SOIL":  al.coral_transform(src_soil, tgt_soil),
}
assert al.transform_distance(T["NONE"], T["CORAL_SELF"]) <= 1e-6
assert al.transform_distance(T["CORAL"], T["CORAL_INDEP"]) > 1e-6, \
    "the placebo collapsed onto the real arm - the flaw that killed CORAL_PERM"
```

- [ ] **Step 4: Author the scoring cell** producing, per arm and direction, the per-soil paired
      difference `Δ(s) = err_NONE(s) − err_arm(s)`, the nested alpha, its boundary status, the
      selected alpha's own LOGO error and the grid spread — one row per (arm, direction), and one
      row per (arm, direction, soil).

- [ ] **Step 5: Author the bootstrap cell.** Pooled `Δ` over all 45 rows with
      `te.clustered_bootstrap(...)`; per-direction `Δ` with the same routine restricted to that
      direction's clusters; plus the naive comparison printed as a diagnostic, labelled as such.

- [ ] **Step 6: Author the gate cell exactly as spec 5 and 5a.**

```python
# Step 5 produces, for every arm: POOL[arm] (45 paired deltas), CLUSTERS[arm] (the soil ids
# behind those rows, so a soil's two rows move together), DIR[arm] = {"A": mean, "B": mean},
# STATUS[direction] from nested_predict, and the per-direction CIs dict
# CI[arm] = {"A": (mean, lo, hi), "B": (mean, lo, hi), "pooled": (mean, lo, hi)}.
pooled, lo, hi = CI[GATED]["pooled"]
mA, a_lo, a_hi = CI[GATED]["A"]; mB, b_lo, b_hi = CI[GATED]["B"]
indep_p = CI["CORAL_INDEP"]["pooled"][0]
dd = POOL[GATED] - POOL["CORAL_INDEP"]
diff, dlo, dhi = te.clustered_bootstrap(dd, CLUSTERS["DIFF"], 4000, BOOT_SEED)
self_p, slo, shi = te.clustered_bootstrap(POOL["CORAL_SELF"], CLUSTERS["CORAL_SELF"], 4000, BOOT_SEED)
self_A, self_B = DIR["CORAL_SELF"]["A"], DIR["CORAL_SELF"]["B"]

G0 = canary_g0_ok                                   # Task 3's G0 number, recomputed in-run
G1 = pooled >= G1_MIN
G2 = (mA >= G2_MIN) and (mB >= G2_MIN) and not (a_lo < 0 and a_hi < 0) and not (b_lo < 0 and b_hi < 0)
G3 = lo > 0.0
G4 = (diff >= G4_MIN) and (dlo > 0.0) and (indep_p <= pooled)
G5 = (abs(self_p) <= G5_MAX) and max(abs(self_A), abs(self_B)) <= G5_MAX
BOUND = [s for s in (STATUS["A"], STATUS["B"]) if s != "INTERIOR"]
G6 = len(BOUND) < 2                      # two boundary hits => INCONCLUSIVE, gate cannot pass
CAVEAT = len(BOUND) == 1
G7 = label_blind_assertions_passed

if not G0 or not G7:        VERDICT = "INVALID-IMPLEMENTATION"
elif len(BOUND) == 2:       VERDICT = "INCONCLUSIVE"
elif G1 and G2 and G3 and G4:
    VERDICT = "SUPPORTED-WITH-BOUNDARY-CAVEAT" if CAVEAT else "SUPPORTED"
elif not G4 and G1 and G2 and G3: VERDICT = "REGULARISATION-ARTEFACT"
elif not G2:                VERDICT = "DIRECTION-INCONSISTENT"
elif G1 and not G3:         VERDICT = "NOT-RESOLVED"
else:                       VERDICT = "REFUTED-WITH-BOUNDARY-CAVEAT" if CAVEAT else "REFUTED"
SUBMIT = (VERDICT == "SUPPORTED")        # the caveat case never submits, by design
```

- [ ] **Step 7: Author the boundary report cell** printing, per arm and per direction, selected
      alpha, `FLOOR|INTERIOR|CEILING`, selected-alpha LOGO error, grid spread, and `symmetric_boundary`
      diagnostic against `NONE`. Nothing here may change a verdict.

- [ ] **Step 8: Author the submission cell** implementing spec section 7: production recipe only,
      verdict must be exactly `SUPPORTED`, the eight structural gates reused from E3 must pass, and
      when the verdict is anything else the arm is still fitted and the frame validated with no file
      written, labelled as a plumbing check.

- [ ] **Step 9: Author the predictions cell** evaluating P1–P6 mechanically, with P4 explicitly
      separated from G5: drift between 0.10 and 0.50 refutes P4 while leaving the run valid.

- [ ] **Step 10: Author the record cell** writing `Experiment1.txt` from live objects, including the
      verbatim gate block, the boundary table, spec section 13's correction, and a
      scope-of-verification section stating what executed.

- [ ] **Step 11: Static self-checks in the builder**, failing the build on any hit: no `import
      torch`/`timm`; no `camera` used as a feature key; the six numeric thresholds appearing exactly
      once each; `SUBMIT = (VERDICT == "SUPPORTED")` present; every verdict name in `VERDICTS`
      reachable.

- [ ] **Step 12: Generate and compile.** `python scratch/mk_run6.py` — regenerate, compile every
      cell independently, then flatten, then report the provenance table.

- [ ] **Step 13: Checkpoint** with the owner before running.

---

## Task 5: Run Model 5 E1 for real, locally

**Files:** outputs in `Model 5/Model 5 Experiment 1/` per spec section 9.

- [ ] **Step 1:** `python scratch/run_nb_m5e1.py` — executes the shipped notebook.
- [ ] **Step 2:** Confirm G0 printed a gap ≤ 0.05 against E3's `43.0217`, and that `canary.json`
      records both G0 and G5 outcomes.
- [ ] **Step 3:** Confirm the placebo actually differs: `placebo_report.csv` shows
      `transform_distance(CORAL, CORAL_INDEP) > 1e-6`; if it is 0, stop — the shuffle collapsed.
- [ ] **Step 4:** Scan for cells with empty output; every cell must print.
- [ ] **Step 5:** Determinism — run twice, byte-compare every emitted data file.
- [ ] **Step 6:** `python -m preprocess.final_check` must PASS; `git status` must show nothing
      modified under `data/`, `Model 1/` or `Model 2/`.
- [ ] **Step 7:** Read, in order: the boundary report, `transfer_results.csv`,
      `gate.json` clause by clause, `predictions.csv`. Report all of it to the owner verbatim,
      including which clause failed if the gate did not pass.

---

## Task 6: Kaggle and dataset — decided, none

- [ ] **Step 1:** Record in `Experiment1.txt`: Model 5 imports no torch, needs no GPU, no Kaggle run,
      no dataset and no zip. Existing zips are untouched.
- [ ] **Step 2:** Confirm at the end that no zip was rebuilt and no dataset created or modified.

---

## Task 7: Decide and record

- [ ] **Step 1:** Transcribe the verdict and apply the spec section 6 interpretation table.
- [ ] **Step 2:** If `REFUTED` with no boundary caveat, apply the section 10 closure: Model 5 closes
      at one experiment. If `REFUTED-WITH-BOUNDARY-CAVEAT`, record that closure is **withheld**.
- [ ] **Step 3:** Write the outcome into `Model 5/instructions.txt` (create it), and into
      `Model 1/instructions.txt`'s cancelled list only if a direction is now dead.
- [ ] **Step 4:** Update `AGENT_BRIEF.md` section 13 and section 9's ledger, tagged CONFIRMED /
      REFUTED / UNVERIFIED, and correct section 13's Model 2 line if it still says E3 is unrun.
- [ ] **Step 5:** Update project memory with the verdict and the two reusable lessons: row permutation
      cannot placebo a moment-matching method, and a gate clause must be tested for the ability to
      fail before it is trusted to discriminate.
- [ ] **Step 6:** Only on verdict `SUPPORTED`: produce `Submission_Model5_E1.csv` (Task 4 step 8),
      and state in the record that the submitted configuration was never itself evaluated by the
      gate. Then stop and hand the owner the file — uploading it is their act, not this plan's step.

---

## Self-review against the spec

- **Coverage.** Spec §0 H5 → Task 1 and the record cell. §1 facts → Task 3 test 1. §2 frozen/prohibited
  → Task 2 test 7, Task 3 tests 2 and 7, Task 4 step 11. §3 arms → Task 4 step 3. §4 measurement and
  resampling unit → Task 3 `clustered_bootstrap` and Task 4 step 5. §5 G0–G7 → Task 4 step 6. §5a
  boundary rules per direction → Task 4 steps 6 and 7. §6 verdicts and P1–P6 → Task 4 steps 9 and 10.
  §7 submission → Task 4 step 8, Task 7 step 6. §8 errors → Tasks 3, 4, 5. §9 artifacts → Task 4.
  §10 closure → Task 7 step 2. §11 pre-registration → Global constraints. §13 placebo correction →
  Task 2 tests 4 and 5, Task 4 step 3's assertion.
- **Placeholders:** none; every code step shows code.
- **Type consistency:** `coral_transform` returns keys `A, b, match_cov, shrink_source, shrink_target`
  used identically in Tasks 2, 3 and 4; `nested_predict` returns `pred, alpha, status, selection_emd,
  best_emd, grid_spread, sweep` consumed by Task 4 step 4 and step 7; `clustered_bootstrap` returns
  `(mean, lo, hi)` everywhere it is called.
