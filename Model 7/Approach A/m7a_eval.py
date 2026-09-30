"""m7a_eval.py - the family-honest arm M7-A is scored with, and the block variants around it.

The head is IMPORTED, never reimplemented: transfer_eval supplies the 12 features, the ridge, the
rank-3 curve basis, the monotone projection and the EMD, exactly as Models 2 and 5 used them. The
only difference between the anchor and the gated arm is the presence of four columns, which is what
makes the registered verdicts (SUPPORTED, SUPPORTED-BELOW-BAND, SELECTION-OR-CAPACITY-ARTEFACT,
NOT-RESOLVED, REFUTED) distinguishable at all rather than artifacts of a re-typed pipeline.

WHAT IS HERE
  soil_design(root, block)  the 24 x 12 anchor design, or the same 12 with the A1-A4 block appended
      as "real", zeroed as "zero", or replaced by a placebo as "rand" / "shuf" / "cols". Blocks
      "rand", "shuf" and "cols" are wired for Task 7 and are not run by Task 4.
  lofo_errors(X, Y, fams)   one honest error per soil: leave-one-CV-family-out, with the held-out
      soil's entire family absent from its fitting set AND from every fit that informs the ridge
      alpha used for it. That nesting is the registered Model 7 protocol (owner ruling, 2026-09-30):
      the outer family never enters fitting or alpha selection, alpha is chosen only inside the
      outer training families, the grid is the frozen 16-value ALPHAS, and the PCA/ridge/projection/
      EMD machinery is the imported one. The reported number is the score of the procedure, never an
      oracle's, and the same folds and protocol are used for the baseline and for every candidate,
      control and placebo, so no gain is ever a protocol difference.
  fold_report               the integrity claim made checkable: which soils were scored, and whether
      any fold ever fitted on the family it scores.
  g0_gap                    |reproduced anchor - 43.0217308796477|, which must be exactly 0.0.

WHAT IS NOT HERE, deliberately
  No gain, no verdict, no bootstrap, no separability decision, no prediction for a test soil, no
  submission. Those are Tasks 5-10. This module builds matrices and scores labelled training soils
  with the registered protocol; it decides nothing.

LEAKAGE NOTES, because they are the reason this file exists rather than a notebook cell
  * cv_families.csv is read for the fold key and nothing else. It is never concatenated into X, and
    no family label is derived from a device or camera name - the same key Model 2 E3 and Model 5 E1
    used, so the folds are byte-identical to the ones that produced the anchor.
  * normalized_ppm is the physical scale argument the statistics need and is dropped before a design
    frame exists; camera, camera_fam, cv_group, source_* are dropped at load in m7a_data, so a
    design assembled here cannot be selected on them.
  * The oracle alpha sweep exists in transfer_eval as a DIAGNOSTIC and is never called here.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

# --- frozen constants. ANCHOR is the registered G0 value, transcribed once and asserted against. --
# FOUR NUMBERS, FOUR DIFFERENT THINGS. They are not interchangeable and none of them is a Kaggle
# score. The only external evidence this project has is an actual submission; an internal CV mean is
# a measurement on 24 labelled soils, evaluated by a ruler whose per-soil SD is 22.75 EMD.
#   43.0217308796477  the HISTORICAL G0 anchor: internal, produced by the frozen
#                       Model 5/Model 2 no_holdout_control_score protocol, reproduced here to 0.0.
#   43.453225201811563 the MODEL 7 INTERNAL no-block BASELINE under the strictly family-honest
#                       nested protocol this module implements (owner ruling, 2026-09-30).
#   60.56167           a PREVIOUS EXTERNAL KAGGLE SUBMISSION RESULT. Different scale entirely.
#   future Kaggle score  unknown until a submission CSV is generated and submitted.
# Model 7 gains are computed against INTERNAL_BASELINE, never against ANCHOR: the two numbers come
# from different selection protocols and mixing them would put 0.431 EMD of protocol difference into
# a quantity being compared to a 3.00 EMD bar.
ANCHOR = 43.0217308796477
INTERNAL_BASELINE = 43.453225201811563
# The comparator every Model 7 gain is measured against. It is the nested baseline, not the
# historical anchor: subtracting one protocol's number from another's would smuggle 0.431494322163864
# EMD of protocol difference into a quantity judged against a 3.00 EMD bar.
GAIN_BASELINE = INTERNAL_BASELINE
# G6's registered numerical tolerance, from the owner's ruling: bit-for-bit equality is not
# required, and the frozen head is not modified to obtain it.
G6_TOLERANCE = 1e-12
HEAD_COLUMNS: Tuple[str, ...] = tuple(tv.FEATURES)
BLOCK_COLUMNS: Tuple[str, ...] = tuple(sf.SPATIAL)
BLOCKS = ("none", "zero", "real", "rand", "shuf", "cols")
# control seeds, fixed now so a run is reproducible from the number it reports rather than from a
# global RNG state set somewhere else. Task 4 builds none of these arms.
SEED_RAND, SEED_SHUF, SEED_COLS, SEED_PERM = 90001, 90002, 90003, 90004

# The frozen 12-feature assembly and the A1-A4 table each cost minutes to build once, so they are
# memoised per root. This is a cache of the SAME imported call, not a second implementation.
_X12: Dict[str, np.ndarray] = {}
_LAB: Dict[str, Tuple[List[str], np.ndarray]] = {}


def families() -> Dict[str, str]:
    """sample_id -> CV family, read as a fold key only. Never a column, never a device label."""
    t = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    return dict(zip(t.sample_id.astype(str), t.cv_family.astype(str)))


def labels(root) -> Tuple[List[str], np.ndarray]:
    """The 24 labelled training soils and their 11-support curves, from the frozen loader."""
    key = str(Path(root))
    if key not in _LAB:
        ids, Y = tv.labels_matrix(root)
        _LAB[key] = ([str(i) for i in ids], np.asarray(Y, float))
    return _LAB[key]


def head_design(root) -> np.ndarray:
    """The existing 12 features, per soil, in the frozen column order. Imported, not recomputed."""
    key = str(Path(root))
    if key not in _X12:
        ids, _ = labels(root)
        img = tv.image_features(root)
        _X12[key] = tv.soil_from_images(img).loc[ids, list(HEAD_COLUMNS)].to_numpy(float)
    return _X12[key]


def block_design(root, block: str) -> np.ndarray:
    """The 24 x 4 side of the design for the requested variant.

    "rand" re-runs the frozen tile pass with phase-scrambled tiles and re-aggregates, so the placebo
    is the same pipeline on rearranged pixels rather than a rescaled copy of the real block.
    """
    ids, _ = labels(root)
    if block == "real":
        return d.spatial_soil_table(root).loc[ids, list(BLOCK_COLUMNS)].to_numpy(float)
    if block == "zero":
        return np.zeros((len(ids), len(BLOCK_COLUMNS)))
    if block == "rand":
        return d.spatial_soil_table(root, scrambled=True, seed=SEED_RAND).loc[
            ids, list(BLOCK_COLUMNS)].to_numpy(float)
    if block == "shuf":
        B = d.spatial_soil_table(root).loc[ids, list(BLOCK_COLUMNS)].to_numpy(float)
        return sf.shuffle_columns(B, SEED_SHUF)[0]
    if block == "cols":
        return sf.linear_combinations(head_design(root), SEED_COLS)
    raise AssertionError("unknown block %r; the registered variants are %s" % (block, BLOCKS))


def soil_design(root, block: str = "none") -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Returns (X, Y, fam_list, ids). fam_list is a per-soil fold key, never a column of X.

    A non-finite cell anywhere is an abort rather than a fill: the pipeline standardises and reduces
    by rank-3 PCA, and a single NaN or inf would travel into every soil's curve silently.
    """
    if block not in BLOCKS:
        raise AssertionError("unknown block %r; the registered variants are %s" % (block, BLOCKS))
    ids, Y = labels(root)
    X12 = head_design(root)
    X = X12 if block == "none" else np.hstack([X12, block_design(root, block)])
    if not np.isfinite(X).all():
        raise AssertionError("non-finite design matrix for block %r" % block)
    if not np.isfinite(Y).all():
        raise AssertionError("non-finite label matrix; the curves are the thing being predicted")
    fam_of = families()
    missing = [i for i in ids if i not in fam_of]
    if missing:
        raise AssertionError("%d soils have no CV family, so a fold would silently drop them: %s"
                             % (len(missing), ", ".join(missing[:5])))
    return X, Y, [fam_of[i] for i in ids], ids


def alpha_curve(X, Y, fams) -> Dict[float, float]:
    """Pooled family-out error at every grid alpha, computed on the training rows handed to it."""
    F = np.asarray(fams)
    inner = sorted(set(F))
    if len(inner) < 2:
        raise AssertionError("alpha selection needs at least two families to be honest")
    curve = {}
    for al in tv.ALPHAS:
        errs = []
        for g in inner:
            hold = np.where(F == g)[0]
            keep = np.where(F != g)[0]
            if not hold.size or not keep.size:
                continue
            P = tv.fit_predict(X[keep], Y[keep], X[hold], al)
            errs.extend(tv.emd_pair(P[k], Y[i]) for k, i in enumerate(hold))
        curve[float(al)] = float(np.mean(errs)) if errs else float("inf")
    return curve


def select_alpha(X, Y, fams) -> float:
    """The grid minimum on this training set. An oracle is never consulted anywhere in this file."""
    curve = alpha_curve(X, Y, fams)
    return min(curve, key=curve.get)


def lofo_errors(X, Y, fams, return_alphas: bool = False):
    """Leave-one-CV-family-out: one honest error per soil, alpha re-selected inside each fit set."""
    F = np.asarray(fams)
    out = np.full(len(X), np.nan)
    alphas = {}
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        if not te.size or not tr.size:
            raise AssertionError("fold %s is empty on one side, so it cannot be scored honestly" % g)
        alphas[g] = select_alpha(X[tr], Y[tr], list(F[tr]))
        P = tv.fit_predict(X[tr], Y[tr], X[te], alphas[g])
        for k, i in enumerate(te):
            out[i] = tv.emd_pair(P[k], Y[i])
    if np.isnan(out).any():
        raise AssertionError("a soil was never scored")
    return (out, alphas) if return_alphas else out


def lofo_predictions(X, Y, fams, alphas: Dict[str, float]) -> np.ndarray:
    """The per-soil predicted curves under the GIVEN alphas, one row per soil, 11 supports wide.

    Separate from lofo_errors on purpose: G6 compares predictions at identical alphas, so the alpha
    selection step must be held fixed while the curves are looked at. The predicted curve is the
    projected output (clip, cummax, 100 at 200 mm), so a difference here is a difference in what the
    metric would see, not in how a mean was accumulated.
    """
    F = np.asarray(fams)
    out = np.full((len(X), np.asarray(Y).shape[1]), np.nan)
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        if g not in alphas:
            raise AssertionError("no alpha recorded for fold %s, so predictions would be invented"
                                 % g)
        P = tv.fit_predict(X[tr], Y[tr], X[te], alphas[g])
        for k, i in enumerate(te):
            out[i] = np.asarray(P[k], float)
    if not np.isfinite(out).all():
        raise AssertionError("a soil's predicted curve is missing or non-finite")
    return out


def fold_report(X, Y, fams) -> Tuple[Dict[str, object], List[int]]:
    """The integrity claim, made checkable: who was scored, and whether any fold leaked its own set."""
    F = np.asarray(fams)
    seen, leaks, sizes = [], 0, {}
    for g in sorted(set(F)):
        te = list(np.where(F == g)[0])
        tr = list(np.where(F != g)[0])
        seen += te
        sizes[g] = len(te)
        if set(F[te]) & set(F[tr]):
            leaks += 1
        if set(te) & set(tr):
            leaks += 1
        if len({F[i] for i in tr}) < 2:
            leaks += 1
    return {"family_leaks": leaks, "n_folds": len(set(F)), "n_scored": len(set(seen)),
            "fold_sizes": sizes}, sorted(seen)


def g0_gap(root) -> float:
    """G0: the deviation of the reproduced anchor from the registered value. Must be exactly 0.0."""
    return abs(float(tv.no_holdout_control_score(root)) - ANCHOR)


def design_hash(X) -> str:
    """Content-sensitive hash of a design, for the determinism legs."""
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(X, float).tobytes()).hexdigest()[:16]
