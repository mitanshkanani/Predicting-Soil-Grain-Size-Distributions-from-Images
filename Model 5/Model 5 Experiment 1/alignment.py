"""alignment.py - the one new idea in Model 5, isolated from everything it could corrupt.

Model 5 inserts exactly one step into an otherwise frozen pipeline:

    tiles -> image median -> ALIGN -> soil median -> frozen head

This module is that step and nothing else. It knows nothing about cameras, soils, labels, files or
experiments: it takes feature matrices and returns a linear map. That narrow surface is what makes
the label-free requirement checkable rather than aspirational - there is no argument here into
which a label or a device name could be smuggled.

What the map is. CORAL finds the linear transform that carries the source feature distribution's
second-order structure onto the target's, on top of a mean shift, so that a model fitted on the
source is expressed in the target's coordinates. Both covariance matrices are shrunk toward a
scaled identity (Ledoit-Wolf) before the inverse square root is taken, because the target here is
35 images against a 12-dimensional covariance - unshrunk, that estimate is barely more than noise
and its inverse square root amplifies exactly the directions where there is no information.

Deliberately absent: any use of labels, any use of camera or device identity, any conditioning on
which image a row came from, and any file access. The alignment target is passed in as a matrix;
deciding what that matrix is remains the notebook's job, where the disjointness assertion lives.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["covariance", "coral_transform", "apply_transform",
           "independent_column_shuffle", "transform_distance"]


def covariance(X: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance and the shrinkage weight that was applied."""
    from sklearn.covariance import LedoitWolf
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] < 2:
        raise AssertionError(f"need at least 2 rows to estimate a covariance, got {X.shape[0]}")
    lw = LedoitWolf().fit(X)
    return lw.covariance_, float(lw.shrinkage_)


def _sym_pow(C: np.ndarray, power: float, tol: float = 1e-10) -> np.ndarray:
    """C**power for a symmetric positive semi-definite matrix, via its eigendecomposition.

    Eigenvalues at or below `tol` are treated as zero and dropped, which makes this a
    pseudo(-inverse) root on the range of C. That is the safe behaviour when a distribution is
    degenerate: the alternative - inverting a near-zero eigenvalue - turns numerical dust into a
    dominant direction, which is precisely how a covariance-matching transform goes wrong loudly
    after looking fine on well-conditioned data.
    """
    C = (C + C.T) / 2.0
    w, V = np.linalg.eigh(C)
    w = np.where(w > tol, w, 0.0)
    f = np.where(w > 0.0, np.abs(w) ** power, 0.0)
    return (V * f) @ V.T


def coral_transform(source: np.ndarray, target: np.ndarray,
                    match_cov: bool = True) -> dict:
    """Linear map T = {"A", "b"} with A = Cs^-1/2 Ct^1/2 and b = mt - ms A.

    Then for row-vector x: x @ A + b. Substituting gives A' Cs A = Ct, i.e. the source ellipse is
    carried exactly onto the (shrunk) target ellipse; `check_alignment.py` asserts that algebra
    rather than trusting an empirical near-match.

    match_cov=False stops at the mean shift, which is Model 5's MEAN arm and also the honest way
    to ask whether the second-order half of the transform is doing anything at all.
    """
    source = np.atleast_2d(np.asarray(source, float))
    target = np.atleast_2d(np.asarray(target, float))
    if source.ndim != 2 or target.ndim != 2:
        raise AssertionError("source and target must be 2-D feature matrices")
    if source.shape[1] != target.shape[1]:
        raise AssertionError(f"width mismatch: source {source.shape[1]} vs target {target.shape[1]}")
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise AssertionError("non-finite values in the feature matrices handed to the aligner")
    d = source.shape[1]
    ms, mt = source.mean(0), target.mean(0)
    ws = wt = 0.0
    if not match_cov:
        A, b = np.eye(d), mt - ms
    else:
        Cs, ws = covariance(source)
        Ct, wt = covariance(target)
        A = _sym_pow(Cs, -0.5) @ _sym_pow(Ct, 0.5)
        b = mt - ms @ A
    if not np.isfinite(A).all():
        raise AssertionError("non-finite entries in the CORAL matrix")
    if not np.isfinite(b).all():
        raise AssertionError("non-finite entries in the CORAL shift")
    return {"A": A, "b": b, "match_cov": bool(match_cov),
            "shrink_source": float(ws), "shrink_target": float(wt)}


def apply_transform(X: np.ndarray, T: dict) -> np.ndarray:
    """X @ A + b. The same map is applied to the fitting camera and the scoring camera, so a
    model expressed in target coordinates is never scored in another coordinate system."""
    X = np.atleast_2d(np.asarray(X, float))
    out = X @ T["A"] + T["b"]
    if not np.isfinite(out).all():
        raise AssertionError("transform produced non-finite features")
    return out


def independent_column_shuffle(target: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle every feature column independently: marginals exact, joint structure destroyed.

    This is Model 5's attribution placebo. Permuting the target's ROWS would be useless - a mean
    and a covariance are invariant to row order, so a row permutation yields a bit-identical
    transform, which is exactly the flaw that killed the first design of this arm. Shuffling each
    column on its own keeps every per-feature mean, variance and marginal distribution of the real
    target and leaves the cross-feature covariance near-diagonal, so the only thing the placebo is
    missing is the target's actual correlation structure. A gain the placebo also achieves is
    variance rescaling, not information about the camera.

    Returns the shuffled table and the stacked per-column permutation vectors, because a control
    nobody can reconstruct is not a control.
    """
    target = np.atleast_2d(np.asarray(target, float))
    if target.ndim != 2:
        raise AssertionError("target must be a 2-D feature matrix")
    rng = np.random.default_rng(int(seed))
    n = target.shape[0]
    vecs = np.array([rng.permutation(n) for _ in range(target.shape[1])])
    out = np.empty_like(target)
    for j in range(target.shape[1]):
        out[:, j] = target[vecs[j], j]
    return out, vecs


def transform_distance(T1: dict, T2: dict) -> float:
    """Max absolute entrywise difference between two transforms, over A and b together."""
    a = np.abs(np.asarray(T1["A"], float) - np.asarray(T2["A"], float)).max()
    b = np.abs(np.asarray(T1["b"], float) - np.asarray(T2["b"], float)).max()
    return float(max(a, b))
