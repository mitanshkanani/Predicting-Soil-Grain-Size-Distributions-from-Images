"""Contract test for alignment.py - run from the experiment folder:

    python check_alignment.py

Model 5's entire claim rests on one linear map, so the map is pinned here before any
experiment runs. Two of these checks exist because of mistakes that were actually made during
this project's design phase, and both are load-bearing:

  test_self_alignment_is_identity   CORAL toward a table's own distribution must be the identity.
                                    If it is not, the transform is doing something the math does
                                    not license, and every number downstream is meaningless.

  test_row_permutation_is_inert     The first placebo arm approved for Model 5 permuted the
                                    target's ROWS. CORAL reads the target only through its mean
                                    and covariance, both of which are invariant to row order, so
                                    that arm would have computed a BIT-IDENTICAL transform to the
                                    real one - the attribution clause G4 could never have passed
                                    and the verdict would have come out REGULARISATION-ARTEFACT
                                    regardless of the data. This test pins the invariance so the
                                    dead design cannot be reintroduced, and the shuffle test below
                                    pins that the replacement (independent per-column shuffling)
                                    actually differs.

No pytest (not installed). Plain asserts, printed labels, non-zero exit on failure.
"""
from __future__ import annotations

import inspect
import pathlib
import re
import sys

import numpy as np

try:
    import alignment as al
    HAVE_MODULE = True
except Exception as exc:                                      # noqa: BLE001
    print(f"IMPORT FAILED: {type(exc).__name__}: {exc}")
    print("This is the expected state before alignment.py exists.")
    HAVE_MODULE = False

FAILS = []
SHUF_SEED = 20260936


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def _skewed(n=23, d=12, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X[:, 0] *= 4.0
    X[:, 1] += 0.9 * X[:, 2]
    return X


def test_coral_satisfies_its_own_definition():
    """The exact algebraic property: A^T Cs A = Ct, for the shrunk covariances used to build A.

    Stated as algebra rather than as 'the empirical covariance looks similar', because shrinkage
    is not invariant under a general linear map, so an empirical comparison can only ever be
    approximate and would need a tolerance chosen after seeing the data.
    """
    S, T = _skewed(23, 12, 1), _skewed(35, 12, 5) + np.array([2.0] + [0.0] * 11)
    tr = al.coral_transform(S, T)
    Cs, ws = al.covariance(S)
    Ct, wt = al.covariance(T)
    A = tr["A"]
    err = float(np.abs(A.T @ Cs @ A - Ct).max())
    check("A maps the shrunk source covariance onto the shrunk target covariance",
          err < 1e-8, f"max |A'CsA - Ct| = {err:.2e}")
    check("the shift maps the source mean onto the target mean",
          np.abs(S.mean(0) @ A + tr["b"] - T.mean(0)).max() < 1e-10)
    check("A is square and matches the feature width", A.shape == (12, 12), str(A.shape))
    check("shrinkage weights are reported and inside [0,1]",
          0.0 <= tr["shrink_source"] <= 1.0 and 0.0 <= tr["shrink_target"] <= 1.0,
          f"src {tr['shrink_source']:.4f} tgt {tr['shrink_target']:.4f}")
    Sm = al.apply_transform(S, tr)
    before = float(np.abs(np.cov(S.T) - Ct).max())
    after = float(np.abs(np.cov(Sm.T) - Ct).max())
    check("transforming strictly reduces the empirical second-order distance to the target",
          after < before, f"{before:.3f} -> {after:.3f} ({100 * (1 - after / before):.1f}% closer)")
    check("the residual is not zero, and must not be claimed to be: Ledoit-Wolf shrinkage is not"
          " invariant under a general linear map, so cov(S A) uses the UNSHRUNK source covariance",
          after > 0.0, f"residual {after:.3f}")


def test_self_alignment_is_identity():
    X = _skewed(22, 12, 7)
    tr = al.coral_transform(X, X)
    check("self-alignment matrix is the identity",
          float(np.abs(tr["A"] - np.eye(12)).max()) < 1e-8,
          f"max |A - I| = {float(np.abs(tr['A'] - np.eye(12)).max()):.2e}")
    check("self-alignment shift is zero", float(np.abs(tr["b"]).max()) < 1e-8)
    identity = {"A": np.eye(12), "b": np.zeros(12)}
    check("distance to identity is within the notebook's 1e-6 canary bound",
          al.transform_distance(tr, identity) <= 1e-6)


def test_mean_only_arm_leaves_covariance_alone():
    X = _skewed(30, 12, 9)
    tgt = _skewed(10, 12, 11)
    tr = al.coral_transform(X, tgt, match_cov=False)
    Y = al.apply_transform(X, tr)
    check("match_cov=False keeps the covariance exactly",
          np.allclose(np.cov(Y.T), np.cov(X.T), atol=1e-10))
    check("match_cov=False still removes the mean offset",
          np.allclose(Y.mean(0), tgt.mean(0), atol=1e-12))
    check("mean-only transform is not the identity when means differ",
          al.transform_distance(tr, {"A": np.eye(12), "b": np.zeros(12)}) > 1e-6)


def test_row_permutation_is_inert():
    """Pinned reason the first placebo design was abandoned: row order cannot change a mean or a
    covariance, so a row-permuted target yields a bit-identical transform."""
    src = _skewed(20, 12, 13)
    tgt = _skewed(35, 12, 15)
    perm = tgt[np.random.default_rng(16).permutation(len(tgt))]
    d = al.transform_distance(al.coral_transform(src, tgt), al.coral_transform(src, perm))
    check("row-permuting the target changes NOTHING (why CORAL_PERM was killed)",
          d < 1e-12, f"transform distance {d:.2e}")


def test_independent_column_shuffle_preserves_marginals_and_kills_joint():
    tgt = _skewed(35, 12, 17)
    src = _skewed(23, 12, 19)
    sh, vecs = al.independent_column_shuffle(tgt, SHUF_SEED)
    check("per-column means preserved exactly",
          np.allclose(sh.mean(0), tgt.mean(0), atol=1e-12))
    check("per-column variances preserved exactly",
          np.allclose(sh.var(0), tgt.var(0), atol=1e-12))
    off_t = float(np.abs(np.corrcoef(tgt.T) - np.eye(12)).max())
    off_s = float(np.abs(np.corrcoef(sh.T) - np.eye(12)).max())
    check("cross-feature correlation is destroyed", off_s < off_t,
          f"max off-diagonal {off_t:.3f} -> {off_s:.3f}")
    d = al.transform_distance(al.coral_transform(src, tgt), al.coral_transform(src, sh))
    check("the placebo transform really differs from the real one", d > 1e-6,
          f"transform distance {d:.3e}")
    check("permutation vectors are one per column", vecs.shape == (12, len(tgt)), str(vecs.shape))


def test_shuffle_is_deterministic_and_a_bijection_per_column():
    tgt = _skewed(35, 12, 21)
    A, v1 = al.independent_column_shuffle(tgt, SHUF_SEED)
    B, v2 = al.independent_column_shuffle(tgt, SHUF_SEED)
    check("same seed reproduces the same shuffled table", np.array_equal(A, B))
    check("same seed reproduces the same permutation vectors", np.array_equal(v1, v2))
    check("a different seed gives a different table",
          not np.array_equal(A, al.independent_column_shuffle(tgt, SHUF_SEED + 1)[0]))
    ok = all(sorted(A[:, j].tolist()) == sorted(tgt[:, j].tolist()) for j in range(12))
    check("every column is a permutation of the original column (no value invented)", ok)


def test_rank_deficient_input_does_not_produce_nan():
    flat = np.tile(np.random.default_rng(23).normal(size=(1, 12)), (35, 1))   # all rows equal
    tr = al.coral_transform(_skewed(20, 12, 24), flat)
    check("collapsed target still yields finite A and b",
          np.isfinite(tr["A"]).all() and np.isfinite(tr["b"]).all())
    out = al.apply_transform(_skewed(20, 12, 24), tr)
    check("and the transform of real data stays finite", np.isfinite(out).all())
    dup = np.tile(_skewed(6, 12, 25), (4, 1))                                  # rank 6 of 12
    tr2 = al.coral_transform(dup, _skewed(35, 12, 26))
    check("rank-deficient SOURCE also yields a finite map", np.isfinite(tr2["A"]).all())


def test_label_blind_and_torch_free():
    src = pathlib.Path(__file__).with_name("alignment.py").read_text(encoding="utf-8")
    check("no torch or timm import in alignment.py",
          not re.search(r"^\s*(import torch|from torch|import timm|from timm)", src, re.M))
    banned = {"y", "labels", "label", "target_y", "Y", "camera", "camera_fam", "sample_id",
              "ppm", "exif"}
    for fn in (al.coral_transform, al.apply_transform, al.independent_column_shuffle,
               al.transform_distance, al.covariance):
        got = set(inspect.signature(fn).parameters)
        check(f"{fn.__name__} takes no label/camera argument", not (got & banned), str(sorted(got)))
    check("the word camera appears only in prose, never as an identifier",
          not re.search(r"\bcamera\b\s*[=:]", src))


def main():
    print("CHECK ALIGNMENT")
    print("-" * 78)
    if not HAVE_MODULE:
        print("\n[FAIL] alignment.py is not importable; nothing else can be verified.")
        return 1
    tests = [t for t in sorted(globals()) if t.startswith("test_")]
    for name in tests:
        print(f"\n{name}")
        try:
            globals()[name]()
        except Exception as exc:                              # noqa: BLE001
            check(f"{name} raised {type(exc).__name__}", False, str(exc)[:140])
    print("-" * 78)
    if FAILS:
        print(f"CHECK ALIGNMENT: FAILED ({len(FAILS)} problems)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("CHECK ALIGNMENT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
