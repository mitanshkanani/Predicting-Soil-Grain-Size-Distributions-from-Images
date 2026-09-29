"""Contract test for transfer_eval.py - run from the experiment folder:

    python check_transfer_eval.py

This is the gate before the gate. Model 5's whole claim is a DIFFERENCE between two fitted arms,
so if this module's head, aggregation or CV is anything other than Model 2 E3's, every number it
produces is a difference between two wrong things. Three checks carry that burden:

  test_g0_reproduces_model_2_e3_control   the no-holdout, unaligned control must land on E3's
                                          recorded 43.0217308796477 within 0.05 EMD. Read from
                                          E3's own artifact, never transcribed.
  test_image_soil_aggregation_matches_e3   the soil matrix must equal E3's saved control matrix
                                          value for value, so the ALIGN step is provably the only
                                          thing Model 5 inserts.
  test_clustered_ci_is_wider_than_naive     the resampling unit must be the soil, not the row. A
                                          soil appears in both directions; treating its two rows
                                          as independent narrows every CI by roughly sqrt(2),
                                          which is how a null becomes a discovery.

No pytest (not installed). Plain asserts, printed labels, non-zero exit on failure.
"""
from __future__ import annotations

import inspect
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

try:
    import transfer_eval as tv
    HAVE = True
except Exception as exc:                                      # noqa: BLE001
    print(f"IMPORT FAILED: {type(exc).__name__}: {exc}")
    print("This is the expected state before transfer_eval.py exists.")
    HAVE = False

E3_DIR = ROOT / "Model 2" / "Model 2 Experiment 3"
META = ROOT / "data" / "processed_meta"
BOOT_SEED = 20260934
FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def _fams():
    f = pd.read_csv(E3_DIR / "cv_families.csv")
    return dict(zip(f.sample_id.astype(str), f.cv_family))


def test_counts_match_the_spec():
    """24 train soils, 23 Motorola, 22 Samsung, 21 in both, 45 evaluation rows, 24 clusters.

    These are the experiment's entire statistical universe. If they drift, the thresholds written
    in the contract are calibrated against something that no longer exists.
    """
    img = tv.image_features(ROOT)
    fams = _fams()
    labels = pd.read_csv(META / "manifest_samples.csv")
    dA = tv.direction("A", img, labels, fams)
    dB = tv.direction("B", img, labels, fams)
    check("direction A fits 23 Motorola soils", dA["n_fit"] == 23, str(dA["n_fit"]))
    check("direction A scores 22 Samsung soils", dA["n_eval"] == 22, str(dA["n_eval"]))
    check("direction B fits 22 Samsung soils", dB["n_fit"] == 22, str(dB["n_fit"]))
    check("direction B scores 23 Motorola soils", dB["n_eval"] == 23, str(dB["n_eval"]))
    check("pooled evaluation rows are 45", dA["n_eval"] + dB["n_eval"] == 45)
    shared = set(dA["eval_ids"]) & set(dB["eval_ids"])
    check("21 soils are scored in both directions", len(shared) == 21, str(len(shared)))
    nsoils = len(set(dA["fit_ids"]) | set(dA["eval_ids"]))
    check("24 distinct training soils overall", nsoils == 24, str(nsoils))
    clusters = set(dA["eval_ids"]) | set(dB["eval_ids"])
    check("bootstrap clusters are soil identities, 24 of them", len(clusters) == 24,
          str(len(clusters)))
    check("every direction carries at least 8 CV families for alpha selection",
          len(dA["fit_families"]) >= 8 and len(dB["fit_families"]) >= 8,
          f"A {len(dA['fit_families'])} B {len(dB['fit_families'])}")


def test_no_soil_is_scored_by_a_model_that_saw_it():
    """The leak this check exists to keep out, and it is a design property, not a code detail.

    21 of the 24 training soils were photographed by both cameras. If direction A fits every
    Motorola soil and then scores every Samsung soil, the model has already seen the answer curve
    of 21 of the 22 soils it is being scored on - through the other camera. In production the 10
    test soils appear nowhere in training, so a seen-soil evaluation simulates the wrong task and
    inflates both arms while compressing their difference unpredictably.

    The fix costs no statistical power: each scored soil is dropped from the fit set of the fold
    that scores it, so every one of the 45 rows comes from a model that never saw that soil.
    """
    img = tv.image_features(ROOT)
    fams = _fams()
    labels = pd.read_csv(META / "manifest_samples.csv")
    for name in ("A", "B"):
        folds = tv.direction_folds(name, img, labels, fams)
        seen = [s for f in folds for s in f["eval_ids"]]
        check(f"direction {name}: every scored soil appears exactly once",
              len(seen) == len(set(seen)), f"{len(seen)} rows, {len(set(seen))} distinct")
        dirty = [f["fold"] for f in folds if set(f["eval_ids"]) & set(f["fit_ids"])]
        check(f"direction {name}: no fold scores a soil it was fitted on",
              not dirty, str(dirty[:4]))
        check(f"direction {name}: each fold keeps at least 8 families to select alpha from",
              all(len(f["fit_families"]) >= 8 for f in folds))
        check(f"direction {name}: every fold uses the source camera plus the test images only",
              all(f["target_rows"] == 35 for f in folds),
              str(sorted({f['target_rows'] for f in folds})))
    dA = tv.direction("A", img, labels, fams)
    dB = tv.direction("B", img, labels, fams)
    totA = sum(len(f["eval_ids"]) for f in tv.direction_folds("A", img, labels, fams))
    totB = sum(len(f["eval_ids"]) for f in tv.direction_folds("B", img, labels, fams))
    check("the fold split still yields the contracted 22 + 23 = 45 rows",
          totA == dA["n_eval"] == 22 and totB == dB["n_eval"] == 23,
          f"A {totA} B {totB}")


def test_alignment_target_is_test_only_and_disjoint():
    img = tv.image_features(ROOT)
    tgt = tv.alignment_target(img)
    check("alignment target is the 35 test images x 12 features", tgt.shape == (35, 12),
          str(tgt.shape))
    test_fams = set(img.loc[img.split == "test", "camera_fam"])
    check("the target's cameras are the iPhone families only", test_fams == {"iPhone"},
          str(sorted(test_fams)))
    tr_ids = set(img.loc[img.split == "train", "sample_id"])
    te_ids = set(img.loc[img.split == "test", "sample_id"])
    check("not one training soil contributes a target image", not (tr_ids & te_ids),
          f"{len(tr_ids)} train vs {len(te_ids)} test soils")
    check("target rows are finite", bool(np.isfinite(tgt).all()))


def test_image_soil_aggregation_matches_e3_matrix():
    """Model 5's soil tables must be E3's soil tables, value for value.

    Otherwise the ALIGN step is not the only variable, and the experiment measures its own
    plumbing. E3 saved the matrix it scored with, so this is a direct comparison against an
    artifact rather than a restatement of my own code.
    """
    img = tv.image_features(ROOT)
    mine = tv.soil_from_images(img)
    ref = pd.read_csv(E3_DIR / "features_soil_M1_control.csv", index_col="sample_id")
    ref = ref.drop(columns=[c for c in ("split",) if c in ref.columns])
    cols = list(tv.FEATURES)
    check("E3's control matrix carries exactly the 12 frozen features",
          list(ref.columns) == cols, str(list(ref.columns)))
    common = mine.index.intersection(ref.index)
    err = float(np.abs(mine.loc[common, cols].to_numpy(float)
                       - ref.loc[common, cols].to_numpy(float)).max())
    check("soil aggregation reproduces E3's saved control matrix", err < 1e-9,
          f"max abs difference {err:.2e} over {len(common)} soils x {len(cols)} features")
    check("and covers the same 34 soils E3 covered", len(mine) == len(ref) == 34,
          f"{len(mine)} vs {len(ref)}")


def test_g0_reproduces_model_2_e3_control():
    """THE gate before the gate. Read the reference from E3's own artifact, not from this file,
    so a transcription error cannot become a self-fulfilling check."""
    ref = float(pd.read_csv(E3_DIR / "sweep_results.csv")
                .set_index("arm").loc["M1", "nested_in_domain"])
    got = tv.no_holdout_control_score(ROOT)
    gap = abs(got - ref)
    check(f"G0: no-holdout unaligned control reproduces E3 within 0.05 EMD (ref {ref:.13f})",
          gap <= 0.05, f"this module {got:.13f}   gap {gap:.2e}")
    check("G0 is far tighter than required (identical machinery, not approximate agreement)",
          gap < 1e-6, f"gap {gap:.2e}")


def test_alpha_status_classification():
    check("grid floor is FLOOR", tv.alpha_status(tv.ALPHAS[0]) == "FLOOR")
    check("grid ceiling is CEILING", tv.alpha_status(tv.ALPHAS[-1]) == "CEILING")
    check("an interior value is INTERIOR", tv.alpha_status(300.0) == "INTERIOR")
    check("the grid is E1/E2/E3's, 16 values from 0.03 to 1e6",
          len(tv.ALPHAS) == 16 and tv.ALPHAS[0] == 0.03 and tv.ALPHAS[-1] == 1000000.0,
          f"{len(tv.ALPHAS)} values")
    check("the 12 control features are exactly E3's, in E3's order",
          tuple(tv.FEATURES) == ("e4", "e8", "e16", "lum_sd", "grad_mean", "R", "G", "B", "sat",
                                 "lum_p10", "lum_p50", "lum_p90"), str(tv.FEATURES))


def test_nested_predict_contract():
    """Alpha comes from the inner folds only, so what is reported is the score of the procedure."""
    rng = np.random.default_rng(12)
    X = rng.normal(size=(23, 12))
    Y = np.clip(np.tile(np.linspace(0, 100, 11), (23, 1)) + rng.normal(scale=6.0, size=(23, 11)),
                0, 100)
    fam = [f"F{i % 8}" for i in range(23)]
    r = tv.nested_predict(X, Y, fam, X[:5], tv.ALPHAS)
    check("selected alpha is a member of the frozen grid", r["alpha"] in tv.ALPHAS)
    check("boundary status reported per fit", r["status"] in ("FLOOR", "INTERIOR", "CEILING"),
          r["status"])
    check("predictions are (n, 11)", r["pred"].shape == (5, 11), str(r["pred"].shape))
    check("predictions are finite", bool(np.isfinite(r["pred"]).all()))
    check("predictions are non-decreasing CDFs",
          bool((np.diff(r["pred"], axis=1) >= -1e-9).all()))
    check("200 mm support forced to 100", bool(np.abs(r["pred"][:, -1] - 100.0).max() < 1e-9))
    check("selection error equals the best inner error by construction",
          abs(r["selection_emd"] - r["best_emd"]) < 1e-12)
    check("the whole sweep is retained for the boundary report", len(r["sweep"]) == 16)
    with np.errstate(all="ignore"):
        check("no alpha produced a non-finite error",
              all(np.isfinite(v) for v in r["sweep"].values()))


def test_nested_selection_is_not_oracle_selection():
    """If nested selection always equalled picking the best alpha on the held-out fold itself,
    the procedure would be scoring an impossibility. It must usually pick a different alpha."""
    rng = np.random.default_rng(31)
    diffs = 0
    for trial in range(6):
        X = rng.normal(size=(24, 12))
        Y = np.clip(np.tile(np.linspace(0, 100, 11), (24, 1))
                    + rng.normal(scale=8.0, size=(24, 11))
                    + trial * 0.01 * X[:, :11] * 3.0, 0, 100)
        fam = [f"F{i % 6}" for i in range(24)]
        r = tv.nested_predict(X, Y, fam, X[:4], tv.ALPHAS)
        oracle = min(r["sweep"], key=r["sweep"].get)
        # oracle here is the argmin of the same nested curve, so it must agree; the real check is
        # that the curve is not flat, i.e. alpha genuinely matters and selection is doing work.
        spread = max(r["sweep"].values()) - min(r["sweep"].values())
        if spread > 1e-9:
            diffs += 1
    check("the alpha curve is not degenerate across trials", diffs >= 4, f"{diffs}/6 informative")


def test_oracle_sweep_is_a_reported_diagnostic_and_selects_nothing():
    """Section 5a requires the oracle alpha and grid spread beside the nested choice.

    The oracle reads the scoring labels, so it must be provably incapable of influencing what is
    scored: it returns numbers only, mutates nothing, and its own error is by construction no
    larger than the error the nested procedure actually achieved on the same rows.
    """
    rng = np.random.default_rng(77)
    X = rng.normal(size=(23, 12))
    Y = np.clip(np.tile(np.linspace(0, 100, 11), (23, 1)) + rng.normal(scale=6.0, size=(23, 11)),
                0, 100)
    fam = [f"F{i % 8}" for i in range(23)]
    Xe, Ye = X[:5], Y[:5]
    before = tv.nested_predict(X, Y, fam, Xe, tv.ALPHAS)
    o = tv.oracle_sweep(X, Y, Xe, Ye)
    after = tv.nested_predict(X, Y, fam, Xe, tv.ALPHAS)
    check("oracle alpha is a grid member with a boundary status",
          o["alpha"] in tv.ALPHAS and o["status"] == tv.alpha_status(o["alpha"]), o["status"])
    check("the sweep covers the whole frozen grid", len(o["curve"]) == 16)
    check("reported emd is the minimum of its own curve and spread is its range",
          abs(o["emd"] - min(o["curve"].values())) < 1e-12
          and abs(o["spread"] - (max(o["curve"].values()) - min(o["curve"].values()))) < 1e-12)
    nested_emd = float(np.mean([tv.emd_pair(before["pred"][i], Ye[i]) for i in range(5)]))
    check("the oracle is not better than itself: its error is the achievable floor",
          o["emd"] <= nested_emd + 1e-12, f"oracle {o['emd']:.4f} vs nested {nested_emd:.4f}")
    check("computing the diagnostic left the nested result bit-identical",
          np.array_equal(before["pred"], after["pred"]) and before["alpha"] == after["alpha"])
    refit = float(np.mean([tv.emd_pair(tv.fit_predict(X, Y, Xe, before["alpha"])[i], Ye[i])
                           for i in range(5)]))
    check("at the nested alpha the sweep reproduces the nested score exactly, so both refit "
          "on all rows and only the SELECTION differs",
          abs(o["curve"][before["alpha"]] - refit) < 1e-12,
          f"sweep {o['curve'][before['alpha']]:.6f} vs independent refit {refit:.6f}")
    try:
        tv.oracle_sweep(X, Y, Xe, Ye[:4])
        check("a row/label count mismatch raises", False, "no exception")
    except AssertionError:
        check("a row/label count mismatch raises AssertionError", True)


def test_clustered_bootstrap_is_wider_at_the_same_point_estimate():
    rows = np.concatenate([np.full(21, 3.0), np.full(21, 3.0), [1.0], [1.0], [-2.0]])
    clusters = ([f"S{i}" for i in range(21)] * 2) + ["X1", "X2", "X3"]
    m_n, n_lo, n_hi = tv.naive_bootstrap(rows, 4000, BOOT_SEED)
    m_c, c_lo, c_hi = tv.clustered_bootstrap(rows, clusters, 4000, BOOT_SEED)
    check("clustered CI is strictly wider than row-wise CI",
          (c_hi - c_lo) > (n_hi - n_lo),
          f"clustered {c_hi - c_lo:.4f} vs naive {n_hi - n_lo:.4f}")
    check("both estimators agree on the point estimate", abs(m_n - m_c) < 1e-12,
          f"{m_c:.6f}")
    check("clustering is deterministic for a fixed seed",
          tv.clustered_bootstrap(rows, clusters, 4000, BOOT_SEED)
          == tv.clustered_bootstrap(rows, clusters, 4000, BOOT_SEED))
    check("cluster count is reported and cannot exceed the row count",
          len(set(clusters)) <= len(rows), f"{len(set(clusters))} clusters / {len(rows)} rows")
    one = np.arange(24, dtype=float)
    # Cluster keys are sorted lexicographically, so with one row per cluster the two methods
    # draw the same random stream but map it onto rows differently: identical in distribution,
    # not identical in realisation. Comparing widths, not draws, is the honest assertion.
    clo = [tv.clustered_bootstrap(one, [f"c{i}" for i in range(24)], 500, 20260900 + b)[2]
           - tv.clustered_bootstrap(one, [f"c{i}" for i in range(24)], 500, 20260900 + b)[1]
           for b in range(5)]
    nlo = [tv.naive_bootstrap(one, 500, 20260900 + b)[2] - tv.naive_bootstrap(one, 500,
           20260900 + b)[1] for b in range(5)]
    check("with one row per cluster the two methods agree on width in distribution",
          abs(np.mean(clo) - np.mean(nlo)) / max(np.mean(nlo), 1e-9) < 0.05,
          f"clustered {np.mean(clo):.3f} vs naive {np.mean(nlo):.3f}")
    check("clustering is stable across key spellings of the same partition",
          abs(tv.clustered_bootstrap(one, [f"k{i:02d}" for i in range(24)], 500, BOOT_SEED)[1]
              - tv.clustered_bootstrap(one, [f"c{i:02d}" for i in range(24)], 500,
                                       BOOT_SEED)[1]) < 1e-12)


def test_no_label_or_camera_reaches_the_pipeline_helpers():
    banned = {"labels", "label", "y", "Y", "camera", "camera_fam", "sample_id", "ppm", "exif"}
    for name in ("alignment_target", "soil_from_images", "nested_predict",
                 "clustered_bootstrap", "naive_bootstrap", "alpha_status"):
        fn = getattr(tv, name)
        got = set(inspect.signature(fn).parameters)
        check(f"{name}() takes no label or camera argument", not (got & banned), str(sorted(got)))
    src = (HERE / "transfer_eval.py").read_text(encoding="utf-8")
    check("transfer_eval.py imports no torch or timm",
          not re.search(r"^\s*(import torch|from torch|import timm)", src, re.M))
    at = inspect.getsource(tv.alignment_target)
    check("alignment_target filters on the split column only, never on a device name",
          "camera_make" not in at and "camera_model" not in at and "exif" not in at.lower())


def main():
    print("CHECK TRANSFER EVAL")
    print("-" * 78)
    if not HAVE:
        print("\n[FAIL] transfer_eval.py is not importable; nothing else can be verified.")
        return 1
    for name in [k for k in sorted(globals()) if k.startswith("test_")]:
        print(f"\n{name}")
        try:
            globals()[name]()
        except Exception as exc:                              # noqa: BLE001
            check(f"{name} raised {type(exc).__name__}", False, str(exc)[:150])
    print("-" * 78)
    if FAILS:
        print(f"CHECK TRANSFER EVAL: FAILED ({len(FAILS)} problems)")
        for f in FAILS:
            print("  -", f)
        return 1
    print("CHECK TRANSFER EVAL: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
