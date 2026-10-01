"""run_e1.py - score the four registered arms of Experiment 1 and apply gates G1-G4.

ARMS, and why each exists
  FROZEN  The rank-3 PCA head under the strictly family-honest nested protocol. This is the
          ruler. Every gain is measured against it, and G1 requires reproducing its recorded
          value before anything else is believed.
  E1-A    Quadratic shape map on predicted D50. The primary.
  E1-B    LINEAR shape map, identical in every other respect. This is the arm that makes the
          experiment an experiment: if A beats B, the degree of the polynomial is the cause;
          if it does not, the gain was never there and the primary result is an artefact of
          having changed something.
  E1-C    The quadratic map with labels permuted. Same dimensionality, same code path, no
          real relationship. This project has been burned by best-of-N selection twice, so a
          gain that a placebo can match is not a gain.
  E1-D    Quadratic map on TRUE D50, no features. The upper bound for any D50-parameterised
          model. E1-A means nothing without it: it separates "the map is wrong" from "the
          D50 estimate is noisy".

GATES, pre-registered in research_and_plan.md section 21
  G1  the harness reproduces 43.453225201811563 to 1e-6. Failure stops the run and reports.
  G2  E1-A beats E1-B by >= 3.00 EMD.
  G3  the E1-A minus frozen difference has a family-cluster bootstrap 95% CI lower bound
      above 0.
  G4  E1-A beats E1-C, and the observed gain exceeds the best-of-N permutation null.

UNCERTAINTY
  The bootstrap resamples the 16 curve-distance FAMILIES, not the 24 soils. Six pairs of
  training soils are near-duplicates in label space and are not independent observations;
  resampling soils would treat H366 and H371 as two separate pieces of evidence and narrow
  every interval. This is the same correction that killed the rank-1 basis in section 9.8.

WHAT IS NOT HERE
  No test prediction, no submission, no leaderboard claim. An internal CV mean is a
  measurement on 24 labelled soils, not a Kaggle score.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "Model 5" / "Model 5 Experiment 1",
          ROOT / "Model 7" / "Approach A",
          HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import d50 as _d50            # noqa: E402
import head_d50 as _h         # noqa: E402
import ruler as R             # noqa: E402  the transcribed registered protocol
import transfer_eval as tv    # noqa: E402

FROZEN_BASELINE = R.INTERNAL_BASELINE       # 43.453225201811563
G2_BAR = 3.00
N_BOOT = 20000
SEED = 20260930


# ------------------------------------------------------------------ data
def load():
    X, Y, fams, ids = R.design()
    ld = _d50.logd50(Y)
    print("[..] %d soils, %d features, %d families" % (len(X), X.shape[1], len(set(fams))))
    return X, Y, fams, ids, ld


# ------------------------------------------------------------------ arms
def frozen_arm(X, Y, fams):
    """The ruler: the registered rank-3 head under the strictly nested protocol."""
    errs, alphas = R.lofo_errors(X, Y, fams, return_alphas=True)
    return errs, alphas


def d50_arm(X, Y, fams, deg, use_true=False):
    """E1-A/B/D. use_true reads D50 off the target curve, so no features are consulted."""
    F = np.asarray(fams)
    out = np.full(len(X), np.nan)
    chosen: Dict[str, float] = {}
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        if use_true:
            h = _h.HeadD50(deg=deg, use_true_d50=True).fit(X[tr], Y[tr])
            P = h.predict(X[te], Y_for_true_d50=Y[te])
            chosen[g] = float("nan")
        else:
            curve = _h.alpha_curve(X[tr], Y[tr], list(F[tr]), deg)
            al = min(curve, key=curve.get)
            chosen[g] = float(al)
            h = _h.HeadD50(deg=deg, alpha=al).fit(X[tr], Y[tr])
            P = h.predict(X[te])
        for k, i in enumerate(te):
            out[i] = _h._emd(P[k], Y[i])
    assert not np.isnan(out).any(), "a soil was never scored"
    return out, chosen


def shape_only_arm(abscissa: np.ndarray, targets: np.ndarray, fams, deg):
    """E1-D proper: true D50 as abscissa, leave-one-family-out on the shape map alone.

    No features and no alpha, so this is a closed-form polynomial fit per fold. Cheap enough
    to run inside the permutation null.

    The abscissa and the targets are SEPARATE arguments on purpose. The permutation null has
    to break the link between a curve and its own D50; passing a single matrix would permute
    rows together, carry each curve's D50 along with it, and leave the very relationship
    under test perfectly intact - a null that measures the real effect is not a null.
    """
    F = np.asarray(fams)
    out = np.full(len(targets), np.nan)
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        A = _h.design(abscissa[tr], deg)
        b = np.linalg.lstsq(A, targets[tr], rcond=None)[0]
        # project() clips/accumulates along axis 1 and returns 2-D, so the folded prediction
        # must stay 2-D here: ravel()ing it merges every held-out soil into one long row.
        P = tv.project(_h.design(abscissa[te], deg) @ b)
        for k, i in enumerate(te):
            out[i] = _h._emd(P[k], targets[i])
    return out


# ------------------------------------------------------------------ uncertainty
def family_cluster_ci(diff: np.ndarray, fams: List[str], n: int = N_BOOT,
                      seed: int = SEED) -> Tuple[float, float, float, float]:
    """Mean paired difference with a 95% CI from resampling FAMILIES, not soils.

    Returns (mean, lo, hi, p_gain_negative). The p-value is the fraction of draws whose mean
    difference is <= 0, which is the quantity the gates are stated against.
    """
    fams = np.asarray(fams)
    by: Dict[str, List[int]] = {}
    for i, f in enumerate(fams):
        by.setdefault(str(f), []).append(i)
    keys = sorted(by)
    idx = [np.array(by[k]) for k in keys]
    rng = np.random.default_rng(seed)
    draws = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(keys), size=len(keys))
        take = np.concatenate([idx[p] for p in pick])
        draws[b] = diff[take].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), float((draws <= 0).mean())


def best_of_null(abscissa: np.ndarray, Y: np.ndarray, fams, n: int,
                 seed: int = SEED) -> Dict[str, object]:
    """Best-of-N permutation null on the SHAPE map, which is the comparison the gates use.

    The D50 abscissa is HELD FIXED and only the target curves are permuted across soils. That
    is the correct placebo for this hypothesis: the claim is that the map from D50 to shape
    is curved, so the thing that must be destroyed is the correspondence between a curve and
    its shape. Permuting D50 alongside the curves would instead destroy the very
    correspondence that makes the shape map predictable at all, and both arms would collapse
    to the same floor - which is what the first version of this function did, producing a
    "null" mean of +5.72 against a real +6.09.

    Both degrees are re-run on every draw and the gap between them is recorded, so the real
    gap is judged against what a search over the same two candidates finds on no signal.
    """
    rng = np.random.default_rng(seed)
    gaps = np.empty(n)
    e1s = np.empty(n)
    e2s = np.empty(n)
    for b in range(n):
        perm = rng.permutation(len(Y))
        Yp = Y[perm]
        e1 = shape_only_arm(abscissa, Yp, fams, 1).mean()
        e2 = shape_only_arm(abscissa, Yp, fams, 2).mean()
        e1s[b], e2s[b] = e1, e2
        gaps[b] = e1 - e2
    return {"n": n, "deg1_mean": float(e1s.mean()), "deg2_mean": float(e2s.mean()),
            "gap_mean": float(gaps.mean()), "gap_sd": float(gaps.std(ddof=1)),
            "gap_p95": float(np.percentile(gaps, 95)),
            "gap_max": float(gaps.max()),
            "construction": "D50 abscissa fixed; target curves permuted across soils"}


def loso_sensitivity(diff: np.ndarray, ids: List[str], fams: List[str]) -> Dict[str, object]:
    """Leave one soil out, and report which soil costs the most if it is removed.

    Section 9.8's rank-1 basis looked like a 2.59 EMD gain and turned out to be one soil.
    This is the check that catches that failure mode before it reaches a verdict.
    """
    full = float(diff.mean())
    worst_i, worst_v = None, None
    for i in range(len(diff)):
        keep = np.array([j for j in range(len(diff)) if j != i])
        v = float(diff[keep].mean())
        if worst_v is None or v < worst_v:
            worst_i, worst_v = i, v
    retained = float(worst_v)
    return {"full_mean": full, "most_influential_soil": str(ids[worst_i]),
            "drop_it_mean": retained,
            "gain_retained_fraction": float(retained / full) if full != 0 else float("nan"),
            "worst_soil_delta": float(diff[worst_i])}


# ------------------------------------------------------------------ main
def main() -> int:
    t0 = time.time()
    np.random.seed(SEED)
    print("=" * 74)
    print("MODEL 8 / EXPERIMENT 1 - quadratic shape map")
    print("ruler: strictly family-honest nested LOFO, %d families" % 16)
    print("=" * 74)

    X, Y, fams, ids, ld = load()

    # ---- family integrity -------------------------------------------------
    rep = R.fold_report(fams)
    if rep["family_leaks"] != 0:
        raise AssertionError("fold integrity failed: %r" % rep)
    print("[ok] fold integrity: %d folds, %d soils scored, %d leaks"
          % (rep["n_folds"], rep["n_scored"], rep["family_leaks"]))
    print("     fold sizes: %s" % sorted(rep["fold_sizes"].values()))

    # ---- G1 ---------------------------------------------------------------
    print("\n--- G1: reproduce the frozen baseline ---")
    t = time.time()
    e_frozen, a_frozen = frozen_arm(X, Y, fams)
    frozen_mean = float(e_frozen.mean())
    print("[..] frozen nested-LOFO = %.6f  (%.1fs)" % (frozen_mean, time.time() - t))
    gap = abs(frozen_mean - FROZEN_BASELINE)
    print("[%s] G1 |observed - recorded| = %.3e  (tolerance 1e-6)"
          % ("ok" if gap <= 1e-6 else "FAIL", gap))
    if gap > 1e-6:
        print("\nGATE G1 FAILED. Nothing else in this run is reported.")
        (HERE / "reproduce_anchor.txt").write_text(
            "G1 FAILED\nobserved %.15f\nrecorded %.15f\ngap %.6e\n"
            % (frozen_mean, FROZEN_BASELINE, gap), encoding="ascii")
        return 1
    (HERE / "reproduce_anchor.txt").write_text(
        "G1 PASSED\nobserved %.15f\nrecorded %.15f\ngap %.6e\n"
        % (frozen_mean, FROZEN_BASELINE, gap), encoding="ascii")

    # ---- arms -------------------------------------------------------------
    print("\n--- arms ---")
    t = time.time()
    e_a, a_a = d50_arm(X, Y, fams, 2)
    print("[..] E1-A quadratic  = %.4f  (%.1fs)" % (e_a.mean(), time.time() - t))
    t = time.time()
    e_b, a_b = d50_arm(X, Y, fams, 1)
    print("[..] E1-B linear     = %.4f  (%.1fs)" % (e_b.mean(), time.time() - t))
    t = time.time()
    e_d = shape_only_arm(ld, Y, fams, 2)
    print("[..] E1-D true D50   = %.4f  (%.1fs)" % (e_d.mean(), time.time() - t))
    e_dlin = shape_only_arm(ld, Y, fams, 1)
    print("[..]     true D50 lin= %.4f" % e_dlin.mean())

    # E1-C placebo: the quadratic map on permuted labels, full pipeline.
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(len(Y))
    e_c, _ = d50_arm(X, Y[perm], fams, 2)
    print("[..] E1-C placebo    = %.4f" % e_c.mean())

    # ---- G2 ---------------------------------------------------------------
    print("\n--- gates ---")
    d_ab = e_b - e_a
    m_ab, lo_ab, hi_ab, p_ab = family_cluster_ci(d_ab, fams)
    g2 = m_ab >= G2_BAR
    print("G2  E1-A vs E1-B      : %+.4f EMD  CI [%+.3f, %+.3f]  P(gain<0)=%.4f  -> %s"
          % (m_ab, lo_ab, hi_ab, p_ab, "PASS" if g2 else "FAIL"))

    d_af = e_frozen - e_a
    m_af, lo_af, hi_af, p_af = family_cluster_ci(d_af, fams)
    g3 = lo_af > 0
    print("G3  E1-A vs frozen    : %+.4f EMD  CI [%+.3f, %+.3f]  P(gain<0)=%.4f  -> %s"
          % (m_af, lo_af, hi_af, p_af, "PASS" if g3 else "FAIL"))

    gain_ac = e_c.mean() - e_a.mean()
    g4_pre = gain_ac > 0
    print("G4  E1-A vs E1-C      : %+.4f EMD  (positive -> %s)"
          % (gain_ac, "PASS" if g4_pre else "FAIL"))

    # ---- placebo null on the shape map -----------------------------------
    n_null = 200
    t = time.time()
    null = best_of_null(ld, Y, fams, n_null)
    real_gap = float(e_dlin.mean() - e_d.mean())
    g4_null = real_gap >= null["gap_p95"]
    null["real_gap"] = real_gap
    null["g4_null_pass"] = bool(g4_null)
    print("    shape-map gap (true D50) : %+.4f EMD" % real_gap)
    print("    best-of-%d permuted null   : mean %+.4f, p95 %+.4f  -> G4 null %s"
          % (n_null, null["gap_mean"], null["gap_p95"], "PASS" if g4_null else "FAIL"))
    print("    (%.1fs)" % (time.time() - t))
    g4 = g4_pre and g4_null
    print("G4  combined           : -> %s" % ("PASS" if g4 else "FAIL"))

    # ---- supporting checks ------------------------------------------------
    loso = loso_sensitivity(d_ab, ids, fams)
    print("\n--- supporting (reported, not gating) ---")
    print("    E1-D reproduces the section 9.4 shape result: %.2f (plan said ~18.02)"
          % e_d.mean())
    print("    gain retained after dropping %s: %.1f%% (want >= 60%%)"
          % (loso["most_influential_soil"], 100 * loso["gain_retained_fraction"]))
    better = int((d_ab > 0).sum())
    print("    E1-A better than E1-B on %d of %d soils" % (better, len(d_ab)))

    # ---- artefacts --------------------------------------------------------
    pd_rows = []
    import pandas as pd
    for i, s in enumerate(ids):
        pd_rows.append({"sample_id": s, "cv_family": fams[i], "logD50": float(ld[i]),
                        "err_frozen": float(e_frozen[i]), "err_E1A_quad": float(e_a[i]),
                        "err_E1B_lin": float(e_b[i]), "err_E1C_placebo": float(e_c[i]),
                        "err_E1D_trueD50": float(e_d[i]),
                        "delta_A_minus_B": float(-d_ab[i]),
                        "alpha_frozen": float(a_frozen[fams[i]]),
                        "alpha_E1A": float(a_a[fams[i]]), "alpha_E1B": float(a_b[fams[i]])})
    pd.DataFrame(pd_rows).to_csv(HERE / "cv_per_soil.csv", index=False)

    pd.DataFrame([
        {"arm": "frozen", "degree": "rank-3 PCA", "mean_emd": frozen_mean,
         "vs_frozen": 0.0, "note": "ruler, strictly nested LOFO"},
        {"arm": "E1-A", "degree": 2, "mean_emd": float(e_a.mean()),
         "vs_frozen": float(-d_af.mean()), "note": "primary, predicted D50"},
        {"arm": "E1-B", "degree": 1, "mean_emd": float(e_b.mean()),
         "vs_frozen": float(e_frozen.mean() - e_b.mean()), "note": "control, linear shape"},
        {"arm": "E1-C", "degree": 2, "mean_emd": float(e_c.mean()),
         "vs_frozen": float(e_frozen.mean() - e_c.mean()), "note": "placebo, permuted labels"},
        {"arm": "E1-D", "degree": 2, "mean_emd": float(e_d.mean()),
         "vs_frozen": float(e_frozen.mean() - e_d.mean()), "note": "true D50, no features"},
        {"arm": "E1-D-linear", "degree": 1, "mean_emd": float(e_dlin.mean()),
         "vs_frozen": float(e_frozen.mean() - e_dlin.mean()), "note": "true D50, linear"},
    ]).to_csv(HERE / "arm_comparison.csv", index=False)

    boot = {"method": "family-cluster bootstrap, %d families resampled" % len(set(fams)),
            "n_draws": N_BOOT, "seed": SEED,
            "comparisons": {
                "E1A_vs_E1B": {"mean": m_ab, "ci95": [lo_ab, hi_ab], "p_gain_negative": p_ab},
                "E1A_vs_frozen": {"mean": m_af, "ci95": [lo_af, hi_af], "p_gain_negative": p_af}}}
    (HERE / "bootstrap.json").write_text(json.dumps(boot, indent=2), encoding="ascii")
    (HERE / "placebo.json").write_text(json.dumps(
        {"E1C_permuted_labels_mean": float(e_c.mean()),
         "shape_map_null": null, "real_shape_gap": real_gap}, indent=2), encoding="ascii")

    verdict = "SUPPORTED" if (g2 and g3 and g4) else "NOT-RESOLVED"
    print("\n" + "=" * 74)
    print("VERDICT: %s   (G1 pass, G2 %s, G3 %s, G4 %s)"
          % (verdict, "pass" if g2 else "fail", "pass" if g3 else "fail",
             "pass" if g4 else "fail"))
    print("elapsed %.1fs" % (time.time() - t0))
    print("=" * 74)
    (HERE / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "g1": True, "g2": bool(g2), "g3": bool(g3), "g4": bool(g4),
         "frozen_baseline": frozen_mean, "e1a": float(e_a.mean()), "e1b": float(e_b.mean()),
         "e1c": float(e_c.mean()), "e1d": float(e_d.mean()),
         "gain_A_vs_frozen": m_af, "ci95_A_vs_frozen": [lo_af, hi_af],
         "gain_A_vs_B": m_ab, "ci95_A_vs_B": [lo_ab, hi_ab]}, indent=2), encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
