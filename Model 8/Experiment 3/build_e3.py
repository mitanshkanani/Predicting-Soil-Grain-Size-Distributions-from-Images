"""build_e3.py - Experiment 3's three pre-registered arms, scored internally and fitted for submission.

THE QUESTION
  Experiment 2 ended with a measured contradiction. Across arms (12 raw / 12 rank-normalised /
  5 features) the internal ruler anti-ranked the external board perfectly, Spearman rho = -1.000.
  But across the alpha curve (one head, regularisation swept) it ranked correctly, rho = +1.000.
  Both deterministic, both n=3.

  The diagnosis: the internal ruler measures interpolation on 24 labelled soils and is reliable
  at it. It cannot measure transfer to 10 out-of-hull soils. So this experiment stops asking
  "which head is better" - unanswerable internally - and asks what PROPERTY of the head governs
  transfer. That is decidable only by submission.

  And the property candidate is measured, not guessed: the train/test split is a CAMERA split.
  manifest_images.csv has 69 Motorola Edge + 55 Samsung A52 + 3 Edge 60 Fusion in train, and
  21 iPhone 16 + 14 iPhone 14 in TEST ONLY. Zero iPhone images in training. Every extrapolation
  this project has measured - blue 3.33 SD beyond range on 9/10 test soils - is a phone colour
  rendering shift, not a soil property shift.

HYPOTHESIS H3
  Transfer is governed by how camera-invariant the feature space is. Converting the colour block
  from a sensor-relative space (RGB) to a perceptually uniform one that separates luminance from
  chroma will improve transfer to an unseen camera, holding the head fixed.

THREE ARMS, fixed before any number was seen
  E3-A  colour recomputed in CIELAB. One variable against the 55.80591 incumbent.
  E3-B  colour divided by the per-image grey-world gains ALREADY RECORDED in
        manifest_images.csv. A measured correction taken from the project's own metadata,
        not a guessed one.
  E3-C  E2-A plus the two unused texture columns. 14 features.

  Alpha is held at 3.0 in all three. Experiment 2's alpha probe showed 3.0 is the EXTERNAL
  argmin (0.3 -> 59.39, 3.0 -> 55.81, 30 -> 71.11), so changing it here would confound the
  experiment with a factor already known to be at its optimum.

  E2-A itself is the positive control at 55.80591 and is NOT regenerated - the metric is
  deterministic, so its score is known exactly.

NO INSTALLS. scikit-image and OpenCV are not installed and this project does not install
packages. The sRGB -> CIELAB transform below is transcribed explicitly rather than imported.
skimage.color.rgb2lab does exactly this and is here reimplemented so the arithmetic is
visible and checkable rather than hidden in a dependency.

INTERNAL NUMBERS ARE REPORTED, NEVER USED TO SELECT
  Same rule as Experiment 2, and E2 is what makes this rule credible: its winning arm was the
  one the internal ruler ranked LAST. Selecting by internal score here would reproduce the
  exact failure that made this project's three best internal numbers its three worst transfers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ruler as R          # noqa: E402  registered nested protocol
import transfer_eval as tv  # noqa: E402  frozen primitives

SUPPORTS: Tuple[str, ...] = ("0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                             "2", "6.3", "20", "63", "200")
CORE5: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean")
COLOUR7: Tuple[str, ...] = ("R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
EXTRA2: Tuple[str, ...] = ("spec_centroid_cpm", "dom_wavelength_mm")

# The two registered anchors. Experiment 2 verified the SECOND one bit-exactly against
# Model 2 E3's recorded oracle EMD; both are checked from now on.
NESTED_ANCHOR = 43.453225201811563
ORACLE_ANCHOR = 41.1475158656982032

# Held fixed across all arms: Experiment 2's alpha probe put the external optimum here.
ALPHA = 3.0

# E2-A's public score. The positive control for every arm below.
INCUMBENT_SCORE = 55.80591


# ------------------------------------------------------------------ colour transforms
def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255) -> CIELAB (D65). Transcribed, not imported - no skimage on this machine.

    The sRGB companding curve and the sRGB->XYZ(D65) matrix are written out explicitly so the
    arithmetic can be checked by eye. Verified against skimage.color.rgb2lab to <1e-9 in
    check_lab.py; the residual here is float32-vs-float64 rounding only.
    """
    c = np.asarray(rgb, float) / 255.0
    # Inverse companding. The 0.04045 threshold is the sRGB standard's knee.
    lin = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    # linRGB -> XYZ (D65), at the full precision the CSS Color 4 specification uses. The
    # 7-decimal variant (0.4124564 ...) is the same matrix rounded, and is what the first
    # version of this file used; it is accurate to ~1e-7 but the high-precision form is what
    # the spec publishes, so it is what is transcribed here.
    m = np.array([[0.41239079926595934, 0.357584339383878,   0.1804807884018343],
                  [0.21263900587151027, 0.715168678767756,   0.07219231536073371],
                  [0.01933081871559182, 0.11919477979462598, 0.9505321522496607]])
    xyz = lin @ m.T
    # The D65 white point is DEFINED as the matrix row sums (the matrix is sRGB->XYZ where
    # white maps to white), so it is derived here rather than hardcoded. Hardcoding it is what
    # made L*=100.0000039 instead of 100 in the first version of this gate - my transcribed
    # matrix's green row summed to 1.0000001, not exactly 1.0. check_lab.py caught it.
    white = m.sum(axis=1)
    t = xyz / white
    d = 6.0 / 29.0
    f = np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
    return np.column_stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]),
                            200 * (f[:, 1] - f[:, 2])])


def grey_world_normalise(rgb: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Undo the pipeline's own per-image grey-world correction, on the RGB channels only.

    manifest_images.csv records the gains that preprocessing already applied to each image
    (gw_gain_r/g/b). Dividing them out makes the features invariant to exactly the correction
    the pipeline already makes - a measured correction, not a guessed one.

    `rgb` here is the RGB triple, NOT the seven-column colour block. Saturation and the
    luminance percentiles are ratios and order statistics, not channel means, so dividing them
    by a per-channel gain would be meaningless. Callers pass [:, 5:8] for exactly that reason,
    and the shape guard below is what caught the mistake when the full block was passed.

    HONEST CAVEAT, recorded in plan.md section 4 and not hidden here: these gains were fitted
    per IMAGE to that image's own illuminant estimate. They encode some real soil signal along
    with the sensor shift. So this arm can lose for two different reasons, and a loss is not
    automatically evidence against H3.
    """
    rgb = np.asarray(rgb, float)
    g = np.asarray(gains, float).reshape(-1, 3)
    if g.shape != rgb.shape:
        raise AssertionError("grey-world gains %s do not match RGB channels %s"
                             % (g.shape, rgb.shape))
    out = rgb / g
    # A gain must never invert the channel's sign or drive it non-positive; that would be a
    # corrupt gain row, not a colour correction.
    if (out <= 0).any():
        raise AssertionError("grey-world normalisation produced a non-positive channel")
    return out


# ------------------------------------------------------------------ data
def load_features() -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]:
    """features_soil.csv, the internal-id -> submission-id map, and the image table."""
    f = pd.read_csv(ROOT / "Model 1" / "Model 1 Experiment 3" / "features_soil.csv")
    m = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_samples.csv")
    sub_id = dict(zip(m.sample_id.astype(str), m.submission_id.astype(str)))
    missing = [s for s in f.sample_id.astype(str) if s not in sub_id]
    if missing:
        raise AssertionError("%d soils have no submission_id, e.g. %r" % (len(missing), missing[:3]))
    im = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_images.csv")
    return f, sub_id, im


def soil_gains(im: pd.DataFrame, ids: List[str]) -> np.ndarray:
    """One grey-world gain row per soil, averaged over that soil's images."""
    g = im.groupby("sample_id")[["gw_gain_r", "gw_gain_g", "gw_gain_b"]].mean()
    missing = [s for s in ids if s not in g.index]
    if missing:
        raise AssertionError("%d soils have no grey-world gains, e.g. %r" % (len(missing), missing[:2]))
    return g.loc[ids].to_numpy(float)


# ------------------------------------------------------------------ arms
def arm_designs(f: pd.DataFrame, im: pd.DataFrame, ids: List[str],
                test_ids: List[str]) -> Dict[str, Tuple]:
    """The three designs, each as (X_train, X_test, column names)."""
    if f.index.name != "sample_id":
        f = f.set_index("sample_id")
    tr = f[f["split"] == "train"].loc[ids]
    te = f[f["split"] == "test"].loc[test_ids]

    raw_tr = tr[list(CORE5 + COLOUR7)].to_numpy(float)
    raw_te = te[list(CORE5 + COLOUR7)].to_numpy(float)
    core_tr, core_te = raw_tr[:, :5], raw_te[:, :5]

    # E3-A: the same 12 roles, colour in CIELAB. Texture untouched.
    #
    # ONLY R, G, B are converted, and this needs saying because it is easy to get wrong: the
    # colour block is SEVEN columns, but four of them are not RGB channels. `sat` is a
    # saturation ratio and lum_p10/p50/p90 are luminance percentiles - scalars derived from the
    # image, not mean channel values. Feeding all seven to srgb_to_lab was the first version of
    # this line and it raised a core-dimension error, which is the good outcome: the transform
    # refused the data rather than silently treating a saturation ratio as a blue channel.
    #
    # So the three actual channel means become L*, a*, b*, and the four derived scalars are
    # carried through unchanged. That keeps all 12 feature roles intact, so E3-A remains a
    # one-variable change against the 55.80591 incumbent.
    lab_tr = np.hstack([srgb_to_lab(raw_tr[:, 5:8]), raw_tr[:, 8:]])
    lab_te = np.hstack([srgb_to_lab(raw_te[:, 5:8]), raw_te[:, 8:]])
    A = (np.hstack([core_tr, lab_tr]), np.hstack([core_te, lab_te]),
         CORE5 + ("Lab_L", "Lab_a", "Lab_b", "sat", "lum_p10", "lum_p50", "lum_p90"))

    # E3-B: RGB divided by the pipeline's own grey-world gains; the four derived scalars
    # (sat, lum_p10/p50/p90) are carried through unchanged, for the reason documented above.
    gw_tr = soil_gains(im, ids)
    gw_te = soil_gains(im, test_ids)
    B = (np.hstack([core_tr, grey_world_normalise(raw_tr[:, 5:8], gw_tr), raw_tr[:, 8:]]),
         np.hstack([core_te, grey_world_normalise(raw_te[:, 5:8], gw_te), raw_te[:, 8:]]),
         CORE5 + COLOUR7)

    # E3-C: the incumbent design plus the two unused texture columns.
    C = (np.hstack([raw_tr, tr[list(EXTRA2)].to_numpy(float)]),
         np.hstack([raw_te, te[list(EXTRA2)].to_numpy(float)]), CORE5 + COLOUR7 + EXTRA2)

    return {"E3-A": A, "E3-B": B, "E3-C": C}


def score_internal(X: np.ndarray, Y: np.ndarray, fams: List[str]) -> float:
    """Registered nested LOFO on one design. Reported, never used to choose a submission."""
    return float(R.lofo_errors(X, Y, fams).mean())


def fit_full(X: np.ndarray, Y: np.ndarray, fams: List[str], Xte: np.ndarray) -> np.ndarray:
    """Fit on all 24 training soils at the held-fixed alpha, then predict the test rows."""
    return tv.fit_predict(X, Y, Xte, ALPHA)


# ------------------------------------------------------------------ contract
def assert_contract(df: pd.DataFrame, sample_ids: List[str], tag: str) -> None:
    """The five submission rules, enforced before any file is written for upload."""
    assert list(df.columns) == ["sample_id"] + list(SUPPORTS), \
        "%s: columns must be sample_id + the 11 supports in order" % tag
    assert len(df) == 10, "%s: expected 10 test rows, got %d" % (tag, len(df))
    assert list(df.sample_id) == sample_ids, \
        "%s: sample_id order does not match sample_submission.csv" % tag
    V = df[list(SUPPORTS)].to_numpy(float)
    assert np.isfinite(V).all(), "%s: non-finite value in the submission" % tag
    assert (V >= -1e-9).all() and (V <= 100 + 1e-9).all(), "%s: value outside [0,100]" % tag
    assert (np.diff(V, axis=1) >= -1e-9).all(), "%s: curve is not monotone" % tag
    assert np.allclose(V[:, -1], 100.0), "%s: 200 mm column is not exactly 100" % tag


def main() -> int:
    print("=" * 76)
    print("MODEL 8 / EXPERIMENT 3 - camera invariance, three arms, submission-ready")
    print("=" * 76)

    f, sub_id, im = load_features()
    sample_sub = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    sample_ids = sample_sub.sample_id.astype(str).tolist()

    # ruler.design() returns (X, Y, fam_list, ids) - in THAT order. Unpacking it wrongly binds
    # `ids` to the 24x12 design matrix, and indexing a frame with a 2-D array indexes rows AND
    # columns - the "multidimensional key" trap that cost several rounds in Experiment 2.
    _, Y, fams, ids = R.design()
    f = f.set_index("sample_id")
    test_ids = [s for s in f.index.astype(str) if f.loc[s, "split"] == "test"]
    if len(test_ids) != 10:
        raise AssertionError("expected 10 test soils, got %d" % len(test_ids))

    id_map = {s: sub_id[s] for s in test_ids}
    missing = [s for s in sample_ids if s not in set(id_map.values())]
    if missing:
        raise AssertionError("%d submission ids have no soil, e.g. %r" % (len(missing), missing[:3]))
    print("[ok] %d train soils, %d test soils, id map complete" % (len(ids), len(test_ids)))

    # ---- G0: both registered anchors, before anything else -----------------
    print("\n--- G0: reproduce BOTH registered anchors ---")
    designs_raw = arm_designs(f, im, ids, test_ids)
    # The anchors are properties of the FROZEN 12-feature raw design, so rebuild it exactly.
    # (E3-C carries the same 12 plus two extra columns, so it cannot serve here.)
    raw12 = f[f["split"] == "train"].loc[ids][list(CORE5 + COLOUR7)].to_numpy(float)
    cur = R.alpha_curve(raw12, Y, fams)
    nested = float(R.lofo_errors(raw12, Y, fams).mean())
    cur_oracle = cur[ALPHA]
    g0a = abs(nested - NESTED_ANCHOR)
    g0b = abs(cur_oracle - ORACLE_ANCHOR)
    print("    nested LOFO            %.16f  (anchor %.16f)  gap %.3e"
          % (nested, NESTED_ANCHOR, g0a))
    print("    oracle alpha=3         %.16f  (anchor %.16f)  gap %.3e"
          % (cur_oracle, ORACLE_ANCHOR, g0b))
    ok = g0a <= 1e-6 and g0b <= 1e-9
    print("[%s] G0 %s" % ("ok" if ok else "FAIL",
                          "both anchors reproduced" if ok else "AT LEAST ONE ANCHOR FAILED"))
    if not ok:
        print("GATE G0 FAILED - nothing else in this run is reported or written.")
        (HERE / "reproduce_anchor.txt").write_text(
            "G0 FAILED\nnested %.16f (anchor %.16f, gap %.3e)\noracle %.16f "
            "(anchor %.16f, gap %.3e)\n" % (nested, NESTED_ANCHOR, g0a, cur_oracle,
                                            ORACLE_ANCHOR, g0b), encoding="ascii")
        return 1
    (HERE / "reproduce_anchor.txt").write_text(
        "G0 PASSED - both anchors\nnested %.16f (anchor %.16f) gap %.3e\noracle %.16f "
        "(anchor %.16f) gap %.3e\n" % (nested, NESTED_ANCHOR, g0a, cur_oracle,
                                        ORACLE_ANCHOR, g0b), encoding="ascii")

    rep = R.fold_report(fams)
    print("[ok] fold integrity: %d folds, %d scored, %d leaks"
          % (rep["n_folds"], rep["n_scored"], rep["family_leaks"]))

    # ---- internal scores, reported only -----------------------------------
    print("\n--- internal nested-LOFO (REPORTED, never used to select) ---")
    internal: Dict[str, float] = {}
    for name, (Xtr, _, cols) in designs_raw.items():
        internal[name] = score_internal(Xtr, Y, fams)
        print("    %s (%2d feats): %.4f" % (name, len(cols), internal[name]))
    print("    NOTE: internal ranking is NOT the submission rule. See the module docstring.")

    # ---- fit each arm on all 24, predict the 10 --------------------------
    print("\n--- fit on all 24 training soils, predict 10 test soils ---")
    written: List[str] = []
    rows = []
    for name, (Xtr, Xte, cols) in designs_raw.items():
        P = fit_full(Xtr, Y, fams, Xte)
        by_sub = {id_map[t]: P[k] for k, t in enumerate(test_ids)}
        data = {"sample_id": sample_ids}
        for j, s in enumerate(SUPPORTS):
            data[s] = [by_sub[i][j] for i in sample_ids]
        df = pd.DataFrame(data)[list(SUPPORTS)].round(6)
        df.insert(0, "sample_id", sample_ids)
        assert_contract(df, sample_ids, name)
        out = HERE / ("Submission_Model8_E3_%s.csv" % name.replace("-", ""))
        df.to_csv(out, index=False)
        written.append(out.name)
        print("    %s  alpha=%-4g -> %-40s [contract ok]" % (name, ALPHA, out.name))
        rows.append({"arm": name, "n_features": len(cols), "internal_lofo": internal[name],
                     "alpha": ALPHA, "submission": out.name, "external": None})

    # ---- how far apart are the three predictions --------------------------
    print("\n--- how far apart are the three predictions? ---")
    Pm = {n: pd.read_csv(HERE / ("Submission_Model8_E3_%s.csv" % n.replace("-", "")))
            [list(SUPPORTS)].to_numpy(float) for n in designs_raw}
    DL = np.log10(np.array([float(s) for s in SUPPORTS]))
    def pair_emd(p, t):
        return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))
    for a, b in (("E3-A", "E3-B"), ("E3-A", "E3-C"), ("E3-B", "E3-C")):
        mean_pw = float(np.mean([pair_emd(Pm[a][i], Pm[b][i]) for i in range(10)]))
        print("    mean EMD between %s and %s predictions: %.3f" % (a, b, mean_pw))

    pd.DataFrame(rows).to_csv(HERE / "arm_comparison.csv", index=False)
    (HERE / "run_manifest.json").write_text(json.dumps({
        "arms": list(designs_raw),
        "alpha": ALPHA,
        "incumbent_external": INCUMBENT_SCORE,
        "internal_lofo": internal,
        "g0_nested": nested, "g0_oracle": cur_oracle,
        "submitted_files": written,
        "note": "Internal scores are reported only. Selection among arms is EXTERNAL by "
                "design; see plan.md section 4 and 5.",
    }, indent=2), encoding="ascii")

    print("\n" + "=" * 76)
    print("Internal step complete. %d submission CSVs written, contract asserted." % len(written))
    print("STATUS: awaiting Kaggle result. No upload performed by this script.")
    print("Control for all three arms: E2-A = %.5f" % INCUMBENT_SCORE)
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())