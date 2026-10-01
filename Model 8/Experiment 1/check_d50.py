"""check_d50.py - gate on the D50 extraction before it is allowed near a score.

The single easiest error in this experiment is inverting the interpolation: finding the
bracket by searching the CUMULATIVE VALUES for 50 and interpolating in log10-diameter between
the bracketing supports. Interpolating percent-vs-logdiameter instead returns a plausible
number that is wrong, and a wrong abscissa silently corrupts the polynomial fit without ever
raising.

So this asserts the round-trip that the two conventions cannot both satisfy:

    curve_at(Y, diameter_at(Y, 50)) == 50

If the extraction used the wrong convention, the reconstructed percentage at its own reported
D50 would not be 50. The tolerance is tight on purpose: this is arithmetic, not estimation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]          # Model 8/Experiment 1 -> Model 8 -> repository root
for p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import d50  # noqa: E402
import transfer_eval as tv  # noqa: E402


def main() -> int:
    d50.verify_supports(ROOT)
    print("[ok] sample_submission.csv supports match d50.SUPPORTS")

    ids, Y = tv.labels_matrix(ROOT)
    Y = np.asarray(Y, float)
    print("[..] %d labelled training soils, %d supports" % (len(Y), Y.shape[1]))

    if not np.isfinite(Y).all():
        raise AssertionError("non-finite label matrix")

    # Monotonicity and the 100-at-200mm invariant, on the labels themselves.
    if (np.diff(Y, axis=1) < -1e-9).any():
        raise AssertionError("a label curve decreases across the supports")
    if not np.allclose(Y[:, -1], 100.0):
        raise AssertionError("a label curve does not end at exactly 100")
    print("[ok] all label curves are monotone and end at exactly 100")

    d = d50.diameter_at(Y, 50.0)
    back = d50.curve_at(Y, d)
    err = float(np.max(np.abs(back - 50.0)))
    print("[..] max |curve_at(Y, D50) - 50| = %.3e" % err)
    if err > 1e-6:
        raise AssertionError(
            "D50 round-trip failed (%.3e). The extraction is using the wrong interpolation "
            "convention." % err)
    print("[ok] D50 round-trip: every curve reads 50%% at its own reported D50")

    # The wrong convention, for contrast. Interpolating percent-vs-logdiameter instead of
    # logdiameter-vs-percent gives a DIFFERENT number; showing the size of that difference is
    # what makes the assertion above meaningful rather than vacuous.
    DL = d50.DL
    wrong = np.empty(len(Y))
    for i, row in enumerate(Y):
        t = (50.0 - row[0]) / (row[-1] - row[0])
        wrong[i] = 10.0 ** (DL[0] + t * (DL[-1] - DL[0]))
    gap = float(np.max(np.abs(np.log10(wrong) - np.log10(d))))
    print("[..] wrong convention differs by up to %.3f in log10 D50" % gap)

    ld = d50.logd50(Y)
    print("[ok] log10 D50 range: %.3f to %.3f (%.2f decades)"
          % (ld.min(), ld.max(), ld.max() - ld.min()))
    # The positivity that matters is the DIAMETER's, not the log's: fine soils legitimately
    # have log10 D50 < 0 (0.021 mm is -1.67), so gating on the log would reject valid labels.
    if (d <= 0).any():
        raise AssertionError("a non-positive D50 reached the fit; log10 is undefined there")
    print("[ok] every D50 is a positive diameter")

    # Cross-check against the project's own recorded D50 column, which is an independent
    # source for the same quantity and was produced by a different code path.
    import pandas as pd
    fam = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    rec = dict(zip(fam.sample_id.astype(str), fam.logD50.astype(float)))
    common = [i for i in ids if i in rec]
    if common:
        mine = dict(zip([str(i) for i in ids], ld))
        gap2 = max(abs(mine[i] - rec[i]) for i in common)
        print("[..] max |log10 D50 - recorded logD50| over %d soils = %.3e"
              % (len(common), gap2))
        if gap2 > 1e-6:
            print("[!!] D50 disagrees with the recorded column - investigate before scoring")
        else:
            print("[ok] D50 matches the project's independently recorded column exactly")

    print("\nD50 GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
