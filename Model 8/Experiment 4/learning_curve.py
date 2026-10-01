"""learning_curve.py - is this project DATA-limited or MODEL-limited?

THE QUESTION
  Ten submissions now exist. Six land between 55.8 and 57.9 and the top four are within 0.7
  EMD of each other. Two readings of that plateau are available and they imply opposite
  decisions:

    (a) MODEL-limited - the head is wrong, and better features/regressors would move us.
    (b) DATA-limited  - 24 soils (~8-10 effectively independent, six near-duplicate pairs, one
                        measurement each, no replicates) simply do not contain enough signal to
                        localise a grain-size curve better, and 55.8 is near the information
                        ceiling.

  Nothing in the project has ever distinguished these. Every previous experiment varied the
  MODEL and reported that it did not help - which is equally consistent with (b).

  The discriminator is the learning curve: hold the model fixed, vary the amount of training
  data, and ask whether the error has stopped falling. A curve still descending steeply at
  n = 24 means more labels would buy accuracy and the model is fine. A curve that has flattened
  means more labels would buy almost nothing and no amount of model work will help.

  Crucially this is measurable INTERNALLY, for free, on the 24 labelled soils we already have.
  That is why it is run before any submission is spent.

DESIGN - it must mimic the real task, not a friendlier one
  The real task is extrapolating to 10 UNSEEN iPhone soils. Evaluating a learning curve by
  random train/test splits of the 24 would measure interpolation among Motorola/Samsung soils
  and would flatter the model enormously.

  So every point on this curve holds out a whole FAMILY (never a soil), exactly as the
  registered protocol does, and then draws n training soils at random from the remaining
  families. Family g is absent from every part of the fit - from the features, the labels, the
  scaler, the curve basis and the ridge. That is the same extrapolation the test set demands,
  just at smaller n.

  Alpha is FIXED at 3.0, not re-selected per point. E2's alpha probe established 3.0 as the
  external optimum, and re-selecting on a subset would confound "less data" with "a
  differently-tuned model". Fixing it means the only thing varying along the curve is n.

WHAT THIS DELIBERATELY CANNOT SHOW
  It measures the learning curve for TRANSFER to unseen families among Android soils. The test
  set is unseen CAMERA as well as unseen soil. Those are different curves, and this one cannot
  speak to the camera part - with zero iPhone training images there is no way to estimate the
  camera axis from this data at all. Recorded now, before the result, so it cannot be
  retrofitted as an excuse.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ruler as R          # noqa: E402  registered nested protocol
import transfer_eval as tv  # noqa: E402  frozen head

ALPHA = 3.0          # fixed: E2's externally-optimal value
N_DRAWS = 120        # per point, per family
SEED = 20261001
SIZES: Tuple[int, ...] = (4, 6, 8, 10, 12, 14, 16, 18, 20, 21)


def curve_point(X: np.ndarray, Y: np.ndarray, F: np.ndarray, n: int,
                rng: np.random.Generator) -> Tuple[float, List[float]]:
    """Mean held-out EMD when only n soils are used for fitting, family g always excluded."""
    per_family: List[float] = []
    for g in sorted(set(F)):
        pool = np.where(F != g)[0]        # family g is NEVER available to the fit
        if n > len(pool):
            continue
        held = np.where(F == g)[0]
        errs: List[float] = []
        for _ in range(N_DRAWS):
            take = rng.choice(pool, size=n, replace=False)
            P = tv.fit_predict(X[take], Y[take], X[held], ALPHA)
            errs.extend(tv.emd_pair(P[k], Y[i]) for k, i in enumerate(held))
        if errs:
            per_family.append(float(np.mean(errs)))
    if not per_family:
        return float("nan"), []
    return float(np.mean(per_family)), per_family


def main() -> int:
    print("=" * 76)
    print("MODEL 8 / EXPERIMENT 4 - LEARNING CURVE (data-limited or model-limited?)")
    print("=" * 76)

    X, Y, fams, ids = R.design()
    F = np.asarray(fams)

    # ---- G0: the anchors must hold before any curve is reported ----------
    nested = float(R.lofo_errors(X, Y, F).mean())
    oracle = float(R.alpha_curve(X, Y, F)[ALPHA])
    g0a, g0b = abs(nested - R.INTERNAL_BASELINE), abs(oracle - 41.1475158656982032)
    print("\n--- G0: both registered anchors ---")
    print("    nested %.16f gap %.3e | oracle %.16f gap %.3e" % (nested, g0a, oracle, g0b))
    if g0a > 1e-6 or g0b > 1e-9:
        print("G0 FAILED - nothing reported.")
        return 1
    print("[ok] both anchors reproduced")

    print("\n--- G1: curve, alpha fixed at %.1f, family held out entirely ---" % ALPHA)
    print("    n     mean EMD    SE       95%% CI            vs previous n")
    rng = np.random.default_rng(SEED)
    rows: List[Dict[str, float]] = []
    prev = None
    for n in SIZES:
        m, per_fam = curve_point(X, Y, F, n, rng)
        se = float(np.std(per_fam, ddof=1) / np.sqrt(len(per_fam))) if len(per_fam) > 1 else float("nan")
        lo, hi = m - 1.96 * se, m + 1.96 * se
        delta = "" if prev is None else "%+7.3f" % (m - prev)
        print("    %-4d  %8.3f  %6.3f   [%7.3f, %7.3f]   %s" % (n, m, se, lo, hi, delta))
        rows.append({"n": n, "mean_emd": m, "se": se, "ci_lo": lo, "ci_hi": hi})
        prev = m

    full = float(R.lofo_errors(X, Y, F).mean())
    print("    %-4d  %8.3f  %6s   %19s   (the registered ruler, all 23 fit soils per fold)"
          % (24, full, "-", "-"))

    # ---- the actual question: is it still descending? --------------------
    print("\n--- G2: is the curve still falling at n = 24? ---")
    ns = np.array([r["n"] for r in rows], float)
    ms = np.array([r["mean_emd"] for r in rows], float)
    # Fit EMD = a + b * n**(-c). c near 0 means a pure offset; the informative quantity is
    # whether the curve is close to flat over the last stretch of n we actually have.
    A = np.column_stack([np.ones_like(ns), ns ** -1.0, ns ** -0.5])
    coef, *_ = np.linalg.lstsq(A, ms, rcond=None)
    pred = A @ coef
    ss_res = float(((ms - pred) ** 2).sum())
    ss_tot = float(((ms - ms.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    last = [r for r in rows if r["n"] >= 14]
    span = last[0]["mean_emd"] - last[-1]["mean_emd"]
    print("    n = 14 -> %d : total improvement %.3f EMD over %d extra soils"
          % (last[-1]["n"], span, last[-1]["n"] - last[0]["n"]))
    print("    n = 20 -> 21 (last single step): %+.3f EMD"
          % (rows[-1]["mean_emd"] - rows[-2]["mean_emd"]))
    print("    power-law fit  EMD = %.3f %+.3f/n %+.3f/sqrt(n)   R^2 = %.3f"
          % (coef[0], coef[1], coef[2], r2))

    # What would doubling the data be worth, by the fit?
    for target in (24, 48, 96):
        est = coef[0] + coef[1] / target + coef[2] / np.sqrt(target)
        print("    extrapolated EMD at n = %-3d : %.3f  (%+.3f vs the n=24 ruler)"
              % (target, est, est - full))

    (HERE / "learning_curve.csv").write_text(
        "n,mean_emd,se,ci_lo,ci_hi\n" + "\n".join(
            "%d,%.6f,%.6f,%.6f,%.6f" % (r["n"], r["mean_emd"], r["se"], r["ci_lo"], r["ci_hi"])
            for r in rows) + "\n%d,%.6f,,\n" % (24, full), encoding="ascii")
    (HERE / "learning_curve.json").write_text(json.dumps({
        "alpha_fixed": ALPHA, "n_draws_per_point": N_DRAWS, "seed": SEED,
        "protocol": "hold out one FAMILY entirely, draw n training soils from the rest",
        "curve": rows, "ruler_at_24": full,
        "powerlaw_fit": {"intercept": coef[0], "inv_n": coef[1], "inv_sqrt_n": coef[2],
                         "r2": r2},
        "improvement_14_to_last": span,
        "LIMIT": "measures transfer among unseen families of ANDROID soils. Says nothing "
                 "about the camera axis - there are 0 iPhone training images, so the camera "
                 "component of the learning curve is not estimable from this data.",
    }, indent=2), encoding="ascii")

    print("\n" + "=" * 76)
    print("Learning curve written to learning_curve.csv / .json. No submission generated yet.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())