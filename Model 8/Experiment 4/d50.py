"""d50.py - extracting the median grain size from a cumulative GSD curve.

WHAT D50 IS, AND THE ONE ERROR THAT IS EASY TO MAKE HERE
  D50 is the diameter at which the cumulative curve EQUALS 50% passing. The labels are
  interpolated, not measured at the 11 supports, so the answer usually sits BETWEEN two
  supports and must be interpolated.

  The two bracketing supports are found by searching the CUMULATIVE VALUES for 50
  (np.searchsorted on the monotone curve, not on the diameters), and the interpolation
  then happens in LOG10-DIAMETER between those two supports, because the metric and the
  DIN EN ISO 14688-1 grid are both logarithmic. Interpolating percent-vs-logdiameter
  instead - i.e. treating the curve as the function and solving for x - is the natural
  mistake and it is wrong: it returns a number where percent is linear in log-diameter,
  not where diameter is linear in percent.

  `check_d50.py` asserts the round-trip below, which is the property that actually
  distinguishes the two conventions.

WHY THIS IS ITS OWN MODULE
  Experiment 1's entire hypothesis is about the functional form of the map from D50 to the
  curve. If D50 itself were computed inconsistently between the fit and the evaluation, the
  experiment would be measuring its own bug. So the extraction lives alone, is asserted, and
  is imported by the head rather than reimplemented there.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

# The 11 supports, as they appear as column names in sample_submission.csv.
# Hardcoded as bare strings deliberately: str(2.0) gives '2.0' and the column is '2'.
SUPPORTS: Tuple[str, ...] = ("0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                             "2", "6.3", "20", "63", "200")
DL = np.log10(np.array([float(s) for s in SUPPORTS]))  # the metric's abscissa


def verify_supports(root) -> None:
    """Abort if the shipped sample_submission.csv disagrees with the hardcoded supports.

    A silent mismatch here would corrupt every interpolated diameter while leaving the
    pipeline looking healthy, so it is checked rather than assumed.
    """
    cols = list(pd.read_csv(str(root) + "/data/sample_submission.csv", nrows=0).columns)
    if cols[0] != "sample_id" or tuple(cols[1:]) != SUPPORTS:
        raise AssertionError(
            "sample_submission.csv supports %r do not match d50.SUPPORTS %r; the labels and "
            "the metric abscissa would disagree" % (tuple(cols[1:]), SUPPORTS))


def diameter_at(Y: np.ndarray, q: float) -> np.ndarray:
    """The diameter at which each curve reaches q percent passing, in the curve's own units.

    Accepts (n, 11) and returns (n,). Out-of-range q clamps to the end supports: a curve
    that never reaches 50% is a real (if odd) label, and inventing a diameter for it would
    put a fabricated value into a fit.
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    out = np.empty(len(Y))
    for i, row in enumerate(Y):
        r = np.maximum.accumulate(row)          # projection's first step; see transfer_eval.project
        j = int(np.searchsorted(r, q))
        if j == 0:                              # already past q at the smallest support
            out[i] = float(SUPPORTS[0])
        elif j >= len(r):                       # never reaches q
            out[i] = float(SUPPORTS[-1])
        else:
            r0, r1 = r[j - 1], r[j]
            t = 0.0 if r1 == r0 else (q - r0) / (r1 - r0)
            out[i] = 10.0 ** (DL[j - 1] + t * (DL[j] - DL[j - 1]))
    return out


def logd50(Y: np.ndarray) -> np.ndarray:
    """log10 of the median grain size, the scalar this experiment parameterises the curve by."""
    return np.log10(diameter_at(Y, 50.0))


def curve_at(Y: np.ndarray, d) -> np.ndarray:
    """The cumulative percentage passing at diameter d. The inverse of diameter_at.

    d may be a scalar (one diameter for every curve) or a length-n array (each curve read
    at its own diameter), which is what the round-trip assertion needs.

    Used only by that assertion, not by the head.
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    d = np.broadcast_to(np.asarray(d, float), (len(Y),)).copy()
    R = np.maximum.accumulate(Y, axis=-1)
    lo_d, hi_d = float(SUPPORTS[0]), float(SUPPORTS[-1])
    out = np.empty(len(Y))
    at_floor = d <= lo_d
    at_ceil = d >= hi_d
    mid = ~(at_floor | at_ceil)

    out[at_floor] = 0.0
    out[at_ceil] = 100.0

    if mid.any():
        dm = np.log10(d[mid])
        # searchsorted on the log-diameter axis, then step back one support so that k is the
        # first support strictly above d and k-1 the last at or below it.
        k = np.clip(np.searchsorted(DL, dm), 1, len(SUPPORTS) - 1)
        t = (dm - DL[k - 1]) / (DL[k] - DL[k - 1])
        rows = np.where(mid)[0]
        out[rows] = R[rows, k - 1] + t * (R[rows, k] - R[rows, k - 1])
    return out
