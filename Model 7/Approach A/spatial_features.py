"""spatial_features.py - the four within-tile statistics M7-A adds, and nothing else.

This module deliberately knows no labels, no camera, no file paths and no experiment. It takes a
tile as a float array and returns four numbers. That narrow surface is what makes the "no leakage"
claim checkable rather than aspirational, and it is the same reason Model 5's alignment.py was
isolated.

The mask and the grey field are taken from the FROZEN preprocessing definitions
(preprocess/verify.py:105-128) so that A1-A4 are computed on exactly the pixel set the existing 12
features are computed on:

    g = mean over channels
    m = g > percentile(g, 12)
    e = residual after removing an illumination base of sigma 12 mm

Every statistic is a function of a tile's own pixels. Nothing here reads another tile, a
position, or a label.

THE DEFINITIONS BELOW ARE THE OWNER'S RULINGS OF 2026-09-29, which supersede this plan's task briefs
and the code that was written from them:

  A1  structure-tensor anisotropy log((l2+eps)/(l1+eps)). Untouched, as ruled.
  A2  a PLAIN ratio L(0 deg)/L(90 deg), not a log-ratio. A profile line whose entire row or column
      falls outside the mask is SKIPPED: excluded from that profile's mean and from every lag
      product. It is never written as 0.0. Fewer than MIN_PROFILE_LINES valid lines in either
      direction and A2 is NaN. A direction whose masked profile has NO variance at all has no
      autocorrelation and therefore no e-folding length, and A2 is NaN for that reason too: the
      EPS floor used to answer it with exactly 1.0, which is what a perfectly isotropic tile
      returns. The owner's follow-up ruling on this cell (option a) adds two more undefined cases,
      both read off the registered 1/e definition and off the integer lag grid rather than off any
      energy or magnitude floor: a direction whose autocorrelation NEVER falls below 1/e inside the
      evaluated lags, and a direction whose 1/e crossing lands at lag 0, below one full lag step.
      Neither has an e-folding length, so A2 is NaN; the maximum lag and a sub-lag-step length are
      both stand-ins the ruling forbids. Its known sensitivity to ppm (through the 12 mm
      illumination base) is accepted by ruling 2 and is not corrected here.
  A3  range / (A3_REFERENCE_LAG * tile extent in mm). The sill LOCATES the range and is never the
      denominator. A lag is ONE measurement over the two registered directions, so a lag is
      measurable only when BOTH directions have at least MIN_PAIRS_PER_LAG masked pairs: one short
      direction undefineds the lag rather than leaving the other to answer alone, which is the same
      all-or-nothing rule A4 applies to its four directions. Such a lag is NaN and is not carried
      forward from the previous lag; the sill is the trailing window's mean over MEASURED lags, so
      an unmeasured lag inside that window leaves the sill undefined rather than being dropped from
      the average, and an unmeasured lag in front of the crossing means the first lag to reach
      A3_SILL_FRACTION of the sill cannot be located. A variogram that never reaches the crossing
      has no range at all, and is answered with NaN rather than with the maximum lag.
  A4  omni - C(45 deg), where omni is the mean of C over 0, 45, 90 and 135 deg. The axis directions
      use lag_px = round(A4_LAG_MM * ppm), which is 5 px at the canonical 4.552516 px/mm; BOTH
      diagonals use A4_DIAG_PX = 3 px per axis, the closest this grid gets to the same physical
      distance (3*sqrt(2) = 4.243 px against 5 px, where 4 px per axis would give 5.657 px). A
      direction short of MIN_PAIRS_PER_DIRECTION pairs makes A4 NaN.

Every numeric parameter the spec's frozen table names is a module constant below, so that "no
parameter per statistic" is auditable without reading function bodies. A4_DIAG_PX is the one
constant NOT in that table: it comes from ruling 4 and from the amended A4 cell. Its absence from
the frozen table is reported to the owner rather than silently amended here.

Two failure modes are kept apart on purpose:

  TileDegenerate - this TILE cannot carry this measurement: too few mask pixels, too few valid
                   profile lines, a profile with no variance to correlate, an autocorrelation that
                   never falls below 1/e inside its evaluated lags, a 1/e crossing that lands below
                   one full lag step, too few masked pairs at a lag or in a direction, no
                   measurable sill, and a sill window or a range search with an unmeasured lag in
                   it. tile_spatial converts it to NaN for the one key and
                   returns. It reaches the caller only when the whole tile is unusable, i.e. only
                   from mask_and_field.
  ValueError     - this CALL is invalid: wrong shape, non-finite pixels, a tile too small to lag,
                   ppm <= 0, NaN, inf or None, a matrix of the wrong rank, or a feature column that
                   holds an inf where the aggregation expects NaN. It is never catchable as
                   TileDegenerate, so a systemic bug cannot be logged as a hard tile.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "preprocess") not in sys.path:
    sys.path.insert(0, str(ROOT))
from preprocess.verify import BASE_SIGMA_MM, _illumination_base  # noqa: E402

SPATIAL: Tuple[str, ...] = ("A1", "A2", "A3", "A4")

# --- the spec's frozen numeric parameter table: one constant per row, nothing else free --------
A4_LAG_MM = 1.0              # the fixed pair lag, in mm
A3_REFERENCE_LAG = 0.5       # A3's denominator is this times the tile extent in mm
A3_SILL_FRACTION = 0.9       # the variogram crossing that defines the range
A3_SILL_WINDOW = 0.10        # trailing fraction of lags averaged to form the sill
MIN_MASK_PIXELS = 16         # below this the whole tile is degenerate
MIN_PAIRS_PER_LAG = 8        # below this a variogram lag is undefined
MIN_PAIRS_PER_DIRECTION = 8  # below this an A4 direction is undefined
MIN_PROFILE_LINES = 4        # below this an A2 direction is undefined
EPS = 1e-12                  # floor inside every log and ratio
ACF_MAX_LAG_FRACTION = 0.5   # A2's autocorrelation spans this fraction of the profile length
A4_DIAG_PX = 3               # ruling 4's per-axis diagonal offset; spec table row A4_DIAG_PX


class TileDegenerate(Exception):
    """This tile cannot carry this measurement. A property of the tile, never of the call.

    Raised by the profile, variogram and direction helpers. tile_spatial catches it per statistic
    and writes NaN into that one key; mask_and_field raises it before any statistic exists, which is
    the only case that reaches the caller.
    """


def _as_tile(a: np.ndarray) -> np.ndarray:
    """Validate the array half of a call. Invalid, not degenerate: a bug in the caller."""
    a = np.asarray(a, float)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("expected an HxWx3 tile, got shape %s" % (a.shape,))
    if not np.all(np.isfinite(a)):
        raise ValueError("tile carries %d non-finite pixels; a tile may not contain NaN or inf"
                         % int(np.count_nonzero(~np.isfinite(a))))
    if a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError("tile is %dx%d, too small to gradient or to lag" % a.shape[:2])
    return a


def _as_ppm(ppm: float) -> float:
    """Validate the scale half of a call. A missing or impossible ppm is a bug, not degeneracy."""
    try:
        p = float(ppm)
    except (TypeError, ValueError):
        raise ValueError("ppm must be a real number, got %r" % (ppm,)) from None
    if not np.isfinite(p) or p <= 0.0:
        raise ValueError("ppm must be positive and finite, got %r" % (ppm,))
    return p


def mask_and_field(a: np.ndarray, ppm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grey field, mask and illumination residual, using the tile's own recorded scale.

    ppm comes from the manifest's normalized_ppm, never from a camera or device name. Deriving it
    from the tile extent instead would silently assume every tile is 256 px at 4.5525 px/mm, and the
    statistics are all in mm, so that assumption would be load-bearing.
    """
    a = _as_tile(a)
    ppm = _as_ppm(ppm)
    g = a.mean(axis=2)
    m = g > np.percentile(g, 12)
    n_masked = int(m.sum())
    if n_masked < MIN_MASK_PIXELS:
        raise TileDegenerate("degenerate mask: only %d pixels survive the 12th pct, need %d"
                             % (n_masked, MIN_MASK_PIXELS))
    base = _illumination_base(g, BASE_SIGMA_MM * ppm)
    e = np.where(m, g - base, 0.0)
    return g, m, e


def _masked_profile(field: np.ndarray, mask: np.ndarray, axis: int) -> Tuple[np.ndarray, int]:
    """Mean of the masked pixels along each line, plus how many lines produced a mean at all.

    axis=1 profiles along x (0 deg): each column is one line, reduced over rows. axis=0 profiles
    along y (90 deg): each row is a line, reduced over columns. A line with no masked pixel has no
    mean, so its entry is NaN and it is not counted. This is ruling 2: a masked-out line is skipped,
    never substituted by 0.0, which is what used to inject a hard step into a profile whose other
    entries were grey levels near 120.
    """
    reduce = 1 - axis
    cnt = mask.sum(axis=reduce)
    tot = np.where(mask, field, 0.0).sum(axis=reduce)
    prof = np.full(np.shape(cnt), np.nan)
    has = cnt > 0
    prof[has] = tot[has] / cnt[has]
    return prof, int(np.count_nonzero(has))


def _acf_along(field: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Normalised autocorrelation of the masked field along one axis, over ACF_MAX_LAG_FRACTION.

    A lag product survives only where BOTH of its profile entries came from a line with masked
    pixels: a skipped entry is zero in `p` and so contributes nothing to either dot product, while
    the denominator is the energy of the lines that do exist. That is the ordinary masked-pair
    autocorrelation, and it is what makes "skip the line" mean something other than "delete the
    sample", which would silently change the physical length of every lag.

    The denominator's energy is also the direction's own degeneracy condition: a masked profile with
    ZERO variance has no autocorrelation to normalise, so every lag would be 0 / EPS and both
    e-folding lengths would be zero, which the ratio then answers as exactly 1.0 - the number a
    perfectly isotropic tile returns. The condition is structural (energy == 0.0, or not finite),
    not a small-value cutoff, so it adds no threshold and a merely tiny variance still measures.
    What decides a tiny-variance direction is its crossing, in _efold, and not its magnitude: a flat
    profile whose 1/e crossing the lag grid cannot resolve is refused there, below one full lag step.
    """
    prof, lines = _masked_profile(field, mask, axis)
    if lines < MIN_PROFILE_LINES:
        raise TileDegenerate("direction %d deg has %d valid profile lines, need %d"
                             % (0 if axis == 1 else 90, lines, MIN_PROFILE_LINES))
    keep = np.isfinite(prof)
    p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
    energy = float(np.dot(p, p))
    if not np.isfinite(energy) or energy == 0.0:
        raise TileDegenerate("direction %d deg has a zero-variance masked profile over %d finite "
                             "lines: nothing correlates with anything, so it has no e-folding "
                             "length" % (0 if axis == 1 else 90, int(np.count_nonzero(keep))))
    den = energy + EPS
    n = prof.size
    nlag = max(1, int(ACF_MAX_LAG_FRACTION * n))
    return np.array([float(np.dot(p[:n - k], p[k:]) / den) for k in range(nlag)])


def _efold(ac: np.ndarray) -> float:
    """First lag at which the autocorrelation falls below 1/e, in samples (float, interpolated).

    Two answers are not measurements and both raise TileDegenerate, which is the owner's follow-up
    ruling on the A2 cell (option a): the ratio needs a measurable decay in BOTH directions.

      * an ACF that never falls below 1/e inside the evaluated lags. The pre-ruling code gave the
        maximum lag for it, which is a range-like stand-in - the same error A3 was ruled to answer
        with NaN when its variogram never reaches its sill;
      * a crossing that lands at lag 0, i.e. below one full lag step of the integer lag grid the
        profile is sampled on. The decorrelation is then unresolvable at this sampling, and the
        pre-ruling code reported it as a tiny positive length that the plain ratio blew up into a
        1e13 or 1e-14 "anisotropy".

    Both tests read the registered 1/e definition and the lag grid. Neither is an energy or
    magnitude floor: the only magnitude-shaped test on the A2 path is the structural
    `energy == 0.0 or not finite` guard in _acf_along, which this ruling leaves exactly as it stood.
    """
    thr = 1.0 / np.e
    for k in range(1, len(ac)):
        if ac[k] < thr:
            span = ac[k - 1] - ac[k]
            frac = (ac[k - 1] - thr) / span if span > EPS else 0.0
            length = (k - 1) + float(frac)
            if length < 1.0:
                raise TileDegenerate("the 1/e crossing lands at lag %r, below one full lag step of "
                                     "a %d-lag profile: the decorrelation is unresolvable at this "
                                     "sampling, so there is no e-folding length"
                                     % (length, len(ac)))
            return length
    raise TileDegenerate("the autocorrelation never falls below 1/e within the %d evaluated lags, "
                         "so this direction has no e-folding length; the maximum lag is not one"
                         % len(ac))


def _a2(e: np.ndarray, m: np.ndarray) -> float:
    """A2 = L(0 deg) / L(90 deg): a plain ratio of two e-folding lengths, each measured in mm along
    its own profile. Scale-free because both lengths share a unit, so mm is a reporting statement,
    not a divisor. Positive, and greater than 1 exactly when the field correlates further along x
    than along y.

    Either direction can raise TileDegenerate - too few valid lines, a profile with no variance to
    correlate, an autocorrelation that never falls below 1/e inside the evaluated lags, or a 1/e
    crossing that lands below one full lag step - which tile_spatial answers with NaN. There is no
    log in this path: ruling 2 removed it, and `thr = 1.0 / np.e` is the only transcendental left.
    The EPS floor stays inside the ratio because a length of one lag step is the shortest thing this
    grid can measure; what is not legal is a direction that produced no measurement at all, and
    _acf_along and _efold now refuse those before the ratio can flatten one into 1.0 or blow the
    other up into 1e13.
    """
    l0 = _efold(_acf_along(e, m, axis=1))
    l90 = _efold(_acf_along(e, m, axis=0))
    return float((l0 + EPS) / (l90 + EPS))


def _variogram(g: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Half the mean squared difference of masked pairs at each lag, averaged over 0 and 90 deg.

    All-or-nothing per lag, which is the same rule _a4 applies to its four directions: the lag is
    ONE measurement over the two registered directions, so a lag is measurable only when BOTH
    directions contribute at least MIN_PAIRS_PER_LAG masked pairs. A lag with one short direction
    is NaN rather than the surviving direction's value, because averaging a single direction answers
    the registered two-direction statistic with a different one. NaN is written explicitly, so no
    mean-of-empty-slice warning can reach the runner.

    NaN also stays NaN: ruling 5 forbids answering an unmeasurable lag with the previous lag's
    value, which is what the pre-ruling fallback did.
    """
    h = np.arange(1, max(2, g.shape[0] // 2))
    gam = []
    for k in h:
        vals = []
        short = 0
        for ax in (0, 1):
            sl_a = [slice(None)] * 2; sl_a[ax] = slice(None, -k)
            sl_b = [slice(None)] * 2; sl_b[ax] = slice(k, None)
            keep = m[tuple(sl_a)] & m[tuple(sl_b)]
            n = int(np.count_nonzero(keep))
            if n < MIN_PAIRS_PER_LAG:
                short += 1
            else:
                d = g[tuple(sl_a)][keep] - g[tuple(sl_b)][keep]
                vals.append(0.5 * float(np.mean(d ** 2)))
        measured = short == 0 and len(vals) == 2
        gam.append(float(np.mean(vals)) if measured else float("nan"))
    return np.asarray(gam, float)


def _variogram_range(g: np.ndarray, m: np.ndarray, ppm: float) -> float:
    """Lag where the masked variogram first reaches A3_SILL_FRACTION of its own sill, in mm.

    Averaged over the horizontal and vertical directions only. No diagonal, no free lag count: the
    two directions are what an omnidirectional range needs and adding more would be a scale sweep.
    The sill locates the range and is not the denominator (ruling 3).

    An unmeasured lag is unmeasured, not skipped, in both places it can sit:
      * in the trailing window - the sill is the mean of that window, so a window with a hole in it
        is not the registered A3_SILL_WINDOW fraction of anything, and dropping the hole and
        averaging the survivors would silently form the sill from a shorter window;
      * in front of the crossing - the range is the FIRST lag to reach the sill fraction, and a hole
        earlier in the variogram makes "first" unknowable.
    A hole BEHIND the crossing changes nothing, because the lags after the range are not read.
    Where the sill cannot be measured, or no lag reaches it, there is no range: TileDegenerate, not
    the last lag.
    """
    gam = _variogram(g, m)
    win = max(1, int(A3_SILL_WINDOW * gam.size))
    tail = gam[-win:]
    unmeasured = ~np.isfinite(gam)
    holes = int(np.count_nonzero(~np.isfinite(tail)))
    if holes:
        raise TileDegenerate("variogram sill window has %d unmeasured lag(s) of its %d, so the "
                             "trailing %g fraction of the variogram cannot be averaged"
                             % (holes, win, A3_SILL_WINDOW))
    sill = float(np.mean(tail))
    if sill <= EPS:
        raise TileDegenerate("variogram sill is %.6g, no range can be located on it" % sill)
    reached = np.isfinite(gam) & (gam >= A3_SILL_FRACTION * sill)
    if not bool(np.any(reached)):
        raise TileDegenerate("variogram never reaches %g of its sill, so it has no range"
                             % A3_SILL_FRACTION)
    hit = int(np.argmax(reached))
    holes_before = int(np.count_nonzero(unmeasured[:hit]))
    if holes_before:
        raise TileDegenerate("variogram has %d unmeasured lag(s) before the crossing at lag %d, so "
                             "lag %d is not the first lag to reach %g of the sill"
                             % (holes_before, hit + 1, hit + 1, A3_SILL_FRACTION))
    return float(np.arange(1, gam.size + 1)[hit]) / ppm


def _pair_slices(shape: Tuple[int, int], oy: int, ox: int):
    """The two shifted views whose overlap holds every pixel pair at offset (oy, ox)."""
    h, w = shape
    r0, r1 = (0, h - oy) if oy >= 0 else (-oy, h)
    c0, c1 = (0, w - ox) if ox >= 0 else (-ox, w)
    return ((slice(r0, r1), slice(c0, c1)),
            (slice(r0 + oy, r1 + oy), slice(c0 + ox, c1 + ox)))


def _direction_contrast(g: np.ndarray, m: np.ndarray, oy: int, ox: int,
                        mad: float) -> Tuple[float, int]:
    """C(d): the fraction of masked pairs at offset (oy, ox) whose grey difference exceeds the MAD.

    Returns (fraction, n_pairs). Raises TileDegenerate below MIN_PAIRS_PER_DIRECTION: ruling 5
    forbids answering that with 0.0, and 0.0 is indistinguishable here from a genuinely uniform tile
    such as the ramp, where every direction has thousands of pairs and none of them exceeds the MAD.
    """
    sl_a, sl_b = _pair_slices(g.shape, oy, ox)
    keep = m[sl_a] & m[sl_b]
    n = int(np.count_nonzero(keep))
    if n < MIN_PAIRS_PER_DIRECTION:
        raise TileDegenerate("direction at offset (%d, %d) has %d masked pairs, need %d"
                             % (oy, ox, n, MIN_PAIRS_PER_DIRECTION))
    d = np.abs(g[sl_a][keep] - g[sl_b][keep])
    return float(np.mean(d > mad)), n


def _a4(g: np.ndarray, m: np.ndarray, ppm: float) -> float:
    """A4 = omni - C(45 deg), omni the mean of C over 0, 45, 90 and 135 deg.

    Axis directions are (0, +lag_px) and (+lag_px, 0); both diagonals are (+A4_DIAG_PX, +A4_DIAG_PX)
    and (+A4_DIAG_PX, -A4_DIAG_PX), which is the 3 px per-axis offset ruling 4 chose. Any one
    direction short of MIN_PAIRS_PER_DIRECTION pairs makes the whole statistic NaN: an omni average
    over three directions is not the registered measurement, and substituting the three that happen
    to exist would be a fallback number.
    """
    lag_px = int(round(A4_LAG_MM * ppm))
    if lag_px < 1:
        raise ValueError("ppm=%r puts the %g mm lag under one pixel (lag_px=%d): the tile's "
                         "recorded scale cannot carry this measurement"
                         % (ppm, A4_LAG_MM, lag_px))
    mad = float(np.median(np.abs(g[m] - np.median(g[m])))) + EPS
    axis = [_direction_contrast(g, m, 0, lag_px, mad)[0],
            _direction_contrast(g, m, lag_px, 0, mad)[0]]
    diag = [_direction_contrast(g, m, A4_DIAG_PX, A4_DIAG_PX, mad)[0],
            _direction_contrast(g, m, A4_DIAG_PX, -A4_DIAG_PX, mad)[0]]
    terms = {0: axis[0], 90: axis[1], 45: diag[0], 135: diag[1]}
    omni = float(np.mean([terms[d] for d in (0, 45, 90, 135)]))
    return omni - terms[45]


def _a3(g: np.ndarray, m: np.ndarray, ppm: float) -> float:
    """A3 = range_mm / (A3_REFERENCE_LAG * extent_mm): the variogram range over half the tile's own
    extent, dimensionless. The tile extent is g.shape[0] / ppm and the range is h[hit] / ppm, so ppm
    cancels exactly and A3 is a pure lag fraction: it carries no scale information (measured, and
    pinned by test_a3_is_ppm_invariant_by_construction).
    """
    range_mm = _variogram_range(g, m, ppm)
    extent_mm = g.shape[0] / ppm          # px -> mm at the tile's own recorded scale
    return float(range_mm / (A3_REFERENCE_LAG * extent_mm))


def tile_spatial(a: np.ndarray, ppm: float) -> Dict[str, float]:
    """The four frozen statistics for one tile. Arrays in, numbers out, NaN where undefined.

    The try/except is per statistic, which is the point of the split: a tile whose 135 deg direction
    has no pairs is still a tile whose A1, A2 and A3 were measured, and only the missing column
    becomes NaN. mask_and_field is outside every try, so a genuinely unusable tile still reaches the
    caller as TileDegenerate instead of arriving as four quiet NaNs.
    """
    g, m, e = mask_and_field(a, ppm)
    gy, gx = np.gradient(g)
    jxx = float(np.mean(gx[m] ** 2))
    jyy = float(np.mean(gy[m] ** 2))
    jxy = float(np.mean(gx[m] * gy[m]))
    tr, det = jxx + jyy, jxx * jyy - jxy ** 2
    disc = max(tr * tr / 4.0 - det, 0.0)
    l1 = tr / 2.0 + np.sqrt(disc)
    l2 = max(tr / 2.0 - np.sqrt(disc), 0.0)
    a1 = float(np.log((l2 + EPS) / (l1 + EPS)))

    out = {"A1": a1, "A2": float("nan"), "A3": float("nan"), "A4": float("nan")}
    for key, fn in (("A2", lambda: _a2(e, m)), ("A3", lambda: _a3(g, m, ppm)),
                    ("A4", lambda: _a4(g, m, ppm))):
        try:
            out[key] = float(fn())
        except TileDegenerate:
            out[key] = float("nan")
    if any(np.isinf(v) for v in out.values()) or np.isnan(out["A1"]):
        raise ValueError("a statistic came back infinite, or A1 came back NaN, which is a bug and "
                         "not a degeneracy: %r" % (out,))
    return out


# --------------------------------------------------------------------------- placebo constructors
#
# The four controls of the pre-registration are constructors over the same arrays, not new
# statistics: nothing here computes a number about a tile, it rearranges one. Each takes its seed as
# a required argument, so a run is reproducible from the seed that was recorded rather than from a
# global state that was set somewhere else, and the statistic functions above stay free of any RNG
# (which check_spatial_features.py scans for).


def phase_scrambled(a: np.ndarray, seed: int) -> np.ndarray:
    """P-RAND: randomise the FFT phase of each channel, then re-rank onto that channel's own values.

    Without the re-ranking a plain phase scramble also Gaussianises the marginal, so a gain the
    placebo reproduces could be about the histogram rather than about arrangement.

    The pre-registration claims this keeps the pixel multiset exactly AND the entire power spectrum
    exactly. Those two claims fight each other and only the first survives: the un-reranked
    reconstruction does have the magnitude spectrum exactly, but re-ranking is a monotone map that
    replaces each pixel by a different one of the tile's own values, which changes the spectrum.
    The implementation is the registered one and the deviation is MEASURED, not argued:
    check_spatial_features.py reports the radial relative L1 distance per tile. What is exact is the
    per-channel multiset, and therefore the mean and the DC power bin.
    """
    a = _as_tile(a)
    out = np.empty_like(a)
    rng = np.random.default_rng(int(seed))
    for ch in range(a.shape[2]):
        g = a[:, :, ch]
        F = np.fft.fft2(g)
        phase = np.exp(2j * np.pi * rng.random(g.shape))
        phase[0, 0] = 1.0
        rec = np.real(np.fft.ifft2(np.abs(F) * phase))
        order = np.argsort(np.argsort(rec.ravel()))
        out[:, :, ch] = np.sort(g.ravel())[order].reshape(g.shape)
    return out


def shuffle_columns(mat: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """P-SHUF: permute every column independently. Marginals exact, joint structure destroyed.

    A ROW permutation would be inert here for the same reason it was inert in Model 5: a per-column
    median and a per-column distribution do not care about row order. The stacked
    permutation vectors are returned because a control nobody can reconstruct is not a control.
    """
    mat = np.asarray(mat, float)
    if mat.ndim != 2:
        raise ValueError("shuffle_columns takes a 2-D matrix, got shape %s" % (mat.shape,))
    rng = np.random.default_rng(int(seed))
    vecs = np.array([rng.permutation(mat.shape[0]) for _ in range(mat.shape[1])])
    out = np.empty_like(mat)
    for j in range(mat.shape[1]):
        out[:, j] = mat[vecs[j], j]
    return out, vecs


def linear_combinations(X12: np.ndarray, seed: int) -> np.ndarray:
    """P-COLS: four fixed random mixtures of the existing 12 columns.

    No new information by construction - every column is in the span of the input - so any gain this
    control also achieves is a feature-count effect rather than a content effect.
    """
    X = np.asarray(X12, float)
    if X.ndim != 2:
        raise ValueError("linear_combinations takes a 2-D matrix, got shape %s" % (X.shape,))
    rng = np.random.default_rng(int(seed))
    W = rng.normal(size=(X.shape[1], 4))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    return X @ W


# --------------------------------------------------------------------------- the frozen aggregation
def _median_column(vals: np.ndarray, key: str, where: str) -> float:
    """The plain median of one column, keeping NaN and non-finite apart.

    Two different conditions were being answered with the same two tokens:

      an ALL-NaN column is a real outcome. np.nanmedian answers it with NaN plus a RuntimeWarning,
      and NaN is the right value after ruling 5 - a dead column must not be revived with a fallback
      number. Under ruling 2 a loose warning is a test failure, so the one expected occurrence is
      handled HERE, at the site that produces it: the case is detected before nanmedian is called,
      re-emitted as an explicit notice naming the statistic, and answered with NaN.

      an inf, or anything else non-finite that is not NaN, is not a missing value at all.
      tile_spatial calls a non-finite statistic a bug rather than a degeneracy, so a column holding
      one is an invalid CALL and raises ValueError. It used to be reported as an all-NaN column
      (which was false: those entries were infs) when every row was bad, and answered with a silent
      inf when only one row was.

    Nothing is filtered globally, and a warning from any other cause still propagates and fails the
    run.
    """
    v = np.asarray(vals, float)
    if v.size == 0:
        raise ValueError("%s: no rows to aggregate for %s" % (where, key))
    bad = v[np.isinf(v)]
    if bad.size:
        raise ValueError("%s: %s has %d non-finite value(s) of %d rows (%s); only NaN is a legal "
                         "missing value here, so this is an invalid call rather than an all-NaN "
                         "column" % (where, key, int(bad.size), v.size,
                                     ", ".join(repr(float(x)) for x in np.unique(bad))))
    if bool(np.isnan(v).all()):
        warnings.warn("%s: %s is NaN in every one of %d rows, so its median is NaN"
                      % (where, key, v.size), RuntimeWarning, stacklevel=3)
        return float("nan")
    return float(np.nanmedian(v))


def _block_median_table(rows, where: str) -> Dict[str, float]:
    if not rows:
        raise ValueError("%s: no rows to aggregate" % where)
    arr = np.array([[r[k] for k in SPATIAL] for r in rows], float)
    return {k: _median_column(arr[:, i], k, where) for i, k in enumerate(SPATIAL)}


def image_block(tile_rows) -> Dict[str, float]:
    """Median of a tile list -> one image row. The frozen first aggregation, same column set."""
    return _block_median_table(list(tile_rows), "image_block")


def soil_block(image_rows) -> Dict[str, float]:
    """Median of image rows -> one soil row. The frozen second aggregation, same column set."""
    return _block_median_table(list(image_rows), "soil_block")
