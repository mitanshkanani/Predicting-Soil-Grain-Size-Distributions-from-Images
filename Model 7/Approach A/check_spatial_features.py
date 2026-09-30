"""Contract test for spatial_features.py. Run: python check_spatial_features.py

WHAT THIS FILE ASSERTS (owner rulings, 2026-09-29, which supersede the plan and the task briefs)
  A1  untouched: log((l2+eps)/(l1+eps)) of the masked structure tensor.
  A2  a PLAIN ratio L(0 deg)/L(90 deg). Not a log-ratio. A profile line whose whole row or column
      lies outside the mask is SKIPPED, never written as 0.0. Fewer than MIN_PROFILE_LINES valid
      lines in a direction -> A2 is NaN. So is a direction whose autocorrelation NEVER falls below
      1/e inside its evaluated lags, and one whose 1/e crossing lands at lag 0, below one full lag
      step: neither has an e-folding length, and the maximum lag and a sub-lag-step length are both
      stand-ins (owner's follow-up ruling on this cell, option a). That ruling adds no numeric
      energy or magnitude floor; the structural energy == 0.0 guard is unchanged.
  A3  range / (0.5 * tile extent in mm), the sill only locating the range. Unchanged, except that a
      lag with too few pairs is no longer carried forward and a range that is never reached is no
      longer answered by the maximum lag: both give NaN.
  A4  omni - C(45 deg), omni = mean of C over 0, 45, 90, 135 deg. Axis directions use the 5 px
      offset at the canonical scale; BOTH diagonals use a 3 px per-axis offset. A direction with
      fewer than MIN_PAIRS_PER_DIRECTION pairs -> A4 is NaN.
  NaN   every undefined statistic is NaN inside an otherwise valid tile; it must not raise and must
      not be answered by a fallback number.
  types TileDegenerate (a genuinely degenerate tile, whole-tile only) vs ValueError (an invalid
      call: wrong shape, non-finite pixels, ppm <= 0 or NaN). Neither is catchable as the other.
  warnings any warning emitted during a check is a FAILURE: main() runs every test with
      warnings.simplefilter("error"), and test_no_module_call_emits_a_warning records warnings
      independently so the guarantee also holds outside this runner.

Checks that used to pin the pre-ruling behaviour were deleted or re-derived. None were relabelled:
the old "pinned:" A4-collapses-to-0.0, A3-max-lag-fallback and degenerate-tile-yields-finite-numbers
checks are gone, and the A2 transpose check was re-derived from "flips sign" to "inverts", which is
what a plain ratio does.

Under the follow-up A2 ruling (option a) five checks that pinned an undefined-returns-a-number
were re-derived, not deleted: "_efold of a non-decaying ACF is the maximum lag", "_efold of an
all-zero ACF is zero samples", "_efold crosses inside the first interval when lag 1 is already below
1/e", "A2 over tiny-but-non-zero variance is a number" (moved onto a profile whose crossing IS
resolvable, so the no-magnitude-floor property it existed for survives), and "MIN_PROFILE_LINES: A2
on the 4-row tile is a number" (that tile clears the line-count floor and is NaN on the crossing
instead, which is now what the check says). Everything else in the 411 that ran before this ruling
still runs and still passes.

MEASURED NUMBERS (brief Step 5: a control that cannot move the statistic is worthless, so these are
recorded here as well as printed by the run; ppm = 4.552516 throughout)

  Mutation sanity, eight healthy tiles plus two degenerate ones:
    A1 from -33.761250 (stripes) to 0.000000 (a 25 px island); A2 from 0.028351 to 35.271945,
    a multiplicative spread of 1244x; A3 from 0.007812 to 0.906250 in its 0.0078125 steps;
    A4 from -0.083333 to 0.006871. A4 on the two stripe orientations is equal to the last bit
    (-0.083333 both), because omni averages 0 and 90 deg: see test_a4_moves_across_texture_regimes.
  Undefined statistics on the constructed degeneracies:
    5 x 5 island  -> A3 and A4 NaN, A1 and A2 finite.
    3 x 40 band    -> A2, A3 and A4 NaN, A1 finite.
    4 x 4 island   -> A3 and A4 NaN (exactly MIN_MASK_PIXELS, so the tile itself is valid).
  P-RAND on blobs(seed=0), real versus scrambled(seed=7):
    real      A1=-0.019575 A2=0.274521 A3=0.398438 A4=-0.004123
    scrambled A1=-0.006563 A2=0.905326 A3=0.046875 A4= 0.004714
    The plan expects A3 to stay close. It does not: 0.351563, i.e. 45 of A3's own quantisation
    steps.
  P-RAND spectrum, radial relative L1 distance, 32 bins, with and without the DC bin:
    blobs 7.157e-04 / 0.74678   coarse16 1.381e-03 / 0.67506   fine3 2.374e-04 / 0.65611
    noise 2.380e-06 / 0.15213   stripe_x 0.000e+00 / 0.00000 (its scramble stays row-constant)
    The pixel multiset is exact on all five; the spectrum is not, which is what the two claims in
    the pre-registration fighting each other looks like in numbers.
  Cost: about 90 ms per 256 px tile single-threaded, roughly 3 minutes for the 1946 materialised
    tiles. The pre-ruling module was 39 ms; A4 now walks four directions instead of two.
"""
import ast
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spatial_features as sf  # noqa: E402

FAIL = []
TOTAL = 0
PPM = 4.552516  # the canonical tile scale; every mm quantity below is derived from it
# The statistics with a registered undefined branch. A1 has none: it is computable for every tile
# that survives MIN_MASK_PIXELS, so a NaN A1 would be a bug rather than a degeneracy.
NAN_KEYS = ("A2", "A3", "A4")


def check(name, ok, detail=""):
    global TOTAL
    TOTAL += 1
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def caught(fn, *args, **kw):
    """The exception a call raises, or None. Checks assert on the TYPE, which is the whole point of
    the ruling's split: knowing that something raised does not tell TileDegenerate from
    ValueError."""
    try:
        fn(*args, **kw)
    except Exception as exc:
        return exc
    return None


def noisy(fn, *args, **kw):
    """Call fn with every warning recorded instead of raised, and keep what happened.

    Returns (result, exception_or_None, ["Category: message", ...]). The three-part return exists
    because a check that looks only at the warning list can pass while the call raises, so every
    caller below is asked to assert on the exception slot too - that omission was itself a review
    finding in fix round 2.

    It is used on the statistic path (to prove it is silent), on the aggregation path (to catch the
    ONE deliberate all-NaN notice, which is a warning by design), and on the new undefined-statistic
    probes, where raising TileDegenerate is the expected outcome and must not be confused with a
    number. It never suppresses anything: the records are handed back to the caller, and the
    runner's simplefilter("error") is still in force for every check that does not call this.
    """
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        res, err = None, None
        try:
            res = fn(*args, **kw)
        except Exception as exc:
            err = exc
    return res, err, [w.category.__name__ + ": " + str(w.message) for w in got]


# --------------------------------------------------------------------------- synthetic tiles
def striped(width=256, height=256, period=8):
    """A tile with stripes along x: strongly anisotropic, and orientation-flippable.

    The broadcast_to is not decoration: `128.0 + 40.0 * cos(...)` has shape (H, 1), and dstack of
    three (H, 1) arrays gives (H, 1, 3), on which np.gradient raises "Shape of array too small".
    Verified 2026-09-29 while checking this task before dispatch.
    """
    y = np.arange(height)[:, None]
    row = 128.0 + 40.0 * np.cos(2 * np.pi * (y % period) / period)
    g = np.broadcast_to(row, (height, width)).copy()
    return np.dstack([g, g, g])


def blobs(seed=0, size=24):
    rng = np.random.default_rng(seed)
    a = np.full((256, 256, 3), 120.0)
    for _ in range(30):
        cy, cx = rng.integers(0, 256, 2)
        yy, xx = np.ogrid[:256, :256]
        m = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (size / 2) ** 2
        a[m] += 60.0
    return a


def grain_noise(seed=3):
    """White pixel noise: no lateral correlation at any lag beyond one pixel."""
    rng = np.random.default_rng(seed)
    return rng.uniform(60.0, 200.0, (256, 256, 3))


def coarse_blocks(seed=4, block=16):
    """A 16 px mosaic: correlation length set by the block, not by the pixel."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(60.0, 200.0, (256 // block, 256 // block))
    g = np.kron(small, np.ones((block, block)))
    return np.dstack([g, g, g])


def fine_grains(seed=5, block=3):
    """The same mosaic recipe at 3 px: much shorter correlation length than block 16."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(60.0, 200.0, (256 // block, 256 // block))
    g = np.kron(small, np.ones((block, block)))
    return np.dstack([g, g, g])


def speck_on_dark(side=5):
    """Mask survives as one 5 px island: 25 pixels, so the tile is NOT degenerate, but no masked
    pair is 5 px apart in any direction and the variogram has no usable lag at all.

    This is the case the owner ruled on: A4 and A3 are undefined and must be NaN, while A1 and A2
    stay finite. Before the ruling the module answered both with well-formed numbers (0.0 for A4,
    127/128 for A3), which is exactly the fallback the ruling forbids.
    """
    a = np.full((256, 256, 3), 10.0)
    a[40:40 + side, 70:70 + side] = 230.0
    return a


def thin_band(rows=3, cols=40, at=(60, 90)):
    """Mask survives as a 3 x 40 island: 120 pixels, so a valid tile, but only 3 profile lines run
    along y, below MIN_PROFILE_LINES = 4, so A2 must be NaN. No masked pixel pair is 5 rows apart
    either, so A4 is NaN for its own reason, and the variogram runs out of pairs by lag 38 so A3 is
    NaN too. A1 is the only finite one.

    The pair of tiles (speck_on_dark, thin_band) is what makes the NaN rule testable per statistic
    rather than per tile: each has a different set of undefined keys, and the exact sets are
    asserted, not just "something is NaN".
    """
    a = np.full((256, 256, 3), 10.0)
    a[at[0]:at[0] + rows, at[1]:at[1] + cols] = 230.0
    return a


def island(side, at=(20, 20), shape=(64, 64)):
    """A tile whose mask is exactly side*side pixels, used to probe MIN_MASK_PIXELS from both sides.

    The default position is inside the default 64 x 64 shape on purpose: an out-of-range slice is
    silently empty in numpy, and a "16 pixel mask" tile that has no pixels in it tests nothing.
    """
    a = np.full(shape + (3,), 10.0)
    a[at[0]:at[0] + side, at[1]:at[1] + side] = 230.0
    return a


def ramp(width=256, height=256):
    """A monotone linear grey ramp: an exactly rank-1 gradient field.

    This is the boundary case for A1's eigenvalue floor. tr/2 - sqrt(disc) evaluates to zero (up
    to roundoff) when every gradient vector is parallel, so it is the only synthetic tile here that
    exercises max(..., 0.0) rather than sailing past it.
    """
    g = np.linspace(0.0, 255.0, width)[None, :]
    g = np.broadcast_to(g, (height, width)).copy()
    return np.dstack([g, g, g])


def value_table(tiles):
    """{name: statistic dict} for the mutation sanity checks, one tile_spatial call per tile."""
    return {name: sf.tile_spatial(a, PPM) for name, a in tiles.items()}


def spread(values, key):
    got = [v[key] for v in values.values()]
    return max(got) - min(got), got


def fmt(values, key):
    return "; ".join("%s=%.6f" % (n, v[key]) for n, v in values.items())


def finite(values, key):
    return {n: v for n, v in values.items() if np.isfinite(v[key])}


def has_nan(out):
    """The keys of a tile_spatial result that are NaN."""
    return tuple(k for k in sf.SPATIAL if np.isnan(out[k]))


# A1 was NOT touched by the rulings, and the spec cell for A3 was amended to match the code rather
# than the other way round. Their values on these four tiles are therefore bit-identical to the
# numbers the Task 1 run recorded (task-1-report.md section 2.2). Asserting exact equality is the
# cheapest guard against a refactor moving a statistic that was ruled out of scope; a tolerance
# would let a real change through under the banner of "unchanged".
#
# Fix round 2 checked that this pin still means what it says after A3's lag rule was made
# all-or-nothing: the four tiles here have NO one-direction lag at all (measured: 0 of 127, of 127,
# of 127 and of 127), so the tightened gate cannot move their values and the pin is testing
# "unchanged", not "unchanged within a tolerance the new rule slips through". The tiles the rule
# DOES move are probed directly in test_a3_lag_is_all_or_nothing_over_its_directions.
A1_A3_PRE_RULING = {
    "blobs": {"A1": -0.019575082103750408, "A3": 0.39843750000000006},
    "stripe_x": {"A1": -33.76125000296601, "A3": 0.015625},
    "noise": {"A1": -0.005142022827908911, "A3": 0.0078125},
    "coarse16": {"A1": -0.0869194997177046, "A3": 0.109375},
}
A1_A3_TILES = {"blobs": blobs, "stripe_x": striped, "noise": grain_noise,
               "coarse16": coarse_blocks}


def test_a1_and_a3_are_untouched_by_the_rulings():
    for name, maker in A1_A3_TILES.items():
        out = sf.tile_spatial(maker(), PPM)
        for key in ("A1", "A3"):
            want, got = A1_A3_PRE_RULING[name][key], out[key]
            check("%s %s bit-identical to the pre-ruling module" % (name, key), got == want,
                  "want %.17g got %.17g" % (want, got))


# Golden reference values, re-measured from the module AFTER the owner's rulings of 2026-09-29 and
# asserted at max(5% relative, 0.02 absolute), the same device preprocess/verify.py uses for the 12.
# The invariance and "can this statistic move" checks are blind to a change that moves all tiles
# together; this table is not. It exists because mutation testing in Task 1 found that replacing
# _acf_along's zero-fill of fully masked lines with neighbour interpolation escaped every other
# check. The A2 and A4 columns of the pre-ruling table (-0.911524 / 3.395993 style log ratios, and
# the 5 px diagonal) were DELETED, not relabelled: they pinned definitions the owner overruled.
# The A1 and A3 columns are unchanged, and test_a1_and_a3_are_untouched_by_the_rulings pins that.
GOLDEN = {
    "blobs": {"A1": -0.019575082103750408, "A2": 0.2745206920717868,
              "A3": 0.39843750000000006, "A4": -0.004123098760535949},
    "coarse16": {"A1": -0.0869194997177046, "A2": 0.4559072102201212,
                 "A3": 0.109375, "A4": -0.006562978878119824},
    "noise": {"A1": -0.005142022827908911, "A2": 1.0089893192060135,
              "A3": 0.0078125, "A4": -0.0010594892656631272},
    "stripe_x": {"A1": -33.76125000296601, "A2": 35.271944936420425,
                 "A3": 0.015625, "A4": -0.08333333333333331},
}
GOLDEN_TILES = {"blobs": blobs, "stripe_x": striped, "noise": grain_noise,
                "coarse16": coarse_blocks}


def golden_reference():
    """Re-measure the table from the module. Reached only by --golden, never by a normal run."""
    for name, maker in sorted(GOLDEN_TILES.items()):
        out = sf.tile_spatial(maker(), PPM)
        print('    "%s": {%s},' % (name, ", ".join('"%s": %r' % (k, out[k]) for k in sf.SPATIAL)))


def test_golden_values_reproduced():
    for name, maker in GOLDEN_TILES.items():
        out = sf.tile_spatial(maker(), PPM)
        if name not in GOLDEN:
            check("golden table has an entry for %s" % name, False,
                  "re-measure with --golden; an empty table is not a pass")
            continue
        for key in sf.SPATIAL:
            want, got = GOLDEN[name][key], out[key]
            tol = max(0.05 * abs(want), 0.02)
            check("%s %s within max(5%%, 0.02) of golden" % (name, key),
                  abs(got - want) <= tol, "want %.6f got %.6f tol %.6f" % (want, got, tol))


# --------------------------------------------------------------------------- ruled definitions
def ref_pair_slices(shape, oy, ox):
    """Independent transcription of "pairs of pixels at offset (oy, ox)", from the spec cell."""
    h, w = shape
    r0, r1 = (0, h - oy) if oy >= 0 else (-oy, h)
    c0, c1 = (0, w - ox) if ox >= 0 else (-ox, w)
    return (slice(r0, r1), slice(c0, c1)), (slice(r0 + oy, r1 + oy), slice(c0 + ox, c1 + ox))


def ref_direction_contrast(g, m, oy, ox, mad):
    """C(d) as the spec words it, plus the pair count. Returns (fraction or None, n_pairs)."""
    a, b = ref_pair_slices(g.shape, oy, ox)
    keep = m[a] & m[b]
    n = int(keep.count_nonzero()) if hasattr(keep, "count_nonzero") else int(np.count_nonzero(keep))
    if n < sf.MIN_PAIRS_PER_DIRECTION:
        return None, n
    d = np.abs(g[a][keep] - g[b][keep])
    return float(np.mean(d > mad)), n


def reference_a4(a, ppm):
    """omni - C(45 deg), omni over 0/45/90/135, 5 px axes and 3 px diagonals. None if undefined."""
    g, m, _ = sf.mask_and_field(a, ppm)
    lag_px = int(round(sf.A4_LAG_MM * ppm))
    diag = sf.A4_DIAG_PX
    mad = float(np.median(np.abs(g[m] - np.median(g[m])))) + sf.EPS
    offsets = {0: (0, lag_px), 90: (lag_px, 0), 45: (diag, diag), 135: (diag, -diag)}
    terms, counts = {}, {}
    for deg, (oy, ox) in offsets.items():
        c, n = ref_direction_contrast(g, m, oy, ox, mad)
        if c is None:
            return None, counts
        terms[deg], counts[deg] = c, n
    omni = float(np.mean([terms[0], terms[45], terms[90], terms[135]]))
    return omni - terms[45], terms


def reference_a2(a, ppm):
    """L(0 deg)/L(90 deg) with fully masked profile lines skipped, computed line by line in a loop.

    Deliberately a different code path from the module's vectorised masked sums, so agreement means
    the two transcriptions of the spec cell agree, not that one function was compared with itself.
    The 1/e interpolation is the module's own _efold, whose crossing and lag-grid rules are pinned
    separately by test_efold_crossing_is_the_registered_one_over_e. Under the owner's follow-up
    ruling on the A2 cell this reference returns None, not a number, when either direction has no
    measurable decay: fewer than MIN_PROFILE_LINES lines, or the _efold refusal (no crossing inside
    the evaluated lags, or a crossing below one full lag step).
    """
    g, m, e = sf.mask_and_field(a, ppm)
    lengths = []
    for axis in (1, 0):                      # axis 1 -> profile along x -> 0 deg, then 90 deg
        n = e.shape[axis]
        prof = np.full(n, np.nan)
        for i in range(n):
            line, ml = (e[:, i], m[:, i]) if axis == 1 else (e[i, :], m[i, :])
            v = line[ml]
            if v.size:
                prof[i] = float(v.mean())    # a line with no masked pixel is skipped, never 0.0
        keep = np.isfinite(prof)
        if int(np.count_nonzero(keep)) < sf.MIN_PROFILE_LINES:
            return None
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        den = float(np.dot(p, p)) + sf.EPS
        nlag = int(sf.ACF_MAX_LAG_FRACTION * n)
        ac = np.array([float(np.dot(p[:n - k], p[k:]) / den) for k in range(nlag)])
        try:
            lengths.append(sf._efold(ac))
        except sf.TileDegenerate:            # no measurable decay in this direction: no ratio
            return None
    return float((lengths[0] + sf.EPS) / (lengths[1] + sf.EPS))


# ------------------------------------------------------------ the rulings, one test each
def test_frozen_parameter_table_is_implemented():
    """Every row of the spec's frozen-parameter table is a named constant with the frozen value.

    The spec freezes these numbers so that "no parameter per statistic" is checkable. A literal
    buried in a function is not a frozen parameter: it cannot be audited without reading the code.
    A4_DIAG_PX is NOT in that table - it comes from ruling 4 and the amended A4 cell, and its
    absence from the table is reported to the owner rather than quietly filled.
    """
    want = {"A4_LAG_MM": 1.0, "A3_REFERENCE_LAG": 0.5, "A3_SILL_FRACTION": 0.9,
            "A3_SILL_WINDOW": 0.10, "MIN_MASK_PIXELS": 16, "MIN_PAIRS_PER_LAG": 8,
            "MIN_PAIRS_PER_DIRECTION": 8, "MIN_PROFILE_LINES": 4, "EPS": 1e-12,
            "ACF_MAX_LAG_FRACTION": 0.5}
    for name, value in sorted(want.items()):
        check("frozen table constant %s == %s" % (name, value),
              getattr(sf, name, None) == value, str(getattr(sf, name, "MISSING")))
    check("A4_DIAG_PX is the 3 px per-axis offset the owner ruled", sf.A4_DIAG_PX == 3,
          str(sf.A4_DIAG_PX))
    check("SPATIAL is the frozen A1-A4 tuple", sf.SPATIAL == ("A1", "A2", "A3", "A4"),
          str(sf.SPATIAL))
    # Finding 4: the old check here counted the token "EPS" in the source, which a docstring
    # satisfies and a correct refactor can fail. What EPS has to DO is pinned where it does it -
    # inside A1's log, inside A4's MAD, inside the sill floor and inside A2's ratio - in
    # test_named_constants_are_the_ones_the_statistics_read.


def src_text():
    return (Path(__file__).resolve().parent / "spatial_features.py").read_text(encoding="utf-8")


def test_exception_types_exist_and_are_disjoint():
    check("TileDegenerate is defined and subclasses Exception",
          isinstance(getattr(sf, "TileDegenerate", None), type)
          and issubclass(sf.TileDegenerate, Exception))
    check("TileDegenerate is not a ValueError subclass, so degeneracy cannot be logged as a bug",
          not issubclass(sf.TileDegenerate, ValueError))
    check("ValueError is not a TileDegenerate subclass, so a bug cannot be logged as degeneracy",
          not issubclass(ValueError, sf.TileDegenerate))
    check("TileDegenerate is not an AssertionError subclass (Task 3 must catch it by name)",
          not issubclass(sf.TileDegenerate, AssertionError))


def test_degenerate_mask_raises_tiledenerate():
    """Too few mask pixels is a WHOLE-TILE degeneracy, and it is the only degeneracy that raises."""
    exc = caught(sf.mask_and_field, np.full((64, 64, 3), 120.0), PPM)
    check("flat tile raises TileDegenerate on the degenerate mask",
          isinstance(exc, sf.TileDegenerate), type(exc).__name__)
    check("degenerate-mask message names the pixel count",
          "degenerate mask" in str(exc).lower() and "0" in str(exc), str(exc)[:72])
    exc2 = caught(sf.tile_spatial, np.full((64, 64, 3), 120.0), PPM)
    check("tile_spatial propagates the degenerate-mask raise", isinstance(exc2, sf.TileDegenerate),
          type(exc2).__name__)
    nine = island(3)                      # 9 pixels survive the percentile, below MIN_MASK_PIXELS
    exc3 = caught(sf.tile_spatial, nine, PPM)
    check("a 9 pixel mask raises TileDegenerate naming 9", isinstance(exc3, sf.TileDegenerate),
          str(exc3)[:72])
    exc4 = caught(sf.mask_and_field, island(4), PPM)   # exactly MIN_MASK_PIXELS = 16
    check("a mask of exactly MIN_MASK_PIXELS pixels is a tile, not a degeneracy", exc4 is None,
          str(exc4)[:72])
    exc5 = caught(sf.tile_spatial, island(4), PPM)
    check("a 16 pixel mask tile returns with NaN where it cannot measure",
          exc5 is None and isinstance(exc5, sf.TileDegenerate) is False, str(exc5)[:60])


def test_invalid_calls_raise_value_error():
    """Wrong shape, non-finite pixels, ppm <= 0 or NaN: an invalid call, never a degeneracy."""
    cases = [("2D tile", np.zeros((32, 32)), PPM, "HxWx3"),
             ("4D tile", np.zeros((8, 8, 3, 1)), PPM, "HxWx3"),
             ("2-channel tile", np.zeros((32, 32, 2)), PPM, "HxWx3"),
             ("ppm 0.0", striped(), 0.0, "ppm"),
             ("ppm -1.0", striped(), -1.0, "ppm"),
             ("ppm nan", striped(), float("nan"), "ppm"),
             ("ppm inf", striped(), float("inf"), "ppm")]
    for name, arr, ppm, needle in cases:
        exc = caught(sf.mask_and_field, arr, ppm)
        check("%s raises ValueError" % name, isinstance(exc, ValueError), type(exc).__name__)
        check("%s message names %s" % (name, needle), needle.lower() in str(exc).lower(),
              str(exc)[:60])
        check("%s is not catchable as TileDegenerate" % name,
              not isinstance(exc, sf.TileDegenerate))
    nan_tile = blobs().copy()
    nan_tile[7, 9, 1] = np.nan
    exc = caught(sf.tile_spatial, nan_tile, PPM)
    check("a NaN pixel raises ValueError, not TileDegenerate",
          isinstance(exc, ValueError) and not isinstance(exc, sf.TileDegenerate),
          "%s | %s" % (type(exc).__name__, str(exc)[:60]))
    inf_tile = blobs().copy()
    inf_tile[3, 3, 0] = np.inf
    exc = caught(sf.tile_spatial, inf_tile, PPM)
    check("an infinite pixel raises ValueError, not TileDegenerate",
          isinstance(exc, ValueError) and not isinstance(exc, sf.TileDegenerate),
          type(exc).__name__)
    exc = caught(sf.mask_and_field, striped(), None)
    check("missing ppm (None) raises ValueError", isinstance(exc, ValueError),
          type(exc).__name__)


def test_undefined_statistic_is_nan_and_does_not_raise():
    """Rulings 2, 3, 5: an undefined statistic is NaN within an otherwise valid tile.

    Two probes with DIFFERENT undefined sets, so the rule is tested per statistic and not per tile.
    speck_on_dark: a 5 x 5 island, 25 mask pixels, a valid tile; lag 1 has pairs but lag 5 has none
    and the variogram's trailing window is empty -> A3 and A4 NaN, A1 and A2 finite.
    thin_band: a 3 x 40 island, 120 mask pixels, only 3 valid y lines and no pair 5 rows apart ->
    A2, A3 and A4 NaN, A1 finite.
    """
    out, err, got = noisy(sf.tile_spatial, speck_on_dark(), PPM)
    check("speck tile: tile_spatial returns instead of raising", err is None, str(err)[:60])
    check("speck tile: no warning while computing", not got, "; ".join(got)[:70])
    check("speck tile: the NaN keys are exactly A3 and A4", has_nan(out) == ("A3", "A4"),
          str(has_nan(out)))
    check("speck tile: A3 is NaN, not the max-lag fallback 127/128", np.isnan(out["A3"]),
          repr(out["A3"]))
    check("speck tile: A4 is NaN, not a 0.0 exceedance fraction", np.isnan(out["A4"]),
          repr(out["A4"]))
    check("speck tile: A1 and A2 are finite", np.isfinite(out["A1"]) and np.isfinite(out["A2"]),
          "A1=%.6f A2=%.6f" % (out["A1"], out["A2"]))
    check("speck tile: no old fallback number survives",
          not any(out[k] == 0.0 for k in NAN_KEYS), str(out))

    out2, err2, got2 = noisy(sf.tile_spatial, thin_band(), PPM)
    check("thin band: tile_spatial returns instead of raising", err2 is None, str(err2)[:60])
    check("thin band: no warning while computing", not got2, "; ".join(got2)[:70])
    check("thin band: A2 is NaN (3 valid lines < MIN_PROFILE_LINES)", np.isnan(out2["A2"]),
          repr(out2["A2"]))
    check("thin band: the NaN keys are exactly A2, A3 and A4",
          has_nan(out2) == ("A2", "A3", "A4"), str(has_nan(out2)))
    check("thin band: A1 is finite", np.isfinite(out2["A1"]), "%.6f" % out2["A1"])
    # The NaN must come from a real degeneracy, not from a silent zero-length guard: the direction
    # itself raises, and tile_spatial is what converts it to NaN.
    g, m, e = sf.mask_and_field(thin_band(), PPM)
    exc = caught(sf._acf_along, e, m, 0)
    check("thin band: the 90 deg direction raises TileDegenerate internally",
          isinstance(exc, sf.TileDegenerate), "%s | %s" % (type(exc).__name__, str(exc)[:60]))
    exc = caught(sf._acf_along, e, m, 1)
    check("thin band: the 0 deg direction still measures (40 valid lines)", exc is None,
          str(exc)[:60])
    g2, m2, _ = sf.mask_and_field(speck_on_dark(), PPM)
    mad = float(np.median(np.abs(g2[m2] - np.median(g2[m2])))) + sf.EPS
    exc = caught(sf._direction_contrast, g2, m2, 0, int(round(sf.A4_LAG_MM * PPM)), mad)
    check("speck: a direction with too few pairs raises TileDegenerate",
          isinstance(exc, sf.TileDegenerate), "%s | %s" % (type(exc).__name__, str(exc)[:60]))
    exc = caught(sf._direction_contrast, g2, m2, 0, 1, mad)
    check("speck: a lag that does have pairs is measured, not refused", exc is None, str(exc)[:60])
    exc = caught(sf._variogram_range, g2, m2, PPM)
    check("speck: a variogram with no measurable sill raises TileDegenerate",
          isinstance(exc, sf.TileDegenerate), "%s | %s" % (type(exc).__name__, str(exc)[:60]))


def test_a2_is_a_plain_ratio():
    """Ruling 2. A log-ratio is negative when L(0) < L(90); a plain ratio is below 1, and the two
    are not interchangeable, so the sign form of the old test had to be re-derived."""
    horiz = sf.tile_spatial(striped(), PPM)["A2"]
    vert = sf.tile_spatial(np.transpose(striped(), (1, 0, 2)).copy(), PPM)["A2"]
    check("A2 on stripes along x is greater than 1 (0 deg decorrelates more slowly)", horiz > 1.0,
          "%.6f" % horiz)
    check("A2 on stripes along y is between 0 and 1 (90 deg decorrelates more slowly)",
          0.0 < vert < 1.0, "%.6f" % vert)
    check("A2 inverts under transpose: the product is 1, not the sum being 0",
          abs(horiz * vert - 1.0) < 1e-9, "%.12f" % (horiz * vert))
    values = value_table({"blobs": blobs(), "noise": grain_noise(), "coarse16": coarse_blocks(),
                          "fine3": fine_grains(), "ramp": ramp(), "stripe_x": striped()})
    a2 = [v["A2"] for v in values.values() if np.isfinite(v["A2"])]
    check("A2 is strictly positive on every measurable tile (a ratio, never a signed log)",
          all(x > 0.0 for x in a2), " ".join("%.6f" % x for x in a2))
    check("A2's reciprocal is not its own negative: the log form is gone",
          abs(horiz + vert) > 1.0, "%.6f + %.6f = %.6f" % (horiz, vert, horiz + vert))
    for name in ("blobs", "coarse16"):
        want = reference_a2(A1_A3_TILES[name](), PPM)
        got = values[name]["A2"]
        check("A2 on %s matches the independent line-by-line reference" % name,
              want is not None and abs(np.log(got / want)) < 1e-9,
              "ref=%s module=%.9f" % ("None" if want is None else "%.9f" % want, got))


def test_a2_skips_masked_out_lines_instead_of_writing_zero():
    """Ruling 2, the fill rule. A whole row or column outside the mask is not a grey level of 0."""
    for name, a in (("striped", striped()), ("blobs", blobs()), ("ramp", ramp()),
                    ("noise", grain_noise())):
        g, m, e = sf.mask_and_field(a, PPM)
        for axis, deg in ((1, 0), (0, 90)):
            prof, lines = sf._masked_profile(e, m, axis)
            empty = m.sum(axis=1 - axis) == 0          # the lines with no masked pixel at all
            check("%s %d deg: every empty line is NaN, never 0.0" % (name, deg),
                  int(np.count_nonzero(empty)) == int(np.count_nonzero(~np.isfinite(prof)))
                  and not np.any(prof[empty] == 0.0),
                  "empty lines=%d NaN entries=%d" % (int(np.count_nonzero(empty)),
                                                     int(np.count_nonzero(~np.isfinite(prof)))))
            check("%s %d deg: the valid line count is the number of finite profile entries"
                  % (name, deg), lines == int(np.count_nonzero(np.isfinite(prof))),
                  "lines=%d finite entries=%d" % (lines, int(np.count_nonzero(np.isfinite(prof)))))
    # The magnitude the Task 1 report measured: swapping the rule moves A2 on blobs by 0.334 log
    # units, so this is not a cosmetic fix. With the skip rule the striped tile's 32 empty rows and
    # blobs' 5 empty columns are gone from the profile rather than folded into it.
    out, err, got_warnings = noisy(sf.tile_spatial, striped(), PPM)
    check("A2 on the 32-empty-row striped tile emits no warning", not got_warnings,
          "; ".join(got_warnings)[:70])
    check("A2 on the 32-empty-row striped tile does not raise", err is None, str(err)[:60])
    out, err, got_warnings = noisy(sf.tile_spatial, blobs(), PPM)
    check("A2 on the 5-empty-column blobs tile emits no warning", not got_warnings,
          "; ".join(got_warnings)[:70])


# -------------------------------- AMENDMENT 1, A2 cell: the owner's follow-up ruling (option a)
#
# A2 is the PLAIN ratio L(0 deg)/L(90 deg) and the ruling requires a measurable decay in BOTH
# directions. A direction that produced no measurable decay undefineds A2, exactly as too few
# profile lines and a zero-variance profile already do. The two new undefined cases are read off
# the registered 1/e definition and off the integer lag grid the profile is sampled on; the ruling
# adds NO numeric energy or magnitude floor of any kind, and the existing structural
# "energy == 0.0 or not finite" guard in _acf_along is untouched:
#   * an ACF that never falls below 1/e inside the evaluated lags. Before the ruling _efold returned
#     the maximum lag for it, which is the range-like stand-in the owner ruled against for A3's
#     unmet crossing ("no code path can answer an unmet crossing with the maximum lag");
#   * a 1/e crossing that lands at lag 0, i.e. below one full lag step. Before the ruling _efold
#     returned a tiny positive length, and the plain ratio turned it into a 1e13 or 1e-14
#     "anisotropy" on tiles that had measured nothing at all in that direction.


def grey_ramp(side=64, along="x"):
    """A monotone 0..255 grey ramp, exactly constant along the other axis.

    along="x" is the owner's row-constant case: every column of the tile is a copy of every other,
    so the 90 deg profile is flat to roundoff (measured centred energy 3.7e-28, four orders below
    the EPS that normalises it, so the EXACTLY-flat guard does not fire) and its 1/e crossing lands
    at lag 0.0, while the 0 deg profile measures normally at lag 12.17. along="y" is its transpose.
    """
    v = np.linspace(0.0, 255.0, side)
    g = (np.broadcast_to(v[None, :], (side, side)) if along == "x"
         else np.broadcast_to(v[:, None], (side, side)))
    g = g.copy()
    return np.dstack([g, g, g])


def checkerboard(side=60, lo=0.0, hi=255.0):
    """A one-pixel checkerboard: the 12th-percentile mask keeps one colour, so profiles alternate.

    Both directions cross 1/e at lag 0.322, inside the first lag step, and the same tile is already
    NaN on A3 and A4. That combination is what made a finite A2 of 1.0000000000036 indefensible.
    """
    y, x = np.mgrid[0:side, 0:side]
    g = np.where((x + y) % 2 == 0, hi, lo).astype(float)
    return np.dstack([g, g, g])


def long_range_ramp(side=5):
    """A small monotone diagonal ramp: a correlation length longer than the profile reading it.

    ACF_MAX_LAG_FRACTION evaluates half the profile, so a 5 px tile gives lags 0 and 1 only, and the
    autocorrelation does not fall below 1/e there (measured minimum 0.38965 against 1/e = 0.36788).
    This is the never-crosses case: before the ruling both directions answered the maximum lag 2.0
    and the ratio answered exactly 1.0, the number a genuinely isotropic tile also returns.
    """
    g = np.add.outer(np.arange(side, dtype=float), np.arange(side, dtype=float))
    return np.dstack([g, g, g])


def efold_pre_ruling(ac):
    """_efold AS IT STOOD before this ruling, transcribed here and not imported from the module.

    Every probe below states its own undefined condition through this function (a sub-lag-step
    crossing, or no crossing at all inside the evaluated lags) and then asserts that the module no
    longer reports the number it produced. A test that only asserted "NaN" could be satisfied by a
    rule that NaNs everything; these cannot.
    """
    thr = 1.0 / np.e
    for k in range(1, len(ac)):
        if ac[k] < thr:
            span = ac[k - 1] - ac[k]
            frac = (ac[k - 1] - thr) / span if span > sf.EPS else 0.0
            return (k - 1) + float(frac)
    return float(len(ac))


def a2_acfs(a, ppm=PPM):
    """(g, m, e) and the two directional ACFs, in the registered 0 deg then 90 deg order."""
    g, m, e = sf.mask_and_field(a, ppm)
    return (g, m, e), [sf._acf_along(e, m, 1), sf._acf_along(e, m, 0)]


# Measured 2026-09-29 through the PUBLIC API (tile_spatial) at ppm 4.552516, against the module as
# it stood BEFORE the follow-up ruling. The A2 column is the answer that must no longer be returned;
# the other three columns are the answers that must NOT move. The tuple at the end is the
# directional (0 deg, 90 deg) lengths that pre-ruling _efold produced, i.e. the diagnosis.
A2_PROBES = {
    "row-constant grey ramp": (lambda: grey_ramp(along="x"), 1.2166e13, 12166331134739.916,
                              {"A1": -30.427278753462396, "A3": 0.90625, "A4": 0.0},
                              (12.166331134738916, 0.0), "lag-0"),
    "col-constant grey ramp": (lambda: grey_ramp(along="y"), 8.2194e-14, 8.219404756661486e-14,
                              {"A1": -30.427278753462396, "A3": 0.90625, "A4": 0.0},
                              (0.0, 12.166331134738916), "lag-0"),
    "pure checkerboard": (lambda: checkerboard(), 1.0000000000036, 1.0000000000035931,
                          {"A1": -0.06669137449867216, "A3": float("nan"), "A4": float("nan")},
                          (0.3224394240255171, 0.32243942402435855), "lag-0"),
    "correlation longer than the tile": (lambda: long_range_ramp(), 1.0, 1.0,
                                         {"A1": -28.324168296488992, "A3": 0.4,
                                          "A4": float("nan")},
                                         (2.0, 2.0), "never-crosses"),
}
# The same table for tiles whose decay IS measurable in both directions. These are the controls that
# stop the ruling from being satisfiable by turning every A2 into NaN.
A2_MEASURABLE = {
    "stripe_x": (striped, 35.271944936420425,
                 {"A1": -33.76125000296601, "A3": 0.015625, "A4": -0.08333333333333331},
                 (47.54757292505386, 1.3480281002572057)),
    "blobs": (blobs, 0.2745206920717868,
              {"A1": -0.019575082103750408, "A3": 0.39843750000000006,
               "A4": -0.004123098760535949},
              (11.440665665740488, 41.67505764100833)),
}


def test_a2_never_crossing_acf_is_nan_not_the_maximum_lag():
    """Ruling (a), first condition: an ACF that never falls below 1/e has NO e-folding length."""
    name = "correlation longer than the tile"
    check("the never-crosses branch has a tile probe of its own",
          [n for n, p in A2_PROBES.items() if p[5] == "never-crosses"] == [name],
          str([n for n, p in A2_PROBES.items() if p[5] == "never-crosses"]))
    a = A2_PROBES[name][0]()
    (g, m, e), acs = a2_acfs(a)
    thr = 1.0 / np.e
    for deg, ac in ((0, acs[0]), (90, acs[1])):
        check("never-crossing probe %d deg: the ACF stays at or above 1/e on every evaluated lag"
              % deg, float(np.min(ac[1:])) >= thr,
              "min(ac[1:]) = %.9f vs 1/e = %.9f over %d lags" % (float(np.min(ac[1:])), thr,
                                                                 ac.size))
        exc, err, got = noisy(sf._efold, ac)
        check("never-crossing probe %d deg: _efold raises TileDegenerate, it does not return a "
              "length" % deg, isinstance(err, sf.TileDegenerate) and exc is None,
              "%s | %s" % (type(err).__name__, str(err)[:60]))
        check("never-crossing probe %d deg: the refusal is not an invalid call" % deg,
              not isinstance(err, ValueError), type(err).__name__)
        check("never-crossing probe %d deg: the refusal emits no warning" % deg, not got,
              "; ".join(got)[:70])
    lengths = [efold_pre_ruling(ac) for ac in acs]
    check("never-crossing probe: the pre-ruling answer WAS the maximum lag for both directions",
          lengths == [float(acs[0].size), float(acs[1].size)],
          "%s of %d lags" % (lengths, acs[0].size))
    out, err, got = noisy(sf.tile_spatial, a, PPM)
    check("never-crossing probe: tile_spatial returns instead of raising", err is None,
          str(err)[:60])
    check("never-crossing probe: no warning while A2 goes undefined", not got, "; ".join(got)[:70])
    check("never-crossing probe: A2 is NaN", np.isnan(out["A2"]), repr(out["A2"]))
    check("never-crossing probe: A2 is not the maximum-lag ratio it used to be",
          out["A2"] != 1.0, "pre-ruling A2 was 1.0 = %r / %r" % tuple(lengths))
    check("never-crossing probe: the NaN keys are exactly A2 and A4",
          has_nan(out) == ("A2", "A4"), str(has_nan(out)))


def test_a2_lag_zero_crossing_is_nan_not_a_tiny_positive_length():
    """Ruling (a), second condition: a crossing inside the first lag step is unresolvable.

    The direction that fails is refused and the direction that measures is still measured, so the
    guard is per direction and not a blanket refusal: on the row-constant ramp the 0 deg length
    stays 12.166 samples while the 90 deg length, which the pre-ruling code reported as 0.0, is the
    one that undefineds the ratio.
    """
    for name, deg in (("row-constant grey ramp", 90), ("col-constant grey ramp", 0),
                      ("pure checkerboard", 0), ("pure checkerboard", 90)):
        i = 0 if deg == 0 else 1
        a = A2_PROBES[name][0]()
        (g, m, e), acs = a2_acfs(a)
        pre = [efold_pre_ruling(ac) for ac in acs]
        check("%s: the %d deg crossing lands below one full lag step (%.6g)"
              % (name, deg, pre[i]), 0.0 <= pre[i] < 1.0, "pre-ruling length %r" % pre[i])
        exc, err, got = noisy(sf._efold, acs[i])
        check("%s: the %d deg direction raises TileDegenerate instead of reporting a length"
              % (name, deg), isinstance(err, sf.TileDegenerate) and exc is None,
              "%s | %s" % (type(err).__name__, str(err)[:60]))
        check("%s: and it is a degeneracy, not a swallowed ValueError" % name,
              not isinstance(err, ValueError), type(err).__name__)
        check("%s: the %d deg refusal emits no warning" % (name, deg), not got,
              "; ".join(got)[:70])
        other = 1 - i
        if pre[other] >= 1.0:
            kept = sf._efold(acs[other])
            check("%s: the OTHER direction is still measured at %.6f samples"
                  % (name, pre[other]), abs(kept - pre[other]) < 1e-12,
                  "got %.12f want %.12f" % (kept, pre[other]))
        out, err2, got2 = noisy(sf.tile_spatial, a, PPM)
        check("%s: tile_spatial returns and A2 is NaN" % name,
              err2 is None and np.isnan(out["A2"]), "%s | %r" % (str(err2)[:40], out["A2"]))
        check("%s: A2 undefined costs no warning" % name, not got2, "; ".join(got2)[:70])


def test_a2_measurable_controls_keep_a_finite_number():
    """The other side of the ruling: a resolvable decay in both directions still measures.

    Without these two the fix could pass by making A2 NaN everywhere. Both directional lengths are
    pinned as numbers, the striped tile's 90 deg length (1.348 samples) is the closest any probe
    here sits above the one-lag-step floor, and A2 is bit-identical to its pre-ruling value.
    """
    for name, (maker, want_a2, others, lengths) in A2_MEASURABLE.items():
        a = maker()
        (g, m, e), acs = a2_acfs(a)
        check("%s: both directions cross 1/e at or after one full lag step" % name,
              min(lengths) >= 1.0, "lengths %s, shortest %.6f" % (lengths, min(lengths)))
        kept = [sf._efold(ac) for ac in acs]
        check("%s: _efold still returns both lengths" % name,
              all(abs(kept[i] - lengths[i]) < 1e-12 for i in (0, 1)),
              "got %s want %s" % (kept, list(lengths)))
        out, err, got = noisy(sf.tile_spatial, a, PPM)
        check("%s: A2 is finite, with no raise and no warning" % name,
              err is None and not got and np.isfinite(out["A2"]),
              "%s | %r" % (str(err)[:40], out["A2"]))
        check("%s: A2 is the same number as before the ruling" % name, out["A2"] == want_a2,
              "want %.17g got %.17g" % (want_a2, out["A2"]))
        check("%s: A1, A3 and A4 did not move" % name,
              all(out[k] == v for k, v in others.items()), str(out))


def test_a2_unmeasurable_tiles_no_longer_return_their_numbers():
    """The three tiles the owner measured, plus the never-crosses probe: all four go NaN.

    The pre-fix values are asserted as the thing that must NO LONGER come back, and A1, A3 and A4
    are pinned bit-for-bit, so the fix cannot be paid for by moving a statistic that was ruled out
    of scope - including the checkerboard's A3 and A4 NaNs, which stay NaN for their own reasons.
    """
    for name, (maker, quoted, pre_fix, others, lengths, why) in A2_PROBES.items():
        out, err, got = noisy(sf.tile_spatial, maker(), PPM)
        check("%s: tile_spatial still returns" % name, err is None, str(err)[:60])
        check("%s: computing A2 emits no warning" % name, not got, "; ".join(got)[:70])
        check("%s: A2 is NaN (%s case)" % (name, why), np.isnan(out["A2"]), repr(out["A2"]))
        check("%s: A2 is not the pre-ruling %.17g" % (name, pre_fix), out["A2"] != pre_fix,
              "quoted by the owner as %.17g" % quoted)
        for key, want in sorted(others.items()):
            got_v = out[key]
            check("%s: %s unchanged at %.17g" % (name, key, want),
                  got_v == want or (np.isnan(got_v) and np.isnan(want)),
                  "want %.17g got %.17g" % (want, got_v))
        first = sf.tile_spatial(maker(), PPM)
        check("%s: two calls agree bit-for-bit, NaN keys included" % name,
              all(first[k] == out[k] or (np.isnan(first[k]) and np.isnan(out[k]))
                  for k in sf.SPATIAL) and has_nan(first) == has_nan(out),
              "%s vs %s" % (first, out))
        check("%s: the ratio of the pre-ruling lengths IS the pre-ruling A2" % name,
              abs((lengths[0] + sf.EPS) / (lengths[1] + sf.EPS) - pre_fix) <= 1e-9 * abs(pre_fix),
              "%.17g vs %.17g" % ((lengths[0] + sf.EPS) / (lengths[1] + sf.EPS), pre_fix))
        ref = noisy(reference_a2, maker(), PPM)
        check("%s: the independent line-by-line reference calls A2 undefined too" % name,
              ref[1] is None and ref[0] is None and not ref[2],
              "%s | %s" % (repr(ref[0]), str(ref[1])[:40]))


def test_a4_is_omni_minus_45_with_the_3px_diagonal():
    """Ruling 4: omni over 0/45/90/135 deg, 5 px axes, 3 px per-axis diagonals."""
    for name, maker in (("blobs", blobs), ("stripe_x", striped), ("noise", grain_noise),
                        ("coarse16", coarse_blocks)):
        a = maker()
        want, terms = reference_a4(a, PPM)
        got = sf.tile_spatial(a, PPM)["A4"]
        check("A4 on %s matches the independent reference" % name,
              want is not None and abs(got - want) < 1e-12,
              "ref=%.12f module=%.12f" % (float("nan") if want is None else want, got))
        if name == "blobs":
            print("INFO   A4 terms | " + " ".join("%d deg=%.6f" % (d, c)
                                                  for d, c in sorted(terms.items())))
            check("A4's four direction terms are all present and distinct from a 0 deg only omni",
                  len(terms) == 4 and abs(terms[0] - terms[90]) > 1e-9,
                  "C0=%.6f C90=%.6f" % (terms[0], terms[90]))
            omni = float(np.mean([terms[0], terms[45], terms[90], terms[135]]))
            check("A4 equals omni minus C(45 deg)", abs(got - (omni - terms[45])) < 1e-12,
                  "%.12f" % got)
            check("A4 is not omni minus C(0 deg), which the pre-ruling code effectively computed",
                  abs(got - (omni - terms[0])) > 1e-9,
                  "minus C0 would be %.12f" % (omni - terms[0]))


def test_a4_diagonal_offset_is_3px_not_5px():
    """The pre-ruling code put the diagonal pair at 1.553 mm against a 1.098 mm axis lag. If that
    ever comes back, every other A4 check could still pass by coincidence, so it is pinned here by
    re-deriving A4 with a 5 px diagonal and asserting the module does not agree with it."""
    a = blobs()
    g, m, _ = sf.mask_and_field(a, PPM)
    lag_px = int(round(sf.A4_LAG_MM * PPM))
    mad = float(np.median(np.abs(g[m] - np.median(g[m])))) + sf.EPS
    def with_diag(d):
        terms = {}
        for deg, (oy, ox) in {0: (0, lag_px), 90: (lag_px, 0), 45: (d, d), 135: (d, -d)}.items():
            c, n = ref_direction_contrast(g, m, oy, ox, mad)
            if c is None:
                return None
            terms[deg] = c
        return float(np.mean([terms[0], terms[45], terms[90], terms[135]]) - terms[45])
    got = sf.tile_spatial(a, PPM)["A4"]
    v3, v5 = with_diag(3), with_diag(5)
    check("A4 equals the 3 px diagonal construction", v3 is not None and abs(got - v3) < 1e-12,
          "3px=%.12f module=%.12f" % (float("nan") if v3 is None else v3, got))
    check("A4 differs from the 5 px diagonal construction it replaced",
          v5 is not None and abs(got - v5) > 1e-6,
          "5px=%.12f 3px=%.12f" % (float("nan") if v5 is None else v5, v3))
    # The arithmetic behind the ruling, recorded rather than assumed. The registered lag is
    # A4_LAG_MM = 1.0 mm, which at 4.552516 px/mm is 4.5525 px, so the exact diagonal per-axis
    # offset would be 1.0 * ppm / sqrt(2) = 3.219 px and the closest integer is 3. At 4 px the
    # diagonal pair sits at 1.2426 mm, 0.243 mm past the registered lag instead of 0.068 mm short
    # of it. The residual mismatch that is left is between the INTEGER axis lag, which is
    # round(4.5525) = 5 px = 1.0983 mm, and the diagonal (4.2426 px = 0.9320 mm): 0.166 mm. The
    # spec states it rather than smoothing it over, and so does this check.
    exact = sf.A4_LAG_MM * PPM / np.sqrt(2.0)
    d3, d4 = 3 * np.sqrt(2.0) / PPM, 4 * np.sqrt(2.0) / PPM
    check("3 px per axis is the closest integer to the exact diagonal offset (%.4f px)" % exact,
          abs(exact - 3) < abs(exact - 4), "exact=%.4f d3=%.4f d4=%.4f" % (exact, 3.0, 4.0))
    check("the 3 px diagonal pair sits at %.4f mm, nearer the registered 1.0 mm lag than 4 px" % d3,
          abs(d3 - sf.A4_LAG_MM) < abs(d4 - sf.A4_LAG_MM),
          "3px=%.4f mm off %.4f, 4px=%.4f mm off %.4f"
          % (d3, abs(d3 - 1.0), d4, abs(d4 - 1.0)))
    axis_mm = int(round(sf.A4_LAG_MM * PPM)) / PPM
    check("the axis/diagonal residual mismatch is 0.166 mm and is stated, not hidden",
          abs((axis_mm - d3) - 0.1663) < 5e-4, "axis=%.4f mm diag=%.4f mm mismatch=%.4f mm"
          % (axis_mm, d3, axis_mm - d3))


def test_a3_denominator_is_half_the_tile_extent():
    """Ruling 3: the sill locates the range and is NOT the denominator. A range/sill form would move
    A3 off the exact lag-count ratio the owner chose to keep, so it is asserted as a value."""
    for name, maker in A1_A3_TILES.items():
        out = sf.tile_spatial(maker(), PPM)
        a3 = out["A3"]
        check("%s: A3 is a multiple of 1/(0.5*256) = 0.0078125, so it is a lag over half the extent"
              % name, abs(a3 / 0.0078125 - round(a3 / 0.0078125)) < 1e-9, "%.9f" % a3)
    g, m, _ = sf.mask_and_field(blobs(), PPM)
    range_mm = sf._variogram_range(g, m, PPM)
    check("A3 = range_mm / (A3_REFERENCE_LAG * extent_mm) exactly",
          abs(sf.tile_spatial(blobs(), PPM)["A3"]
              - range_mm / (sf.A3_REFERENCE_LAG * 256 / PPM)) < 1e-12,
          "range_mm=%.6f" % range_mm)
    check("A3's range is in mm and its denominator is in mm, so ppm cancels",
          all(sf.tile_spatial(blobs(), p)["A3"] == sf.tile_spatial(blobs(), PPM)["A3"]
              for p in (2.276258, 4.552516, 9.105032)))


def test_a3_lags_are_not_carried_forward():
    """A lag with too few pairs is NaN, not the previous lag repeated. The band tile is the probe:
    its 3-row mask runs out of vertical pairs at lag 3, so under the all-or-nothing rule only lags 1
    and 2 are measurable and A3 must be NaN rather than a sill built on one-direction values."""
    g, m = sf.mask_and_field(blobs(), PPM)[:2]
    gam = sf._variogram(g, m)
    check("a healthy tile's variogram has no NaN lag", not np.any(np.isnan(gam)),
          "%d NaN lags of %d" % (int(np.count_nonzero(np.isnan(gam))), gam.size))
    g2, m2 = sf.mask_and_field(thin_band(), PPM)[:2]
    gam2 = sf._variogram(g2, m2)
    first_nan = int(np.argmax(np.isnan(gam2)))
    check("the band tile's lags 1 and 2 are measured over both directions",
          np.isfinite(gam2[0]) and np.isfinite(gam2[1])
          and all(min(lag_direction_counts(m2, k)) >= sf.MIN_PAIRS_PER_LAG for k in (1, 2)),
          "counts at lag 3: %s" % (lag_direction_counts(m2, 3),))
    check("the band tile's lag 3 is NaN because one direction has run out of pairs",
          np.isnan(gam2[2]) and lag_direction_counts(m2, 3)[0] < sf.MIN_PAIRS_PER_LAG,
          "counts at lag 3: %s" % (lag_direction_counts(m2, 3),))
    check("the band tile's late lags are NaN, not copies of the last measured lag",
          np.all(np.isnan(gam2[first_nan:])), "NaN from index %d to the end" % first_nan)
    check("a carried-forward sill would have produced a number where A3 is now NaN",
          np.isnan(sf.tile_spatial(thin_band(), PPM)["A3"]))
    check("the speck tile's A3 is NaN rather than the 127/128 maximum-lag range",
          np.isnan(sf.tile_spatial(speck_on_dark(), PPM)["A3"]))
    # Pinned as STRUCTURE, not as behaviour, and the reason is a proof rather than a gap: the sill
    # is the mean of the trailing window, whose largest lag is >= sill > 0.9 * sill, which means
    # `reached` can never be empty when the sill exists. No tile can reach the guard that stops a
    # maximum-lag range, so deleting it would be invisible to every value check - mutation testing
    # confirmed exactly that (disabling the guard leaves every check in this file green). The pin is
    # the honest device for a load-bearing branch no input can exercise. Fix round 2 changed its
    # form from a whitespace-exact line to the syntax tree, so a correct refactor of the same guard
    # (reformatting, a renamed local) keeps the pin and a deleted guard does not; the proof above is
    # why the structural form is kept at all rather than downgraded to a behavioural test.
    check("no code path can answer an unmet crossing with the maximum lag",
          crossing_guard_is_structural("_variogram_range"),
          "one return path, and the unmet-crossing raise still tests `reached`")


def crossing_guard_is_structural(fn_name):
    """The guard against a maximum-lag range, read as syntax rather than as a formatted string.

    Two conditions, both of which the old verbatim-line pin was standing in for:
      * the function has exactly one `return`, so there is no second path that could answer the
        unmet crossing with anything;
      * one of its `if` tests is a test on the crossing mask by name (`reached`), so the branch that
        fires when no lag reaches the sill is still the crossing branch and not a renamed accident.
    """
    tree = ast.parse((Path(__file__).resolve().parent / "spatial_features.py")
                     .read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        return False
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None]
    guards = [ast.dump(n.test) for n in ast.walk(fn) if isinstance(n, ast.If)]
    tested = any("reached" in g and "any" in g for g in guards)
    return len(returns) == 1 and tested


# ---------------------------------------------------------------- fix round 2: review findings 1-3
def lag_direction_counts(m, k):
    """Masked-pair counts at lag k for each direction, counted here rather than by the module.

    A test that only knows a lag came back NaN cannot say WHICH direction was short; this can, and
    it is the independent transcription of "below this a variogram lag is undefined" that finding 1
    is about.
    """
    out = []
    for ax in (0, 1):
        sl_a = [slice(None)] * 2
        sl_a[ax] = slice(None, -k)
        sl_b = [slice(None)] * 2
        sl_b[ax] = slice(k, None)
        out.append(int(np.count_nonzero(m[sl_a[0], sl_a[1]] & m[sl_b[0], sl_b[1]])))
    return out


def mask_tile(shape, pixels):
    """A tile whose mask_and_field mask is exactly `pixels`; values vary along BOTH axes.

    The per-pixel value term is not decoration: an island of one flat grey has zero-variance A2
    profiles, so it probes finding 2's zero-energy rule instead of the threshold a test is gating.
    Each caller asserts the achieved mask, so a design that silently misses its target fails loudly.
    """
    h, w = shape
    a = np.full((h, w, 3), 10.0)
    for n, (i, j) in enumerate(pixels):
        a[i, j] = 200.0 + 0.5 * j + 3.0 * i + 0.01 * n
    return a


def profile_energy(e, m, axis):
    """(valid line count, profile variance) for one A2 direction, computed independently."""
    prof, lines = sf._masked_profile(e, m, axis)
    keep = np.isfinite(prof)
    p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
    return lines, float(np.dot(p, p))


def flat_band(rows, side=64, at=20):
    """A band of `rows` whole rows at one grey level: an exactly zero-variance A2 profile."""
    a = np.full((side, side, 3), 10.0)
    a[at:at + rows, :] = 200.0
    return a


def test_a3_lag_is_all_or_nothing_over_its_directions():
    """Finding 1. The spec freezes MIN_PAIRS_PER_LAG as "below this a VARIOGRAM LAG is undefined",
    and a lag is one measurement over the two registered directions, so one short direction
    undefineds the lag rather than leaving the other direction to answer alone. The A4 path already
    treats its directions this way (_a4's docstring: an omni over three directions is not the
    registered measurement); this makes A3 consistent with it. No new threshold, no softening.
    """
    lag = 5
    for n_common, measured in ((7, False), (8, True)):
        m = np.zeros((30, 40), bool)
        m[3, :n_common] = True                 # n_common pairs with row 3 + lag, same columns
        m[3 + lag, :n_common] = True
        m[20, :30] = True                       # horizontal pairs at this lag, no vertical pairs
        g = np.arange(30 * 40, dtype=float).reshape(30, 40) / 39.0
        counts = lag_direction_counts(m, lag)
        check("hand-built lag %d: one direction at %d pairs, the other above the floor"
              % (lag, n_common),
              counts[0] == n_common and counts[1] >= sf.MIN_PAIRS_PER_LAG, str(counts))
        gam = sf._variogram(g, m)
        if measured:
            check("lag with BOTH directions at or above the floor is measured",
                  np.isfinite(gam[lag - 1]), "gam[%d] = %r" % (lag - 1, gam[lag - 1]))
        else:
            check("lag with ONE direction below the floor is NaN, not that direction's value",
                  np.isnan(gam[lag - 1]),
                  "counts=%s gam=%r" % (counts, gam[lag - 1]))
            check("and the refusal is silent: NaN is written explicitly, not warned about",
                  not noisy(sf._variogram, g, m)[2], str(noisy(sf._variogram, g, m)[2])[:60])

    # The reviewer's tile: a thin band whose variogram has many one-direction lags. The count of
    # lags whose NaN status does NOT match the two-direction floor is the defect, measured.
    g, m = sf.mask_and_field(thin_band(), PPM)[:2]
    gam = sf._variogram(g, m)
    agreed = mism = single = 0
    for i in range(gam.size):
        counts = lag_direction_counts(m, i + 1)
        n_short = sum(1 for c in counts if c < sf.MIN_PAIRS_PER_LAG)
        if n_short == 1:
            single += 1
        should_be_nan = n_short > 0
        if should_be_nan == bool(np.isnan(gam[i])):
            agreed += 1
        else:
            mism += 1
    check("thin band: every lag's NaN status matches the all-or-nothing floor",
          mism == 0, "%d lag(s) reported as measured on one direction; %d of %d lags have a short "
                     "direction" % (mism, single, gam.size))
    check("thin band: the lags that are reported are measured over both directions",
          all(min(lag_direction_counts(m, i + 1)) >= sf.MIN_PAIRS_PER_LAG
              for i in range(gam.size) if np.isfinite(gam[i])),
          "%d finite lag(s)" % int(np.count_nonzero(np.isfinite(gam))))

    # The sill must not be formed by dropping the unmeasured lags out of the trailing window: that
    # is the same silent skip in a different place. Island 120 x 200 in a 256 tile has a tail window
    # in which 8 of the 12 lags have a short direction, so the sill is undefined, not "the mean of
    # the four that survived".
    g2 = np.arange(256 * 256, dtype=float).reshape(256, 256) / 511.0
    m2 = np.zeros((256, 256), bool)
    m2[10:130, 20:220] = True
    gam2 = sf._variogram(g2, m2)
    win = max(1, int(sf.A3_SILL_WINDOW * gam2.size))
    short_in_window = sum(1 for i in range(gam2.size - win, gam2.size)
                          if any(c < sf.MIN_PAIRS_PER_LAG for c in lag_direction_counts(m2, i + 1)))
    check("sill window probe: its trailing %d lags are %d short of measurable"
          % (win, short_in_window),
          short_in_window > 0 and short_in_window < win,
          "%d of %d" % (short_in_window, win))
    exc = caught(sf._variogram_range, g2, m2, PPM)
    check("a sill window with an unmeasured lag raises rather than averaging the survivors",
          isinstance(exc, sf.TileDegenerate), "%s | %s" % (type(exc).__name__, str(exc)[:70]))
    check("the raise names the unmeasured lags in the window",
          "window" in str(exc).lower() and str(win) in str(exc), str(exc)[:88])

    # The range search must not step over an unmeasured lag on its way to the crossing. Masked rows
    # are the even rows, so every odd lag has no vertical pair at all while the horizontal count
    # stays high; the tail window (win = 1, lag 8) is measured, so the sill is well defined and the
    # only thing that can refuse the range is the hole in front of the crossing.
    g3 = np.broadcast_to(5.0 + 0.7 * np.arange(19)[:, None] + 0.3 * np.arange(19)[None, :],
                         (19, 19)).copy()
    m3 = np.zeros((19, 19), bool)
    m3[::2, :] = True
    gam3 = sf._variogram(g3, m3)
    odd_short = [i for i in range(gam3.size)
                 if lag_direction_counts(m3, i + 1)[0] < sf.MIN_PAIRS_PER_LAG]
    check("crossing probe: unmeasured lags sit in front of the window's sill",
          bool(odd_short) and np.isfinite(gam3[-1]),
          "unmeasured lag indices %s, last lag %.4f" % (odd_short, gam3[-1]))
    exc = caught(sf._variogram_range, g3, m3, PPM)
    check("an unmeasured lag before the crossing raises, it is not skipped",
          isinstance(exc, sf.TileDegenerate), "%s | %s" % (type(exc).__name__, str(exc)[:70]))
    check("the raise says the first lag that reaches the sill cannot be located",
          "first" in str(exc).lower() or "before" in str(exc).lower(), str(exc)[:88])
    check("and the module never answers either refusal with a number",
          isinstance(caught(sf._variogram_range, g2, m2, PPM), sf.TileDegenerate)
          and isinstance(caught(sf._variogram_range, g3, m3, PPM), sf.TileDegenerate),
          "the window rule and the crossing rule both fire, never a max-lag stand-in")


def test_each_frozen_floor_gates_its_own_statistic():
    """Finding 4. The deleted pins said a threshold NAME appears in the source, which the constant's
    own definition line satisfies no matter what the code does with it. What has to be true instead
    is that each floor GATES a statistic: a tile just above it yields a number, a tile just below it
    yields NaN. One pair per threshold, at the exact value, with the counts re-derived here.
    """
    # MIN_MASK_PIXELS: 16 pixels is a tile, 15 is not.
    sixteen = [(i, j) for i in range(4) for j in range(4)]
    for n, want_valid in ((15, False), (16, True)):
        a = mask_tile((20, 20), sixteen[:n])
        exc = caught(sf.mask_and_field, a, PPM)
        check("a mask of %d pixels is %s" % (n, "a tile" if want_valid else "a degeneracy"),
              (exc is None) == want_valid, "%s | %s" % (type(exc).__name__, str(exc)[:60]))
        check("a mask of %d pixels: the count is in the message and matches the tile" % n,
              exc is None or (str(n) in str(exc) and "need %d" % sf.MIN_MASK_PIXELS in str(exc)),
              str(exc)[:72])
    a = mask_tile((20, 20), sixteen)
    out = sf.tile_spatial(a, PPM)
    check("MIN_MASK_PIXELS: exactly 16 mask pixels gives a finite A1, 15 never gets that far",
          int(sf.mask_and_field(a, PPM)[1].sum()) == sf.MIN_MASK_PIXELS
          and np.isfinite(out["A1"]), "%.6f" % out["A1"])

    # MIN_PROFILE_LINES: 4 masked rows is a measurable 90 deg direction, 3 is not.
    for nrows, want in ((3, False), (4, True)):
        a = mask_tile((20, 20), [(i, j) for i in range(8, 8 + nrows) for j in range(20)])
        g, m, e = sf.mask_and_field(a, PPM)
        lines_y, energy_y = profile_energy(e, m, 0)
        check("MIN_PROFILE_LINES: %d masked rows give %d valid 90 deg lines" % (nrows, lines_y),
              lines_y == nrows, str(lines_y))
        check("MIN_PROFILE_LINES: the %d-row tile's other direction has profile variance" % nrows,
              energy_y > 0.0 and profile_energy(e, m, 1)[1] > 0.0,
              "90deg=%.6g 0deg=%.6g" % (energy_y, profile_energy(e, m, 1)[1]))
        exc = caught(sf._acf_along, e, m, 0)
        check("MIN_PROFILE_LINES: %d lines is %s" % (nrows, "measured" if want else "undefined"),
              (exc is None) == want, "%s | %s" % (type(exc).__name__, str(exc)[:60]))
        out = sf.tile_spatial(a, PPM)
        # RE-DERIVED under the owner's follow-up A2 ruling (option a). The line-count floor gate is
        # the check just above, and it is unchanged: 4 lines clears it, 3 does not. What changes is
        # the A2 VALUE the 4-row tile reports. Only 4 of its 20 y-lines carry a mean at all, so its
        # 90 deg 1/e crossing lands at lag 0.843, below one full lag step, and the ratio has no
        # measurable decay in that direction: NaN, on the resolvable-crossing rule rather than on
        # the line count. Both tiles are now NaN and the REASON is what is asserted here, because
        # the pre-ruling check ("a number" for 4 lines) was one of the undefined-returns-a-number
        # answers the ruling removes.
        check("MIN_PROFILE_LINES: A2 on the %d-row tile is NaN, %s"
              % (nrows, "on the line count" if not want else "on the crossing instead"),
              np.isnan(out["A2"]), repr(out["A2"]))
        if want:
            ac90 = sf._acf_along(e, m, 0)
            check("MIN_PROFILE_LINES: the 4-row tile's direction IS measured, and its crossing is "
                  "the thing that is not", 0.0 <= efold_pre_ruling(ac90) < 1.0,
                  "pre-ruling length %.6f of %d lags" % (efold_pre_ruling(ac90), ac90.size))
        else:
            check("MIN_PROFILE_LINES: the 3-row tile is refused on the line count before any "
                  "crossing is looked at", "profile lines" in str(exc), str(exc)[:70])

    # MIN_PAIRS_PER_DIRECTION: A4's diagonal pair count exactly 8 is measured, 7 is not.
    lag_px = int(round(sf.A4_LAG_MM * PPM))
    for run_len, want in ((10, False), (11, True)):
        a = mask_tile((20, 20), [(9 + off, 3 + j) for off in (0, 3, 5) for j in range(run_len)])
        g, m, _ = sf.mask_and_field(a, PPM)
        counts = {}
        for deg, (oy, ox) in ((0, (0, lag_px)), (90, (lag_px, 0)),
                              (45, (sf.A4_DIAG_PX, sf.A4_DIAG_PX)),
                              (135, (sf.A4_DIAG_PX, -sf.A4_DIAG_PX))):
            sl_a, sl_b = sf._pair_slices(m.shape, oy, ox)
            counts[deg] = int(np.count_nonzero(m[sl_a] & m[sl_b]))
        check("MIN_PAIRS_PER_DIRECTION: run %d puts the diagonals at %d pairs"
              % (run_len, 11 - run_len + 8),
              counts[45] == run_len - sf.A4_DIAG_PX and counts[0] >= sf.MIN_PAIRS_PER_DIRECTION
              and counts[90] >= sf.MIN_PAIRS_PER_DIRECTION, str(counts))
        out = sf.tile_spatial(a, PPM)
        check("MIN_PAIRS_PER_DIRECTION: A4 on run %d (diagonals at %d pairs) is %s"
              % (run_len, counts[45], "a number" if want else "NaN"),
              np.isfinite(out["A4"]) == want, "A4=%r counts=%s" % (out["A4"], counts))
    for n_common, want in ((7, False), (8, True)):
        m = np.zeros((24, 24), bool)
        m[2, :n_common] = True
        m[2 + lag_px, :n_common] = True
        m[20, :20] = True                        # horizontal pairs, no vertical pair of its own
        g = np.arange(24 * 24, dtype=float).reshape(24, 24) / 23.0
        counts = lag_direction_counts(m, lag_px)
        mad = float(np.median(np.abs(g[m] - np.median(g[m])))) + sf.EPS
        exc = caught(sf._direction_contrast, g, m, lag_px, 0, mad)
        check("_direction_contrast at %d pairs in the offset direction (counts %s) is %s"
              % (n_common, counts, "measured" if want else "undefined"),
              counts[0] == n_common and counts[1] >= sf.MIN_PAIRS_PER_DIRECTION
              and (exc is None) == want, "%s | %s" % (counts, str(exc)[:60]))

    # MIN_PAIRS_PER_LAG: the same 10 x 10 block, with the last row trimmed to 7 or 8 shared columns.
    full = [(i, j) for i in range(9, 19) for j in range(10)]
    for keep_cols, want in ((7, False), (8, True)):
        px = [p for p in full if p[0] != 18 or p[1] < keep_cols]
        a = mask_tile((20, 20), px)
        g, m, _ = sf.mask_and_field(a, PPM)
        counts = lag_direction_counts(m, 9)
        check("MIN_PAIRS_PER_LAG: trimmed block has %d pairs at lag 9 one way, %d the other"
              % (keep_cols, counts[1]), counts[0] == keep_cols, str(counts))
        gam = sf._variogram(g, m)
        check("MIN_PAIRS_PER_LAG: lag 9 with a %d-pair direction is %s"
              % (keep_cols, "measured" if want else "NaN"),
              np.isfinite(gam[8]) == want, "gam[8]=%r" % gam[8])
        out = sf.tile_spatial(a, PPM)
        check("MIN_PAIRS_PER_LAG: A3 is %s when one direction of the sill lag is at %d pairs"
              % ("a number" if want else "NaN", keep_cols),
              np.isfinite(out["A3"]) == want, "A3=%r" % out["A3"])
    a = mask_tile((20, 20), full)
    out = sf.tile_spatial(a, PPM)
    check("MIN_PAIRS_PER_LAG: the untrimmed block clears every lag and A3 is a number",
          np.isfinite(out["A3"]) and lag_direction_counts(sf.mask_and_field(a, PPM)[1], 9)[0]
          >= sf.MIN_PAIRS_PER_LAG, "A3=%.6f" % out["A3"])


def test_a2_zero_energy_profile_is_nan_not_one():
    """Finding 2. A masked profile with no variance has no autocorrelation, so it has no e-folding
    length, and the EPS floor turned the ratio of the two absent lengths into exactly 1.0 - the same
    number a perfectly isotropic tile returns. The condition is structural (zero variance, or no
    finite pairs), so it adds no threshold; and it is a degenerate TILE, so it raises TileDegenerate
    and arrives as NaN rather than being routed as an invalid call.
    """
    val, err, got = noisy(sf._a2, np.zeros((64, 64)), np.ones((64, 64), dtype=bool))
    check("the reviewer's zero-energy probe no longer answers 1.0",
          val is None or np.isnan(val), "got %r (was 1.0)" % (val,))
    check("the zero-energy probe raises TileDegenerate, which tile_spatial turns into NaN",
          isinstance(err, sf.TileDegenerate) and not isinstance(err, ValueError),
          "%s | %s" % (type(err).__name__, str(err)[:66]))
    check("the zero-energy probe emits no warning", not got, "; ".join(got)[:70])
    exc = caught(sf._acf_along, np.zeros((64, 64)), np.ones((64, 64), dtype=bool), 1)
    check("a zero-variance direction raises TileDegenerate, not ValueError",
          isinstance(exc, sf.TileDegenerate) and not isinstance(exc, ValueError),
          "%s | %s" % (type(exc).__name__, str(exc)[:60]))
    check("and the message says the profile has no variance to correlate",
          "variance" in str(exc).lower() and "0" in str(exc), str(exc)[:88])
    const = caught(sf._a2, np.full((64, 64), 7.0), np.ones((64, 64), bool))
    check("a constant non-zero field is the same degeneracy, not a ratio of two floors",
          isinstance(const, sf.TileDegenerate), "%s | %s" % (type(const).__name__,
                                                             str(const)[:70]))
    check("and it is not an invalid call wearing the degeneracy's type",
          not isinstance(const, ValueError), type(const).__name__)

    # The same thing through the real tile path: a flat band has an exactly constant masked profile
    # along x, so A2 is undefined whether or not the other direction has variance.
    for rows, why in ((6, "both directions"), (8, "one direction only")):
        a = flat_band(rows)
        g, m, e = sf.mask_and_field(a, PPM)
        energies = [profile_energy(e, m, ax) for ax in (1, 0)]
        check("flat %d-row band: %s has exactly zero profile variance" % (rows, why),
              any(en[1] == 0.0 for en in energies) and all(en[0] >= sf.MIN_PROFILE_LINES
                                                          for en in energies),
              "lines/energy per direction: %s" % (energies,))
        out, err, got = noisy(sf.tile_spatial, a, PPM)
        check("flat %d-row band: tile_spatial returns, it does not raise" % rows,
              err is None, str(err)[:60])
        check("flat %d-row band: A2 is NaN, not the isotropic-looking number" % rows,
              np.isnan(out["A2"]), "A2 was %r" % (out["A2"],))
        check("flat %d-row band: the other statistics are untouched" % rows,
              np.isfinite(out["A1"]), "A1=%.6f" % out["A1"])
        check("flat %d-row band: no warning while A2 goes undefined" % rows, not got,
              "; ".join(got)[:70])

    # Structural, not small: a profile whose variance is merely tiny is still measured. This is the
    # boundary the ruling requires - an undefined statistic is NaN on its OWN condition, and a
    # value-based cutoff would be a new parameter in everything but name.
    tiny = 1e-15 * np.tile(np.arange(64.0)[:, None], (1, 64))
    full = np.ones((64, 64), dtype=bool)
    en_tiny = profile_energy(tiny, full, 0)[1]
    en_flat = profile_energy(np.zeros((64, 64)), full, 1)[1]
    check("a 1e-15 profile variance is non-zero, so that direction is measured",
          en_tiny > 0.0 and np.isfinite(sf._acf_along(tiny, full, 0)).all(),
          "energy=%.6g" % en_tiny)
    check("the exactly-constant direction of the same shaped field is the one that refuses",
          en_flat == 0.0 and isinstance(caught(sf._acf_along, np.zeros((64, 64)), full, 1),
                                       sf.TileDegenerate), "energy=%.6g" % en_flat)
    # RE-DERIVED under the owner's follow-up A2 ruling (option a). `tiny + tiny.T` above has a
    # centred energy of 2.2e-26, four orders BELOW the EPS that normalises it, so its 1/e crossing
    # lands at lag 0 in both directions: the pre-ruling code answered it with exactly 1.0, which is
    # the number this ruling removes. The "no magnitude floor" property the check was standing for
    # is asserted on a profile that is still astronomically flat in grey levels - 1e-6 per pixel -
    # but whose crossing the lag grid CAN resolve, so a value-based cutoff is still excluded.
    check("the 1e-15 probe is undefined on the new rule: its crossing is below a lag step",
          isinstance(caught(sf._a2, tiny + tiny.T, full), sf.TileDegenerate),
          "%s | %s" % (type(caught(sf._a2, tiny + tiny.T, full)).__name__,
                       str(caught(sf._a2, tiny + tiny.T, full))[:60]))
    flat_x = 1e-6 * np.tile(np.arange(96.0)[None, :], (64, 1))
    flat_y = 1e-6 * np.tile(np.arange(64.0)[:, None], (1, 96))
    flat_field, flat_full = flat_x + flat_y, np.ones((64, 96), dtype=bool)
    flat_en = [profile_energy(flat_field, flat_full, ax)[1] for ax in (1, 0)]
    check("the flat control's energies are tiny in grey levels yet non-zero",
          all(0.0 < en < 1e-5 for en in flat_en), "energies=%.6g / %.6g" % tuple(flat_en))
    check("A2 over tiny-but-non-zero variance is a number: no value cutoff was introduced",
          np.isfinite(sf._a2(flat_field, flat_full)), repr(sf._a2(flat_field, flat_full)))
    check("and it is a measured ratio, not the isotropic 1.0 the EPS floor used to hand out",
          abs(sf._a2(flat_field, flat_full) - 1.0) > 1e-6, "%.9f" % sf._a2(flat_field, flat_full))
    # And the honest 1.0 survives: a tile that really is isotropic still returns about 1.
    iso = sf.tile_spatial(grain_noise(), PPM)["A2"]
    check("a genuinely isotropic tile still returns A2 within 2 percent of 1.0",
          abs(iso - 1.0) < 0.02, "%.6f" % iso)
    check("so 1.0 is no longer reachable by a tile that measured nothing",
          abs(iso - 1.0) < 0.02 and np.isnan(sf.tile_spatial(flat_band(6), PPM)["A2"]),
          "%.6f vs NaN" % iso)


def test_aggregation_keeps_all_nan_and_non_finite_apart():
    """Finding 3. tile_spatial:333 calls a non-finite statistic a bug, so a feature column
    holding an inf is an invalid CALL. The aggregation used to answer it with NaN while announcing an all-NaN
    column that did not exist, and answered [inf, 2.0] with inf and no notice at all.
    """
    for bad in (float("inf"), float("-inf")):
        rows = [{"A1": bad, "A2": 2.0, "A3": 3.0, "A4": 4.0} for _ in range(2)]
        exc = caught(sf.image_block, rows)
        check("image_block over A1 = %r raises ValueError" % bad,
              isinstance(exc, ValueError) and not isinstance(exc, sf.TileDegenerate),
              "%s | %s" % (type(exc).__name__, str(exc)[:70]))
        check("image_block over A1 = %r is not reported as an all-NaN column" % bad,
              "nan in every" not in str(exc).lower(), str(exc)[:88])
        check("image_block over A1 = %r names the statistic and the bad value" % bad,
              "A1" in str(exc) and "inf" in str(exc).lower(), str(exc)[:88])
        check("image_block over A1 = %r emits no warning on the invalid path" % bad,
              not noisy(sf.image_block, rows)[2], str(noisy(sf.image_block, rows)[2])[:70])
        check("soil_block over A1 = %r raises the same way" % bad,
              isinstance(caught(sf.soil_block, rows), ValueError),
              type(caught(sf.soil_block, rows)).__name__)
    mixed = [{"A1": float("inf"), "A2": 2.0, "A3": 3.0, "A4": 4.0},
             {"A1": 2.0, "A2": 2.0, "A3": 3.0, "A4": 4.0}]
    exc = caught(sf.image_block, mixed)
    check("a column with ONE inf and one finite value raises instead of returning inf",
          isinstance(exc, ValueError), "%s | %s" % (type(exc).__name__, str(exc)[:70]))
    check("and the returned table is never allowed to carry an inf",
          not isinstance(exc, sf.TileDegenerate) and "inf" in str(exc).lower(), str(exc)[:88])
    both = [{"A1": float("inf"), "A2": 2.0, "A3": 3.0, "A4": 4.0},
            {"A1": float("nan"), "A2": 2.0, "A3": 3.0, "A4": 4.0}]
    check("inf mixed with NaN in the same column is still the invalid case",
          isinstance(caught(sf.image_block, both), ValueError),
          type(caught(sf.image_block, both)).__name__)

    # The distinction has to go BOTH ways: a genuine all-NaN column stays what it was.
    allnan = [{"A1": float("nan"), "A2": 2.0, "A3": 3.0, "A4": 4.0},
              {"A1": float("nan"), "A2": 6.0, "A3": 7.0, "A4": 8.0}]
    out, err, got = noisy(sf.image_block, allnan)
    check("an all-NaN column is still answered with NaN and one notice",
          err is None and np.isnan(out["A1"]) and len(got) == 1, "%s | %s" % (out, got))
    check("the notice still names A1 and never claims an inf",
          got and "A1" in got[0] and "NaN" in got[0] and "inf" not in got[0].lower(), str(got)[:88])
    check("a column with NaN and a finite value stays a plain median",
          sf.image_block([{"A1": float("nan"), "A2": 1.0, "A3": 1.0, "A4": 1.0},
                          {"A1": 3.0, "A2": 1.0, "A3": 1.0, "A4": 1.0}])["A1"] == 3.0,
          "plain median")
    check("no finite column is disturbed by another column's inf raise message",
          "A2" not in str(caught(sf.image_block, mixed)), str(caught(sf.image_block, mixed))[:88])


def test_efold_crossing_is_the_registered_one_over_e():
    """Finding 4. Behavioural replacement for the whitespace-exact `thr = 1.0 / np.e` pin: probe the
    crossing itself, including the boundary case where an ACF value sits exactly on 1/e.

    Three checks here were RE-DERIVED under the owner's follow-up ruling on the A2 cell (option a),
    because they pinned exactly the two undefined answers the ruling removes: "a non-decaying ACF is
    the maximum lag", "an all-zero ACF is zero samples", and a crossing interpolated inside the
    first interval. Each now asserts the refusal, and the interpolation formula it used to test is
    kept alive on a crossing that lands at or after one full lag step, where it is a measurement.
    """
    thr = 1.0 / np.e
    got = sf._efold(np.array([1.0, thr, 0.0]))
    check("_efold does not cross at a value exactly on 1/e (the test is strict <)",
          got == 1.0, "%.12f" % got)
    check("and a crossing landing exactly on the first lag step IS one full lag step, so it stays",
          got >= 1.0, "%.12f" % got)
    got = sf._efold(np.array([1.0, 0.5, 0.0]))
    want = 1.0 + (0.5 - thr) / (0.5 - 0.0)
    check("_efold interpolates the crossing at 1/e, not at any other fraction",
          abs(got - want) < 1e-12, "got %.12f want %.12f" % (got, want))
    got = sf._efold(np.array([1.0, 0.9, 0.3, 0.0]))
    want = 1.0 + (0.9 - thr) / (0.9 - 0.3)
    check("_efold interpolates a crossing inside the SECOND interval, one full lag step or later",
          abs(got - want) < 1e-12 and got >= 1.0, "got %.12f want %.12f" % (got, want))
    exc = caught(sf._efold, np.array([1.0, 0.3, 0.0]))
    check("_efold refuses a crossing inside the FIRST interval, below one full lag step, rather "
          "than returning (1 - 1/e)/(1 - 0.3) = 0.903 samples",
          isinstance(exc, sf.TileDegenerate), "%s | %s" % (type(exc).__name__, str(exc)[:66]))
    exc = caught(sf._efold, np.ones(5))
    check("_efold of a non-decaying ACF raises instead of returning the maximum lag 5.0",
          isinstance(exc, sf.TileDegenerate) and not isinstance(exc, ValueError),
          "%s | %s" % (type(exc).__name__, str(exc)[:66]))
    exc = caught(sf._efold, np.zeros(5))
    check("_efold of an all-zero ACF raises instead of returning a zero length",
          isinstance(exc, sf.TileDegenerate) and not isinstance(exc, ValueError),
          "%s | %s" % (type(exc).__name__, str(exc)[:66]))
    check("and neither refusal is an invalid call wearing the degeneracy's type",
          not isinstance(exc, ValueError), type(exc).__name__)


def test_named_constants_are_the_ones_the_statistics_read():
    """Finding 4. Behavioural replacement for the "appears exactly once" and "EPS used six times"
    pins: prove the arithmetic reads the NAMED frozen constant, which is the property the literal
    counts were standing in for. Each patch is restored, and the frozen value re-asserted after.
    """
    g, m = sf.mask_and_field(blobs(), PPM)[:2]
    gam = sf._variogram(g, m)
    win = max(1, int(sf.A3_SILL_WINDOW * gam.size))
    sill = float(np.mean(gam[-win:]))
    original = sf.A3_SILL_FRACTION
    try:
        crossings = {}
        for f in (0.9, 0.5):
            sf.A3_SILL_FRACTION = f
            rng = sf._variogram_range(g, m, PPM)
            crossings[f] = int(round(rng * PPM))
            want = next(k for k in range(1, gam.size + 1) if gam[k - 1] >= f * sill)
            check("A3's crossing is the first lag at %g of the sill read from the constant" % f,
                  crossings[f] == want, "module lag %d, independent lag %d" % (crossings[f], want))
        check("a lower registered fraction cannot put the range later",
              crossings[0.5] <= crossings[0.9], "%d vs %d" % (crossings[0.5], crossings[0.9]))
    finally:
        sf.A3_SILL_FRACTION = original
    check("A3_SILL_FRACTION is restored to its frozen value",
          sf.A3_SILL_FRACTION == 0.9, str(sf.A3_SILL_FRACTION))
    check("and the range is back to what the frozen fraction gives",
          abs(sf._variogram_range(g, m, PPM) * PPM - crossings[0.9]) < 1e-12, "restored")

    # A4's MAD belongs to the tile, not to the direction: four calls, one value, and it is the
    # tile's own MAD.
    a = blobs()
    g, m, _ = sf.mask_and_field(a, PPM)
    want_mad = float(np.median(np.abs(g[m] - np.median(g[m])))) + sf.EPS
    real = sf._direction_contrast
    seen = []

    def spy(gg, mm, oy, ox, mad):
        seen.append((oy, ox, mad))
        return real(gg, mm, oy, ox, mad)

    sf._direction_contrast = spy
    try:
        out = sf._a4(g, m, PPM)
    finally:
        sf._direction_contrast = real
    check("A4 asks for exactly the four registered directions",
          sorted((oy, ox) for oy, ox, _ in seen)
          == sorted([(0, int(round(sf.A4_LAG_MM * PPM))),
                     (int(round(sf.A4_LAG_MM * PPM)), 0),
                     (sf.A4_DIAG_PX, sf.A4_DIAG_PX), (sf.A4_DIAG_PX, -sf.A4_DIAG_PX)]),
          str([(oy, ox) for oy, ox, _ in seen]))
    check("A4's MAD is computed once per tile and passed to every direction",
          len(seen) == 4 and all(abs(mad - want_mad) < 1e-15 for _, _, mad in seen),
          "%d calls, mads=%s" % (len(seen), sorted(set(round(x[2], 12) for x in seen))))
    check("the spy did not change the statistic", abs(out - sf._a4(g, m, PPM)) < 1e-15,
          "%.12f" % out)

    # EPS is a floor inside the arithmetic, which is observable at each site that uses it.
    out = sf.tile_spatial(ramp(), PPM)
    gy, gx = np.gradient(sf.mask_and_field(ramp(), PPM)[0])
    mm = sf.mask_and_field(ramp(), PPM)[1]
    jxx, jyy = float(np.mean(gx[mm] ** 2)), float(np.mean(gy[mm] ** 2))
    jxy = float(np.mean(gx[mm] * gy[mm]))
    tr, det = jxx + jyy, jxx * jyy - jxy ** 2
    l1 = tr / 2.0 + np.sqrt(max(tr * tr / 4.0 - det, 0.0))
    check("A1 on the rank-1 ramp sits on EPS, so the log floor is EPS and not a smaller number",
          abs(out["A1"] - np.log((0.0 + sf.EPS) / (l1 + sf.EPS))) < 1e-9,
          "A1=%.9f floor=%.9f" % (out["A1"], np.log(sf.EPS / l1)))
    flat = sf.tile_spatial(flat_band(6), PPM)
    check("A4 on a tile whose MAD is exactly zero returns a finite fraction, not NaN",
          np.isfinite(flat["A4"]), "A4=%r" % (flat["A4"],))
    g0 = np.zeros((32, 32))
    m0 = np.ones((32, 32), bool)            # every lag measurable, every difference zero: sill 0.0
    exc = caught(sf._variogram_range, g0, m0, PPM)
    check("a zero-sill variogram raises instead of dividing by the floor",
          isinstance(exc, sf.TileDegenerate) and "sill" in str(exc).lower(),
          "%s | %s" % (type(exc).__name__, str(exc)[:70]))


def test_frozen_floors_are_still_the_frozen_numbers():
    """The behavioural gates above are only meaningful while the constants they move are the frozen
    ones, so the pairing is asserted here rather than left to the value table alone."""
    for name in ("MIN_MASK_PIXELS", "MIN_PAIRS_PER_LAG", "MIN_PAIRS_PER_DIRECTION",
                 "MIN_PROFILE_LINES"):
        check("%s still has its frozen value after the gating tests" % name,
              getattr(sf, name) == {"MIN_MASK_PIXELS": 16, "MIN_PAIRS_PER_LAG": 8,
                                    "MIN_PAIRS_PER_DIRECTION": 8, "MIN_PROFILE_LINES": 4}[name],
              str(getattr(sf, name)))
    check("the gating tiles are the ones designed, not accidental near-misses",
          int(sf.mask_and_field(mask_tile((20, 20), [(i, j) for i in range(4) for j in range(4)]),
                                PPM)[1].sum()) == 16, "mask size")


# ------------------------------------------------------------------ carried over from Task 1
def test_returns_four_keys_in_order():
    out = sf.tile_spatial(striped(), 4.552516)
    check("returns exactly A1-A4", tuple(out) == sf.SPATIAL, str(tuple(out)))
    check("all four are finite floats on a healthy tile",
          all(isinstance(v, float) and np.isfinite(v) for v in out.values()))
    check("every value is a Python float, never a numpy scalar or None",
          all(type(v) is float for v in out.values()),
          str([type(v).__name__ for v in out.values()]))


def test_orientation_is_detected():
    """A1 is an anisotropy MAGNITUDE, so it is orientation-INVARIANT by construction.

    Measured 2026-09-29: a striped tile and its transpose both give exactly -33.761250, so an
    assertion that they differ could never have passed. What A1 must do is ignore WHICH axis is
    dominant and separate a striped tile from a near-isotropic one (measured difference 33.74).
    A2 is the directional one, and after ruling 2 it is a ratio, so it INVERTS under transpose
    instead of flipping sign: the old assertion abs(h + v) < 1e-6 was re-derived to h * v == 1.
    """
    horiz = sf.tile_spatial(striped(), 4.552516)
    vert = sf.tile_spatial(np.transpose(striped(), (1, 0, 2)).copy(), 4.552516)
    iso = sf.tile_spatial(blobs(), 4.552516)
    check("A1 is invariant under transposing the tile",
          abs(horiz["A1"] - vert["A1"]) < 1e-9, "%.6f vs %.6f" % (horiz["A1"], vert["A1"]))
    check("A1 separates a striped tile from a near-isotropic one",
          abs(horiz["A1"] - iso["A1"]) > 0.5, "%.4f vs %.4f" % (horiz["A1"], iso["A1"]))
    check("A2 inverts under transpose, because it is a directional ratio",
          abs(horiz["A2"] * vert["A2"] - 1.0) < 1e-9 and abs(horiz["A2"] - vert["A2"]) > 1e-6,
          "%.4f vs %.4f" % (horiz["A2"], vert["A2"]))


def test_mask_and_field_match_frozen_definitions():
    """g, m, e must be the pixel set the existing 12 features are computed on.

    The frozen definitions are preprocess/verify.py:115-119: g = channel mean,
    m = g > percentile(g, 12), e = (g - illumination base of sigma BASE_SIGMA_MM * ppm) inside m
    and 0.0 outside it. Recomputing them here independently pins the reuse rather than trusting
    the import. Three tiles, not one: blobs() has a flat 120.0 background covering well over 88% of
    its pixels, so its 12th and 0th percentiles are the same number and a module using the wrong
    percentile would still match on it.
    """
    from preprocess.verify import BASE_SIGMA_MM, _illumination_base
    for name, a in (("blobs", blobs()), ("noise", grain_noise()), ("coarse16", coarse_blocks())):
        g, m, e = sf.mask_and_field(a, PPM)
        gg = np.asarray(a, float).mean(axis=2)
        mm = gg > np.percentile(gg, 12)
        ee = np.where(mm, gg - _illumination_base(gg, BASE_SIGMA_MM * PPM), 0.0)
        check("%s: grey field equals the channel mean" % name, np.array_equal(g, gg))
        check("%s: mask equals g > percentile(g, 12)" % name, np.array_equal(m, mm),
              "%d of %d px" % (int(m.sum()), m.size))
        check("%s: residual equals the frozen illumination residual" % name, np.array_equal(e, ee))
        check("%s: residual is zero outside the mask" % name,
              float(np.abs(e[~m]).max()) == 0.0)
        _, mf_err, mf_warn = noisy(sf.mask_and_field, a, PPM)
        check("%s: mask_and_field neither warns nor raises" % name,
              not mf_warn and mf_err is None,
              "%s | %s" % ("; ".join(mf_warn)[:50], str(mf_err)[:50]))
    check("BASE_SIGMA_MM is the frozen 12 mm", BASE_SIGMA_MM == 12.0, str(BASE_SIGMA_MM))
    check("the illumination base is imported, not re-derived here",
          "_illumination_base" in src_text() and "gaussian_filter" not in code_only_text(
              Path(__file__).resolve().parent / "spatial_features.py"))


def test_rank1_gradient_exercises_the_eigenvalue_floor():
    """A1 on a rank-1 gradient must land on the EPS floor, not on a negative eigenvalue.

    tr/2 - sqrt(disc) is exactly zero, up to roundoff, when every gradient vector is parallel, so a
    linear ramp is the case where max(..., 0.0) is load-bearing: without it l2 goes slightly
    negative, log((l2+eps)/(l1+eps)) is NaN, and A1 becomes non-finite. With eps = 1e-12 and a
    255-over-255-px ramp l1 is about 1.0, so the expected value is log(1e-12 / 1) = -27.63.
    """
    out = sf.tile_spatial(ramp(), PPM)
    check("ramp: A1 is finite (the l2 floor held)", np.isfinite(out["A1"]), "%.6f" % out["A1"])
    check("ramp: A1 sits on the eps floor, within 5 log units of -27.63",
          abs(out["A1"] + 27.631) < 5.0, "%.6f" % out["A1"])
    check("ramp: A1 is at least as anisotropic as -20", out["A1"] < -20.0, "%.6f" % out["A1"])


def test_sub_half_mm_scale_path_is_loud():
    """ppm below 1 px/mm makes A4's lag_px round to zero, which is not a measurable 1 mm lag.

    Re-derived after the rulings: the raise is now this module's own ValueError naming the lag, not
    a numpy broadcast clash, because an unrepresentable physical lag is an invalid call rather than
    tile degeneracy. Canonical tiles are at 4.552516 px/mm, where lag_px is 5, so this path is
    unreachable in the experiment and exists to make the boundary explicit.
    """
    exc = caught(sf.tile_spatial, blobs(), 0.4)
    check("ppm=0.4 raises", isinstance(exc, Exception), type(exc).__name__)
    check("ppm=0.4 raises ValueError naming the lag", isinstance(exc, ValueError)
          and "lag" in str(exc).lower(), "%s | %s" % (type(exc).__name__, str(exc)[:70]))
    check("ppm=0.4 is not catchable as tile degeneracy", not isinstance(exc, sf.TileDegenerate))
    exc = caught(sf.tile_spatial, blobs(), 0.6)
    check("ppm=0.6 (lag_px rounds to 1) runs cleanly", exc is None, str(exc)[:60])
    exc = caught(sf.tile_spatial, blobs(), 1.0)
    check("ppm=1.0 (lag_px exactly 1) runs cleanly", exc is None, str(exc)[:60])


def test_a3_is_ppm_invariant_by_construction():
    """Recorded property, not a wish: A3's mm units cancel, so ppm cannot move it.

    range_mm = h[hit] / ppm and extent_mm = height / ppm, so A3 = h[hit] / (0.5 * height) is a pure
    lag-count ratio. A4 moves instead, because its lag is round(1 mm * ppm), and A2 moves because
    the illumination-base sigma is 12 mm * ppm. The owner ruled that A2's scale sensitivity is
    accepted and the statistic is NOT to be altered because of it.
    """
    for name, a in (("blobs", blobs()), ("coarse16", coarse_blocks()), ("stripe", striped())):
        got = [sf.tile_spatial(a, p)["A3"] for p in (2.276258, 4.552516, 9.105032)]
        check("%s: A3 identical across three ppm values" % name,
              max(got) - min(got) == 0.0, " ".join("%.6f" % v for v in got))
    a4 = [sf.tile_spatial(blobs(), p)["A4"] for p in (2.276258, 4.552516, 9.105032)]
    check("blobs: A4 does move across the same three ppm values",
          max(a4) - min(a4) > 0.005, " ".join("%.6f" % v for v in a4))
    a2 = [sf.tile_spatial(blobs(), p)["A2"] for p in (2.276258, 4.552516, 9.105032)]
    check("blobs: A2 moves across the same three ppm values (accepted, not corrected)",
          max(a2) / min(a2) > 1.05, " ".join("%.6f" % v for v in a2))
    print("INFO   A2 vs ppm | " + " ".join("%.6f" % v for v in a2))
    print("INFO   A4 vs ppm | " + " ".join("%.6f" % v for v in a4))


def test_determinism():
    """Bit-identical repeats, NaN included: two NaNs in the same key are the same answer.

    A plain `first[k] == second[k]` would call that a difference, and the pre-ruling module could
    get away with it because it never returned NaN. After ruling 5 it does, so the comparison has
    to say what it means.
    """
    def same(a, b):
        return a == b or (np.isnan(a) and np.isnan(b))
    for name, a in (("blobs", blobs()), ("striped", striped()),
                    ("coarse", coarse_blocks()), ("noise", grain_noise()),
                    ("speck", speck_on_dark()), ("band", thin_band())):
        first = sf.tile_spatial(a, PPM)
        second = sf.tile_spatial(a, PPM)
        check("%s: repeat call is bit-identical" % name,
              all(same(first[k], second[k]) for k in sf.SPATIAL), str(first))
    before = blobs().copy()
    a = before.copy()
    sf.tile_spatial(a, PPM)
    check("tile_spatial does not mutate its input tile", np.array_equal(a, before))
    g, m, e = sf.mask_and_field(before, PPM)
    sf._acf_along(e, m, 1)
    sf._variogram(g, m)
    check("the internal helpers do not mutate the grey field or the mask",
          np.array_equal(before, a) and np.array_equal(m, sf.mask_and_field(before, PPM)[1]))


def test_a2_moves_across_texture_regimes():
    """A2 must be able to change: it is a statistic, not a constant.

    Re-derived for the ratio form. The old floor was 0.05 LOG units, i.e. a 5% difference between
    the two directional e-folding lengths, so the ratio-domain statement of the same floor is
    max/min > exp(0.05). Asserting a spread of 0.05 on raw ratios would be a different and much
    weaker bar, and silently lowering a threshold is the methodology weakening the rulings forbid.
    """
    tiles = {"stripe_x": striped(), "blobs": blobs(), "noise": grain_noise(),
             "coarse16": coarse_blocks(), "fine3": fine_grains()}
    values = finite(value_table(tiles), "A2")
    ratio = max(v["A2"] for v in values.values()) / min(v["A2"] for v in values.values())
    bar = float(np.exp(0.05))
    check("A2's multiplicative spread across five texture regimes exceeds exp(0.05)",
          ratio > bar, "max/min ratio %.6f over %.6f (exp(0.05)) on %d tiles" % (ratio, bar,
                                                                                len(values)))
    stripe, noise = values["stripe_x"]["A2"], values["noise"]["A2"]
    check("A2 on a striped tile is more than 5%% away from A2 on white noise",
          abs(np.log(stripe / noise)) > 0.05, "stripe %.6f vs noise %.6f" % (stripe, noise))
    print("INFO   A2 | " + fmt(values, "A2"))


def test_a3_moves_across_texture_regimes():
    """A3 must separate a 16 px mosaic from a 3 px mosaic and from white noise."""
    tiles = {"stripe_x": striped(), "blobs": blobs(), "noise": grain_noise(),
             "coarse16": coarse_blocks(), "fine3": fine_grains()}
    values = value_table(tiles)
    rng, _ = spread(values, "A3")
    check("A3 moves across five texture regimes", rng > 0.05, "spread %.6f" % rng)
    check("A3 separates the 16 px mosaic from white noise",
          abs(values["coarse16"]["A3"] - values["noise"]["A3"]) > 0.05,
          "coarse %.6f vs noise %.6f" % (values["coarse16"]["A3"], values["noise"]["A3"]))
    check("A3 separates the 16 px mosaic from the 3 px mosaic",
          abs(values["coarse16"]["A3"] - values["fine3"]["A3"]) > 0.05,
          "coarse %.6f vs fine %.6f" % (values["coarse16"]["A3"], values["fine3"]["A3"]))
    print("INFO   A3 | " + fmt(values, "A3"))


def test_a4_moves_across_texture_regimes():
    """A4 must still discriminate after the omni rewrite.

    Reported finding, asserted rather than hidden: with omni averaging 0 and 90 deg symmetrically,
    the registered A4 cannot separate stripes along x from stripes along y on this tile, because
    C(45 deg) and C(135 deg) are equal there and the omni mean is invariant to the swap. The
    pre-ruling code (C(0) - C(45)) could. That is a consequence of the definition the owner chose,
    so it is recorded as an invariance of the statistic instead of being smoothed over.
    """
    tiles = {"stripe_x": striped(), "stripe_y": np.transpose(striped(), (1, 0, 2)).copy(),
             "noise": grain_noise(), "coarse16": coarse_blocks(), "fine3": fine_grains(),
             "blobs": blobs(), "speck5": speck_on_dark()}
    values = finite(value_table(tiles), "A4")
    rng, _ = spread(values, "A4")
    check("A4 moves across the measurable texture regimes", rng > 0.05,
          "spread %.6f over %d tiles" % (rng, len(values)))
    check("A4 separates stripes from white noise",
          abs(values["stripe_x"]["A4"] - values["noise"]["A4"]) > 0.05,
          "stripe %.6f vs noise %.6f" % (values["stripe_x"]["A4"], values["noise"]["A4"]))
    check("A4 on the two stripe orientations is equal, by the omni construction",
          abs(values["stripe_x"]["A4"] - values["stripe_y"]["A4"]) < 1e-12,
          "%.6f vs %.6f" % (values["stripe_x"]["A4"], values["stripe_y"]["A4"]))
    print("INFO   A4 | " + fmt(values, "A4"))


# --------------------------------------------------------------------------- Task 2: placebos
def radial_power(img, nbin=32):
    """Azimuthally averaged |FFT|^2 of the grey field; the DC bin is index 0."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(np.asarray(img, float).mean(axis=2))) ** 2)
    h, w = f.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - h // 2) ** 2 + (xx - w // 2) ** 2).astype(int)
    total = np.bincount(r.ravel(), f.ravel())
    count = np.bincount(r.ravel())
    out = total / np.maximum(count, 1)
    return out[:min(nbin, out.size)]


def spectrum_distance(a, s, nbin=32):
    """Relative L1 distance between two radial power spectra, with and without the DC bin.

    The DC term is one number the re-ranking cannot move (the multiset, and so the mean, is exact)
    and it dominates the sum, so both readings are reported: including it flatters the placebo,
    excluding it is the honest size of the perturbation.
    """
    pa, ps = radial_power(a, nbin), radial_power(s, nbin)
    k = min(pa.size, ps.size)
    full = float(np.abs(pa[:k] - ps[:k]).sum() / max(pa[:k].sum(), 1e-30))
    nodd = float(np.abs(pa[1:k] - ps[1:k]).sum() / max(pa[1:k].sum(), 1e-30))
    return full, nodd


def test_phase_scramble_preserves_the_pixel_multiset():
    """The half of the P-RAND claim that is exactly true."""
    a = blobs()
    s = sf.phase_scrambled(a, seed=7)
    check("scramble keeps the pixel multiset (marginals exact)",
          np.array_equal(np.sort(a.ravel()), np.sort(s.ravel())),
          "compared with array_equal, so the equality is exact and not an rtol")
    check("scramble keeps the pixel multiset on every channel",
          all(np.array_equal(np.sort(a[:, :, c].ravel()), np.sort(s[:, :, c].ravel()))
              for c in range(3)))
    check("scramble keeps the mean exactly, which is the DC term", abs(a.mean() - s.mean()) < 1e-9,
          "%.9f vs %.9f" % (a.mean(), s.mean()))
    check("scramble is not a no-op", not np.allclose(a, s))
    check("scramble returns the same shape, as a float array",
          s.shape == a.shape and np.issubdtype(s.dtype, np.floating), str(s.dtype))
    before = a.copy()
    sf.phase_scrambled(a, seed=7)
    check("phase_scrambled does not mutate its input", np.array_equal(a, before))


def test_phase_scramble_power_spectrum_is_measured_not_claimed():
    """The half of the P-RAND claim that is NOT exactly true, measured instead of asserted away.

    The pre-registration says P-RAND keeps "every pixel-value marginal and the entire power
    spectrum". Those two fight each other: the un-reranked reconstruction has the magnitude spectrum
    exactly, but re-ranking is a monotone map onto the tile's own values, so it changes which pixel
    holds which value and therefore changes the spectrum. This check neither weakens the placebo nor
    restates the claim as satisfied. It reports the radial relative L1 distance per tile, with the
    DC bin and without it, and it also runs the BRIEF'S OWN device (`rp`: the 100 smallest |FFT|^2
    values, rtol 1e-6) on every tile and prints its verdict, so the assertion the brief wrote is on
    the record rather than replaced by a metric of my choosing.

    Measured, at seed 7, 32 radial bins:

      tile      brief's rp() rtol=1e-6   radial rel L1 with DC   radial rel L1 without DC
      blobs     FAIL                     7.157e-04               0.74678
      coarse16  FAIL                     1.381e-03               0.67506
      fine3     FAIL                     2.374e-04               0.65611
      noise     FAIL                     2.380e-06               0.15213
      stripe_x  PASS                     0.000e+00               0.00000

    The DC term is exactly preserved (the mean is a function of the multiset) and it dominates the
    power sum, which is why the with-DC numbers look small. Stripped of DC the perturbation is 15%
    to 75% of the spectrum. On the striped tile the re-ranking happens to stay row-constant, so both
    readings are zero there: a property of that tile, not a general guarantee, and the four others
    say so. The pixel multiset, by contrast, is exact on all five.
    """
    tiles = {"blobs": blobs(), "noise": grain_noise(), "coarse16": coarse_blocks(),
             "fine3": fine_grains(), "stripe_x": striped()}

    def rp(img):
        """The brief's own spectrum device, verbatim: the 100 smallest squared-modulus values."""
        f = np.abs(np.fft.fft2(np.asarray(img, float).mean(axis=2))) ** 2
        return np.sort(f.ravel())[:100]

    perturbed, preserved = [], []
    for name, a in sorted(tiles.items()):
        s = sf.phase_scrambled(a, seed=7)
        full, nodc = spectrum_distance(a, s)
        brief_ok = bool(np.allclose(rp(a), rp(s), rtol=1e-6))
        check("%s: both radial readings are finite and inside [0, 1]" % name,
              np.isfinite(full) and np.isfinite(nodc) and 0.0 <= full <= 1.0 and 0.0 <= nodc <= 1.0,
              "with DC=%.6e, without DC=%.6f" % (full, nodc))
        print("INFO   P-RAND spectrum | %-9s brief rp() rtol=1e-6 %s | rel L1 with DC %.6e | "
              "without DC %.6f" % (name, "PASS" if brief_ok else "FAIL", full, nodc))
        (preserved if nodc <= 1e-6 else perturbed).append(name)
    check("the spectrum is NOT preserved on the four tiles whose scramble breaks the structure",
          sorted(perturbed) == ["blobs", "coarse16", "fine3", "noise"], str(perturbed))
    check("the striped tile is the measured exception: its scramble is still row constant",
          preserved == ["stripe_x"], str(preserved))
    check("so the pre-registration's 'entire power spectrum' wording is false as a guarantee",
          len(perturbed) > 0, "%d of 5 tiles lose their spectrum" % len(perturbed))
    check("while the pixel multiset is exact on all five, so the other half of the claim holds",
          all(np.array_equal(np.sort(a.ravel()), np.sort(sf.phase_scrambled(a, seed=7).ravel()))
              for a in tiles.values()))
    check("and the brief's rp() device agrees with the radial measure on all five tiles",
          all(np.allclose(rp(a), rp(sf.phase_scrambled(a, seed=7)), rtol=1e-6) == (n in preserved)
              for n, a in sorted(tiles.items())),
          "rp() passes exactly on the tiles the radial measure calls preserved")


# Fix round 2: test_column_shuffle_keeps_marginals_and_breaks_joint was deleted here. Its three
# checks (per-column marginals preserved, planted correlation destroyed, permutation vectors
# reconstruct the shuffle) were repeated verbatim, on the same matrix and the same seed, inside
# test_shuffle_columns_gives_a_reconstructable_control below, which additionally checks that the
# vectors are row-index permutations, that the shuffle is per-column rather than one shared
# permutation, that the input is not mutated and that the per-column median is inert to it. Keeping
# one copy keeps the count of what is asserted honest.


def test_placebos_are_seeded_and_reproducible():
    a = blobs()
    one = sf.phase_scrambled(a, seed=7)
    two = sf.phase_scrambled(a, seed=7)
    check("same seed gives a bit-identical scramble", np.array_equal(one, two))
    check("a different seed gives a different scramble",
          not np.array_equal(one, sf.phase_scrambled(a, 8)))
    M = np.random.default_rng(3).normal(size=(16, 4))
    check("same seed gives a bit-identical column shuffle",
          np.array_equal(sf.shuffle_columns(M, seed=5)[0], sf.shuffle_columns(M, seed=5)[0]))
    check("a different seed gives a different column shuffle",
          not np.array_equal(sf.shuffle_columns(M, seed=5)[0], sf.shuffle_columns(M, seed=6)[0]))
    C = sf.linear_combinations(np.random.default_rng(2).normal(size=(8, 12)), seed=13)
    check("same seed gives bit-identical linear combinations",
          np.array_equal(C, sf.linear_combinations(
              np.random.default_rng(2).normal(size=(8, 12)), 13)))


def test_shuffle_columns_gives_a_reconstructable_control():
    rng = np.random.default_rng(3)
    M = rng.normal(size=(20, 4))
    M[:, 1] = M[:, 0] * 2.0
    before = M.copy()
    S, vecs = sf.shuffle_columns(M, seed=11)
    check("shuffled columns have identical per-column values",
          all(np.allclose(np.sort(M[:, j]), np.sort(S[:, j])) for j in range(4)))
    check("the planted correlation is destroyed",
          abs(np.corrcoef(S[:, 0], S[:, 1])[0, 1]) < 0.5,
          "corr=%.6f" % abs(np.corrcoef(S[:, 0], S[:, 1])[0, 1]))
    ok = all(np.allclose(M[vecs[j], j], S[:, j]) for j in range(M.shape[1]))
    check("the permutation vectors reconstruct the shuffle", bool(ok))
    check("every returned vector is a permutation of the row indices",
          vecs.shape == (M.shape[1], M.shape[0]) and
          all(sorted(vecs[j].tolist()) == list(range(M.shape[0])) for j in range(4)),
          "vecs %s for a %s matrix: one row index vector per column" % (vecs.shape, M.shape))
    check("the shuffle is per-column, not one shared row permutation",
          not np.array_equal(vecs[0], vecs[1]))
    check("shuffle_columns does not mutate its input matrix", np.array_equal(M, before))
    check("the per-column median, which is what the pipeline takes, is inert to the shuffle",
          all(np.median(S[:, j]) == np.median(M[:, j]) for j in range(4)))
    check("a 1-D matrix is an invalid call", isinstance(caught(sf.shuffle_columns, M[:, 0], 1),
          ValueError), type(caught(sf.shuffle_columns, M[:, 0], 1)).__name__)


def test_aggregation_is_two_plain_medians_like_the_frozen_pipeline():
    rows = [{"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0},
            {"A1": 5.0, "A2": 6.0, "A3": 7.0, "A4": 8.0}]
    check("image_block medians over tiles", sf.image_block(rows)["A1"] == 3.0,
          str(sf.image_block(rows)))
    check("soil_block medians over images", sf.soil_block([rows[0], rows[1]])["A2"] == 4.0,
          str(sf.soil_block([rows[0], rows[1]])))
    check("both aggregations return exactly the A1-A4 column set, in order",
          tuple(sf.image_block(rows)) == sf.SPATIAL and tuple(sf.soil_block(rows)) == sf.SPATIAL,
          str(tuple(sf.image_block(rows))))
    three = [{"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0},
             {"A1": 5.0, "A2": 6.0, "A3": 7.0, "A4": 8.0},
             {"A1": 9.0, "A2": 10.0, "A3": 11.0, "A4": 12.0}]
    check("an odd-length median is the middle value, not a mean of two",
          sf.image_block(three)["A1"] == 5.0, str(sf.image_block(three)))
    # The symmetric cases above cannot tell a median from a mean: 1,5 -> 3 either way, and
    # 1,5,9 -> 5 either way. The frozen pipeline's aggregation is a per-column MEDIAN
    # (model_spec_m7a.md:23), so a mean would be a methodology substitution that no balanced
    # test row can see. Found by mutation testing: replacing nanmedian with nanmean escaped every
    # other check in this file.
    skewed = [{"A1": 1.0, "A2": 1.0, "A3": 1.0, "A4": 1.0},
              {"A1": 2.0, "A2": 2.0, "A3": 2.0, "A4": 2.0},
              {"A1": 100.0, "A2": 100.0, "A3": 100.0, "A4": 100.0}]
    check("the aggregation is a MEDIAN, not a mean: a skewed column gives the middle value",
          sf.image_block(skewed)["A1"] == 2.0 and sf.soil_block(skewed)["A1"] == 2.0,
          "image_block %.3f (median 2.0, mean would be 34.333)" % sf.image_block(skewed)["A1"])
    check("and it survives an outlier in the same direction a median must",
          sf.image_block(skewed + [{"A1": 1e6, "A2": 1.0, "A3": 1.0, "A4": 1.0}])["A1"] == 51.0,
          "four rows 1, 2, 100 and 1e6: median 51.0")
    check("the aggregation is order-free, as a median must be",
          sf.image_block(three) == sf.image_block(list(reversed(three))))
    mixed = [{"A1": 1.0, "A2": 2.0, "A3": 3.0, "A4": 4.0},
             {"A1": float("nan"), "A2": 6.0, "A3": 7.0, "A4": 8.0},
             {"A1": 1.0, "A2": 6.0, "A3": 7.0, "A4": 8.0},
             {"A1": 1.0, "A2": 6.0, "A3": 7.0, "A4": 8.0}]
    out, err, got = noisy(sf.image_block, mixed)
    check("one NaN tile does not poison its column", err is None and out["A1"] == 1.0, str(out))
    check("a partial NaN column emits no warning", not got, str(got)[:70])
    allnan = [{"A1": float("nan"), "A2": 2.0, "A3": 3.0, "A4": 4.0},
              {"A1": float("nan"), "A2": 6.0, "A3": 7.0, "A4": 8.0}]
    out2, err2, got2 = noisy(sf.image_block, allnan)
    check("an all-NaN column aggregates to NaN, never to a fallback number",
          err2 is None and np.isnan(out2["A1"]), str(out2))
    check("that all-NaN column is REPORTED at the one site that produces it",
          len(got2) == 1 and got2[0].startswith("RuntimeWarning") and "A1" in got2[0],
          str(got2)[:90])
    check("the notice names only the dead column", all("A2" not in w and "A3" not in w and "A4"
          not in w for w in got2), str(got2)[:90])
    check("soil_block reports an all-NaN column the same way",
          len(noisy(sf.soil_block, allnan)[2]) == 1, str(noisy(sf.soil_block, allnan)[2])[:90])
    check("no notice is emitted when no column is all-NaN", not noisy(sf.image_block, three)[2],
          str(noisy(sf.image_block, three)[2])[:60])
    exc = caught(sf.image_block, [])
    check("aggregating zero rows is an invalid call, not an empty median",
          isinstance(exc, ValueError), "%s | %s" % (type(exc).__name__, str(exc)[:50]))


def test_linear_combination_control_has_no_new_information():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(24, 12))
    before = X.copy()
    C = sf.linear_combinations(X, seed=13)
    check("P-COLS returns 4 columns", C.shape == (24, 4), str(C.shape))
    check("every P-COLS column lies in the span of the original 12",
          all(np.linalg.matrix_rank(np.column_stack([X, C[:, j]])) == 12 for j in range(4)))
    check("P-COLS is reproducible from its seed", np.array_equal(sf.linear_combinations(X, 13), C))
    check("a different seed gives a different mixture",
          not np.allclose(sf.linear_combinations(X, 14), C))
    W = np.linalg.lstsq(X, C, rcond=None)[0]
    check("the recovered mixture weights have unit norm, so the four columns are comparable",
          np.allclose(np.linalg.norm(W, axis=0), 1.0, atol=1e-9),
          str(np.round(np.linalg.norm(W, axis=0), 9)))
    check("P-COLS is not a copy of four existing columns",
          not any(np.allclose(C[:, j], X[:, j]) for j in range(4)))
    check("linear_combinations does not mutate its input", np.array_equal(X, before))
    check("a 1-D matrix is an invalid call",
          isinstance(caught(sf.linear_combinations, X[:, 0], 13), ValueError),
          type(caught(sf.linear_combinations, X[:, 0], 13)).__name__)


def test_placebos_move_the_statistics_the_controls_claim_to_move():
    """Brief Step 5's sanity probe, run as a check so the numbers cannot be forgotten.

    The plan predicts the scrambled vector "differs from the real one on A1/A2/A4 while A3 stays
    close". Measured on the blobs tile, the first half holds (A1 by 0.013012, A2 by 0.630805, A4 by
    0.008837, each far above the 1e-9 bit-noise floor) and the second does not: A3 goes from
    0.398438 to 0.046875, a move of 45 of its own 0.0078125 quantisation steps. That is the same
    fact as the spectrum measurement above, seen from the other end - re-ranking perturbs the power
    spectrum by 75% once the DC bin is excluded, so a statistic built on the spatial covariance has
    no reason to stay put. The check records the measured numbers rather than the prediction, and
    task-2-report.md carries the deviation.
    """
    a = blobs()
    real = sf.tile_spatial(a, PPM)
    scr = sf.tile_spatial(sf.phase_scrambled(a, seed=7), PPM)
    print("INFO   P-RAND probe | real      "
          + " ".join("%s=%.6f" % (k, real[k]) for k in sf.SPATIAL))
    print("INFO   P-RAND probe | scrambled "
          + " ".join("%s=%.6f" % (k, scr[k]) for k in sf.SPATIAL))
    check("the scrambled tile is computable and gives no all-NaN row",
          all(not np.isnan(scr[k]) for k in sf.SPATIAL), str(scr))
    check("the scramble moves A1, A2 and A4, as the plan predicts",
          all(abs(scr[k] - real[k]) > 1e-3 for k in ("A1", "A2", "A4")),
          "dA1=%.6f dA2=%.6f dA4=%.6f" % tuple(abs(scr[k] - real[k]) for k in ("A1", "A2", "A4")))
    check("A3 does NOT stay close, contrary to the plan: recorded as a deviation, not smoothed",
          abs(scr["A3"] - real["A3"]) > 0.05,
          "real %.6f scrambled %.6f = %.1f quantisation steps"
          % (real["A3"], scr["A3"], abs(scr["A3"] - real["A3"]) / 0.0078125))
    check("the scrambled vector differs from the real one",
          any(scr[k] != real[k] for k in sf.SPATIAL))
    sh = sf.tile_spatial(sf.phase_scrambled(a, seed=11), PPM)
    check("a second scramble seed gives a different placebo row, so P-RAND is a real arm",
          any(sh[k] != scr[k] for k in sf.SPATIAL), str(sh))


def test_no_module_call_emits_a_warning():
    """The statistic path is silent AND it returns: an exception captured here is a failure, not a
    reason to look only at the warning list. Fix round 2 adds the missing err condition, which the
    pre-fix version discarded - a raise on a tile would still have printed PASS."""
    tiles = {"striped": striped(), "blobs": blobs(), "noise": grain_noise(),
             "coarse16": coarse_blocks(), "fine3": fine_grains(), "ramp": ramp(),
             "speck5": speck_on_dark(), "band": thin_band(), "flat6": flat_band(6),
             "band3": mask_tile((20, 20), [(i, j) for i in range(8, 11) for j in range(20)])}
    for name, a in sorted(tiles.items()):
        out, err, got = noisy(sf.tile_spatial, a, PPM)
        check("tile_spatial on %s neither warns nor raises" % name, not got and err is None,
              "%s | %s" % ("; ".join(got)[:50], str(err)[:50]))
        check("tile_spatial on %s returns the four keys, with A1 finite" % name,
              err is None and tuple(out) == sf.SPATIAL and np.isfinite(out["A1"]),
              str(out)[:66])


def test_warning_policy_is_in_force():
    """Proves the runner's filter is real: an ordinary RuntimeWarning arrives as an exception.

    np.nanmean of an empty slice is the exact call that made A2 noisy before ruling 2. If
    main()'s simplefilter("error") were ever removed this check fails, so the policy cannot decay
    into a comment.
    """
    exc = caught(np.nanmean, np.asarray([]))
    check("the runner turns RuntimeWarning into an error",
          exc is not None and isinstance(exc, RuntimeWarning), type(exc).__name__)


def test_no_scale_rederivation_or_data_access_in_source():
    """The module's leakage surface is its argument list: source-scan the prohibitions.

    Scanned over CODE ONLY (docstrings and comments stripped): the module says in prose that it
    reads no labels, camera or manifest, and prose is not a surface. Calls and imports are.
    """
    src = code_only_text(Path(__file__).resolve().parent / "spatial_features.py")
    text = src.lower()
    for banned in ("_ppm_from_extent", "56.23", "read_csv", "manifest", "pandas",
                   "image.open", "data/", "label", "camera", "open("):
        check("module code contains no %s" % banned, banned.lower() not in text)
    check("mask_and_field takes ppm as a required argument",
          "def mask_and_field(a: np.ndarray, ppm: float)" in src
          or "def mask_and_field(a, ppm)" in src)
    check("tile_spatial takes ppm as a required argument",
          "def tile_spatial(a: np.ndarray, ppm: float)" in src)
    # The frozen literals that are written inline rather than as named constants, and that no value
    # test can always reach: replacing A4's MAD with a mean absolute deviation moves the fraction by
    # less than the regime floor, and the l2 floor is a literal no-op on the only tile that reaches
    # it (measured raw l2 = exactly 0.0 on the ramp). Those definitions are pinned as text because
    # no tile can exercise them, which is stated rather than glossed.
    #
    # Deleted here in fix round 2, because each was a pure-text pin duplicating the value assertions
    # of test_frozen_parameter_table_is_implemented, and each is now asserted as behaviour instead:
    #   the four threshold NAMES ("MIN_PAIRS_PER_LAG" in src, and its three siblings) were satisfied
    #     by the constant's own definition line however the code used it, and are replaced by
    #     test_each_frozen_floor_gates_its_own_statistic;
    #   "thr = 1.0 / np.e" was whitespace-exact, and is replaced by the boundary probes in
    #     test_efold_crossing_is_the_registered_one_over_e;
    #   the two OCCURRENCE COUNTS (== 1) are replaced by test_named_constants_are_the_ones_the_
    #     statistics_read, which shows the arithmetic reads the named constant and that A4's MAD is
    #     computed once per tile, both of which are the properties the counts were standing in for.
    for literal, label in (("A3_SILL_FRACTION * sill", "A3's sill crossing threshold"),
                           ("np.percentile(g, 12)", "the frozen 12th-percentile mask"),
                           ("BASE_SIGMA_MM * ppm",
                            "the 12 mm illumination base at the caller's ppm"),
                           ("A4_LAG_MM * ppm", "A4's lag in px from the caller's ppm"),
                           ("A3_REFERENCE_LAG * extent_mm", "A3's half-extent reference"),
                           ("for ax in (0, 1):", "A3's horizontal+vertical variogram, no diagonal"),
                           ("(l2 + EPS) / (l1 + EPS)", "A1's log eigenvalue ratio"),
                           ("max(tr / 2.0 - np.sqrt(disc), 0.0)", "A1's non-negative eigenvalue"),
                           ("np.median(np.abs(g[m] - np.median(g[m])))", "A4's MAD threshold"),
                           ("A4_DIAG_PX", "A4's 3 px per-axis diagonal offset")):
        check("frozen inline literal: %s" % label, literal in src)
    check("the finiteness guard is present",
          "np.isinf" in src and "np.isnan" in src)


PLACEBO_FNS = ("phase_scrambled", "shuffle_columns", "linear_combinations")


def test_rng_is_confined_to_the_placebo_functions():
    """The statistics must be deterministic given a tile, so no RNG may appear in their code.

    Task 2's placebo constructors are seeded, explicitly, and take seed as a required argument:
    that is the P-RAND and P-COLS definition, not a leak. The Task 1 scan banned "random" and
    "seed" across the whole file, which the placebos legitimately trip, so the ban is re-scoped to
    the statistic side and the placebo side is pinned to an explicit seed instead. The guarantee
    ("no hidden state in a statistic") is unchanged; only its scope is now accurate.
    """
    fns = dict(inspect.getmembers(sf, inspect.isfunction))
    fns = {n: f for n, f in fns.items() if f.__module__ == "spatial_features"}
    statistic_fns = sorted(n for n in fns if n not in PLACEBO_FNS)
    for name in statistic_fns:
        body = inspect.getsource(fns[name]).lower()
        check("statistic-side %s contains no random, seed or rng" % name,
              "random" not in body and "seed" not in body and "rng" not in body)
    check("no statistic-side function is a placebo name by accident",
          not any(n in PLACEBO_FNS for n in statistic_fns), str(statistic_fns)[:80])


def code_only_text(path):
    """Module source with docstrings and comments removed, so the scan below reads CODE.

    Without the strip the scan would trip on the module's own docstring, which says in prose that
    it reads no labels, no camera and no manifest. Prose is not a surface; calls and imports are.
    """
    src = path.read_text(encoding="utf-8")
    drop = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = list(getattr(node, "body", []))
            head = body[0] if body else None
            if (isinstance(head, ast.Expr) and isinstance(head.value, ast.Constant)
                    and isinstance(head.value.value, str)):
                drop.update(range(head.lineno, head.end_lineno + 1))
    return "\n".join(line for i, line in enumerate(src.splitlines(), 1)
                     if i not in drop and not line.lstrip().startswith("#"))


def test_every_check_passed():
    """Pytest bridge, and it must stay the last test_ function in the file.

    check() prints instead of asserting, which is the project convention and is what makes a check
    visible, but it also means pytest would report this file green while FAIL is non-empty. This
    one function asserts on the accumulated list, so a green pytest run means a green contract.
    The runner below sorts by definition line, so it still runs last there too. The detail string
    counts TOTAL + 1 because this check's own line is the one being printed: the Task 1 version
    reported "0 of 104 failed" under a summary that said 105 checks.
    """
    check("all checks above passed", not FAIL, "%d of %d failed" % (len(FAIL), TOTAL + 1))
    assert not FAIL, "%d check(s) failed: %s" % (len(FAIL), ", ".join(FAIL[:6]))


def main():
    fns = []
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            fns.append((obj.__code__.co_firstlineno, name, obj))
    fns.sort()
    if "--golden" in sys.argv:
        golden_reference()
        return 0
    for _, name, fn in fns:
        try:
            with warnings.catch_warnings():
                # Ruling 2: a warning emitted during any check is a failure of that check. Scoped to
                # each test so a warning cannot escape into the runner's own bookkeeping.
                warnings.simplefilter("error")
                fn()
        except Exception as exc:  # a test that crashes is a failure, not an aborted run
            check("%s raised %s" % (name, type(exc).__name__), False, str(exc)[:90])
    print("\n%d checks, %d failed" % (TOTAL, len(FAIL)))
    for f in FAIL:
        print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
