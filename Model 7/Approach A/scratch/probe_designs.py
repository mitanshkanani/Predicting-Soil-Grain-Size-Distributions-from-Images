"""Validate the tile/mask designs the new tests will use, and capture before-numbers."""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_features as sf

PPM = 4.552516
warnings.simplefilter("ignore")


def set_mask(shape, pixels, high=200.0, low=10.0):
    """Tile whose mask_and_field mask is exactly `pixels` (set of (i, j))."""
    h, w = shape
    a = np.full((h, w, 3), low)
    for (i, j) in pixels:
        a[i, j] = high + (0.5 * j + 3.0 * i)
    return a


def show(label, a, keys=("A1", "A2", "A3", "A4")):
    g, m, e = sf.mask_and_field(a, PPM)
    out = sf.tile_spatial(a, PPM)
    print("%-34s mask=%4d  A2=%.6g A3=%.6g A4=%.6g  nan=%s" %
          (label, int(m.sum()), out["A2"], out["A3"], out["A4"],
             [k for k in keys if np.isnan(out[k])]))
    for axis in (1, 0):
        prof, lines = sf._masked_profile(e, m, axis)
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        print("     %s lines=%3d energy=%.6g" % ("0deg" if axis == 1 else "90deg", lines,
                                                 float(np.dot(p, p))))
    gam = sf._variogram(g, m)
    print("     variogram size=%d nan_idx=%s" % (gam.size, list(np.flatnonzero(np.isnan(gam)))))
    return g, m, e, out


def counts_at(shape, m, k):
    out = {}
    for ax, name in ((0, "ax0"), (1, "ax1")):
        sl_a = [slice(None)] * 2
        sl_a[ax] = slice(None, -k)
        sl_b = [slice(None)] * 2
        sl_b[ax] = slice(k, None)
        out[name] = int(np.count_nonzero(m[tuple(sl_a)] & m[tuple(sl_b)]))
    return out


print("== MIN_MASK_PIXELS boundary ==")
px16 = [(i, j) for i in range(4) for j in range(4)]
px15 = px16[:15]
t16 = set_mask((64, 64), px16, )
t15 = set_mask((64, 64), px15)
for lbl, t in (("16 px block", t16), ("15 px block", t15)):
    try:
        g, m, e = sf.mask_and_field(t, PPM)
        print("%-12s -> mask ok, n=%d" % (lbl, int(m.sum())))
    except Exception as exc:
        print("%-12s -> %s: %s" % (lbl, type(exc).__name__, str(exc)[:70]))

print("\n== MIN_PROFILE_LINES boundary (tile level, 20x20) ==")
for nrows in (3, 4, 5):
    px = [(i, j) for i in range(8, 8 + nrows) for j in range(20)]
    show("rows band %d" % nrows, set_mask((20, 20), px))

print("\n== MIN_PAIRS_PER_DIRECTION boundary (rows {r,r+3,r+5} x L cols) ==")
for L in (10, 11, 12):
    px = [(9 + off, 4 + j) for off in (0, 3, 5) for j in range(L)]
    a = set_mask((20, 20), px)
    g, m, e = sf.mask_and_field(a, PPM)
    lag = int(round(sf.A4_LAG_MM * PPM))
    c = {}
    for name, (oy, ox) in (("0", (0, lag)), ("90", (lag, 0)),
                           ("45", (sf.A4_DIAG_PX, sf.A4_DIAG_PX)),
                           ("135", (sf.A4_DIAG_PX, -sf.A4_DIAG_PX))):
        sl_a, sl_b = sf._pair_slices((20, 20), oy, ox)
        c[name] = int(np.count_nonzero(m[sl_a] & m[sl_b]))
    out = sf.tile_spatial(a, PPM)
    print("L=%2d pairs=%s A4=%.6g (nan=%s)" % (L, c, out["A4"], np.isnan(out["A4"])))

print("\n== MIN_PAIRS_PER_LAG boundary at the statistic (20x20 block) ==")
full = [(i, j) for i in range(9, 19) for j in range(10)]
cut = [p for p in full if not (p[0] == 18 and p[1] >= 7)]
for lbl, px in (("10x10 block", full), ("row18 trimmed to 7 px", cut)):
    a = set_mask((20, 20), px)
    g, m, e = sf.mask_and_field(a, PPM)
    gam = sf._variogram(g, m)
    print("%-22s mask=%d lag9 counts=%s gam=%s" %
          (lbl, int(m.sum()), counts_at((20, 20), m, 9),
           np.array2string(gam, precision=4, separator=",")))
    show(lbl, a)

print("\n== single-direction lag, hand-built (finding 1 core) ==")
for nv in (7, 8, 9):
    m = np.zeros((40, 60), bool)
    m[3, :nv] = True
    m[8, :nv] = True           # vertical pairs at lag 5 = nv
    m[30, :40] = True          # horizontal pairs at lag 5 = 35, no vertical pairs
    g = np.arange(40 * 60, dtype=float).reshape(40, 60) / 7.0
    gam = sf._variogram(g, m)
    print("n_vert=%2d counts=%s gam[4]=%.6g nan=%s" %
          (nv, counts_at((40, 60), m, 5), gam[4], np.isnan(gam[4])))

print("\n== range search: interior NaN, measured tail (19-row even-row mask) ==")
m = np.zeros((19, 19), bool)
m[::2, :] = True
g = np.tile(np.arange(19.0)[:, None], (1, 19))
gam = sf._variogram(g, m)
print("lags=%d nan_idx=%s gam=%s" % (gam.size, list(np.flatnonzero(np.isnan(gam))),
                                     np.array2string(gam, precision=4, separator=",")))
win = max(1, int(sf.A3_SILL_WINDOW * gam.size))
tail = gam[-win:]
print("win=%d tail=%s finite=%d" % (win, np.array2string(tail, precision=4), int(np.isfinite(tail).sum())))
try:
    print("_variogram_range ->", sf._variogram_range(g, m, PPM))
except Exception as exc:
    print("_variogram_range ->", type(exc).__name__, str(exc)[:80])

print("\n== partial NaN sill window (island 120x200 in a 256 tile) ==")
g = np.arange(256 * 256, dtype=float).reshape(256, 256) / 511.0
m = np.zeros((256, 256), bool)
m[10:130, 20:220] = True
gam = sf._variogram(g, m)
win = max(1, int(sf.A3_SILL_WINDOW * gam.size))
tail = gam[-win:]
print("lags=%d nan total=%d win=%d tail_finite=%d" %
      (gam.size, int(np.isnan(gam).sum()), win, int(np.isfinite(tail).sum())))
print("_variogram_range ->", sf._variogram_range(g, m, PPM))
