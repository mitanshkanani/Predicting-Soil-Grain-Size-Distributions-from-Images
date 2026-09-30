"""Validate the four threshold-gating tile designs and capture before-numbers."""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_features as sf

PPM = 4.552516
warnings.simplefilter("ignore")


def mask_tile(shape, pixels, order=0.0):
    h, w = shape
    a = np.full((h, w, 3), 10.0)
    for n, (i, j) in inenumerate(pixels):
        a[i, j] = 200.0 + 0.5 * j + 3.0 * i + 0.01 * n
    return a


def inenumerate(seq):
    return enumerate(seq)


def lag_counts(m, k):
    out = []
    for ax in (0, 1):
        sl_a = [slice(None)] * 2
        sl_a[ax] = slice(None, -k)
        sl_b = [slice(None)] * 2
        sl_b[ax] = slice(k, None)
        out.append(int(np.count_nonzero(m[tuple(sl_a)] & m[tuple(sl_b)])))
    return out


def stats(a, label):
    g, m, e = sf.mask_and_field(a, PPM)
    out = sf.tile_spatial(a, PPM)
    print("%-30s mask=%4d A1=%.4f A2=%.6g A3=%.6g A4=%.6g" %
          (label, int(m.sum()), out["A1"], out["A2"], out["A3"], out["A4"]))
    return g, m, e, out


print("== MIN_MASK_PIXELS: exactly 16 vs 15 pixels ==")
for n in (15, 16, 17):
    px = [(i, j) for i in range(4) for j in range(4)][:n]
    a = mask_tile((20, 20), px)
    try:
        g, m = sf.mask_and_field(a, PPM)[:2]
        print("  %2d px -> mask=%d, tile valid" % (n, int(m.sum())))
        stats(a, "  %2d px" % n)
    except sf.TileDegenerate as exc:
        print("  %2d px -> TileDegenerate: %s" % (n, str(exc)[:60]))

print("\n== MIN_PROFILE_LINES: masked rows 3 vs 4 ==")
for nrows in (3, 4, 5):
    px = [(i, j) for i in range(8, 8 + nrows) for j in range(20)]
    g, m, e, out = stats(mask_tile((20, 20), px), "band rows=%d" % nrows)
    for axis in (1, 0):
        prof, lines = sf._masked_profile(e, m, axis)
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        print("      axis=%d lines=%d energy=%.6g" % (axis, lines, float(np.dot(p, p))))

print("\n== MIN_PAIRS_PER_DIRECTION: rows {0,3,5} x L cols ==")
lag = int(round(sf.A4_LAG_MM * PPM))
for L in (10, 11, 12):
    px = [(9 + off, 3 + j) for off in (0, 3, 5) for j in range(L)]
    g, m, e, out = stats(mask_tile((20, 20), px), "run L=%d" % L)
    counts = {}
    for name, (oy, ox) in (("0deg", (0, lag)), ("90deg", (lag, 0)),
                           ("45deg", (sf.A4_DIAG_PX, sf.A4_DIAG_PX)),
                           ("135deg", (sf.A4_DIAG_PX, -sf.A4_DIAG_PX))):
        sl_a, sl_b = sf._pair_slices(m.shape, oy, ox)
        counts[name] = int(np.count_nonzero(m[sl_a] & m[sl_b]))
    print("      direction pairs:", counts)

print("\n== _direction_contrast direct at exactly 8 vs 7 pairs ==")
m = np.zeros((24, 24), bool)
g = np.arange(24 * 24, dtype=float).reshape(24, 24)
for ncol in (7, 8, 9):
    mm = np.zeros((24, 24), bool)
    mm[2, :ncol] = True
    mm[2 + lag, :ncol] = True
    mad = float(np.median(np.abs(g[mm] - np.median(g[mm])))) + sf.EPS
    exc = None
    try:
        val, n = sf._direction_contrast(g, mm, 0, lag, mad)
    except Exception as got:
        val, n, exc = None, None, got
    print("   run=%d pairs(ax0 at lag %d)=%d -> %s %s" %
          (ncol, lag, lag_counts(mm, lag)[0], ("%.6f" % val) if val is not None else type(exc).__name__,
           str(exc)[:50] if exc else "(n=%d)" % n))

print("\n== MIN_PAIRS_PER_LAG: 10x10 block vs trimmed row ==")
full = [(i, j) for i in range(9, 19) for j in range(10)]
for label, px in (("block 10x10", full), ("block, row 18 -> 7 px", [p for p in full if p[0] != 18 or p[1] < 7]),
                  ("block, row 18 -> 8 px", [p for p in full if p[0] != 18 or p[1] < 8])):
    g, m, e, out = stats(mask_tile((20, 20), px), label)
    gam = sf._variogram(g, m)
    print("      lag9 counts=%s gam=%s" % (lag_counts(m, 9),
                                           np.array2string(gam, precision=3, separator=",")))

print("\n== Finding 2 tiles: flat bands, tile-level ==")
for nrows in (6, 8):
    t = np.full((64, 64, 3), 10.0)
    t[20:20 + nrows, :] = 200.0
    g, m, e = sf.mask_and_field(t, PPM)
    out = sf.tile_spatial(t, PPM)
    en = []
    for axis in (1, 0):
        prof, lines = sf._masked_profile(e, m, axis)
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        en.append((axis, lines, float(np.dot(p, p))))
    print("  flat band %d rows: mask=%d A2=%r A1=%.4f A3=%.4f A4=%.4f | %s" %
          (nrows, int(m.sum()), out["A2"], out["A1"], out["A3"], out["A4"], en))

print("\n== existing tiles: A3 must not move ==")
import check_spatial_features as csf
for name, a in (("blobs", csf.blobs()), ("stripe_x", csf.striped()), ("noise", csf.grain_noise()),
                ("coarse16", csf.coarse_blocks()), ("fine3", csf.fine_grains()),
                ("ramp", csf.ramp()), ("stripe_y", np.transpose(csf.striped(), (1, 0, 2)).copy())):
    g, m, e = sf.mask_and_field(a, PPM)
    out = sf.tile_spatial(a, PPM)
    single = sum(1 for k in range(1, sf._variogram(g, m).size + 1)
                 if (lambda c: (c[0] >= 8) != (c[1] >= 8))(lag_counts(m, k)))
    print("  %-9s A3=%.9f single_dir_lags=%d A2=%.6f" % (name, out["A3"], single, out["A2"]))
