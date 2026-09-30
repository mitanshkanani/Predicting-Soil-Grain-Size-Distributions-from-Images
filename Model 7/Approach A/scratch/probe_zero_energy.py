"""Is a tile-level zero-energy A2 direction reachable? Probe x/y constancy of the residual."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_features as sf
import check_spatial_features as csf

PPM = csf.PPM

# g constant along x (band/stripes): is e constant along x inside the mask?
for name, a in (("striped", csf.striped()), ("y_ramp_64", None)):
    if a is None:
        col = np.linspace(0.0, 255.0, 64)[None, :].T
        g = np.broadcast_to(col, (64, 64)).copy()
        a = np.dstack([g, g, g])
    g, m, e = sf.mask_and_field(a, PPM)
    prof_x, lx = sf._masked_profile(e, m, 1)
    prof_y, ly = sf._masked_profile(e, m, 0)
    def energy(prof):
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        return float(np.dot(p, p))
    print("%-10s x-const=%s e-range-across-cols=%s" %
          (name, np.allclose(e, e[:, :1]), float(np.abs(e - e[:, :1]).max())))
    print("   0deg lines=%d energy=%.6g | 90deg lines=%d energy=%.6g | A2=%.6g" %
          (lx, energy(prof_x), ly, energy(prof_y), sf._a2(e, m)))

# A flat horizontal band of one constant value inside a larger tile.
for band in (6, 8, 20):
    t = np.full((64, 64, 3), 10.0)
    t[20:20 + band, :] = 200.0
    g, m, e = sf.mask_and_field(t, PPM)
    prof_x, lx = sf._masked_profile(e, m, 1)
    prof_y, ly = sf._masked_profile(e, m, 0)
    def energy(prof):
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        return float(np.dot(p, p))
    print("flat band %d rows: mask=%d px, 0deg lines=%d energy=%.6g, 90deg lines=%d energy=%.6g,"
          " A2=%r" % (band, int(m.sum()), lx, energy(prof_x), ly, energy(prof_y), sf._a2(e, m)))

# A flat square island.
for side in (8, 16, 24):
    t = np.full((64, 64, 3), 10.0)
    t[20:20 + side, 20:20 + side] = 200.0
    g, m, e = sf.mask_and_field(t, PPM)
    print("flat island %d: A2=%r" % (side, sf._a2(e, m)))

# Direct helper constructions with an explicit mask (no mask_and_field).
cases = {
    "zeros+allTrue": (np.zeros((64, 64)), np.ones((64, 64), bool)),
    "const+allTrue": (np.full((64, 64), 7.0), np.ones((64, 64), bool)),
    "x-constant only": (np.tile(np.arange(64.0)[:, None], (1, 64)), np.ones((64, 64), bool)),
    "y-constant only": (np.tile(np.arange(64.0)[None, :], (64, 64)), np.ones((64, 64), bool)),
    "single col mask": (np.tile(np.arange(64.0)[:, None], (1, 64)),
                        np.tile(np.array([True] + [False] * 63)[None, :], (64, 64))),
}
for label, (fld, msk) in cases.items():
    out = []
    for axis in (1, 0):
        prof, lines = sf._masked_profile(fld, msk, axis)
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        out.append("axis%d lines=%d energy=%.4g" % (axis, lines, float(np.dot(p, p))))
    try:
        val = repr(sf._a2(fld, msk))
    except Exception as exc:
        val = "%s: %s" % (type(exc).__name__, str(exc)[:60])
    print("%-18s %s | _a2 = %s" % (label, " | ".join(out), val))
