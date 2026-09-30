"""Re-run the two range-search designs with unambiguous per-lag output."""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_features as sf

PPM = 4.552516
warnings.simplefilter("ignore")


def per_lag(g, m):
    gam = sf._variogram(g, m)
    rows = []
    for i, k in enumerate(range(1, gam.size + 1)):
        c = []
        for ax in (0, 1):
            sl_a = [slice(None)] * 2
            sl_a[ax] = slice(None, -k)
            sl_b = [slice(None)] * 2
            sl_b[ax] = slice(k, None)
            c.append(int(np.count_nonzero(m[tuple(sl_a)] & m[tuple(sl_b)])))
        rows.append((k, c[0], c[1], gam[i], np.isnan(gam[i])))
    return rows


print("== A: 19x19, masked even rows, 2-D g ==")
m = np.zeros((19, 19), bool)
m[::2, :] = True
g = 5.0 + 0.7 * np.arange(19)[:, None] + 0.3 * np.arange(19)[None, :]
g = np.broadcast_to(g, (19, 19)).copy()
for k, c0, c1, v, isn in per_lag(g, m):
    print("  lag %2d ax0=%4d ax1=%4d gam=%.6g nan=%s" % (k, c0, c1, v, isn))
win = max(1, int(sf.A3_SILL_WINDOW * (2 * (19 // 2) - 2)))
try:
    print("  _variogram_range ->", sf._variogram_range(g, m, PPM))
except Exception as exc:
    print("  _variogram_range ->", type(exc).__name__, str(exc)[:90])

print("\n== B: 120x200 island in a 256 tile (partial sill window) ==")
g2 = np.arange(256 * 256, dtype=float).reshape(256, 256) / 511.0
m2 = np.zeros((256, 256), bool)
m2[10:130, 20:220] = True
rows = per_lag(g2, m2)
gam = sf._variogram(g2, m2)
win = max(1, int(sf.A3_SILL_WINDOW * gam.size))
print("  lags=%d win=%d (lags %d..%d)" % (gam.size, win, gam.size - win + 1, gam.size))
for k, c0, c1, v, isn in rows[-(win + 3):]:
    print("  lag %3d ax0=%5d ax1=%5d gam=%9.4f nan=%s" % (k, c0, c1, v, isn))
single = [r[0] for r in rows if (r[1] >= sf.MIN_PAIRS_PER_LAG) != (r[2] >= sf.MIN_PAIRS_PER_LAG)]
both_short = [r[0] for r in rows if r[1] < sf.MIN_PAIRS_PER_LAG and r[2] < sf.MIN_PAIRS_PER_LAG]
print("  single-direction lags=%d  both-short=%d  nan_today=%d" %
      (len(single), len(both_short), int(np.isnan(gam).sum())))
try:
    print("  _variogram_range ->", sf._variogram_range(g2, m2, PPM))
except Exception as exc:
    print("  _variogram_range ->", type(exc).__name__, str(exc)[:90])
