"""Before/after diagnostics for the fix round. Not part of the contract suite."""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spatial_features as sf
import check_spatial_features as csf

warnings.simplefilter("ignore")
PPM = csf.PPM

TILES = {
    "blobs": csf.blobs(),
    "stripe_x": csf.striped(),
    "noise": csf.grain_noise(),
    "coarse16": csf.coarse_blocks(),
    "fine3": csf.fine_grains(),
    "ramp": csf.ramp(),
    "thin_band": csf.thin_band(),
    "speck5": csf.speck_on_dark(),
}


def lag_breakdown(gam, g, m):
    """Recount, per lag index, how many of the two directions clear MIN_PAIRS_PER_LAG."""
    n_both = n_single = n_none = 0
    single_idx = []
    for i, k in enumerate(range(1, gam.size + 1)):
        counts = []
        for ax in (0, 1):
            sl_a = [slice(None)] * 2
            sl_a[ax] = slice(None, -k)
            sl_b = [slice(None)] * 2
            sl_b[ax] = slice(k, None)
            counts.append(int(np.count_nonzero(m[tuple(sl_a)] & m[tuple(sl_b)])))
        ok = [c for c in counts if c >= sf.MIN_PAIRS_PER_LAG]
        if len(ok) == 2:
            n_both += 1
        elif len(ok) == 1:
            n_single += 1
            single_idx.append(i)
        else:
            n_none += 1
    return n_both, n_single, n_none, single_idx


print("== LAG BREAKDOWN (finding 1) ==")
for name, a in TILES.items():
    g, m = sf.mask_and_field(a, PPM)[:2]
    gam = sf._variogram(g, m)
    both, single, none, idx = lag_breakdown(gam, g, m)
    nan_now = int(np.count_nonzero(np.isnan(gam)))
    out = sf.tile_spatial(a, PPM)
    print("%-10s lags=%3d both=%3d single=%3d none=%3d nan_today=%3d first_single=%s A3=%.6f"
          % (name, gam.size, both, single, none, nan_now,
             (idx[0] + 1) if idx else "-", out["A3"]))

print("\n== SILL WINDOW / CROSSING ==")
for name, a in TILES.items():
    g, m = sf.mask_and_field(a, PPM)[:2]
    gam = sf._variogram(g, m)
    win = max(1, int(sf.A3_SILL_WINDOW * gam.size))
    tail = gam[-win:]
    finite = tail[np.isfinite(tail)]
    sill = float(finite.mean()) if finite.size else float("nan")
    reached = np.isfinite(gam) & (gam >= sf.A3_SILL_FRACTION * sill) if finite.size else None
    hit = int(np.argmax(reached)) if reached is not None and reached.any() else None
    nan_before_hit = int(np.count_nonzero(np.isnan(gam[:hit]))) if hit is not None else None
    print("%-10s win=%2d tail_finite=%2d nan_before_cross=%s cross_lag=%s"
          % (name, win, int(finite.size), nan_before_hit, (hit + 1) if hit is not None else None))

print("\n== A2 ZERO ENERGY (finding 2) ==")
z = np.zeros((64, 64))
full = np.ones((64, 64), dtype=bool)
print("_a2(zeros, all-True mask) =", repr(sf._a2(z, full)))
p, lines = sf._masked_profile(z, full, 1)
print("  profile lines:", lines, "dot(p,p) with p=prof-mean:",
      float(np.dot(p - p.mean(), p - p.mean())))
# a constant island inside a real tile: the mask covers only pixels of one value
flat = sf.mask_and_field(np.full((64, 64, 3), 120.0) + np.random.default_rng(0).normal(
    0, 1, (64, 64, 3)), PPM)
print("_a2 on a random tile:", repr(sf._a2(flat[2], flat[1])))

print("\n== AGGREGATION NON-FINITE (finding 3) ==")


def probe(fn, rows, label):
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        try:
            res = fn(rows)
            err = None
        except Exception as exc:
            res, err = None, exc
    print("%-28s -> %s | raised=%s | warnings=%s"
          % (label, res, type(err).__name__ if err else None,
             [w.category.__name__ + ": " + str(w.message)[:40] for w in got]))


def rows_of(vals):
    return [{"A1": v, "A2": 2.0, "A3": 3.0, "A4": 4.0} for v in vals]


probe(sf.image_block, rows_of([float("inf"), float("inf")]), "image_block [inf, inf]")
probe(sf.image_block, rows_of([float("inf"), 2.0]), "image_block [inf, 2.0]")
probe(sf.soil_block, rows_of([float("inf"), float("inf")]), "soil_block [inf, inf]")
probe(sf.image_block, rows_of([float("nan"), float("nan")]), "image_block [nan, nan]")
probe(sf.image_block, rows_of([float("-inf"), 2.0]), "image_block [-inf, 2.0]")
