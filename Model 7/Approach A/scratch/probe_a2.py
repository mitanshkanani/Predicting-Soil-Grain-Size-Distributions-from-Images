"""A2 energy diagnostics: which directions have exactly zero profile variance today."""
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
    "stripe_y": np.transpose(csf.striped(), (1, 0, 2)).copy(),
    "noise": csf.grain_noise(),
    "coarse16": csf.coarse_blocks(),
    "fine3": csf.fine_grains(),
    "ramp": csf.ramp(),
    "thin_band": csf.thin_band(),
    "speck5": csf.speck_on_dark(),
}

print("%-10s %-8s %-7s %-7s %-10s %-10s %-10s" %
      ("tile", "axis", "lines", "energy", "acf[0:3]", "efold", "A2"))
for name, a in TILES.items():
    g, m, e = sf.mask_and_field(a, PPM)
    a2 = sf._a2(e, m)
    for axis in (1, 0):
        prof, lines = sf._masked_profile(e, m, axis)
        keep = np.isfinite(prof)
        p = np.where(keep, prof - float(prof[keep].mean()), 0.0)
        energy = float(np.dot(p, p))
        ac = sf._acf_along(e, m, axis)
        ef = sf._efold(ac)
        print("%-10s %-8s %-7d %-7.3g %-10s %-10.6f" %
              (name, "0deg" if axis == 1 else "90deg", lines, energy,
               np.array2string(ac[:3], precision=4, separator=","), ef))
    print("%-10s A2 = %.12f" % (name, a2))

print()
print("zeros probe _a2:", sf._a2(np.zeros((64, 64)), np.ones((64, 64), dtype=bool)))
print("constant nonzero field:", sf._a2(np.full((64, 64), 5.0), np.ones((64, 64), dtype=bool)))
print("one nonzero entry:", sf._a2(np.eye(64), np.ones((64, 64), dtype=bool)))
