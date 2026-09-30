"""Task 5 diagnostic: which parts of the Model 6-3 record does the registered rule reproduce?

Read-only. The record (research_map_1_2_5_6.md lines 189-205) contains FIVE checkable quantities, not
three:
    worst-10 share of summed CV error   61.2 %
    worst single soil share             10.1 %
    median neighbour distance           worst 1.65   best 1.58
    median neighbour curve gap          worst 65.7   best 28.1
    the H038/G190 pair                  distance 1.32, gap 125.4 EMD
This script prints what the registered rule produces for all five, and then - for diagnosis only, not
as a candidate rule - what two neighbouring definitions would have produced, so the discrepancy can be
located rather than guessed at. Nothing here changes the registered rule; the choice is the owner's.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("Model 7/Approach A").resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path("Model 5/Model 5 Experiment 1").resolve()))
import alias_gate as ag  # noqa: E402
import transfer_eval as tv  # noqa: E402

ROOT = Path(".").resolve()
ids, X, Y, fam, err = ag.design(ROOT)
z = ag._standardise(X)
order = list(np.argsort(err)[::-1])
worst, best = order[:10], order[10:]
df = ag.derive_pairs(ROOT)

print("== the registered rule, exactly as pre-registered (cross-family neighbour)")
print(df.to_string(index=False))
al = df[df.kind == "alias"]
co = df[df.kind == "control"]
print("\nalias   : median distance %.4f (recorded 1.65)  median gap %.4f (recorded 65.7)  "
      "mean gap %.4f" % (al.feature_distance.median(), al.curve_emd.median(), al.curve_emd.mean()))
print("control : median distance %.4f (recorded 1.58)  median gap %.4f (recorded 28.1)  "
      "mean gap %.4f" % (co.feature_distance.median(), co.curve_emd.median(), co.curve_emd.mean()))
h = df[(df.soil_a == "H038")]
print("H038 pair: %s distance %.4f (recorded 1.32) gap %.4f (recorded 125.4)"
      % (list(h.itertuples())[0].soil_b, list(h.itertuples())[0].feature_distance,
         list(h.itertuples())[0].curve_emd))


def neighbour(q, cross_family=True):
    cand = [j for j in range(len(ids)) if j != q
            and (not cross_family or fam[ids[j]] != fam[ids[q]])]
    d = np.sqrt(((z[cand] - z[q]) ** 2).sum(axis=1))
    j = int(cand[int(np.argmin(d))])
    return j, float(d.min()), float(tv.emd_pair(Y[q], Y[j]))


for label, cross in (("nearest neighbour with NO family constraint", False),
                     ("nearest neighbour from a DIFFERENT family (registered)", True)):
    gaps, dists = [], []
    for q in worst:
        _, dd, gg = neighbour(q, cross)
        dists.append(dd)
        gaps.append(gg)
    bg, bd = [], []
    for q in best:
        _, dd, gg = neighbour(q, cross)
        bd.append(dd)
        bg.append(gg)
    print("\n%s:" % label)
    print("  worst: median dist %.4f  median gap %.4f  mean gap %.4f" % (np.median(dists),
                                                                        np.median(gaps),
                                                                        np.mean(gaps)))
    print("  best : median dist %.4f  median gap %.4f  mean gap %.4f" % (np.median(bd),
                                                                        np.median(bg),
                                                                        np.mean(bg)))

# is the discrepancy in one pair? print each alias pair's gap sorted, to see what a median of 65.7
# would require
print("\n== alias gaps, sorted (median of 10 values is the mean of the 5th and 6th)")
gaps = sorted(al.curve_emd.tolist())
print("  " + "  ".join("%.2f" % g for g in gaps))
print("  recorded 65.7 would sit at rank %d of these 10"
      % (1 + sum(1 for g in gaps if g < 65.7)))
print("\n== per-soil M1 errors, worst first (the set the shares are computed over)")
print("  " + "  ".join("%s:%.2f" % (ids[q], err[q]) for q in worst))
print("  summed over 24 = %.4f; top10 = %.4f; worst = %.4f"
      % (err.sum(), err[worst].sum(), err[worst[0]]))
