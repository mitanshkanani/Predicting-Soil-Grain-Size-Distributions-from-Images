"""Pins the two things the contract says about Delta and about the cluster unit.

Spec section 3 step 4:  Delta(s) = err_NONE(s) - err_arm(s), POSITIVE MEANS THE ARM IS BETTER.
Spec section 3:        resampling unit = SOIL IDENTITY; the 21 soils photographed by both cameras
                       move together as ONE cluster, giving 45 rows over 24 clusters.

Both are easy to write down and easy to get wrong, and neither failure raises on its own: a sign
flip turns a degradation into a "gain", and prefixing a cluster key with the direction turns the
soil-clustered bootstrap into the row bootstrap it was introduced to replace, narrowing every CI by
about sqrt(2). G3 and G4 are decided by those intervals, so this is not bookkeeping.

This file execs the SHIPPED cell F1 helpers, sliced between two markers, against a synthetic design
that mirrors the real one (22 + 23 rows over 24 soils, 21 shared). It says nothing about any
measured number.

Run:  python scratch/check_m5_delta.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 5" / "Model 5 Experiment 1" / "Model5_Experiment1.ipynb"
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import transfer_eval as te                                            # noqa: E402

FAILS, LINES = [], []


def check(name, ok, detail=""):
    (LINES if ok else FAILS).append(f"  {'PASS' if ok else '**FAIL**'}  {name}"
                                    + (f"   {detail}" if detail else ""))


src = ""
for c in json.load(io.open(NB, encoding="utf-8"))["cells"]:
    s = "".join(c["source"])
    if s.startswith('"""Cell F1 -'):
        src = s
assert src, "cell F1 not found in the shipped notebook"
HELPERS = src[src.index("def arm_rows"):src.index("GAIN = {")]
assert "def gain(" in HELPERS and "def _bs(" in HELPERS, "sliced too short"

A_ONLY = ["S21"]
B_ONLY = ["S22", "S23"]
SHARED = [f"S{i:02d}" for i in range(21)]
IDS_A = sorted(SHARED + A_ONLY)
IDS_B = sorted(SHARED + B_ONLY)
N_SOILS = len(set(IDS_A) | set(IDS_B))
print("CHECK M5 DELTA AND CLUSTERING - the shipped F1 helpers, on a synthetic mirror of the design")
print("-" * 78)
check("the synthetic mirror has the contracted shape", len(IDS_A) == 22 and len(IDS_B) == 23
      and N_SOILS == 24 and len(SHARED) == 21,
      f"{len(IDS_A)}+{len(IDS_B)} rows, {N_SOILS} soils, {len(SHARED)} shared")

BETTER_BY = 5.0                      # the arm beats NONE by ~5 EMD, varying soil to soil so the
SPREAD = 0.35                        # bootstrap has something to resample and a wrong partition
DLT = {s: BETTER_BY + SPREAD * (((int(s[1:]) * 7) % 9) - 4)          # can actually be detected
       for s in sorted(set(IDS_A) | set(IDS_B))}
WORSE_BY = -3.0
rows = []
for d, ids in (("A", IDS_A), ("B", IDS_B)):
    for s in ids:
        rows.append(dict(direction=d, arm="NONE", sample_id=s, emd=50.0))
        rows.append(dict(direction=d, arm="BETTER", sample_id=s, emd=50.0 - DLT[s]))
        rows.append(dict(direction=d, arm="WORSE", sample_id=s, emd=50.0 - WORSE_BY))
TRANS = pd.DataFrame(rows)
BASE = {d: TRANS[(TRANS.direction == d) & (TRANS.arm == "NONE")].set_index("sample_id").emd
        for d in ("A", "B")}
ns = {"np": np, "pd": pd, "te": te, "TRANS": TRANS, "BASE": BASE,
      "DIRECTIONS": ("A", "B"), "ARMS": ("NONE", "BETTER", "WORSE"),
      "CFG": type("C", (), dict(n_boot=4000, n_rows_eval=45, n_clusters=N_SOILS))(),
      "BOOT_SEED": 20260934}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(HELPERS, "cellF1-helpers", "exec"), ns)
gain, arm_rows = ns["gain"], ns["arm_rows"]

EXP = np.mean([DLT[s] for s in list(IDS_A) + list(IDS_B)])
EXP_A = np.mean([DLT[s] for s in IDS_A])
EXP_B = np.mean([DLT[s] for s in IDS_B])
g = gain("BETTER")
check("Delta is err_NONE minus err_arm, so an arm that improves every soil gives POSITIVE gain",
      abs(g["pooled_gain"] - EXP) < 1e-9, f"pooled {g['pooled_gain']:+.4f} (want {EXP:+.4f})")
check("and both directions carry that same sign and their own means",
      abs(g["gain_A"] - EXP_A) < 1e-9 and abs(g["gain_B"] - EXP_B) < 1e-9,
      f"A {g['gain_A']:+.3f}  B {g['gain_B']:+.3f}")
w = gain("WORSE")
check("an arm that degrades every soil gives a NEGATIVE gain, not a positive one",
      abs(w["pooled_gain"] - WORSE_BY) < 1e-9, f"pooled {w['pooled_gain']:+.3f} (want -3.000)")
check("so a gate of the form gain >= 3 cannot be passed by a degradation",
      not w["pooled_gain"] >= 3.0, f"pooled {w['pooled_gain']:+.3f}")

_, clusters, per_dir = arm_rows("BETTER")
check("45 evaluation rows collapse to 24 SOIL clusters, not 45",
      len(clusters) == 45 and len(set(clusters)) == 24,
      f"{len(clusters)} rows, {len(set(clusters))} clusters")
check("a cluster key is a soil identity and carries no direction prefix",
      set(clusters) == set(IDS_A) | set(IDS_B),
      "keys look like " + str(sorted(set(clusters))[:2]))
by = {}
for k, s in zip(clusters, list(IDS_A) + list(IDS_B)):
    by.setdefault(k, []).append(s)
sizes = sorted(by.values(), key=len, reverse=True)
check("the 21 soils seen by both cameras each hold both of their rows; the 3 single-camera soils "
      "hold one",
      [len(v) for v in sizes] == [2] * 21 + [1] * 3,
      f"{sum(1 for v in by.values() if len(v) == 2)} clusters of 2, "
      f"{sum(1 for v in by.values() if len(v) == 1)} of 1")
for d, ids in (("A", IDS_A), ("B", IDS_B)):
    check(f"within direction {d} the {len(ids)} rows are {len(set(ids))} distinct soils, so the "
          f"per-direction CI is not double-counting",
          len(per_dir[d][1]) == len(set(per_dir[d][1])) == len(ids), str(len(per_dir[d][2])))

# the clustering must be doing real work: same rows, finer partition => narrower interval
rows_v = np.concatenate([per_dir["A"][2], per_dir["B"][2]])
bare = list(IDS_A) + list(IDS_B)
prefixed = [f"A:{s}" for s in IDS_A] + [f"B:{s}" for s in IDS_B]
lo_bare, hi_bare = ns["_bs"](rows_v, bare)[1:]
lo_pref, hi_pref = ns["_bs"](rows_v, prefixed)[1:]
check("soil clustering genuinely widens the interval over a direction-split partition",
      (hi_bare - lo_bare) > (hi_pref - lo_pref) + 1e-9,
      f"soil [{lo_bare:+.2f},{hi_bare:+.2f}] width {hi_bare - lo_bare:.3f} vs prefixed width "
      f"{hi_pref - lo_pref:.3f}")
check("a direction-split partition really is narrower, i.e. the bug is worth catching",
      (hi_bare - lo_bare) > (hi_pref - lo_pref) + 1e-9,
      f"soil-partition width {hi_bare - lo_bare:.3f} vs split-partition {hi_pref - lo_pref:.3f}")
check("the shipped gain() is the soil-clustered one, so its interval is the WIDER of the two",
      abs(g["pooled_lo"] - lo_bare) < 1e-9 and abs(g["pooled_hi"] - hi_bare) < 1e-9,
      f"gain() CI [{g['pooled_lo']:+.3f},{g['pooled_hi']:+.3f}] vs soil [{lo_bare:+.3f},"
      f"{hi_bare:+.3f}]")
check("a null effect gives an interval straddling zero, so G3 cannot be passed by nothing",
      gain("NONE")["pooled_lo"] <= 0.0 <= gain("NONE")["pooled_hi"],
      str((gain("NONE")["pooled_lo"], gain("NONE")["pooled_hi"])))
print("\n".join(LINES))
print("-" * 78)
if FAILS:
    print("\n".join(FAILS))
    print(f"CHECK M5 DELTA AND CLUSTERING: FAILED ({len(FAILS)} problems)")
    sys.exit(1)
print(f"CHECK M5 DELTA AND CLUSTERING: PASSED ({len(LINES)} checks)")
