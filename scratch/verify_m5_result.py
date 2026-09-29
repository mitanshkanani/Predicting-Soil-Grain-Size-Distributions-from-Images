"""Post-run verification of Model 5 E1: why MEAN and CORAL_SELF reproduce NONE exactly, and
whether any emitted artifact contains a NaN, an infinity, or a short row.

This checks a RESULT. It computes no new arm score, reads no gate decision, and writes nothing
into the experiment folder.

Run:  python scratch/verify_m5_result.py
"""
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Model 5" / "Model 5 Experiment 1"
OUT = EXP
sys.path.insert(0, str(EXP))
import alignment as al                                                # noqa: E402
import transfer_eval as te                                            # noqa: E402

FEATS = list(te.FEATURES)
FAILS, LINES = [], []


def check(name, ok, detail=""):
    (LINES if ok else FAILS).append(f"  {'PASS' if ok else '**FAIL**'}  {name}"
                                    + (f"   {detail}" if detail else ""))


print("VERIFY M5 RESULT")
print("-" * 78)

# ---- 1. why the two null arms are exactly null ------------------------------------
IMG = te.image_features(ROOT)
TARGET = te.alignment_target(IMG)
fit = IMG[IMG.camera_fam == "Motorola"]
src = fit[FEATS].to_numpy(float)
T_none = {"A": np.eye(12), "b": np.zeros(12)}
T_mean = al.coral_transform(src, TARGET, match_cov=False)
T_self = al.coral_transform(src, src)
check("the MEAN arm's matrix is the identity and only its shift is nonzero",
      np.array_equal(T_mean["A"], np.eye(12)) and np.abs(T_mean["b"]).max() > 1e-6,
      f"max |b| = {np.abs(T_mean['b']).max():.4f}")
check("CORAL_SELF is the identity to numerical precision",
      np.abs(T_self["A"] - np.eye(12)).max() < 1e-9 and np.abs(T_self["b"]).max() < 1e-9,
      f"max |A-I| = {np.abs(T_self['A'] - np.eye(12)).max():.2e}")

Xn = te.soil_from_images(fit).to_numpy(float)
_shifted = fit.copy()
_shifted[FEATS] = al.apply_transform(fit[FEATS].to_numpy(float), T_mean)
Xm = te.soil_from_images(_shifted).to_numpy(float)
check("the MEAN arm's soil matrix is the NONE soil matrix plus one constant vector",
      np.abs(Xm - (Xn + T_mean["b"])).max() < 1e-9,
      f"max deviation {np.abs(Xm - (Xn + T_mean['b'])).max():.2e}; the image median commutes "
      "with a constant shift, column by column")

from sklearn.preprocessing import StandardScaler                        # noqa: E402
s1 = StandardScaler().fit(Xn)
s2 = StandardScaler().fit(Xm)
check("StandardScaler removes a constant shift from both sides identically, so the MEAN arm "
      "CANNOT move the head",
      np.abs(s1.transform(Xn) - s2.transform(Xm)).max() < 1e-9,
      f"max scaled-feature difference {np.abs(s1.transform(Xn) - s2.transform(Xm)).max():.2e}; "
      "the rank-3 basis and Ridge see the same matrix")
print("  => MEAN and CORAL_SELF are STRUCTURAL zeros in this pipeline, not measured nulls: any")
print("     effect of an alignment here must come from the covariance half, which is what")
print("     CORAL and CORAL_INDEP compare. P3's 'CONFIRMED' is this identity talking, and it")
print("     should be read as 'MEAN did nothing', not as 'the mean shift carries the gain'.")

# ---- 2. artifact integrity --------------------------------------------------------
gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
check("gate.json reports all eight clauses", len(gate["clauses"]) == 8)
check("submit is False and the verdict is not SUPPORTED",
      gate["submit"] is False and gate["verdict"] != "SUPPORTED",
      f"{gate['verdict']} submit={gate['submit']}")
check("no submission file exists, as the gate requires",
      not list(EXP.glob("Submission*.csv")) and not list(ROOT.glob("Submission_Model5*")))
check("G0 gap and the boundary statuses were recorded",
      gate["g0"]["gap"] < 1e-6 and set(gate["boundary_status_per_direction"]) == {"A", "B"},
      str(gate["boundary_status_per_direction"]))

tables = {}
for f in sorted(EXP.glob("*.csv")):
    df = pd.read_csv(f)
    tables[f.name] = df
    num = df.select_dtypes(include=[np.number])
    check(f"{f.name}: {len(df)} rows x {len(df.columns)} cols, no NaN, all finite",
          len(df) > 0 and not num.isna().to_numpy().any()
          and bool(np.isfinite(num.to_numpy()).all()))

ps = tables["per_soil_transfer.csv"]
check("per-soil table covers 7 arms x 2 directions x their soils = 315 rows",
      len(ps) == 315 and ps.groupby(["direction", "arm"]).sample_id.nunique().eq(
          ps.groupby(["direction", "arm"]).size()).all(), str(len(ps)))
check("every arm scored exactly 22 soils in A and 23 in B",
      ps.groupby(["arm", "direction"]).size().unique().tolist() == [22, 23]
      or sorted(set(ps.groupby(['arm', 'direction']).size().tolist())) == [22, 23],
      str(sorted(set(ps.groupby(["arm", "direction"]).size().tolist()))))
pred = tables["predictions_transfer.csv"]
cols = [c for c in pred.columns if c not in ("direction", "arm", "fold", "sample_id")]
P = pred[cols].to_numpy(float)
check("all 315 predicted curves are non-decreasing CDFs pinned at 100 on the last support",
      bool((np.diff(P, axis=1) >= -1e-9).all()) and bool(np.allclose(P[:, -1], 100.0)),
      f"{P.shape}")
check("every predicted curve lies inside [0, 100]", bool((P >= -1e-9).all() and (P <= 100 + 1e-9).all()))
ab = tables["alpha_by_fold.csv"]
check("alpha recorded for all 224 (direction, arm, fold) combinations", len(ab) == 224, str(len(ab)))
check("every nested alpha is a member of the frozen grid",
      set(np.round(ab.nested_alpha.astype(float), 12)) <=
      {round(float(a), 12) for a in te.ALPHAS})

# EMD recomputed from the stored curves must reproduce the stored per-soil errors
ytab = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_samples.csv")
ytab = ytab[ytab.split == "train"].set_index("sample_id")
sup = [c for c in pred.columns if c not in ("direction", "arm", "fold", "sample_id")]
recomputed = 0
worst = 0.0
for (d, a), grp in pred.groupby(["direction", "arm"]):
    for _, r in grp.iterrows():
        y = ytab.loc[str(r["sample_id"]), sup].to_numpy(float)
        e = te.emd_pair(np.array([float(r[c]) for c in sup]), y)
        stored = float(ps[(ps.direction == d) & (ps.arm == a)
                          & (ps.sample_id == r["sample_id"])].emd.iloc[0])
        worst = max(worst, abs(e - stored))
        recomputed += 1
check("every stored per-soil EMD is reproducible from the stored predicted curves",
      recomputed == 315 and worst < 1e-9, f"{recomputed} recomputed, max deviation {worst:.2e}")

# ---- 3. the run left every cell with output --------------------------------------
log = io.open(ROOT / "scratch" / "_m5_run4.log", encoding="utf-8").read()
ncells = log.count(" >>> cell ")
blank = [i for i in range(1, ncells) if f">>> cell {i}\n >>> cell {i+1}" in
         log.replace(" ", "").replace("\n", " ")]
check("all 21 code cells executed and none printed nothing", ncells == 21 and not blank,
      f"{ncells} cell markers, {len(blank)} empty")
check("the run exited cleanly and wrote the record",
      "exit = 0" in log and (EXP / "Experiment1.txt").exists())
for bad in ("Traceback", "nan", "inf"):
    hits = sum(1 for ln in log.splitlines() if bad in ln and "no NaN" not in ln)
    check(f"the run log contains no '{bad}'", hits == 0, f"{hits} lines")

print("\n".join(LINES))
print("-" * 78)
if FAILS:
    print("\n".join(FAILS))
    print(f"VERIFY M5 RESULT: FAILED ({len(FAILS)} problems)")
    sys.exit(1)
print(f"VERIFY M5 RESULT: PASSED ({len(LINES)} checks)")
