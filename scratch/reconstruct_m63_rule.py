"""Reconstruction of Model 6-3's actual alias rule, from the verbatim probe source recovered out of
this project's own session transcript (scratch/_m63_commands_verbatim.txt, transcript lines 11543 /
11551 / 11557 of session c6165df8).

This is a HISTORICAL RECONSTRUCTION, not a search for a definition that makes the Task 5 numbers
match. It reproduces what the original printed, and separately reports what the registered Task 5
rule printed, and the exact difference between the two rules.

Read-only. It writes no artifact, touches no Model 6 record, scores no A1-A4 block, runs no control,
fits nothing, predicts nothing.

Run:  python scratch/reconstruct_m63_rule.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
sys.path.insert(0, str(ROOT / "Model 7" / "Approach A"))
import transfer_eval as te  # noqa: E402

DL = np.log10(np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200]))
F = list(te.FEATURES)

# ---- the Model 6-3 recorded figures, quoted from research_map_1_2_5_6.md:189-208 and from the
# ---- original probe output in the transcript. Nothing here is fitted.
RECORDED = {
    "worst8_nn_dist": 1.65, "best8_nn_dist": 1.58,
    "worst8_nn_emd": 65.7, "best8_nn_emd": 28.1,
    "alias_pair_maxd_median": 1.95, "lowerror_pair_maxd_median": 1.37,
}
ROWS = {  # soil: (err, nn-dist, nn-EMD, D50 mm) exactly as the original printed them
    "H038": (104.73, 1.32, 125.42, 2.4338), "H616": (79.38, 2.22, 85.33, 0.2107),
    "H037": (73.91, 1.23, 83.95, 0.8144), "H405": (70.34, 1.64, 67.30, 0.0225),
    "F827": (56.98, 1.65, 64.13, 0.0213), "H126": (54.41, 2.00, 34.08, 0.0350),
    "H368": (16.42, 1.58, 48.07, 6.1261), "H615": (11.24, 1.86, 22.05, 5.3387),
    "G190": (7.27, 1.32, 125.42, 0.0622),
}
ALIAS_PAIRS_HAND = [("H038", "G190"), ("H037", "H368"), ("H405", "H615"),
                    ("F827", "H372"), ("H616", "H183")]
OK_PAIRS_HAND = [("H368", "H615"), ("H181", "H183"), ("H372", "H181"), ("H615", "G190")]

FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def emd(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.trapezoid(np.abs(a - b), DL))


def base():
    ids, Y = te.labels_matrix(str(ROOT))
    img = te.image_features(str(ROOT))
    X = te.soil_from_images(img).loc[ids, F].to_numpy(float)
    err = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_per_soil.csv"
                      ).set_index("sample_id")["M1"]
    err = pd.Series({s: float(err[s]) for s in ids})
    return ids, Y, X, img, err


print("== 1. the recovered rule, run verbatim ==")
ids, Y, X, img, err = base()
check("the per-soil error series reproduces the internal anchor mean 43.0217308796477",
      abs(err.mean() - 43.0217308796477) < 5e-13, "%.13f" % err.mean())

# EXACTLY as recovered: corpus mean, std + 1e-12 (additive, not a dead-column guard), Euclidean
# nearest neighbour over ALL 23 other soils with NO family constraint, worst8/best8 halves.
mu, sd = X.mean(0), X.std(0) + 1e-12
Z = (X - mu) / sd
d50 = np.array([10 ** np.interp(50., np.maximum(Y[i], np.linspace(1e-6, 1, 11)), DL)
                for i in range(len(ids))])
rows = []
for i in range(len(ids)):
    dd = np.linalg.norm(Z - Z[i], axis=1)
    dd[i] = 1e9
    j = int(np.argmin(dd))
    rows.append((ids[i], ids[j], dd[j], emd(Y[j], Y[i])))
R = pd.DataFrame(rows, columns=["soil", "nn_soil", "nnfeat", "nnemd"]).set_index("soil")
srt = err.sort_values(ascending=False)
worst8, best8 = list(srt.index[:8]), list(srt.index[-8:])

print("   %-7s %8s %9s %9s %9s   %s" % ("soil", "err", "nn-feat-d", "nn-EMD", "D50 mm", "neighbour"))
for s in list(srt.index[:6]) + list(srt.index[-3:]):
    print("   %-7s %8.2f %9.2f %9.2f %9.4f   %s"
          % (s, err[s], R.loc[s, "nnfeat"], R.loc[s, "nnemd"], d50[list(ids).index(s)],
             R.loc[s, "nn_soil"]))

for s, (we, wd, we_md, wd50) in ROWS.items():
    got = (err[s], R.loc[s, "nnfeat"], R.loc[s, "nnemd"], d50[list(ids).index(s)])
    check("row %s reproduces the printed (err, nn-dist, nn-EMD, D50)" % s,
          all(abs(a - b) < (0.005 if k != 3 else 0.00005) for k, (a, b) in enumerate(zip(got, (we, wd, we_md, wd50)))),
          "got %.4f %.4f %.4f %.4f vs recorded %.2f %.2f %.2f %.4f" % (got + (we, wd, we_md, wd50)))

med = (float(R.loc[worst8, "nnfeat"].median()), float(R.loc[worst8, "nnemd"].median()),
       float(R.loc[best8, "nnfeat"].median()), float(R.loc[best8, "nnemd"].median()))
print("   medians  worst8: nn-dist %.2f  nn-curve-EMD %.1f | best8: nn-dist %.2f  nn-curve-EMD %.1f"
      % med)
for label, got, want in (("worst8 nn-dist", med[0], RECORDED["worst8_nn_dist"]),
                         ("worst8 nn-curve-EMD", med[1], RECORDED["worst8_nn_emd"]),
                         ("best8 nn-dist", med[2], RECORDED["best8_nn_dist"]),
                         ("best8 nn-curve-EMD", med[3], RECORDED["best8_nn_emd"])):
    tol = 0.05 if want < 10 else 0.5
    check("%s reproduces the recorded %.2f" % (label, want), abs(got - want) < tol,
          "got %.4f" % got)

# H038's neighbour under the recovered rule, which is the map's irreducible claim.
check("H038's nearest neighbour under the recovered rule is G190",
      R.loc["H038", "nn_soil"] == "G190", str(R.loc["H038", "nn_soil"]))
check("G190's nearest neighbour is H038 (the pair is mutual)",
      R.loc["G190", "nn_soil"] == "H038", str(R.loc["G190", "nn_soil"]))
check("the H038/G190 pair reproduces at 1.32 distance and 125.42 EMD",
      abs(R.loc["H038", "nnfeat"] - 1.32) < 0.005 and abs(R.loc["H038", "nnemd"] - 125.42) < 0.005,
      "%.4f %.4f" % (R.loc["H038", "nnfeat"], R.loc["H038", "nnemd"]))

print("\n== 2. the separability leg, same hand-listed pairs, same formula ==")


def pair_test(a, b):
    A = img[(img.sample_id == a) & (img.split == "train")][F].to_numpy(float)
    B = img[(img.sample_id == b) & (img.split == "train")][F].to_numpy(float)
    ts, ds = [], []
    for j in range(len(F)):
        x, y = A[:, j], B[:, j]
        s = np.sqrt((x.var(ddof=1) / len(x)) + (y.var(ddof=1) / len(y)))
        ts.append((x.mean() - y.mean()) / (s + 1e-12))
        ds.append(abs(x.mean() - y.mean()) / (np.std(np.r_[x, y]) + 1e-12))
    return np.array(ts), np.array(ds)


al_max, ok_max = [], []
for a, b in ALIAS_PAIRS_HAND:
    ts, ds = pair_test(a, b)
    al_max.append(ds.max())
    print("   %-6s vs %-6s : max d %5.2f (%s) | |t|>3: %2d/12" % (a, b, ds.max(), F[int(np.argmax(ds))], int((np.abs(ts) > 3).sum())))
for a, b in OK_PAIRS_HAND:
    ts, ds = pair_test(a, b)
    ok_max.append(ds.max())
    print("   %-6s vs %-6s : max d %5.2f (%s) | |t|>3: %2d/12  [low-error pair]"
          % (a, b, ds.max(), F[int(np.argmax(ds))], int((np.abs(ts) > 3).sum())))
ma, mo = float(np.median(al_max)), float(np.median(ok_max))
print("   aliasing-pair max-d median %.2f | low-error-pair max-d median %.2f" % (ma, mo))
check("alias-pair median max d reproduces 1.95", abs(ma - 1.95) < 0.005, "%.4f" % ma)
check("control-pair median max d reproduces 1.37", abs(mo - 1.37) < 0.005, "%.4f" % mo)
check("H038/G190 reproduces 0 of 12 features at |t| > 3 with max d 1.08",
      int((np.abs(pair_test("H038", "G190")[0]) > 3).sum()) == 0
      and abs(pair_test("H038", "G190")[1].max() - 1.08) < 0.005)

print("\n== 3. what the REGISTERED Task 5 rule produced, for the record (read-only) ==")
reg = pd.read_csv(ROOT / "Model 7" / "Approach A" / "data" / "alias_pairs.csv")
print(reg.to_string(index=False))
print("\n== 4. the two rules differ in exactly these ways ==")
print("   group size      : recovered worst-8 + best-8  |  registered top-10 + remaining 14")
print("   neighbour pool  : recovered allows a SAME-FAMILY sibling  |  registered requires a")
print("                     different CV family from both soils")
print("   error series    : both the recorded M1 column of cv_per_soil.csv (M6-3 proved its own")
print("                     recompute equal to it at max dev 0.00e+00)")
print("   standardisation : recovered std + 1e-12  |  registered std with dead columns set to 1.0")

print("\n== 5. full recovered table, with the family relation of each neighbour ==")
ft = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
FM = dict(zip(ft.sample_id.astype(str), ft.cv_family))
same = 0
print("   %-7s %8s %9s %7s %9s %9s" % ("soil", "err", "nn-soil", "family", "nn-dist", "nn-EMD"))
for s in srt.index:
    rel = "SAME" if FM[s] == FM[R.loc[s, "nn_soil"]] else "-"
    same += rel == "SAME"
    print("   %-7s %8.2f %9s %7s %9.4f %9.2f" % (s, err[s], R.loc[s, "nn_soil"], rel,
                                                 R.loc[s, "nnfeat"], R.loc[s, "nnemd"]))
print("   soils whose recovered nearest neighbour is a same-CV-family sibling: %d of 24" % same)
print("   ...in the worst-8 half: %d | in the best-8 half: %d"
      % (sum(1 for s in worst8 if FM[s] == FM[R.loc[s, "nn_soil"]]),
         sum(1 for s in best8 if FM[s] == FM[R.loc[s, "nn_soil"]])))

print("\n%d checks failed" % len(FAIL))
for f in FAIL:
    print("  - " + f)
sys.exit(1 if FAIL else 0)
