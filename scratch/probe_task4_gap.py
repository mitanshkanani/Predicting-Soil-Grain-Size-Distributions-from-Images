"""Why does m7a_eval.lofo_errors(12 columns) differ from tv.no_holdout_control_score by 0.431 EMD?

Read-only diagnostic. It reproduces both protocols side by side on the same 12-column design and
prints, per held-out family, the alpha each one selected and the per-soil error it produced. The
prediction step is identical in both; the hypothesis under test is the alpha-selection step.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("Model 7/Approach A").resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path("Model 5/Model 5 Experiment 1").resolve()))
import m7a_eval as ev  # noqa: E402
import transfer_eval as tv  # noqa: E402

ROOT = Path(".").resolve()
ANCHOR = 43.0217308796477

X, Y, fams, ids = ev.soil_design(ROOT, "none")
F = np.asarray(fams)
fams_sorted = sorted(set(F))
print("design", X.shape, "families", len(fams_sorted), "soils", len(ids))

# --- protocol A: the frozen Model 5 function's own structure, spelled out here for comparison.
# For every alpha it fits every family-out fold of the FULL 24, then, when choosing alpha for
# family G, it pools the errors of the other families from those full-corpus folds - so the fit
# that produced a fold used in G's alpha selection contained G itself.
err = {}
for al in tv.ALPHAS:
    pf = {}
    for G in fams_sorted:
        hold = np.where(F == G)[0]
        keep = np.where(F != G)[0]
        P = tv.fit_predict(X[keep], Y[keep], X[hold], al)
        pf[G] = {i: tv.emd_pair(P[k], Y[i]) for k, i in enumerate(hold)}
    err[float(al)] = pf
A = {}
for G in fams_sorted:
    inner = [g for g in fams_sorted if g != G]
    curve = {al: float(np.mean([v for g in inner for v in err[al][g].values()]))
             for al in err}
    best = min(curve, key=curve.get)
    A[G] = (best, float(np.mean(list(err[best][G].values()))))

# --- protocol B: m7a_eval's strictly nested selection. For family G, alpha is chosen using only
# folds fitted on the 24 minus G minus the inner family, so the scored family never enters the
# training set of any fit that informs its own alpha.
errsB, alphasB = ev.lofo_errors(X, Y, fams, return_alphas=True)
per_B = {g: float(np.mean([errsB[i] for i in range(len(F)) if F[i] == g])) for g in fams_sorted}

print("\n%-22s %-10s %-10s %-12s %-12s" % ("family", "A alpha", "B alpha", "A mean EMD",
                                            "B mean EMD"))
same = 0
for g in fams_sorted:
    a, ea = A[g]
    b = alphasB[g]
    same += int(a == b)
    print("%-22s %-10g %-10s %-12.6f %-12.6f %s" % (g, a, b, ea, per_B[g],
                                                     "" if a == b else "  <- alpha differs"))
print("\nfamilies where the two protocols chose the same alpha: %d of %d" % (same, len(fams_sorted)))
print("protocol A pooled mean : %.15f" % float(np.mean([A[g][1] for g in fams_sorted])))
print("  (this is what no_holdout_control_score returns; A recomputed here for the per-family table)")
print("frozen function value  : %.15f" % float(tv.no_holdout_control_score(ROOT)))
print("protocol B pooled mean : %.15f" % float(np.mean(ev.lofo_errors(X, Y, fams))))
print("anchor                 : %.15f" % ANCHOR)
print("frozen - B             : %+.15f" % (float(tv.no_holdout_control_score(ROOT))
                                            - float(np.mean(ev.lofo_errors(X, Y, fams)))))

# --- and the second failure: zeroing the block is not bit-exact. Where does the difference arise?
Xz = ev.soil_design(ROOT, "zero")[0]
ez = ev.lofo_errors(Xz, Y, fams)
e0 = ev.lofo_errors(X, Y, fams)
d = np.abs(ez - e0)
print("\nG6 none-vs-zero: max |diff| %.3e, nonzero soils %d of %d"
      % (float(d.max()), int((d > 0).sum()), len(d)))
al0 = ev.lofo_errors(X, Y, fams, return_alphas=True)[1]
alz = ev.lofo_errors(Xz, Y, fams, return_alphas=True)[1]
print("alpha choices identical between the two arms: %s" % (al0 == alz))
print("design column scales: std of the 12 head cols equal? %s"
      % bool(np.array_equal(X[:, :12].std(axis=0), Xz[:, :12].std(axis=0))))
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
for name, M in (("12 cols", X[:, :12]), ("16 cols with 4 zero", Xz)):
    Z = StandardScaler().fit_transform(M)
    p = PCA(n_components=tv.PC_RANK, svd_solver="full").fit(Z)
    print("%-20s PCA explained_variance_ %s  components[0][:4] %s"
          % (name, np.round(p.explained_variance_, 12), np.round(p.components_[0][:4], 12)))
