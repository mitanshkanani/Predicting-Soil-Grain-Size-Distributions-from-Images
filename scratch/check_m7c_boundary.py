"""Boundary verification for M7-C. Read-only: it re-runs the frozen fold protocol, asserts the
honesty properties the 1.68 EMD figure depends on, and characterises what the adaptive estimator
actually changes. It claims no gain and selects no model.

The question the owner asked: is the 1.68 EMD alpha-selection cost genuinely family-honest, and is
the adaptive selector fixed before evaluation rather than plucked from the 15-head sweep?
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
sys.path.insert(0, str(ROOT / "scratch"))
import transfer_eval as tv  # noqa: E402
import probe_head_spread as ps  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def load():
    ft = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    fam = dict(zip(ft.sample_id.astype(str), ft.cv_family.astype(str)))
    img = tv.image_features(ROOT)
    soil = tv.soil_from_images(img)
    ids, Y = tv.labels_matrix(ROOT)
    ids = [str(i) for i in ids]
    X = soil.loc[ids, list(tv.FEATURES)].to_numpy(float)
    F = np.array([fam[i] for i in ids])
    return ids, X, Y, F


def main() -> int:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, BayesianRidge

    ids, X, Y, F = load()
    fams = sorted(set(F))
    check("24 soils, 16 families, sizes 4 + five pairs + ten singletons",
          len(ids) == 24 and len(fams) == 16
          and sorted(pd.Series(F).value_counts().tolist(), reverse=True)
          == [4, 2, 2, 2, 2, 2] + [1] * 10)

    # ---- 1. HONESTY OF THE OUTER FOLDS, asserted rather than remembered ----------------------
    scored = np.zeros(len(ids), int)
    leak = 0
    for g in fams:
        te = [i for i, k in enumerate(F) if k == g]
        tr = [i for i, k in enumerate(F) if k != g]
        scored[te] += 1
        if set(F[te]) & set(F[tr]):
            leak += 1
        for i in te:
            if i in tr:
                leak += 1
    check("every soil is scored exactly once", bool((scored == 1).all()))
    check("no scored soil, and no soil from its family, is ever in the fitting set",
          leak == 0, "leaks=%d" % leak)

    # ---- 2. THE TWO ARMS DIFFER ONLY IN HOW ALPHA IS CHOSEN ---------------------------------
    def arms():
        fixed, nested, chosen = {}, {}, []
        for a in tv.ALPHAS:
            fixed[a] = []
        for g in fams:
            te = [i for i, k in enumerate(F) if k == g]
            tr = [i for i, k in enumerate(F) if k != g]
            sc = StandardScaler().fit(X[tr])
            Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
            mu, V = ps.basis(Y[tr])
            Tt = (Y[tr] - mu) @ V.T
            a_sel = ps.ridge_alpha_nested(Xt, Tt, Y[tr], [F[i] for i in tr])
            chosen.append((g, a_sel))
            for row, j in zip(tv.project(
                    Ridge(alpha=a_sel).fit(Xt, Tt).predict(Xe) @ V + mu), te):
                nested[j] = ps.emd(row, Y[j])
            for a in tv.ALPHAS:
                for row, j in zip(tv.project(
                        Ridge(alpha=a).fit(Xt, Tt).predict(Xe) @ V + mu), te):
                    fixed[a].append(ps.emd(row, Y[j]))
        return {a: float(np.mean(v)) for a, v in fixed.items()}, float(np.mean(
            list(nested.values()))), chosen

    fixed, nested, chosen = arms()
    best_a = min(fixed, key=lambda a: fixed[a])
    oracle = fixed[best_a]
    print("  nested (honest) %.2f | best fixed alpha %.2f at %g | regret %.2f"
          % (nested, oracle, best_a, nested - oracle))
    check("the nested arm reproduces the probe's 42.83 and the oracle arm 41.15",
          abs(nested - 42.83) < 0.02 and abs(oracle - 41.15) < 0.02
          and best_a == 3.0)
    check("the 1.68 figure is ORACLE REGRET, not an achievable gain",
          abs((nested - oracle) - 1.68) < 0.03, "nested-oracle=%.2f" % (nested - oracle))

    # ---- 3. THE ORACLE ARM PEEKS: say so in the artifact, not just in prose -----------------
    # The oracle minimum is the argmin of the mean LOFO error over all 24 soils, i.e. it is chosen
    # with the held-out errors of the soils it scores. No deployed or fold-local procedure can
    # access that, which is exactly why it is a regret bound and not a gain. Evidence that it is
    # not fold-local: the per-fold honest choices need not agree with it.
    agree = sum(1 for _, a in chosen if a == best_a)
    check("the oracle alpha is a global quantity, not reachable by any single fold's information",
          agree < len(chosen), "%d of %d folds independently pick %g"
          % (agree, len(chosen), best_a))

    # ---- 4. SELECTION INSTABILITY: the mechanism under test ------------------------------------
    picks = pd.Series([a for _, a in chosen]).value_counts()
    print("  alpha picked per outer fold: %s" % picks.to_dict())
    check("the nested selector is not constant across folds (so selection really is a procedure)",
          len(picks) > 1, "%d distinct choices" % len(picks))

    # ---- 5. CHARACTERISE THE ADAPTIVE ESTIMATOR: is it just an off-grid alpha? ----------------
    # BayesianRidge learns its own regularisation per target column, so the question the boundary
    # needs answered is whether that is a better *choice from the same family* or a larger model
    # class. Report the learned values; claim nothing from them.
    eff = []
    learned = []
    for g in fams:
        te = [i for i, k in enumerate(F) if k == g]
        tr = [i for i, k in enumerate(F) if k != g]
        sc = StandardScaler().fit(X[tr])
        Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
        mu, V = ps.basis(Y[tr])
        Tt = (Y[tr] - mu) @ V.T
        ms = [BayesianRidge().fit(Xt, Tt[:, j]) for j in range(3)]
        learned.append([float(m.alpha_) for m in ms])
        coef = np.stack([m.predict(Xe) for m in ms], axis=1)
        for row, j in zip(tv.project(coef @ V + mu), te):
            eff.append(ps.emd(row, Y[j]))
    learned = np.array(learned)
    print("  adaptive arm mean %.2f" % float(np.mean(eff)))
    print("  learned alpha_ per column: min %.3g median %.3g max %.3g"
          % (learned.min(), np.median(learned), learned.max()))
    inside = [(learned > a * 0.5) & (learned < a * 2.0) for a in tv.ALPHAS]
    near_grid = np.zeros_like(learned, bool)
    for m in inside:
        near_grid |= m
    print("  share of learned penalties within a factor 2 of a grid value: %.2f"
          % near_grid.mean())
    check("the adaptive arm reproduces the sweep's 40.20",
          abs(float(np.mean(eff)) - 40.20) < 0.02)

    # ---- 6. DETERMINISM: two fresh loads must give identical numbers ---------------------------
    fixed2, nested2, _ = arms()
    check("fold protocol is deterministic across a second full pass",
          nested2 == nested and all(fixed2[a] == fixed[a] for a in tv.ALPHAS))

    facts = {"nested_honest": round(nested, 4), "oracle_fixed": round(oracle, 4),
             "oracle_alpha": float(best_a), "oracle_regret": round(nested - oracle, 4),
             "picks": {str(a): int(c) for a, c in picks.items()},
             "n_folds": len(chosen), "n_folds_choosing_oracle": int(agree),
             "adaptive_mean": round(float(np.mean(eff)), 4),
             "adaptive_alpha_median": round(float(np.median(learned)), 6),
             "adaptive_alpha_max": round(float(learned.max()), 6),
             "adaptive_share_near_grid": round(float(near_grid.mean()), 4),
             "grid_floor": float(min(tv.ALPHAS)), "anchor_recorded": 43.0217308796477}
    import json
    (ROOT / "scratch" / "_m7c_boundary.json").write_text(json.dumps(facts, indent=2),
                                                         encoding="ascii")
    print("facts -> scratch/_m7c_boundary.json")
    print("\n%d checks failed" % len(FAIL))
    for f in FAIL:
        print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
