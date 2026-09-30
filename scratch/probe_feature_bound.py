"""Read-only probe for research_map_1_2_5_6.md section 1. Reuses Model 5's frozen loaders and
EMD; computes nothing that is fed to a model and writes nothing outside scratch/.

Question: how much exploitable signal is left in the 12 hand features?
Estimator: leave-one-out k-nearest-neighbour lookup in standardised feature space, scoring the
mean of the k neighbours' TRUE curves. Two candidate pools:

  naive     - all 23 other soils, including same-CV-family siblings.
  honest    - same-CV-family siblings excluded, which is what nested CV enforces.

A sibling shares a CV family by construction (similar D50), so borrowing its curve is not a
prediction, it is a leak. The gap between the two columns is the size of that leak.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import transfer_eval as tv  # noqa: E402

FEATS = list(tv.FEATURES)


def main() -> int:
    fam_tbl = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    fams = dict(zip(fam_tbl.sample_id.astype(str), fam_tbl.cv_family.astype(str)))
    img = tv.image_features(ROOT)
    soil = tv.soil_from_images(img)
    ids, Y = tv.labels_matrix(ROOT)
    ids = [str(i) for i in ids]
    X = soil.loc[ids, FEATS].to_numpy(float)
    assert np.isfinite(X).all() and np.isfinite(Y).all()

    counts = pd.Series([fams[i] for i in ids]).value_counts()
    print("cv families: n=%d  sizes=%s" % (len(counts), sorted(counts.tolist(), reverse=True)))
    print("soils with >=1 same-family sibling: %d of %d"
          % (sum(1 for i in ids if counts[fams[i]] > 1), len(ids)))

    sup, dl = tv._supports()
    print("supports (%d): %s" % (len(sup), " ".join(sup)))

    def emd(p, t):
        return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), dl))

    def project(F):
        F = np.clip(np.atleast_2d(np.asarray(F, float)), 0, 100)
        F = np.maximum.accumulate(F, axis=1)
        F[:, -1] = 100.0
        return F

    def run(pool_of, scaler, label):
        """scaler='pool' standardises on the candidate pool only (honest).
        scaler='all' standardises on all 24 rows including the query - the leak that produced the
        withdrawn 44.23 figure."""
        rec = {}
        for k in (1, 2, 3, 4, 5, 6, 8):
            errs = []
            for q in range(len(ids)):
                cand = [j for j in range(len(ids)) if j != q and pool_of(q, j)]
                base = X if scaler == "all" else X[cand]
                mu = base.mean(0, keepdims=True)
                sd = np.where(base.std(0, keepdims=True) > 1e-12, base.std(0, keepdims=True), 1.0)
                d = np.sqrt((((X[cand] - mu) / sd - (X[q] - mu) / sd) ** 2).sum(1))
                order = np.argsort(d)[:k]
                errs.append(emd(project(Y[cand][order].mean(0))[0], Y[q]))
            rec[k] = float(np.mean(errs))
        print("%-22s %s" % (label, " ".join("k%d=%.2f" % (k, v) for k, v in rec.items())))
        return rec

    def siblings(q, j):
        return fams[ids[q]] == fams[ids[j]]

    any_pool = lambda q, j: True
    no_siblings = lambda q, j: not siblings(q, j)

    tables = {}
    tables[("all", "naive")] = run(any_pool, "all", "scaler=all / pool=all")
    tables[("all", "honest")] = run(no_siblings, "all", "scaler=all / no siblings")
    tables[("pool", "naive")] = run(any_pool, "pool", "scaler=pool / pool=all")
    tables[("pool", "honest")] = run(no_siblings, "pool", "scaler=pool / no siblings")
    naive = tables[("all", "naive")]
    honest = tables[("pool", "honest")]
    print("leak vs the fully honest version: %s"
          % " ".join("k%d%+.2f" % (k, honest[k] - naive[k]) for k in honest))
    pd.DataFrame([{"scaler": s, "pool": p, "k": k, "emd": v}
                  for (s, p), t in tables.items() for k, v in t.items()]
                 ).to_csv(ROOT / "scratch" / "_feature_bound.csv", index=False)

    # --- constant-curve references and the recorded model score ------------------------------
    med = project(np.median(Y, 0))[0]
    mean = project(Y.mean(0))[0]
    print("constant train-median %.2f | constant train-mean %.2f"
          % (np.mean([emd(med, y) for y in Y]), np.mean([emd(mean, y) for y in Y])))
    print("recorded shipped nested ridge: 43.0217308796477")

    # --- rank-k representation floor, both conventions ---------------------------------------
    mu = Y.mean(0)
    _, _, Vt = np.linalg.svd(Y - mu, full_matrices=False)
    floors = {}
    for rank in (2, 3, 4, 5, 7):
        V = Vt[:rank]
        rec = (Y - mu) @ V.T @ V + mu
        floors[rank] = {"bare": float(np.mean([emd(r, y) for r, y in zip(rec, Y)])),
                        "projected": float(np.mean([emd(p, y) for p, y in zip(project(rec), Y)]))}
        print("rank %d floor: bare %.2f | monotone-projected %.2f"
              % (rank, floors[rank]["bare"], floors[rank]["projected"]))
    pd.DataFrame(floors).T.rename_axis("rank").to_csv(ROOT / "scratch" / "_rank_floors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
