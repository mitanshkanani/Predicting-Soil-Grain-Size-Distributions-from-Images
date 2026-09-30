"""Read-only probe for research_map_1_2_5_6.md buckets A and D.

Question: is there a defensible gain left in the EXISTING 12 features, or does the shipped head
already extract about all this representation supports at n=24?

Protocol, fixed before any head is scored:
  * Fold unit = CV family (16 families, 24 soils). A scored soil's family is never in its fitting
    set - the same rule nested CV imposes, and the same rule the honest k-NN bound obeys.
  * Every head maps 12 standardised features onto the SAME rank-3 curve basis (fitted from that
    fold's training soils only) and gets the SAME monotone projection. Head family is the only
    manipulated variable.
  * No hyperparameter is tuned anywhere. Ridge uses the frozen 16-value alpha grid selected by
    nested family-out CV inside the training set (the shipped rule). Every other head runs at
    library defaults.
  * Null: the whole sweep replayed on soils whose feature-to-curve pairing is randomly permuted.
    Under the null no head has signal, so best-of-N on null data measures the inflation that comes
    from comparing N heads and reporting the winner. A real gain must clear that.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import transfer_eval as tv  # noqa: E402

FEATS = list(tv.FEATURES)
RANK = 3
N_NULL = 12


def make_heads():
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.svm import SVR
    from sklearn.linear_model import BayesianRidge
    return {
        "knn1": KNN(1), "knn2": KNN(2), "knn3": KNN(3), "knn4": KNN(4),
        "knn2w": KNN(2, weights=True), "knn4w": KNN(4, weights=True),
        "pls2": PLSRegression(n_components=2), "pls3": PLSRegression(n_components=3),
        "krr": KernelRidge(), "svr": SVR(), "bayes": BayesianRidge(),
        "rf": RandomForestRegressor(n_estimators=300, random_state=0),
        "et": ExtraTreesRegressor(n_estimators=300, random_state=0),
        "gb": GradientBoostingRegressor(random_state=0),
    }


class KNN:
    def __init__(self, k, weights=False):
        self.k, self.weights = k, weights

    def fit(self, X, T):
        self.X = np.asarray(X, float)
        self.T = np.asarray(T, float).reshape(len(self.X), -1)
        return self

    def predict(self, X):
        rows = []
        for row in np.atleast_2d(X):
            d = np.sqrt(((self.X - row) ** 2).sum(1))
            idx = np.argsort(d)[:self.k]
            w = 1.0 / np.maximum(d[idx], 1e-9) if self.weights else np.ones(len(idx))
            rows.append((self.T[idx] * (w / w.sum())[:, None]).sum(0))
        out = np.asarray(rows)
        return out[:, 0] if out.shape[1] == 1 else out


def emd(pred, truth):
    _, dl = tv._supports()
    return float(np.trapezoid(np.abs(np.asarray(pred, float)
                                     - np.asarray(truth, float)), dl))


def ridge_alpha_nested(Xt, Tt, Yt_tr, Ftr):
    """Frozen rule: pooled inner family-out error over every alpha, take the argmin."""
    from sklearn.linear_model import Ridge
    mu, V = basis(Yt_tr)
    curve = {}
    for a in tv.ALPHAS:
        errs = []
        for g in sorted(set(Ftr)):
            itr = [i for i, k in enumerate(Ftr) if k != g]
            ite = [i for i, k in enumerate(Ftr) if k == g]
            if not itr or not ite:
                continue
            Tloc = (Yt_tr[itr] - mu) @ V.T
            pred = Ridge(alpha=a).fit(Xt[itr], Tloc).predict(Xt[ite]) @ V + mu
            for row, j in zip(tv.project(pred), ite):
                errs.append(emd(row, Yt_tr[j]))
        curve[a] = float(np.mean(errs)) if errs else np.inf
    return min(curve, key=lambda a: curve[a])


def basis(Ytr):
    mu = Ytr.mean(0)
    _, _, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    return mu, Vt[:RANK]


def run_sweep(X, Y, F, verbose=True):
    rows = {}
    for name in list(make_heads()) + ["ridge_nested"]:
        rows[name] = float(np.mean(per_soil_errors(X, Y, F, name)))
    if verbose:
        for name, e in sorted(rows.items(), key=lambda kv: kv[1]):
            print("  %-13s %7.2f" % (name, e))
    return rows


def per_soil_errors(X, Y, F, name):
    """Per-soil honest errors for one head, returned in soil order (NaN-free: every soil is
    held out exactly once, since each fold is a whole family)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    heads = make_heads()
    out = np.full(len(X), np.nan)
    for g in sorted(set(F)):
        te = [i for i, k in enumerate(F) if k == g]
        tr = [i for i, k in enumerate(F) if k != g]
        sc = StandardScaler().fit(X[tr])
        Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
        mu, V = basis(Y[tr])
        Tt = (Y[tr] - mu) @ V.T
        if name == "ridge_nested":
            a = ridge_alpha_nested(Xt, Tt, Y[tr], [F[i] for i in tr])
            coef = Ridge(alpha=a).fit(Xt, Tt).predict(Xe)
        else:
            est = copy.deepcopy(heads[name])
            if name in ("pls2", "pls3"):
                coef = np.asarray(est.fit(Xt, Tt).predict(Xe)).reshape(len(te), -1)
            else:
                coef = np.stack([np.asarray(copy.deepcopy(est).fit(Xt, Tt[:, j])
                                            .predict(Xe), float) for j in range(RANK)], axis=1)
        assert coef.shape == (len(te), RANK), (name, np.shape(coef))
        for row, j in zip(tv.project(coef @ V + mu), te):
            out[j] = emd(row, Y[j])
    assert not np.isnan(out).any()
    return out


def family_bootstrap(e_a, e_b, F, n_boot=4000, seed=7):
    """Paired, clustered on CV family (siblings share a fold, so their errors are not independent).
    d = e_b - e_a, so a POSITIVE d means head a has the lower error."""
    rng = np.random.default_rng(seed)
    fams = sorted(set(F))
    groups = [np.where(F == g)[0] for g in fams]
    d = e_a - e_b
    point = float(d.mean())
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        draws.append(d[idx].mean())
    draws = np.array(draws)
    return point, np.percentile(draws, [2.5, 50, 97.5]), float((draws <= 0).mean())


def alpha_curve(X, Y, F):
    """Family-honest LOFO at every FIXED alpha on the frozen grid. Reports the curve the nested
    rule is competing against; the minimum here is an oracle and is never used to pick a model."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    rows = {}
    for a in tv.ALPHAS:
        errs = []
        for g in sorted(set(F)):
            te = [i for i, k in enumerate(F) if k == g]
            tr = [i for i, k in enumerate(F) if k != g]
            sc = StandardScaler().fit(X[tr])
            Xt, Xe = sc.transform(X[tr]), sc.transform(X[te])
            mu, V = basis(Y[tr])
            Tt = (Y[tr] - mu) @ V.T
            coef = Ridge(alpha=a).fit(Xt, Tt).predict(Xe)
            for row, j in zip(tv.project(coef @ V + mu), te):
                errs.append(emd(row, Y[j]))
        rows[a] = float(np.mean(errs))
        print("  alpha %-9.2f LOFO EMD %7.2f" % (a, rows[a]))
    return rows


def main() -> int:
    fam_tbl = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    fams = dict(zip(fam_tbl.sample_id.astype(str), fam_tbl.cv_family.astype(str)))
    img = tv.image_features(ROOT)
    soil = tv.soil_from_images(img)
    ids, Y = tv.labels_matrix(ROOT)
    ids = [str(i) for i in ids]
    X = soil.loc[ids, FEATS].to_numpy(float)
    F = np.array([fams[i] for i in ids])
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    print("soils %d  features %d  families %d  sizes %s"
          % (len(ids), X.shape[1], len(set(F)),
             sorted(pd.Series(F).value_counts().tolist(), reverse=True)))

    print("--- real labels, family-honest LOFO ---")
    real = run_sweep(X, Y, F)

    rng = np.random.default_rng(0)
    bias, all_null = [], []
    for t in range(N_NULL):
        perm = rng.permutation(len(ids))
        rows = run_sweep(X, Y[perm], F, verbose=False)
        vals = np.array(sorted(rows.values()))
        b = float(np.median(vals) - vals[0])
        bias.append(b)
        all_null.append(rows)
        print("  null %2d: best-of-%d %.2f | median head %.2f | selection bias %.2f"
              % (t, len(rows), vals[0], np.median(vals), b))
    bn = np.array(bias)
    print("null selection bias (median head - best head): median %.2f  p90 %.2f  max %.2f"
          % (np.median(bn), np.percentile(bn, 90), bn.max()))

    print("--- paired family-clustered bootstrap vs ridge_nested ---")
    base = per_soil_errors(X, Y, F, "ridge_nested")
    order = sorted(real.items(), key=lambda kv: kv[1])[:4]
    for name, _ in order:
        e = per_soil_errors(X, Y, F, name)
        d, ci, p = family_bootstrap(base, e, F)
        print("  %-13s gain over ridge_nested %+.2f EMD  CI95 [%+.2f, %+.2f]  P(no gain) %.3f"
              % (name, d, ci[0], ci[2], p))
    print("--- fixed-alpha curve: an oracle diagnostic, never a selector ---")
    ac = alpha_curve(X, Y, F)
    pd.DataFrame({"lofo_emd": ac}).to_csv(ROOT / "scratch" / "_alpha_curve.csv")
    print("  best fixed alpha %.2f scores %.2f | nested scores %.2f | selection cost %.2f"
          % (min(ac, key=ac.get), min(ac.values()), real["ridge_nested"],
             real["ridge_nested"] - min(ac.values())))
    print("--- verdict ---")
    rb = min(real, key=lambda k: real[k])
    print("  real best head: %s at %.2f | strict-nested ridge here %.2f | recorded 43.0217308796477"
          % (rb, real[rb], real["ridge_nested"]))
    pd.DataFrame({"real": real,
                  "null_median": {k: np.median([r[k] for r in all_null]) for k in real}}
                 ).to_csv(ROOT / "scratch" / "_head_spread.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
