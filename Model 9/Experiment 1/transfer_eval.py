"""transfer_eval.py - assembles Model 5's two camera directions on top of the FROZEN Model 1/2
head, and scores them.

Nothing here is a new model. The head, the curve basis, the ridge, the monotone projection, the
EMD, the CV families and the nested alpha rule are Model 2 E3's, reimplemented line-for-line so
that Model 5's only difference from E3 is the ALIGN step. That claim is not asserted in a comment:
`check_transfer_eval.py` verifies it two ways, by reproducing E3's saved soil matrix value for
value and by reproducing E3's recorded nested control score of 43.0217308796477 (gate G0).

Two things this module is deliberately strict about:

  The resampling unit is the SOIL, not the row. A soil scored in both directions contributes two
  correlated rows; naive row resampling understates every CI by roughly sqrt(2), and an
  understated CI is how a null turns into a discovery.

  Alpha is chosen from inner folds only, and the inner curve is pooled over all inner soils
  rather than averaged across folds - E3's exact rule. A nested score is the score of a
  procedure, and the moment it becomes the score of the best alpha the number is worthless.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

FEATURES: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean",
                             "R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
ALPHAS: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                             300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0,
                             300000.0, 1000000.0)
TILE_PX = 256
MIN_SOIL_FRACTION = 0.50
PC_RANK = 3
DIRECTION_SPEC = {"A": ("Motorola", "Samsung"), "B": ("Samsung", "Motorola")}

ROOT_DEFAULT = Path(__file__).resolve().parents[2]
_SUP = None
_DL = None


def _supports() -> Tuple[List[str], np.ndarray]:
    global _SUP, _DL
    if _SUP is None:
        sub = pd.read_csv(ROOT_DEFAULT / "data" / "sample_submission.csv")
        _SUP = [str(c) for c in sub.columns if str(c) != "sample_id"]
        _DL = np.log10(np.array([float(c) for c in _SUP]))
    return _SUP, _DL


def emd_pair(p, t) -> float:
    _, dl = _supports()
    return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), dl))


def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F, float)), 0, 100)
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 100.0
    return F


def fit_predict(Xtr, Ytr, Xte, alpha, rank=PC_RANK):
    """Model 1's closed-form head, transcribed: scale, rank-k curve basis, ridge, project."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    mu = Ytr.mean(0)
    _, _, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    V = Vt[:rank]
    sc = StandardScaler().fit(Xtr)
    T = (Ytr - mu) @ V.T
    r = Ridge(alpha=alpha).fit(sc.transform(Xtr), T)
    return project(r.predict(sc.transform(Xte)) @ V + mu)


# ---------------------------------------------------------------- data assembly
def config_hash(root) -> str:
    h = sorted(set(pd.read_csv(Path(root) / "data" / "processed_meta" /
                               "manifest_images.csv").config_hash.astype(str)))
    if len(h) != 1:
        raise AssertionError(f"ambiguous config_hash {h}")
    return h[0]


def image_features(root) -> pd.DataFrame:
    """One row per IMAGE: the median of its qualifying tiles' 12 features, plus provenance.

    This is E3's tile->image median step and nothing more. Model 5's ALIGN is applied to this
    table, which is why the table stops here: aggregating further first would smuggle the
    transform past the point where it is defined.
    """
    root = Path(root)
    meta = root / "data" / "processed_meta"
    imgs = pd.read_csv(meta / "manifest_images.csv")
    tiles = pd.read_csv(meta / "manifest_tiles.csv")
    feats = pd.read_csv(root / "Model 1" / "Model 1 Experiment 3" / ".cache" /
                        f"tile_features_{TILE_PX}_{config_hash(root)}.csv")
    keep = tiles[(tiles.tile_size_px == TILE_PX) & tiles.materialized
                 & (tiles.soil_fraction >= MIN_SOIL_FRACTION)]
    k = keep[["tile_path", "sample_id", "parent_image_path", "camera"]].merge(
        feats.drop_duplicates("tile_path"), on="tile_path", how="inner")
    assert len(k) == len(keep), f"{len(keep) - len(k)} qualifying tiles lack features"
    k["camera_fam"] = k.camera.str.split(" ").str[0]
    im = k.groupby(["parent_image_path", "sample_id", "camera_fam"])[list(FEATURES)].median()
    im = im.reset_index().set_index("parent_image_path")
    im["split"] = im.index.map(dict(zip(imgs.processed_path, imgs.split)))
    assert im.split.notna().all(), "an image in the feature table is not in the image manifest"
    assert not im[list(FEATURES)].isna().to_numpy().any(), "NaN in image-level features"
    return im


def soil_from_images(img: pd.DataFrame) -> pd.DataFrame:
    """Median over a soil's images - the frozen second aggregation step.

    Grouping happens before column selection on purpose: `img[FEATURES]` would drop sample_id
    out from under the groupby and raise a KeyError that looks like a data problem.
    """
    return img.groupby("sample_id")[list(FEATURES)].median()


def alignment_target(img: pd.DataFrame) -> np.ndarray:
    """The unlabeled target: image features of every TEST-split image.

    Selected on the split column alone - never on a device name, make or model - and the caller
    has no way to pass labels in, because there is no labels parameter.
    """
    rows = img[img["split"] == "test"]
    if rows.empty:
        raise AssertionError("no test-split images found; the alignment target would be empty")
    return rows[list(FEATURES)].to_numpy(float)


def training_soils(root) -> List[str]:
    s = pd.read_csv(Path(root) / "data" / "processed_meta" / "manifest_samples.csv")
    return s[s.split == "train"].sample_id.astype(str).tolist()


def labels_matrix(root) -> Tuple[List[str], np.ndarray]:
    s = pd.read_csv(Path(root) / "data" / "processed_meta" / "manifest_samples.csv")
    sup, _ = _supports()
    tr = s[s.split == "train"].set_index("sample_id")
    return tr.index.astype(str).tolist(), tr[sup].to_numpy(float)


def direction(name: str, img: pd.DataFrame, labels: pd.DataFrame,
              fams: Dict[str, str]) -> Dict[str, object]:
    """One camera-holdout configuration: fit on the source family, score the other family.

    A soil enters a side only if it has an image from that camera, so the fit and score sets are
    disjoint by construction - a soil cannot be both trained on and scored in the same direction.
    """
    if name not in DIRECTION_SPEC:
        raise AssertionError(f"unknown direction {name!r}")
    fit_fam, eval_fam = DIRECTION_SPEC[name]
    tr_ids = set(labels[labels.split == "train"].sample_id.astype(str))
    have = {f: set(g.sample_id.astype(str)) for f, g in img.groupby("camera_fam")}
    for f in (fit_fam, eval_fam):
        if f not in have:
            raise AssertionError(f"camera family {f!r} has no images")
    fit_ids = sorted(have[fit_fam] & tr_ids)
    eval_ids = sorted(have[eval_fam] & tr_ids)
    # A soil may legitimately appear on both sides of a DIRECTION (21 of 24 were photographed by
    # both cameras). What must never happen is a soil being scored by a model that was fitted on
    # it, and that is enforced per fold in direction_folds(), not here.
    fam_list = sorted({fams[s] for s in fit_ids if s in fams})
    if len(fam_list) < 2:
        raise AssertionError(f"direction {name} has {len(fam_list)} families; cannot select alpha")
    return {"name": name, "fit_camera": fit_fam, "eval_camera": eval_fam,
            "fit_ids": fit_ids, "eval_ids": eval_ids, "fit_families": fam_list,
            "n_fit": len(fit_ids), "n_eval": len(eval_ids),
            "fit_img": img[img.camera_fam == fit_fam],
            "eval_img": img[img.camera_fam == eval_fam]}


def direction_folds(name: str, img: pd.DataFrame, labels: pd.DataFrame,
                    fams: Dict[str, str]) -> List[Dict[str, object]]:
    """Split one direction into folds so that no soil is ever scored by a model that saw it.

    Scoring a soil whose curve was in the fit set would not simulate production, where the 10 test
    soils appear nowhere in training, and it inflates both the aligned and unaligned arms while
    compressing their difference unpredictably. So each fold drops a CV family from the fitting
    camera, estimates the alignment from the surviving fitting-camera images plus the test images,
    and scores the held-out family's images from the evaluation camera.

    A soil with no image in the fitting camera can never be fitted on, so it is scored once
    against the full fitting pool rather than being dropped from the experiment.
    """
    d = direction(name, img, labels, fams)
    fitpool, evals = d["fit_ids"], d["eval_ids"]
    fam_list = sorted({fams[s] for s in fitpool if s in fams})
    folds: List[Dict[str, object]] = []
    for F in fam_list:
        ev = [s for s in evals if fams.get(s) == F and s in set(fitpool)]
        if not ev:
            continue
        fit = [s for s in fitpool if fams.get(s) != F]
        sub_fams = sorted({fams[s] for s in fit})
        if len(sub_fams) < 2:
            raise AssertionError(f"fold {F} leaves {len(sub_fams)} families; cannot select alpha")
        folds.append(_fold(name, F, fit, ev, img, sub_fams))
    always = [s for s in evals if s not in set(fitpool)]
    if always:
        folds.append(_fold(name, "__unseen__", list(fitpool), always, img, fam_list))
    seen = [x for f in folds for x in f["eval_ids"]]
    if len(seen) != len(set(seen)) or set(seen) != set(evals):
        raise AssertionError(f"direction {name}: soils scored {len(seen)} times for "
                             f"{len(evals)} evaluation soils")
    for f in folds:
        if set(f["eval_ids"]) & set(f["fit_ids"]):
            raise AssertionError(f"fold {f['fold']} scores a soil it was fitted on")
    return folds


def _fold(name: str, F: str, fit_ids: List[str], eval_ids: List[str],
          img: pd.DataFrame, fit_families: List[str]) -> Dict[str, object]:
    fit_fam, eval_fam = DIRECTION_SPEC[name]
    return {"name": name, "fold": F, "fit_ids": sorted(fit_ids), "eval_ids": sorted(eval_ids),
            "fit_families": fit_families,
            "n_fit": len(fit_ids), "n_eval": len(eval_ids),
            "fit_img": img[(img.camera_fam == fit_fam) & img.sample_id.isin(fit_ids)],
            "eval_img": img[(img.camera_fam == eval_fam) & img.sample_id.isin(eval_ids)],
            "target_img": img[img.split == "test"], "target_rows": int((img.split == "test").sum())}


# ---------------------------------------------------------------- scoring
def alpha_status(alpha: float) -> str:
    if abs(float(alpha) - ALPHAS[0]) < 1e-12:
        return "FLOOR"
    if abs(float(alpha) - ALPHAS[-1]) < 1e-12:
        return "CEILING"
    return "INTERIOR"


def nested_predict(train_X, train_Y, train_fam, test_X, alphas=ALPHAS) -> Dict[str, object]:
    """LOGO-CV inside the FIT set only, alpha by inner folds, then fit-on-all and predict.

    The held-out evaluation set never influences alpha, and no label from it is ever read here.
    """
    train_X = np.asarray(train_X, float)
    train_Y = np.asarray(train_Y, float)
    fams = sorted(set(train_fam))
    if len(fams) < 2:
        raise AssertionError("need at least two families to select alpha honestly")
    err: Dict[float, Dict[str, Dict[str, float]]] = {}
    for al in alphas:
        per_fold = {}
        for F in fams:
            hold = np.array([i for i, f in enumerate(train_fam) if f == F])
            keep = np.array([i for i in range(len(train_fam)) if train_fam[i] != F])
            P = fit_predict(train_X[keep], train_Y[keep], train_X[hold], al)
            per_fold[F] = {i: emd_pair(P[k], train_Y[i]) for k, i in enumerate(hold)}
        err[al] = per_fold
    curve = {al: float(np.mean([v for F in fams for v in err[al][F].values()])) for al in alphas}
    alpha = min(curve, key=curve.get)
    pred = fit_predict(train_X, train_Y, np.asarray(test_X, float), alpha)
    return {"pred": pred, "alpha": alpha, "status": alpha_status(alpha),
            "selection_emd": curve[alpha], "best_emd": min(curve.values()),
            "grid_spread": max(curve.values()) - min(curve.values()), "sweep": curve,
            "n_families": len(fams)}


def oracle_sweep(train_X, train_Y, test_X, test_Y, alphas=ALPHAS) -> Dict[str, object]:
    """Fit-on-all then score the eval rows at every grid alpha, and report the best.

    This is a DIAGNOSTIC, never a selector: choosing an alpha from it would be selecting on the
    scoring labels. It exists because section 5a requires the oracle alpha and the grid spread to
    be reported next to the nested choice, so a reader can see how much the alpha mattered at all
    rather than being told.
    """
    train_X = np.asarray(train_X, float)
    train_Y = np.asarray(train_Y, float)
    test_X = np.asarray(test_X, float)
    test_Y = np.atleast_2d(np.asarray(test_Y, float))
    if len(test_X) != len(test_Y):
        raise AssertionError(f"{len(test_X)} eval rows vs {len(test_Y)} eval labels")
    curve = {al: float(np.mean([emd_pair(fit_predict(train_X, train_Y, test_X, al)[i], test_Y[i])
                                for i in range(len(test_Y))])) for al in alphas}
    best = min(curve, key=curve.get)
    return {"alpha": best, "status": alpha_status(best), "emd": curve[best],
            "spread": max(curve.values()) - min(curve.values()), "curve": curve}


def no_holdout_control_score(root) -> float:
    """E3's own nested protocol on all 24 labelled soils with no alignment - gate G0's number."""
    root = Path(root)
    tr_ids, Y = labels_matrix(root)
    fam_tbl = pd.read_csv(root / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    fams = dict(zip(fam_tbl.sample_id.astype(str), fam_tbl.cv_family))
    X = soil_from_images(image_features(root)).loc[tr_ids, list(FEATURES)].to_numpy(float)
    fam_list = [fams[s] for s in tr_ids]
    fams_sorted = sorted(set(fam_list))
    err: Dict[float, Dict[str, Dict[str, float]]] = {}
    for al in ALPHAS:
        pf = {}
        for F in fams_sorted:
            hold = np.array([i for i, f in enumerate(fam_list) if f == F])
            keep = np.array([i for i in range(len(fam_list)) if fam_list[i] != F])
            P = fit_predict(X[keep], Y[keep], X[hold], al)
            pf[F] = {i: emd_pair(P[k], Y[i]) for k, i in enumerate(hold)}
        err[al] = pf
    nested = {}
    for F in fams_sorted:
        inner = [G for G in fams_sorted if G != F]
        curve = {al: float(np.mean([v for G in inner for v in err[al][G].values()]))
                 for al in ALPHAS}
        ba = min(curve, key=curve.get)
        nested[F] = err[ba][F]
    all_e = [v for F in fams_sorted for v in nested[F].values()]
    return float(np.mean(all_e))


# ---------------------------------------------------------------- uncertainty
def _percentiles(draws: np.ndarray) -> Tuple[float, float]:
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def naive_bootstrap(rows, n: int = 4000, seed: int = 20260934) -> Tuple[float, float, float]:
    rows = np.asarray(rows, float)
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(rows))
    draws = np.array([rows[rng.choice(idx, size=len(idx), replace=True)].mean() for _ in range(n)])
    return float(rows.mean()), *_percentiles(draws)


def clustered_bootstrap(rows, clusters, n: int = 4000,
                        seed: int = 20260934) -> Tuple[float, float, float]:
    """Resample SOILS. Two rows from one soil are never separated.

    Row-level resampling would treat a soil's appearance in both directions as two independent
    observations and narrow every interval, flattering exactly the clause that decides whether a
    submission is allowed.
    """
    rows = np.asarray(rows, float)
    clusters = list(clusters)
    if len(rows) != len(clusters):
        raise AssertionError(f"{len(rows)} values vs {len(clusters)} clusters")
    if any(c is None for c in clusters):
        raise AssertionError("a row has no cluster id; clustering would be silently skipped")
    by = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    keys = sorted(by)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n))
    for b in range(int(n)):
        pick = rng.choice(np.arange(len(keys)), size=len(keys), replace=True)
        take = np.concatenate([np.array(by[keys[k]]) for k in pick])
        draws[b] = rows[take].mean()
    lo, hi = _percentiles(draws)
    return float(rows.mean()), lo, hi
