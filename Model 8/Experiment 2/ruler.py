"""ruler.py - the frozen rank-3 head under the strictly family-honest nested protocol.

WHY THIS IS TRANSCRIBED RATHER THAN IMPORTED
  Model 7's `m7a_eval.py` holds the registered implementation, and the honest thing is that
  Experiment 1 is measured against THAT protocol. But importing Model 7's module would make
  this experiment's ruler a function of another model's source tree - if that file changed,
  Model 8's numbers would move for reasons that have nothing to do with the hypothesis.

  So the protocol is transcribed here, built only on the frozen primitives in
  `transfer_eval` (the 12 features, the ridge, the rank-3 curve basis, the projection, the
  EMD). It is a copy of a RULE, not of a result.

  Being a copy, it could in principle be a wrong copy - which is exactly why gate G1 exists.
  G1 requires this module to reproduce the registered value 43.453225201811563 before any
  other number in the run is reported. A transcription error cannot survive that gate.

THE RULE, precisely
  For each held-out family g:
    1. Fit set = every soil whose family is not g.
    2. Choose alpha by leave-one-family-out WITHIN the fit set. The inner folds are built
       from the fit set's families only, so family g is absent from every inner fold's
       training data as well as from the final fit.
    3. Refit the scaler and the curve basis on the fit set and predict the held-out family.

  Step 2 is the whole point. The older Models 1/2 rule chose alpha from inner folds whose
  fits still contained family g (disclosed at AGENT_BRIEF.md:716-721), which is why the
  historical anchor 43.0217 and this baseline 43.4532 differ by 0.431 EMD. Mixing the two
  would put a protocol difference into a quantity judged against a 3.00 EMD bar.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import transfer_eval as tv  # noqa: E402  frozen: FEATURES, ALPHAS, fit_predict, emd_pair

# The registered Model 7 internal baseline under this protocol. Transcribed once, used only
# by the G1 gate. It is an INTERNAL CV number on 24 labelled soils and is not a Kaggle score.
INTERNAL_BASELINE = 43.453225201811563

FEATURES: Tuple[str, ...] = tuple(tv.FEATURES)
ALPHAS: Tuple[float, ...] = tuple(tv.ALPHAS)


def families() -> Dict[str, str]:
    """sample_id -> CV family. Read as a fold key only; never a column of the design."""
    t = pd.read_csv(ROOT / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    return dict(zip(t.sample_id.astype(str), t.cv_family.astype(str)))


def design(root=None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """The frozen 24 x 12 design, its 11-column labels, the family key and the soil ids."""
    root = ROOT if root is None else Path(root)
    ids, Y = tv.labels_matrix(root)
    ids = [str(i) for i in ids]
    img = tv.image_features(root)
    X = tv.soil_from_images(img).loc[ids, list(FEATURES)].to_numpy(float)
    fam_of = families()
    missing = [i for i in ids if i not in fam_of]
    if missing:
        raise AssertionError("%d soils have no CV family, so a fold would silently drop them"
                             % len(missing))
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise AssertionError("non-finite design or label matrix")
    return X, Y, [fam_of[i] for i in ids], ids


def alpha_curve(X: np.ndarray, Y: np.ndarray, fams: Sequence[str]) -> Dict[float, float]:
    """Pooled family-out error at every grid alpha, computed on the rows handed to it.

    The pool is over soils, not an average of per-family means: folds are unequal in size
    (1x10, 2x5, 4) and averaging the two would weight a 4-soil family the same as a 1-soil
    one. This matches the frozen rule.
    """
    F = np.asarray(fams)
    inner = sorted(set(F))
    if len(inner) < 2:
        raise AssertionError("alpha selection needs at least two families to be honest")
    curve: Dict[float, float] = {}
    for al in ALPHAS:
        errs: List[float] = []
        for g in inner:
            hold = np.where(F == g)[0]
            keep = np.where(F != g)[0]
            if not hold.size or not keep.size:
                continue
            P = tv.fit_predict(X[keep], Y[keep], X[hold], al)
            errs.extend(tv.emd_pair(P[k], Y[i]) for k, i in enumerate(hold))
        curve[float(al)] = float(np.mean(errs)) if errs else float("inf")
    return curve


def select_alpha(X: np.ndarray, Y: np.ndarray, fams: Sequence[str]) -> float:
    """The grid minimum on THIS training set. An oracle is never consulted anywhere here."""
    curve = alpha_curve(X, Y, fams)
    return min(curve, key=curve.get)


def lofo_errors(X: np.ndarray, Y: np.ndarray, fams: Sequence[str],
                return_alphas: bool = False):
    """One honest error per soil under the registered protocol."""
    F = np.asarray(fams)
    out = np.full(len(X), np.nan)
    alphas: Dict[str, float] = {}
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        if not te.size or not tr.size:
            raise AssertionError("fold %s is empty on one side; it cannot be scored honestly"
                                 % g)
        al = select_alpha(X[tr], Y[tr], list(F[tr]))
        alphas[g] = float(al)
        P = tv.fit_predict(X[tr], Y[tr], X[te], al)
        for k, i in enumerate(te):
            out[i] = tv.emd_pair(P[k], Y[i])
    if np.isnan(out).any():
        raise AssertionError("a soil was never scored")
    return (out, alphas) if return_alphas else out


def fold_report(fams: Sequence[str]) -> Dict[str, object]:
    """The integrity claim, made checkable: no fold may be scored by a fit that saw it."""
    F = np.asarray(fams)
    leaks, sizes, seen = 0, {}, []
    for g in sorted(set(F)):
        te = list(np.where(F == g)[0])
        tr = list(np.where(F != g)[0])
        seen += te
        sizes[g] = len(te)
        if set(F[te]) & set(F[tr]):
            leaks += 1
        if set(te) & set(tr):
            leaks += 1
        if len({F[i] for i in tr}) < 2:
            leaks += 1
    return {"family_leaks": leaks, "n_folds": len(set(F)), "n_scored": len(set(seen)),
            "fold_sizes": sizes}
