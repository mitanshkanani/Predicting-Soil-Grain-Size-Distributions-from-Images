"""head_d50.py - the two-stage head Experiment 1 is testing, against the frozen one.

THE HYPOTHESIS UNDER TEST
  The frozen head treats the map from median grain size to curve shape as LINEAR. Because the
  label space is 91.4% one-dimensional (PC1 vs log10 D50, rho = 0.9949), that head spends three
  free coefficients on a near-line. If the true relation is quadratic in log10 D50, the frozen
  form is leaving real accuracy on the table - and, crucially, it is leaving it in the SHAPE,
  which section 9.3 showed no feature can predict (every feature gives negative R-squared on
  the shape residual). Making a fixed map a better map is the only route to that block.

WHY IT IS BUILT AS TWO STAGES RATHER THAN ONE
  Stage 1 predicts a scalar; stage 2 maps that scalar to 11 columns. The alternative -
  predicting the 11 columns directly, or predicting rank-3 PCA coefficients - throws away the
  1-D structure that is the entire reason this experiment exists. Section 9.7 measured that
  the matched-loss alternative is 28.84 EMD WORSE, because the 11 columns are not 11
  independent targets.

THE THREE ARMS THIS FILE PROVIDES
  deg=2  Arm E1-A, the primary. The polynomial map [1, y, y^2].
  deg=1  Arm E1-B, the control that makes the experiment an experiment. Identical in every
        other respect; it isolates the degree of the polynomial and nothing else.
  deg=3  Reported as a supporting check, not a candidate. The plan predicts it does not beat
        degree 2 (17.85 vs 18.02 at true D50), which would show the gain saturates at two
        parameters rather than drifting on with parameter count.

  `use_true_d50` supplies the target's own D50 instead of a prediction. That is arm E1-D, the
  upper bound for any D50-parameterised model: it removes feature noise entirely and measures
  the shape map alone. E1-A means nothing without it.

THE PROJECTION IS THE FROZEN ONE, UNCHANGED
  clip to [0,100], then maximum.accumulate along the supports, then force exactly 100 at
  200 mm. It is applied to the reconstructed curve, not to the polynomial output alone, so a
  quadratic that overshoots is corrected exactly as a linear one would be.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

import d50 as _d50

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ALPHAS: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                             300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0,
                             300000.0, 1000000.0)


def design(v: np.ndarray, deg: int) -> np.ndarray:
    """The polynomial basis [1, v, v^2, ...] as an (n, deg+1) design matrix.

    v must already be a well-scaled parameter. Callers pass log10 D50, whose training range
    is about 2.5 decades; passing raw millimetres instead would make the design matrix
    numerically hopeless and is the kind of unit slip this function's shape is meant to make
    obvious.
    """
    v = np.asarray(v, float).ravel()
    return np.column_stack([v ** k for k in range(deg + 1)])


class HeadD50:
    """predict D50 by ridge, then map through a polynomial in log10 D50, then project.

    Every fitted quantity - the scaler, the ridge, and the polynomial coefficients - is
    learned from the rows passed to fit() and from nothing else. That is what makes the
    leave-one-family-out score honest: the held-out family cannot influence anything about
    the model that scores it.
    """

    def __init__(self, deg: int = 2, alpha: float = 3.0, use_true_d50: bool = False):
        if deg < 1:
            raise AssertionError("deg must be >= 1; the linear form is the control arm, not a limit")
        self.deg = deg
        self.alpha = float(alpha)
        self.use_true_d50 = bool(use_true_d50)
        self.sc_: StandardScaler | None = None
        self.ridge_: Ridge | None = None
        self.coef_: np.ndarray | None = None   # (deg+1, 11)
        self.alpha_: float | None = None

    # -- stage 1 ---------------------------------------------------------------
    def fit_scalar(self, X: np.ndarray, Y: np.ndarray) -> "HeadD50":
        """Learn scaler + ridge mapping features -> log10 D50. Ignored when use_true_d50."""
        y = _d50.logd50(Y)
        self.sc_ = StandardScaler().fit(X)
        self.ridge_ = Ridge(alpha=self.alpha).fit(self.sc_.transform(X), y)
        return self

    def predict_scalar(self, X: np.ndarray) -> np.ndarray:
        if self.ridge_ is None or self.sc_ is None:
            raise AssertionError("fit_scalar has not run")
        return self.ridge_.predict(self.sc_.transform(X))

    # -- stage 2 ---------------------------------------------------------------
    def fit_shape(self, Y: np.ndarray) -> "HeadD50":
        """Least-squares polynomial map from log10 D50 to the 11 curve columns.

        Fitted on the TRAINING rows only. When use_true_d50 is set, the abscissa is each
        training curve's own D50, so the map is the one E1-D measures.
        """
        y = _d50.logd50(Y) if not self.use_true_d50 else np.log10(_d50.diameter_at(Y, 50.0))
        A = design(y, self.deg)
        self.coef_ = np.linalg.lstsq(A, np.asarray(Y, float), rcond=None)[0]
        return self

    def predict_shape(self, y: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise AssertionError("fit_shape has not run")
        return design(y, self.deg) @ self.coef_

    # -- projection ------------------------------------------------------------
    def predict(self, X: np.ndarray, Y_for_true_d50: np.ndarray | None = None) -> np.ndarray:
        """Predict curves for X. Ignores Y_for_true_d50 unless use_true_d50 is set.

        Y_for_true_d50 is required only for the true-D50 arm, where the abscissa IS the
        label; passing it for the predicted arms would leak the answer into its own score.
        """
        if self.use_true_d50:
            if Y_for_true_d50 is None:
                raise AssertionError("use_true_d50 needs the true curves to read D50 from")
            y = np.log10(_d50.diameter_at(Y_for_true_d50, 50.0))
        else:
            y = self.predict_scalar(X)
        from transfer_eval import project  # the frozen projection, imported not copied
        return project(self.predict_shape(y))

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "HeadD50":
        if not self.use_true_d50:
            self.fit_scalar(X, Y)
        self.fit_shape(Y)
        return self


def alpha_curve(X: np.ndarray, Y: np.ndarray, fams: Sequence[str], deg: int,
                alphas: Sequence[float] = ALPHAS) -> Dict[float, float]:
    """Pooled family-out error at every grid alpha, on the training rows handed in.

    Structure matches m7a_eval.alpha_curve exactly - including its inner leave-one-family-out
    loop - so that the alpha-selection rule is byte-for-byte the same rule the frozen baseline
    uses. Changing the ruler and the treatment at the same time would confound the result.
    """
    F = np.asarray(fams)
    inner = sorted(set(F))
    if len(inner) < 2:
        raise AssertionError("alpha selection needs at least two families to be honest")
    curve: Dict[float, float] = {}
    for al in alphas:
        errs: List[float] = []
        for g in inner:
            hold = np.where(F == g)[0]
            keep = np.where(F != g)[0]
            if not hold.size or not keep.size:
                continue
            h = HeadD50(deg=deg, alpha=al).fit(X[keep], Y[keep])
            P = h.predict(X[hold])
            errs.extend(_emd(P[k], Y[i]) for k, i in enumerate(hold))
        curve[float(al)] = float(np.mean(errs)) if errs else float("inf")
    return curve


def _emd(P: np.ndarray, T: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(P, float) - np.asarray(T, float)), _d50.DL))


def lofo(X: np.ndarray, Y: np.ndarray, fams: Sequence[str], deg: int,
         alphas: Sequence[float] = ALPHAS) -> Tuple[np.ndarray, Dict[str, float]]:
    """One honest error per soil: leave-one-family-out, alpha selected inside the fit set.

    This is the registered Model 7 protocol, transcribed. For outer family g, the rows used
    to choose alpha exclude g entirely - not merely from the final fit, but from every inner
    fold - which is the correction over the older Models 1/2 rule that AGENT_BRIEF.md:716-721
    discloses and that the audit independently confirmed.
    """
    F = np.asarray(fams)
    out = np.full(len(X), np.nan)
    chosen: Dict[str, float] = {}
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        if not te.size or not tr.size:
            raise AssertionError("fold %s is empty on one side" % g)
        curve = alpha_curve(X[tr], Y[tr], list(F[tr]), deg, alphas)
        al = min(curve, key=curve.get)
        chosen[g] = float(al)
        h = HeadD50(deg=deg, alpha=al).fit(X[tr], Y[tr])
        P = h.predict(X[te])
        for k, i in enumerate(te):
            out[i] = _emd(P[k], Y[i])
    if np.isnan(out).any():
        raise AssertionError("a soil was never scored")
    return out, chosen
