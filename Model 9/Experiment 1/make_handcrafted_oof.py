"""make_handcrafted_oof.py - family-held-out OOF predictions of the FROZEN handcrafted head.

WHY THIS EXISTS
  Model 9's whole point is a CNN that is a DIFFERENT model class from the 55.80591 handcrafted
  head, so that a blend of the two can beat either alone (a blend helps only when the component
  errors decorrelate; Model 8 E4 found handcrafted-vs-handcrafted blending dead precisely because
  those errors correlated 0.86-0.95). To choose the blend weight HONESTLY - on the 24 labelled
  soils, family-held-out, never on the test set - the notebook needs the handcrafted head's
  out-of-fold predictions on the same soils the CNN is scored on.

  This script produces them, once, locally, and writes handcrafted_oof_train.csv. It is shipped
  INTO the Kaggle ZIP so the notebook never has to re-run the frozen head on Kaggle.

THE HEAD, EXACTLY
  alpha is FIXED at 3.0 - the externally-optimal value Model 8 E2 established and the value the
  55.80591 submission (E2-A) itself used. Fixing it means the OOF here corresponds to the SAME
  head whose test predictions are the 55.80591 CSV, so blending the two is coherent. As a
  consequence the OOF mean EMD must reproduce the registered oracle-alpha-3 anchor 41.1475...,
  which is this script's self-check.

  An internal CV mean on 24 soils is NOT a Kaggle score and is never subtracted from one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (HERE,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ruler as R          # noqa: E402  copied, family-held-out protocol
import transfer_eval as tv  # noqa: E402  copied, frozen head + EMD

ALPHA = 3.0
ORACLE_ANCHOR = 41.1475158656982032   # oracle alpha=3.0, registered in ruler/E4


def main() -> int:
    X, Y, fams, ids = R.design()
    F = np.asarray(fams)
    _, dl = tv._supports()
    sup = [str(c) for c in tv._supports()[0]]

    oof = np.full_like(Y, np.nan)
    for g in sorted(set(F)):
        te = np.where(F == g)[0]
        tr = np.where(F != g)[0]
        P = tv.fit_predict(X[tr], Y[tr], X[te], ALPHA)   # family g never in the fit
        for k, i in enumerate(te):
            oof[i] = P[k]
    assert not np.isnan(oof).any(), "a soil was never predicted"

    per = np.array([tv.emd_pair(oof[i], Y[i]) for i in range(len(ids))])
    mean_emd = float(per.mean())
    gap = abs(mean_emd - ORACLE_ANCHOR)
    print("handcrafted OOF (alpha=3.0, family-held-out)")
    print("  mean EMD %.13f   anchor %.13f   gap %.2e" % (mean_emd, ORACLE_ANCHOR, gap))
    assert gap < 1e-9, "OOF does not reproduce the oracle-alpha-3 anchor; head mismatch"

    cols = ["d_%s" % s for s in sup]
    out = pd.DataFrame(oof, columns=cols)
    out.insert(0, "sample_id", ids)
    out.insert(1, "cv_family", [fams[i] for i in range(len(ids))])
    out["oof_emd"] = per
    out.to_csv(HERE / "handcrafted_oof_train.csv", index=False, encoding="ascii")

    # labels, shipped alongside so the notebook scores the blend without re-reading manifests
    lab = pd.DataFrame(Y, columns=cols)
    lab.insert(0, "sample_id", ids)
    lab.insert(1, "cv_family", [fams[i] for i in range(len(ids))])
    lab.to_csv(HERE / "train_labels_matrix.csv", index=False, encoding="ascii")

    print("  wrote handcrafted_oof_train.csv  (%d soils x 11)" % len(ids))
    print("  wrote train_labels_matrix.csv")
    print("  diameter columns:", ", ".join(sup))
    print("[ok] anchor reproduced; OOF is coherent with the 55.80591 head")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
