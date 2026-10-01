"""build_e2.py - Experiment 2's three pre-registered arms, scored internally and fitted for submission.

THE QUESTION
  Experiment 1 confirmed a real structural fact about the label space (the log10 D50 -> shape
  map is curved, +6.09 EMD, P=0.0073) and then showed it cannot be reached by predicting D50
  first. That left the project with 24 internal numbers and zero understanding of why internal
  43 becomes external 60.

  This experiment stops measuring interpolation and produces actual submissions. Internal CV
  cannot answer the transfer question: all 10 test soils sit outside the training feature
  hull, so there is no honest way to make our own validation look like the test set. We do not
  have 10 out-of-hull soils WITH LABELS. So we submit and observe.

THE THREE ARMS, fixed before any number was seen
  E2-A  the incumbent: the frozen head, 12 features. The number to beat. Not a new arm.
  E2-B  colour dropped: the 5 texture-core features. Tests whether removing the block the test
        data actually moved in helps transfer.
  E2-C  colour made INVARIANT: the same 5 features plus each colour channel converted to its
        within-training-set percentile. Keeps the information, removes the scale shift.

  E2-C is not a third guess. The project's best external result (60.56167) came from the arm
  with the SMALLEST transfer penalty among its pair, which is an invariance argument rather
  than a deletion argument. Section 9.6 also showed the internal verdict on dropping colour
  flips sign by scored subset (-0.47 all 24, -2.28 on the 12 most isolated, +4.75 on the 6
  most isolated) - undecidable internally for two experiments running.

WHY ONLY THREE ARMS
  Every prior experiment tested 3-5 arms and reported the winner. The project's own measurement
  is that selecting the best of 15 heads on SHUFFLED LABELS buys a median 15.29 EMD apparent
  gain - so a 2.96 EMD "improvement" is exactly what picking the luckiest of five buys. Three
  arms, declared in advance, no grid. That discipline is the actual fix for 39.26 -> 83.94.

THE ID CONTRACT, and the trap in it
  The competition's sample_id values are NOT the internal sample ids. The test soil is called
  'Muenster_BS6_9,0-10m' internally and 'HPC_Muenster_BS6_9_0-10m' on the board, and the
  internal one arrives mojibake-corrupted ('M?nster...') through the feature CSV's encoding.
  So the join goes through manifest_samples.submission_id, which is the authoritative mapping,
  and never through string surgery on the id. Getting this wrong would produce a submission
  with plausible-looking rows attached to the wrong soils.

INTERNAL NUMBERS ARE REPORTED, NEVER USED TO SELECT THE SUBMISSION
  That is the whole point. If we pick by internal score we reproduce the exact failure that
  made the three best internal numbers in project history map to the three worst transfers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ruler as R          # noqa: E402  the registered nested protocol (transcribed in E1)
import transfer_eval as tv  # noqa: E402  frozen primitives

SUPPORTS: Tuple[str, ...] = ("0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                             "2", "6.3", "20", "63", "200")
CORE5: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean")
COLOUR7: Tuple[str, ...] = ("R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
G0_VALUE = 43.453225201811563


# ------------------------------------------------------------------ data
def load_features() -> Tuple[pd.DataFrame, Dict[str, str]]:
    """The cached soil-level features, and the internal-id -> submission-id mapping.

    Reads features_soil.csv directly: no image tiles are opened anywhere in this experiment.
    """
    f = pd.read_csv(ROOT / "Model 1" / "Model 1 Experiment 3" / "features_soil.csv")
    m = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_samples.csv")
    sub_id = dict(zip(m.sample_id.astype(str), m.submission_id.astype(str)))
    missing = [s for s in f.sample_id.astype(str) if s not in sub_id]
    if missing:
        raise AssertionError("%d soils have no submission_id, e.g. %r"
                             % (len(missing), missing[:3]))
    return f, sub_id


def rank_normalise(train_X: np.ndarray, other_X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map each column to its percentile within the TRAIN distribution.

    Two properties matter here. It is monotone, so it destroys no ordinal information about
    colour; and its only fitted quantity is the training quantile function, so a test soil far
    outside the training range lands near 0 or 1 rather than at an extrapolated value. That
    second property is the entire point - it is what makes the head invariant to exactly the
    shift the test set exhibits.

    Training columns are mapped through themselves (so the transform is identical at fit and
    predict time); the training quantiles are the only thing carried across.
    """
    tr = np.empty_like(train_X, dtype=float)
    te = np.empty_like(other_X, dtype=float)
    for j in range(train_X.shape[1]):
        col = train_X[:, j]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(len(col), float)
        ranks[order] = np.arange(len(col), dtype=float)
        tr[:, j] = (ranks + 0.5) / len(col)
        te[:, j] = np.searchsorted(np.sort(col), other_X[:, j], side="left") / len(col)
    return tr, te


# ------------------------------------------------------------------ arms
def arm_designs(f: pd.DataFrame, ids: List[str], test_ids: List[str]) -> Dict[str, Tuple]:
    """The three designs, each as (X_train, X_test, column names).

    Accepts the frame indexed by sample_id or as read, so the caller's ordering cannot
    silently change what is selected.
    """
    if f.index.name != "sample_id":
        f = f.set_index("sample_id")
    tr = f[f["split"] == "train"].loc[ids]
    te = f[f["split"] == "test"].loc[test_ids]
    A = (tr[list(R.FEATURES)].to_numpy(float), te[list(R.FEATURES)].to_numpy(float),
         tuple(R.FEATURES))
    B = (tr[list(CORE5)].to_numpy(float), te[list(CORE5)].to_numpy(float), CORE5)

    ctr, cte = rank_normalise(tr[list(COLOUR7)].to_numpy(float),
                              te[list(COLOUR7)].to_numpy(float))
    C = (np.hstack([B[0], ctr]), np.hstack([B[1], cte]), CORE5 + COLOUR7)
    return {"E2-A": A, "E2-B": B, "E2-C": C}


def score_internal(X: np.ndarray, Y: np.ndarray, fams: List[str]) -> float:
    """Registered nested LOFO on one design. Reported, never used to choose a submission."""
    return float(R.lofo_errors(X, Y, fams).mean())


def fit_full(X: np.ndarray, Y: np.ndarray, fams: List[str], Xte: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit on all 24 training soils with alpha chosen by nested LOFO, then predict the test rows."""
    al = R.select_alpha(X, Y, fams)
    return tv.fit_predict(X, Y, Xte, al), float(al)


# ------------------------------------------------------------------ contract
def assert_contract(df: pd.DataFrame, sample_ids: List[str], tag: str) -> None:
    """The five submission rules, enforced before any file is written for upload.

    Checked against Model 1 Experiment 3's real submission this session rather than assumed,
    because a malformed submission wastes the slot it was meant to buy.
    """
    assert list(df.columns) == ["sample_id"] + list(SUPPORTS), \
        "%s: columns must be sample_id + the 11 supports in order" % tag
    assert len(df) == 10, "%s: expected 10 test rows, got %d" % (tag, len(df))
    assert list(df.sample_id) == sample_ids, \
        "%s: sample_id order does not match sample_submission.csv" % tag
    V = df[list(SUPPORTS)].to_numpy(float)
    assert np.isfinite(V).all(), "%s: non-finite value in the submission" % tag
    assert (V >= -1e-9).all() and (V <= 100 + 1e-9).all(), "%s: value outside [0,100]" % tag
    assert (np.diff(V, axis=1) >= -1e-9).all(), "%s: curve is not monotone" % tag
    assert np.allclose(V[:, -1], 100.0), "%s: 200 mm column is not exactly 100" % tag


def main() -> int:
    print("=" * 76)
    print("MODEL 8 / EXPERIMENT 2 - colour robustness, three arms, submission-ready")
    print("=" * 76)

    f, sub_id = load_features()
    sample_sub = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    sample_ids = sample_sub.sample_id.astype(str).tolist()

    # ruler.design() returns (X, Y, fam_list, ids) - in THAT order. Unpacking it wrongly
    # binds `ids` to the 24x12 design matrix, and indexing a frame with a 2-D array is
    # exactly the "multidimensional key" error below.
    _, Y, fams, ids = R.design()
    f = f.set_index("sample_id")
    test_ids = [s for s in f.index.astype(str) if f.loc[s, "split"] == "test"]
    if len(test_ids) != 10:
        raise AssertionError("expected 10 test soils in features_soil.csv, got %d" % len(test_ids))

    # The internal id -> competition id map, in the board's row order.
    id_map = {s: sub_id[s] for s in test_ids}
    missing = [s for s in sample_ids if s not in set(id_map.values())]
    if missing:
        raise AssertionError("%d submission ids have no soil, e.g. %r" % (len(missing), missing[:3]))
    print("[ok] %d train soils, %d test soils, id map complete" % (len(ids), len(test_ids)))

    designs = arm_designs(f, ids, test_ids)

    # ---- G0: the ruler, before anything else ------------------------------
    print("\n--- G0: reproduce the registered frozen baseline ---")
    g0 = score_internal(designs["E2-A"][0], Y, fams)
    print("[%s] G0 frozen 12-feature LOFO = %.15f (|gap| = %.3e)"
          % ("ok" if abs(g0 - G0_VALUE) <= 1e-6 else "FAIL", g0, abs(g0 - G0_VALUE)))
    if abs(g0 - G0_VALUE) > 1e-6:
        print("GATE G0 FAILED - nothing else in this run is reported.")
        (HERE / "reproduce_anchor.txt").write_text(
            "G0 FAILED\nobserved %.15f\nregistered %.15f\n" % (g0, G0_VALUE), encoding="ascii")
        return 1
    (HERE / "reproduce_anchor.txt").write_text(
        "G0 PASSED\nobserved %.15f\nregistered %.15f\ngap %.3e\n"
        % (g0, G0_VALUE, abs(g0 - G0_VALUE)), encoding="ascii")

    rep = R.fold_report(fams)
    print("[ok] fold integrity: %d folds, %d scored, %d leaks"
          % (rep["n_folds"], rep["n_scored"], rep["family_leaks"]))

    # ---- internal scores, reported only -----------------------------------
    print("\n--- internal nested-LOFO (REPORTED, never used to select) ---")
    internal: Dict[str, float] = {}
    for name, (Xtr, _, cols) in designs.items():
        internal[name] = score_internal(Xtr, Y, fams)
        print("    %s (%2d feats): %.4f" % (name, len(cols), internal[name]))
    print("    NOTE: internal ranking is NOT the submission rule. See the module docstring.")

    # ---- fit each arm on all 24, predict the 10 --------------------------
    print("\n--- fit on all 24 training soils, predict 10 test soils ---")
    written: List[str] = []
    rows = []
    for name, (Xtr, Xte, cols) in designs.items():
        P, al = fit_full(Xtr, Y, fams, Xte)
        # Map internal soil ids -> competition ids, in sample_submission.csv order.
        by_sub = {id_map[t]: P[k] for k, t in enumerate(test_ids)}
        data = {"sample_id": sample_ids}
        for j, s in enumerate(SUPPORTS):
            data[s] = [by_sub[i][j] for i in sample_ids]
        df = pd.DataFrame(data)[list(SUPPORTS)].round(6)
        df.insert(0, "sample_id", sample_ids)
        assert_contract(df, sample_ids, name)
        out = HERE / ("Submission_Model8_E2_%s.csv" % name.replace("-", ""))
        df.to_csv(out, index=False)
        written.append(out.name)
        print("    %s  alpha=%-8g -> %s  [contract ok]" % (name, al, out.name))
        rows.append({"arm": name, "n_features": len(cols), "internal_lofo": internal[name],
                     "alpha": al, "submission": out.name})

    # ---- what the arms actually disagree about --------------------------
    print("\n--- how far apart are the three predictions? ---")
    P = {n: pd.read_csv(HERE / ("Submission_Model8_E2_%s.csv" % n.replace("-", "")))
            [list(SUPPORTS)].to_numpy(float) for n in designs}
    # Compare the arms against each other on the metric's own abscissa, computed here rather
    # than via transfer_eval.emd_pair: that helper builds its grid from sample_submission.csv
    # by str()-ing the column names, which yields a 2-D array on this pandas version.
    DL = np.log10(np.array([float(s) for s in SUPPORTS]))
    def pair_emd(p, t):
        return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))
    for a, b in (("E2-A", "E2-B"), ("E2-A", "E2-C"), ("E2-B", "E2-C")):
        mean_pairwise = float(np.mean([pair_emd(P[a][i], P[b][i]) for i in range(10)]))
        print("    mean EMD between %s and %s predictions: %.3f" % (a, b, mean_pairwise))

    pd.DataFrame(rows).to_csv(HERE / "arm_comparison.csv", index=False)
    (HERE / "run_manifest.json").write_text(json.dumps({
        "arms": list(designs),
        "internal_lofo": internal,
        "g0_reproduced": g0,
        "submitted_files": written,
        "note": "Internal scores are reported only. Selection among arms is EXTERNAL by "
                "design; see plan.md section 6.",
    }, indent=2), encoding="ascii")

    print("\n" + "=" * 76)
    print("Internal step complete. %d submission CSVs written, contract asserted." % len(written))
    print("STATUS: awaiting Kaggle result. No upload performed by this script.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())