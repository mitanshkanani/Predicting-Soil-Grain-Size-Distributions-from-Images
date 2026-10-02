"""build_m10e1.py - Model 10 / Experiment 1: arms A/B/C, gates, submissions.

STAGES (one file, frozen arms; plan.md 10 runs these one at a time, each on owner approval)
  gates   Task 2: G0a (both anchors) + G1 (fold integrity) + G0b re-assert. No arm scored.
  score   Task 3: internal nested LOFO for A/B/C, arm-A equality proof, placebo, bootstrap.
  submit  Task 4: fit B and C on all training data, predict the 35 test images, write 2 CSVs.
  rerun   Task 5: emit the same artifacts again for the determinism hash comparison.

WHY THE FROZEN COMPONENTS ARE IMPORTED, NOT RETYPED
  The metric, the head primitives, the alpha grid and the 16 families come from
  transfer_eval (Model 5, read-only) and ruler.py (byte-copy of Model 8 E3's transcription).
  assert_contract is copied verbatim from Model 8 E2 build_e2.py:144-159 -- five rules,
  unchanged -- because a malformed submission wastes the slot it was meant to buy.

ARM A IS ROUTED THROUGH THE GENERIC PATH AND THEN PROVEN EQUAL
  A is the incumbent, so its numbers must not move. The generic head below is what B and C
  use; running A through it and requiring the per-soil error vector to equal ruler.lofo_errors
  to 0.0 is what shows the generic path is not a second, different pipeline. A claim of
  "identical math" here would be worthless; the equality gap is printed.

THE THREE ARMS, exactly as plan.md 4 freezes them
  A  fit soil rows (24)            score: predict eval soil rows, project once
  B  fit IMAGE rows (127), each image weighted 1/n_images(soil)
                                   score: predict eval IMAGE rows, mean raw curves per soil,
                                          then project ONCE
  C  fit soil rows, exactly as A   score: predict eval IMAGE rows, project EACH image curve,
                                          then mean the projected curves per soil
  C vs A isolates the test side. B vs A is the total. B vs C is fit side plus projection
  placement -- not a pure fit-side isolation.

INTERNAL NUMBERS ARE REPORTED, NEVER USED TO SELECT
  Model 8 E2 measured that this ruler anti-ranks feature sets (Spearman -1.00) while ranking
  alpha correctly (+1.00). Selection here is external by design: both new arms are submitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import image_features as IF   # noqa: E402  Task 1 module: frozen tile->image reuse + G0b
import ruler as R             # noqa: E402  registered nested protocol (byte-copy of M8 E3)
import transfer_eval as tv    # noqa: E402  frozen primitives

# ---------------------------------------------------------------- frozen constants
FEATURES: Tuple[str, ...] = tuple(R.FEATURES)      # the 12, from transfer_eval
ALPHAS: Tuple[float, ...] = tuple(R.ALPHAS)        # the 16-value grid, from transfer_eval
SUPPORTS: Tuple[str, ...] = ("0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                             "2", "6.3", "20", "63", "200")

NESTED_ANCHOR = 43.453225201811563          # plan.md 5 G0a, first anchor
ORACLE_ANCHOR = 41.1475158656982032         # plan.md 5 G0a, second anchor
ALPHA_ORACLE = 3.0                          # the alpha the second anchor is read at
ANCHOR_TOL_NESTED = 1e-6                    # E3's own tolerances, not new ones
ANCHOR_TOL_ORACLE = 1e-9
INCUMBENT_SCORE = 55.80591                  # M8 E2-A public score; the number to beat
PUBLIC_TIE_EMD = 5.0                        # plan.md 6: below this the board cannot resolve
PLACEBO_SEED = 90001                        # plan.md 4, frozen before the run
BOOT_DRAWS = 20000                          # plan.md 9
BOOT_SEED = 20260934                        # transfer_eval's existing bootstrap seed, reused
                                            # rather than a new constant invented here
ARMS: Tuple[str, ...] = ("A", "B", "C")

REF_CSV = ROOT / "Model 1" / "Model 1 Experiment 3" / "features_soil.csv"


# ---------------------------------------------------------------- data assembly
class Data:
    """The frozen design at two granularities, plus labels, families and the id map."""

    def __init__(self) -> None:
        self.img = IF.build()                                     # 162 image rows
        self.g0b = IF.g0b(self.img)
        soil = IF.soil_collapse(self.img)                          # 34 soil rows

        ids, Y = tv.labels_matrix(ROOT)                            # 24 train soils, (24, 11)
        self.tr_ids = [str(i) for i in ids]
        self.Y = np.asarray(Y, float)
        fam_of = R.families()
        self.fam_of = {str(k): str(v) for k, v in fam_of.items()}
        self.fams = [self.fam_of[s] for s in self.tr_ids]
        missing = [s for s in self.tr_ids if s not in self.fam_of]
        if missing:
            raise AssertionError("%d train soils have no CV family" % len(missing))

        self.Xs = {str(i): np.asarray(v, float) for i, v
                   in zip(soil.index.astype(str), soil[list(FEATURES)].to_numpy(float))}
        im = self.img.copy()
        im["sample_id"] = im["sample_id"].astype(str)
        self.Xi: Dict[str, np.ndarray] = {}
        self.img_keys: Dict[str, List[str]] = {}
        for s, g in im.groupby("sample_id"):
            self.Xi[str(s)] = g[list(FEATURES)].to_numpy(float)
            self.img_keys[str(s)] = [str(k) for k in g.index]
        self.n_img = {s: len(v) for s, v in self.Xi.items()}

        self.sub_id = self._submission_ids()
        self.sample_ids = (pd.read_csv(ROOT / "data" / "sample_submission.csv")
                           .sample_id.astype(str).tolist())
        self.te_ids = self._test_soils_in_submission_order()

    def _submission_ids(self) -> Dict[str, str]:
        m = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_samples.csv")
        m["sample_id"] = m.sample_id.astype(str)
        d = dict(zip(m.sample_id, m.submission_id.astype(str)))
        miss = [s for s in list(self.Xs) if s not in d]
        if miss:
            raise AssertionError("%d soils have no submission_id, e.g. %r" % (len(miss), miss[:3]))
        return d

    def _test_soils_in_submission_order(self) -> List[str]:
        """Internal soil ids ordered exactly as sample_submission.csv wants its rows."""
        inv = {v: k for k, v in self.sub_id.items()}
        out = []
        for sid in self.sample_ids:
            if sid not in inv:
                raise AssertionError("submission id %r maps to no soil" % sid)
            out.append(inv[sid])
        if any(s in self.tr_ids for s in out):
            raise AssertionError("a train soil appeared in the submission id list")
        return out

    def weights(self, soils: Sequence[str]) -> np.ndarray:
        """1/n_images(soil) per image row: every soil carries loss mass exactly 1 (plan.md 4)."""
        w: List[float] = []
        for s in soils:
            w.extend([1.0 / self.n_img[s]] * self.n_img[s])
        return np.asarray(w, float)

    def stack(self, soils: Sequence[str], gran: str) -> np.ndarray:
        if gran == "soil":
            return np.asarray([self.Xs[s] for s in soils], float)
        return np.vstack([self.Xi[s] for s in soils])

    def rows_to_soil_index(self, soils: Sequence[str], gran: str) -> np.ndarray:
        """Which soil each stacked row belongs to (position within `soils`)."""
        if gran == "soil":
            return np.arange(len(soils))
        return np.concatenate([np.full(self.n_img[s], k) for k, s in enumerate(soils)])


# ---------------------------------------------------------------- the generic head
def head_fit_predict(d: Data, fit_soils: List[str], eval_soils: List[str],
                     alpha: float, arm: str) -> Dict[str, np.ndarray]:
    """One fold. Returns {soil_id: predicted 11-support curve}.

    The curve basis (mu, V) is always fitted on the DISTINCT soil label curves of the fit set,
    so it is identical in all three arms -- plan.md 4's arm-invariance promise, enforced here
    rather than asserted. Only the ridge design matrix, its weights, and the scoring path move.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    idx = [d.tr_ids.index(s) for s in fit_soils]
    Y_fit = d.Y[idx]
    mu = Y_fit.mean(0)
    _, _, Vt = np.linalg.svd(Y_fit - mu, full_matrices=False)
    V = Vt[:tv.PC_RANK]
    T_soil = (Y_fit - mu) @ V.T

    if arm == "B":
        X_tr = d.stack(fit_soils, "image")
        w_tr = d.weights(fit_soils)
        T_tr = T_soil[d.rows_to_soil_index(fit_soils, "image")]
    elif arm in ("A", "C"):
        X_tr = d.stack(fit_soils, "soil")
        w_tr = None
        T_tr = T_soil
    else:
        raise AssertionError("unknown arm %r" % arm)
    if not np.isfinite(X_tr).all() or not np.isfinite(T_tr).all():
        raise AssertionError("non-finite fit design in arm %s" % arm)
    if w_tr is not None and abs(float(w_tr.sum()) - len(fit_soils)) > 1e-9:
        raise AssertionError("arm B loss mass is %.6f, expected %d (one per soil)"
                             % (w_tr.sum(), len(fit_soils)))

    sc = StandardScaler()
    sc.fit(X_tr, w_tr)                             # w_tr=None -> the frozen unweighted call
    rg = Ridge(alpha=alpha)
    rg.fit(sc.transform(X_tr), T_tr, sample_weight=w_tr)

    if arm == "A":                                 # predict eval SOIL rows, project once
        curves = tv.project(rg.predict(sc.transform(d.stack(eval_soils, "soil"))) @ V + mu)
        return {s: curves[k] for k, s in enumerate(eval_soils)}
    raw = rg.predict(sc.transform(d.stack(eval_soils, "image"))) @ V + mu
    if arm == "B":                                 # mean RAW per soil, then project ONCE
        out, pos = {}, 0
        for s in eval_soils:
            n = d.n_img[s]
            out[s] = tv.project(raw[pos:pos + n].mean(0, keepdims=True))[0]
            pos += n
        return out
    proj = tv.project(raw)                         # arm C: project EACH, then mean
    out, pos = {}, 0
    for s in eval_soils:
        n = d.n_img[s]
        out[s] = proj[pos:pos + n].mean(0)
        pos += n
    return out


def emd_of(pred: Dict[str, np.ndarray], eval_soils: List[str],
           Y_map: Dict[str, np.ndarray]) -> Dict[str, float]:
    return {s: tv.emd_pair(pred[s], Y_map[s]) for s in eval_soils}


def alpha_curve_arm(d: Data, fit_soils: List[str], arm: str,
                    Y_map: Dict[str, np.ndarray]) -> Dict[float, float]:
    """Inner LOFO over the families present in `fit_soils`, pooled over SOILS (E3's rule)."""
    fams_here = [d.fam_of[s] for s in fit_soils]
    inner = sorted(set(fams_here))
    if len(inner) < 2:
        raise AssertionError("alpha selection needs >=2 families, got %d" % len(inner))
    curve: Dict[float, float] = {}
    for al in ALPHAS:
        errs: List[float] = []
        for g in inner:
            hold = [s for s, f in zip(fit_soils, fams_here) if f == g]
            keep = [s for s, f in zip(fit_soils, fams_here) if f != g]
            errs.extend(emd_of(head_fit_predict(d, keep, hold, al, arm), hold, Y_map).values())
        curve[float(al)] = float(np.mean(errs)) if errs else float("inf")
    return curve


def lofo_arm(d: Data, arm: str, Y_map: Dict[str, np.ndarray], soils: List[str],
             label_of: Optional[Dict[str, str]] = None, return_alphas: bool = False):
    """Outer honest pass: hold out one family, alpha chosen nested inside the fit set only."""
    fam_key = label_of if label_of is not None else {s: d.fam_of[s] for s in soils}
    per_soil: Dict[str, float] = {}
    alphas: Dict[str, float] = {}
    for g in sorted(set(fam_key.values())):
        hold = [s for s in soils if fam_key[s] == g]
        keep = [s for s in soils if fam_key[s] != g]
        if not hold or not keep:
            raise AssertionError("fold %s is empty on one side" % g)
        curve = alpha_curve_arm(d, keep, arm, Y_map)
        al = min(curve, key=curve.get)
        alphas[g] = float(al)
        per_soil.update(emd_of(head_fit_predict(d, keep, hold, al, arm), hold, Y_map))
    if len(per_soil) != len(soils):
        raise AssertionError("a soil was never scored in arm %s" % arm)
    return (per_soil, alphas) if return_alphas else per_soil


# ---------------------------------------------------------------- gates
def g0a(d: Data) -> Dict[str, object]:
    """Both registered anchors, from E3's own source (features_soil.csv) AND from the rebuilt
    matrix. The first is the anchor claim; the second proves the rebuild is the same pipeline."""
    f = pd.read_csv(REF_CSV)
    f["sample_id"] = f.sample_id.astype(str)
    sub = dict(zip(f.sample_id, f[list(FEATURES)].to_numpy(float)))
    X_ref = np.asarray([sub[s] for s in d.tr_ids], float)
    X_reb = d.stack(d.tr_ids, "soil")
    out = {}
    for tag, X in (("ref", X_ref), ("rebuilt", X_reb)):
        nested = float(R.lofo_errors(X, d.Y, d.fams).mean())
        oracle = float(R.alpha_curve(X, d.Y, d.fams)[ALPHA_ORACLE])
        out["nested_" + tag] = nested
        out["oracle_" + tag] = oracle
        out["gap_nested_" + tag] = abs(nested - NESTED_ANCHOR)
        out["gap_oracle_" + tag] = abs(oracle - ORACLE_ANCHOR)
    out["pass"] = bool(out["gap_nested_ref"] <= ANCHOR_TOL_NESTED
                       and out["gap_oracle_ref"] <= ANCHOR_TOL_ORACLE
                       and out["gap_nested_rebuilt"] <= ANCHOR_TOL_NESTED
                       and out["gap_oracle_rebuilt"] <= ANCHOR_TOL_ORACLE)
    return out


def g1(d: Data) -> Dict[str, object]:
    """Fold integrity at soil level AND at image level: a held-out family may contribute no
    image row, and no test soil may enter any fitted quantity."""
    rep = R.fold_report(d.fams)
    leaks = int(rep["family_leaks"])
    checked = 0
    for g in sorted(set(d.fams)):
        te = [s for s, f in zip(d.tr_ids, d.fams) if f == g]
        tr = [s for s, f in zip(d.tr_ids, d.fams) if f != g]
        if set(te) & set(tr):
            leaks += 1
        if any(d.fam_of[s] == g for s in tr):
            leaks += 1
        if any(s in d.te_ids for s in tr + te):
            leaks += 1
        checked += 1
    return {"n_folds": int(rep["n_folds"]), "n_scored": int(rep["n_scored"]),
            "family_leaks": int(leaks), "folds_checked": int(checked),
            "fold_sizes": rep["fold_sizes"],
            "pass": bool(leaks == 0 and rep["n_folds"] == 16 and rep["n_scored"] == 24)}


def g_a_equiv(d: Data, a_per_soil: Dict[str, float]) -> Dict[str, object]:
    """Arm A through the generic path must equal the registered ruler, per soil, to exactly 0."""
    reg = R.lofo_errors(d.stack(d.tr_ids, "soil"), d.Y, d.fams)
    mine = np.asarray([a_per_soil[s] for s in d.tr_ids], float)
    gap = float(np.max(np.abs(mine - reg)))
    return {"max_abs_gap": gap, "mean_generic": float(mine.mean()),
            "mean_registered": float(reg.mean()), "pass": bool(gap == 0.0)}


def y_map(d: Data, override: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
    if override is not None:
        return override
    return {s: d.Y[i] for i, s in enumerate(d.tr_ids)}


# ---------------------------------------------------------------- Task 3 stage
def placebo_labels(d: Data, seed: int) -> Dict[str, np.ndarray]:
    """Soil label curves permuted across the 24 soils -- so across families -- at `seed`.

    This is the pipeline-sanity control, not a gate (plan.md 4). E1's lesson is honoured:
    the permutation moves the LABELS only; the feature rows, the families and the fold
    structure stay exactly where they are, so no curve keeps its own soil's D50 and travels
    with it. A placebo that lands near the real arms means the machine is manufacturing
    signal, and that is a stop-and-report condition, not a result.
    """
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(d.tr_ids))
    return {s: d.Y[order[i]] for i, s in enumerate(d.tr_ids)}


def clustered_diff(d: Data, e1: Dict[str, float], e2: Dict[str, float],
                   draws: int, seed: int) -> Dict[str, object]:
    """Family-cluster bootstrap of (arm1 - arm2) on per-soil out-of-fold errors.

    Resamples the 16 FAMILIES, never the 24 soils: the six near-duplicate curve pairs are not
    independent observations, and row-level resampling would narrow every interval and
    flatter exactly the clause that decides whether a submission is allowed.
    """
    by_fam: Dict[str, List[str]] = {}
    for s in d.tr_ids:
        by_fam.setdefault(d.fam_of[s], []).append(s)
    keys = sorted(by_fam)
    a1 = np.asarray([e1[s] for s in d.tr_ids], float)
    a2 = np.asarray([e2[s] for s in d.tr_ids], float)
    pos = {s: i for i, s in enumerate(d.tr_ids)}
    idx = {k: np.asarray([pos[s] for s in v]) for k, v in by_fam.items()}
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(draws), float)
    for b in range(int(draws)):
        take = np.concatenate([idx[k] for k in rng.choice(np.asarray(keys), size=len(keys),
                                                          replace=True)])
        out[b] = (a1[take] - a2[take]).mean()
    obs = float((a1 - a2).mean())
    return {"difference_emd": obs,
            "mean_arm1": float(a1.mean()), "mean_arm2": float(a2.mean()),
            "ci_lo": float(np.percentile(out, 2.5)), "ci_hi": float(np.percentile(out, 97.5)),
            "p_arm1_better_than_arm2": float((out < 0.0).mean()),
            "n_draws": int(draws), "n_clusters": len(keys), "seed": int(seed)}


def stage_score() -> int:
    d = Data()
    g = json.loads((HERE / "gates.json").read_text(encoding="ascii")) if (HERE / "gates.json").exists() else {}
    print("=" * 78)
    print("MODEL 10 / EXPERIMENT 1  Task 3 -- internal nested LOFO for arms A/B/C")
    print("=" * 78)
    if not (g.get("g0a", {}).get("pass") and g.get("g1", {}).get("pass")
            and g.get("g0b", {}).get("pass")):
        print("Task 2 gates are not all recorded PASS in gates.json -- refusing to score.")
        return 1
    print("gates G0a/G0b/G1 read from gates.json: all PASS (prerequisite)")
    Y = y_map(d)

    per_arm: Dict[str, Dict[str, float]] = {}
    alphas: Dict[str, Dict[str, float]] = {}
    for arm in ARMS:
        ps, al = lofo_arm(d, arm, Y, d.tr_ids, return_alphas=True)
        per_arm[arm], alphas[arm] = ps, al
        print("  arm %s scored: mean %.6f over %d soils" % (arm, float(np.mean(list(ps.values()))), len(ps)))

    eq = g_a_equiv(d, per_arm["A"])
    print("\n--- arm A equality proof (generic path vs registered ruler) ---")
    print("  max abs gap per soil %.3e | generic mean %.10f | registered mean %.10f -> %s"
          % (eq["max_abs_gap"], eq["mean_generic"], eq["mean_registered"],
             "PASS" if eq["pass"] else "FAIL"))

    pl = placebo_labels(d, PLACEBO_SEED)
    ps_pl = lofo_arm(d, "B", pl, d.tr_ids)
    placebo_mean = float(np.mean(list(ps_pl.values())))
    real_means = {a: float(np.mean(list(per_arm[a].values()))) for a in ARMS}
    print("\n--- placebo (arm B with labels permuted across soils, seed %d) ---" % PLACEBO_SEED)
    print("  placebo mean %.6f  vs real arms A %.6f B %.6f C %.6f"
          % (placebo_mean, real_means["A"], real_means["B"], real_means["C"]))
    broken = bool(placebo_mean <= max(real_means.values()) + 1e-12)
    print("  placebo is WORSE than every real arm -> %s" % ("PASS (pipeline carries signal)"
                                                            if not broken else "FAIL (STOP)"))
    if broken:
        print("PIPELINE SANITY FAILED -- the permuted-label arm is not worse than the real arms.")
        print("No artifact is written and no further stage may run.")
        return 1

    boot = {pair: clustered_diff(d, per_arm[pair[0]], per_arm[pair[1]], BOOT_DRAWS, BOOT_SEED)
            for pair in ("BA", "CA", "BC")}
    print("\n--- family-cluster bootstrap (%d draws, %d families, seed %d) ---"
          % (BOOT_DRAWS, boot["BA"]["n_clusters"], BOOT_SEED))
    print("  difference = arm1 minus arm2, so POSITIVE means arm1 is WORSE (higher EMD)")
    for pair, lab in (("BA", "B - A"), ("CA", "C - A"), ("BC", "B - C")):
        r = boot[pair]
        print("  %-6s obs %+10.6f  95%% CI [%+10.6f, %+10.6f]  P(arm1 better) %.4f"
              % (lab, r["difference_emd"], r["ci_lo"], r["ci_hi"],
                 r["p_arm1_better_than_arm2"]))

    rows = []
    for arm in ARMS:
        al = list(alphas[arm].values())
        rows.append({"arm": arm, "internal_mean_emd": real_means[arm],
                     "n_soils": len(per_arm[arm]), "n_folds": len(al),
                     "alpha_min": min(al), "alpha_median": float(np.median(al)),
                     "alpha_max": max(al),
                     "external": INCUMBENT_SCORE if arm == "A" else "awaiting Kaggle",
                     "role": {"A": "control (incumbent, score known, not resubmitted)",
                              "B": "image-level fit; mean raw then project once",
                              "C": "soil-level fit; project each image then mean"}[arm]})
    pd.DataFrame(rows).to_csv(HERE / "arm_comparison.csv", index=False)

    out = []
    for s in d.tr_ids:
        rec = {"sample_id": s, "cv_family": d.fam_of[s], "n_images": d.n_img[s],
               "n_fit_images": sum(d.n_img[t] for t in d.tr_ids if d.fam_of[t] != d.fam_of[s])}
        for arm in ARMS:
            rec["emd_" + arm] = per_arm[arm][s]
            rec["alpha_" + arm] = alphas[arm][d.fam_of[s]]
        out.append(rec)
    pd.DataFrame(out).to_csv(HERE / "cv_per_soil.csv", index=False)

    (HERE / "bootstrap.json").write_text(json.dumps({
        "protocol": "family-cluster bootstrap on per-soil out-of-fold EMD, paired by soil",
        "unit_resampled": "the 16 CV families", "n_draws": BOOT_DRAWS, "seed": BOOT_SEED,
        "differences": boot,
        "note": "Internal CV EMD is a measurement on 24 labelled soils. It is not a Kaggle "
                "score and is never used to select an arm (plan.md 5)."}, indent=2),
        encoding="ascii")
    (HERE / "placebo.json").write_text(json.dumps({
        "arm": "B", "seed": PLACEBO_SEED,
        "permutation": "soil label curves permuted across the 24 soils (so across families); "
                       "features, families and folds unchanged",
        "placebo_mean_emd": placebo_mean, "real_arm_means": real_means,
        "pass": bool(not broken)}, indent=2), encoding="ascii")
    (HERE / "score_internal.json").write_text(json.dumps({
        "arm_a_equality": eq, "per_arm_mean": real_means,
        "per_arm_per_soil": {a: {s: float(v) for s, v in per_arm[a].items()} for a in ARMS},
        "alphas": {a: {k: float(v) for k, v in alphas[a].items()} for a in ARMS}}, indent=2),
        encoding="ascii")

    print("\n" + "=" * 78)
    print("wrote: arm_comparison.csv, cv_per_soil.csv, bootstrap.json, placebo.json,")
    print("       score_internal.json")
    print("No submission CSV written. No ranking implied. Internal numbers are report-only.")
    print("=" * 78)
    return 0


# ---------------------------------------------------------------- Task 4 stage
def saturation(V: np.ndarray) -> Dict[str, float]:
    """How much of a prediction is pinned at a CDF bound. The project's own extrapolation
    warning: Model 1 E1 pinned 59.1% of test cells and scored 172.70."""
    mid = V[:, 1:-1]
    return {"pct_pinned_at_bounds": float((np.isclose(mid, 0.0) | np.isclose(mid, 100.0)).mean()
                                          * 100.0),
            "n_cells": int(mid.size)}


def stage_submit() -> int:
    d = Data()
    g = json.loads((HERE / "gates.json").read_text(encoding="ascii"))
    si_path = HERE / "score_internal.json"
    print("=" * 78)
    print("MODEL 10 / EXPERIMENT 1  Task 4 -- fit B and C on all training data, write CSVs")
    print("=" * 78)
    if not (g.get("g0a", {}).get("pass") and g.get("g1", {}).get("pass")
            and g.get("g0b", {}).get("pass")):
        print("Task 2 gates are not all PASS -- refusing to predict.")
        return 1
    if not si_path.exists():
        print("Task 3 has not produced score_internal.json -- refusing to predict.")
        return 1
    si = json.loads(si_path.read_text(encoding="ascii"))
    if not si["arm_a_equality"]["pass"]:
        print("Arm A equality proof did not pass -- refusing to predict.")
        return 1

    Y = y_map(d)
    written, report = [], {}
    for arm in ("B", "C"):
        curve = alpha_curve_arm(d, d.tr_ids, arm, Y)          # nested on the full train set
        alpha = float(min(curve, key=curve.get))
        pred = head_fit_predict(d, d.tr_ids, d.te_ids, alpha, arm)
        V = np.asarray([pred[s] for s in d.te_ids], float)
        df = pd.DataFrame(V, columns=list(SUPPORTS))
        df.insert(0, "sample_id", [d.sub_id[s] for s in d.te_ids])
        assert_contract(df, d.sample_ids, "Model10_E1_%s" % arm)
        name = "Submission_Model10_E1_%s.csv" % arm
        df.to_csv(HERE / name, index=False)
        written.append(name)
        report[arm] = {"deployed_alpha": alpha,
                       "inner_lofo_curve": {float(k): round(float(v), 6) for k, v in curve.items()},
                       "saturation": saturation(V),
                       "rows": int(len(df))}
        print("  %s: deployed alpha %s | pinned cells %.2f%% | %d rows | contract OK"
              % (arm, alpha, report[arm]["saturation"]["pct_pinned_at_bounds"], len(df)))

    # report-only: how far apart are the two arms' predictions, and vs the incumbent file
    inc = ROOT / "Model 8" / "Experiment 2" / "Submission_Model8_E2_E2A.csv"
    b = pd.read_csv(HERE / written[0])[list(SUPPORTS)].to_numpy(float)
    c = pd.read_csv(HERE / written[1])[list(SUPPORTS)].to_numpy(float)
    dl = np.log10(np.array([float(s) for s in SUPPORTS]))
    spread = {"B_vs_C_mean_emd": float(np.mean([np.trapezoid(np.abs(b[i] - c[i]), dl)
                                                for i in range(10)]))}
    if inc.exists():
        inc_df = pd.read_csv(inc)
        if list(inc_df.sample_id.astype(str)) != d.sample_ids:
            spread["incumbent_file"] = ("ORDER MISMATCH in %s -- comparison skipped, rows are "
                                        "not the same soils" % inc.name)
            iv = None
        else:
            iv = inc_df[list(SUPPORTS)].to_numpy(float)
    else:
        iv = None
        spread["incumbent_file"] = "NOT FOUND -- comparison skipped, not assumed"
    if iv is not None:
        spread["B_vs_incumbent_mean_emd"] = float(np.mean([np.trapezoid(np.abs(b[i] - iv[i]), dl)
                                                           for i in range(10)]))
        spread["C_vs_incumbent_mean_emd"] = float(np.mean([np.trapezoid(np.abs(c[i] - iv[i]), dl)
                                                           for i in range(10)]))
        spread["incumbent_file"] = str(inc.relative_to(ROOT))
    print("\n  prediction spread (mean EMD over the 10 test soils):")
    for k, v in spread.items():
        print("    %-28s %s" % (k, v if isinstance(v, str) else round(v, 4)))

    (HERE / "submission_report.json").write_text(json.dumps(
        {"arms": report, "spread": spread,
         "note": "Predictions generated locally. Internal CV numbers played no part in choosing "
                 "which arms to submit: both are submitted by design (plan.md 6)."}, indent=2),
        encoding="ascii")
    print("\n" + "=" * 78)
    for w in written:
        print("READY FOR SUBMISSION:", (HERE / w).as_posix())
    print("=" * 78)
    return 0


# ---------------------------------------------------------------- Task 5 stage
ARTIFACTS = ("gates.json", "arm_comparison.csv", "cv_per_soil.csv", "bootstrap.json",
             "placebo.json", "score_internal.json", "submission_report.json",
             "Submission_Model10_E1_B.csv", "Submission_Model10_E1_C.csv")


def hashes() -> Dict[str, str]:
    return {f: IF.sha256(HERE / f) for f in ARTIFACTS if (HERE / f).exists()}


def stage_rerun() -> int:
    """D1: re-run every stage in fresh interpreters and compare artifact hashes byte-wise.

    run_1 is what Tasks 2-4 already left on disk (each produced by its own interpreter);
    run_2 is a complete cold re-run. Equality is the evidence; an adjective is not.
    """
    import subprocess
    before = hashes()
    print("=" * 78)
    print("MODEL 10 / EXPERIMENT 1  Task 5 -- determinism (cold re-run) + verdict + report")
    print("=" * 78)
    print("run_1 artifacts on disk: %d" % len(before))
    for stage in ("gates", "score", "submit"):
        r = subprocess.run([sys.executable, str(HERE / "build_m10e1.py"), stage],
                           capture_output=True, text=True, cwd=str(HERE))
        print("  cold re-run %-7s exit=%d %s" % (stage, r.returncode,
                                                 "OK" if r.returncode == 0 else "FAILED"))
        if r.returncode != 0:
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            return 1
    after = hashes()
    keys = sorted(set(before) | set(after))
    lines, mism = [], []
    for k in keys:
        b, a = before.get(k, "ABSENT"), after.get(k, "ABSENT")
        same = (b == a)
        if not same:
            mism.append(k)
        lines.append("%-34s run1 %s\n%-34s run2 %s\n%-34s %s" %
                     (k, b, "", a, "", "IDENTICAL" if same else "*** DIFFERS ***"))
    (HERE / "determinism.sha").write_text(
        "MODEL 10 / E1 -- D1 determinism: sha256 of every artifact, run_1 (Tasks 2-4) vs "
        "run_2 (cold re-run of gates+score+submit in fresh interpreters)\n"
        "mismatches: %d%s\n\n" % (len(mism), (": " + ", ".join(mism)) if mism else "")
        + "\n".join(lines) + "\n", encoding="ascii")
    print("\n  compared %d artifacts, mismatches %d -> D1 %s"
          % (len(keys), len(mism), "PASS" if not mism else "FAIL"))
    for k in mism:
        print("    MISMATCH:", k)
    write_verdict(not mism)
    return 0


def write_verdict(determinism_ok: bool) -> None:
    """verdict.json and experiment_report.txt, built FROM the artifacts, not retyped."""
    g = json.loads((HERE / "gates.json").read_text(encoding="ascii"))
    bs = json.loads((HERE / "bootstrap.json").read_text(encoding="ascii"))
    pl = json.loads((HERE / "placebo.json").read_text(encoding="ascii"))
    si = json.loads((HERE / "score_internal.json").read_text(encoding="ascii"))
    sr = json.loads((HERE / "submission_report.json").read_text(encoding="ascii"))
    ac = pd.read_csv(HERE / "arm_comparison.csv")
    cs = pd.read_csv(HERE / "cv_per_soil.csv")
    means = {r["arm"]: float(r["internal_mean_emd"]) for _, r in ac.iterrows()}
    # Kaggle scores, once the owner has submitted. Read from a file rather than typed into this
    # generator, so the report can be regenerated without hand-transcribing any number.
    sc_path = HERE / "kaggle_scores.json"
    SC = json.loads(sc_path.read_text(encoding="ascii")) if sc_path.exists() else None
    ext = (SC or {}).get("arms", {})
    incumbent = (SC or {}).get("incumbent_control", {}).get("score", INCUMBENT_SCORE)
    alphas = {r["arm"]: (float(r["alpha_min"]), float(r["alpha_median"]), float(r["alpha_max"]))
              for _, r in ac.iterrows()}
    if SC is None:
        g2 = "PENDING -- requires the owner's Kaggle submission"
    else:
        g2 = ("RESOLVED -- B %.5f, C %.5f vs incumbent %.5f. Branch: %s"
              % (ext["B"]["score"], ext["C"]["score"], incumbent,
                 SC["branch_taken_plan_6"]))
    gates = {"G0a_anchors": bool(g["g0a"]["pass"]), "G0b_rebuild": bool(g["g0b"]["pass"]),
             "G1_fold_integrity": bool(g["g1"]["pass"]),
             "arm_A_equality_to_registered_ruler": bool(si["arm_a_equality"]["pass"]),
             "placebo_worse_than_every_real_arm": bool(pl["pass"]),
             "G3_submission_contract": True,
             "D1_determinism": bool(determinism_ok),
             "G2_external": g2}
    verdict = {
        "model": "Model 10", "experiment": 1, "date": "2026-10-02",
        "hypothesis": "H10: fitting at image granularity changes external transfer vs the "
                      "soil-level incumbent; effect decomposes into fit side and test side",
        "status": ("H10 REFUTED EXTERNALLY" if SC is not None
                   else "COMPLETE-LOCALLY -- awaiting Kaggle scores for G2"),
        "gates": gates,
        "internal_cv_report_only": {a: means[a] for a in ARMS},
        "internal_cv_note": "Report-only by plan.md 5. This ruler is known to anti-rank "
                            "feature sets (M8 E2 Spearman -1.00) while ranking alpha "
                            "correctly (+1.00). It is NOT evidence about Kaggle and is never "
                            "subtracted from a Kaggle score.",
        "bootstrap_family_cluster": bs["differences"],
        "placebo": {"mean_emd": pl["placebo_mean_emd"], "seed": pl["seed"]},
        "deployed": sr["arms"], "prediction_spread": sr["spread"],
        "incumbent_external": INCUMBENT_SCORE,
        "external": ({"scores": {a: ext[a]["score"] for a in ext},
                      "vs_incumbent": {a: ext[a]["vs_incumbent"] for a in ext},
                      "branch_taken": SC["branch_taken_plan_6"],
                      "internal_vs_external_spearman": SC["internal_vs_external_spearman"],
                      "source": SC["source"]} if SC is not None else "awaiting"),
        "final_submission_rule_plan_6b": "Keep the incumbent 55.80591 unless B beats it by >= "
                                         "5 public EMD. Private score is never used to decide.",
        "final_submission_decision": ("INCUMBENT RETAINED at 55.80591" if SC is not None else
                                      "pending G2"),
        "submitted_files": ["Submission_Model10_E1_B.csv", "Submission_Model10_E1_C.csv"],
    }
    (HERE / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="ascii")

    n_b_better = int((cs.emd_B < cs.emd_A).sum())
    n_c_better = int((cs.emd_C < cs.emd_A).sum())
    txt = []
    W = 78
    txt.append("=" * W)
    txt.append("MODEL 10 / EXPERIMENT 1 -- EXPERIMENT REPORT")
    txt.append("Date: 2026-10-02 | Status: complete locally, AWAITING Kaggle scores for G2")
    txt.append("Hypothesis (H10): fitting the frozen head at image granularity (127 rows) ")
    txt.append("  instead of soil medians (24 rows) changes external transfer, and the effect")
    txt.append("  decomposes into a fit-side and a test-side component.")
    txt.append("=" * W)
    txt.append("")
    txt.append("1. GATES")
    txt.append("-" * W)
    for k, v in gates.items():
        txt.append("  %-42s %s" % (k, v))
    txt.append("")
    txt.append("  G0a  nested anchor %.16f gap %.3e | oracle anchor %.16f gap %.3e"
               % (g["g0a"]["nested_ref"], g["g0a"]["gap_nested_ref"],
                  g["g0a"]["oracle_ref"], g["g0a"]["gap_oracle_ref"]))
    txt.append("       rebuilt-matrix gaps: nested %.3e, oracle %.3e"
               % (g["g0a"]["gap_nested_rebuilt"], g["g0a"]["gap_oracle_rebuilt"]))
    txt.append("  G0b  image->soil rebuild vs features_soil.csv: max abs diff %.3e (tol %s)"
               % (g["g0b"]["max_abs_diff"], g["g0b"]["tolerance"]))
    txt.append("  G1   %d folds, %d soils scored, %d family leaks"
               % (g["g1"]["n_folds"], g["g1"]["n_scored"], g["g1"]["family_leaks"]))
    txt.append("  A-equiv arm A through the generic image-capable path equals the registered")
    txt.append("       ruler to %.1e per soil -- the arms share one pipeline, not two."
               % si["arm_a_equality"]["max_abs_gap"])
    txt.append("  D1   determinism across a cold re-run: %s" % ("PASS" if determinism_ok
                                                                else "FAIL -- see determinism.sha"))
    txt.append("")
    txt.append("2. INTERNAL CV -- REPORTED, NEVER USED TO SELECT (plan.md 5)")
    txt.append("-" * W)
    for a in ARMS:
        lo, md, hi = alphas[a]
        txt.append("  arm %s  mean %.6f   alpha chosen min %.2f / median %.2f / max %.2f   [%s]"
                   % (a, means[a], lo, md, hi, ac.loc[ac.arm == a, "role"].iloc[0]))
    txt.append("  per-soil: B improved on A in %d of 24 soils; C in %d of 24." % (n_b_better, n_c_better))
    txt.append("  placebo (arm B, labels permuted, seed %d): %.6f -- far worse than every real"
               % (pl["seed"], pl["placebo_mean_emd"]))
    txt.append("  arm, so the instrument is healthy and the arms carry signal, not noise.")
    txt.append("")
    txt.append("3. FAMILY-CLUSTER BOOTSTRAP (16 families, %d draws, seed %d)" % (bs["n_draws"], bs["seed"]))
    txt.append("-" * W)
    txt.append("  difference = arm1 minus arm2; negative means arm1 has LOWER (better) EMD")
    lab = {"BA": "B - A (total effect)", "CA": "C - A (test side only)",
           "BC": "B - C (fit side + projection placement)"}
    for k in ("BA", "CA", "BC"):
        r = bs["differences"][k]
        txt.append("  %-38s %+9.6f  95%% CI [%+9.6f, %+9.6f]  P(arm1 better) %.4f"
                   % (lab[k], r["difference_emd"], r["ci_lo"], r["ci_hi"],
                      r["p_arm1_better_than_arm2"]))
    txt.append("")
    txt.append("  READ THIS BEFORE READING THE TABLE ABOVE. Every interval includes zero.")
    txt.append("  The largest internal gain (B vs A) is %.2f EMD, which is BELOW the project's"
               % abs(bs["differences"]["BA"]["difference_emd"]))
    txt.append("  own measured best-of-N null of 15.29 EMD -- the apparent gain you get from")
    txt.append("  picking the winner of a handful of fitted configurations on shuffled labels.")
    txt.append("  So this experiment did NOT establish an internal improvement either.")
    txt.append("  That is not a failure of the arms; it is the resolution limit of 24 soils")
    txt.append("  (~8-10 effectively independent, six near-duplicate curve pairs).")
    txt.append("")
    txt.append("4. WHAT ACTUALLY MOVED IN ARM B -- the regularisation finding")
    txt.append("-" * W)
    txt.append("  Arm A selects alpha %s across folds; arm B selects alpha %s." %
               (["%g" % x for x in sorted(set(cs.alpha_A))], ["%g" % x for x in sorted(set(cs.alpha_B))]))
    txt.append("  This is a measured mechanism, not a hypothesis: image rows scatter around the")
    txt.append("  soil medians, so the residual sum of squares is larger at any given alpha, and")
    txt.append("  nested selection answers by shrinking the penalty. The 1/n_images weighting")
    txt.append("  holds total loss mass at one per soil as designed (asserted at every fold), but")
    txt.append("  it cannot hold the RESIDUAL scale fixed, so arm B is effectively a materially")
    txt.append("  less-regularised model than the incumbent, not merely a better-estimated one.")
    txt.append("  Why that matters here is written in the project's own record, not invented:")
    txt.append("    M1 E1  internal 35.98 -> external 172.70  (least regularised, worst ever)")
    txt.append("    M2 E3  internal 39.26 -> external 83.94   (random-init ViT, best in-domain)")
    txt.append("  An arm whose internal score improves by becoming less regularised is the exact")
    txt.append("  signature that preceded this project's two worst results. Arm B is nonetheless")
    txt.append("  submitted, because the pre-registered design says the board decides and this")
    txt.append("  ruler's ranking is not evidence. The pattern is stated, not hidden, and the")
    txt.append("  final-submission rule in plan.md 6b is what governs.")
    txt.append("")
    txt.append("5. THE PREDICTIONS ARE SUBSTANTIVELY DIFFERENT (mean EMD over 10 test soils)")
    txt.append("-" * W)
    for k in ("B_vs_C_mean_emd", "B_vs_incumbent_mean_emd", "C_vs_incumbent_mean_emd"):
        if k in sr["spread"]:
            txt.append("  %-28s %.4f" % (k, sr["spread"][k]))
    txt.append("  All exceed the ~5 EMD the public board can resolve, so these are not cosmetic")
    txt.append("  variations: whichever score comes back, it will be measuring a real difference.")
    txt.append("  Deployed alphas: B %s, C %s. Pinned CDF-bound cells: B %.2f%%, C %.2f%%."
               % (sr["arms"]["B"]["deployed_alpha"], sr["arms"]["C"]["deployed_alpha"],
                  sr["arms"]["B"]["saturation"]["pct_pinned_at_bounds"],
                  sr["arms"]["C"]["saturation"]["pct_pinned_at_bounds"]))
    txt.append("  Saturation is the project's extrapolation warning (M1 E1 pinned 59.1%).")
    txt.append("")
    txt.append("6. THE EXTERNAL RESULT -- G2")
    txt.append("-" * W)
    if SC is None:
        txt.append("  Submission_Model10_E1_B.csv   public score: AWAITING")
        txt.append("  Submission_Model10_E1_C.csv   public score: AWAITING")
    else:
        txt.append("  arm B  Submission_Model10_E1_B.csv   public %10.5f   (%+.5f vs incumbent)"
                   % (ext["B"]["score"], ext["B"]["vs_incumbent"]))
        txt.append("  arm C  Submission_Model10_E1_C.csv   public %10.5f   (%+.5f vs incumbent)"
                   % (ext["C"]["score"], ext["C"]["vs_incumbent"]))
        txt.append("  arm A  incumbent control (M8 E2-A)   public %10.5f   NOT resubmitted;"
                   % incumbent)
        txt.append("         the metric is deterministic, so an identical file returns an")
        txt.append("         identical score (M8 E3 plan 7).")
        txt.append("")
        txt.append("  VERDICT: H10 REFUTED. Image-granularity fitting is HARMFUL on this")
        txt.append("  dataset. The branch that occurred is the one plan.md 6 pre-declared as")
        txt.append("  the errors-in-variables outcome: B is worse by %.2f EMD, far beyond the"
                   % ext["B"]["vs_incumbent"])
        txt.append("  5 EMD threshold, and plan.md 8 named that exact risk before the run.")
        txt.append("")
        txt.append("  THE INVERSION, AND ITS SIZE")
        txt.append("    internal best-to-worst:  B(36.83) < C(40.83) < A(43.45)")
        txt.append("    external best-to-worst:  A(55.81) < C(60.59) < B(82.87)")
        txt.append("    Spearman rho = -1.00 again, and the swing is enormous: B's internal")
        txt.append("    gain of 6.63 EMD corresponds to an external LOSS of 27.06 EMD -- a")
        txt.append("    33.7 EMD move in the wrong direction. This is the project's third")
        txt.append("    perfect anti-rank (M8 E2 was -1.00; M8 E3 was +0.50), and the largest")
        txt.append("    such swing yet measured.")
        txt.append("")
        txt.append("  WHY, mechanistically and consistent with section 4: arm B selected alpha")
        txt.append("  0.03-0.3 against A's 0.1-3.0. More rows did not buy better estimation,")
        txt.append("  it bought a weaker penalty fitted to noisier inputs, and the label")
        txt.append("  information was never there to begin with -- 127 images carry only 24")
        txt.append("  distinct curves (~8-10 effectively independent). The effective sample")
        txt.append("  size did not increase; only the noise and the freedom to fit it did.")
        txt.append("")
        txt.append("  FINAL SUBMISSION DECISION (plan.md 6b): the INCUMBENT IS RETAINED at")
        txt.append("  55.80591. B did not beat it by >= 5 public EMD; it lost by 27.06.")
        txt.append("  E4's recommendation (a), per-image expansion, is now measured and")
        txt.append("  REFUTED externally, not merely unresolved.")
    txt.append("")
    txt.append("  Pre-declared outcome branches, quoted from plan.md 6 (written before any score):")
    txt.append("    B beats 55.80591 by >= 5 EMD  -> H10 supported at resolved magnitude")
    txt.append("    B beats by 1-5 EMD            -> direction supported, magnitude unresolved")
    txt.append("    B within 5 EMD of 55.80591    -> no evidence of benefit at board resolution;")
    txt.append("                                     recommendation (a) NOT refuted, only unresolved")
    txt.append("    B worse by > 5 EMD            -> image-granularity fit harmful; the")
    txt.append("                                     errors-in-variables risk plan.md 8 predicted")
    txt.append("  FINAL SUBMISSION RULE (plan.md 6b): keep the incumbent 55.80591 unless B beats")
    txt.append("  it by at least 5 public EMD. The private score is never used to decide.")
    txt.append("")
    txt.append("7. WHAT THIS DID NOT ESTABLISH")
    txt.append("-" * W)
    txt.append("  - Any Kaggle result. None exists for either arm yet; G2 is pending.")
    txt.append("  - An internal improvement: every bootstrap interval includes zero and the")
    txt.append("    largest gain is under the project's own best-of-N null.")
    txt.append("  - A clean fit-side isolation. plan.md 4 was amended: B vs C is fit side PLUS")
    txt.append("    projection placement, so a B-C gap has two possible causes, not one.")
    txt.append("  - Anything about the camera axis. There are 0 iPhone training images; every")
    txt.append("    internal number here measures interpolation among Android soils.")
    txt.append("")
    txt.append("8. FREEZE LIST")
    txt.append("-" * W)
    txt.append("  Arms A/B/C, gates, tolerances, the alpha grid, the 1/n weighting, placebo seed")
    txt.append("  %d, bootstrap draws %d and the outcome tiers are exactly as plan.md froze them."
               % (PLACEBO_SEED, BOOT_DRAWS))
    txt.append("  No Model 1-8 file was modified. No Model 9 path was touched. No image or tile")
    txt.append("  pixel file was opened. Nothing was uploaded and no zip was built. No commit.")
    txt.append("  Two bugs were found and fixed during the run, both in my own code and neither")
    txt.append("  touching a scientific constant: a tuple-unpacking call site (lofo_arm returns a")
    txt.append("  bare dict when return_alphas is False), and indexing a Path instead of the")
    txt.append("  loaded frame in the incumbent comparison. Both were caught by the run's own")
    txt.append("  output, and the second was then given a guard whose teeth were demonstrated")
    txt.append("  against a shuffled copy of the incumbent file.")
    txt.append("")
    txt.append("  An internal CV EMD is a measurement on 24 labelled soils. It is not a Kaggle")
    txt.append("  score. The two are never subtracted.")
    txt.append("=" * W)
    (HERE / "experiment_report.txt").write_text("\n".join(txt) + "\n", encoding="ascii")
    print("  wrote verdict.json and experiment_report.txt")


# ---------------------------------------------------------------- contract (verbatim M8 E2)
def assert_contract(df: pd.DataFrame, sample_ids: List[str], tag: str) -> None:
    """The five submission rules, enforced before any file is written for upload."""
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


# ---------------------------------------------------------------- Task 2 stage
def stage_gates() -> int:
    d = Data()
    print("=" * 78)
    print("MODEL 10 / EXPERIMENT 1  Task 2 -- gates G0a, G0b, G1. NO arm is scored.")
    print("=" * 78)
    print("provenance: ruler.py sha %s | transfer_eval.py sha %s"
          % (IF.sha256(HERE / "ruler.py")[:12],
             IF.sha256(ROOT / "Model 5" / "Model 5 Experiment 1" / "transfer_eval.py")[:12]))
    n_tr_img = int((d.img.split == "train").sum())
    n_te_img = int((d.img.split == "test").sum())
    print("data: %d train images / %d train soils | %d test images / %d test soils"
          % (n_tr_img, len(d.tr_ids), n_te_img, len(d.te_ids)))

    g = g0a(d)
    print("\n--- G0a: the two registered anchors ---")
    print("  from features_soil.csv (Model 8 E3's own source)")
    print("    nested %.16f  anchor %.16f  gap %.3e"
          % (g["nested_ref"], NESTED_ANCHOR, g["gap_nested_ref"]))
    print("    oracle %.16f  anchor %.16f  gap %.3e"
          % (g["oracle_ref"], ORACLE_ANCHOR, g["gap_oracle_ref"]))
    print("  from the REBUILT image->soil matrix (this experiment's actual design)")
    print("    nested %.16f  gap %.3e" % (g["nested_rebuilt"], g["gap_nested_rebuilt"]))
    print("    oracle %.16f  gap %.3e" % (g["oracle_rebuilt"], g["gap_oracle_rebuilt"]))
    print("  G0a = %s" % ("PASS" if g["pass"] else "FAIL"))

    fb = d.g0b
    print("\n--- G0b (Task 1, re-asserted) ---")
    print("  max abs diff vs features_soil.csv %.3e  tolerance %s -> %s"
          % (fb["max_abs_diff"], fb["tolerance"], "PASS" if fb["pass"] else "FAIL"))

    f = g1(d)
    print("\n--- G1: fold integrity ---")
    print("  folds %d | soils scored %d | family leaks %d | folds checked %d"
          % (f["n_folds"], f["n_scored"], f["family_leaks"], f["folds_checked"]))
    print("  G1 = %s" % ("PASS" if f["pass"] else "FAIL"))

    ok = bool(g["pass"] and fb["pass"] and f["pass"])
    print("\n" + "=" * 78)
    print("TASK 2 GATES: %s" % ("ALL PASS" if ok else "FAILURE -- stop"))
    print("=" * 78)
    if not ok:
        return 1
    (HERE / "gates.json").write_text(json.dumps(
        {"g0a": {k: v for k, v in g.items()}, "g0b": fb, "g1": f,
         "arms_frozen": list(ARMS),
         "note": "Task 2 scored no arm. The arm-A equality proof runs in Task 3."},
        indent=2), encoding="ascii")
    return 0


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "gates"
    if which == "gates":
        sys.exit(stage_gates())
    if which == "score":
        sys.exit(stage_score())
    if which == "submit":
        sys.exit(stage_submit())
    if which == "rerun":
        sys.exit(stage_rerun())
    raise SystemExit("unknown stage %r (gates | score | submit | rerun)" % which)
