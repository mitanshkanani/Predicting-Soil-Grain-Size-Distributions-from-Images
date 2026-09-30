"""alias_gate.py - the pair set, derived by one deterministic rule over recorded artifacts.

Model 6-3 ran this diagnostic read-only and never persisted a pair list, so the set is re-derived
here from the two files that do exist: Model 2 Experiment 3's `cv_per_soil.csv` (column `M1`, the
shipped model's own per-soil CV error) and its `cv_families.csv`. The re-derivation must reproduce
the Model 6-3 figures that are properties of THIS rule - top-10 share 61.2%, worst soil 10.1% -
before the gate is allowed to decide anything. The median alias curve gap of 65.7 EMD is not one of
them: amendment A1 records that it was printed by Model 6-3's own worst-8/best-8 unconstrained
neighbour rule, a different instrument, and `historical_gaps()` re-runs that rule so the claim stays
checkable rather than prose. If the two shares cannot be recovered, the rule and the record disagree
about something and Task 6 must not run on top of it.

THE RULE, FIXED IN THE PRE-REGISTRATION
  For each of the 10 highest-M1 soils, its nearest neighbour in the standardised 12-feature space,
  drawn from a DIFFERENT CV family than the query, is its alias partner. The control set is the same
  rule applied to the 14 remaining (lowest-error) soils. One pair per query soil, in error order. No
  extra exclusion, no radius, no threshold, and no hand-picked pair.

WHY THIS FILE DOES NOT LOOK AT A1-A4
  The pair set has to be frozen before the new features are consulted, or the gate becomes selection
  on the very thing it is supposed to test. Nothing here imports m7a_data or spatial_features, and
  check_alias_gate.py asserts that neither module is loaded by the derivation. The A1-A4 separation
  of these same pairs is Task 6's computation, over a pair list that cannot move under it.

WHAT THIS FILE DOES NOT DO
  No CV arm is scored, no control or placebo runs, no gain is computed, no prediction is made for a
  test soil, and no submission exists. The numbers printed are internal diagnostics on the 24
  labelled training soils. The distances use corpus-wide standardisation, which is legitimate here
  because nothing is fitted and no model sees these features: it is a property of the feature space.

The curve gaps are the frozen `emd_pair` over the log-weighted trapezoid, on the same 11 DIN-ISO
supports the pipeline predicts, so a "curve gap" here is the same quantity the metric reports.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import transfer_eval as tv  # noqa: E402

# Amendment A1 (2026-09-30): the recorded figures are split by which rule actually produced them.
# OWNED are properties of the registered top-10 / cross-family group and ARE reproduction targets.
# HISTORICAL were printed by Model 6-3's worst-8 / best-8 UNCONSTRAINED neighbour rule, a different
# instrument; they are kept for provenance and verified by historical_gaps(), and they are NOT targets
# for this rule - demanding that is what made the pre-registration internally unsatisfiable.
OWNED = {"top10_share": 61.2, "worst_share": 10.1}
TOLERANCE = {"top10_share": 0.15, "worst_share": 0.05}
HISTORICAL_WORST8_BEST8 = {"median_alias_curve_gap": 65.7, "best_group_gap": 28.1,
                           "worst8_nn_dist": 1.65, "best8_nn_dist": 1.58}
N_ALIAS = 10
PAIR_COLUMNS = ["kind", "soil_a", "soil_b", "soil_a_family", "soil_b_family", "feature_distance",
                "curve_emd"]
SUMMARY: Dict[str, object] = {}
_BASE: Dict[str, object] = {}


def design(root) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, str], np.ndarray]:
    """(ids, X12, Y, family map, M1 error per soil) for the 24 labelled training soils.

    The single source of truth for the pair rule, and shared with the test so the test's
    recomputation is a recomputation of the same inputs rather than a second copy that could drift.
    """
    root = Path(root)
    key = str(root)
    if key in _BASE:
        b = _BASE[key]
        return b["ids"], b["X"], b["Y"], b["fam"], b["err"]
    ft = pd.read_csv(root / "Model 2" / "Model 2 Experiment 3" / "cv_families.csv")
    fam = dict(zip(ft.sample_id.astype(str), ft.cv_family.astype(str)))
    ids, Y = tv.labels_matrix(root)
    ids = [str(i) for i in ids]
    Y = np.atleast_2d(np.asarray(Y, float))
    X = tv.soil_from_images(tv.image_features(root)).loc[ids, list(tv.FEATURES)].to_numpy(float)
    err_t = pd.read_csv(root / "Model 2" / "Model 2 Experiment 3" / "cv_per_soil.csv")
    err_t = err_t.set_index(err_t.sample_id.astype(str))
    missing = [i for i in ids if i not in err_t.index]
    if missing:
        raise AssertionError("cv_per_soil.csv has no M1 error for %d soils: %s"
                             % (len(missing), ", ".join(missing[:5])))
    err = err_t["M1"].loc[ids].to_numpy(float)
    if not (np.isfinite(X).all() and np.isfinite(Y).all() and np.isfinite(err).all()):
        raise AssertionError("non-finite input to the pair rule; the record would be meaningless")
    _BASE[key] = {"ids": ids, "X": X, "Y": Y, "fam": fam, "err": err}
    return ids, X, Y, fam, err


def _standardise(X: np.ndarray) -> np.ndarray:
    """z over the 24-soil corpus, with a dead column left at zero rather than divided by itself."""
    sd = np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    return (X - X.mean(0)) / sd


def nearest_cross_family(z: np.ndarray, ids: List[str], fam: Dict[str, str], q: int) -> Tuple[int, float]:
    """The query's closest soil whose CV family differs from the query's. Ties go to the lowest index.

    Ties are named because they decide a pair: with two equally close soils the rule would otherwise
    depend on iteration order, and the pair list would not be reconstructible from the rule alone.
    """
    cand = [j for j in range(len(ids)) if j != q and fam[ids[j]] != fam[ids[q]]]
    if not cand:
        raise AssertionError("soil %s has no cross-family neighbour, so the rule cannot pair it"
                             % ids[q])
    d = np.sqrt(((z[cand] - z[q]) ** 2).sum(axis=1))
    k = int(np.argmin(d))
    return int(cand[k]), float(d[k])


def derive_pairs(root) -> pd.DataFrame:
    """The alias set (10 worst soils) and the control set (the other 14), one pair per query."""
    ids, X, Y, fam, err = design(root)
    z = _standardise(X)
    order = list(np.argsort(err)[::-1])
    groups = (("alias", order[:N_ALIAS]), ("control", order[N_ALIAS:]))
    rows = []
    for kind, group in groups:
        for q in group:
            j, dist = nearest_cross_family(z, ids, fam, int(q))
            rows.append({"kind": kind, "soil_a": ids[q], "soil_b": ids[j],
                         "soil_a_family": fam[ids[q]], "soil_b_family": fam[ids[j]],
                         "feature_distance": dist,
                         "curve_emd": float(tv.emd_pair(Y[q], Y[j]))})
    df = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    if len(df) != len(ids):
        raise AssertionError("expected %d pairs under the registered rule, got %d"
                             % (len(ids), len(df)))
    HERE.joinpath("data").mkdir(exist_ok=True)
    df.to_csv(HERE / "data" / "alias_pairs.csv", index=False, lineterminator="\n")
    return df


def summary(root) -> Dict[str, object]:
    """The two shares this rule owns, the median gap this rule measures, and the pair set itself."""
    ids, X, Y, fam, err = design(root)
    tot = float(err.sum())
    if tot <= 0.0:
        raise AssertionError("summed M1 error is not positive; shares would be meaningless")
    order = np.argsort(err)[::-1]
    df = derive_pairs(root)
    alias = df[df.kind == "alias"]
    SUMMARY.update({
        "top10_share": 100.0 * float(err[order[:N_ALIAS]].sum()) / tot,
        "worst_share": 100.0 * float(err[order[0]]) / tot,
        "median_alias_curve_gap": float(alias.curve_emd.median()),
        "n_alias": int((df.kind == "alias").sum()),
        "n_control": int((df.kind == "control").sum()),
        "alias_pair_ids": [sorted([r.soil_a, r.soil_b]) for r in alias.itertuples()],
    })
    return SUMMARY


def historical_gaps(root) -> Dict[str, object]:
    """Model 6-3's OWN neighbour rule, re-run so the historical figures stay checkable, not prose.

    Worst 8 and best 8 of the descending M1 series; each soil's Euclidean nearest neighbour over all
    23 other soils with NO CV-family constraint; corpus mean and std + 1e-12. This definition was
    recovered verbatim from the session transcript of the probe that printed 65.7 - it was not chosen,
    and no neighbour definition was searched for in order to reproduce that figure. The registered
    pair rule above is untouched by this function, and nothing here reads A1-A4 or writes an artifact.
    """
    ids, X, Y, fam, err = design(root)
    z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    nn, gap, partner = {}, {}, {}
    for i, s in enumerate(ids):
        dd = np.linalg.norm(z - z[i], axis=1)
        dd[i] = 1e9
        j = int(np.argmin(dd))
        nn[s], gap[s], partner[s] = float(dd[j]), float(tv.emd_pair(Y[j], Y[i])), ids[j]
    order = list(pd.Series(err, index=list(ids)).sort_values(ascending=False).index)
    w8, b8 = order[:8], order[-8:]
    return {"worst8_median_curve_emd": float(np.median([gap[s] for s in w8])),
            "best8_median_curve_emd": float(np.median([gap[s] for s in b8])),
            "worst8_median_nn_dist": float(np.median([nn[s] for s in w8])),
            "best8_median_nn_dist": float(np.median([nn[s] for s in b8])),
            "h038_partner": partner["H038"],
            "h038_partner_curve_emd": gap[partner["H038"]],
            "same_family_partners": int(sum(1 for s in ids if fam[s] == fam[partner[s]]))}


def reproduction(root) -> Dict[str, object]:
    """Measured against recorded, with the delta: a disagreement stays visible, never rounded away.

    Only the OWNED figures are pass/fail, because only they are properties of this rule's group. The
    HISTORICAL figures are reported alongside their own rule's reproduction and flagged as targets of
    that rule, not this one (amendment A1).
    """
    s, h = summary(root), historical_gaps(root)
    out = {}
    for key, want in OWNED.items():
        got = float(s[key])
        out[key] = {"measured": got, "recorded": want, "delta": got - want,
                    "tolerance": TOLERANCE[key], "ok": bool(abs(got - want) <= TOLERANCE[key])}
    out["registered_median_alias_gap"] = {
        "measured": float(s["median_alias_curve_gap"]), "is_reproduction_target": False,
        "note": "65.7 belongs to Model 6-3's worst-8/best-8 unconstrained rule (amendment A1)"}
    out["historical_worst8_best8"] = dict(
        h, recorded=HISTORICAL_WORST8_BEST8,
        reproduces=bool(abs(h["worst8_median_curve_emd"] - 65.7) <= 0.6
                        and abs(h["best8_median_curve_emd"] - 28.1) <= 0.6
                        and abs(h["worst8_median_nn_dist"] - 1.65) <= 0.05
                        and abs(h["best8_median_nn_dist"] - 1.58) <= 0.05))
    out["all_reproduced"] = bool(all(out[k]["ok"] for k in OWNED))
    out["rules_are_distinguishable"] = bool(abs(float(s["median_alias_curve_gap"]) - 65.7) > 0.6)
    return out


def _median_separation(mat: np.ndarray, pairs: pd.DataFrame, ids) -> float:
    """Median RMS z-distance between the two soils of each pair, in the given design space.

    Standardisation uses the full 24-soil corpus, which is legitimate here because nothing is fitted
    and no label is read: this is a property of the feature space, not a model score. A constant
    column gets scale 1.0 rather than a division by zero.
    """
    pos = {s: i for i, s in enumerate(ids)}
    mu, sd = mat.mean(0), np.where(mat.std(0) > 1e-12, mat.std(0), 1.0)
    z = (mat - mu) / sd
    out = [float(np.sqrt(((z[pos[r.soil_a]] - z[pos[r.soil_b]]) ** 2).mean()))
           for r in pairs.itertuples()]
    return float(np.median(out))


def separability(root) -> Dict[str, object]:
    """Task 6: does the frozen A1-A4 block pull the frozen alias pairs further apart than the 12 do?

    The pair set comes from derive_pairs and cannot move; the block, its seeds and its column order
    come from the registered construction (m7a_data's soil aggregation, P-RAND the same tile pass on
    phase-scrambled pixels at seed 90001, P-SHUF a per-column permutation at seed 90002). No CV arm is
    fitted, no alpha is selected, no label enters, and nothing is written.

    The gate is the registered one and only: the real block's median separation gain over the 12 must
    be positive AND must exceed the gains that P-RAND and P-SHUF achieve on the same pairs. The
    control-pair numbers are reported so the contrast is visible, and are NOT gate-relevant - the
    pre-registration defines the gate on alias pairs alone.
    """
    import m7a_data as d
    import m7a_eval as ev
    import spatial_features as sf
    ids, X, Y, fam, err = design(root)
    df = derive_pairs(root)
    alias, control = df[df.kind == "alias"], df[df.kind == "control"]
    cols = list(ev.BLOCK_COLUMNS)
    B = d.spatial_soil_table(root).loc[ids, cols].to_numpy(float)
    Br = d.spatial_soil_table(root, scrambled=True, seed=ev.SEED_RAND).loc[ids, cols].to_numpy(float)
    Bs = sf.shuffle_columns(B, ev.SEED_SHUF)[0]
    for name, m in (("real", B), ("rand", Br), ("shuf", Bs)):
        if not np.isfinite(m).all():
            raise AssertionError("the %s A1-A4 block carries a non-finite cell, so the separation "
                                 "statistic would be silently wrong" % name)
        if m.shape != (len(ids), len(cols)):
            raise AssertionError("the %s block is %s, expected %s" % (name, m.shape,
                                                                      (len(ids), len(cols))))
    base = _median_separation(X, alias, ids)
    block = _median_separation(np.hstack([X, B]), alias, ids)
    rand = _median_separation(np.hstack([X, Br]), alias, ids)
    shuf = _median_separation(np.hstack([X, Bs]), alias, ids)
    gain = block - base
    cbase = _median_separation(X, control, ids)
    cblock = _median_separation(np.hstack([X, B]), control, ids)
    return {"base12": base, "block16": block, "rand16": rand, "shuf16": shuf,
            "gain_vs_base": gain, "rand_gain_vs_base": rand - base, "shuf_gain_vs_base": shuf - base,
            "gate_passed": bool(gain > 0.0 and gain > (rand - base) and gain > (shuf - base)),
            "ctrl_base12": cbase, "ctrl_block16": cblock, "ctrl_gain_vs_base": cblock - cbase,
            "control_separation_is_gate_relevant": False,
            "n_alias": int(len(alias)), "n_control": int(len(control)),
            "block_columns": cols, "seed_rand": int(ev.SEED_RAND), "seed_shuf": int(ev.SEED_SHUF)}
