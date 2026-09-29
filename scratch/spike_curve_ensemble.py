"""SPIKE - throwaway. NOT registered. NOT a submission. Nothing here is an experiment.

Question it exists to answer, in one line: does averaging the predicted CURVES of the hand
feature model and the DINOv2 model beat either model alone, at a fixed 50/50 weight?

Why that is a different question from Model 2 E3: E3 fused the two families as FEATURES
(396 columns into one ridge) and that was refuted. Averaging curves is a different mechanism -
it reduces error when the components are wrong about different soils, which is exactly what E1
found (D invariant but weaker; M1 accurate but fragile) and what E3 did not test.

Why it might be nothing: an ensemble of a model with a genuinely useful second view and an
ensemble of a model with pure noise can look identical. So M1 + SHUF (the row-permuted DINOv2
block, the arm E3 built as its noise control) is in this spike as the control arm. If M1+SHUF
gains as much as M1+D, the gain is averaging, not information, and there is nothing to register.

Protocol: the same nested LOGO-CV as E1/E2/E3 - 16 curve-distance families over 24 soils,
alpha chosen inside each outer fold from inner folds only, trapezoid EMD on log10 d. Curve
averaging happens AFTER per-arm projection, and the average is re-projected, so the blend is a
legal CDF and not an artefact of arithmetic.

Outputs go to scratch/ only. Delete freely. Run:
    python scratch/spike_curve_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "Model 2" / "Model 2 Experiment 3"
sys.path.insert(0, str(E3))
sys.path.insert(0, str(ROOT / "Model 2" / "Model 2 Experiment 2"))

import embedding_store as estore                                 # noqa: E402
import crop_geometry as cg                                       # noqa: E402

SUP = [str(c) for c in pd.read_csv(ROOT / "data" / "sample_submission.csv").columns
       if str(c) != "sample_id"]
SUPPORT = np.array([float(c) for c in SUP])
DL = np.log10(SUPPORT)
CONTROL = ("e4", "e8", "e16", "lum_sd", "grad_mean", "R", "G", "B", "sat",
           "lum_p10", "lum_p50", "lum_p90")
ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
          300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0, 300000.0, 1000000.0)
RANK, PC_SEED, N_BOOT = 3, 20260932, 4000


def expected_from_e3():
    """Read Model 2 E3's own recorded nested scores rather than transcribing them.

    The sanity gate is only worth something if the reference numbers come from the artifact
    that produced them. Typing 43.0217 by hand would let a transcription error masquerade as a
    pipeline difference, which is the exact confusion this gate exists to rule out.
    """
    d = pd.read_csv(E3 / "sweep_results.csv").set_index("arm")
    return {a: float(d.loc[a, "nested_in_domain"])
            for a in ("M1", "dinov2@c256", "vit_random@c256", "shuf@c256")}


def emd_pair(p, t):
    return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))


def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F, float)), 0, 100)
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 100.0
    return F


def fit_predict(Xtr, Ytr, Xte, alpha):
    """The frozen Model 1 / Model 2 head, transcribed from the notebook cells."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    mu = Ytr.mean(0)
    _, _, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    V = Vt[:RANK]
    sc = StandardScaler().fit(Xtr)
    T = (Ytr - mu) @ V.T
    r = Ridge(alpha=alpha).fit(sc.transform(Xtr), T)
    return project(r.predict(sc.transform(Xte)) @ V + mu)


def load_data():
    META = ROOT / "data" / "processed_meta"
    CONFIG_HASH = sorted(set(pd.read_csv(META / "manifest_images.csv")
                             .config_hash.astype(str)))[0]
    samples = pd.read_csv(META / "manifest_samples.csv")
    t_all = pd.read_csv(META / "manifest_tiles.csv")
    tiles = t_all[(t_all.tile_size_px == 256) & t_all.materialized].copy()
    good = tiles[tiles.soil_fraction >= 0.50].copy()
    good["camera_fam"] = good.camera.str.split(" ").str[0]
    tr = samples[samples.split == "train"].set_index("sample_id")
    tr_ids = list(tr.index)
    Y = tr[SUP].to_numpy(float)

    fam = pd.read_csv(E3 / "cv_families.csv")
    fams = dict(zip(fam.sample_id.astype(str), fam.cv_family))
    groups = sorted(set(fams.values()))

    hand = pd.read_csv(ROOT / "Model 1" / "Model 1 Experiment 3" / ".cache" /
                       f"tile_features_256_{CONFIG_HASH}.csv")
    hand = hand.drop_duplicates("tile_path").set_index("tile_path")[list(CONTROL)]

    def agg(mat, cols, index_name):
        sub = good[good.tile_path.isin(mat.index)]
        d = sub[["tile_path", "sample_id", "camera_fam",
                 "parent_image_path"]].merge(
            mat.rename_axis("tile_path").reset_index(), on="tile_path", how="inner")
        assert len(d) == len(sub), f"{index_name}: lost tiles in the join"
        im = d.groupby(["sample_id", "camera_fam", "parent_image_path"])[cols].median()
        return im.groupby("sample_id")[cols].median()

    emb_cols = [f"emb_{i:03d}" for i in range(384)]
    out, EMB_MAT = {}, {}
    for arm, (backend, crop) in {"dinov2@c256": ("dinov2", 256),
                                 "vit_random@c256": ("vit_random", 256)}.items():
        rec = estore.load_one(E3 / "embeddings", backend, crop, "real", META, 256,
                              CONFIG_HASH, 20260930)
        m = pd.DataFrame(rec["arr"], index=rec["names"], columns=emb_cols)
        EMB_MAT[arm] = m
        out[arm] = agg(m, emb_cols, arm)
    out["M1"] = agg(hand, list(CONTROL), "M1")
    # The control arm must be E3's SHUF exactly: hand block PLUS a row-permuted embedding
    # block (396 columns), not the permuted block on its own. Building the wrong arm here
    # would make the sanity gate fail for a reason that has nothing to do with the pipeline,
    # which is precisely the confusion the gate is meant to remove.
    import fusion as fu
    perm_emb, _vec = fu.permute_rows(out["dinov2@c256"], 20260933)
    out["shuf@c256"] = fu.fuse(out["M1"], perm_emb, list(CONTROL), emb_cols)

    # 34 soils carry tiles (24 labelled train + 10 test); the CV uses the 24 labelled ones.
    assert len(out["M1"]) == 34, f"{len(out['M1'])} soils, expected 34"
    for a in out:
        out[a] = out[a].loc[tr_ids]
        assert len(out[a]) == len(tr_ids) == 24, f"{a}: {len(out[a])} rows after slicing"
        assert not out[a].isna().to_numpy().any()
    print(f"loaded {len(out)} arms over {len(tr_ids)} soils, {len(groups)} CV families, "
          f"config_hash {CONFIG_HASH}")
    return out, EMB_MAT, Y, tr_ids, fams, groups, _vec


def cg_permute(df):
    """Row-permute the data, keep the index - E3's fusion.permute_rows semantics."""
    import fusion as fu
    return fu.permute_rows(df, 20260933)


def nested(XA, Y, tr_ids, fams, groups):
    """Per-arm nested LOGO-CV returning per-soil ERRORS and per-soil CURVES.

    Curves are what makes this a spike and not a re-print of E3: the errors alone cannot be
    blended, only predictions can. Alpha is chosen inside each outer fold from the inner folds
    only, so the reported number is the score of the procedure.
    """
    LERR, LPRED, err, curves = {}, {}, {}, {}
    for arm, X in XA.items():
        Xn = X.to_numpy(float)
        LERR[arm], LPRED[arm] = {}, {}
        for al in ALPHAS:
            for F in groups:
                hold = [i for i, s in enumerate(tr_ids) if fams[s] == F]
                keep = np.array([i for i in range(len(tr_ids)) if i not in hold])
                hidx = np.array(hold)
                P = fit_predict(Xn[keep], Y[keep], Xn[hidx], al)
                LERR[arm][(al, F)] = {tr_ids[i]: emd_pair(P[k], Y[i])
                                      for k, i in enumerate(hidx)}
                LPRED[arm][(al, F)] = {tr_ids[i]: P[k] for k, i in enumerate(hidx)}
        e, c = {}, {}
        for F in groups:
            inner = [G for G in groups if G != F]
            sc = {al: float(np.mean([v for G in inner for v in LERR[arm][(al, G)].values()]))
                  for al in ALPHAS}
            ba = min(sc, key=sc.get)
            for s, v in LERR[arm][(ba, F)].items():
                e[s] = v
                c[s] = LPRED[arm][(ba, F)][s]
        err[arm], curves[arm] = e, c
        print(f"  {arm:16s} nested {np.mean(list(e.values())):7.3f}")
    return err, curves


def blend(curva, curvb, w=0.5):
    return {s: project(w * np.asarray(curva[s], float) +
                       (1 - w) * np.asarray(curvb[s], float))[0] for s in curva}


def paired_boot(da, db, n=N_BOOT):
    d = np.asarray(da, float) - np.asarray(db, float)
    r = np.random.default_rng(PC_SEED)
    idx = np.arange(len(d))
    b = np.array([d[r.choice(idx, size=len(idx), replace=True)].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), \
        int((d < 0).sum())


def main():
    print("=" * 78)
    print("SPIKE: 50/50 curve-space ensemble of the hand model and the DINOv2 model")
    print("unregistered | exploratory | no submission | code is throwaway")
    print("=" * 78)
    XA, _, Y, tr_ids, fams, groups, perm = load_data()
    print("\nnested in-domain LOGO-CV, single arms:")
    err, curves = nested(XA, Y, tr_ids, fams, groups)

    print("\nsanity gate - single arms must reproduce Model 2 E3's recorded numbers:")
    EXPECTED = expected_from_e3()
    ok = True
    for a, ref in EXPECTED.items():
        got = float(np.mean(list(err[a].values())))
        good = abs(got - ref) <= 0.05
        ok &= good
        print(f"  [{'PASS' if good else '**FAIL**'}] {a:16s} spike {got:7.3f}  "
              f"E3 {ref:7.3f}  gap {abs(got-ref):.4f}")
    if not ok:
        print("\nABORT: the spike's own single-arm numbers do not match E3, so any ensemble")
        print("difference here is attributable to this script, not to blending.")
        return 1

    print("\nensembles at a FIXED 0.5 weight (no weight was selected on this data):")
    E = {a: np.array([err[a][s] for s in tr_ids]) for a in err}
    combos = [("M1 + D", "M1", "dinov2@c256"), ("M1 + R", "M1", "vit_random@c256"),
              ("M1 + SHUF  [CONTROL]", "M1", "shuf@c256")]
    rows = []
    tables = []
    _d50 = dict(zip(pd.read_csv(E3 / "cv_families.csv").sample_id.astype(str),
                    pd.read_csv(E3 / "cv_families.csv").D50_mm))
    for name, a, b in combos:
        cb = blend(curves[a], curves[b], 0.5)
        eb = [emd_pair(cb[s], Y[tr_ids.index(s)]) for s in tr_ids]
        gain, lo, hi, wins = paired_boot(eb, E[a])
        best_alone = min(float(E[a].mean()), float(E[b].mean()))
        sig = "SIGNIFICANT" if (lo > 0 or hi < 0) else "not significant"
        print(f"  {name:22s} blend {np.mean(eb):7.3f}  vs {a} {E[a].mean():7.3f}  "
              f"delta {gain:+7.3f}  CI [{lo:+7.3f}, {hi:+7.3f}]  {sig}")
        print(f"  {'':22s} (better single arm alone: {best_alone:.3f}; "
              f"soil improvements {wins}/{len(tr_ids)})")
        rows.append(dict(combo=name, blend=float(np.mean(eb)), base_a=float(E[a].mean()),
                         base_b=float(E[b].mean()), delta=gain, ci_lo=lo, ci_hi=hi,
                         significant=bool(lo > 0 or hi < 0), wins_a=wins))
        tables.append(pd.DataFrame(
            dict(sample_id=tr_ids, cv_family=[fams[s] for s in tr_ids],
                 soil_group=[cg.soil_group(float(_d50[s])) for s in tr_ids],
                 err_a=E[a], err_b=E[b], err_blend=eb,
                 delta_vs_a=np.asarray(eb) - E[a])).assign(combo=name))
    TAB = pd.concat(tables, ignore_index=True)
    TAB.drop(columns=["soil_group"], inplace=True)

    print("\nper-FOLD mean delta vs M1 (negative = blend better), by CV family:")
    for name in [c[0] for c in combos]:
        sub = TAB[TAB.combo == name]
        g = sub.groupby("cv_family").delta_vs_a.mean().sort_values()
        print(f"\n  {name}")
        print("    " + "  ".join(f"{k}:{v:+.1f}" for k, v in g.items()))
        print(f"    improved {int((g < 0).sum())}/{len(g)} folds; "
              f"the two best folds contribute {g.head(2).sum():+.1f} of "
              f"{g.sum():+.1f} total")

    print("\nper-SOIL delta vs M1 for the arm under test (M1 + D), worst-first:")
    sub = TAB[TAB.combo == "M1 + D"].sort_values("delta_vs_a")
    for r in sub.itertuples(index=False):
        print(f"    {r.sample_id}  M1 {r.err_a:7.2f}  D {r.err_b:7.2f}  "
              f"blend {r.err_blend:7.2f}  delta {r.delta_vs_a:+7.2f}  {r.cv_family}")
    d = sub.delta_vs_a.to_numpy()
    print(f"    spread: best {d.min():+.2f}, worst {d.max():+.2f}, "
          f"median {np.median(d):+.2f}; 1 soil carries "
          f"{100 * abs(d.min()) / max(abs(d).sum(), 1e-9):.0f}% of all movement")
    TAB.to_csv(ROOT / "scratch" / "spike_curve_ensemble_per_soil.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "scratch" / "spike_curve_ensemble.csv", index=False)

    print("\nwhat this spike can and cannot license:")
    print("  - M1 + SHUF is the control. If it gains like M1 + D, the effect is averaging,")
    print("    not information, and there is nothing to register.")
    print("  - A gain here is in-domain. CAM+RES is ordinal-only and the public LB scores 3")
    print("    of 10 soils, so nothing in this table forecasts a leaderboard number.")
    print("  - No weight tuning was performed. Choosing 0.5 in advance is the only reason the")
    print("    CI printed above means anything; a swept weight would make it meaningless.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
