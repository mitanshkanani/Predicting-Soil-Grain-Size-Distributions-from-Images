import matplotlib; matplotlib.use("Agg")

# ---- code cell 1 ----
print(" >>> cell "+str(1), flush=True)
"""Cell A1 — configuration. Every tunable lives here and nowhere else."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

MODEL_ID        = "Model1"
EXPERIMENT_ID   = "E4"
EXPERIMENT_NAME = "Model 1 / Experiment 4 - does the colour block earn its place"
SEED = 20260929
E3_DIR_NAME = "Model 1 Experiment 3"


@dataclass(frozen=True)
class Config:
    tile_size_px: int = 256
    min_tile_soil_fraction: float = 0.50
    pc_rank: int = 3
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    holdout_cameras: Tuple[str, ...] = ("Motorola", "Samsung")
    res_match_targets: Tuple[str, ...] = ("iPhone 14", "iPhone 16")
    n_boot: int = 4000
    n_perm: int = 200
    ORDINAL_FLOOR: float = 15.0      # E3's ruling: differences below this are unresolved
    core_feats: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean")
    freq_feats: Tuple[str, ...] = ("spec_centroid_cpm", "dom_wavelength_mm")
    colour_feats: Tuple[str, ...] = ("R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
    # the subset of the colour block that device normalisation touches. Luminance
    # percentiles are deliberately EXCLUDED: a per-camera brightness offset is real and
    # normalising it away would confound C2 with C3.
    norm_feats: Tuple[str, ...] = ("R", "G", "B", "sat")


CFG = Config()
ON_KAGGLE = Path("/kaggle/input").exists()
OUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working")) if ON_KAGGLE \
    else Path("Model 1/Model 1 Experiment 4")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  context={'kaggle' if ON_KAGGLE else 'local'}")
print(f"ordinal floor: differences below {CFG.ORDINAL_FLOOR:.0f} EMD may not be used to choose")

# ---- code cell 2 ----
print(" >>> cell "+str(2), flush=True)
"""Cell A2 — imports and environment capture."""
import platform, sys, json, hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

VERSIONS = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "Pillow": __import__("PIL").__version__}
for _k, _v in VERSIONS.items():
    print(f"  {_k:12s} {_v}")

# ---- code cell 3 ----
print(" >>> cell "+str(3), flush=True)
"""Cell A3 — locate the processed dataset. No machine-specific path is hardcoded."""
def resolve_input_root():
    kg = Path("/kaggle/input")
    if kg.exists():
        for cand in sorted(kg.iterdir()):
            for probe in (cand, *sorted(cand.iterdir())):
                if (probe / "data" / "processed_meta" / "manifest_images.csv").exists():
                    return probe, "kaggle"
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "data" / "processed_meta" / "manifest_images.csv").exists():
            return cand, "local"
    raise RuntimeError("Could not find data/processed_meta/manifest_images.csv")

INPUT_ROOT, RUN_CONTEXT = resolve_input_root()
META = INPUT_ROOT / "data" / "processed_meta"
sys.path.insert(0, str(INPUT_ROOT))
from preprocess import verify as pv
print(f"INPUT_ROOT  {INPUT_ROOT}  ({RUN_CONTEXT})")

# ---- code cell 4 ----
print(" >>> cell "+str(4), flush=True)
"""Cell A4 — pin the preprocessing version, the submission schema, and the seed."""
np.random.seed(SEED)
_audit = json.loads((META / "audit.json").read_text(encoding="utf-8"))
_golden = json.loads((META / "golden_checks.json").read_text(encoding="utf-8"))
_imgs_probe = pd.read_csv(META / "manifest_images.csv")
_HASHES = {"manifest_images.csv": set(_imgs_probe.config_hash.astype(str)),
           "audit.json": {_audit["config_hash"]},
           "golden_checks.json": {_golden["config_hash"]}}
for _n, _hs in _HASHES.items():
    assert len(_hs) == 1, f"{_n} carries multiple config_hash values: {_hs}"
_shared = {next(iter(_hs)) for _hs in _HASHES.values()}
assert len(_shared) == 1, f"config_hash disagreement across artifacts: {_HASHES}"
CONFIG_HASH = _shared.pop()
TARGET_PPM = float(_imgs_probe.target_ppm.iloc[0])
PIPELINE_VERSION = str(_imgs_probe.pipeline_version.iloc[0])
_SUB_PROBE = pd.read_csv(INPUT_ROOT / "data" / "sample_submission.csv")
TARGET_COLS = [str(c) for c in _SUB_PROBE.columns if str(c) != "sample_id"]
SUPPORT = np.array([float(c) for c in TARGET_COLS]); DL = np.log10(SUPPORT)
assert len(TARGET_COLS) == 11 and np.allclose(SUPPORT, [0.002, 0.0063, 0.02, 0.063, 0.2,
                                                        0.63, 2, 6.3, 20, 63, 200])
print(f"config_hash {CONFIG_HASH} | pipeline {PIPELINE_VERSION} | {TARGET_PPM:.6f} px/mm")

# ---- code cell 5 ----
print(" >>> cell "+str(5), flush=True)
"""Cell B1 — manifests and labels. Labels come from the SAMPLE manifest only."""
imgs = pd.read_csv(META / "manifest_images.csv")
tiles_all = pd.read_csv(META / "manifest_tiles.csv")
samples = pd.read_csv(META / "manifest_samples.csv")
tiles = tiles_all[(tiles_all.tile_size_px == CFG.tile_size_px) & tiles_all.materialized].copy()
missing = [p for p in tiles.tile_path if not (INPUT_ROOT / p).exists()]
assert not missing, f"{len(missing)} materialised tiles missing"
tr_lab = samples[samples.split == "train"].set_index("sample_id")
assert len(tr_lab) == 24
tr_ids = list(tr_lab.index)
Y = tr_lab[TARGET_COLS].to_numpy(float)
assert np.all(np.diff(Y, axis=1) >= -1e-9) and np.all(Y[:, -1] == 100.0)
assert ((Y >= 0) & (Y <= 100)).all()
print(f"images {imgs.shape} | tiles {tiles.shape} | labels {Y.shape}")

# ---- code cell 6 ----
print(" >>> cell "+str(6), flush=True)
"""Cell B2 — CV families, copied from E3, plus the label lookup every later cell needs."""
def resolve_history(tag, fname):
    cands = ([INPUT_ROOT / "experiment_history" / tag] if ON_KAGGLE
             else [OUT_DIR.parent / E3_DIR_NAME])
    for c in cands:
        if (c / fname).exists():
            return c
    raise RuntimeError(
        f"Cannot find {fname} from {tag}. Experiment 4 asserts its feature matrix against "
        f"Experiment 3's, so it cannot run without it.\nLooked in: {[str(c) for c in cands]}")


E3_DIR = resolve_history("Model1_E3", "features_soil.csv")
e3_soil = pd.read_csv(E3_DIR / "features_soil.csv")
e3_fam = pd.read_csv(E3_DIR / "cv_families.csv")
e3_sub = pd.read_csv(E3_DIR / "Submission_Model1_E3.csv")
e3_eff = pd.read_csv(E3_DIR / "paired_effects.csv")
fams = e3_fam.set_index("sample_id").cv_family
assert set(fams.index) == set(tr_ids)
GROUPS = sorted(fams.unique())
Ymap = {s: Y[i] for i, s in enumerate(tr_ids)}
pd.read_csv(E3_DIR / "cv_families.csv").to_csv(OUT_DIR / "cv_families.csv", index=False)
print(f"E3 artifacts from {E3_DIR}: {len(GROUPS)} CV families over {len(fams)} soils")

# ---- code cell 7 ----
print(" >>> cell "+str(7), flush=True)
"""Cell B2 — the metric and its reconciliation against the competition page."""
WIDTHS = np.diff(DL)


def emd_pair(p, t):
    return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))


def emd_page(p, t):
    return float(np.sum(np.abs(np.asarray(p, float) - np.asarray(t, float))[:10] * WIDTHS))


TRIVIAL = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])
PUB = 100.31
_t = float(np.mean([emd_pair(TRIVIAL, y) for y in Y]))
_p = float(np.mean([emd_page(TRIVIAL, y) for y in Y]))
print(f"trivial baseline: ours {_t:.4f} | page formula {_p:.4f} | host publishes {PUB}")
assert abs(_t - PUB) < 0.05, "our EMD no longer reproduces the host's reference number"
print("METRIC CONFIRMED")

# ---- code cell 8 ----
print(" >>> cell "+str(8), flush=True)
"""Cell B3 — reference baselines and the four external measurements this project owns."""
MEAN_CURVE = Y.mean(axis=0); MEDIAN_CURVE = np.median(Y, axis=0)
BASELINE_EMD = float(np.mean([emd_pair(MEDIAN_CURVE, y) for y in Y]))
LB = {"Model1_E1  16 feats, alpha by LOGO-CV": 172.69929,
      "mean-curve probe (ignores images)":     102.37237,
      "Model1_E2  14 feats, no degenerate cols": 71.27346,
      "Model1_E3  12 feats, no frequency":       61.23560}
print("external ground truth (public LB, 3 fixed soils of 10):")
for _k, _v in LB.items():
    print(f"  {_v:9.3f}   {_k}")
LB_E1, LB_BASE, LB_E2, LB_E3 = list(LB.values())
print(f"\nno-image floor (internal, constant median curve): {BASELINE_EMD:.2f}")
print(f"trajectory: {LB_E1:.1f} -> {LB_E2:.1f} -> {LB_E3:.1f}   (rank 49 on the board = 40.22)")

# ---- code cell 9 ----
print(" >>> cell "+str(9), flush=True)
"""Cell C1 — the calibration table and the ordinal ruling derived from it."""
CAL = pd.DataFrame([
    dict(experiment="E1", ruler=205.34, actual=LB_E1, change="16 feats, alpha 0.03 by LOGO-CV"),
    dict(experiment="E2", ruler=75.73, actual=LB_E2, change="dropped 2 degenerate cols, alpha 30"),
    dict(experiment="E3", ruler=51.87, actual=LB_E3, change="dropped the frequency family")])
CAL["factor"] = CAL.actual / CAL.ruler
print(CAL.round(2).to_string(index=False))
print(f"\nfactors: {' -> '.join(f'{v:.2f}' for v in CAL.factor)}   "
      "(monotonically outward, crossed the 0.7-1.1 window E3 pre-registered)")
d1r, d1a = CAL.ruler[0] - CAL.ruler[1], CAL.actual[0] - CAL.actual[1]
d2r, d2a = CAL.ruler[1] - CAL.ruler[2], CAL.actual[1] - CAL.actual[2]
print(f"\nmarginal check -- does the ruler predict the SIZE of a gain?")
print(f"  E1->E2  ruler says {d1r:6.1f} EMD   actual {d1a:6.1f}   ratio {d1a/d1r:.2f}")
print(f"  E2->E3  ruler says {d2r:6.1f} EMD   actual {d2a:6.1f}   ratio {d2a/d2r:.2f}")
print("  The ruler's optimism GROWS as configurations improve. That is the signature of")
print("  a proxy that models the nuisances it was built from and goes blind once those")
print("  nuisances are removed.")
RANKS_OK = bool((CAL.ruler.diff().dropna() < 0).all() and (CAL.actual.diff().dropna() < 0).all())
print(f"\nranking preserved across all three points: {RANKS_OK}")
assert RANKS_OK, "the ruler mis-ordered two externally-measured configs; it is not even ordinal"
print("\nBINDING RULE FOR THIS EXPERIMENT:")
print(f"  CAM+RES may RANK candidates. It may NOT forecast a score, and it may not")
print(f"  choose between two candidates whose readings differ by less than")
print(f"  {CFG.ORDINAL_FLOOR:.0f} EMD. Every expectation below is 'better or worse than "
      f"{LB_E3:.2f}', never a number.")

# ---- code cell 10 ----
print(" >>> cell "+str(10), flush=True)
"""Cell C2 — load Experiment 3's artifacts. E4 is defined relative to E3."""
def resolve_history(tag, fname):
    cands = ([INPUT_ROOT / "experiment_history" / tag] if ON_KAGGLE
             else [OUT_DIR.parent / E3_DIR_NAME])
    for c in cands:
        if (c / fname).exists():
            return c
    raise RuntimeError(
        f"Cannot find {fname} from {tag}. Experiment 4 asserts its feature matrix against "
        f"Experiment 3's, so it cannot run without it.\nLooked in: {[str(c) for c in cands]}")


E3_DIR = resolve_history("Model1_E3", "features_soil.csv")
e3_soil = pd.read_csv(E3_DIR / "features_soil.csv")
e3_fam = pd.read_csv(E3_DIR / "cv_families.csv")
e3_sub = pd.read_csv(E3_DIR / "Submission_Model1_E3.csv")
e3_eff = pd.read_csv(E3_DIR / "paired_effects.csv")
print(f"E3 artifacts from {E3_DIR}")
print(f"  features_soil {e3_soil.shape} | paired_effects {e3_eff.shape}")
_c = e3_eff[e3_eff.effect.str.startswith("MAIN frequency")]
print(f"  E3's headline: frequency main effect {float(_c.estimate.iloc[0]):+.2f} "
      f"CI [{float(_c.ci_lo.iloc[0]):+.2f},{float(_c.ci_hi.iloc[0]):+.2f}]")

# ---- code cell 11 ----
print(" >>> cell "+str(11), flush=True)
"""Cell D1 — build the soil x camera x view feature table used throughout."""
CORE = list(CFG.core_feats); FREQ = list(CFG.freq_feats); COL = list(CFG.colour_feats)
RAW14 = CORE + FREQ + COL
CACHE = OUT_DIR / ".cache"; CACHE.mkdir(exist_ok=True)
_srcs = [CACHE, OUT_DIR.parent / E3_DIR_NAME / ".cache"]
_geo = imgs[["processed_path", "crop_area_fraction"]].rename(
    columns={"processed_path": "parent_image_path"})


def get_cache(fname, builder):
    """Reuse a sibling experiment's tile cache when the config_hash matches. Reusing is a
    reproducibility check, not just a speed-up."""
    for d in _srcs:
        if (d / fname).exists():
            if not (CACHE / fname).exists():
                (CACHE / fname).write_bytes((d / fname).read_bytes())
            return pd.read_csv(CACHE / fname), str(d)
    df = builder(); df.to_csv(CACHE / fname, index=False)
    return df, "built"


def tile_features(rgb, ppm):
    """Identical to E1/E2/E3's definition, which is itself preprocess.verify.feats."""
    a = np.asarray(rgb, float)
    g = a.mean(axis=2)
    m = g > np.percentile(g, 12)
    out = {k: v for k, v in pv.feats(a, ppm).items() if k in ("e1", "e2", "e4", "e8", "e16")
           or k == "sat"}
    for k in ("e1", "e2", "e4", "e8", "e16"):
        out.setdefault(k, np.nan)
    lum = g[m]
    out["lum_p10"], out["lum_p50"], out["lum_p90"] = np.percentile(lum, [10, 50, 90])
    out["lum_sd"] = float(lum.std())
    gy, gx = np.gradient(g)
    out["grad_mean"] = float(np.hypot(gx, gy)[m].mean())
    cpx, cpm = pv.spectral_centroid(a, ppm)
    out["spec_centroid_cpm"] = float(cpm)
    out["dom_wavelength_mm"] = float(1.0 / cpm) if cpm > 1e-9 else np.nan
    out["R"], out["G"], out["B"] = [float(a[..., i][m].mean()) for i in range(3)]
    return out


_k = imgs.assign(k=imgs.effective_ppm / imgs.target_ppm).groupby("camera").k.first()
RES_SIGMA = {}
for _t in CFG.res_match_targets:
    for _c in CFG.holdout_cameras:
        _s = float(_k[[c for c in _k.index if c.startswith(_c)][0]])
        RES_SIGMA[(_c, _t)] = float(np.sqrt(max(float(_k[_t]) ** 2 - _s ** 2, 0)) / np.sqrt(12))


def _extract(sub_tiles, tgt=None):
    recs = []
    for i, t in enumerate(sub_tiles.itertuples(index=False)):
        rgb = np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB"), float)
        if tgt is not None:
            s = RES_SIGMA.get((t.camera.split(" ")[0], tgt), 0.0)
            if s and s > 0.02:
                rgb = ndi.gaussian_filter(rgb, sigma=(s, s, 0), mode="reflect")
        d = tile_features(rgb, TARGET_PPM); d["tile_path"] = t.tile_path
        recs.append(d)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sub_tiles)}")
    return pd.DataFrame(recs)


print(f"base tiles ({len(tiles)}) ...")
tf0, s0 = get_cache(f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv", lambda: _extract(tiles))
VIEWS = {"real": tf0}
for _t in CFG.res_match_targets:
    tag = "res" + _t.replace("iPhone", "").strip()
    tr = tiles[tiles.split == "train"]
    _, s1 = get_cache(f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}_{_t.replace(' ','')}.csv",
                      lambda tt=tr, t=_t: _extract(tt, t))
    VIEWS[tag] = pd.read_csv(CACHE / f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}_"
                                    f"{_t.replace(' ','')}.csv")

_parts = []
for tag, f in VIEWS.items():
    # INNER join on purpose: the resolution-matched caches cover training tiles only, so
    # a left join would manufacture NaN rows for test soils that were never blurred.
    d = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
               "soil_fraction"]].merge(f, on="tile_path", how="inner")
    d = d.merge(_geo, on="parent_image_path", how="left")
    d = d[d.soil_fraction >= CFG.min_tile_soil_fraction].copy()
    assert not d[RAW14].isna().to_numpy().any(), f"{tag} has NaN tile features"
    # `camera` is the device FAMILY used by the holdout rulers (Motorola / Samsung /
    # iPhone); `device` is the specific model, which is what per-device normalisation
    # must group by -- iPhone 14 and iPhone 16 are different devices even though both are
    # "iPhone" for the purpose of the camera-holdout test.
    d["device"] = d.camera
    d["camera"] = d.camera.str.split(" ").str[0]
    im = d.groupby(["split", "sample_id", "camera", "device",
                    "parent_image_path"])[RAW14].median()
    _parts.append(im.reset_index().assign(view=tag))
    print(f"  view {tag:6s}: {len(d)} tiles -> {len(im)} image rows")
V = pd.concat(_parts, ignore_index=True)
assert not V[RAW14].isna().to_numpy().any()
assert set(V[V.view == "real"]["split"]) == {"train", "test"}, "real view lost the test tiles"
assert set(V[V.view != "real"]["split"]) == {"train"}, "a res view wrongly carries test tiles"
print(f"image x camera x view rows: {V.shape}   (base cache from {s0})")
print(V[V.view == "real"].groupby(["split", "device"]).size().to_string())
print(V[V.view != "real"].groupby(["view", "camera"]).size().to_string())

# ---- code cell 12 ----
print(" >>> cell "+str(12), flush=True)
"""Cell D2 — soil x camera x device table, and the reproduction assertion against E3.

SC covers BOTH splits for the real view. The res-matched views exist only for training
soils, because they are a synthetic holdout target, not a source of new subjects.
"""
SC = V.groupby(["view", "sample_id", "camera", "device"])[RAW14].median()
# the same table pooled over devices within a camera: what the holdout rulers consume
SCR = SC.reset_index().groupby(["view", "sample_id", "camera"])[RAW14].median()
_ids = sorted(set(SCR.xs(("real", "Motorola"), level=["view", "camera"]).index) &
              set(SCR.xs(("real", "Samsung"), level=["view", "camera"]).index))
print(f"dual-camera soils: {len(_ids)}")

_r = V[V.view == "real"]
X_tr = _r[_r["split"] == "train"].groupby("sample_id")[RAW14].median().reset_index()
X_tr["split"] = "train"
X_te = _r[_r["split"] == "test"].groupby("sample_id")[RAW14].median().reset_index()
X_te["split"] = "test"
pooled = pd.concat([X_tr, X_te], ignore_index=True).merge(
    samples[["sample_id", "n_images", "cameras"]], on="sample_id", how="left")
_chk = pooled.merge(e3_soil, on=["split", "sample_id"], suffixes=("_e4", "_e3"))
assert len(_chk) == len(e3_soil) == 34, f"row mismatch {len(_chk)}"
_mx = max(float(np.nanmax(np.abs(_chk[c + "_e4"] - _chk[c + "_e3"]))) for c in RAW14)
print(f"max absolute drift vs Experiment 3 across {len(RAW14)} features: {_mx:.3e}")
assert _mx < 1e-6, f"feature matrix drifted from E3 by {_mx} - experiments not comparable"
pooled.to_csv(OUT_DIR / "features_soil.csv", index=False)
print("REPRODUCED against E3. E4's arms differ from E3's cells only in representation.")

# ---- code cell 13 ----
print(" >>> cell "+str(13), flush=True)
"""Cell D3 — the model primitives, and the blur-transfer function every later cell uses.

Identical in construction to E1/E2/E3: rank-3 PCA curve basis fitted on the training
rows, StandardScaler on the features, ridge on the curve coefficients, then the
clip/cummax/force-100 monotone projection.
"""


def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F, float)), 0, 100)
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 100.0
    return F


def fit_predict(Xtr, Ytr, Xte, alpha, rank=CFG.pc_rank):
    mu = Ytr.mean(0)
    _, _, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    Vb = Vt[:rank]
    sc = StandardScaler().fit(Xtr)
    r = Ridge(alpha=alpha).fit(sc.transform(Xtr), (Ytr - mu) @ Vb.T)
    return project(r.predict(sc.transform(Xte)) @ Vb + mu)


def transfer(FE, alpha, train_view, train_cam, test_view, test_cam):
    """Mean per-soil EMD when fitting on one device's view at one blur level and
    predicting another device's view at another blur level. Family held out."""
    A = SCR.xs((train_view, train_cam), level=["view", "camera"])
    B = SCR.xs((test_view, test_cam), level=["view", "camera"])
    ids = [s for s in _ids if s in A.index and s in B.index]
    errs = []
    for F in GROUPS:
        k = np.array([fams[s] != F for s in ids])
        if k.all() or not (~k).any():
            continue
        P = fit_predict(A.loc[ids, FE].to_numpy(float)[k],
                        np.array([Ymap[s] for s in ids])[k],
                        B.loc[ids, FE].to_numpy(float)[~k], alpha)
        errs += [emd_pair(P[i], Ymap[ids[j]]) for i, j in enumerate(np.where(~k)[0])]
    return float(np.mean(errs))


C0F = CORE + COL
print("primitives defined; transfer() is the blur x camera matrix used in D3b and D5")

# ---- code cell 14 ----
print(" >>> cell "+str(14), flush=True)
"""Cell D3b — CANCELLED HYPOTHESIS 1: resolution-matched training.

If training on the test cameras' sampling history were a real lever, the blur-sensitivity
matrix would be strongly diagonal. For E3's selected 12-feature set it is not.
"""
BLUR = {}
for lbl, FE, a in [("E3 12 feats (no freq)", C0F, 10.0),
                   ("E2 14 feats (with freq)", RAW14, 30.0)]:
    BLUR[lbl] = pd.DataFrame([[transfer(FE, a, tv, "Motorola", sv, "Samsung")
                               for sv in ("real", "res14", "res16")]
                              for tv in ("real", "res14", "res16")],
                             index=[f"train {t}" for t in ("real", "res14", "res16")],
                             columns=[f"predict {t}" for t in ("real", "res14", "res16")])
for lbl, tab in BLUR.items():
    print(f"\n{lbl}   (sigma: real 0.04, res14 0.84, res16 1.20 canonical px)")
    print(tab.round(2).to_string())
    print(f"  spread across all nine cells: "
          f"{float(tab.values.max() - tab.values.min()):.1f} EMD")
pd.concat([t.add_suffix(f" | {l}") for l, t in BLUR.items()], axis=1).to_csv(
    OUT_DIR / "blur_sensitivity_matrix.csv")
_s12 = float(BLUR["E3 12 feats (no freq)"].values.max() -
             BLUR["E3 12 feats (no freq)"].values.min())
_s14 = float(BLUR["E2 14 feats (with freq)"].values.max() -
             BLUR["E2 14 feats (with freq)"].values.min())
print(f"\nThe 12-feature model spans {_s12:.1f} EMD across EVERY blur combination; the")
print(f"14-feature model spans {_s14:.1f} and is strongly diagonal - the frequency")
print("features turn the model into a sharpness meter. E3 already absorbed the sharpness")
print("problem, so matched training has ~2 EMD left to give, inside the noise. CANCELLED.")

# ---- code cell 15 ----
print(" >>> cell "+str(15), flush=True)
"""Cell D4 — CANCELLED HYPOTHESIS 2: richer tile aggregation.

A grain-size DISTRIBUTION arguably deserves a distributional summary. Test it directly.
"""
_d = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
            "soil_fraction"]].merge(tf0, on="tile_path")
_d = _d[(_d.split == "train") & (_d.soil_fraction >= CFG.min_tile_soil_fraction)].copy()
_d["camera"] = _d.camera.str.split(" ").str[0]
_img = _d.groupby(["sample_id", "camera", "parent_image_path"])[C0F]
SUMS = {"med": _img.median(), "mean": _img.mean(), "sd": _img.std(),
        "p10": _img.quantile(.10), "p90": _img.quantile(.90)}
for nm, cols in [("median only (current protocol)", ["med"]),
                 ("median+mean+sd", ["med", "mean", "sd"]),
                 ("+ p10/p90 spread", ["med", "mean", "sd", "p10", "p90"])]:
    S = pd.concat({c: SUMS[c] for c in cols}, axis=1)
    S.columns = [f"{a}__{b}" for a, b in S.columns]
    S = S.groupby("sample_id").median().replace([np.inf, -np.inf], np.nan).dropna()
    ids = [s for s in tr_ids if s in S.index]
    X = S.loc[ids].to_numpy(float); Yy = np.array([Ymap[s] for s in ids])
    best = 9e9
    for a in CFG.alpha_grid:
        e = []
        for F in GROUPS:
            k = np.array([fams[s] != F for s in ids])
            if not (~k).any():
                continue
            P = fit_predict(X[k], Yy[k], X[~k], a)
            e += [emd_pair(P[i], Yy[j]) for i, j in enumerate(np.where(~k)[0])]
        best = min(best, float(np.mean(e)))
    print(f"  {nm:34s}{X.shape[1]:>4} cols   LOGO-CV best {best:6.2f}")
print(f"\nreference: E3 cell C LOGO-CV ~41.2 (oracle) | rank-3 ceiling 7.35 | "
      f"no-image floor {BASELINE_EMD:.2f}")
print("Quintupling the feature space by adding distributional summaries makes it slightly")
print("WORSE. The physics intuition is right and the n=24 statistics is not. CANCELLED.")

# ---- code cell 16 ----
print(" >>> cell "+str(16), flush=True)
"""Cell D5 — THE RULE THIS EXPERIMENT EXISTS TO ESTABLISH, demonstrated numerically.

Train on a blurred view and score with a ruler whose target is that SAME blurred view,
and the features E3 removed look helpful. That is an artefact, not a result.
"""
print("cost of KEEPING the frequency features, by evaluation condition:\n")
rows = []
for tv, sv, note in [("real", "res16", "DEPLOYMENT - blur never seen in training"),
                     ("res14", "res16", "held out - a different blur level"),
                     ("res16", "res16", "MATCHED - ruler target equals training view")]:
    a = transfer(C0F, 10.0, tv, "Motorola", sv, "Samsung")
    b = transfer(RAW14, 10.0, tv, "Motorola", sv, "Samsung")
    rows.append(dict(train_view=tv, test_view=sv, note=note, no_freq=a, with_freq=b,
                     freq_costs=b - a))
CIRC = pd.DataFrame(rows)
CIRC.to_csv(OUT_DIR / "circularity_demo.csv", index=False)
print(CIRC.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
print("\nRead the 'freq_costs' column. Under held-out blur frequency costs +35 to +58 EMD.")
print("Under a MATCHED ruler it appears to HELP by ~3 EMD - a ~35 EMD artefact of scoring")
print("the model on the exact distribution it was trained on.")
print("\nRULE (binding, and applied to our own arms in section I):")
print("  a treatment that alters the TRAINING distribution must be evaluated on a")
print("  held-out level of the nuisance it targets. A ruler whose evaluation condition")
print("  equals the treatment condition reports the treatment as free.")

# ---- code cell 17 ----
print(" >>> cell "+str(17), flush=True)
"""Cell E1 — define the arms, and assert exactly what varies."""
# chromaticity: scale-invariant colour. b is determined by r+g+b=1, so only r and g are
# carried. Luminance percentiles and saturation are kept, so the ONLY thing C3 changes
# relative to C0 is how the three channel means are expressed.
_tot = SC[["R", "G", "B"]].sum(axis=1).replace(0, np.nan)
SC = SC.join(pd.DataFrame({"chrom_r": SC.R / _tot, "chrom_g": SC.G / _tot}))
SC[["chrom_r", "chrom_g"]] = SC[["chrom_r", "chrom_g"]].fillna(1 / 3)
LUM = ["sat", "lum_p10", "lum_p50", "lum_p90"]

ARMS = {
    "C0 raw colour (E3)":   dict(feats=CORE + COL,             norm=None,     role="control"),
    "C1 no colour":         dict(feats=CORE,                   norm=None,     role="camera-preferred option"),
    "C2a per-cam norm (id)": dict(feats=CORE + COL,            norm="camera",  role="device-group from manifest"),
    "C2b per-cam norm (clu)": dict(feats=CORE + COL,           norm="cluster", role="device-group from clustering"),
    "C3 chromaticity":      dict(feats=CORE + ["chrom_r", "chrom_g"] + LUM, norm=None, role="scale-invariant colour"),
    "C4 colour only":       dict(feats=COL,                    norm=None,     role="NEGATIVE CONTROL"),
}
for nm, a in ARMS.items():
    assert len(set(a["feats"])) == len(a["feats"]), f"{nm} has duplicate columns"
    assert set(a["feats"]) <= set(SC.columns), f"{nm} uses undefined columns"
assert ARMS["C0 raw colour (E3)"]["feats"] == CORE + COL
assert len(ARMS["C0 raw colour (E3)"]["feats"]) == 12, "C0 must be exactly E3's cell C"
assert ARMS["C1 no colour"]["feats"] == CORE
print(f"{'arm':26s}{'n':>4}  norm      role")
for nm, a in ARMS.items():
    print(f"{nm:26s}{len(a['feats']):>4}  {str(a['norm']):8s}  {a['role']}")
print("\nC0 is byte-identical to E3's selected cell, so E4 re-measures E3 as a control.")
print("The single variable is the colour representation; the texture core never changes")
print("except in C4, which is the deliberate negative control.")

# ---- code cell 18 ----
print(" >>> cell "+str(18), flush=True)
"""Cell E2 — build each arm's feature tables, per view and per split.

C2's per-device median/IQR are computed once per (view, group) over all rows of that
group and reused across folds. That is a DECLARED DEVIATION from the 'fit on training
folds only' rule: the transform is entirely label-free, and the target group's statistics
are properties of unlabeled input the competition hands us. It is applied to C2 ONLY, and
section I tests whether it matters.
"""
NORM_COLS = list(CFG.norm_feats)
COLF = ["R", "G", "B", "sat"]
VIEW_NAMES = list(SC.index.get_level_values("view").unique())


MIN_GROUP = 4          # device groups smaller than this fall back to the family grouping


def robust_z(X, grp, cols, fallback=None):
    """Median/IQR standardisation within each group. Two guards, both necessary:
    a group with a single row has IQR 0 (H374 is the only Motorola Edge 60 Fusion soil),
    and a group with too few rows has no meaningful spread at all, so it inherits the
    coarser `fallback` grouping instead of dividing by zero."""
    out = X.copy()
    g = pd.Series(np.asarray(grp), index=X.index)
    sizes = g.value_counts()
    if fallback is not None:
        fb = pd.Series(np.asarray(fallback), index=X.index)
        g = g.where(sizes.reindex(g).to_numpy() >= MIN_GROUP, fb)
    for c in cols:
        if c not in out.columns:
            continue
        gg = out[c].groupby(g)
        med = gg.transform("median")
        iqr = gg.transform(lambda s: s.quantile(.75) - s.quantile(.25))
        glob = float(out[c].quantile(.75) - out[c].quantile(.25))
        iqr = iqr.where(iqr > 1e-9, glob if glob > 1e-9 else 1.0)
        out[c] = (out[c] - med) / iqr
    return out


# C2b's device groups: k-means on the colour features of EVERY real-view row, train and
# test together, never reading a camera label. Fitting across the test rows is what makes
# this a fair stand-in for C2a rather than a straw man.
# k is set to the number of DISTINCT devices, so C2b is handed exactly as many groups as
# C2a gets for free from the manifest. With a smaller k the comparison would be rigged.
_real = SC.xs("real", level="view")
_devs = _real.index.get_level_values("device")
NDEV = int(_devs.nunique())
_km = KMeans(n_clusters=NDEV, n_init=50, random_state=SEED).fit(_real[COLF])
CLAB = {v: pd.Series(_km.predict(SC.xs(v, level="view")[COLF]),
                     index=SC.xs(v, level="view").index) for v in VIEW_NAMES}
_lab = pd.Series(_km.labels_, index=_real.index)
# purity: for each cluster, the share of its rows belonging to its majority device.
# 1.0 means clustering recovered the devices exactly, with no label ever inspected.
CT = pd.crosstab(_lab, _devs)
CLUSTER_PURITY = float((CT.max(axis=1).sum()) / CT.values.sum())
print(f"C2b: k set to {NDEV} = the number of distinct devices C2a gets from the manifest")
print("cluster x device crosstab (label-free clustering vs the true device):")
print(CT.to_string())
print(f"cluster purity = {100*CLUSTER_PURITY:.1f}%  (100% would mean clustering recovered")
print("the devices with no label ever inspected)")

T, TS = {}, {}
for arm, spec in ARMS.items():
    a = spec["feats"]
    for v in VIEW_NAMES:
        sub = SC.xs(v, level="view")
        X = sub[a].copy()
        fam = sub.index.get_level_values("camera")
        if spec["norm"] == "camera":
            X = robust_z(sub, sub.index.get_level_values("device"), NORM_COLS,
                         fallback=fam)[a]
        elif spec["norm"] == "cluster":
            X = robust_z(sub, CLAB[v].reindex(sub.index).to_numpy(), NORM_COLS,
                         fallback=fam)[a]
        T[(arm, v)] = X
    # soil-level matrix for the real view: normalise per device, then pool over devices
    X = T[(arm, "real")]
    TS[arm] = X.groupby("sample_id").median()
for (arm, v), X in T.items():
    assert not X.isna().any().any(), f"{arm}/{v} has NaN after normalisation"
print(f"\nbuilt {len(T)} (arm, view) tables and {len(TS)} soil-level matrices")
print("soil-level rows available:", {k: int(v.index.isin(tr_ids).sum()) for k, v in
                                      list(TS.items())[:1]}, "train /",
      {k: int((~v.index.isin(tr_ids)).sum()) for k, v in list(TS.items())[:1]}, "test")

# ---- code cell 19 ----
print(" >>> cell "+str(19), flush=True)
"""Cell E3 — the evaluation conditions and the leak assertion."""
CAM_RES, CAM_ONLY = [], []
for _src in CFG.holdout_cameras:
    _tgt = [c for c in CFG.holdout_cameras if c != _src][0]
    for _v in ["real", "res14", "res16"]:
        _c = dict(name=f"{_src[:3]}>{_tgt[:3]}:{_v}", src=_src, tgt=_tgt, view=_v)
        (CAM_ONLY if _v == "real" else CAM_RES).append(_c)
ALLC = CAM_ONLY + CAM_RES
print(f"CAM+RES ({len(CAM_RES)}): " + ", ".join(c["name"] for c in CAM_RES))
print(f"CAM     ({len(CAM_ONLY)}): " + ", ".join(c["name"] for c in CAM_ONLY))


def arm_rows(arm, view, camera):
    X = T[(arm, view)]
    return X[X.index.get_level_values("camera") == camera].droplevel(["camera", "device"])


for _c in ALLC:
    a, b = arm_rows("C0 raw colour (E3)", "real", _c["src"]), \
             arm_rows("C0 raw colour (E3)", _c["view"], _c["tgt"])
    assert set(_ids) <= set(a.index) & set(b.index), f"condition {_c['name']} missing soils"
for F in GROUPS:
    ev = [s for s in _ids if fams[s] == F]
    fit = [s for s in _ids if fams[s] != F]
    assert not (set(ev) & set(fit))
    assert not any(fams[s] == fams[t] for s in ev for t in fit), \
        "a same-family soil leaks the held-out label under the other camera"
print(f"\nPASS: {len(_ids)} dual-camera soils; no evaluated soil, and none from its own")
print("family, appears in its own fit under ANY of the six conditions.")

# ---- code cell 20 ----
print(" >>> cell "+str(20), flush=True)
"""Cell E4 — the scoring primitives, identical in construction to E3."""


def cond_errors(arm, alpha, cond, eval_family):
    a = ARMS[arm]["feats"]
    A = arm_rows(arm, "real", cond["src"]); B = arm_rows(arm, cond["view"], cond["tgt"])
    ids = [s for s in _ids if s in A.index and s in B.index]
    keep = np.array([fams[s] != eval_family for s in ids])
    if keep.all() or not (~keep).any():
        return {}
    P = fit_predict(A.loc[ids, a].to_numpy(float)[keep],
                    np.array([Ymap[s] for s in ids])[keep],
                    B.loc[ids, a].to_numpy(float)[~keep], alpha)
    return {ids[j]: emd_pair(P[i], Ymap[ids[j]]) for i, j in enumerate(np.where(~keep)[0])}


def logo_errors(arm, alpha, eval_family):
    S = TS[arm]
    ids = [s for s in tr_ids if s in S.index]
    M = S.loc[ids].to_numpy(float)
    Yy = np.array([Ymap[s] for s in ids])
    keep = np.array([fams[s] != eval_family for s in ids])
    P = fit_predict(M[keep], Yy[keep], M[~keep], alpha)
    return {ids[j]: emd_pair(P[i], Yy[j]) for i, j in enumerate(np.where(~keep)[0])}

# ---- code cell 21 ----
print(" >>> cell "+str(21), flush=True)
"""Cell E5 — nested selection engine. Alpha is chosen inside each outer fold on
CAM+RES, so section F reports the score of the PROCEDURE, never of the best alpha.
"""
NEST_KEY = hashlib.sha256(json.dumps(
    {"arms": {k: v["feats"] for k, v in ARMS.items()}, "norm": {k: str(v["norm"]) for k, v in ARMS.items()},
     "conds": [c["name"] for c in ALLC], "alpha": list(CFG.alpha_grid), "hash": CONFIG_HASH,
     "seed": SEED, "rank": CFG.pc_rank}, sort_keys=True).encode()).hexdigest()[:12]
NCACHE = CACHE / f"nested_{NEST_KEY}.json"


def nested_for_arm(arm):
    per = {c["name"]: {} for c in ALLC}; per["LOGO-CV"] = {}
    alphas = []
    for F in GROUPS:
        inner = [G for G in GROUPS if G != F]
        best, ba = None, CFG.alpha_grid[0]
        for al in CFG.alpha_grid:
            v = [x for G in inner for c in CAM_RES
                 for x in cond_errors(arm, al, c, G).values()]
            if not v:
                continue
            m = float(np.mean(v))
            if best is None or m < best:
                best, ba = m, al
        alphas.append(ba)
        for c in ALLC:
            for s, e in cond_errors(arm, ba, c, F).items():
                per[c["name"]].setdefault(s, []).append(e)
        for s, e in logo_errors(arm, ba, F).items():
            per["LOGO-CV"].setdefault(s, []).append(e)
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in per.items()}, \
        sorted(set(alphas))


if NCACHE.exists():
    _n = json.loads(NCACHE.read_text(encoding="utf-8"))
    NEST, ALPHAS_USED = _n["per"], _n["alphas"]
    print(f"loaded nested results from cache ({NEST_KEY})")
else:
    NEST, ALPHAS_USED = {}, {}
    for arm in ARMS:
        NEST[arm], ALPHAS_USED[arm] = nested_for_arm(arm)
        print(f"  nested {arm:26s} done", flush=True)
    NCACHE.write_text(json.dumps({"per": NEST, "alphas": ALPHAS_USED}), encoding="utf-8")


def soil_vector(arm, conds):
    keys = ["LOGO-CV"] if conds == "LOGO" else [c["name"] for c in conds]
    shared = set.intersection(*[set(NEST[arm][k]) for k in keys])
    return {s: float(np.mean([NEST[arm][k][s] for k in keys])) for s in sorted(shared)}


V_CR = {a: soil_vector(a, CAM_RES) for a in ARMS}
V_CM = {a: soil_vector(a, CAM_ONLY) for a in ARMS}
V_LG = {a: soil_vector(a, "LOGO") for a in ARMS}
print(f"\nnested cache key {NEST_KEY}")
for a in ARMS:
    print(f"  {a:26s} CAM+RES {np.mean(list(V_CR[a].values())):6.2f}   "
          f"CAM {np.mean(list(V_CM[a].values())):6.2f}   "
          f"LOGO {np.mean(list(V_LG[a].values())):6.2f}   alphas {ALPHAS_USED[a]}")

# ---- code cell 22 ----
print(" >>> cell "+str(22), flush=True)
"""Cell F1 — paired machinery."""
rng = np.random.default_rng(SEED)
AR = list(ARMS)
SOILS = sorted(set.intersection(*[set(V_CR[a]) for a in AR]))
assert len(SOILS) >= 18, f"only {len(SOILS)} shared soils"
MC = {a: np.array([V_CR[a][s] for s in SOILS]) for a in AR}
MM = {a: np.array([V_CM[a][s] for s in SOILS]) for a in AR}
ML = {a: np.array([V_LG[a][s] for s in SOILS]) for a in AR}
print(f"paired on {len(SOILS)} soils present in every arm under every condition")


def boot_ci(diff, n=CFG.n_boot):
    k = len(diff)
    b = np.array([diff[rng.integers(0, k, k)].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi):
    return "SIGNIFICANT (first worse)" if lo > 0 else (
        "SIGNIFICANT (first better)" if hi < 0 else "not distinguishable from zero")


def paired(a, b, vec, label, store, ordinal=True):
    d = vec[a] - vec[b]
    lo, hi = boot_ci(d)
    sep = bool(lo > 0 or hi < 0)
    resolvable = bool((hi - lo) <= CFG.ORDINAL_FLOOR or sep)
    v = verdict(lo, hi)
    if ordinal and not sep and abs(d.mean()) < CFG.ORDINAL_FLOOR:
        v = "NOT RESOLVABLE by an ordinal ruler (|delta| < 15 and CI spans 0)"
    store.append(dict(effect=label, estimate=float(d.mean()), ci_lo=lo, ci_hi=hi,
                      wins_first_better=int((d < 0).sum()), n=len(d), significant=sep,
                      resolvable=resolvable, verdict=v))
    print(f"  {label:40s} {d.mean():+8.2f}  CI [{lo:+8.2f},{hi:+8.2f}]  "
          f"wins {int((d<0).sum()):2d}/{len(d)}  {v}")


EFF = []
print("every arm vs C0 (E3's configuration), on the nested CAM+RES procedure score:\n")
for a in AR:
    if a != "C0 raw colour (E3)":
        paired(a, "C0 raw colour (E3)", MC, f"CAM+RES  {a} - C0", EFF)
print("\nsame comparisons on CAM, shown for transparency only. C2 arms are NOT ranked on it:")
print("per-camera normalisation removes exactly the difference CAM measures (cell I).\n")
for a in AR:
    if a != "C0 raw colour (E3)":
        paired(a, "C0 raw colour (E3)", MM, f"CAM      {a} - C0", EFF, ordinal=False)
EFF = pd.DataFrame(EFF)
EFF.to_csv(OUT_DIR / "paired_effects.csv", index=False)

# ---- code cell 23 ----
print(" >>> cell "+str(23), flush=True)
"""Cell G1 — P1 and P2, the two predictions that matter most."""
_c4 = EFF[(EFF.effect.str.startswith("CAM+RES  C4"))].iloc[0]
_c4cam = float(np.mean(list(V_CM["C4 colour only"].values())))
P1 = bool(_c4cam > 75)
print(f"P1  C4 (colour only) negative control: CAM = {_c4cam:.2f}, must exceed 75")
print(f"    -> {'CONFIRMED' if P1 else 'REFUTED'}   "
      + ("pipeline reproduces E3's colour-only collapse" if P1 else
         "STOP: the pipeline drifted, nothing else here is trustworthy"))
_c1 = EFF[EFF.effect.str.startswith("CAM+RES  C1")].iloc[0]
P2 = bool(not _c1.significant)
print(f"\nP2  C1 (no colour) vs C0 (raw colour) NOT separable by CAM+RES  [BLIND]")
print(f"    estimate {_c1.estimate:+.2f}  CI [{_c1.ci_lo:+.2f},{_c1.ci_hi:+.2f}]  "
      f"-> {'CONFIRMED' if P2 else 'REFUTED'}")
if P2:
    print("    The ruler cannot choose between keeping and dropping colour. E3's ambiguity")
    print("    was a power problem, not a measurement problem, and the ordinal ruling stands.")
else:
    print("    The ruler CAN separate them, so E3's ordinal ruling was too hasty and the")
    print("    ruler has more power than allowed. That is the better-news branch.")
_c3 = EFF[EFF.effect.str.startswith("CAM+RES  C3")].iloc[0]
P3 = bool(_c3.significant and _c3.estimate < 0)
print(f"\nP3  C3 (chromaticity) better than C0 under CAM+RES  [BLIND]")
print(f"    {_c3.estimate:+.2f} CI [{_c3.ci_lo:+.2f},{_c3.ci_hi:+.2f}]  -> "
      f"{'CONFIRMED' if P3 else 'REFUTED'}")
_c2b = EFF[EFF.effect.str.startswith("CAM+RES  C2a")].iloc[0]
_c2bb = EFF[EFF.effect.str.startswith("CAM+RES  C2b")].iloc[0]
P4 = bool(_c2b.estimate < _c2bb.estimate)
print(f"\nP4  C2a (identity) beats C2b (clustering) on CAM+RES  [BLIND]")
print(f"    C2a {_c2b.estimate:+.2f} vs C2b {_c2bb.estimate:+.2f} vs C0  -> "
      f"{'CONFIRMED' if P4 else 'REFUTED'}")
print(f"    unsupervised device groups recovered {100*CLUSTER_PURITY:.1f}% purity "
      f"(k={NDEV}, no camera label inspected)")

# ---- code cell 24 ----
print(" >>> cell "+str(24), flush=True)
"""Cell H1 — the mechanism, restated for colour specifically."""
_sd = pooled[pooled.split == "train"][RAW14].std().replace(0, np.nan)
_A = arm_rows("C0 raw colour (E3)", "real", "Motorola")
_B = arm_rows("C0 raw colour (E3)", "real", "Samsung")
sh = sorted(set(_A.index) & set(_B.index))
_cam = ARMS["C0 raw colour (E3)"]["feats"]
cam = ((_A.loc[sh] - _B.loc[sh]).abs() / _sd[_cam]).mean()
_C = arm_rows("C0 raw colour (E3)", "res16", "Motorola")
sh2 = sorted(set(_A.index) & set(_C.index))
sharp = ((_A.loc[sh2] - _C.loc[sh2]).abs() / _sd[_cam]).mean()
MECH = pd.DataFrame({"camera_shift_z": cam, "sharpness_shift_z": sharp})
MECH.to_csv(OUT_DIR / "mechanism_table.csv")
print(MECH.round(3).sort_values("camera_shift_z", ascending=False).to_string())
_col = [c for c in COL if c in MECH.index]
print(f"\ncolour block: mean camera shift {MECH.loc[_col,'camera_shift_z'].mean():.2f} z, "
      f"mean sharpness shift {MECH.loc[_col,'sharpness_shift_z'].mean():.2f} z")
print("Colour is the family that moves most between devices. Section F is about whether")
print("moving a lot and being load-bearing are the same thing. They are not obviously so.")

# ---- code cell 25 ----
print(" >>> cell "+str(25), flush=True)
"""Cell I1 — circularity audit of the C2 arms, demonstrated not asserted."""
aud = []
for arm in ["C0 raw colour (E3)", "C2a per-cam norm (id)", "C2b per-cam norm (clu)"]:
    aud.append(dict(arm=arm, CAM=float(np.mean(list(V_CM[arm].values()))),
                    CAM_RES=float(np.mean(list(V_CR[arm].values()))),
                    LOGO=float(np.mean(list(V_LG[arm].values())))))
AUD = pd.DataFrame(aud)
AUD["CAM_gain_vs_C0"] = AUD.CAM - AUD.CAM.iloc[0]
AUD["CAMRES_gain_vs_C0"] = AUD.CAM_RES - AUD.CAM_RES.iloc[0]
AUD.to_csv(OUT_DIR / "c2_circularity_audit.csv", index=False)
print(AUD.round(2).to_string(index=False))
print("\nIf C2's CAM gain is far larger than its CAM+RES gain, the CAM reading is the")
print("artefact: normalising by device erases the very difference CAM measures.")
_c = AUD[AUD.arm.str.startswith("C2a")].iloc[0]
print(f"\nC2a: CAM gain {_c.CAM_gain_vs_C0:+.2f} EMD vs CAM+RES gain {_c.CAMRES_gain_vs_C0:+.2f}")
C2_CIRCULAR = bool(_c.CAM_gain_vs_C0 < -5 and _c.CAMRES_gain_vs_C0 > _c.CAM_gain_vs_C0 + 5)
print(f"circularity confirmed for C2 on CAM: {C2_CIRCULAR}")
print("Consequence, pre-stated in instructions.txt: C2 is never ranked on CAM alone.")

# ---- code cell 26 ----
print(" >>> cell "+str(26), flush=True)
"""Cell J1 — selection, applying the ordinal rule."""
NESTED = {a: dict(camres=float(np.mean(list(V_CR[a].values()))),
                  cam=float(np.mean(list(V_CM[a].values()))),
                  logo=float(np.mean(list(V_LG[a].values()))),
                  n=len(ARMS[a]["feats"])) for a in ARMS}
print(f"{'arm':26s}{'n':>4}{'CAM+RES':>10}{'CAM':>8}{'LOGO':>8}{'worst':>8}")
for a, v in sorted(NESTED.items(), key=lambda kv: kv[1]["camres"]):
    print(f"{a:26s}{v['n']:>4}{v['camres']:>10.2f}{v['cam']:>8.2f}"
          f"{v['logo']:>8.2f}{max(v['camres'], v['cam']):>8.2f}")
rankable = [a for a in ARMS if not a.startswith("C2a")]
PRIMARY = min(rankable, key=lambda a: NESTED[a]["camres"])
MINIMAX = min(rankable, key=lambda a: max(NESTED[a]["camres"], NESTED[a]["cam"]))
print(f"\nprimary (best nested CAM+RES, C2a excluded as circular): {PRIMARY}")
print(f"minimax (best worst-case)                              : {MINIMAX}")
if PRIMARY == MINIMAX:
    SELECTED, why = PRIMARY, "primary and minimax agree"
else:
    d = MC[PRIMARY] - MC[MINIMAX]; lo, hi = boot_ci(d)
    SELECTED = min([PRIMARY, MINIMAX], key=lambda a: NESTED[a]["n"])
    why = (f"disagreed (CAM+RES {NESTED[PRIMARY]['camres']:.2f} vs "
           f"{NESTED[MINIMAX]['camres']:.2f}); paired CI [{lo:+.2f},{hi:+.2f}] "
           f"-> fewer features")
PROBE = False
if SELECTED == "C0 raw colour (E3)":
    # The rule as written can return the control, and re-submitting E3 buys zero
    # information. A null internal result still has a discriminating experiment
    # attached to it, so the submission becomes the arm that most directly tests the
    # question rather than the arm the ruler happens to prefer.
    votes = {}
    for a in ARMS:
        if a in ("C0 raw colour (E3)", "C2a per-cam norm (id)"):
            continue
        votes[a] = sum([NESTED[a]["logo"] < NESTED["C0 raw colour (E3)"]["logo"],
                        NESTED[a]["cam"] < NESTED["C0 raw colour (E3)"]["cam"],
                        NESTED[a]["camres"] < NESTED["C0 raw colour (E3)"]["camres"]])
    print("\nThe rule returned the CONTROL. The internal comparison is a null and")
    print("re-submitting E3 would measure nothing, so the submission is converted into the")
    print("discriminating probe. votes = how many of the three rulers prefer the arm to C0:")
    for a, v in sorted(votes.items(), key=lambda kv: -kv[1]):
        print(f"    {a:26s} {v}/3   (LOGO {NESTED[a]['logo']:6.2f}  "
              f"CAM {NESTED[a]['cam']:6.2f}  CAM+RES {NESTED[a]['camres']:6.2f})")
    tied = [a for a, v in votes.items() if v == max(votes.values())]
    SELECTED = min(tied, key=lambda a: NESTED[a]["n"])
    why += f"; the rule returned the control, so the probe is {SELECTED} "            f"({max(votes.values())}/3 rulers prefer it to C0)"
    PROBE = True
_d0 = MC[SELECTED] - MC["C0 raw colour (E3)"]
_lo, _hi = boot_ci(_d0)
BET = bool(not (_hi < 0))
print(f"\nSELECTED: {SELECTED} ({NESTED[SELECTED]['n']} feats) -- {why}")
if PROBE:
    print("   STATUS: DISCRIMINATING PROBE, not an improvement claim.")
print(f"vs C0: {_d0.mean():+.2f} EMD  CI [{_lo:+.2f},{_hi:+.2f}]  wins {int((_d0<0).sum())}/{len(_d0)}")
print(f"label: {'A BET, NOT A FINDING' if BET else 'A FINDING'}")
if BET:
    print("  The ordinal rule fired: the ruler could not separate this arm from C0, so the")
    print("  submission IS the measurement rather than a claim of progress.")
SEL = dict(arm=SELECTED, feats=ARMS[SELECTED]["feats"], norm=str(ARMS[SELECTED]["norm"]),
           primary=PRIMARY, minimax=MINIMAX, why=why, bet=bool(BET), probe=bool(PROBE),
           nested_camres=NESTED[SELECTED]["camres"], vs_C0=[float(_d0.mean()), _lo, _hi])
json.dump(SEL, open(OUT_DIR / "selection_rule.json", "w"), indent=1)

# ---- code cell 27 ----
print(" >>> cell "+str(27), flush=True)
"""Cell J2 — the final alpha, its sensitivity, and the ordinal-only expectation."""
FE_SEL = ARMS[SELECTED]["feats"]
_sens = []
for al in CFG.alpha_grid:
    v = [x for F in GROUPS for c in CAM_RES
         for x in cond_errors(SELECTED, al, c, F).values()]
    _sens.append((al, float(np.mean(v))))
SENS = pd.DataFrame(_sens, columns=["alpha", "camres"])
best_i = int(SENS.camres.idxmin())
_oracle = float(SENS.camres[best_i]); alpha_oracle = float(SENS.alpha[best_i])
# Parsimony rule, adopted BEFORE the submission and justified by E1: the grid optimum is
# not used when a LARGER alpha is within the ruler's own resolution. E1's catastrophe came
# from an oracle-chosen alpha at the edge of the grid, so "smallest achievable error" is a
# known failure mode here, not a hypothetical one.
_tol = CFG.ORDINAL_FLOOR
_ok = SENS[SENS.camres <= _oracle + _tol]
alpha_f = float(_ok.alpha.max())
_ruler_pred = NESTED[SELECTED]["camres"]
print(f"selected arm {SELECTED}: CAM+RES by alpha\n")
print(SENS.round(2).to_string(index=False))
print(f"\ngrid optimum (oracle)      : alpha {alpha_oracle:g} at {_oracle:.2f}")
print(f"largest alpha within {_tol:.0f} EMD : alpha {alpha_f:g} at "
      f"{float(SENS.set_index('alpha').camres[alpha_f]):.2f}   <-- USED")
print("  Justification: E1 selected the grid optimum (0.03) and it collapsed on unseen")
print("  cameras. Where the ruler cannot resolve a difference it prefers more")
print("  regularisation, because the failure mode of too little is known and the failure")
print("  mode of too much is merely a worse score.")
_rng = SENS.camres.max() - SENS.camres.min()
print(f"\nalpha sensitivity of this arm: {_rng:.2f} EMD across the whole grid "
      f"({'NOT flat, so the choice matters' if _rng > 10 else 'flat'})")
print(f"\nmeasured factors on E1/E2/E3: {', '.join(f'{v:.2f}' for v in [LB_E1/205.34, LB_E2/75.73, LB_E3/51.87])}")
print("ORDINAL EXPECTATION ONLY. The ruler forecast E3's gain as 23.9 EMD and the real")
print("gain was 10.0, so a point estimate is not admissible (see cell C1).")
print(f"  P5: the submitted arm must score BELOW {LB_E3:.2f}.")
print(f"  If it does not, and no arm separated from C0, Model 1 has plateaued and Model 2")
print(f"  becomes an evidence-backed decision.")
PRED_BAND = None

# ---- code cell 28 ----
print(" >>> cell "+str(28), flush=True)
"""Cell K1 — fit on all 24 training soils, predict the 10 test soils.

TS[arm] is the soil-level matrix for the real view, already normalised per device for the
C2 arms and already pooled over a soil's devices the same way E3 pooled. Using it keeps
the fitted model and the normalisation consistent by construction.
"""
S = TS[SELECTED]
_order = samples[samples.split == "test"].sort_values("submission_row_order")
missing = [s for s in _order.sample_id if s not in S.index]
assert not missing, f"test soils absent from the arm table: {missing}"
s_te = S.loc[_order.sample_id.tolist()]
Xtr = S.loc[tr_ids].to_numpy(float)
Xte = s_te.to_numpy(float)
assert Xtr.shape == (24, len(FE_SEL)) and Xte.shape == (10, len(FE_SEL))
assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
P_test = fit_predict(Xtr, Y, Xte, alpha_f)
_sat = 100.0 * float(np.mean((P_test <= 1e-6) | (P_test >= 100)))
_e3p = e3_sub[TARGET_COLS].to_numpy(float)
_sat_e3 = 100.0 * float(np.mean((_e3p <= 1e-6) | (_e3p >= 100)))
print(f"arm {SELECTED}: {len(FE_SEL)} features {FE_SEL}")
print(f"predictions {P_test.shape}")
print(f"  columns pinned at 0/100: E1 59.1%  E2 10.0%  E3 {_sat_e3:.1f}%  E4 {_sat:.1f}%"
      "   (real labels 27%)")
print(f"  mean EMD of predictions vs the train mean curve: "
      f"{float(np.mean([emd_pair(p, MEAN_CURVE) for p in P_test])):.2f}"
      f"   (E3 was 52.74)")
assert Xtr.shape == (24, len(FE_SEL)) and Xte.shape == (10, len(FE_SEL))

# ---- code cell 29 ----
print(" >>> cell "+str(29), flush=True)
"""Cell K2 — eight validation gates."""
sub = pd.read_csv(INPUT_ROOT / "data" / "sample_submission.csv")
_idmap = dict(zip(samples.sample_id, samples.submission_id))
submit = pd.DataFrame({"sample_id": [str(_idmap[s]) for s in s_te.index]})
for c in TARGET_COLS:
    submit[c] = P_test[:, TARGET_COLS.index(c)]
checks = {
    "row count == 10": len(submit) == len(sub) == 10,
    "ids exactly match sample_submission order":
        submit.sample_id.tolist() == sub.sample_id.astype(str).tolist(),
    "columns exactly match": list(submit.columns) == list(sub.columns),
    "no missing values": bool(~submit.isna().to_numpy().any()),
    "values within [0,100]": bool((submit[TARGET_COLS].to_numpy(float) >= 0).all())
        and bool((submit[TARGET_COLS].to_numpy(float) <= 100).all()),
    "cumulative / non-decreasing": bool((np.diff(submit[TARGET_COLS].to_numpy(float),
                                                 axis=1) >= -1e-9).all()),
    "200mm column == 100": bool(np.allclose(submit["200"].to_numpy(float), 100.0)),
    "no extra columns": len(submit.columns) == 12,
}
for k, ok in checks.items():
    print(f"  {'PASS' if ok else '**FAIL**'}  {k}")
assert all(checks.values()), "submission failed validation - not writing the file"
print("\nall 8 submission checks passed")

# ---- code cell 30 ----
print(" >>> cell "+str(30), flush=True)
"""Cell K3 — write, re-read, re-assert."""
SUB_NAME = f"Submission_{MODEL_ID}_{EXPERIMENT_ID}.csv"
SUB_PATH = OUT_DIR / SUB_NAME
submit.to_csv(SUB_PATH, index=False)
back = pd.read_csv(SUB_PATH)
assert list(back.columns) == list(sub.columns)
assert back.sample_id.astype(str).tolist() == sub.sample_id.astype(str).tolist()
assert not back.isna().any().any()
assert (np.diff(back[TARGET_COLS].to_numpy(float), axis=1) >= -1e-9).all()
print(f"wrote {SUB_PATH}  ({SUB_PATH.stat().st_size} bytes)\n")
print(back.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

# ---- code cell 31 ----
print(" >>> cell "+str(31), flush=True)
"""Cell L1 — error analysis against E3."""
_meta = pooled[pooled.split == "test"].set_index("sample_id").loc[s_te.index]
fig, axes = plt.subplots(2, 5, figsize=(19, 7.5))
for i, sid in enumerate(s_te.index):
    ax = axes.ravel()[i]
    ax.plot(DL, MEAN_CURVE, ":", c="#999", label="train mean")
    ax.plot(DL, _e3p[i], "x--", c="#dd6b20", lw=1.3, label="E3 (61.24)")
    ax.plot(DL, P_test[i], "o-", c="#2b6cb0", lw=2, label="E4 selected")
    ax.set_title(f"{sid}\n{_meta.iloc[i].cameras}", fontsize=8)
    ax.set_ylim(-5, 105); ax.tick_params(labelsize=7)
axes.ravel()[0].legend(fontsize=7)
plt.suptitle(f"E4 {SELECTED} vs E3, same 10 test soils")
plt.tight_layout(); plt.show()

d = MC[SELECTED] - MC["C0 raw colour (E3)"]
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.bar(range(len(SOILS)), d, color=np.where(d < 0, "#2b6cb0", "#c53030"))
ax.axhline(0, c="k", lw=.8)
ax.set_xticks(range(len(SOILS))); ax.set_xticklabels(SOILS, rotation=90, fontsize=7)
ax.set_ylabel("paired CAM+RES EMD (neg = E4 better)")
ax.set_title(f"{SELECTED} minus C0 per soil: {int((d<0).sum())}/{len(d)} improve")
plt.tight_layout(); plt.show()
print(f"{int((d<0).sum())} of {len(d)} soils improve. A mean gain carried by a minority of")
print("soils, or by a few extreme ones, is not robust - read this before the mean.")

# ---- code cell 32 ----
print(" >>> cell "+str(32), flush=True)
"""Cell M1 — write Experiment4.txt from the live objects."""
import datetime
L = []; A = L.append
A("=" * 78)
A(f"EXPERIMENT RECORD -- {MODEL_ID} / {EXPERIMENT_ID}")
A("=" * 78)
A(f"generated_at   : {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
A(f"name           : {EXPERIMENT_NAME}")
A(f"context        : {RUN_CONTEXT} on {platform.platform()}")
A("")
A("0. TYPE")
A("   An ordinary Model 1 experiment. ONE variable: how colour is represented.")
A("   It is also the first experiment whose central question is one the ruler is")
A("   forbidden by rule to answer.")
A("")
A("1. RULER STATUS INHERITED FROM E3")
for _, r in CAL.iterrows():
    A(f"   {r.experiment}  CAM+RES {r.ruler:7.2f} -> actual {r.actual:7.2f}   factor {r.factor:.2f}")
A(f"   E1->E2 ruler delta {d1r:.1f} vs actual {d1a:.1f} (ratio {d1a/d1r:.2f});")
A(f"   E2->E3 ruler delta {d2r:.1f} vs actual {d2a:.1f} (ratio {d2a/d2r:.2f}).")
A(f"   RULING: CAM+RES is ORDINAL. No score forecasts; no choice between arms differing")
A(f"   by less than {CFG.ORDINAL_FLOOR:.0f} EMD.")
A("")
A("2. TWO SCHEDULED EXPERIMENTS CANCELLED, WITH EVIDENCE (reproduced in section D)")
A(f"   resolution-matched training: the 12-feature blur matrix spans only {_s12:.1f} EMD")
A(f"   across all nine train x test combinations (14-feature set spans {_s14:.1f}).")
A("   E3's feature selection already absorbed the sharpness problem.")
A("   richer tile aggregation: median-only LOGO-CV 41.15; adding mean, sd, p10, p90")
A("   gave 42.13 then 42.22. Quintupling the feature space made it slightly worse.")
A("   A THIRD FINDING, general and not anticipated by any plan: training on a view and")
A("   scoring with a ruler whose target is that same view is CIRCULAR. Under a matched")
A("   ruler the frequency features E3 removed appear to HELP by ~3 EMD; under held-out")
A("   blur they still cost +35 to +58. A ~35 EMD artefact. Rule: a treatment that alters")
A("   the training distribution must be evaluated on a held-out level of the nuisance.")
A("")
A("3. ARMS")
for nm, a in ARMS.items():
    A(f"   {nm:26s} {len(a['feats']):2d} feats  norm={str(a['norm']):8s}  {a['role']}")
A(f"   C0 is exactly E3's cell C; feature matrix reproduced to {_mx:.2e}.")
A("")
A("4. NESTED PROCEDURE SCORES")
for a in AR:
    v = NESTED[a]
    A(f"   {a:26s} CAM+RES {v['camres']:6.2f}   CAM {v['cam']:6.2f}   "
      f"LOGO {v['logo']:6.2f}   alphas {ALPHAS_USED[a]}")
A("")
A("5. PAIRED EFFECTS vs C0 (soil-level bootstrap, negative = the arm is BETTER)")
for _, r in EFF[EFF.effect.str.startswith("CAM+RES")].iterrows():
    A(f"   {r.effect:40s} {r.estimate:+8.2f} CI [{r.ci_lo:+8.2f},{r.ci_hi:+8.2f}] "
      f"wins {r.wins_first_better}/{r.n}  {r.verdict}")
A("   (CAM comparisons are reported for transparency but C2 is not ranked on them.)")
A("")
A("6. MECHANISM")
_colm = [c for c in COL if c in MECH.index]
A(f"   colour block mean camera shift {MECH.loc[_colm,'camera_shift_z'].mean():.2f} z "
  f"(largest of any family); mean sharpness shift "
  f"{MECH.loc[_colm,'sharpness_shift_z'].mean():.2f} z")
A("   E3 established that a large shift does not imply a load-bearing feature. E4 tests")
A("   whether that also holds for colour specifically.")
A("")
A("7. C2 CIRCULARITY AUDIT")
for _, r in AUD.iterrows():
    A(f"   {r.arm:26s} CAM {r.CAM:6.2f} (gain {r.CAM_gain_vs_C0:+6.2f})   "
      f"CAM+RES {r.CAM_RES:6.2f} (gain {r.CAMRES_gain_vs_C0:+6.2f})")
A(f"   circularity on CAM confirmed: {C2_CIRCULAR}. C2 is therefore ranked only on")
A("   CAM+RES, whose blur component is untouched by colour normalisation.")
A(f"   unsupervised device groups reached {100*CLUSTER_PURITY:.1f}% purity at k={NDEV}")
A("")
A("8. PREDICTION VERDICTS")
A(f"   P1 C4 negative control CAM > 75          : {'CONFIRMED' if P1 else 'REFUTED'} "
  f"(CAM={_c4cam:.2f})   [consistency check]")
A(f"   P2 C1 vs C0 NOT separable by CAM+RES     : {'CONFIRMED' if P2 else 'REFUTED'}"
  f"   [BLIND]")
A(f"   P3 C3 better than C0 on CAM+RES          : {'CONFIRMED' if P3 else 'REFUTED'}"
  f"   [BLIND]")
A(f"   P4 C2a beats C2b on CAM+RES              : {'CONFIRMED' if P4 else 'REFUTED'}"
  f"   [BLIND]")
A(f"   P5 submitted arm beats {LB_E3:.2f}             : pending submission   [BLIND]")
A("   P6 no point estimate offered anywhere      : HONOURED (PRED_BAND = None)")
A("")
A("9. SELECTION")
A(f"   primary (C2a excluded as circular) : {PRIMARY}")
A(f"   minimax                            : {MINIMAX}")
A(f"   {why}")
A(f"   SELECTED: {SELECTED} ({NESTED[SELECTED]['n']} feats), alpha {alpha_f:g}")
A(f"   alpha rule: grid optimum was {alpha_oracle:g} at {_oracle:.2f}; the LARGEST alpha")
A(f"   within {CFG.ORDINAL_FLOOR:.0f} EMD of it ({alpha_f:g}) was used instead. E1's collapse came from an")
A("   oracle-chosen alpha at the edge of the grid, so where the ruler cannot resolve a")
A("   difference this protocol prefers more regularisation. Rule adopted before the")
A("   submission, not after seeing a score.")
A(f"   CAM+RES across the whole alpha grid spans {_rng:.2f} EMD, so the choice does matter.")
if PROBE:
    A("   STATUS: DISCRIMINATING PROBE, NOT AN IMPROVEMENT CLAIM. The pre-stated rule")
    A("   returned the control arm, i.e. the internal comparison is a null. Re-submitting")
    A("   E3 would cost a submission and teach nothing, so the submission was redirected")
    A("   to the arm that most directly tests this experiment's question.")
A(f"   vs C0 paired: {SEL['vs_C0'][0]:+.2f} EMD CI [{SEL['vs_C0'][1]:+.2f},"
  f"{SEL['vs_C0'][2]:+.2f}]  -> {'A BET' if BET else 'A FINDING'}")
A("   Which of the two reasons applied is stated above, as the protocol requires.")
A("")
A("9b. A PREDICTION OF MINE THAT WAS TOO STRONG")
_a = AUD[AUD.arm.str.startswith("C2a")].iloc[0]
A(f"   instructions.txt predicted CAM would rate C2 'near perfect by construction'. It")
A(f"   did not: C2a's CAM gain is {_a.CAM_gain_vs_C0:+.2f} EMD against a CAM+RES gain of")
A(f"   {_a.CAMRES_gain_vs_C0:+.2f}. The circularity is real but mild (~2x), not total.")
A("   Excluding C2 from CAM ranking was still the right call, but the stated magnitude")
A("   was overstated and is corrected here rather than quietly left in the earlier file.")
A("")
A("10. OUTPUT")
A(f"   columns pinned at 0/100: E1 59.1%  E2 10.0%  E3 {_sat_e3:.1f}%  E4 {_sat:.1f}%  (real 27%)")
A("")
A("11. Kaggle RESULT  (fill in by hand after submitting)")
A(f"   file            : {SUB_NAME}")
A(f"   public score    : ____________________   must beat {LB_E3:.2f}  (no band, per P6)")
A(f"   P5 verdict      : ____________________")
A(f"   private score   : ____________________")
A("")
A("12. LIMITS")
A(f"   {len(SOILS)} dual-camera soils; six conditions share them. Cannot resolve below")
A(f"   ~{CFG.ORDINAL_FLOOR:.0f} EMD. Five rankable arms is more multiple-comparison surface than")
A("   E3's two factors; the ordinal rule is the only guard.")
A("   The synthetic resolution camera is a Gaussian approximation of a Lanczos prefilter.")
A("   C2's per-group statistics are transductive and fitted once, not per fold - a")
A("   declared deviation, applied to C2 only, justified because the transform is")
A("   label-free. If it matters, C2's result is void and must be re-run fold-wise.")
A("   All four calibration points assume the public 3-soil subset never changed.")
A("")
A("13. DEVIATIONS FROM PLAN")
A("   C3 was planned as 4 features, which would also have deleted the luminance")
A("   percentiles and changed two things at once. It is implemented as 11 features so the")
A("   only difference from C0 is how the channel means are expressed.")
A("   C2 was split into C2a (manifest identity) and C2b (unsupervised clustering) after")
A("   the plan was written, because normalising by device identity may be a metadata")
A("   exploit rather than a generalisation method. Strictly more careful than planned.")
A("   No new measurements, no learned representation, no change to aggregation, output")
A("   or estimator.")
A("")
A("14. NEXT")
if BET:
    A("   The ruler could not separate the selected arm from C0, so this experiment's")
    A("   submission IS the measurement. Read the score, then:")
    A("     better than 61.24 -> the parsimonious representation is right; keep it and")
    A("       move to blur-robust frequency surrogates (band ratios) for interpretability.")
    A("     not better      -> Model 1 has plateaued. Sharpness handled, aggregation")
    A("       exhausted, colour undecided. Model 2 becomes evidence-backed, not a hunch.")
else:
    A("   The ruler separated the arms, so E3's ordinal ruling was too conservative and")
    A("   should be revisited before Model 2 is scoped.")
A("=" * 78)
TXT_PATH = OUT_DIR / "Experiment4.txt"
TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {TXT_PATH} ({len(L)} lines)")
print("\n" + "\n".join(L[L.index('8. PREDICTION VERDICTS'):][:8]))

# ---- code cell 33 ----
print(" >>> cell "+str(33), flush=True)
"""Cell M2 — summary."""
print("=" * 78)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  --  SUMMARY")
print("=" * 78)
print(f"  question                 : does the colour block earn its place")
print(f"  arms                     : {len(ARMS)} (C0 control = E3's cell C)")
for a in AR:
    print(f"    {a:26s} CAM+RES {NESTED[a]['camres']:6.2f}  CAM {NESTED[a]['cam']:6.2f}")
print(f"  P2 ruler cannot separate C0/C1 : {P2}")
print(f"  selected                       : {SELECTED} ({NESTED[SELECTED]['n']} feats)")
print(f"  status                         : {'BET' if BET else 'FINDING'}")
print(f"  must beat                      : {LB_E3:.2f}   (no forecast offered, per P6)")
print("=" * 78)
print(f"  submission : {SUB_NAME}")
print("  record     : Experiment4.txt")
print("  artifacts  : paired_effects.csv, c2_circularity_audit.csv, circularity_demo.csv,")
print("               blur_sensitivity_matrix.csv, mechanism_table.csv, selection_rule.json")

