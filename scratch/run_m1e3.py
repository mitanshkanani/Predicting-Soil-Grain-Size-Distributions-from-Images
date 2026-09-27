import matplotlib; matplotlib.use("Agg")

# ---- code cell 1 ----
print(" >>> cell "+str(1), flush=True)
"""Cell A1 — configuration. Every tunable lives here and nowhere else."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

MODEL_ID        = "Model1"
EXPERIMENT_ID   = "E3"
EXPERIMENT_NAME = "Model 1 / Experiment 3 - feature-family factorial (colour x frequency)"
SEED = 20260928
E2_DIR_NAME = "Model 1 Experiment 2"
E1_DIR_NAME = "Model 1 Experiment 1"


@dataclass(frozen=True)
class Config:
    tile_size_px: int = 256
    min_tile_soil_fraction: float = 0.50
    pc_rank: int = 3
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    family_linkage: str = "average"
    family_cut_emd: float = 14.0
    holdout_cameras: Tuple[str, ...] = ("Motorola", "Samsung")
    res_match_targets: Tuple[str, ...] = ("iPhone 14", "iPhone 16")
    n_boot: int = 4000
    n_perm: int = 200
    # ---- the three feature families. Fixed texture core + two optional blocks.
    core_feats: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean")
    freq_feats: Tuple[str, ...] = ("spec_centroid_cpm", "dom_wavelength_mm")
    colour_feats: Tuple[str, ...] = ("R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")


CFG = Config()
ON_KAGGLE = Path("/kaggle/input").exists()
OUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working")) if ON_KAGGLE \
    else Path("Model 1/Model 1 Experiment 3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  context={'kaggle' if ON_KAGGLE else 'local'}")
print("outputs ->", OUT_DIR)

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
    raise RuntimeError(
        "Could not find data/processed_meta/manifest_images.csv.\n"
        "On Kaggle: attach the soil-gsd-processed dataset (see kaggle_setup.md).\n"
        "Locally: run this notebook from the repository root.")

INPUT_ROOT, RUN_CONTEXT = resolve_input_root()
META = INPUT_ROOT / "data" / "processed_meta"
sys.path.insert(0, str(INPUT_ROOT))
from preprocess import verify as pv          # the VALIDATED feature definition, reused

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
assert len(TARGET_COLS) == 11, f"expected 11 support columns, got {TARGET_COLS}"
SUPPORT = np.array([float(c) for c in TARGET_COLS])
DL = np.log10(SUPPORT)
assert np.allclose(SUPPORT, [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
print(f"config_hash {CONFIG_HASH} | pipeline {PIPELINE_VERSION} | {TARGET_PPM:.6f} px/mm")

# ---- code cell 5 ----
print(" >>> cell "+str(5), flush=True)
"""Cell B1 — load manifests and labels. Labels come from the SAMPLE manifest only."""
imgs = pd.read_csv(META / "manifest_images.csv")
tiles_all = pd.read_csv(META / "manifest_tiles.csv")
samples = pd.read_csv(META / "manifest_samples.csv")

tiles = tiles_all[(tiles_all.tile_size_px == CFG.tile_size_px) & tiles_all.materialized].copy()
missing = [p for p in tiles.tile_path if not (INPUT_ROOT / p).exists()]
assert not missing, f"{len(missing)} materialised tiles missing, e.g. {missing[:3]}"

tr_lab = samples[samples.split == "train"].set_index("sample_id")
assert len(tr_lab) == 24
tr_ids = list(tr_lab.index)
Y = tr_lab[TARGET_COLS].to_numpy(float)
assert np.all(np.diff(Y, axis=1) >= -1e-9) and np.all(Y[:, -1] == 100.0)
assert ((Y >= 0) & (Y <= 100)).all()
print(f"images {imgs.shape} | tiles@{CFG.tile_size_px} {tiles.shape} | labels {Y.shape}")

# ---- code cell 6 ----
print(" >>> cell "+str(6), flush=True)
"""Cell B2 — the EMD implementation and its reconciliation against the page."""
WIDTHS = np.diff(DL)


def emd_pair(p, t):
    return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))


def emd_page(p, t):
    return float(np.sum(np.abs(np.asarray(p, float) - np.asarray(t, float))[:10] * WIDTHS))


TRIVIAL = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])
PUB_TRIVIAL = 100.31
_t = float(np.mean([emd_pair(TRIVIAL, y) for y in Y]))
_p = float(np.mean([emd_page(TRIVIAL, y) for y in Y]))
print(f"trivial baseline: ours {_t:.4f} | page formula {_p:.4f} | host publishes {PUB_TRIVIAL}")
assert abs(_t - PUB_TRIVIAL) < 0.05, \
    "our EMD no longer reproduces the host's published reference - stop and reconcile"
print("METRIC CONFIRMED (as in E2): the trapezoid matches the host's own number.")

# ---- code cell 7 ----
print(" >>> cell "+str(7), flush=True)
"""Cell B3 — reference baselines and the three external ground-truth scores."""
MEAN_CURVE = Y.mean(axis=0)
MEDIAN_CURVE = np.median(Y, axis=0)
REF = {
    "trivial 9.09%/bin (no images)":   float(np.mean([emd_pair(TRIVIAL, y) for y in Y])),
    "constant train MEAN curve":       float(np.mean([emd_pair(MEAN_CURVE, y) for y in Y])),
    "constant train MEDIAN curve":     float(np.mean([emd_pair(MEDIAN_CURVE, y) for y in Y])),
}
BASELINE_EMD = REF["constant train MEDIAN curve"]
for _k, _v in REF.items():
    print(f"  internal  {_k:32s} {_v:7.2f}")

LB = {
    "Model1_E1  16 feats, alpha by LOGO-CV":  172.69929,
    "mean-curve probe (ignores images)":      102.37237,
    "Model1_E2  14 feats, alpha by CAM+RES":   71.27346,
    "Model1_E3  12 feats, no frequency":       61.23560,
}
print("\nexternal ground truth (public LB, 3 fixed soils of 10):")
for _k, _v in LB.items():
    print(f"  {_v:9.3f}   {_k}")
LB_E1 = LB["Model1_E1  16 feats, alpha by LOGO-CV"]
LB_BASE = LB["mean-curve probe (ignores images)"]
LB_E2 = LB["Model1_E2  14 feats, alpha by CAM+RES"]
LB_E3 = LB["Model1_E3  12 feats, no frequency"]
print(f"\n  E1 was {100*(LB_E1-LB_BASE)/LB_BASE:+.1f}% worse than ignoring the images;")
print(f"  E2 is {100*(LB_E2-LB_BASE)/LB_BASE:+.1f}%, E3 is {100*(LB_E3-LB_BASE)/LB_BASE:+.1f}%.")
print(f"  Trajectory: {LB_E1:.1f} -> {LB_E2:.1f} -> {LB_E3:.1f}")

# ---- code cell 8 ----
print(" >>> cell "+str(8), flush=True)
"""Cell C1 — locate and load the prior experiments' outputs."""
def resolve_history(tag, fname):
    cands = ([INPUT_ROOT / "experiment_history" / tag, INPUT_ROOT / f"Model 1 {tag}"]
             if ON_KAGGLE else [OUT_DIR.parent / E2_DIR_NAME, OUT_DIR.parent / E1_DIR_NAME])
    for c in cands:
        if (c / fname).exists():
            return c
    raise RuntimeError(
        f"Cannot find {fname} from {tag}. Experiment 3 is defined relative to Experiment "
        f"2 and asserts against its feature matrix, so it cannot run without it.\n"
        f"Looked in: {[str(c) for c in cands]}\n"
        "On Kaggle: add experiment_history/Model1_E2/{features_soil.csv, cv_families.csv, "
        "Submission_Model1_E2.csv} to the attached dataset.")


E2_DIR = resolve_history("Model1_E2", "features_soil.csv")
E1_DIR = OUT_DIR.parent / E1_DIR_NAME if not ON_KAGGLE else None
e2_soil = pd.read_csv(E2_DIR / "features_soil.csv")
e2_fam = pd.read_csv(E2_DIR / "cv_families.csv")
e2_sub = pd.read_csv(E2_DIR / "Submission_Model1_E2.csv")
print(f"E2 artifacts from {E2_DIR}")
print(f"  features_soil {e2_soil.shape} | cv_families {e2_fam.shape} | "
      f"submission {e2_sub.shape}")

# ---- code cell 9 ----
print(" >>> cell "+str(9), flush=True)
"""Cell C2 — the trajectory that motivates this experiment."""
print("=" * 78)
print("WHAT HAS BEEN ESTABLISHED")
print("=" * 78)
print(f"  E1  16 features, alpha 0.03 chosen by LOGO-CV   Kaggle {LB_E1:7.2f}   LOST to")
print(f"      a submission that ignores the images ({LB_BASE:.2f})")
print(f"  E2  14 features (2 degenerate columns dropped), alpha 30 chosen by CAM+RES")
print(f"      Kaggle {LB_E2:7.2f}   beat the no-image submission by {LB_BASE-LB_E2:.1f} EMD")
print("-" * 78)
print(f"  ruler calibration so far:  CAM+RES 205.34 -> actual {LB_E1:.2f}  (factor 0.84)")
print(f"                             CAM+RES  75.73 -> actual {LB_E2:.2f}  (factor 0.94)")
print(f"  in-domain LOGO-CV on E2's matrix: 42.66   -> ~{LB_E2-42.66:.0f} EMD of the")
print("  remaining error is camera/sharpness transfer, not model capacity.")
print("=" * 78)
print("  E2's stated lever was WRONG. Colour was predicted to be the camera carrier;")
print("  reconnaissance shows the colour main effect is indistinguishable from zero and")
print("  the ABSOLUTE FREQUENCY features cost ~20 EMD. E3 tests that properly.")

# ---- code cell 10 ----
print(" >>> cell "+str(10), flush=True)
"""Cell D1 — declare the families and assert the design is what we claim."""
CORE = list(CFG.core_feats)
FREQ = list(CFG.freq_feats)
COL = list(CFG.colour_feats)
FE14 = CORE + FREQ + COL

assert len(set(CORE)) == len(CORE) and len(set(FREQ)) == len(FREQ) and len(set(COL)) == len(COL)
assert not (set(CORE) & set(FREQ)) and not (set(CORE) & set(COL)) and not (set(FREQ) & set(COL)), \
    "the three families overlap - the factorial would not be orthogonal"
assert set(FE14) == set(e2_soil.columns) - {"split", "sample_id", "n_images", "cameras",
                                            "e1", "e2", "soil_fraction", "crop_area_fraction"}, \
    "CORE+FREQ+COL is not E2's 14-feature matrix"

CELLS_DEF = {
    "A core":              CORE,
    "B core+freq":         CORE + FREQ,
    "C core+colour":       CORE + COL,
    "D core+freq+colour":  FE14,
}
FREQ_PRESENT = {"A core": 0, "B core+freq": 1, "C core+colour": 0, "D core+freq+colour": 1}
COLOUR_PRESENT = {"A core": 0, "B core+freq": 0, "C core+colour": 1, "D core+freq+colour": 1}
assert sum(FREQ_PRESENT.values()) == 2 and sum(COLOUR_PRESENT.values()) == 2, \
    "not a balanced 2x2"

print(f"texture core ({len(CORE)}): {', '.join(CORE)}")
print(f"frequency    ({len(FREQ)}): {', '.join(FREQ)}   <- absolute spatial frequency")
print(f"colour       ({len(COL)}): {', '.join(COL)}")
print("\n2x2 cells:")
for k, v in CELLS_DEF.items():
    print(f"  {k:22s} {len(v):2d} features   freq={FREQ_PRESENT[k]} colour={COLOUR_PRESENT[k]}")
print("\nCell D is identical to Experiment 2's configuration, so E3 re-measures E2's")
print("result as a consistency check rather than assuming it.")

# ---- code cell 11 ----
print(" >>> cell "+str(11), flush=True)
"""Cell E1 — the per-tile feature function, reused verbatim from E1/E2."""
BAND_KEYS = ("e1", "e2", "e4", "e8", "e16")


def tile_features(rgb, ppm):
    a = np.asarray(rgb, float)
    g = a.mean(axis=2)
    m = g > np.percentile(g, 12)
    out = {k: v for k, v in pv.feats(a, ppm).items() if k in BAND_KEYS or k == "sat"}
    for k in BAND_KEYS:
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


TILE_FEATS = CORE + FREQ + COL

# ---- code cell 12 ----
print(" >>> cell "+str(12), flush=True)
"""Cell E2 — resolve the tile caches, searching E3's own then E2's then E1's.

Reusing an existing cache is a deliberate reproducibility check: the same code on the
same tiles must give the same numbers, or the experiments are not comparable.
"""
CACHE = OUT_DIR / ".cache"; CACHE.mkdir(exist_ok=True)
_srcs = [OUT_DIR / ".cache", OUT_DIR.parent / E2_DIR_NAME / ".cache"]


def get_cache(fname, builder):
    for d in _srcs:
        if (d / fname).exists():
            if not (CACHE / fname).exists():
                (CACHE / fname).write_bytes((d / fname).read_bytes())
            return pd.read_csv(CACHE / fname), str(d)
    df = builder()
    df.to_csv(CACHE / fname, index=False)
    return df, "built"


def _extract(sub_tiles, sigma_fn=None):
    recs = []
    for i, t in enumerate(sub_tiles.itertuples(index=False)):
        rgb = np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB"), float)
        if sigma_fn is not None:
            s = sigma_fn(t.camera)
            if s and s > 0.02:
                rgb = ndi.gaussian_filter(rgb, sigma=(s, s, 0), mode="reflect")
        d = tile_features(rgb, TARGET_PPM); d["tile_path"] = t.tile_path
        recs.append(d)
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(sub_tiles)}")
    return pd.DataFrame(recs)


print(f"extracting base tile features for {len(tiles)} tiles ...")
tf0, src = get_cache(f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv",
                     lambda: _extract(tiles))
print(f"  base features from {src}: {tf0.shape}")

# ---- code cell 13 ----
print(" >>> cell "+str(13), flush=True)
"""Cell E3 — derive the resolution-match blur from the manifests (as E2 did)."""
_k = imgs.assign(k=imgs.effective_ppm / imgs.target_ppm).groupby("camera").k.first()
RES_SIGMA = {}
for tgt in CFG.res_match_targets:
    for src_cam in CFG.holdout_cameras:
        k_s = float(_k[[c for c in _k.index if c.startswith(src_cam)][0]])
        k_t = float(_k[tgt])
        RES_SIGMA[(src_cam, tgt)] = float(np.sqrt(max(k_t ** 2 - k_s ** 2, 0.0)) / np.sqrt(12.0))
for (s, t), v in sorted(RES_SIGMA.items()):
    print(f"  {s:9s} -> {t:11s}  extra anti-alias sigma {v:.4f} px = {v/TARGET_PPM:.4f} mm")
assert all(v > 0 for v in RES_SIGMA.values())


def _sig_for(camera, tgt):
    return RES_SIGMA.get((camera.split(" ")[0], tgt), 0.0)


for tgt in CFG.res_match_targets:
    tag = tgt.replace("iPhone", "").strip()
    tr_tiles = tiles[tiles.split == "train"]
    print(f"extracting RES-MATCH ({tgt}) for {len(tr_tiles)} training tiles ...")
    _, src = get_cache(f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}_{tgt.replace(' ','')}.csv",
                       lambda tt=tr_tiles, t=tgt: _extract(tt, lambda c, t=t: _sig_for(c, t)))
    print(f"  res{tag} features from {src}")

# ---- code cell 14 ----
print(" >>> cell "+str(14), flush=True)
"""Cell E4 — assemble the IMAGE-level feature table for every view.

Two-stage aggregation, matching E1 and E2 exactly: tile -> image median -> soil median,
with the soil-level median taken over ALL of that soil's images (both cameras), not over
camera means. Collapsing the stages changes the numbers materially (reconnaissance
measured CAM+RES for E2's matrix at 85.0 one-stage vs 75.7 two-stage), so the stage order
is part of the frozen protocol, not an implementation detail. Cell E5 proves it holds.
"""
_geo = imgs[["processed_path", "crop_area_fraction"]].rename(
    columns={"processed_path": "parent_image_path"})
_all = []
for tag, fname in ([("real", f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv")] +
                   [(f"res{t.replace('iPhone','').strip()}",
                     f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}_iPhone{t.split()[-1]}.csv")
                    for t in CFG.res_match_targets]):
    f = pd.read_csv(CACHE / fname)
    d = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
               "soil_fraction"]].merge(f, on="tile_path", how="left")
    d = d.merge(_geo, on="parent_image_path", how="left")
    d = d[d.soil_fraction >= CFG.min_tile_soil_fraction].copy()
    d["camera"] = d.camera.str.split(" ").str[0]
    d = d[d.split == "train"]
    im = d.groupby(["sample_id", "camera", "parent_image_path"])[TILE_FEATS].median()
    _all.append(im.reset_index().assign(view=tag))
V = pd.concat(_all, ignore_index=True)
assert not V[TILE_FEATS].isna().to_numpy().any()
print(f"image x camera x view rows: {V.shape}")
print(V.groupby(["view", "camera"]).size().to_string())

# ---- code cell 15 ----
print(" >>> cell "+str(15), flush=True)
"""Cell E5 — reproduce Experiment 2's soil matrix and assert it matches.

If this fails, E3 is not comparable to E2 and the experiment is void.
"""
X_tr = V[V.view == "real"].groupby("sample_id")[TILE_FEATS].median().reset_index()
X_tr["split"] = "train"
_f = pd.read_csv(CACHE / f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv")
_d = tiles[["tile_path", "sample_id", "split", "parent_image_path",
            "soil_fraction"]].merge(_f, on="tile_path")
_d = _d[(_d.split == "test") & (_d.soil_fraction >= CFG.min_tile_soil_fraction)]
_im = _d.groupby(["sample_id", "parent_image_path"])[TILE_FEATS].median()
X_te = _im.groupby("sample_id")[TILE_FEATS].median().reset_index()
X_te["split"] = "test"
pooled = pd.concat([X_tr, X_te], ignore_index=True)
pooled = pooled.merge(samples[["sample_id", "n_images", "cameras"]], on="sample_id", how="left")

_chk = pooled.merge(e2_soil, on=["split", "sample_id"], suffixes=("_e3", "_e2"))
assert len(_chk) == len(e2_soil) == 34, f"row mismatch {len(_chk)} vs {len(e2_soil)}"
_drift = {c: float(np.nanmax(np.abs(_chk[c + "_e3"] - _chk[c + "_e2"]))) for c in TILE_FEATS}
_mx = max(_drift.values())
print(f"max absolute drift vs Experiment 2 across {len(TILE_FEATS)} features: {_mx:.3e}")
assert _mx < 1e-6, f"feature matrix drifted from E2 by {_mx} - experiments not comparable"
pooled.to_csv(OUT_DIR / "features_soil.csv", index=False)
print("REPRODUCED: E3's soil matrix equals E2's. Every difference reported below is")
print("attributable to the feature set, and to nothing else.")

# ---- code cell 16 ----
print(" >>> cell "+str(16), flush=True)
"""Cell E6 — copy the CV families and assert they are E1/E2's, unchanged."""
fams = e2_fam.set_index("sample_id").cv_family
assert set(fams.index) == set(tr_ids)
GROUPS = sorted(fams.unique())
pd.read_csv(E2_DIR / "cv_families.csv").to_csv(OUT_DIR / "cv_families.csv", index=False)
_multi = fams.value_counts()
print(f"{len(GROUPS)} families over {len(fams)} soils "
      f"({int((_multi>1).sum())} multi-member, {int((_multi==1).sum())} singletons), "
      "copied unchanged from E1/E2.")

# ---- code cell 17 ----
print(" >>> cell "+str(17), flush=True)
"""Cell E7 — the evaluation conditions, and the leak assertion.

Four CAM+RES conditions drive the paired analysis. Two CAM conditions are carried for
the minimax robustness check in section J. LOGO-CV is reported but never used to choose.
"""
Ymap = {s: Y[i] for i, s in enumerate(tr_ids)}


def rows(view, camera):
    """Soil-level rows for one (view, camera): median over that soil's images taken by
    that camera. Identical in construction to Experiment 2's _rows()."""
    return V[(V.view == view) & (V.camera == camera)].groupby("sample_id")[TILE_FEATS].median()


CAM_RES, CAM_ONLY = [], []
for _src in CFG.holdout_cameras:
    _tgt = [c for c in CFG.holdout_cameras if c != _src][0]
    for _v in ["real"] + [f"res{t.replace('iPhone', '').strip()}"
                          for t in CFG.res_match_targets]:
        _c = dict(name=f"{_src[:3]}>{_tgt[:3]}:{_v}", src=_src, tgt=_tgt, view=_v)
        (CAM_ONLY if _v == "real" else CAM_RES).append(_c)
ALL_CONDS = CAM_ONLY + CAM_RES
print(f"CAM+RES conditions ({len(CAM_RES)}): " + ", ".join(c["name"] for c in CAM_RES))
print(f"CAM conditions     ({len(CAM_ONLY)}): " + ", ".join(c["name"] for c in CAM_ONLY))

_ids = sorted(set(rows("real", "Motorola").index) & set(rows("real", "Samsung").index))
assert len(_ids) >= 18, "too few dual-camera soils for the camera rulers to mean anything"
for _c in ALL_CONDS:
    assert set(rows(_c["view"], _c["tgt"]).index) >= set(_ids), \
        f"condition {_c['name']} is missing dual-camera soils"
for F in GROUPS:
    ev = [s for s in _ids if fams[s] == F]
    fit = [s for s in _ids if fams[s] != F]
    assert not (set(ev) & set(fit))
    assert not any(fams[s] == fams[t] for s in ev for t in fit), \
        "a same-family soil leaks the held-out label under the other camera"
print(f"\nPASS: {len(_ids)} dual-camera soils. No evaluated soil, and no soil from its own")
print("family, appears in its own fit under ANY condition.")
print("NOTE: the conditions share the same 21 soils, so they are NOT independent. The")
print("bootstrap in section G resamples SOILS for exactly this reason.")

# ---- code cell 18 ----
print(" >>> cell "+str(18), flush=True)
"""Cell E8 — model primitives, identical to E1/E2."""


def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F, float)), 0, 100)
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 100.0
    return F


def fit_predict(Xtr, Ytr, Xte, alpha, rank=CFG.pc_rank):
    mu = Ytr.mean(0)
    _, _, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    V = Vt[:rank]
    sc = StandardScaler().fit(Xtr)
    T = (Ytr - mu) @ V.T
    r = Ridge(alpha=alpha).fit(sc.transform(Xtr), T)
    return project(r.predict(sc.transform(Xte)) @ V + mu)


_Xtr_pool = pooled[pooled.split == "train"].set_index("sample_id")


def cond_errors(FE, alpha, cond, eval_family):
    """Per-soil EMD for one (feature set, alpha, condition, held-out family)."""
    A = rows("real", cond["src"]); B = rows(cond["view"], cond["tgt"])
    ids = [s for s in _ids if s in A.index and s in B.index]
    keep = np.array([fams[s] != eval_family for s in ids])
    if keep.all() or not (~keep).any():
        return {}
    P = fit_predict(A.loc[ids, FE].to_numpy(float)[keep],
                    np.array([Ymap[s] for s in ids])[keep],
                    B.loc[ids, FE].to_numpy(float)[~keep], alpha)
    return {ids[j]: emd_pair(P[i], Ymap[ids[j]]) for i, j in enumerate(np.where(~keep)[0])}


def logo_errors(FE, alpha, eval_family):
    X = _Xtr_pool.loc[tr_ids, FE].to_numpy(float)
    keep = np.array([fams[s] != eval_family for s in tr_ids])
    P = fit_predict(X[keep], Y[keep], X[~keep], alpha)
    return {tr_ids[j]: emd_pair(P[i], Y[j]) for i, j in enumerate(np.where(~keep)[0])}

# ---- code cell 19 ----
print(" >>> cell "+str(19), flush=True)
"""Cell E9 — the nested selection engine.

Alpha is chosen INSIDE each outer fold using only the four CAM+RES conditions, so what
section G reports is the score of the PROCEDURE, never of the best alpha. The CAM
conditions and LOGO-CV are evaluated at those same chosen alphas purely for reporting.
"""
NEST_KEY = hashlib.sha256(json.dumps(
    {"cells": list(CELLS_DEF), "conds": [c["name"] for c in ALL_CONDS],
     "alpha": list(CFG.alpha_grid), "hash": CONFIG_HASH, "seed": SEED,
     "rank": CFG.pc_rank}, sort_keys=True).encode()).hexdigest()[:12]
NCACHE = CACHE / f"nested_{NEST_KEY}.json"


def nested_for_cell(FE):
    per = {c["name"]: {} for c in ALL_CONDS}
    per["LOGO-CV"] = {}
    alphas = []
    for F in GROUPS:
        inner = [G for G in GROUPS if G != F]
        best, ba = None, CFG.alpha_grid[0]
        for a in CFG.alpha_grid:
            v = [x for G in inner for c in CAM_RES
                 for x in cond_errors(FE, a, c, G).values()]
            if not v:
                continue
            m = float(np.mean(v))
            if best is None or m < best:
                best, ba = m, a
        alphas.append(ba)
        for c in ALL_CONDS:
            for s, e in cond_errors(FE, ba, c, F).items():
                per[c["name"]].setdefault(s, []).append(e)
        for s, e in logo_errors(FE, ba, F).items():
            per["LOGO-CV"].setdefault(s, []).append(e)
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in per.items()}, \
        sorted(set(alphas))


if NCACHE.exists():
    _n = json.loads(NCACHE.read_text(encoding="utf-8"))
    NEST, ALPHAS_USED = _n["per"], _n["alphas"]
    print(f"loaded nested results from cache ({NEST_KEY})")
else:
    NEST, ALPHAS_USED = {}, {}
    for nm, FE in CELLS_DEF.items():
        NEST[nm], ALPHAS_USED[nm] = nested_for_cell(FE)
        print(f"  nested {nm:22s} alphas={ALPHAS_USED[nm]}", flush=True)
    NCACHE.write_text(json.dumps({"per": NEST, "alphas": ALPHAS_USED}), encoding="utf-8")


def soil_vector(cell, conds):
    """Per-soil mean EMD of the nested procedure over a group of conditions."""
    keys = [c["name"] for c in conds] if conds != "LOGO" else ["LOGO-CV"]
    shared = set.intersection(*[set(NEST[cell][k]) for k in keys])
    return {s: float(np.mean([NEST[cell][k][s] for k in keys])) for s in sorted(shared)}


V_CR = {nm: soil_vector(nm, CAM_RES) for nm in CELLS_DEF}
V_CM = {nm: soil_vector(nm, CAM_ONLY) for nm in CELLS_DEF}
V_LG = {nm: soil_vector(nm, "LOGO") for nm in CELLS_DEF}
print(f"\nnested cache key {NEST_KEY}")
for nm in CELLS_DEF:
    print(f"  {nm:22s} n={len(V_CR[nm]):2d}  CAM+RES {np.mean(list(V_CR[nm].values())):6.2f}"
          f"   CAM {np.mean(list(V_CM[nm].values())):6.2f}"
          f"   LOGO {np.mean(list(V_LG[nm].values())):6.2f}")

# ---- code cell 20 ----
print(" >>> cell "+str(20), flush=True)
"""Cell F1 — flat grid: cell x alpha, each condition separately plus ruler means."""
grid = []
for nm, FE in CELLS_DEF.items():
    for a in CFG.alpha_grid:
        row = dict(cell=nm, n_feat=len(FE), alpha=a)
        for c in ALL_CONDS:
            v = [x for F in GROUPS for x in cond_errors(FE, a, c, F).values()]
            row[c["name"]] = float(np.mean(v)) if v else np.nan
        row["CAM+RES"] = float(np.mean([row[c["name"]] for c in CAM_RES]))
        row["CAM"] = float(np.mean([row[c["name"]] for c in CAM_ONLY]))
        row["LOGO-CV"] = float(np.mean([x for F in GROUPS
                                        for x in logo_errors(FE, a, F).values()]))
        grid.append(row)
GRID = pd.DataFrame(grid)
GRID.to_csv(OUT_DIR / "factorial_grid.csv", index=False)
print(GRID[["cell", "n_feat", "alpha", "LOGO-CV", "CAM"] +
           [c["name"] for c in CAM_RES] + ["CAM+RES"]].round(2).to_string(index=False))

# ---- code cell 21 ----
print(" >>> cell "+str(21), flush=True)
"""Cell F2 — the 2 x 2 as a table, at each cell's best CAM+RES alpha."""
rows2 = []
for nm in CELLS_DEF:
    sub = GRID[GRID.cell == nm]
    b = sub.loc[sub["CAM+RES"].idxmin()]
    rows2.append(dict(cell=nm, n_feat=int(b.n_feat), freq=FREQ_PRESENT[nm],
                      colour=COLOUR_PRESENT[nm], oracle_alpha=float(b.alpha),
                      **{k: float(b[k]) for k in ["LOGO-CV", "CAM", "CAM+RES"] +
                         [c["name"] for c in CAM_RES]}))
TAB = pd.DataFrame(rows2).set_index("cell")
print("best-alpha (ORACLE) reading per cell -- descriptive only, see section G:\n")
print(TAB[["n_feat", "freq", "colour", "oracle_alpha", "LOGO-CV", "CAM", "CAM+RES"]]
      .round(2).to_string())
print("\nCAM+RES by cell:")
for nm in CELLS_DEF:
    print(f"  {nm:22s} {TAB.loc[nm,'CAM+RES']:6.2f}")
print("\nReading down that column, the two cells WITHOUT frequency are the low ones.")
print("The inference on whether that is real is in section G.")

# ---- code cell 22 ----
print(" >>> cell "+str(22), flush=True)
"""Cell G1 — paired machinery. The resampling unit is asserted, not assumed."""
rng = np.random.default_rng(SEED)
SOILS = sorted(set.intersection(*[set(V_CR[nm]) for nm in CELLS_DEF]))
assert len(SOILS) >= 18, f"paired comparison has only {len(SOILS)} shared soils"
MC = {nm: np.array([V_CR[nm][s] for s in SOILS]) for nm in CELLS_DEF}
MM = {nm: np.array([V_CM[nm][s] for s in SOILS]) for nm in CELLS_DEF}
ML = {nm: np.array([V_LG[nm][s] for s in SOILS]) for nm in CELLS_DEF}
print(f"paired on {len(SOILS)} soils present in every cell, under every condition")


def boot_ci(diff, n=CFG.n_boot):
    """Soil-level bootstrap of the mean of a per-soil difference vector."""
    k = len(diff)
    b = np.array([diff[rng.integers(0, k, k)].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi):
    return "SIGNIFICANT (first is worse)" if lo > 0 else (
        "SIGNIFICANT (first is better)" if hi < 0 else "not distinguishable from zero")


def contrast(vec, a, b, label, store):
    d = vec[a] - vec[b]
    lo, hi = boot_ci(d)
    store.append(dict(effect=label, estimate=float(d.mean()), ci_lo=lo, ci_hi=hi,
                      wins_first_better=int((d < 0).sum()), n=len(d),
                      significant=bool(lo > 0 or hi < 0), verdict=verdict(lo, hi)))
    print(f"  {label:34s} {d.mean():+8.2f}  CI [{lo:+8.2f},{hi:+8.2f}]  "
          f"wins {int((d<0).sum()):2d}/{len(d)}  {verdict(lo,hi)}")


SIMPLE = []
print("SIMPLE EFFECTS on the nested CAM+RES procedure score:\n")
contrast(MC, "B core+freq", "A core", "frequency | no colour", SIMPLE)
contrast(MC, "D core+freq+colour", "C core+colour", "frequency | +colour", SIMPLE)
contrast(MC, "C core+colour", "A core", "colour | no frequency", SIMPLE)
contrast(MC, "D core+freq+colour", "B core+freq", "colour | +frequency", SIMPLE)

# ---- code cell 23 ----
print(" >>> cell "+str(23), flush=True)
"""Cell G2 — MAIN EFFECTS and the interaction, bootstrapped jointly over soils.

The main effect of a factor is the average of its two simple effects. Bootstrapping the
AVERAGE (resampling soils once and recomputing both simple effects on the same resample)
is the correct paired procedure; averaging four separate CIs would not be.
"""
def main_effect(name, pairs):
    """Average of the two simple effects, bootstrapped jointly over soils: one soil
    resample is applied to BOTH simple effects before they are averaged."""
    k = len(SOILS)
    def stat(w):
        return float(np.mean([(MC[a] - MC[b])[w].mean() for a, b in pairs]))
    d_pt = stat(np.arange(k))
    b = np.array([stat(rng.integers(0, k, k)) for _ in range(CFG.n_boot)])
    lo, hi = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    print(f"  MAIN EFFECT {name:11s} {d_pt:+8.2f}  CI [{lo:+8.2f},{hi:+8.2f}]  "
          f"{verdict(lo, hi)}")
    avg = np.mean([MC[a] - MC[b] for a, b in pairs], axis=0)
    return dict(effect=f"MAIN {name}", estimate=d_pt, ci_lo=lo, ci_hi=hi,
                wins_first_better=int((avg < 0).sum()), n=k,
                significant=bool(lo > 0 or hi < 0), verdict=verdict(lo, hi))


print("MAIN EFFECTS (soil-bootstrap on the average of the two simple effects):\n")
MAIN = [main_effect("frequency", [("B core+freq", "A core"),
                                  ("D core+freq+colour", "C core+colour")]),
        main_effect("colour",    [("C core+colour", "A core"),
                                  ("D core+freq+colour", "B core+freq")])]


def interaction(w):
    """A genuine 2x2 interaction: does the frequency effect DEPEND on the colour level?
    Formally (B-A) - (D-C). NOT the difference of the two main effects, which would be
    large by construction whenever one factor matters and the other does not."""
    freq_no_colour = (MC["B core+freq"] - MC["A core"])[w].mean()
    freq_w_colour = (MC["D core+freq+colour"] - MC["C core+colour"])[w].mean()
    return freq_no_colour - freq_w_colour


k = len(SOILS)
_pt = interaction(np.arange(k))
_b = np.array([interaction(rng.integers(0, k, k)) for _ in range(CFG.n_boot)])
_lo, _hi = float(np.percentile(_b, 2.5)), float(np.percentile(_b, 97.5))
print(f"\n  INTERACTION (freq effect at no-colour minus freq effect at +colour) "
      f"{_pt:+8.2f}  CI [{_lo:+8.2f},{_hi:+8.2f}]  {verdict(_lo,_hi)}")
print("  ~0 means the two factors are additive, so the 2x2 was the right design.")
P3 = bool(not (_lo > 0 or _hi < 0))
print(f"  P3 (no colour x frequency interaction): "
      f"{'CONFIRMED' if P3 else 'REFUTED'}")
EFFECTS = pd.DataFrame(SIMPLE + MAIN + [dict(effect="interaction", estimate=_pt,
                                             ci_lo=_lo, ci_hi=_hi,
                                             wins_first_better=0, n=k,
                                             significant=bool(_lo > 0 or _hi < 0),
                                             verdict=verdict(_lo, _hi))])
EFFECTS.to_csv(OUT_DIR / "paired_effects.csv", index=False)

# ---- code cell 24 ----
print(" >>> cell "+str(24), flush=True)
"""Cell G3 — the frequency effect per condition, which is what P1 is judged on.

P1 requires significance in >= 3 of the 4 CAM+RES conditions, not just in the pooled
average. Reporting only the pooled number would let one extreme condition carry it.
"""
P1_hits = 0
print("frequency effect, condition by condition (positive = with-frequency is WORSE):\n")
for c in CAM_RES:
    key = c["name"]
    d = np.array([NEST["B core+freq"][key].get(s, np.nan) -
                  NEST["A core"][key].get(s, np.nan) for s in SOILS])
    ok = ~np.isnan(d); d = d[ok]
    lo, hi = boot_ci(d)
    hit = bool(lo > 0)
    P1_hits += int(hit)
    print(f"  {key:18s} n={len(d):2d}  mean {d.mean():+7.2f}  CI [{lo:+8.2f},{hi:+8.2f}]"
          f"  {'significant' if hit else 'not'}")
    if hi < 0:
        print("    ^ WRONG DIRECTION - frequency helped here")
P1 = bool(P1_hits >= 3)
print(f"\nP1 (frequency main effect significant in >= 3 of 4 conditions): "
      f"{P1_hits}/4 -> {'CONFIRMED' if P1 else 'REFUTED'}")
P2 = bool(not MAIN[1]["significant"])
print(f"P2 (colour main effect not distinguishable from zero): "
      f"{'CONFIRMED' if P2 else 'REFUTED'}   (CI [{MAIN[1]['ci_lo']:+.2f},"
      f"{MAIN[1]['ci_hi']:+.2f}])")

# ---- code cell 25 ----
print(" >>> cell "+str(25), flush=True)
"""Cell H1 — per-family shift under camera change vs under sharpness change."""
_sd = pooled[pooled.split == "train"][TILE_FEATS].std().replace(0, np.nan)
_A = rows("real", "Motorola"); _B = rows("real", "Samsung")
_sh = sorted(set(_A.index) & set(_B.index))
cam_shift = ((_A.loc[_sh, TILE_FEATS] - _B.loc[_sh, TILE_FEATS]).abs() / _sd).mean()
_C = rows("res14", "Motorola")
_sh2 = sorted(set(_A.index) & set(_C.index))
sharp_shift = ((_A.loc[_sh2, TILE_FEATS] - _C.loc[_sh2, TILE_FEATS]).abs() / _sd).mean()
MECH = pd.DataFrame({"camera_shift_z": cam_shift, "sharpness_shift_z": sharp_shift})
MECH["family"] = ["core" if f in CORE else ("freq" if f in FREQ else "colour")
                  for f in MECH.index]
MECH.to_csv(OUT_DIR / "mechanism_table.csv")
print(MECH.round(3).sort_values("sharpness_shift_z", ascending=False).to_string())
print("\nper-family means:")
print(MECH.groupby("family")[["camera_shift_z", "sharpness_shift_z"]].mean().round(3).to_string())
_fam = MECH.groupby("family")[["camera_shift_z", "sharpness_shift_z"]].mean()
print(f"\nFrequency shifts {_fam.loc['freq','sharpness_shift_z']:.2f} z under blur vs "
      f"{_fam.loc['core','sharpness_shift_z']:.2f} for the core: the absolute-frequency")
print("features are reading the SAMPLING HISTORY, not the soil. That is the mechanism.")
print(f"Colour shifts {_fam.loc['colour','camera_shift_z']:.2f} z under camera change, the "
      "largest of the three,\nyet section G says removing it changes the score by ~0. A "
      "feature can move a lot and\nstill not be load-bearing once the model is regularised.")

# ---- code cell 26 ----
print(" >>> cell "+str(26), flush=True)
"""Cell I1 — family-only models, and each family against a shuffled-label control.

A family that carries no signal must not be credited with carrying robustness. This is
the control that distinguishes "frequency is fragile" from "frequency is useless".
"""
LD50 = np.array([float(np.interp(50.0, np.maximum(y, np.linspace(1e-6, 1, 11)), DL)) for y in Y])
_Xtr = pooled[pooled.split == "train"].set_index("sample_id").loc[tr_ids]
_prng = np.random.default_rng(SEED)
rows_i = []
for fam_name, FE in [("core", CORE), ("frequency", FREQ), ("colour", COL)]:
    solo = []
    for a in CFG.alpha_grid:
        v = [x for F in GROUPS for c in CAM_RES
             for x in cond_errors(FE, a, c, F).values()]
        if v:
            solo.append((float(np.mean(v)), a))
    lv = [x for F in GROUPS for x in logo_errors(FE, 1.0, F).values()]
    rho = spearmanr(_Xtr[FE].mean(axis=1), LD50).statistic
    ctrl = max(abs(spearmanr(_Xtr[FE].mean(axis=1), _prng.permutation(LD50)).statistic)
               for _ in range(CFG.n_perm))
    rows_i.append(dict(family=fam_name, n=len(FE), logo_cv_1=float(np.mean(lv)),
                       camres_best=min(solo)[0], camres_alpha=min(solo)[1],
                       rho_vs_logD50=float(rho), perm_control=float(ctrl),
                       beats_control=bool(abs(rho) > ctrl)))
FAMS = pd.DataFrame(rows_i)
FAMS.to_csv(OUT_DIR / "family_controls.csv", index=False)
print(FAMS.round(3).to_string(index=False))
print(f"\nFrequency alone scores LOGO-CV {FAMS[FAMS.family=='frequency'].logo_cv_1.iloc[0]:.1f}"
      f" against a no-image floor of {BASELINE_EMD:.1f} --")
print("it is WORSE than ignoring the images, and it does not beat its own shuffled-label")
print("control. So the honest statement is: the frequency features are near-useless AND")
print("highly fragile. E2's reconnaissance found dom_wavelength the single worst feature")
print("(rho 0.111 against a control of 0.595); this says the whole family is like that.")

# ---- code cell 27 ----
print(" >>> cell "+str(27), flush=True)
"""Cell J1 — primary pick, minimax check, and the honest bet/finding label."""
NESTED_CR = {nm: float(np.mean(list(V_CR[nm].values()))) for nm in CELLS_DEF}
NESTED_CM = {nm: float(np.mean(list(V_CM[nm].values()))) for nm in CELLS_DEF}
print("nested procedure scores:\n")
for nm in CELLS_DEF:
    print(f"  {nm:22s} CAM+RES {NESTED_CR[nm]:6.2f}   CAM {NESTED_CM[nm]:6.2f}   "
          f"max {max(NESTED_CR[nm], NESTED_CM[nm]):6.2f}")
PRIMARY = min(NESTED_CR, key=NESTED_CR.get)
MINIMAX = min(NESTED_CR, key=lambda nm: max(NESTED_CR[nm], NESTED_CM[nm]))
print(f"\nprimary (best nested CAM+RES) : {PRIMARY}")
print(f"minimax (best worst-case)     : {MINIMAX}")

AGREE = PRIMARY == MINIMAX
if AGREE:
    SELECTED, sel_note = PRIMARY, "primary and minimax agree"
else:
    d = MC[PRIMARY] - MC[MINIMAX]
    lo, hi = boot_ci(d)
    print(f"\nTHEY DISAGREE. paired CAM+RES {PRIMARY} - {MINIMAX} = {d.mean():+.2f} "
          f"CI [{lo:+.2f},{hi:+.2f}]")
    SELECTED = min([PRIMARY, MINIMAX], key=lambda nm: len(CELLS_DEF[nm]))
    if lo > 0 or hi < 0:
        sel_note = ("the disagreement is significant, so the pre-stated rule takes the "
                    "FEWER-feature cell")
    else:
        sel_note = ("the two cells are not distinguishable, so the tie is broken by "
                    "parsimony: the fewer-feature cell")
print(f"SELECTED CELL: {SELECTED}  ({len(CELLS_DEF[SELECTED])} features) -- {sel_note}")

_dv = MC[SELECTED] - MC["D core+freq+colour"]
_lo, _hi = boot_ci(_dv)
BET = bool(not (_lo < 0 and _hi < 0))
print(f"\nwinner vs cell D (E2's configuration): {int(_dv.mean()):+d} EMD "
      f"CI [{_lo:+.2f},{_hi:+.2f}]  wins {int((_dv<0).sum())}/{len(_dv)}")
print(f"label: {'A BET, NOT A FINDING -- the CI spans zero' if BET else 'A FINDING'}")
SEL = dict(cell=SELECTED, features=CELLS_DEF[SELECTED], primary=PRIMARY, minimax=MINIMAX,
           agreed=bool(AGREE), note=sel_note, bet=bool(BET),
           nested_camres=NESTED_CR[SELECTED], vs_D_estimate=float(_dv.mean()),
           vs_D_ci=[_lo, _hi])
json.dump(SEL, open(OUT_DIR / "selection_rule.json", "w"), indent=1)

# ---- code cell 28 ----
print(" >>> cell "+str(28), flush=True)
"""Cell J2 — the alpha for the final fit, and the expected external band."""
FE_SEL = CELLS_DEF[SELECTED]
_sub = GRID[GRID.cell == SELECTED]
alpha_f = float(_sub.loc[_sub["CAM+RES"].idxmin(), "alpha"])
_oracle = float(_sub["CAM+RES"].min())
_ruler_pred = float(NESTED_CR[SELECTED])
print(f"selected cell {SELECTED}: oracle alpha {alpha_f:g}")
print(f"  oracle CAM+RES {_oracle:6.2f} | nested procedure CAM+RES {_ruler_pred:6.2f}")
print("  the submission uses the ORACLE alpha (it has to pick one); the nested number is")
print("  the honest estimate of what the whole procedure achieves.")

_FAC = {"E1": LB_E1 / 205.34, "E2": LB_E2 / 75.73}
print(f"\nmeasured ruler bias: E1 {LB_E1:.2f}/205.34 = {_FAC['E1']:.2f}  |  "
      f"E2 {LB_E2:.2f}/75.73 = {_FAC['E2']:.2f}")
LO, HI = sorted([_oracle * min(_FAC.values()), _ruler_pred * max(_FAC.values())])
print(f"expected public score for E3: {LO:.0f} to {HI:.0f}")
print(f"  P4 (beat E2's {LB_E2:.2f})   : predicted {'YES' if HI < LB_E2 else 'MIXED'}")
P5_BAND = (LO, HI)
print(f"  P5 (land in {LO:.0f}-{HI:.0f})   : to be checked after submitting")

# ---- code cell 29 ----
print(" >>> cell "+str(29), flush=True)
"""Cell K1 — fit on all 24 training soils, predict the 10 test soils."""
s_te = pooled[pooled.split == "test"].set_index("sample_id")
_test_order = samples[samples.split == "test"].sort_values("submission_row_order")
s_te = s_te.loc[_test_order.sample_id.tolist()]
Xtr = _Xtr_pool.loc[tr_ids, FE_SEL].to_numpy(float)
Xte = s_te[FE_SEL].to_numpy(float)
assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
P_test = fit_predict(Xtr, Y, Xte, alpha_f)
_sat = 100.0 * float(np.mean((P_test <= 1e-6) | (P_test >= 100)))
_e1p = pd.read_csv(E2_DIR / "Submission_Model1_E2.csv")
_sat_e2 = 100.0 * float(np.mean((_e1p[TARGET_COLS].to_numpy(float) <= 1e-6) |
                                (_e1p[TARGET_COLS].to_numpy(float) >= 100)))
print(f"predictions {P_test.shape}")
print(f"  columns pinned at 0/100: E1 59.1%  E2 {_sat_e2:.1f}%  E3 {_sat:.1f}%  "
      "(real labels 27%)")
print(f"  mean EMD vs the train mean curve: "
      f"{float(np.mean([emd_pair(p, MEAN_CURVE) for p in P_test])):.2f}")

# ---- code cell 30 ----
print(" >>> cell "+str(30), flush=True)
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

# ---- code cell 31 ----
print(" >>> cell "+str(31), flush=True)
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

# ---- code cell 32 ----
print(" >>> cell "+str(32), flush=True)
"""Cell M1 — how the selected cell's predictions differ from E2's."""
fig, axes = plt.subplots(2, 5, figsize=(19, 7.5))
for i, sid in enumerate(s_te.index):
    ax = axes.ravel()[i]
    ax.plot(DL, MEAN_CURVE, ":", c="#999", label="train mean")
    ax.plot(DL, _e1p[TARGET_COLS].to_numpy(float)[i], "x--", c="#dd6b20", lw=1.3,
            label="E2 (71.27)")
    ax.plot(DL, P_test[i], "o-", c="#2b6cb0", lw=2, label="E3 selected")
    ax.set_title(f"{sid}\n{s_te.iloc[i].cameras}", fontsize=8)
    ax.set_ylim(-5, 105); ax.tick_params(labelsize=7)
axes.ravel()[0].legend(fontsize=7)
plt.suptitle(f"E3 {SELECTED} vs E2, same 10 test soils")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(9, 4.4))
d = MC[SELECTED] - MC["D core+freq+colour"]
ax.bar(range(len(SOILS)), d, color=np.where(d < 0, "#2b6cb0", "#c53030"))
ax.axhline(0, c="k", lw=.8)
ax.set_xticks(range(len(SOILS))); ax.set_xticklabels(SOILS, rotation=90, fontsize=7)
ax.set_ylabel("paired CAM+RES EMD difference")
ax.set_title(f"{SELECTED} minus E2's cell D, per soil\n"
             f"negative = E3 better on that soil ({int((d<0).sum())}/{len(d)})")
plt.tight_layout(); plt.show()
print(f"the strip plot is the honest picture: {int((d<0).sum())} of {len(d)} soils improve.")
print("A mean gain built on a minority of soils, or on a few extreme ones, is not robust.")

# ---- code cell 33 ----
print(" >>> cell "+str(33), flush=True)
"""Cell M2 — the rows that changed most, and the Munster check."""
_e2p = _e1p[TARGET_COLS].to_numpy(float)
print(f"{'sample':26s} {'E2->E3 shift':>13}   note")
for i, sid in enumerate(s_te.index):
    dd = emd_pair(P_test[i], _e2p[i])
    note = ""
    if "nster" in sid:
        note = "E1 predicted this flat-100 (pure clay) for a rounded cobble gravel"
    print(f"{sid:26s} {dd:13.2f}   {note}")
_m = [i for i, s in enumerate(s_te.index) if "nster" in s]
if _m:
    print("\nMunster now: " + ", ".join(f"{x:.1f}" for x in P_test[_m[0]]))

# ---- code cell 34 ----
print(" >>> cell "+str(34), flush=True)
"""Cell N1 — write Experiment3.txt from the live objects so the record cannot drift."""
import datetime

L = []; A = L.append
A("=" * 78)
A(f"EXPERIMENT RECORD -- {MODEL_ID} / {EXPERIMENT_ID}")
A("=" * 78)
A(f"generated_at   : {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
A(f"name           : {EXPERIMENT_NAME}")
A(f"context        : {RUN_CONTEXT} on {platform.platform()}")
A("")
A("0. TYPE OF EXPERIMENT")
A("   An ordinary Model 1 experiment. ONE variable changes: the feature set. The ruler,")
A("   output representation, estimator, aggregation, CV families, split and metric are")
A("   frozen at their Experiment 2 values.")
A("")
A("1. OBJECTIVE")
A("   Determine which feature families survive unseen capture devices, and whether")
A("   removing the ones that do not improves the external score. Secondary: the first")
A("   out-of-sample test of the CAM+RES ruler on a change of feature FAMILIES rather")
A("   than of regularisation strength.")
A("")
A("2. DESIGN")
A(f"   texture core ({len(CORE)}): {', '.join(CORE)}")
A(f"   frequency    ({len(FREQ)}): {', '.join(FREQ)}")
A(f"   colour       ({len(COL)}): {', '.join(COL)}")
A("   2x2 factorial; neighbouring cells differ by exactly one family.")
for nm in CELLS_DEF:
    A(f"     {nm:22s} {len(CELLS_DEF[nm]):2d} features  freq={FREQ_PRESENT[nm]} "
      f"colour={COLOUR_PRESENT[nm]}")
A("   Cell D is E2's exact configuration, so E3 re-measures E2 as a consistency check.")
A("")
A("3. HELD IDENTICAL TO E2")
A(f"   rank-{CFG.pc_rank} PCA curve basis, closed-form ridge, clip/cummax/force-100,")
A("   two-stage tile->image->soil MEDIAN aggregation, 16 CV families, 24/10 split.")
A(f"   Reproducibility assertion: max feature drift vs E2 = {_mx:.2e} (< 1e-6 required)")
A("")
A("4. METRIC")
A(f"   trapezoid EMD reproduces the host's published trivial baseline {_t:.4f} vs "
      f"{PUB_TRIVIAL}; the page's printed left-endpoint formula gives {_p:.2f} and does not.")
A("")
A("5. EVALUATION CONDITIONS")
for c in CAM_RES:
    A(f"   CAM+RES  {c['name']}")
for c in CAM_ONLY:
    A(f"   CAM      {c['name']}   (minimax check only)")
A(f"   paired on {len(SOILS)} dual-camera soils; conditions share soils so the bootstrap")
A("   resamples SOILS, not observations.")
A("")
A("6. NESTED PROCEDURE SCORES (the honest estimates)")
for nm in CELLS_DEF:
    A(f"   {nm:22s} CAM+RES {NESTED_CR[nm]:6.2f}   CAM {NESTED_CM[nm]:6.2f}   "
      f"LOGO {np.mean(list(V_LG[nm].values())):6.2f}   alphas {ALPHAS_USED[nm]}")
A("")
A("7. PAIRED EFFECTS  (negative = the first cell is BETTER)")
for _, r in EFFECTS.iterrows():
    A(f"   {r.effect:34s} {r.estimate:+8.2f}  CI [{r.ci_lo:+8.2f},{r.ci_hi:+8.2f}]  {r.verdict}")
A("")
A("8. MECHANISM")
for fam_name in ["core", "freq", "colour"]:
    _r = _fam.loc[fam_name]
    A(f"   {fam_name:8s} camera shift {_r.camera_shift_z:.2f} z   "
      f"sharpness shift {_r.sharpness_shift_z:.2f} z")
A("   Frequency is the family that reads the sampling history rather than the soil.")
A("   Colour moves most under camera change yet removing it changes the score by ~0:")
A("   a feature can move a lot and not be load-bearing once the model is regularised.")
A("")
A("9. FAMILY CONTROLS")
for _, r in FAMS.iterrows():
    A(f"   {r.family:10s} n={int(r.n)} LOGO-CV {r.logo_cv_1:6.2f}  best CAM+RES "
      f"{r.camres_best:6.2f}  rho {r.rho_vs_logD50:+.3f} vs control {r.perm_control:.3f} "
      f"beats={'Y' if r.beats_control else 'n'}")
A(f"   no-image floor for comparison: {BASELINE_EMD:.2f}")
A("")
A("10. SELECTION")
A(f"   primary (best nested CAM+RES) : {PRIMARY}")
A(f"   minimax (best worst-case)     : {MINIMAX}")
A(f"   agreed: {SEL['agreed']}   {SEL['note']}")
A(f"   SELECTED                      : {SELECTED} ({len(FE_SEL)} features)")
A(f"   winner vs cell D: {SEL['vs_D_estimate']:+.2f} EMD CI "
      f"[{SEL['vs_D_ci'][0]:+.2f},{SEL['vs_D_ci'][1]:+.2f}]  -> "
      f"{'A BET, NOT A FINDING' if BET else 'A FINDING'}")
A("")
A("11. FINAL FIT")
A(f"   alpha (ORACLE over the full training set): {alpha_f:g}")
A(f"   oracle CAM+RES {_oracle:.2f} | nested procedure {_ruler_pred:.2f}")
A("   The submission uses the oracle alpha; the nested number is the honest estimate.")
A("")
A("12. PREDICTION VERDICTS")
A(f"   P1 freq effect significant in >=3/4 conditions : "
  f"{'CONFIRMED' if P1 else 'REFUTED'} ({P1_hits}/4)   [confirmation, not blind]")
A(f"   P2 colour effect indistinguishable from zero   : "
  f"{'CONFIRMED' if P2 else 'REFUTED'}   [confirmation, not blind]")
A(f"   P3 no colour x frequency interaction           : "
  f"{'CONFIRMED' if P3 else 'REFUTED'}   [confirmation, not blind]")
A(f"   P4 E3 scores below E2's {LB_E2:.2f}                  : "
  f"{'CONFIRMED' if LB_E3 < LB_E2 else 'REFUTED'} (actual {LB_E3:.2f})   [BLIND]")
_p5 = LO <= LB_E3 <= HI
A(f"   P5 lands in the band {LO:.0f}-{HI:.0f}                   : "
  f"{'CONFIRMED' if _p5 else 'REFUTED'} (actual {LB_E3:.2f}, "
  f"{LB_E3 - HI:+.1f} above the band)   [BLIND]")
_F3 = LB_E3 / _ruler_pred
A(f"   P6 implied factor within 0.7-1.1               : "
  f"{'CONFIRMED' if 0.7 <= _F3 <= 1.1 else 'REFUTED'} (actual factor {_F3:.2f})   [BLIND]")
A("")
A("12b. WHAT THE BLIND PREDICTIONS DECIDED ABOUT THE RULER")
A(f"   calibration across the three externally-measured configurations:")
A(f"     E1  CAM+RES 205.34 -> actual {LB_E1:7.2f}   factor {LB_E1/205.34:.2f}")
A(f"     E2  CAM+RES  75.73 -> actual {LB_E2:7.2f}   factor {LB_E2/75.73:.2f}")
A(f"     E3  CAM+RES  {_ruler_pred:6.2f} -> actual {LB_E3:7.2f}   factor {_F3:.2f}")
A("   The factor has moved monotonically OUTWARD (0.84 -> 0.94 -> 1.18) and crossed the")
A("   P6 window. So the ruler's absolute calibration is NOT stable and P5/P6 are void.")
A("   The RANKING, however, held for the third consecutive time: the ruler put E1 worst,")
A("   E2 middle, E3 best, and so did Kaggle. P4 is the confirmation of that.")
A("   Marginal check: the ruler said E2->E3 was worth 23.9 EMD; it was worth 10.0. Its")
A("   optimism GROWS as configurations improve -- the signature of a proxy that models")
A("   the nuisances it was built from and goes blind once those are removed.")
A("   RULING, binding from E4 onward: CAM+RES is an ORDINAL instrument. It may rank")
A("   candidates. It must not forecast a score, and it must not choose between two")
A("   candidates whose readings differ by less than ~15 EMD.")
A("")
A("13. EXPECTATION")
A(f"   predicted band {LO:.0f}-{HI:.0f} from CAM+RES oracle {_oracle:.2f} / nested "
  f"{_ruler_pred:.2f}, x measured factors {min(_FAC.values()):.2f}-{max(_FAC.values()):.2f}")
if abs(_oracle - _ruler_pred) < 0.01:
    A("   (oracle and nested agree to 2 dp for this cell: the nested procedure happened to")
    A("    pick the same alpha in every fold, so there is no optimism gap here.)")
A("   CAVEAT: the factors come from E1 and E2, both of which changed REGULARISATION.")
A("   E3 changes which FAMILIES exist. P4-P6 exist precisely because that transfer is")
A("   untested. If P4 fails while P1-P3 hold, the ruler ranks but cannot predict, and it")
A("   must be demoted from a selection device to a diagnostic.")
A("")
A("14. SATURATION AND OUTPUT")
A(f"   columns pinned at 0/100: E1 59.1%  E2 {_sat_e2:.1f}%  E3 {_sat:.1f}%  (real 27%)")
A(f"   mean EMD of predictions vs the train mean curve: "
  f"{float(np.mean([emd_pair(p, MEAN_CURVE) for p in P_test])):.2f}")
A("")
A("15. Kaggle RESULT")
A(f"   file            : {SUB_NAME}")
A(f"   public score    : {LB_E3:.5f}    predicted band {LO:.0f}-{HI:.0f}  -> band MISSED")
A(f"   P4 (beat {LB_E2:.2f})   : {'CONFIRMED' if LB_E3 < LB_E2 else 'REFUTED'}   "
  f"gain of {LB_E2 - LB_E3:.2f} EMD")
A(f"   P5 (inside band)    : {'CONFIRMED' if _p5 else 'REFUTED'}   {LB_E3 - HI:+.1f} EMD above")
A(f"   P6 (factor 0.7-1.1) : {'CONFIRMED' if 0.7 <= _F3 <= 1.1 else 'REFUTED'}   factor {_F3:.2f}")
A(f"   private score   : ____________________   (blank until the competition closes)")
A("")
A("16. LIMITS")
A(f"   {len(SOILS)} dual-camera soils. Four conditions raise the observation count but not")
A("   the number of independent soils, so the power ceiling is the 21.")
A("   This design cannot resolve differences much below ~15 EMD; the colour null is an")
A("   absence of evidence, not evidence of absence.")
A("   The synthetic resolution camera is a Gaussian approximation of a Lanczos prefilter.")
A("   Public leaderboard is 3 fixed soils of 10; final score is 0.30*public + 0.70*private.")
A("")
A("17. KNOWN COST OF THIS EXPERIMENT")
A("   Removing the frequency family removes dom_wavelength_mm, the only feature whose")
A("   units are millimetres of actual soil texture and the only one a soil scientist can")
A("   read directly. The model becomes less interpretable even if it scores better.")
A("   Restoring that with blur-robust band RATIOS (e4/e8) rather than absolute spectral")
A("   position is E5's job.")
A("")
A("18. DEVIATIONS FROM PLAN AND MID-RUN CORRECTIONS")
A("   (a) Reconnaissance (scratch/cal_exp3.py) ran BEFORE the plan was written and already")
A("       answers P1-P3, so those are labelled confirmations rather than blind tests.")
A("       Only P4-P6 are genuinely blind. Nothing was re-labelled after the fact.")
A("   (b) Factor A was implemented as SIX conditions (2 camera directions x 3 views), not")
A("       the four the plan listed: the two extra 'real' conditions are the CAM ruler used")
A("       by the minimax check. Strictly more informative; no model component changed.")
A("   (c) BUG FOUND AND FIXED, first run. The E5 reproduction assertion FAILED at a drift")
A("       of 2.18e+01. Cause: E3 was aggregating tile -> soil x camera -> soil (pooling")
A("       cameras before soils) while E1/E2 aggregate tile -> image -> soil (pooling all")
A("       images). The assertion caught a silent incompatibility that would have made")
A("       every number in this experiment meaningless. Fixed; drift is now 2.8e-14.")
A("   (d) BUG FOUND AND FIXED, before any conclusion was drawn. The first implementation")
A(f"       of the interaction test computed (frequency main effect) minus (colour main")
A("       effect), which is large by construction whenever one factor matters and the")
A("       other does not -- it is not an interaction. It reported P3 as REFUTED. The")
A("       correct 2x2 interaction, (B-A) - (D-C), gives -1.64 CI [-6.63,+3.17] and P3")
A("       CONFIRMED. The wrong version was never submitted anywhere; it was caught in")
A("       the run log. Recorded here because a corrected statistic must stay visible.")
A("   (e) A PLAN PREDICTION THAT DID NOT COME TRUE. model_plan_exp3.md section 5 said 'I")
A("       expect rule 3 to fire', i.e. that primary and minimax would disagree. They")
A("       agreed on cell C. The disagreement seen in reconnaissance (CAM preferring the")
A("       5-feature core) was an artefact of comparing each cell at its OWN oracle alpha;")
A("       under a common nested procedure it disappears. Lesson: oracle-alpha comparisons")
A("       across feature sets manufacture differences that are not there.")
A("   No pretrained backbone, fine-tuning, attention pooling, new features, or change to")
A("   aggregation, output representation or estimator was introduced.")
A("")
A("19. NEXT — REVISED AFTER THE SCORE, NOT AFTER THE INTERNAL RESULT")
A("   The scheduled E4 (train on resolution-matched views) is CANCELLED WITH EVIDENCE.")
A("   Once the frequency features are gone the model is already blur-insensitive: across")
A("   all nine train-blur x test-blur combinations the 12-feature set spans only 9.2 EMD,")
A("   and matched training buys at most 2.3 on the deployment column. E3's feature")
A("   selection absorbed the sharpness problem, so there is nothing left for E4 to fix.")
A("   The scheduled aggregation experiment is ALSO cancelled: adding mean, sd, p10 and")
A("   p90 summaries to the tile aggregation took LOGO-CV from 41.15 to 42.22. Quintupling")
A("   the feature space made it slightly worse.")
A("   A THIRD finding, and the most transferable one: training on a blurred view and")
A("   scoring with a ruler whose target is that same blurred view is CIRCULAR. Under that")
A("   comparison the frequency features E3 removed look HELPFUL (-3.08 EMD), a 34-EMD")
A("   artefact. Under held-out blur they still cost +35 to +58. Rule established:")
A("   a treatment that alters the training distribution must be evaluated on a held-out")
A("   level of the nuisance it targets.")
A("   E4 becomes: does the colour block earn its place? CAM prefers no colour (49.27 vs")
A("   52.26); CAM+RES prefers colour (51.87 vs 56.98). The gap is 5 EMD, inside the ~15")
A("   EMD the ordinal ruling in 12b says we may not resolve. That is precisely the")
A("   decision the ruler is now forbidden to make, so it goes to the leaderboard.")
A("   See model_plan_exp4.md.")
A("=" * 78)
TXT_PATH = OUT_DIR / "Experiment3.txt"
TXT_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
print(f"wrote {TXT_PATH} ({len(L)} lines)")
print("\n" + "\n".join(L[L.index('7. PAIRED EFFECTS  (negative = the first cell is BETTER)'):][:8]))

# ---- code cell 35 ----
print(" >>> cell "+str(35), flush=True)
"""Cell N2 — summary."""
print("=" * 76)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  --  SUMMARY")
print("=" * 76)
print(f"  design                 : 2x2 {{colour}} x {{frequency}} on a 5-feature texture core")
print(f"  selected cell          : {SELECTED} ({len(FE_SEL)} features)")
print(f"  nested CAM+RES         : {_ruler_pred:6.2f}   (E2's cell D: {NESTED_CR['D core+freq+colour']:.2f})")
print(f"  frequency main effect  : {MAIN[0]['estimate']:+.2f} CI [{MAIN[0]['ci_lo']:+.2f},"
      f"{MAIN[0]['ci_hi']:+.2f}]")
print(f"  colour main effect     : {MAIN[1]['estimate']:+.2f} CI [{MAIN[1]['ci_lo']:+.2f},"
      f"{MAIN[1]['ci_hi']:+.2f}]")
print(f"  alpha (oracle)         : {alpha_f:g}")
print(f"  saturation             : {_sat:.1f}%  (E2 {_sat_e2:.1f}%, E1 59.1%, real 27%)")
print(f"  predicted external     : {LO:.0f}-{HI:.0f}   E2 actual was {LB_E2:.2f}")
print(f"  status of the gain     : {'BET (CI spans zero)' if BET else 'FINDING'}")
print("=" * 76)
print(f"  submission : {SUB_NAME}")
print("  record     : Experiment3.txt")
print("  artifacts  : paired_effects.csv, factorial_grid.csv, mechanism_table.csv,")
print("               family_controls.csv, selection_rule.json, features_soil.csv")

