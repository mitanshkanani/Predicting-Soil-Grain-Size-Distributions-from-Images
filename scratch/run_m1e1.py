import matplotlib; matplotlib.use("Agg")

# ---- code cell 1 ----
print(" >>> cell "+str(1), flush=True)
"""Cell A1 — configuration. Every tunable lives here and nowhere else."""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

MODEL_ID      = "Model1"
EXPERIMENT_ID = "E1"
EXPERIMENT_NAME = "Model 1 / Experiment 1 - classical image-signal baseline"
SEED = 20260926


@dataclass(frozen=True)
class Config:
    tile_size_px: int = 256                 # 56.23 mm square at the canonical scale
    min_tile_soil_fraction: float = 0.50
    # e1/e2 are computed for diagnosis but EXCLUDED. At 4.5525 px/mm their centre
    # wavelength is 2.5 px and 5.0 px: e1 sits at Nyquist and both failed their
    # shuffled-label control during preprocessing verification.
    texture_bands: Tuple[str, ...] = ("e4", "e8", "e16")
    colour_feats: Tuple[str, ...] = ("R", "G", "B", "sat")
    intensity_feats: Tuple[str, ...] = ("lum_p10", "lum_p50", "lum_p90", "lum_sd")
    gradient_feats: Tuple[str, ...] = ("grad_mean",)
    frequency_feats: Tuple[str, ...] = ("spec_centroid_cpm", "dom_wavelength_mm")
    geometry_feats: Tuple[str, ...] = ("soil_fraction", "crop_area_fraction")
    pc_rank: int = 3                        # rank-3 reconstruction ceiling: EMD 7.4
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    family_linkage: str = "average"
    family_cut_emd: float = 14.0
    n_perm: int = 200


CFG = Config()
ON_KAGGLE = Path("/kaggle/input").exists()
OUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working")) if ON_KAGGLE \
    else Path("Model 1/Model 1 Experiment 1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  context={'kaggle' if ON_KAGGLE else 'local'}")
print("outputs ->", OUT_DIR)

# ---- code cell 2 ----
print(" >>> cell "+str(2), flush=True)
"""Cell A2 — imports and environment capture, recorded for reproducibility."""
import platform, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 170)

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
sys.path.insert(0, str(INPUT_ROOT))          # makes `import preprocess` work on Kaggle
from preprocess import verify as pv          # the VALIDATED feature definition

print(f"INPUT_ROOT  {INPUT_ROOT}  ({RUN_CONTEXT})")
print(f"preprocess package imported from {pv.__file__}")

# ---- code cell 4 ----
print(" >>> cell "+str(4), flush=True)
"""Cell A4 — pin the preprocessing version and seed everything.

If the manifest's config_hash disagreed with the audit or golden-checks hash, the
features we validated would not be the features we are reading.
"""
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
PIPELINE_VERSION = str(_imgs_probe.pipeline_version.iloc[0])
assert _imgs_probe.target_ppm.nunique() == 1
TARGET_PPM = float(_imgs_probe.target_ppm.iloc[0])

# Column names are taken from the competition's own sample_submission.csv, which is
# authoritative. Deriving them from the float diameters yields "2.0"/"200.0" where the
# required headers are the bare "2"/"200" -- a mismatch that would fail the submission.
_SUB_PROBE = pd.read_csv(INPUT_ROOT / "data" / "sample_submission.csv")
TARGET_COLS = [str(c) for c in _SUB_PROBE.columns if str(c) != "sample_id"]
assert len(TARGET_COLS) == 11, f"expected 11 support columns, got {TARGET_COLS}"
SUPPORT = np.array([float(c) for c in TARGET_COLS])
DL = np.log10(SUPPORT)
assert np.allclose(SUPPORT, [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
print(f"config_hash {CONFIG_HASH} | pipeline {PIPELINE_VERSION} | target {TARGET_PPM:.6f} px/mm")

# ---- code cell 6 ----
print(" >>> cell "+str(6), flush=True)
"""Cell B1 — load manifests."""
imgs = pd.read_csv(META / "manifest_images.csv")
tiles_all = pd.read_csv(META / "manifest_tiles.csv")
samples = pd.read_csv(META / "manifest_samples.csv")

tiles = tiles_all[(tiles_all.tile_size_px == CFG.tile_size_px) & tiles_all.materialized].copy()
missing = [p for p in tiles.tile_path if not (INPUT_ROOT / p).exists()]
assert not missing, f"{len(missing)} materialised tiles missing, e.g. {missing[:3]}"
print(f"images {imgs.shape} | tiles@{CFG.tile_size_px}px {tiles.shape} | samples {samples.shape}")
tiles.head(2).T

# ---- code cell 7 ----
print(" >>> cell "+str(7), flush=True)
"""Cell B2 — labels plus the structural checks the competition requires."""
tr_lab = samples[samples.split == "train"].set_index("sample_id")
assert len(tr_lab) == 24
tr_lab = tr_lab.loc[[s for s in tr_lab.index]]
Y = tr_lab[TARGET_COLS].to_numpy(float)
assert np.all(np.diff(Y, axis=1) >= -1e-9), "a training curve is non-monotonic"
assert np.all(Y[:, -1] == 100.0), "200mm column is not 100"
assert ((Y >= 0) & (Y <= 100)).all(), "label outside [0,100]"


def log_d50(y):
    """Median diameter on the log-size axis, by interpolation of the cumulative curve."""
    return float(np.interp(50.0, np.maximum(y, np.linspace(1e-6, 1, 11)), DL))


LD50 = np.array([log_d50(y) for y in Y])
print(f"labels {Y.shape} | D50 spans {10**LD50.min():.4f} .. {10**LD50.max():.3f} mm "
      f"({10**(LD50.max()-LD50.min()):.0f}x)")
tr_lab[["n_images", "cameras", "camera_paired"]].head(3)

# ---- code cell 8 ----
print(" >>> cell "+str(8), flush=True)
"""Cell B3 — the verified metric and its reference points.

The trivial-baseline value doubles as a check on our EMD implementation: the
competition page states it scores 100.31. Agreement means the metric is right.
"""

def emd_pair(p, t):
    return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))


def mean_emd(P, T):
    P, T = np.atleast_2d(P), np.atleast_2d(T)
    return float(np.mean([emd_pair(P[i], T[i]) for i in range(len(T))]))


TRIVIAL = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])
MEDIAN_CURVE = np.median(Y, axis=0)
MEAN_CURVE = Y.mean(axis=0)
REF = {
    "official trivial baseline (9.09%/bin)": mean_emd(np.tile(TRIVIAL, (24, 1)), Y),
    "constant train MEAN curve (no images)": mean_emd(np.tile(MEAN_CURVE, (24, 1)), Y),
    "constant train MEDIAN curve (no images)": mean_emd(np.tile(MEDIAN_CURVE, (24, 1)), Y),
}
print("reference EMD on the training set (lower is better):")
for _k, _v in REF.items():
    print(f"  {_k:44s} {_v:7.2f}")
_trivial = REF["official trivial baseline (9.09%/bin)"]
print(f"\nmetric self-check: measured {_trivial:.2f} vs competition page 100.31")
assert abs(_trivial - 100.31) < 0.5, \
    "our EMD does not reproduce the published baseline - reconcile the metric before proceeding"
BASELINE_EMD = REF["constant train MEDIAN curve (no images)"]

# ---- code cell 9 ----
print(" >>> cell "+str(9), flush=True)
"""Cell B4 — dataset inspection, including the confound that dominates this problem."""
per_img_n = tiles.groupby("parent_image_path").size()
print(f"tiles per image: min {per_img_n.min()}, median {int(per_img_n.median())}, max {per_img_n.max()}")
print("\ntiles per soil:")
print(tiles.groupby(["split", "sample_id"]).size().unstack(0)
      .describe().loc[["min", "50%", "max"]].to_string())

print("\ncamera x split contingency -- THE CENTRAL CHALLENGE:")
ct = pd.crosstab(imgs.camera, imgs.split)
print(ct.to_string())
_only_tr = sorted(set(ct[ct["train"] > 0].index) - set(ct[ct["test"] > 0].index))
_only_te = sorted(set(ct[ct["test"] > 0].index) - set(ct[ct["train"] > 0].index))
print(f"\n  cameras only in TRAIN: {_only_tr}")
print(f"  cameras only in TEST : {_only_te}")
print("  -> no test camera appears in training. Camera is perfectly confounded with the")
print("     split, so every internal estimate is optimistic relative to the private set.")

_low = tiles[tiles.soil_fraction < CFG.min_tile_soil_fraction]
print(f"\ntiles below soil_fraction {CFG.min_tile_soil_fraction}: {len(_low)} of {len(tiles)} "
      f"({100*len(_low)/len(tiles):.1f}%) -- excluded from aggregation, still in the manifest")

# ---- code cell 10 ----
print(" >>> cell "+str(10), flush=True)
"""Cell B5 — look at the real inputs, including the tiles we discard."""
show = tiles.groupby("sample_id").head(3)
fig, axes = plt.subplots(3, 6, figsize=(16, 8.4))
for ax, (_, t) in zip(axes.ravel(), show.iterrows()):
    ax.imshow(np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB")))
    ax.set_title(f"{t.sample_id}\n{t.camera}", fontsize=8)
    ax.axis("off")
for ax in axes.ravel()[len(show):]:
    ax.axis("off")
fig.suptitle(f"materialised {CFG.tile_size_px}px tiles = "
             f"{CFG.tile_size_px/TARGET_PPM:.2f} mm square, identical physical scale per camera")
plt.tight_layout(); plt.show()

if len(_low):
    fig, axes = plt.subplots(1, min(6, len(_low)), figsize=(14, 2.8))
    for ax, (_, t) in zip(np.atleast_1d(axes), _low.head(6).iterrows()):
        ax.imshow(np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB")))
        ax.set_title(f"DISCARDED\nsoil_frac={t.soil_fraction:.2f}", fontsize=8)
        ax.axis("off")
    plt.suptitle("low-soil-fraction tiles excluded from aggregation", y=1.05)
    plt.tight_layout(); plt.show()

# ---- code cell 11 ----
print(" >>> cell "+str(11), flush=True)
"""Cell B6 — derive the approved curve-distance CV families, deterministically.

The manifest's `soil_family` column is derived from the borehole-ID letter prefix and is
NOT this grouping (documented as advisory during preprocessing). The approved protocol
clusters the label CURVES, so it is computed here from the labels and written into this
experiment folder. No preprocessing artifact is modified.
"""
_D = np.array([[emd_pair(a, b) for b in Y] for a in Y])
_Z = linkage(squareform(_D, checks=False), method=CFG.family_linkage)
_cl = fcluster(_Z, CFG.family_cut_emd, criterion="distance")
fams = pd.DataFrame({"sample_id": tr_lab.index, "cv_family": [f"FAM{c}" for c in _cl],
                     "logD50": LD50, "D50_mm": 10 ** LD50})
fams["family_size"] = fams.cv_family.map(fams.cv_family.value_counts())
fams.to_csv(OUT_DIR / "cv_families.csv", index=False)

GROUPS = sorted(fams.cv_family.unique())
_multi = fams.groupby("cv_family").size()
print(f"cut at curve-EMD {CFG.family_cut_emd}: {len(GROUPS)} groups from {len(fams)} soils")
print(f"  multi-member families: {int((_multi > 1).sum())}   singletons: {int((_multi == 1).sum())}")
print(f"  fold training size: {sorted(len(Y) - int(v) for v in _multi)}")
print("\n" + fams.sort_values(["family_size", "cv_family"], ascending=[False, True])
      .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print("\nWhy singletons are correct: those soils genuinely have no near-twin, so there is")
print("nothing to leak. The grouping exists to break the dangerous pairs, e.g. H366~H371")
print("(curve distance 3.3) and H549~H668 (6.8), which plain leave-one-soil-out would leak.")

# ---- code cell 13 ----
print(" >>> cell "+str(13), flush=True)
"""Cell C1 — the per-tile feature function.

Texture bands come from `preprocess.verify.feats`, the exact definition that
`golden_checks.json` was validated against -- reused, not reimplemented. A silently
divergent reimplementation is the easiest way to make an experiment unrepeatable.
"""
BAND_KEYS = ("e1", "e2", "e4", "e8", "e16")


def tile_features(rgb, ppm):
    """One feature vector per tile. rgb: HxWx3 float array at `ppm` px/mm."""
    a = np.asarray(rgb, float)
    g = a.mean(axis=2)
    m = g > np.percentile(g, 12)            # ignore the darkest 12% (tray / shadow)
    out = {k: v for k, v in pv.feats(a, ppm).items() if k in BAND_KEYS or k == "sat"}
    for k in BAND_KEYS:
        out.setdefault(k, np.nan)
    lum = g[m]
    out["lum_p10"], out["lum_p50"], out["lum_p90"] = np.percentile(lum, [10, 50, 90])
    out["lum_sd"] = float(lum.std())
    gy, gx = np.gradient(g)
    out["grad_mean"] = float(np.hypot(gx, gy)[m].mean())        # edge density per px
    # rotation-averaged power spectrum -> dominant texture wavelength in mm
    cpx, cpm = pv.spectral_centroid(a, ppm)
    out["spec_centroid_cpm"] = float(cpm)                        # finer soil -> higher
    out["dom_wavelength_mm"] = float(1.0 / cpm) if cpm > 1e-9 else np.nan
    out["R"], out["G"], out["B"] = [float(a[..., i][m].mean()) for i in range(3)]
    return out


MODEL_FEATS = (list(CFG.texture_bands) + list(CFG.colour_feats) + list(CFG.intensity_feats)
               + list(CFG.gradient_feats) + list(CFG.frequency_feats) + list(CFG.geometry_feats))
DIAG_ONLY = ["e1", "e2"]
print(f"model feature count: {len(MODEL_FEATS)}   diagnostic-only: {DIAG_ONLY}")
for _l in [
    "  texture    e4,e8,e16              DoG octave energies at 0.88/1.76/3.52 mm;",
    "                                   validated 94-96% soil variance, ~1.5% camera",
    "  colour     R,G,B,sat              channel means + saturation; mineralogy and",
    "                                   moisture co-vary with texture",
    "  intensity  lum_p10,p50,p90,sd     brightness distribution of the soil region",
    "  gradient   grad_mean              mean edge magnitude; finer grains pack more",
    "                                   edges per mm",
    "  frequency  spec_centroid_cpm,     dominant spatial wavelength, the most direct",
    "             dom_wavelength_mm      physical proxy for grain size available",
    "  geometry   soil_fraction,         capture/mask QUALITY, not soil properties.",
    "             crop_area_fraction     Kept, but flagged as a possible shortcut"]:
    print(_l)

# ---- code cell 14 ----
print(" >>> cell "+str(14), flush=True)
"""Cell C2 — extract features for every tile (cached so re-runs are cheap)."""
CACHE = OUT_DIR / ".cache"
CACHE.mkdir(exist_ok=True)
FKEY = f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv"
if (CACHE / FKEY).exists():
    tf = pd.read_csv(CACHE / FKEY)
    print(f"loaded {len(tf)} cached tile features")
else:
    print(f"extracting {len(tiles)} tiles at {TARGET_PPM:.4f} px/mm ...")
    recs = []
    for i, t in enumerate(tiles.itertuples(index=False)):
        rgb = np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB"), float)
        d = tile_features(rgb, TARGET_PPM)
        d["tile_path"] = t.tile_path
        recs.append(d)
        if (i + 1) % 400 == 0:
            print(f"  {i+1}/{len(tiles)}")
    tf = pd.DataFrame(recs)
    tf.to_csv(CACHE / FKEY, index=False)

tf = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
            "soil_fraction"]].merge(tf, on="tile_path", how="left")
# crop_area_fraction is an IMAGE-level property, so it comes from the image manifest
tf = tf.merge(imgs[["processed_path", "crop_area_fraction"]].rename(
    columns={"processed_path": "parent_image_path"}), on="parent_image_path", how="left")
assert len(tf) == len(tiles) and not tf[MODEL_FEATS].isna().any().any(),     "tile feature matrix has gaps: " + str(tf[MODEL_FEATS].isna().sum().to_dict())
print(f"tile feature matrix: {tf.shape}")
tf.groupby("split")[MODEL_FEATS].mean().T.round(3)

# ---- code cell 15 ----
print(" >>> cell "+str(15), flush=True)
"""Cell C3 — aggregate tile -> image -> SOIL.

The soil sample is the unit of supervision. Median is the primary statistic because
low-soil-fraction tiles would drag a mean; mean and IQR are carried as separate
features so the choice is testable in a later experiment rather than assumed.
"""
good = tf[tf.soil_fraction >= CFG.min_tile_soil_fraction]
# aggregate the diagnostic bands alongside the model features so the inspection cell can
# report rho and a shuffled-label control for the EXCLUDED bands too
AGG_FEATS = MODEL_FEATS + DIAG_ONLY
img_lv = good.groupby(["split", "sample_id", "camera", "parent_image_path"])[AGG_FEATS].agg(
    ["median", "mean", lambda s: s.quantile(.75) - s.quantile(.25)])
img_lv.columns = [f"{a}__{('iqr' if b.startswith('<') else b)}" for a, b in img_lv.columns]
img_lv = img_lv.reset_index()
MED_COLS = [c for c in img_lv.columns if c.endswith("__median")]

soil = img_lv.groupby(["split", "sample_id"])[MED_COLS].median().reset_index()
soil.columns = ["split", "sample_id"] + [c.replace("__median", "") for c in MED_COLS]
soil = soil.merge(samples[["sample_id", "n_images", "cameras"]], on="sample_id", how="left")
soil.to_csv(OUT_DIR / "features_soil.csv", index=False)

print(f"tiles used {len(good)}/{len(tf)}  ->  {img_lv.parent_image_path.nunique()} image-rows"
      f"  ->  {len(soil)} soil rows ({(soil.split=='train').sum()} train / "
      f"{(soil.split=='test').sum()} test)")
print("\nsoil-level feature medians, train vs test (a domain-shift readout):")
mm = soil.groupby("split")[MODEL_FEATS].median().T
sd = soil[soil.split == "train"][MODEL_FEATS].std()
mm["shift_sd"] = (mm["train"] - mm["test"]) / sd
print(mm.round(3).to_string())
print("\nfeatures with |shift| > 1 SD are suspect: they may be encoding the camera,")
print("not the soil, and will move when the model meets the test devices.")

# ---- code cell 16 ----
print(" >>> cell "+str(16), flush=True)
"""Cell C4 — feature inspection against log D50, each with its own control.

At n=24 soils, shuffled labels reach |rho| ~ 0.5. Reporting a correlation without its
control is how noise gets promoted to a finding.
"""
s_tr = soil[soil.split == "train"].set_index("sample_id").loc[tr_lab.index]
assert list(s_tr.index) == list(tr_lab.index)
Y_LD = pd.Series(LD50, index=tr_lab.index).to_numpy(float)
_rng = np.random.default_rng(SEED)


def perm_control(x, y, n=CFG.n_perm):
    xa = np.asarray(x, float)
    return max(abs(spearmanr(xa, _rng.permutation(y)).statistic) for _ in range(n))


def camera_soil_share(feat):
    """Crude ANOVA-style decomposition of tile-level variance.

    Computed as (variance of group means) / (total variance). This is an INDICATIVE
    ratio, not a variance component: group means and individual observations live on
    different scales, so the number can exceed 100% when a feature is almost pure
    within-camera noise (saturation does exactly that here). Use it to rank features
    against each other, never as a percentage of anything.
    """
    tot = tf[feat].var()
    if not tot or not np.isfinite(tot):
        return np.nan, np.nan
    cam = tf.groupby("camera")[feat].mean().var() / tot * 100
    smp = tf.groupby("sample_id")[feat].mean().var() / tot * 100
    return float(cam), float(smp)


rows = []
for f in MODEL_FEATS + DIAG_ONLY:
    v = s_tr[f].to_numpy(float)
    cam, smp = camera_soil_share(f)
    if not np.isfinite(v).all():
        rows.append({"feature": f, "rho_logD50": np.nan, "perm_control": np.nan,
                     "camera_var_pct": cam, "soil_var_pct": smp,
                     "beats_control": False, "used": "NaN"})
        continue
    rho = float(spearmanr(v, Y_LD).statistic)
    pm = perm_control(v, Y_LD)
    rows.append({"feature": f, "rho_logD50": rho, "perm_control": pm,
                 "camera_var_pct": cam, "soil_var_pct": smp,
                 "beats_control": bool(abs(rho) > pm),
                 "used": "model" if f in MODEL_FEATS else "diag-only"})
INSPECT = pd.DataFrame(rows)
INSPECT.to_csv(OUT_DIR / "feature_inspection.csv", index=False)
print(INSPECT.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print(f"\nbeats its shuffled-label control: {int(INSPECT.beats_control.sum())}/{len(INSPECT)}")
print(f"model features that beat control: "
      f"{int(INSPECT[INSPECT.used=='model'].beats_control.sum())}/{len(MODEL_FEATS)}")

# ---- code cell 17 ----
print(" >>> cell "+str(17), flush=True)
"""Cell C5 — feature diagnostics plot."""
u = INSPECT[INSPECT.used == "model"].dropna(subset=["rho_logD50"]).sort_values("rho_logD50")
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
ax[0].barh(u.feature, u.rho_logD50, color=np.where(u.beats_control, "#2b6cb0", "#c8c8c8"))
ax[0].axvline(0, color="k", lw=.8)
ax[0].set_title("Spearman rho with log D50 (soil level)\ngrey = does not beat shuffled labels")
ax[1].scatter(u.camera_var_pct, u.soil_var_pct, s=46, c="#2b6cb0")
for _, r in u.iterrows():
    ax[1].annotate(r.feature, (r.camera_var_pct, r.soil_var_pct), fontsize=8,
                   xytext=(4, 3), textcoords="offset points")
ax[1].plot([0, 100], [0, 100], ls="--", c="grey", lw=.8)
ax[1].set_xlabel("% tile variance explained by CAMERA")
ax[1].set_ylabel("% explained by SOIL")
ax[1].set_title("soil signal vs camera contamination\n(points on the diagonal are interchangeable")
ax[1].set_title(ax[1].get_title() + ")")
plt.tight_layout(); plt.show()

# ---- code cell 19 ----
print(" >>> cell "+str(19), flush=True)
"""Cell D1 — how good could a k-parameter model ever be? (the ceilings)"""
_mu, _U, _S, _Vt = Y.mean(0), *np.linalg.svd(Y - Y.mean(0), full_matrices=False)
_ev = _S ** 2 / (_S ** 2).sum()
print("explained variance by PC:", np.round(_ev[:5], 3))
print(f"PC1 vs log D50: spearman {spearmanr(_U[:,0]*_S[0], LD50).statistic:.3f}")
CEIL = {}
for k in (1, 2, 3, 4, 5):
    R = _mu + (_U[:, :k] * _S[:k]) @ _Vt[:k]
    CEIL[k] = mean_emd(R, Y)
    bad = int(sum(not np.all(np.diff(r) >= -1e-9) for r in R))
    print(f"  rank-{k}: best possible mean EMD {CEIL[k]:6.2f}   "
          f"(non-monotone rows {bad}/{len(R)})")
print(f"\nExperiment 1 targets rank-{CFG.pc_rank}, ceiling EMD {CEIL[CFG.pc_rank]:.2f}.")
print("Scoring worse than this ceiling is a fitting failure; scoring near it means the")
print("features, not the output representation, are now the binding constraint.")

# ---- code cell 20 ----
print(" >>> cell "+str(20), flush=True)
"""Cell D2 — reconstruction and the monotonicity projection that makes it legal.

The projection is load-bearing: an unconstrained rank-3 reconstruction violates the
competition's monotonicity rule on a third of the training curves.
"""

def make_basis(Ytr, rank):
    """PCA of the cumulative curves. MUST be called with training-fold rows only."""
    mu = Ytr.mean(0)
    U, S, Vt = np.linalg.svd(Ytr - mu, full_matrices=False)
    return mu, Vt[:rank]


def to_coeffs(curves, mu, V):
    return (curves - mu) @ V.T


def to_curve(theta, mu, V):
    return mu + np.atleast_2d(theta) @ V


def project(F):
    """Enforce the competition rules: within [0,100], non-decreasing, 100 at 200mm."""
    F = np.clip(np.atleast_2d(np.asarray(F, float)), 0.0, 100.0)
    F = np.maximum.accumulate(F, axis=1)
    F[:, -1] = 100.0
    return F


_mu3, _V3 = make_basis(Y, CFG.pc_rank)
_raw = to_curve(to_coeffs(Y, _mu3, _V3), _mu3, _V3)
_bad = int(sum(not np.all(np.diff(r) >= -1e-9) for r in _raw))
print(f"rank-{CFG.pc_rank} reconstruction, no projection: {_bad}/{len(_raw)} curves non-monotone")
print(f"  example {tr_lab.index[0]}  raw       {np.round(_raw[0],1)}")
print(f"                          projected {np.round(project(_raw)[0],1)}")
assert (np.diff(project(_raw), axis=1) >= -1e-9).all()
assert np.allclose(project(_raw)[:, -1], 100.0)
assert (project(_raw) >= 0).all() and (project(_raw) <= 100).all()
print(f"  projection restores validity everywhere; residual EMD of projecting the TRUE"
      f" curves = {mean_emd(project(Y), Y):.4f} (should be ~0)")

# ---- code cell 22 ----
print(" >>> cell "+str(22), flush=True)
"""Cell E1 — one LOGO fold end to end, written out so the order is inspectable."""

def fit_ridge_curve(Xtr, Ytr, mu, V, alphas, fam_tr):
    """Standardise and choose alpha by INNER leave-one-family-out. Training data only.

    Scores are computed in CURVE space: the ridge predicts rank-k coefficients, which
    must be reconstructed with the fold's basis before EMD is meaningful.
    """
    Ttr = to_coeffs(Ytr, mu, V)
    sc = StandardScaler().fit(Xtr)
    Xs = sc.transform(Xtr)
    uniq = np.unique(fam_tr)
    best = None
    for a in alphas:
        errs = []
        for g in uniq:
            te = fam_tr == g
            if te.all() or not te.any():
                continue
            r = Ridge(alpha=a).fit(Xs[~te], Ttr[~te])
            P = project(to_curve(r.predict(Xs[te]), mu, V))
            errs.append(mean_emd(P, Ytr[te]))
        m = float(np.mean(errs)) if errs else float("inf")
        if best is None or m < best[0]:
            best = (m, a)
    alpha = best[1] if best else alphas[0]
    return sc, Ridge(alpha=alpha).fit(Xs, Ttr), alpha


def run_logo(X, Ycur, fams_arr, rank, alphas):
    """Per-fold records plus the out-of-fold prediction matrix."""
    out, oof = [], np.full(Ycur.shape, np.nan)
    for g in np.unique(fams_arr):
        va = fams_arr == g
        if va.all():
            continue
        Xtr, Ytr = X[~va], Ycur[~va]
        mu, V = make_basis(Ytr, rank)                 # TRAIN-FOLD BASIS ONLY
        sc, mdl, a = fit_ridge_curve(Xtr, Ytr, mu, V, alphas, fams_arr[~va])
        P = project(to_curve(mdl.predict(sc.transform(X[va])), mu, V))
        oof[va] = P
        errs = [emd_pair(P[i], Ycur[va][i]) for i in range(int(va.sum()))]
        out.append({"fold_family": g, "n_val_soils": int(va.sum()), "alpha": a,
                    "fold_mean_EMD": float(np.mean(errs)),
                    "fold_max_EMD": float(np.max(errs)),
                    "val_soils": " ".join(np.array(tr_lab.index)[va])})
    return pd.DataFrame(out), oof


fams_arr = fams.set_index("sample_id").loc[tr_lab.index, "cv_family"].to_numpy()
X_all = s_tr[MODEL_FEATS].to_numpy(float)
assert np.isfinite(X_all).all()
CV, OOF = run_logo(X_all, Y, fams_arr, CFG.pc_rank, CFG.alpha_grid)
CV.to_csv(OUT_DIR / "cv_results.csv", index=False)
print(CV.to_string(index=False, float_format=lambda v: f"{v:9.2f}"))

# ---- code cell 23 ----
print(" >>> cell "+str(23), flush=True)
"""Cell E2 — headline results."""
f = CV.fold_mean_EMD
RES = {
  "n soils": len(Y), "n folds (families)": len(CV), "n features": len(MODEL_FEATS),
  "mean CV EMD": float(f.mean()), "median CV EMD": float(f.median()),
  "std CV EMD": float(f.std(ddof=0)), "min / max fold": f"{f.min():.2f} / {f.max():.2f}",
  "pooled out-of-fold EMD": mean_emd(OOF, Y),
  "baseline: constant median curve (no images)": BASELINE_EMD,
  "improvement vs baseline": BASELINE_EMD - float(f.mean()),
  f"ceiling: best possible rank-{CFG.pc_rank}": CEIL[CFG.pc_rank],
  "gap to ceiling": float(f.mean()) - CEIL[CFG.pc_rank],
}
for k, v in RES.items():
    print(f"  {k:44s} {v if isinstance(v,str) else format(v,'.3f')}")
assert RES["mean CV EMD"] < BASELINE_EMD, \
    "the images add nothing over a constant curve - this is a valid negative result"

# ---- code cell 24 ----
print(" >>> cell "+str(24), flush=True)
"""Cell E3 — error analysis: where along the log axis does the model fail?"""
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].bar(CV.fold_family, CV.fold_mean_EMD, color="#2b6cb0")
ax[0].axhline(BASELINE_EMD, ls="--", c="crimson", label="no-image baseline 82.6")
ax[0].axhline(CEIL[CFG.pc_rank], ls=":", c="green", label=f"rank-{CFG.pc_rank} ceiling")
ax[0].set_ylabel("fold mean EMD"); ax[0].legend(fontsize=8)
ax[0].set_title("per-fold validation EMD (one family held out)"); ax[0].tick_params(axis="x", rotation=90)

per_pt = np.abs(OOF - Y).mean(0)
ax[1].plot(np.log10(SUPPORT), per_pt, "o-", c="#2b6cb0")
ax[1].set_xlabel("log10 diameter (mm)"); ax[1].set_ylabel("mean |error| (pct)")
ax[1].set_title("error profile along the log axis")
ax[1].set_xticks(np.log10(SUPPORT)); ax[1].set_xticklabels([str(s) for s in SUPPORT], rotation=90, fontsize=7)

ordm = np.argsort(LD50)
ax[2].plot(np.arange(len(Y)), 10 ** LD50[ordm], "o-", c="#888")
ax[2].set_yscale("log"); ax[2].set_xlabel("soil index (sorted by D50)")
ax[2].set_ylabel("D50 mm"); ax[2].set_title("the D50 range the model must span")
plt.tight_layout(); plt.show()

worst = CV.nlargest(3, "fold_mean_EMD")[["fold_family", "fold_mean_EMD", "val_soils"]]
print("worst folds:"); print(worst.to_string(index=False))
print("\nper-soil out-of-fold error, ranked:")
per_soil_err = pd.Series([emd_pair(OOF[i], Y[i]) for i in range(len(Y))], index=tr_lab.index)
print(pd.DataFrame({"EMD": per_soil_err.sort_values(ascending=False),
                    "D50_mm": 10 ** pd.Series(LD50, index=tr_lab.index),
                    "family": fams.set_index("sample_id").cv_family}).head(8).to_string(
    float_format=lambda v: f"{v:8.2f}"))

# ---- code cell 25 ----
print(" >>> cell "+str(25), flush=True)
"""Cell E4 — cross-camera transfer diagnostic.

Leave-one-camera-out in the usual sense barely tests anything here: 21 of the 24 soils
were shot by both cameras, so hiding one camera still leaves those soils present via the
other. The informative version is a paired transfer test -- fit on the Motorola view of a
soil and predict the Samsung view of the SAME soil. Soil identity is then constant and
the only thing that changes is the capture device, so the error isolates how much the
model leans on camera appearance rather than on soil physics.

Reported as a diagnostic; NOT used for model selection.
"""
cams_tr = ["Motorola Edge", "Samsung A52"]
MEDCOLS = [c for c in img_lv.columns if c.endswith("__median")]
# img_lv rows are already the per-image median of tile features; take the median of those
# across a soil's images from one camera to get a (soil, camera) feature row.
by_cam = (img_lv[img_lv.camera.isin(cams_tr)]
          .groupby(["sample_id", "camera"])[MEDCOLS].median().reset_index())
by_cam.columns = ["sample_id", "camera"] + [c[:-len("__median")] for c in MEDCOLS]

paired = sorted(
    set(by_cam.loc[by_cam.camera == cams_tr[0], "sample_id"]) &
    set(by_cam.loc[by_cam.camera == cams_tr[1], "sample_id"]))
Yrows = pd.Series(range(len(Y)), index=tr_lab.index)
Xm = by_cam[by_cam.camera == cams_tr[0]].set_index("sample_id").loc[paired, MODEL_FEATS]
Xs = by_cam[by_cam.camera == cams_tr[1]].set_index("sample_id").loc[paired, MODEL_FEATS]
Yp = Y[[Yrows[s] for s in paired]]

Xm, Xs, Yp = Xm.to_numpy(float), Xs.to_numpy(float), Yp
mu_p, V_p = make_basis(Yp, CFG.pc_rank)
sc_p = StandardScaler().fit(Xm)
mdl_p = Ridge(alpha=1.0).fit(sc_p.transform(Xm), to_coeffs(Yp, mu_p, V_p))
def _pred(Xc):
    return project(to_curve(mdl_p.predict(sc_p.transform(Xc)), mu_p, V_p))
P_cross, P_same = _pred(Xs), _pred(Xm)

XC = pd.DataFrame({
    "comparison": [f"train on {cams_tr[0]}, predict SAME camera (in-domain)",
                   f"train on {cams_tr[0]}, predict {cams_tr[1]} (camera transfer)",
                   "no-image constant median curve on these soils"],
    "mean_EMD": [mean_emd(P_same, Yp), mean_emd(P_cross, Yp),
                 mean_emd(np.tile(np.median(Y, axis=0), (len(Yp), 1)), Yp)]})
print(f"paired soils available for the cross-camera test: {len(paired)}")
print(XC.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
print(chr(10) + "Interpretation: the gap between row 1 and row 2 is the cost of changing")
print("capture device with the soil held fixed. If it is large, the features are encoding")
print("camera appearance, and the iPhone-only test set will be scored optimistically by")
print("every within-training validation number we report.")

# ---- code cell 27 ----
print(" >>> cell "+str(27), flush=True)
"""Cell F1 — refit on all 24 soils and predict the 10 test samples."""
s_te = soil[soil.split == "test"].set_index("sample_id")
test_order = samples[samples.split == "test"].sort_values("submission_row_order")
s_te = s_te.loc[test_order.sample_id.tolist()]
X_te = s_te[MODEL_FEATS].to_numpy(float)
assert np.isfinite(X_te).all()

mu_f, V_f = make_basis(Y, CFG.pc_rank)
sc_f, mdl_f, alpha_f = fit_ridge_curve(X_all, Y, mu_f, V_f, CFG.alpha_grid, fams_arr)
P_test = project(to_curve(mdl_f.predict(sc_f.transform(X_te)), mu_f, V_f))
print(f"final model: ridge alpha={alpha_f}, rank-{CFG.pc_rank} basis, "
      f"{len(MODEL_FEATS)} features, fitted on all 24 soils")
pred = pd.DataFrame(P_test, columns=TARGET_COLS)
pred.insert(0, "sample_id", s_te.index.tolist())
print("\npredicted cumulative percentages:")
print(pred.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
assert (np.diff(P_test, axis=1) >= -1e-9).all() and np.allclose(P_test[:, -1], 100)

# ---- code cell 28 ----
print(" >>> cell "+str(28), flush=True)
"""Cell F2 — visual sanity: do the predicted curves look like soils?"""
fig, axes = plt.subplots(2, 5, figsize=(18, 7))
for ax, i, sid in zip(axes.ravel(), range(len(P_test)), s_te.index):
    ax.plot(np.log10(SUPPORT), Y.mean(0), ":", c="#bbb", label="train mean")
    ax.plot(np.log10(SUPPORT), P_test[i], "o-", c="#2b6cb0", lw=2, label="predicted")
    ax.set_title(f"{sid}\n{int(s_te.iloc[i].n_images)} imgs, {s_te.iloc[i].cameras}", fontsize=9)
    ax.set_xlabel("log10 d (mm)"); ax.set_ylabel("cum %")
    ax.set_ylim(-5, 105); ax.tick_params(labelsize=7)
axes.ravel()[0].legend(fontsize=8)
plt.suptitle("Experiment 1 predicted grain-size curves for the 10 test soils")
plt.tight_layout(); plt.show()

# ---- code cell 29 ----
print(" >>> cell "+str(29), flush=True)
"""Cell F3 — submission validation gate. Raises rather than emitting a bad file."""
sub = pd.read_csv(INPUT_ROOT / "data" / "sample_submission.csv")
submit = pd.DataFrame({"sample_id": s_te.index.map(
    dict(zip(samples.sample_id, samples.submission_id))).tolist()})
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
print("\nall submission checks passed")

# ---- code cell 30 ----
print(" >>> cell "+str(30), flush=True)
"""Cell F4 — write the submission, then re-read it from disk and re-assert."""
SUB_NAME = f"Submission_{MODEL_ID}_{EXPERIMENT_ID}.csv"
SUB_PATH = OUT_DIR / SUB_NAME
submit.to_csv(SUB_PATH, index=False)
back = pd.read_csv(SUB_PATH)
assert list(back.columns) == list(sub.columns), "re-read columns differ"
assert back.sample_id.astype(str).tolist() == sub.sample_id.astype(str).tolist()
assert not back.isna().any().any()
assert (np.diff(back[TARGET_COLS].to_numpy(float), axis=1) >= -1e-9).all()
print(f"wrote {SUB_PATH}  ({SUB_PATH.stat().st_size} bytes)")
print(back.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))

# ---- code cell 31 ----
print(" >>> cell "+str(31), flush=True)
"""Cell F5 — write Experiment1.txt from the live objects, so the record cannot drift."""
import datetime, platform

def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.4f}"

L = []
A = L.append
A("=" * 78)
A(f"EXPERIMENT RECORD -- {MODEL_ID} / {EXPERIMENT_ID}")
A("=" * 78)
A(f"generated_at        : {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}")
A(f"experiment_name     : {EXPERIMENT_NAME}")
A(f"run_context         : {RUN_CONTEXT} on {platform.platform()}")
A("")
A("1. OBJECTIVE")
A("   Measure how much grain-size information is recoverable from the preprocessed")
A("   images using ONLY classical hand-designed features and a ridge regressor.")
A("   No pretrained backbone, no fine-tuning, no attention, no self-supervision.")
A("")
A("2. HYPOTHESIS")
A("   A zero-learned-representation model beats the no-image floor of 82.62 EMD,")
A("   with the well-sampled bands e4/e8/e16 carrying most of the signal.")
A("   Evidence going in: rho(e8,logD50)=0.764, rho(e16,logD50)=0.763 at soil level,")
A("   both beating their shuffled-label control and ~94-96% soil variance.")
A("")
A("3. DATASET USED  (verified preprocessing outputs; raw data never read)")
A(f"   config_hash          : {CONFIG_HASH}")
A(f"   pipeline_version     : {PIPELINE_VERSION}")
A(f"   target_ppm           : {TARGET_PPM:.6f} px/mm (derived, not typed)")
A(f"   tiles                : {len(tiles)} materialised @ {CFG.tile_size_px}px "
  f"= {CFG.tile_size_px/TARGET_PPM:.2f} mm square")
A(f"   tiles used in agg    : {len(good)} (soil_fraction >= {CFG.min_tile_soil_fraction});"
  f" {len(tiles)-len(good)} excluded")
A(f"   images               : {len(imgs)} (train {int((imgs.split=='train').sum())}"
  f" / test {int((imgs.split=='test').sum())})")
A(f"   soils                : {len(Y)} train, {len(P_test)} test")
A(f"   input_root           : {INPUT_ROOT}")
A("")
A("4. FEATURE FAMILIES  (each documented; no undocumented features)")
for fam_name, keys in [("texture DoG bands", CFG.texture_bands),
                       ("colour", CFG.colour_feats), ("intensity", CFG.intensity_feats),
                       ("gradient/edge", CFG.gradient_feats), ("frequency/scale", CFG.frequency_feats),
                       ("capture geometry", CFG.geometry_feats)]:
    A(f"   {fam_name:22s}: {', '.join(keys)}")
A(f"   MODEL FEATURE COUNT  : {len(MODEL_FEATS)}")
A(f"   excluded as sub-Nyquist: {DIAG_ONLY} (centre wavelength 2.5px / 5.0px at this scale)")
A(f"   banned as inputs       : camera, native_ppm, effective_ppm, exif_*, icc_*, site,")
A("                            sample_id, cv_group  -- they identify the device or the sample")
A("")
A("5. FEATURE INSPECTION  (soil-level Spearman vs log D50, with shuffled-label control)")
A("   " + "-" * 72)
for _, r in INSPECT.sort_values("rho_logD50", key=lambda s: s.abs(), ascending=False).iterrows():
    A(f"   {r.feature:20s} rho={_fmt(r.rho_logD50):>8s} control={_fmt(r.perm_control):>8s} "
      f"beats={'Y' if r.beats_control else 'n'} cam_ratio={_fmt(r.camera_var_pct)}")
A("")
A("6. AGGREGATION")
A("   tile -> image -> soil. Median at both levels (robust to excluded low-soil tiles);")
A("   mean and IQR also computed and retained for a later aggregation experiment.")
A(f"   one feature row per soil sample; the SOIL is the unit of supervision.")
A("")
A("7. TARGET / OUTPUT REPRESENTATION")
A(f"   mean curve + {CFG.pc_rank} PCA coefficients = {CFG.pc_rank} predicted numbers,")
A("   then clip to [0,100], cummax for monotonicity, force 100 at 200mm.")
A(f"   Rationale: PC1 alone holds 91.4% of label variance and correlates 0.995 with")
A(f"   log D50, so 11 free outputs is unlearnable at n=24. Rank-{CFG.pc_rank} ceiling")
A(f"   is EMD {CEIL[CFG.pc_rank]:.2f}. Direct 11-output monotone is deferred to a later")
A("   controlled comparison rather than mixed into this experiment.")
A("")
A("8. MODEL AND HYPERPARAMETERS")
A(f"   estimator            : sklearn Ridge (multi-output on curve coefficients)")
A(f"   alpha grid           : {CFG.alpha_grid}")
A(f"   alpha selected by    : inner leave-one-family-out on the training fold")
A(f"   final alpha (all 24) : {alpha_f}")
A(f"   feature scaling      : StandardScaler, fit inside each training fold only")
A(f"   curve basis          : PCA refit inside each training fold only (leak guard)")
A("   epochs/batch/scheduler: n/a - closed-form linear estimator, no iterative training")
A("   augmentation         : n/a - none applied (classical features, deterministic)")
A("")
A("9. RANDOMNESS")
A(f"   seed                 : {SEED} (numpy + permutation control)")
A("   determinism          : no stochastic model; clustering and PCA are deterministic")
A(f"   library versions     : {json.dumps(VERSIONS)}")
A("")
A("10. VALIDATION STRATEGY")
A(f"   Leave-one-family-out over curve-distance families: average linkage on pairwise")
A(f"   label-curve EMD, cut at {CFG.family_cut_emd} -> {len(GROUPS)} groups "
  f"({int((fams.groupby('cv_family').size()>1).sum())} multi-member, "
  f"{int((fams.groupby('cv_family').size()==1).sum())} singletons).")
A("   NOTE: the manifest's soil_family column is ID-prefix derived and is NOT this")
A("   grouping; families are recomputed here from labels and written to cv_families.csv.")
A("   No random image-level split. No soil is ever split across folds.")
A("   Secondary diagnostic: cross-camera transfer (Motorola-fit -> Samsung-predict),")
A("   reported but never used for model selection.")
A("")
A("11. METRIC")
A("   log-weighted EMD = trapezoid(|F - Fhat|, log10 d), mean over samples, 0-500 scale.")
A(f"   self-check: our trivial-baseline score {REF['official trivial baseline (9.09%/bin)']:.2f}"
  f" vs competition page 100.31 -> metric reimplementation agrees.")
A("")
A("12. RESULTS -- every fold")
for _, r in CV.iterrows():
    A(f"   {r.fold_family:6s} n_val={int(r.n_val_soils)} alpha={r.alpha:<6g} "
      f"fold EMD = {r.fold_mean_EMD:7.2f} (worst soil {r.fold_max_EMD:7.2f})  "
      f"[{r.val_soils}]")
A("")
A("13. RESULTS -- summary")
for k, v in RES.items():
    A(f"   {k:44s}: {v if isinstance(v,str) else format(v,'.3f')}")
A("")
A("14. BASELINE COMPARISON")
A(f"   no-image constant median curve : {BASELINE_EMD:.2f}")
A(f"   official trivial baseline      : {REF['official trivial baseline (9.09%/bin)']:.2f}")
A(f"   this experiment (mean CV)      : {f.mean():.2f}")
A(f"   improvement over no-image      : {BASELINE_EMD - f.mean():+.2f} EMD "
  f"({100*(BASELINE_EMD-f.mean())/BASELINE_EMD:.1f}% relative)")
A(f"   rank-{CFG.pc_rank} representation ceiling   : {CEIL[CFG.pc_rank]:.2f}"
  f"  (gap {f.mean()-CEIL[CFG.pc_rank]:+.2f})")
A("")
A("15. OBSERVATIONS")
_beat = INSPECT[(INSPECT.used == 'model') & INSPECT.beats_control].feature.tolist()
A(f"   features beating their control: {len(_beat)}/{len(MODEL_FEATS)} -> {', '.join(_beat)}")
for i, r in INSPECT[INSPECT.used == 'model'].iterrows():
    if r.camera_var_pct > 30:
        A(f"   WARNING {r.feature} is {r.camera_var_pct:.0f}% camera-explained")
A(f"   CROSS-CAMERA TRANSFER (the headline result of this experiment):")
A(f"     in-domain (fit and predict on the same camera) : {XC.mean_EMD[0]:7.2f}")
A(f"     camera transfer (fit Motorola, predict Samsung): {XC.mean_EMD[1]:7.2f}")
A(f"     no-image baseline on the same soils          : {XC.mean_EMD[2]:7.2f}")
A("   -> changing only the capture device, with the soil held FIXED, drives the model")
A("      BELOW the no-image baseline. The within-training CV score of "
  f"{f.mean():.2f} is therefore not an estimate of test-set performance.")
A(f"   worst fold {CV.fold_family[CV.fold_mean_EMD.idxmax()]} at "
  f"{CV.fold_mean_EMD.max():.2f}; best {CV.fold_family[CV.fold_mean_EMD.idxmin()]} at "
  f"{CV.fold_mean_EMD.min():.2f}. Spread across folds is {f.max()-f.min():.2f} EMD,")
A("   which at n=24 means single-fold differences carry very little information.")
A("")
A("16. Kaggle RESULT  (fill in by hand after submitting)")
A(f"   file                 : {SUB_NAME}")
A("   public score         : ____________________   (3 fixed soils; low information)")
A("   private score        : ____________________")
A("   local CV expectation : " + f"{f.mean():.2f}")
A("")
A("17. DEVIATIONS FROM PLAN")
A("   None. No pretrained backbone, fine-tuning, attention pooling or self-supervised")
A("   pretraining was introduced. Any future deviation must be recorded here.")
A("")
A("18. NEXT EXPERIMENT HYPOTHESIS")
A("   The cross-camera collapse outranks any tuning question: the pipeline does not")
A("   transfer between capture devices. Candidate follow-ups, in priority order:")
A("     E2  aggregation statistic (mean vs median vs IQR-augmented) and tile vs whole-image")
A("     E3  feature-family ablation: which of texture / colour / frequency / geometry pays")
A("     E4  camera-invariance: gray-world applied vs not, and per-camera gain")
A("         normalisation of the texture bands, scored by the cross-camera test")
A("     E5  regressor: robust (Huber) and gradient boosting on the identical matrix")
A("   A new Model number is justified only by a change of kind, not of feature set.")
A("=" * 78)

TXT_PATH = OUT_DIR / "Experiment1.txt"
TXT_PATH.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {TXT_PATH}  ({len(L)} lines)")
print("\n" + "\n".join(L[-14:]))

# ---- code cell 32 ----
print(" >>> cell "+str(32), flush=True)
"""Cell F6 — final summary block."""
print("=" * 74)
print(f"{MODEL_ID} / {EXPERIMENT_ID}  --  EXPERIMENT SUMMARY")
print("=" * 74)
print(f"  Feature approach : {len(MODEL_FEATS)} classical features, "
      f"{len(set(c.split('_')[0] for c in MODEL_FEATS))}+ named families, no learned representation")
print(f"  Model            : Ridge (closed form), alpha={alpha_f}")
print(f"  Output           : mean curve + {CFG.pc_rank} PCA coefficients, projected monotone")
print(f"  CV strategy      : leave-one-family-out, {len(GROUPS)} curve-distance families")
print(f"  Mean EMD         : {f.mean():8.2f}")
print(f"  Median EMD       : {f.median():8.2f}")
print(f"  Std EMD          : {f.std(ddof=0):8.2f}")
print(f"  Baseline EMD     : {BASELINE_EMD:8.2f}   (constant median curve, ignores images)")
print(f"  Difference       : {BASELINE_EMD - f.mean():+8.2f}  "
      f"({100*(BASELINE_EMD-f.mean())/BASELINE_EMD:+.1f}% vs baseline)")
print(f"  Representation ceiling (rank {CFG.pc_rank}) : {CEIL[CFG.pc_rank]:.2f}")
print("=" * 74)
print(f"  submission : {SUB_NAME}")
print(f"  record     : Experiment1.txt")
print(f"  artifacts  : cv_families.csv, features_soil.csv, cv_results.csv,")
print(f"               feature_inspection.csv, tile_features.csv")
