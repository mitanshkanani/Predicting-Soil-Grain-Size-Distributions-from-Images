import matplotlib; matplotlib.use("Agg")

# ---- code cell 1 ----
print(" >>> cell 1", flush=True)
"""Cell A1 - configuration. Every tunable lives here and nowhere else.

The numbers below are transcribed from model_spec_m5e1.md section 5 and instructions.txt. They
are the gate, and the gate was fixed before the run: changing one here after a score exists is a
scientific change, not an edit.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

MODEL_ID        = "Model5"
EXPERIMENT_ID   = "E1"
EXPERIMENT_NAME = "Model 5 / Experiment 1 - transductive feature alignment, specificity-gated"
SEED = 20260935
BOOT_SEED = 20260934           # bootstrap draws, every clause, same stream
SHUF_SEED = 20260936           # CORAL_INDEP placebo's column permutation
E3_DIR_NAME = "Model 2 Experiment 3"


@dataclass(frozen=True)
class Config:
    # ---- frozen from Model 1 / Model 2; changing any of these is a different experiment
    tile_size_px: int = 256
    min_tile_soil_fraction: float = 0.50
    pc_rank: int = 3
    control_feats: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean",
                                      "R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                                     300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0,
                                     300000.0, 1000000.0)
    n_boot: int = 4000
    ORDINAL_FLOOR: float = 15.0
    # ---- gate thresholds, fixed in model_spec_m5e1.md section 5 BEFORE the run
    G0_TOL: float = 0.05        # control must reproduce E3's 43.0217308796477
    G1_MIN: float = 3.00        # pooled gain over NONE
    G2_MIN: float = 1.00        # gain in EACH direction
    G3_MIN: float = 0.00        # pooled CI lower bound must be strictly above this
    G4_MIN: float = 1.50        # gain over the same-marginal placebo
    G5_MAX: float = 0.50        # self-alignment canary band
    # ---- pre-registered predictions: magnitudes, not validity bounds
    P2_TOL: float = 1.00        # placebo within this of CORAL
    P3_TOL: float = 1.00        # MEAN carries most of any gain
    P4_MAX: float = 0.10        # predicted self-alignment drift (distinct from G5)
    P5_MAX: float = 0.25        # predicted ceiling on the fraction of the penalty any arm closes
    n_images_test: int = 35     # the alignment target, asserted in D1
    n_rows_eval: int = 45       # 22 soils in direction A + 23 in direction B
    n_clusters: int = 24        # training soils, the bootstrap's resampling unit
    probe_arm = None            # Model 5 pre-authorises no probe


CFG = Config()

ARMS = ("NONE", "CORAL", "MEAN", "CORAL_INDEP", "CORAL_SELF", "CORAL_TILE", "CORAL_SOIL")
GATED = "CORAL"
DIRECTIONS = ("A", "B")
SENSITIVITY = ("CORAL_TILE", "CORAL_SOIL")     # reported; cannot fire the gate
VERDICTS = ("SUPPORTED", "SUPPORTED-BELOW-BAND", "SUPPORTED-WITH-BOUNDARY-CAVEAT",
            "REFUTED", "REFUTED-WITH-BOUNDARY-CAVEAT", "REGULARISATION-ARTEFACT",
            "DIRECTION-INCONSISTENT", "NOT-RESOLVED", "INCONCLUSIVE",
            "INVALID-IMPLEMENTATION")

_EXP_DIR = Path("Model 5 / Model 5 Experiment 1".replace(" / ", "/"))
OUT_DIR = _EXP_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  boot_seed={BOOT_SEED}  shuffle_seed={SHUF_SEED}")
print(f"gated arm: {GATED}   arms: {ARMS}")
print(f"thresholds: G0 {CFG.G0_TOL}  G1 {CFG.G1_MIN}  G2 {CFG.G2_MIN}  G3 {CFG.G3_MIN}"
      f"  G4 {CFG.G4_MIN}  G5 {CFG.G5_MAX}")
print("outputs ->", OUT_DIR)
print("torch/timm: NOT IMPORTED. Model 5 has no GPU path and no mock path.")

# ---- code cell 2 ----
print(" >>> cell 2", flush=True)
"""Cell A2 - imports and environment capture."""
import platform, sys, json, hashlib, time, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)

VERSIONS = {"python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "sklearn": __import__("sklearn").__version__,
            "matplotlib": __import__("matplotlib").__version__,
            "Pillow": __import__("PIL").__version__}
for _k, _v in VERSIONS.items():
    print(f"  {_k:12s} {_v}")

# ---- code cell 3 ----
print(" >>> cell 3", flush=True)
"""Cell A3 - resolve the repository and the two Model 5 modules. LOCAL ONLY.

E3's version of this cell walked the Kaggle input mount to a bounded depth because its level
is not stable. Model 5 runs locally by design, so that walk is dropped rather than kept as a
branch no run ever takes - an untested branch is where Model 2 E2's two Kaggle crashes lived.

backbones.py is not imported: Model 5 embeds nothing.
"""
def _find_dir_with(*markers):
    here = Path.cwd()
    for d in [here, *here.parents]:
        if all((d / m).exists() for m in markers):
            return d
    return None


INPUT_ROOT = _find_dir_with(Path("data") / "processed_meta" / "manifest_images.csv",
                            "Model 2", "Model 5")
if INPUT_ROOT is None:
    raise RuntimeError(
        "Could not find the repository root (data/processed_meta/manifest_images.csv plus the "
        "Model 2 and Model 5 folders). Run this notebook from the repository root.")
RUN_CONTEXT = "local"
META = INPUT_ROOT / "data" / "processed_meta"
E3_DIR = INPUT_ROOT / "Model 2" / "Model 2 Experiment 3"
sys.path.insert(0, str(INPUT_ROOT))
sys.path.insert(0, str(_EXP_DIR))

import alignment as al                     # the transform, and nothing else
import transfer_eval as te                 # image-level assembly, folds, nested head, bootstrap

for _fn in ("covariance", "coral_transform", "apply_transform", "independent_column_shuffle",
            "transform_distance"):
    assert callable(getattr(al, _fn)), f"alignment.{_fn} missing"
for _fn in ("image_features", "soil_from_images", "alignment_target", "direction_folds",
            "fit_predict", "nested_predict", "oracle_sweep", "alpha_status",
            "no_holdout_control_score", "clustered_bootstrap"):
    assert callable(getattr(te, _fn)), f"transfer_eval.{_fn} missing"
assert tuple(te.FEATURES) == tuple(CFG.control_feats), "feature drift between A1 and transfer_eval"
assert tuple(te.ALPHAS) == tuple(CFG.alpha_grid), "alpha grid drift between A1 and transfer_eval"
assert len(ARMS) == 7 and GATED in ARMS and GATED not in SENSITIVITY
assert ARMS == ("NONE", "CORAL", "MEAN", "CORAL_INDEP", "CORAL_SELF", "CORAL_TILE",
                "CORAL_SOIL"), "the arm set is frozen by the approved spec"

print(f"INPUT_ROOT  {INPUT_ROOT}")
print(f"alignment   {al.__file__}")
print(f"transfer    {te.__file__}")
print(f"E3 evidence {E3_DIR}")
print(f"torch in sys.modules: {'torch' in sys.modules}")

# ---- code cell 4 ----
print(" >>> cell 4", flush=True)
"""Cell A4 - pin the preprocessing version, the submission schema, and the seed."""
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
assert CONFIG_HASH == "010f44c36c74", (
    f"config_hash moved to {CONFIG_HASH}. E2 is only comparable to E1 and Model 1 on the "
    "frozen preprocessing - stop and reconcile the dataset version before continuing.")
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
print(" >>> cell 5", flush=True)
"""Cell B1 - load manifests and labels. Labels come from the SAMPLE manifest only."""
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
assert np.all(np.diff(Y, axis=1) >= -1e-9), "a training curve is non-monotonic"
assert np.all(Y[:, -1] == 100.0) and ((Y >= 0) & (Y <= 100)).all()


def log_d50(y):
    return float(np.interp(50.0, np.maximum(y, np.linspace(1e-6, 1, 11)), DL))


LD50 = np.array([log_d50(y) for y in Y])
D50_MM = 10 ** LD50
te_ids = samples[samples.split == "test"].sort_values("submission_row_order").sample_id.tolist()
print(f"images {imgs.shape} | tiles@{CFG.tile_size_px} {tiles.shape} "
      f"(train {int((tiles.split=='train').sum())} / test {int((tiles.split=='test').sum())}) "
      f"| labels {Y.shape}")
print(f"D50 spans {D50_MM.min():.4f} .. {D50_MM.max():.3f} mm")

# ---- code cell 6 ----
print(" >>> cell 6", flush=True)
"""Cell B2 - the EMD implementation and its reconciliation against the page."""
WIDTHS = np.diff(DL)


def emd_pair(p, t):
    return float(np.trapezoid(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))


def emd_page(p, t):
    return float(np.sum(np.abs(np.asarray(p, float) - np.asarray(t, float))[:10] * WIDTHS))


def mean_emd(P, T):
    P, T = np.atleast_2d(P), np.atleast_2d(T)
    return float(np.mean([emd_pair(P[i], T[i]) for i in range(len(T))]))


TRIVIAL = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])
PUB_TRIVIAL = 100.31
_t = float(np.mean([emd_pair(TRIVIAL, y) for y in Y]))
_p_page = float(np.mean([emd_page(TRIVIAL, y) for y in Y]))
print(f"trivial baseline: ours {_t:.4f} | page formula {_p_page:.4f} | host publishes "
      f"{PUB_TRIVIAL}")
assert abs(_t - PUB_TRIVIAL) < 0.05, \
    "our EMD no longer reproduces the host's published reference - stop and reconcile"
print("METRIC CONFIRMED (as in E2, E3, E4 and Model 2 E1).")

# ---- code cell 7 ----
print(" >>> cell 7", flush=True)
"""Cell B3 - reference baselines and every external score this project owns."""
MEAN_CURVE = Y.mean(axis=0)
MEDIAN_CURVE = np.median(Y, axis=0)
FLOOR_NO_IMAGE = mean_emd(np.tile(MEDIAN_CURVE, (24, 1)), Y)
print(f"no-image in-domain floor {FLOOR_NO_IMAGE:.2f} | rank-3 representation ceiling 7.35")

LB = {"M1 E1  16 feats, alpha 0.03 by LOGO-CV": 172.69929,
      "no-image probe (train median curve)":    102.37237,
      "M1 E2  14 feats, alpha 30 by CAM+RES":    71.27346,
      "M1 E3  12 feats, alpha 10 by CAM+RES":    61.23560,
      "M1 E4   5 feats, alpha 30 (probe)":       77.12257,
      "M2 E1 384 feats, DINOv2 (declared probe)": 60.56167}
LB_E3 = LB["M1 E3  12 feats, alpha 10 by CAM+RES"]
LB_M2E1 = LB["M2 E1 384 feats, DINOv2 (declared probe)"]
print("\nexternal ground truth:")
for _k, _v in sorted(LB.items(), key=lambda kv: kv[1]):
    print(f"  {_v:8.2f}  {_k}")
print(f"\n  best so far {LB_M2E1:.2f} (Model 2 E1's probe). The public set is a fixed 3 of the")
print("  10 test soils and the final ranking is 0.30*public + 0.70*private, so a difference")
print("  of a few tenths on three soils is not evidence of a large effect.")

# ---- code cell 8 ----
print(" >>> cell 8", flush=True)
"""Cell C1 - the prior evidence Model 5 is built on, read from disk, not recalled.

G0's reference value is read from Model 2 E3's own sweep_results.csv. If it were typed into this
notebook, a transcription error would become a self-fulfilling check - the failure mode that
makes a sanity gate worthless.
"""
E3_SWEEP = pd.read_csv(E3_DIR / "sweep_results.csv").set_index("arm")
G0_REF = float(E3_SWEEP.loc["M1", "nested_in_domain"])
E3_FUSE = float(E3_SWEEP.loc["fuse@c256", "nested_in_domain"])
E3_SHUF = float(E3_SWEEP.loc["shuf@c256", "nested_in_domain"])
E3_FR = float(E3_SWEEP.loc["fr@c256", "nested_in_domain"])
E2_GATE = json.loads((E3_DIR.parent / "Model 2 Experiment 2" / "kaggle run 1 files" /
                      "gate.json").read_text(encoding="utf-8"))

print(f"Model 2 E3 control (G0 reference): nested in-domain {G0_REF:.13f}")
print(f"Model 2 E3 fusion {E3_FUSE:.4f} vs control {G0_REF:.4f}  -> H3 REFUTED")
print(f"Model 2 E3 permuted-noise arm    {E3_SHUF:.4f}  "
      f"({E3_SHUF - G0_REF:+.2f} EMD against the control)")
print(f"Model 2 E3 random-ViT fused arm  {E3_FR:.4f}  (best in-domain, worst transfer)")
print(f"Model 2 E2 gate verdict: submit={E2_GATE['submit']}  -> magnification REFUTED")
assert abs(E3_FUSE - 43.431353) < 0.05 and abs(E3_SHUF - 52.926537) < 0.05, (
    "E3's published artifacts no longer say what Model 5's premise was built on them to say")
print("\nWHAT MAY BE CARRIED FORWARD FROM MODEL 2: its frozen configuration, its aggregation,")
print("its cached artifacts and the fact that D is far more invariant than the hand features.")
print("WHAT MAY NOT: any belief that magnification helped, or that DINOv2 features help.")
print("Model 5's premise is that invariance without accuracy is still worth something when the")
print("shift itself is the target. That is untested, and P1 predicts it is false.")

# ---- code cell 9 ----
print(" >>> cell 9", flush=True)
"""Cell D1 - run both contract suites as subprocesses, then verify G0 numerically.

A subprocess rather than an import so a failed assertion cannot be half-absorbed into this
kernel's namespace, and so the exit code is the evidence."""
import subprocess

for script in ("check_alignment.py", "check_transfer_eval.py"):
    r = subprocess.run([sys.executable, script], cwd=str(_EXP_DIR), capture_output=True,
                       text=True)
    tail = [ln for ln in (r.stdout or "").splitlines() if "CHECK" in ln and (
        "PASSED" in ln or "FAILED" in ln)]
    print(f"  {script:24s} exit={r.returncode}  {tail[-1] if tail else '(no verdict line)'}")
    assert r.returncode == 0, (f"{script} failed with exit {r.returncode}: "
                               + (r.stdout or "")[-2000:].replace("\n", " | "))

_img = te.image_features(INPUT_ROOT)
assert te.alignment_target(_img).shape == (CFG.n_images_test, len(CFG.control_feats)), (
    "the alignment target is not the 35 unlabeled test images")
_train_soils = set(_img.loc[_img.split == "train", "sample_id"])
_test_soils = set(_img.loc[_img.split == "test", "sample_id"])
assert not (_train_soils & _test_soils), "a training soil contributes target images: circular"
assert len(_test_soils) == 10, f"the target covers {len(_test_soils)} soils, not 10"

G0_GOT = te.no_holdout_control_score(INPUT_ROOT)
G0_GAP = abs(G0_GOT - G0_REF)
G0_OK = bool(G0_GAP <= CFG.G0_TOL)
print(f"\nG0  no-holdout unaligned control   this notebook {G0_GOT:.13f}")
print(f"    reference (E3 artifact)          {G0_REF:.13f}")
print(f"  [{'PASS' if G0_OK else '**FAIL**'}] gap {G0_GAP:.3e} vs tolerance {CFG.G0_TOL}")
assert G0_OK, ("G0 failed: this notebook's head, aggregation or CV is not Model 2 E3's. Fix the "
               "pipeline; do NOT read any Model 5 number as a result.")

_e = np.random.default_rng(0).normal(size=11)
assert abs(te.emd_pair(_e, _e) - emd_pair(_e, _e)) < 1e-12, (
    "transfer_eval and E3 disagree on the metric")
assert te.project(np.array([[0.0, 50.0, 10.0, 40.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0, 3.0]]))[0, 0] == 0.0
print("metric agreement between transfer_eval and E3's B2 cell: exact")
_canary = {"G0_ok": G0_OK, "G0_reference": G0_REF, "G0_measured": G0_GOT, "G0_gap": G0_GAP,
           "G0_tolerance": CFG.G0_TOL, "contract_suites": "both passed",
           "alignment_target_rows": int(CFG.n_images_test), "target_soils": len(_test_soils)}
json.dump(_canary, open(OUT_DIR / "canary.json", "w"), indent=2)
print("wrote canary.json")

# ---- code cell 10 ----
print(" >>> cell 10", flush=True)
"""Cell E1 - base tables: images, soils, families, labels, and the tile-level frame.

The tile-level frame serves only the CORAL_TILE sensitivity arm, which exists to answer whether
the answer depends on the estimation unit. It reads the frozen tile-feature cache; it never
re-extracts a feature.

transfer_eval reassembles the labels from the sample manifest independently of B1. Rather than let
either version silently win, this cell asserts they agree row-for-row and value-for-value, so a
drift in either one raises here instead of quietly changing a score later.
"""
IMG = te.image_features(INPUT_ROOT)
_FAMT = pd.read_csv(E3_DIR / "cv_families.csv")
FAMS = dict(zip(_FAMT.sample_id.astype(str), _FAMT.cv_family))
_TE_IDS, _TE_Y = te.labels_matrix(INPUT_ROOT)
assert _TE_IDS == list(tr_ids), "transfer_eval and B1 disagree on training soil order"
assert np.array_equal(_TE_Y, Y), "transfer_eval and B1 disagree on the label matrix"
TR_IDS, YMAP = list(tr_ids), {s: Y[i] for i, s in enumerate(tr_ids)}
TARGET = te.alignment_target(IMG)
FEATS = list(CFG.control_feats)
TE_IDS = list(te_ids)                       # B1's submission-row order, used by cell H1

_t = pd.read_csv(META / "manifest_tiles.csv")
_t = _t[(_t.tile_size_px == CFG.tile_size_px) & _t.materialized
        & (_t.soil_fraction >= CFG.min_tile_soil_fraction)]
_tf = pd.read_csv(INPUT_ROOT / "Model 1" / "Model 1 Experiment 3" / ".cache" /
                  f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv")
TILE = _t[["tile_path", "parent_image_path", "sample_id", "camera", "split"]].merge(
    _tf.drop_duplicates("tile_path"), on="tile_path", how="inner")
TILE["camera_fam"] = TILE.camera.str.split(" ").str[0]
assert len(TILE) == len(_t), f"{len(_t) - len(TILE)} qualifying tiles lack cached features"
assert set(TILE.split) == {"train", "test"}
assert set(TILE.loc[TILE.split == "test", "sample_id"]).isdisjoint(set(TR_IDS)), (
    "a training soil contributes target tiles: the tile-level arm would be circular")


def _soil_of(img_rows):
    return img_rows.groupby("sample_id")[FEATS].median()


PLACEBO_INDEP, PLACEBO_VECS = al.independent_column_shuffle(TARGET, SHUF_SEED)
pd.DataFrame(PLACEBO_VECS, columns=[f"perm_{j}" for j in range(PLACEBO_VECS.shape[1])]).to_csv(
    OUT_DIR / "placebo_permutation.csv", index=False)
assert not np.array_equal(PLACEBO_INDEP, TARGET), "the placebo is the target it replaced"

print(f"images {IMG.shape} | tiles {TILE.shape} | train soils {len(TR_IDS)} | "
      f"target rows {TARGET.shape}")
print("target camera families:", sorted(set(IMG.loc[IMG.split == 'test', 'camera_fam'])))
print(f"tile-level target rows: {int((TILE.split == 'test').sum())} | "
      f"placebo covariance differs: "
      f"{al.transform_distance(al.coral_transform(TARGET, TARGET), al.coral_transform(TARGET, PLACEBO_INDEP)):.4f}")

# ---- code cell 11 ----
print(" >>> cell 11", flush=True)
"""Cell E2 - build every arm's transform for one fold.

CORAL_TILE estimates its moments at tile level and CORAL_SOIL at soil level; both are then still
APPLIED at image level, because the pipeline that feeds the head is image-level. Estimation unit
and application unit are separate choices and only the first varies across these arms.

Nothing in fold_transforms receives a label, a soil identity or a device name. The target is the
whole unlabeled test-image table, identical for every fold, which is what makes this transductive
rather than leaky: the 10 test soils appear nowhere in the fitting pool (asserted in D1 and E1).
"""
_IDENT = {"A": np.eye(len(FEATS)), "b": np.zeros(len(FEATS))}


def fold_transforms(fit_img):
    """{arm: transform} for one fitting pool. Feature matrices in, linear maps out."""
    src = fit_img[FEATS].to_numpy(float)
    src_tile = TILE[TILE.sample_id.isin(fit_img.sample_id.unique())][FEATS].to_numpy(float)
    tgt_tile = TILE[TILE.split == "test"][FEATS].to_numpy(float)
    src_soil = _soil_of(fit_img).to_numpy(float)
    tgt_soil = _soil_of(IMG[IMG.split == "test"]).to_numpy(float)
    assert len(src_soil) == fit_img.sample_id.nunique() and len(tgt_soil) == 10
    return {
        "NONE":        _IDENT,
        "CORAL":       al.coral_transform(src, TARGET),
        "MEAN":        al.coral_transform(src, TARGET, match_cov=False),
        "CORAL_INDEP": al.coral_transform(src, PLACEBO_INDEP),
        "CORAL_SELF":  al.coral_transform(src, src),
        "CORAL_TILE":  al.coral_transform(src_tile, tgt_tile),
        "CORAL_SOIL":  al.coral_transform(src_soil, tgt_soil),
    }


def fold_matrices(fold, T):
    """Transform each side at image level, then aggregate to soil: the frozen pipeline order."""
    cols = list(range(len(FEATS)))

    def _agg(img_rows):
        a = al.apply_transform(img_rows[FEATS].to_numpy(float), T)
        d = pd.DataFrame(a, columns=cols).assign(sample_id=img_rows.sample_id.to_numpy())
        return d.groupby("sample_id")[cols].median()

    Xfit = _agg(fold["fit_img"])
    Xeval = _agg(fold["eval_img"])
    ids_f = list(fold["fit_ids"])
    assert set(ids_f) <= set(Xfit.index) and set(fold["eval_ids"]) <= set(Xeval.index), (
        "a fold asked for a soil its own images cannot supply")
    return (Xfit.reindex(ids_f).to_numpy(float),
            np.array([YMAP[s] for s in ids_f]),
            [FAMS[s] for s in ids_f],
            Xeval.reindex(list(fold["eval_ids"])).to_numpy(float))


_D = {n: te.direction_folds(n, IMG, samples, FAMS) for n in DIRECTIONS}
for _n in DIRECTIONS:
    _seen = [s for f in _D[_n] for s in f["eval_ids"]]
    assert len(_seen) == len(set(_seen))
    assert set(_seen) == set(IMG[IMG.camera_fam == te.DIRECTION_SPEC[_n][1]].sample_id.unique())
    for f in _D[_n]:
        assert not (set(f["eval_ids"]) & set(f["fit_ids"])), f"leak in {_n}/{f['fold']}"
        assert f["target_rows"] == CFG.n_images_test
_NROWS = sum(len(f["eval_ids"]) for v in _D.values() for f in v)
print("folds: " + ", ".join(f"{k}={len(v)} folds / {sum(len(f['eval_ids']) for f in v)} rows"
                            for k, v in _D.items()))
print("total evaluation rows:", _NROWS)
assert _NROWS == CFG.n_rows_eval, f"{_NROWS} rows, not the contracted 45"

# ---- code cell 12 ----
print(" >>> cell 12", flush=True)
"""Cell E3 - score every arm in every fold of every direction.

For each (direction, arm) this yields one EMD per evaluation soil, plus per fold the nested
alpha, its boundary status, the oracle alpha and the grid spread that spec section 5a requires
to be reported. The oracle column reads the scoring labels and therefore selects nothing; it is
printed beside the nested choice so a reader can see how much the alpha mattered at all.

The prediction curves are written out so every EMD below can be recomputed without re-running a
single fit.
"""
ROWS, PRED_ROWS, FOLDSTATS = [], [], []
_t0 = time.time()
for _dn in DIRECTIONS:
    per_arm = {a: {} for a in ARMS}
    for f in _D[_dn]:
        T = fold_transforms(f["fit_img"])
        Yev = np.array([YMAP[s] for s in f["eval_ids"]])
        for _arm in ARMS:
            Xf, Yf, famf, Xe = fold_matrices(f, T[_arm])
            r = te.nested_predict(Xf, Yf, famf, Xe)
            o = te.oracle_sweep(Xf, Yf, Xe, Yev)
            PRED_ROWS.extend([dict(direction=_dn, arm=_arm, fold=f["fold"], sample_id=s,
                                   **{c: P for c, P in zip(TARGET_COLS, r["pred"][k])})
                              for k, s in enumerate(f["eval_ids"])])
            FOLDSTATS.append(dict(direction=_dn, arm=_arm, fold=f["fold"],
                                  nested_alpha=r["alpha"], status=r["status"],
                                  selection_emd=r["selection_emd"], best_emd=r["best_emd"],
                                  grid_spread=r["grid_spread"], oracle_alpha=o["alpha"],
                                  oracle_status=o["status"], oracle_emd=o["emd"],
                                  oracle_spread=o["spread"], n_fit=len(Yf), n_eval=len(Yev)))
            for k, s in enumerate(f["eval_ids"]):
                per_arm[_arm][s] = emd_pair(r["pred"][k], YMAP[s])
    for _arm in ARMS:
        alphas = [x["nested_alpha"] for x in FOLDSTATS
                  if x["direction"] == _dn and x["arm"] == _arm]
        for s, v in per_arm[_arm].items():
            ROWS.append(dict(direction=_dn, arm=_arm, sample_id=s, emd=v))
        print(f"  {_dn} {_arm:12s} soils={len(per_arm[_arm]):3d} "
              f"mean={np.mean(list(per_arm[_arm].values())):7.3f}"
              f"  alphas " + " ".join(f"{a:g}" for a in alphas)
              + f"  ({time.time() - _t0:.0f}s)", flush=True)

TRANS = pd.DataFrame(ROWS)
FOLDS = pd.DataFrame(FOLDSTATS)
BASE = {d: TRANS[(TRANS.direction == d) & (TRANS.arm == "NONE")].set_index("sample_id").emd
        for d in DIRECTIONS}
for d in DIRECTIONS:
    assert len(BASE[d]) == (22 if d == "A" else 23), f"{d} scored {len(BASE[d])} soils"
assert len(FOLDS) == len(ARMS) * sum(len(v) for v in _D.values())
PRED_DF = pd.DataFrame(PRED_ROWS)
PRED_DF.to_csv(OUT_DIR / "predictions_transfer.csv", index=False)
FOLDS.to_csv(OUT_DIR / "alpha_by_fold.csv", index=False)
print(f"\n{len(TRANS)} arm-soil rows over {len(ARMS)} arms x {len(DIRECTIONS)} directions")
print(f"{len(PRED_DF)} predicted curves in predictions_transfer.csv, "
      f"{len(FOLDS)} fold records in alpha_by_fold.csv")

# ---- code cell 13 ----
print(" >>> cell 13", flush=True)
"""Cell F1 - paired differences and the soil-clustered bootstrap.

Resampling unit is the SOIL IDENTITY. 21 soils appear in both directions and their two rows share
one underlying true curve; resampling rows instead would narrow every interval by roughly sqrt(2)
and that is exactly the error that turns a null into a discovery - and a narrow CI is what decides
whether a submission is allowed.

Every clause below draws from the same 4000 resamples of the same stream (BOOT_SEED), so no
clause gets a resample the others did not.
"""
def arm_rows(arm):
    """Per-soil Delta against NONE, pooled and per direction.

    Delta(s) = err_NONE(s) - err_arm(s), so POSITIVE MEANS THE ARM IS BETTER (spec section 3,
    step 4). The pooled cluster key is the bare soil identity: a soil photographed by both cameras
    contributes two rows that must move together, and prefixing the key with the direction splits
    them and silently degrades the bootstrap to row level.
    """
    rows, clusters, per_dir = [], [], {}
    for d in DIRECTIONS:
        ids = sorted(BASE[d].index)
        got = TRANS[(TRANS.direction == d) & (TRANS.arm == arm)].set_index("sample_id").emd
        diff = np.array([float(BASE[d].loc[s]) - float(got.loc[s]) for s in ids])
        rows.append(diff)
        clusters += list(ids)
        per_dir[d] = (float(diff.mean()), ids, diff)
    return np.concatenate(rows), clusters, per_dir


def _bs(v, c):
    return te.clustered_bootstrap(v, c, CFG.n_boot, BOOT_SEED)


def gain(arm):
    rows, clusters, per_dir = arm_rows(arm)
    pooled = _bs(rows, clusters)
    A = _bs(per_dir["A"][2], per_dir["A"][1])
    B = _bs(per_dir["B"][2], per_dir["B"][1])
    return dict(arm=arm, pooled_gain=pooled[0], pooled_lo=pooled[1], pooled_hi=pooled[2],
                gain_A=A[0], lo_A=A[1], hi_A=A[2], gain_B=B[0], lo_B=B[1], hi_B=B[2],
                n_rows=len(rows), n_clusters=len(set(clusters)))


GAIN = {a: gain(a) for a in ARMS}
GD = pd.DataFrame(GAIN.values())
print("pooled gain over NONE (positive = arm better), with soil-clustered 95% CI:\n")
for r in GD.itertuples(index=False):
    print(f"  {r.arm:12s} pooled {r.pooled_gain:+7.3f} CI [{r.pooled_lo:+7.3f}, "
          f"{r.pooled_hi:+7.3f}]   A {r.gain_A:+7.3f}  B {r.gain_B:+7.3f}")
assert GD.loc[GD.arm == "NONE", "pooled_gain"].abs().max() == 0.0, "NONE must be its own zero"
assert int(GD.n_rows.eq(CFG.n_rows_eval).all()) and int(GD.n_clusters.eq(CFG.n_clusters).all()), (
    "row/cluster counts drifted from the contracted 45 rows over 24 soils")

# Pin the sign convention and the cluster unit against one soil recomputed by hand, so neither can
# invert silently: an inverted Delta turns a degradation into a "gain", and a direction-prefixed
# cluster key turns the soil bootstrap into the row bootstrap it exists to replace.
_ids_A = sorted(BASE["A"].index)
_got_A = TRANS[(TRANS.direction == "A") & (TRANS.arm == GATED)].set_index("sample_id").emd
_by, _cl, _pd = arm_rows(GATED)
assert abs(_pd["A"][2][_ids_A.index(_ids_A[0])]
           - (float(BASE["A"].loc[_ids_A[0]]) - float(_got_A.loc[_ids_A[0]]))) < 1e-12, (
    "Delta must be err_NONE - err_arm, positive when the arm is better (spec section 3 step 4)")
assert len(set(_cl)) == 24 and all(c in set(TR_IDS) for c in _cl), (
    f"the pooled cluster key must be the bare soil identity, got {sorted(set(_cl))[:3]}")
print(f"  Delta convention checked: positive = arm better. 45 rows over {len(set(_cl))} soil "
      f"clusters ({sum(1 for c in set(_cl) if _cl.count(c) == 2)} span both directions)")

# ---- code cell 14 ----
print(" >>> cell 14", flush=True)
"""Cell F2 - paired contrasts of the gated arm against every other arm.

The contrast that decides clause G4 is CORAL - CORAL_INDEP: the same gain recomputed as a
difference of the SAME per-soil rows, so the placebo comparison is paired rather than two
independent intervals being eyeballed for overlap.
"""
def paired_contrast(a, b):
    """Clustered bootstrap of (Delta of arm a) - (Delta of arm b), per soil and per direction."""
    _, clusters, pa = arm_rows(a)
    _, _, pb = arm_rows(b)
    d = np.concatenate([pa["A"][2] - pb["A"][2], pa["B"][2] - pb["B"][2]])
    clu = list(pa["A"][1]) + list(pa["B"][1])
    pooled = _bs(d, clu)
    A = _bs(pa["A"][2] - pb["A"][2], pa["A"][1])
    B = _bs(pa["B"][2] - pb["B"][2], pa["B"][1])
    return dict(arm_a=a, arm_b=b, mean=pooled[0], lo=pooled[1], hi=pooled[2],
                mean_A=A[0], lo_A=A[1], hi_A=A[2], mean_B=B[0], lo_B=B[1], hi_B=B[2])


CONTRASTS = pd.DataFrame([paired_contrast(GATED, a) for a in ARMS if a != GATED])
print("paired contrasts vs CORAL (positive = CORAL better):\n")
for r in CONTRASTS.itertuples(index=False):
    sig = "SIGNIFICANT" if (r.lo > 0 or r.hi < 0) else "not significant"
    print(f"  CORAL - {r.arm_b:12s} {r.mean:+7.3f} CI [{r.lo:+7.3f}, {r.hi:+7.3f}]  {sig}"
          f"   A {r.mean_A:+6.3f}  B {r.mean_B:+6.3f}")
assert set(CONTRASTS.arm_b) == set(ARMS) - {GATED}
assert abs(CONTRASTS.loc[CONTRASTS.arm_b == "NONE", "mean"].iloc[0]
           - GAIN[GATED]["pooled_gain"]) < 1e-12, (
    "the contrast against NONE must reproduce the pooled gain; if it does not, the two are not "
    "the same paired difference")
CONTRASTS.to_csv(OUT_DIR / "placebo_report.csv", index=False)
print("wrote placebo_report.csv")

# ---- code cell 15 ----
print(" >>> cell 15", flush=True)
"""Cell G1 - boundary status per direction, then the gate.

Model 5 selects alpha inside each fold, so a direction has as many alphas as it has folds. The
spec's per-direction status is therefore defined here, before any score: a direction is FLOOR or
CEILING only when a strict majority of its folds picked that same edge, and INTERIOR otherwise.
Every fold's own alpha, status, oracle alpha and grid spread are in alpha_by_fold.csv, written by
cell E3, so the rule can be audited against the raw data rather than taken on trust - and a reader
who disagrees with the majority definition can see exactly what a different one would imply.

Spec section 6 lists ten outcomes but two of its rows can describe the same measurement (a one-
direction boundary plus a failing G2 matches both "DIRECTION-INCONSISTENT" and "REFUTED-WITH-
BOUNDARY-CAVEAT"). Section 6a fixes the order: validity > both-directions inconclusive > full pass
> placebo > direction > below-band > unresolved > refuted. The caveat label is applied to exactly
the two outcomes section 5a attaches it to, a full pass and a plain refutation, because a specific
diagnosis says more than a caveat does. Ordering chooses LABELS only; no threshold moves.

And SUBMIT is true if and only if VERDICT is SUPPORTED (section 5). That identity is asserted below
rather than left to the reader, so a future edit that lets a caveat submit fails the run instead of
writing a file.
"""
def dir_boundary_status(dn, arm):
    st = [te.alpha_status(a) for a in
          FOLDS[(FOLDS.direction == dn) & (FOLDS.arm == arm)].nested_alpha]
    n = len(st)
    if st.count("CEILING") * 2 > n:
        return "CEILING"
    if st.count("FLOOR") * 2 > n:
        return "FLOOR"
    return "INTERIOR"


ARMS_LIST = list(ARMS)
ALPHAS_BY_FOLD = {(d, a): list(FOLDS[(FOLDS.direction == d) & (FOLDS.arm == a)].nested_alpha)
                  for d in DIRECTIONS for a in ARMS_LIST}
STATUS = {(dn, arm): dir_boundary_status(dn, arm) for dn in DIRECTIONS for arm in ARMS_LIST}
ANY_EDGE = {(dn, arm): any(te.alpha_status(a) != "INTERIOR" for a in ALPHAS_BY_FOLD[(dn, arm)])
            for dn in DIRECTIONS for arm in ARMS_LIST}
SYMMETRIC = {d: bool(STATUS[(d, GATED)] == STATUS[(d, "NONE")]
                     and STATUS[(d, GATED)] != "INTERIOR") for d in DIRECTIONS}

for _d in DIRECTIONS:
    for _a in (GATED, "NONE"):
        print(f"  fold alphas {_d}/{_a}: "
              + " ".join(f"{x:g}" for x in ALPHAS_BY_FOLD[(_d, _a)]))
        print(f"    per-fold statuses: "
              + " ".join(te.alpha_status(x) for x in ALPHAS_BY_FOLD[(_d, _a)]))
        print(f"    direction status: {STATUS[(_d, _a)]}"
              f"   (any fold at an edge: {ANY_EDGE[(_d, _a)]})")
print("  symmetric_boundary (diagnostic only, changes no verdict):", SYMMETRIC)

_B = [STATUS[(d, GATED)] for d in DIRECTIONS]
BOUNDARY = [s for s in _B if s != "INTERIOR"]
CAVEAT = len(BOUNDARY) == 1

_G = GAIN[GATED]
_c = CONTRASTS.set_index("arm_b").loc["CORAL_INDEP"]
_s = GAIN["CORAL_SELF"]
_p = GAIN["CORAL_INDEP"]

G0 = bool(G0_OK)
G1 = bool(_G["pooled_gain"] >= CFG.G1_MIN)
G2 = bool(_G["gain_A"] >= CFG.G2_MIN and _G["gain_B"] >= CFG.G2_MIN
          and not (_G["lo_A"] < CFG.G3_MIN and _G["hi_A"] < CFG.G3_MIN)
          and not (_G["lo_B"] < CFG.G3_MIN and _G["hi_B"] < CFG.G3_MIN))
G3 = bool(_G["pooled_lo"] > CFG.G3_MIN)
G4 = bool(_c["mean"] >= CFG.G4_MIN and _c["lo"] > CFG.G3_MIN
          and _p["pooled_gain"] <= _G["pooled_gain"])
G5 = bool(abs(_s["pooled_gain"]) <= CFG.G5_MAX and abs(_s["gain_A"]) <= CFG.G5_MAX
          and abs(_s["gain_B"]) <= CFG.G5_MAX)
G6 = bool(len(BOUNDARY) < 2)
G7 = True   # label-blindness and target disjointness are asserted in A3/D1/E1/E2, which raise
            # rather than flag; reaching this line at all means they held

CLAUSES = []


def clause(text, ok, detail=""):
    CLAUSES.append(dict(clause=text, passed=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}   {detail}")


clause(f"G0 control reproduces E3 within {CFG.G0_TOL} EMD", G0, f"gap {G0_GAP:.2e}")
clause(f"G1 pooled gain over NONE >= {CFG.G1_MIN} EMD", G1, f"{_G['pooled_gain']:+.3f}")
clause(f"G2 gain >= {CFG.G2_MIN} EMD in both directions and neither significantly worse", G2,
       f"A {_G['gain_A']:+.3f} [{_G['lo_A']:+.3f},{_G['hi_A']:+.3f}]  "
       f"B {_G['gain_B']:+.3f} [{_G['lo_B']:+.3f},{_G['hi_B']:+.3f}]")
clause("G3 pooled clustered CI excludes zero", G3,
       f"[{_G['pooled_lo']:+.3f}, {_G['pooled_hi']:+.3f}]")
clause(f"G4 beats the same-marginal placebo by >= {CFG.G4_MIN} EMD with CI excluding zero", G4,
       f"{_c['mean']:+.3f} CI [{_c['lo']:+.3f}, {_c['hi']:+.3f}]")
clause(f"G5 self-alignment canary inside {CFG.G5_MAX} EMD", G5,
       f"pooled {_s['pooled_gain']:+.4f}  A {_s['gain_A']:+.4f}  B {_s['gain_B']:+.4f}")
clause("G6 fewer than two directions at an alpha boundary", G6,
       f"per-direction status {_B}  (never pooled across directions)")
clause("G7 alignment is label-free and the target is disjoint", G7, "asserted in A3/D1/E1/E2")

SUBMIT = bool(G0 and G1 and G2 and G3 and G4 and G5 and G6 and G7)
# Section 5's conjunction alone would let the one-direction boundary case submit, because G6 only
# fails when BOTH directions are clipped. Sections 5a and 7 say the opposite in as many words, so
# the caveat is folded into the one variable that gates the write rather than being a second rule
# someone has to remember to consult. This removes a contradiction; it changes no threshold.
SUBMIT = bool(SUBMIT and not CAVEAT)

if not (G0 and G5 and G7):
    VERDICT = "INVALID-IMPLEMENTATION"
elif len(BOUNDARY) == 2:
    VERDICT = "INCONCLUSIVE"
elif G1 and G2 and G3 and G4:
    VERDICT = "SUPPORTED-WITH-BOUNDARY-CAVEAT" if CAVEAT else "SUPPORTED"
elif G1 and G2 and G3 and not G4:
    VERDICT = "REGULARISATION-ARTEFACT"
elif not G2:
    VERDICT = "DIRECTION-INCONSISTENT"
elif not G1 and G4:
    VERDICT = "SUPPORTED-BELOW-BAND"
elif G1 and not G3:
    VERDICT = "NOT-RESOLVED"
else:
    VERDICT = "REFUTED-WITH-BOUNDARY-CAVEAT" if CAVEAT else "REFUTED"
assert VERDICT in VERDICTS
assert SUBMIT == (VERDICT == "SUPPORTED"), (
    f"spec section 7 admits a submission for exactly one verdict, got {VERDICT} with "
    f"submit={SUBMIT}")
print("  verdict precedence (the spec table leaves one overlap; this is the order fixed "
      "before the run): validity > both-directions-inconclusive > full pass > placebo > "
      "direction > below-band > unresolved > refuted")

_BASE_MEAN = {d: float(np.mean(BASE[d].to_numpy())) for d in DIRECTIONS}
PENALTY = float(np.mean(list(_BASE_MEAN.values()))) - G0_GOT
print(f"\nGATE -> {VERDICT}   submit={SUBMIT}")
print(f"  cross-camera error, unaligned: direction A {_BASE_MEAN['A']:.2f}, "
      f"B {_BASE_MEAN['B']:.2f} EMD")
print(f"  camera penalty measured here (cross-camera minus same-camera in-domain {G0_GOT:.2f}): "
      f"{PENALTY:.2f} EMD")
if VERDICT.startswith("SUPPORTED") and not SUBMIT:
    print("  Passed on magnitude but the grid clipped one direction: no submission (spec 5a).")
if not SUBMIT:
    print("  No submission. This is the predicted outcome (P1) and it is a complete answer.")

GATE = dict(submit=SUBMIT, verdict=VERDICT, gated_arm=GATED, clauses=CLAUSES,
            gains={a: GAIN[a] for a in ARMS_LIST},
            coral_minus_placebo={k: (v.item() if hasattr(v, "item") else v)
                                 for k, v in _c.items()},
            boundary_status_per_direction={d: STATUS[(d, GATED)] for d in DIRECTIONS},
            baseline_status_per_direction={d: STATUS[(d, "NONE")] for d in DIRECTIONS},
            symmetric_boundary=SYMMETRIC,
            any_fold_at_edge={f"{k[0]}|{k[1]}": bool(v) for k, v in ANY_EDGE.items()},
            boundary_inclusive_thresholds=True,
            verdict_precedence=["validity (G0,G5,G7)", "both-directions-inconclusive",
                                "full pass", "regularisation", "direction-inconsistent",
                                "below-band", "not-resolved", "refuted"],
            thresholds=dict(G0=CFG.G0_TOL, G1=CFG.G1_MIN, G2=CFG.G2_MIN, G3=CFG.G3_MIN,
                            G4=CFG.G4_MIN, G5=CFG.G5_MAX),
            n_rows=int(CFG.n_rows_eval), n_clusters=int(CFG.n_clusters),
            penalty_emd=PENALTY, base_mean_emd=_BASE_MEAN,
            g0=dict(reference=G0_REF, measured=G0_GOT, gap=G0_GAP))
json.dump(GATE, open(OUT_DIR / "gate.json", "w"), indent=2, default=str)
GD.to_csv(OUT_DIR / "transfer_results.csv", index=False)
TRANS.to_csv(OUT_DIR / "per_soil_transfer.csv", index=False)
print("wrote gate.json, transfer_results.csv, per_soil_transfer.csv")

# ---- code cell 16 ----
print(" >>> cell 16", flush=True)
"""Cell H1 - fit the submission arm and predict the 10 test soils."""
SUB_ARM = GATED if SUBMIT else "NONE"
WRITES_FILE = bool(SUBMIT and CFG.probe_arm is None)
PLUMBING_ONLY = not WRITES_FILE


def predict_for(fit_ids, fit_img, eval_ids, eval_img, arm):
    T = fold_transforms(fit_img)[arm]
    Xf, Yf, famf, Xe = fold_matrices({"fit_ids": list(fit_ids), "fit_img": fit_img,
                                      "eval_ids": list(eval_ids), "eval_img": eval_img}, T)
    return te.nested_predict(Xf, Yf, famf, Xe)


_TEST_IMG = IMG[IMG.split == "test"]
if SUBMIT:
    _fit_ids, _fit_img = TR_IDS, IMG[IMG.split == "train"]
    _note = "production recipe: all 24 labelled soils, aligned onto the 35 test images"
else:
    _f0 = _D["A"][-1]
    _fit_ids, _fit_img = _f0["fit_ids"], _f0["fit_img"]
    _note = "plumbing check through a real fold on the unaligned arm; no file will be written"
_r = predict_for(_fit_ids, _fit_img, TE_IDS, _TEST_IMG, SUB_ARM)
P_test = _r["pred"]
assert len(TR_IDS) == 24 and set(_fit_ids) <= set(IMG.sample_id)
assert _fit_img.sample_id.nunique() == len(_fit_ids)
assert P_test.shape == (10, 11), f"expected 10 test soils x 11 supports, got {P_test.shape}"
assert (np.diff(P_test, axis=1) >= -1e-9).all() and np.allclose(P_test[:, -1], 100.0)
print({"gate": "SUBMISSION - the gate fired", "plumb": "PLUMBING CHECK - the gate did not fire"}
      ["gate" if WRITES_FILE else "plumb"])
print(f"  arm {SUB_ARM}: {len(FEATS)} features, alpha {_r['alpha']:g}, "
      f"boundary {_r['status']}   [{_note}]")
print("  NOTE the submitted configuration is NOT the gate-evaluated one: it uses all 24 labelled")
print("  soils and its alignment target is the same 10 soils it predicts. Spec section 7.")
print("  No camera, ppm, EXIF, ICC profile or sample id reaches the model: cell A3's transforms")
print("  take feature matrices only, and the target was selected on the split column alone.")

# ---- code cell 17 ----
print(" >>> cell 17", flush=True)
"""Cell P2 - eight validation gates, then write only if the gate fired."""
_idmap = dict(zip(samples.sample_id, samples.submission_id))
submit = pd.DataFrame({"sample_id": [str(_idmap[s]) for s in te_ids]})
for c in TARGET_COLS:
    submit[c] = P_test[:, TARGET_COLS.index(c)]
_ref = pd.read_csv(INPUT_ROOT / "data" / "sample_submission.csv")
checks = {
    "row count == 10": len(submit) == len(_ref) == 10,
    "ids exactly match sample_submission order":
        submit.sample_id.tolist() == _ref.sample_id.astype(str).tolist(),
    "columns exactly match": list(submit.columns) == list(_ref.columns),
    "no missing values": bool(~submit.isna().to_numpy().any()),
    "values within [0,100]": bool((submit[TARGET_COLS].to_numpy(float) >= 0).all())
        and bool((submit[TARGET_COLS].to_numpy(float) <= 100).all()),
    "cumulative / non-decreasing": bool((np.diff(submit[TARGET_COLS].to_numpy(float),
                                                 axis=1) >= -1e-9).all()),
    "200mm column == 100": bool(np.allclose(submit["200"].to_numpy(float), 100.0)),
    "no extra columns": len(submit.columns) == 12,
}
print("eight submission gates "
      + ("(plumbing check on the control arm - the file is NOT written):" if PLUMBING_ONLY
         else ("on the arm being SUBMITTED (gate passed):" if SUBMIT
               else "on the arm being SUBMITTED as a DECLARED PROBE:")))
for k, ok in checks.items():
    print(f"  {'PASS' if ok else '**FAIL**'}  {k}")
assert all(checks.values()), "the submission machinery produces a malformed frame"
print("\nall 8 submission checks passed")
SUB_PATH = None
if WRITES_FILE:
    SUB_PATH = OUT_DIR / f"Submission_{MODEL_ID}_{EXPERIMENT_ID}.csv"
    submit.to_csv(SUB_PATH, index=False)
    back = pd.read_csv(SUB_PATH)
    assert list(back.columns) == list(_ref.columns)
    assert back.sample_id.astype(str).tolist() == _ref.sample_id.astype(str).tolist()
    assert not back.isna().any().any()
    assert (np.diff(back[TARGET_COLS].to_numpy(float), axis=1) >= -1e-9).all()
    print(f"wrote {SUB_PATH}  ({SUB_PATH.stat().st_size} bytes)"
          + ("  [PROBE - not a gate pass]" if not SUBMIT else ""))
    print(back.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
else:
    print("NO SUBMISSION WRITTEN. The gates above validated the machinery only.")
    for c in CLAUSES:
        print(f"  {'ok ' if c['passed'] else 'NOT '} {c['clause']}")

# ---- code cell 18 ----
print(" >>> cell 18", flush=True)
"""Cell Q1 - per-soil gains, both directions, and the placebo's own gain."""
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
for k, _dn in enumerate(DIRECTIONS):
    # read the per-soil Deltas from F1's helper rather than redefining the difference a third time;
    # a second implementation is how a sign convention gets quietly inverted
    ids = arm_rows(GATED)[2][_dn][1]
    g, p = arm_rows(GATED)[2][_dn][2], arm_rows("CORAL_INDEP")[2][_dn][2]
    ax[k].bar(np.arange(len(ids)), g, color=["#2a6" if v > 0 else "#b33" for v in g],
              label=f"{GATED} mean {g.mean():+.2f}")
    ax[k].plot(np.arange(len(ids)), p, ls="none", marker="x", ms=5, color="#888",
               label=f"placebo mean {p.mean():+.2f}")
    ax[k].axhline(0, color="k", lw=0.8)
    ax[k].set_title(f"direction {_dn}: {GATED} Delta over NONE (positive = better)")
    ax[k].set_xlabel("soil, sorted by eval id")
    ax[k].legend(frameon=False, fontsize=7)
    print(f"  direction {_dn}: {len(ids)} soils, {int((g > 0).sum())} improved by CORAL, "
          f"{int((p > 0).sum())} improved by the placebo")
plt.tight_layout()
plt.show()
print("Q1 drew both panels; an empty cell here would mean the plot silently skipped.")

# ---- code cell 19 ----
print(" >>> cell 19", flush=True)
"""Cell R1 - the pre-registered predictions, evaluated mechanically."""
PREDS = []


def pred(pid, text, blind, status, evidence):
    PREDS.append(dict(id=pid, prediction=text, blind="yes" if blind else "no", status=status,
                      evidence=evidence))


_g, _pl, _mn, _sl = GAIN[GATED], GAIN["CORAL_INDEP"], GAIN["MEAN"], GAIN["CORAL_SELF"]
_C = CONTRASTS.set_index("arm_b").loc["CORAL_INDEP"]
_best_arm = max(ARMS_LIST, key=lambda a: GAIN[a]["pooled_gain"])
_best_gain = GAIN[_best_arm]["pooled_gain"]

pred("P1", f"H5 is refuted: pooled gain below {CFG.G1_MIN} EMD, or its CI includes zero", True,
     "CONFIRMED - H5 refuted" if (not G1 or not G3) else "REFUTED - H5 survives",
     f"gain {_g['pooled_gain']:+.3f} CI [{_g['pooled_lo']:+.3f}, {_g['pooled_hi']:+.3f}]")
pred("P2", f"placebo gain within +/-{CFG.P2_TOL} EMD of CORAL's (it is variance rescaling)",
     True, "CONFIRMED" if abs(_pl["pooled_gain"] - _g["pooled_gain"]) <= CFG.P2_TOL else "REFUTED",
     f"placebo {_pl['pooled_gain']:+.3f} vs CORAL {_g['pooled_gain']:+.3f}, paired delta "
     f"{_C['mean']:+.3f} CI [{_C['lo']:+.3f}, {_C['hi']:+.3f}]")
pred("P3", f"MEAN carries most of it: gain(MEAN) >= gain(CORAL) - {CFG.P3_TOL}", True,
     "CONFIRMED" if _mn["pooled_gain"] >= _g["pooled_gain"] - CFG.P3_TOL else "REFUTED",
     f"MEAN {_mn['pooled_gain']:+.3f} vs CORAL {_g['pooled_gain']:+.3f}")
pred("P4", f"|gain(CORAL_SELF)| < {CFG.P4_MAX} EMD (a magnitude prediction, not the G5 bound)",
     True, "CONFIRMED" if abs(_sl["pooled_gain"]) < CFG.P4_MAX else "REFUTED",
     f"{_sl['pooled_gain']:+.5f} EMD; the G5 validity band is {CFG.G5_MAX} so the run stays valid")
pred("P5", f"no arm closes more than {int(CFG.P5_MAX * 100)}% of the camera penalty", True,
     "not evaluated - the penalty measured here is not positive, so a fraction of it is undefined"
     if PENALTY <= 0 else
     ("CONFIRMED" if _best_gain <= CFG.P5_MAX * PENALTY else "REFUTED"),
     f"best arm {_best_arm} {_best_gain:+.2f} vs penalty {PENALTY:.2f} EMD "
     f"({100.0 * _best_gain / PENALTY:.1f}% closed)" if PENALTY > 0
     else f"best arm {_best_arm} {_best_gain:+.2f}; penalty {PENALTY:+.2f}")
_o = {a: GAIN[a]["pooled_gain"] for a in SENSITIVITY}
pred("P6", "sensitivity units agree in SIGN with the gated arm, differ in magnitude", True,
     "CONFIRMED" if all(np.sign(v) == np.sign(_g["pooled_gain"]) for v in _o.values())
     else "REFUTED - the result is an artefact of the estimation unit",
     " | ".join([f"CORAL {_g['pooled_gain']:+.2f}"]
                + [f"{k} {v:+.2f}" for k, v in _o.items()]))
pred("X1", "MEAN vs CORAL is a finding about the shift, not a gate clause (spec section 5)",
     True, "covariance half adds nothing measurable" if _mn["pooled_gain"] >= _g["pooled_gain"]
     else "covariance half adds something",
     f"CORAL - MEAN = {CONTRASTS.set_index('arm_b').loc['MEAN', 'mean']:+.3f} EMD")
PRED_DF = pd.DataFrame(PREDS)
PRED_DF.to_csv(OUT_DIR / "predictions.csv", index=False)
print("PRE-REGISTERED PREDICTIONS\n")
for r in PRED_DF.itertuples(index=False):
    print(f"  {r.id} [{'blind' if r.blind == 'yes' else '  n/a'}] {r.prediction[:70]}")
    print(f"       -> {r.status}: {r.evidence[:110]}")

# ---- code cell 20 ----
print(" >>> cell 20", flush=True)
"""Cell R2 - write Experiment1.txt from live objects."""
L = []
W = L.append
W("=" * 78)
W(f"{MODEL_ID} / {EXPERIMENT_ID}  {EXPERIMENT_NAME}")
W("=" * 78)
W(f"run context      : {RUN_CONTEXT}  (no torch, no GPU, no Kaggle run, no dataset, no zip)")
W(f"seeds            : SEED={SEED}  bootstrap={BOOT_SEED}  placebo shuffle={SHUF_SEED}")
W(f"config_hash      : {CONFIG_HASH}   pipeline {PIPELINE_VERSION}")
W(f"arms             : {list(ARMS)}   gated: {GATED}")
W(f"verdict          : {VERDICT}   submit={SUBMIT}")
W("see section 16   : two implementation defects were corrected during this run, BEFORE any gate")
W("                   was evaluated. Read it before interpreting any number below.")
W("")
W("1. HYPOTHESIS, AND WHY IT IS THE LAST LARGE LOSS ANYONE HAS ATTACKED")
W("   H5: aligning the fitting camera's image features onto the 35 unlabeled test images")
W("   lowers EMD on a camera that was neither fitted on nor the alignment target.")
W("   Model 1: replacement refuted. Model 2: scale refuted, then concatenation refuted.")
W("   Camera penalty measured earlier: +16.2 EMD (D) and +18.2 (hand). Output basis ceiling:")
W("   4.9 EMD. Curve-space blending: 0.9 EMD of information-specific gain. This is the")
W("   remaining large loss, and P1 predicted H5 would still be refuted.")
W("")
W("2. THE ONE VARIABLE, AND WHAT IS FROZEN")
W("   Inserted: image median -> ALIGN -> soil median. Frozen: features, aggregation,")
W("   StandardScaler, rank-3 basis, closed-form ridge, alpha grid, monotone projection,")
W("   16 CV families, trapezoid EMD, parsimony rule, config_hash.")
W(f"   Cells reused byte-for-byte from Model 2 E3: A2, A4, B1, B2, B3, P2")
W("")
W("3. GATE G0 - THE PIPELINE PROOF")
W(f"   no-holdout unaligned control: {G0_GOT:.13f}  vs E3 artifact {G0_REF:.13f}")
W(f"   gap {G0_GAP:.3e} against tolerance {CFG.G0_TOL}  ->  {'PASS' if G0 else 'FAIL'}")
W("   The reference is read from Model 2 E3's sweep_results.csv at run time, never typed.")
W("")
W("4. FOLD STRUCTURE (spec 14) - NO SOIL IS SCORED BY A MODEL THAT SAW IT")
for d in DIRECTIONS:
    W(f"   direction {d}: {len(_D[d])} folds, {sum(len(f['eval_ids']) for f in _D[d])} rows, "
      f"fit pools from {min(len(f['fit_ids']) for f in _D[d])} to "
      f"{max(len(f['fit_ids']) for f in _D[d])} soils")
W("   21 soils sit on both sides of the camera split, so a set-level holdout would still have")
W("   leaked them; folds are what makes the 45 rows honest without changing any threshold.")
W("")
W("5. PRIMARY RESULT - mean cross-camera EMD by arm (lower is better)")
for d in DIRECTIONS:
    W(f"   direction {d} ({te.DIRECTION_SPEC[d][0]} fitted -> {te.DIRECTION_SPEC[d][1]} scored):")
    for a in ARMS:
        m = float(TRANS[(TRANS.direction == d) & (TRANS.arm == a)].emd.mean())
        W(f"      {a:12s} {m:8.3f}")
W("")
W(f"6. PAIRED GAINS OVER NONE, SOIL-CLUSTERED 95% CI "
  f"({CFG.n_rows_eval} rows, {CFG.n_clusters} clusters)")
for r in GD.itertuples(index=False):
    W(f"   {r.arm:12s} pooled {r.pooled_gain:+7.3f} CI [{r.pooled_lo:+7.3f}, {r.pooled_hi:+7.3f}]"
      f"   A {r.gain_A:+6.3f}  B {r.gain_B:+6.3f}")
W("")
W("7. CONTRAST AGAINST THE SAME-MARGINAL PLACEBO (clause G4)")
W(f"   CORAL - CORAL_INDEP {_C['mean']:+.3f} CI [{_C['lo']:+.3f}, {_C['hi']:+.3f}]")
W("   CORAL_INDEP preserves every per-feature mean, variance and marginal of the real target")
W("   and destroys only the joint structure. A gain it also achieves is variance rescaling.")
W("   Row permutation is NOT used as a placebo here: a mean and a covariance are invariant to")
W("   row order, so it is bit-identical to the real arm (spec 13).")
W("")
W("8. ALPHA BOUNDARY, PER DIRECTION (spec 5a)")
for d in DIRECTIONS:
    W(f"   {d}: gated arm {STATUS[(d, GATED)]}; baseline NONE {STATUS[(d, 'NONE')]}; "
      f"per-fold alphas " + " ".join(f"{x:g}" for x in ALPHAS_BY_FOLD[(d, GATED)]))
W(f"   directions at a boundary: {BOUNDARY}   caveat={CAVEAT}")
W(f"   symmetric_boundary diagnostic: {SYMMETRIC}")
W("   Majority-of-folds defines the direction status; every fold's own status, oracle alpha and")
W("   grid spread are in alpha_by_fold.csv so the rule can be audited rather than trusted.")
W("")
W("9. DECISION GATE")
for c in CLAUSES:
    W(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['clause']}   {c['detail']}")
W(f"   verdict {VERDICT}   submit {SUBMIT}")
W("   submit is true if and only if the verdict is SUPPORTED (spec section 5); the notebook asserts")
W("   that identity, so a boundary caveat can never license a file. Where two spec rows overlap,")
W("   spec section 6a fixes the order: validity > both-directions > full pass > placebo > direction")
W("   > below-band > unresolved > refuted. Ordering assigns labels only; no threshold moved.")
W("")
W("10. SUBMISSION")
if SUB_PATH is not None:
    W(f"   {SUB_PATH.name} written by the gate. Production recipe (24 soils -> 35 test images).")
    W("   DECLARED DIFFERENCE: this configuration was never itself gate-evaluated - it uses more")
    W("   fitting soils than either direction and its alignment target is the set it predicts.")
else:
    W("   NO SUBMISSION WAS PRODUCED. submit is False and probe_arm is None: Model 5 pre-")
    W("   authorises no probe. The eight structural gates still ran, as a labelled plumbing")
    W("   check, so the submission machinery is not untested code.")
W("")
W("11. PRE-REGISTERED PREDICTIONS")
for r in PRED_DF.itertuples(index=False):
    W(f"   {r.id} [{'blind' if r.blind == 'yes' else '  n/a'}] {r.prediction}")
    W(f"        -> {r.status}: {r.evidence}")
W("")
W("12. WHAT THIS DECIDES")
if VERDICT == "REFUTED":
    W("   MODEL 5 CLOSES AT ONE EXPERIMENT (spec section 10): the closed-form, label-free,")
    W("   transductive form of domain adaptation was tested on the only two measurable camera")
    W("   directions, against a same-marginal placebo, and found insufficient.")
elif VERDICT == "REFUTED-WITH-BOUNDARY-CAVEAT":
    W("   Refuted on the measured rows, but the grid clipped one direction, so the permanent")
    W("   closure rule does NOT fire (spec 5a). A grid amendment is a separate owner decision.")
elif VERDICT == "SUPPORTED":
    W("   Supported on both directions and beyond the placebo: submission permitted.")
elif VERDICT == "SUPPORTED-WITH-BOUNDARY-CAVEAT":
    W("   The effect is there but the grid clipped a direction: no submission, and a documented")
    W("   reason to reopen with an amended grid.")
elif VERDICT == "SUPPORTED-BELOW-BAND":
    W("   The mechanism is real but smaller than the pre-registered floor: no submission, and no")
    W("   dressing it up as a win (spec section 6).")
else:
    W(f"   See spec section 6 for the meaning of {VERDICT} and the action it licenses.")
W("")
W("13. LIMITATION, STATED IN ADVANCE")
W("   Two camera pairs and 45 rows is the entire measurable universe here. A pass rests on two")
W("   domain pairs; a refutation is weaker than it looks for the same reason. iPhone itself is")
W("   never scored locally, CAM+RES is ordinal-only, and the public leaderboard is 3 of 10")
W("   soils while the final ranking is 0.30*public + 0.70*private.")
W("")
W("14. SCOPE OF VERIFICATION")
W("   No mock mode: this run executed every cell on the real cached features. Both contract")
W(f"   suites passed in-subprocess before any score was computed, G0 held to {G0_GAP:.2e},")
W(f"   every arm produced exactly {CFG.n_rows_eval} soil rows over {CFG.n_clusters} clusters,")
W("   and the fold-leakage and placebo-distinctness assertions are live code, not prose.")
W("   What this run does NOT verify: it cannot verify transfer to a third camera, and the")
W("   submitted configuration is a declared extrapolation beyond the tested one.")
W("")
W("15. ENVIRONMENT")
for k, v in VERSIONS.items():
    W(f"   {k:12s} {v}")
W("")
W("16. DISCLOSURE - TWO IMPLEMENTATION DEFECTS CORRECTED DURING THIS RUN")
W("   A first attempt at this run aborted in cell F1 on its own n_clusters assertion, before the")
W("   gate was evaluated and before any score was written. The assertion was right and the code")
W("   was wrong. Two defects were found, both in the difference-and-bootstrap helpers (F1, F2),")
W("   both corrected to match the contract as ALREADY WRITTEN in instructions.txt and spec")
W("   section 3 - no threshold, arm, seed, precedence, clause or interpretation rule changed:")
W("     (1) SIGN. Delta was computed as err_arm - err_NONE. The contract defines")
W("         Delta(s) = err_NONE(s) - err_arm(s), positive meaning the arm is better. The inverted")
W("         form would have reported an 8.75 EMD DEGRADATION as an 8.75 EMD IMPROVEMENT, and had")
W("         G2 been satisfied it would have produced SUPPORTED and written a submission file on a")
W("         result that harms the model.")
W("     (2) CLUSTER UNIT. Pooled bootstrap keys were prefixed with the direction ('A:S07'), so 45")
W("         rows became 45 clusters and the soil-clustered bootstrap silently degenerated to the")
W("         row bootstrap it was specified to replace. Keys are now the bare soil identity: 45")
W("         rows, 24 clusters, 21 of them spanning both directions.")
W("   WHAT WAS OBSERVED BEFORE THE FIX, and it is reported here rather than hidden: the per-arm")
W("   means and per-fold alphas were already on screen and are UNAFFECTED by either defect (they")
W("   come from cells upstream of F1). Direction A arm means 51.784 / 70.248 / 79.366 for")
W("   NONE / CORAL / CORAL_INDEP, direction B 45.366 / 44.820 / 45.291, are the same numbers this")
W("   record prints in section 5. Only the CIs and the sign of every Delta changed.")
W("   MEASURED EFFECT OF DEFECT (2), reported because it contradicts a prediction in the contract:")
W("   for THIS data the soil-clustered intervals are NARROWER than the row-level ones, not wider")
W("   (CORAL: soil [-12.78,-3.97] width 8.81 vs row [-14.34,-3.10] width 11.23; CORAL_SOIL 8.35")
W("   vs 11.41). The sqrt(2)-narrowing argument is about within-cluster correlation and its")
W("   DIRECTION depends on the sign of that correlation, which the contract assumed rather than")
W("   measured. Row resampling is still invalid here - two rows for one soil are not two")
W("   observations - so the correction stands, but the contract's stated expectation about which")
W("   way it moves the interval was wrong and is corrected as a fact, not argued away.")
W("   Neither defect changes the verdict: with the sign fixed and the old clustering the verdict")
W("   is still DIRECTION-INCONSISTENT, and with the old sign the same label is reached by")
W("   accident. The sign fix is what turns G3 from a spurious pass into a fail.")
W("   A regression test now pins both properties by executing the shipped F1 helpers against a")
W("   synthetic mirror of the design: scratch/check_m5_delta.py. It failed 8 ways on the pre-fix")
W("   code and passes 14 on this one, and cell F1 recomputes one soil's Delta by hand and asserts")
W("   the cluster count at run time so the notebook can no longer produce an inverted or")
W("   row-level number silently. G0 was 43.0217308796477 (gap 0.00e+00) in both attempts, and the")
W("   emitted CSVs are byte-identical across independent runs of the corrected notebook.")
W("")
W("17. WHAT THE CAMERA PENALTY ACTUALLY MEASURED, AND A PREMISE CORRECTED")
W(f"   The penalty defined and measured here (cross-camera mean minus same-camera in-domain")
W(f"   control) is {PENALTY:.2f} EMD, not the +18.2 quoted in Model 2 E1's record. The two are not")
W("   the same quantity: +18.2 is M1's nested in-domain 43.02 against its EXTERNAL leaderboard")
W("   61.24, a gap that bundles unseen-soil generalisation, camera transfer and the fixed 3-of-10")
W("   public subset. The camera-only component, measured inside the training distribution on 45")
W("   rows, is roughly a third of it. Model 5 therefore attacked a target whose size had been")
W("   overstated by conflating two gaps, and P5's 'no arm closes more than 25%' was evaluated")
W(f"   against {PENALTY:.2f}. Any future decision about where the remaining loss lives should")
W("   start from this decomposition rather than from 18.2.")
W("")
W("18. TWO READINGS THAT MUST NOT BE OVER-INTERPRETED")
W("   P3 came back CONFIRMED and X1 'covariance adds nothing measurable', but both are artifacts")
W("   of an identity, not evidence: the MEAN arm is a constant shift, the soil median commutes")
W("   with it, and the frozen StandardScaler then removes it, so MEAN is EXACTLY NONE by")
W("   construction (proved in scratch/verify_m5_result.py: max scaled-feature difference 6.9e-15).")
W("   CORAL_SELF is likewise the identity to 3.7e-15, so G5 is a real canary that can only fire on")
W("   a wiring fault. X1's label is worded for the case where MEAN carries a gain; here MEAN")
W("   carries nothing at all, and the correct reading is that the covariance half COST 8.75 EMD.")
W("   The honest positive result is that the instrument worked: the placebo degraded even more")
W("   than CORAL did (-27.58 vs -18.46 in direction A), so the two arms separate in the")
W("   direction the design was built to detect.")
W("")
W("19. WHERE THE RAW EVIDENCE LIVES")
W("   attempt 1 (aborted at F1, pre-fix, no gate): scratch/_m5_run1.log")
W("   attempt 2 (this record's numbers):              scratch/_m5_run2.log")
W("   attempt 3 (determinism, 10/10 artifacts byte-identical): scratch/_m5_run3.log")
W("   post-run verification: scratch/verify_m5_result.py (27 checks)")
W("   sign/cluster regression: scratch/check_m5_delta.py (14 checks; 8 failures pre-fix)")
REC_PATH = OUT_DIR / "Experiment1.txt"
REC_PATH.write_text("\n".join(L) + "\n", encoding="utf-8")
_records = sorted(p.name for p in OUT_DIR.glob("Experiment*.txt"))
assert _records == ["Experiment1.txt"], f"Model 5 must own exactly one record, found {_records}"
print(f"wrote {REC_PATH} ({len(L)} lines)")

# ---- code cell 21 ----
print(" >>> cell 21", flush=True)
"""Cell R3 - summary."""
print("=" * 70)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  verdict={VERDICT}  submit={SUBMIT}")
for a in ARMS_LIST:
    print(f"  {a:12s} pooled gain {GAIN[a]['pooled_gain']:+7.3f} CI "
          f"[{GAIN[a]['pooled_lo']:+7.3f}, {GAIN[a]['pooled_hi']:+7.3f}]")
print(f"  G0 gap {G0_GAP:.2e} | boundary {BOUNDARY} | placebo delta {_C['mean']:+.3f} | "
      f"penalty {PENALTY:.2f}")
print("files:")
for f in sorted(OUT_DIR.glob("*")):
    if f.is_file():
        print(f"  ok   {f.name}")

