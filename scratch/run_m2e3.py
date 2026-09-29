import matplotlib; matplotlib.use("Agg")

# ---- code cell 1 ----
print(" >>> cell 1", flush=True)
"""Cell A1 - configuration. Every tunable lives here and nowhere else."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

MODEL_ID        = "Model2"
EXPERIMENT_ID   = "E3"
EXPERIMENT_NAME = "Model 2 / Experiment 3 - hand features + DINOv2 block, information or capacity"
SEED = 20260932
E1_DIR_NAME = "Model 2 Experiment 1"
E2_DIR_NAME = "Model 2 Experiment 2"
E3_DIR_NAME = "Model 1 Experiment 3"


@dataclass(frozen=True)
class Config:
    # ---- frozen at Model 2 E1/E2's values; changing any of these is a different experiment
    tile_size_px: int = 256
    min_tile_soil_fraction: float = 0.50
    pc_rank: int = 3
    holdout_cameras: Tuple[str, ...] = ("Motorola", "Samsung")
    res_match_targets: Tuple[str, ...] = ("iPhone 14", "iPhone 16")
    n_boot: int = 4000
    n_perm: int = 200
    ORDINAL_FLOOR: float = 15.0
    control_feats: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean",
                                      "R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
    embed_seed: int = 20260930      # PINNED BY E2'S CACHE FILENAMES. Changing it does not
                                    # re-key anything - it invalidates E3's entire premise,
                                    # because the arrays would no longer be E2's.
    batch_size: int = 64            # unused without torch; kept so CFG matches E1/E2 field for field
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                                     300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0,
                                     300000.0, 1000000.0)
    crops: Tuple[int, ...] = (256, 128, 64, 32)
    # ---- THE ONE NEW FACT: the gated arm is the fusion at E1's frozen magnification.
    # 128 is D's best crop in E2 and is NOT used for the gate - choosing it after seeing E2's
    # numbers is the post-hoc selection the pre-registration discipline exists to prevent.
    gated_crop: int = 256
    permutation_seed: int = 20260933
    embed_dim: int = 384
    # ---- gate thresholds, fixed before the run
    REGRESSION_TOL: float = 0.05     # EMD; all three base arms must reproduce E2 this closely
    probe_arm: Optional[str] = None  # E3 PRE-AUTHORISES NO PROBE. See instructions.txt.
    fine_d50_mm: float = 0.11
    mid_d50_mm: float = 1.5


CFG = Config()

# Arm naming keeps E2's "@c<crop>" convention so the reused scoring cells parse it unchanged.
GATED = f"fuse@c{CFG.gated_crop}"     # matches the "@c<crop>" arm-naming convention
ARMS_LIST_EXPECTED = ["M1", "dinov2@c256", "vit_random@c256",
                      "fuse@c256", "shuf@c256", "fr@c256",
                      "fuse@c128", "fuse@c64", "fuse@c32"]
assert GATED in ARMS_LIST_EXPECTED, (
    f"the gated arm name {GATED!r} is not one of the arms this experiment builds - an arm "
    "name typo must fail in A1, not eleven cells later in J1")
LABEL = {"M1": "M1", "dinov2@c256": "D", "vit_random@c256": "R", "fuse@c256": "FUSE",
         "shuf@c256": "SHUF", "fr@c256": "FR", "fuse@c128": "FUSE@128",
         "fuse@c64": "FUSE@64", "fuse@c32": "FUSE@32"}

_EXP_DIR = Path("Model 2 / Model 2 Experiment 3".replace(" / ", "/"))
OUT_DIR = _EXP_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  permutation_seed={CFG.permutation_seed}")
print(f"gated arm: {GATED}   embed_seed: {CFG.embed_seed} (pinned by E2's cache names)")
print(f"expected arms ({len(ARMS_LIST_EXPECTED)}): {ARMS_LIST_EXPECTED}")
print("outputs ->", OUT_DIR)
print("torch/timm: NOT IMPORTED. E3 has no GPU path and no mock path by design.")

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
"""Cell A3 - locate the processed data and E3's own modules. LOCAL ONLY.

E2's version of this cell carried a bounded walk over /kaggle/input because Kaggle's mount
depth is not stable (both /kaggle/input/<slug> and /kaggle/input/datasets/<user>/<slug> have
been observed, three levels down). E3 runs locally by the user's decision on 2026-09-29, so
that walk is DROPPED rather than kept as a branch no local run ever exercises - an untested
branch is exactly where E2's two Kaggle crashes lived.

backbones.py is deliberately NOT imported. E3 embeds nothing, so importing the torch surface
would be dead code with a failure mode.
"""
def _find_module(fname, must_be_in=None):
    """Nearest candidate root containing fname, checked in a fixed, printed order."""
    here = Path.cwd()
    cands = [here, *here.parents]
    if must_be_in:
        cands = [Path(must_be_in)] + cands
    for d in cands:
        try:
            if (d / fname).exists():
                return d
        except OSError:
            continue
    return None


PROBE = Path("data") / "processed_meta" / "manifest_images.csv"
INPUT_ROOT = None
for _d in [Path.cwd(), *Path.cwd().parents]:
    if (_d / PROBE).exists():
        INPUT_ROOT = _d
        break
if INPUT_ROOT is None:
    raise RuntimeError(
        f"Could not find {PROBE} from {Path.cwd()} or any parent. E3 is local-only; run the "
        "notebook from the repository root.")
RUN_CONTEXT = "local"
META = INPUT_ROOT / "data" / "processed_meta"
sys.path.insert(0, str(INPUT_ROOT))

# Located by CONTENT, not by __file__: this notebook also runs flattened as a script, where
# __file__ points at scratch/ and every sibling import would break.
EXP_DIR = _find_module("fusion.py", must_be_in=str(INPUT_ROOT / "Model 2" / "Model 2 Experiment 3"))
assert EXP_DIR is not None, ("fusion.py not found. It is E3's one new variable and lives in "
                             "Model 2/Model 2 Experiment 3/; run from the repository root.")

CG_DIR = _find_module("crop_geometry.py",
                      must_be_in=str(INPUT_ROOT / "Model 2" / "Model 2 Experiment 2"))
assert CG_DIR is not None, ("crop_geometry.py not found in Model 2 Experiment 2. E3 imports "
                            "it READ-ONLY for the geometry table and the D50 grouping; it "
                            "does not own any E3 variable.")
sys.path.insert(0, str(CG_DIR))
sys.path.insert(0, str(EXP_DIR))

def _attachment_roots():
    """Directories that could hold a previous experiment's outputs. E3-LOCAL VERSION.

    E2's version of this function walked /kaggle/input to a bounded depth because Kaggle's
    mount level is not stable (both /kaggle/input/<slug> and /kaggle/input/datasets/<user>/
    <slug> have been observed). E3 runs locally only, so that walk is replaced by an explicit
    list of repo roots rather than left as a branch nothing here executes. C1 is reused
    byte-for-byte and calls this, so the name and the dedupe-by-resolve contract are kept.
    """
    here = Path.cwd()
    cands = [here, here / "Model 1", here / "Model 2", EXP_DIR,
             here / "Model 2" / "Model 2 Experiment 1",
             here / "Model 2" / "Model 2 Experiment 2",
             here / "Model 1" / "Model 1 Experiment 3"]
    seen, out = set(), []
    for d in cands:
        try:
            r = d.resolve()
        except OSError:
            continue
        if r.is_dir() and r not in seen:
            seen.add(r); out.append(r)
    return out


from preprocess import verify as pv          # the validated hand-built feature definition
import crop_geometry as cg                   # E2's module, unmodified, read-only here
import fusion as fu                          # E3's one new variable
import embedding_store as estore             # the suspicious loader

EMBED_STORE = EXP_DIR / "embeddings"
assert fu.EMBED_DIM == CFG.embed_dim == estore.EMBED_DIM == 384, "embedding width disagreement"

print(f"INPUT_ROOT    {INPUT_ROOT}")
print(f"context       {RUN_CONTEXT} (E3 has no Kaggle path)")
print(f"crop_geometry {cg.__file__}")
print(f"fusion        {fu.__file__}")
print(f"store         {EMBED_STORE}")
print(f"torch imported: {'torch' in sys.modules}")

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
"""Cell C1 - locate E1's and E3's outputs by content, not by folder name.

Plumbing folders are excluded from the search. E1's `local mock dry-run/` contains a
complete set of files with exactly the same names as its real results, and reading the
wrong one would make the regression check compare E2 against a mock run and pass
meaninglessly. Preference order: a non-plumbing folder that has the backbone arms, then a
non-plumbing folder, then anything - with a loud warning if it falls back.
"""
PLUMBING = ("local mock dry-run", "__pycache__")
# E1's arm means, pinned from its own record (cell H4 of E1, 2 d.p.). Pinned constants are
# legitimate here for the same reason the leaderboard scores are: they are external facts
# about a finished experiment, and they let E2's regression check work even when E1's
# per-soil file is not attached.
E1_PINNED = {"M1": 43.02, "D": 44.33, "R": 39.78}


def _is_plumbing(p):
    return any(part in PLUMBING for part in Path(p).parts)


def _dir_candidates(signature):
    out = []
    for d in _attachment_roots():
        if all((d / f).exists() for f in signature):
            out.append(d)
    for d in _attachment_roots():
        try:
            hits = list(d.rglob(signature[0]))
        except OSError:
            hits = []
        for p in hits:
            if all((p.parent / f).exists() for f in signature) and p.parent not in out:
                out.append(p.parent)
    # Never accept this experiment's own output directory. cv_per_soil.csv is a filename E2
    # writes itself, so without this the regression check would compare E2 against E2 and
    # pass unconditionally - the most convincing kind of worthless green tick.
    _self = OUT_DIR.resolve()
    return [d for d in out if _self not in d.resolve().parents and d.resolve() != _self]


def _find_dir(signature, label, want_cols=(), required=True):
    cands = _dir_candidates(signature)
    if not cands:
        if required:
            raise AssertionError(f"no folder containing {signature} was found for {label}. "
                                 "Download that experiment's Kaggle outputs into its folder "
                                 "and rebuild the dataset - see kaggle_setup.md section 1.")
        return None
    clean = [d for d in cands if not _is_plumbing(d)]
    pool = clean or cands
    if want_cols:
        best = [d for d in pool
                if set(want_cols) <= set(pd.read_csv(d / signature[0], nrows=0).columns)]
        if best:
            pool = best
    d = pool[0]
    if _is_plumbing(d):
        print(f"  NOTE: {label} resolved to a PLUMBING folder, {d}. Any check against it is")
        print("        weaker than intended.")
    return d


E3_DIR = _find_dir(("features_soil.csv", "cv_families.csv", "Submission_Model1_E3.csv"),
                   "Model 1 E3 artifacts")
E1_DIR = _find_dir(("cv_per_soil.csv", "Experiment1.txt"), "Model 2 E1 per-soil scores",
                   want_cols=("D", "R"), required=False)
if E1_DIR is None:
    print("  Model 2 E1's cv_per_soil.csv is not attached. The crop-256 regression check "
          "falls\n          back to E1's pinned arm means, which is a weaker but still "
          "real check.")
    e1_cv = pd.DataFrame()
    E1_ARMS = []
elif _is_plumbing(E1_DIR):
    print("  The only E1 per-soil file found is inside a plumbing folder, so it is not "
          "used.\n          The regression check falls back to E1's pinned arm means.")
    e1_cv = pd.DataFrame()
    E1_ARMS = []
else:
    e1_cv = pd.read_csv(E1_DIR / "cv_per_soil.csv").set_index("sample_id")
    E1_ARMS = [c for c in ("M1", "D", "R") if c in e1_cv.columns]
    if not {"D", "R"} <= set(E1_ARMS):
        print(f"  NOTE: E1's file carries only {E1_ARMS}. The crop-256 regression check will "
              "verify the\n        control arm per soil and the backbone arms against pinned "
              "means only.")
e3_soil = pd.read_csv(E3_DIR / "features_soil.csv").set_index("sample_id")
e3_fam = pd.read_csv(E3_DIR / "cv_families.csv")
e3_sub = pd.read_csv(E3_DIR / "Submission_Model1_E3.csv")
print(f"E1_DIR {E1_DIR}")
print(f"E3_DIR {E3_DIR}")
for _c in list(CFG.control_feats):
    assert _c in e3_soil.columns, f"E3's matrix is missing control feature {_c}"

# ---- code cell 9 ----
print(" >>> cell 9", flush=True)
"""Cell C2 - E2's per-soil table, located and loaded. This is E3's reference distribution.

Stronger than E2 managed: E2 had to check its crop-256 arm against E1's PINNED MEANS because
E1's per-soil file was not attached. E2's own per-soil file IS on disk here, so E3 can
reproduce all three base arms as 24-element vectors rather than three scalars. A matching
mean with a scrambled per-soil vector is possible in principle; this rules it out.
"""
E2_RES_DIR = _find_dir(("cv_per_soil.csv", "Experiment2.txt"), "Model 2 E2 per-soil scores",
                       want_cols=("dinov2@c256", "vit_random@c256"))
if E2_RES_DIR is None:
    e2_cv = pd.DataFrame()
    E2_LEVEL = "pinned means only (E2 per-soil file not attached)"
    print("  E2's cv_per_soil.csv was not found. H3 will fall back to E2's pinned arm means.")
else:
    e2_cv = pd.read_csv(E2_RES_DIR / "cv_per_soil.csv").set_index("sample_id")
    need = {"M1", "dinov2@c256", "vit_random@c256"}
    if not need <= set(e2_cv.columns):
        raise AssertionError(f"E2's per-soil file lacks {sorted(need - set(e2_cv.columns))}")
    E2_LEVEL = f"per-soil vectors ({len(e2_cv)} soils) from {E2_RES_DIR.name}"
print(f"  regression check level: {E2_LEVEL}")

E2_PINNED = {"M1": 43.02173087964769, "D": 44.330245615450906, "R": 39.77693008453303}

print("\nWHAT E1 CONCLUDED (REFUTED - replacement):  D 44.33 vs M1 43.02 in-domain;")
print("  D camera/soil 0.209 and blur/soil 0.029 against M1's 1.105 and 0.818, i.e. vastly")
print("  more invariant and slightly LESS accurate. R's apparent 39.78 win was the capacity")
print("  trap: grid-floor alpha, worst transfer, soil_distance 0.0000. Declared probe scored")
print("  60.56167 - a near tie with E3's 61.24, recorded as a probe, not a gate pass.")
print("WHAT E2 CONCLUDED (REFUTED - magnification):  D lost to R at every crop")
print("  (+4.55 / +0.96 / +3.14 / +6.24 EMD), no trend (rho -0.40, p 0.60), no gain located")
print("  in the patch-limited fines, and no submission was produced.")
print("WHAT E3 MUST NOT INHERIT:  any belief that magnification helped, and crop 128 as a")
print("  preference. E3 gates on crop 256 = E1's frozen configuration. Its machinery and its")
print("  frozen configuration are inherited; E2's hypothesis is dead.")

# ---- code cell 10 ----
print(" >>> cell 10", flush=True)
"""Cell C3 - provenance of the arrays E3 is about to trust, printed before any result.

E3's inputs are 15 .npy files produced by a different experiment in a different session on a
different machine. Before scoring anything, print what they are: shape, dtype, byte count,
sha256, and the minimum row norm (a zero-norm row would mean an embedding collapsed to the
origin, which standardisation cannot recover).
"""
_rows, _fails = estore.verify_all(EMBED_STORE, META, CFG.tile_size_px, CONFIG_HASH,
                                  CFG.embed_seed)
for _f in _fails:
    print(f"  FAIL {_f}")
assert not _fails, (
    f"{len(_fails)} of {len(estore.REQUIRED)} required embedding arrays are missing or "
            "malformed. E3 cannot embed them itself (no torch) and has no mock fallback. "
            "Retrieve E2's Kaggle cache - model_plan_m2e3.md section 10.")
_prov = estore.manifest_frame(_rows)
print(_prov[["file", "crop", "view", "rows", "cols", "dtype", "bytes",
             "min_abs_row_norm"]].to_string(index=False))
print(f"\nloaded {len(_prov)} arrays, {sum(_prov.bytes)/1e6:.1f} MB, "
      f"{len(set(_prov.sha256))} distinct sha256")
assert len(_prov) == len(estore.REQUIRED) == 15
assert len(set(_prov.sha256)) == len(_prov), "two arrays are byte-identical: a cache collision"
for _v, _g in _prov.groupby("view"):
    assert len(set(_g.rows)) == 1, f"view {_v} has inconsistent row counts"
print(f"rows per view: {dict((v, int(g.rows.iloc[0])) for v, g in _prov.groupby('view'))}")

# ---- code cell 11 ----
print(" >>> cell 11", flush=True)
"""Cell D1 - run check_fusion.py, then re-assert the loader in this process."""
import subprocess

_r = subprocess.run([sys.executable, "check_fusion.py"], cwd=str(EXP_DIR),
                    capture_output=True, text=True)
_tail = (_r.stdout or "").strip().splitlines()[-4:]
print("\n".join(_tail))
assert _r.returncode == 0, ("check_fusion.py failed with exit "
                            f"{_r.returncode}.\n{(_r.stdout or '')[-1500:]}\n{(_r.stderr or '')[-800:]}")
print("fusion.py contract test PASSED in this process's terms")

print("\nloader self-test (5 corruption classes) lives in scratch/check_e3_loader.py and is")
print("run by the plan's verification task, not here, because it fabricates fixtures.")

# ---- code cell 12 ----
print(" >>> cell 12", flush=True)
"""Cell F1 - tile metadata and the soil-fraction filter, identical to E1."""
_geo = imgs[["processed_path", "crop_area_fraction"]].rename(
    columns={"processed_path": "parent_image_path"})
META_T = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
               "soil_fraction"]].merge(_geo, on="parent_image_path", how="left")
META_T["camera_fam"] = META_T.camera.str.split(" ").str[0]
GOOD = META_T[META_T.soil_fraction >= CFG.min_tile_soil_fraction].copy()
print(f"tiles {len(META_T)} -> soil_fraction >= {CFG.min_tile_soil_fraction}: {len(GOOD)} "
      f"({100*len(GOOD)/len(META_T):.1f}%)")
print(GOOD.groupby(["split", "camera_fam"]).size().to_string())
_tps = GOOD.groupby(["split", "sample_id"]).size().groupby("split").median()
print(f"\nmedian tiles per soil: train {_tps.get('train', 0):.0f}, test {_tps.get('test', 0):.0f}"
      "   (test soils already see fewer views; a fine crop shrinks the area per view too)")

# ---- code cell 13 ----
print(" >>> cell 13", flush=True)
"""Cell F2 - the hand-built features for the control arm, reused from E1/E3's caches.

Identical to E2's cell F2 except that the backbone module's own .cache is no longer a search
source: E3 imports no torch, so `bb.__file__` does not exist. Dropping that path is also
what forces the hand-feature CSVs to come from Model 1 E3's cache, which is where they belong.
"""
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


CACHE = OUT_DIR / ".cache"; CACHE.mkdir(exist_ok=True)
VIEWS = ["real"] + [f"res{t.replace('iPhone', '').strip()}" for t in CFG.res_match_targets]
_srcs = [CACHE, E3_DIR / ".cache"]
# E2's guard, kept deliberately: E1_DIR above resolved to E1's "local mock dry-run" folder,
# whose files share names with its real outputs. Reading features from a plumbing folder makes
# a reproduction check pass for the wrong reason, so plumbing is excluded even though the CSVs
# in this particular case happen to be identical.
if E1_DIR is not None and not _is_plumbing(E1_DIR):
    _srcs.append(E1_DIR / ".cache")
_srcs = [s for s in _srcs if Path(s).is_dir()]
print(f"cache sources: {[str(s) for s in _srcs]}")


def get_cache(fname, builder):
    for d in _srcs:
        if (d / fname).exists():
            if not (CACHE / fname).exists():
                (CACHE / fname).write_bytes((Path(d) / fname).read_bytes())
            return pd.read_csv(CACHE / fname), str(d)
    df = builder()
    df.to_csv(CACHE / fname, index=False)
    return df, "built here"


def _blur_rgb(rgb, sigma):
    if sigma is None or sigma <= 0.02:
        return rgb
    return ndi.gaussian_filter(rgb, sigma=(sigma, sigma, 0), mode="reflect")


def _extract_hand(sub_tiles, sigma_fn=None):
    recs = []
    for i, t in enumerate(sub_tiles.itertuples(index=False)):
        rgb = np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB"), float)
        s = 0.0 if sigma_fn is None else sigma_fn(t.camera)
        d = tile_features(_blur_rgb(rgb, s), TARGET_PPM)
        d["tile_path"] = t.tile_path
        recs.append(d)
        if (i + 1) % 500 == 0:
            print(f"    hand {i+1}/{len(sub_tiles)}")
    return pd.DataFrame(recs)


tf0, _hsrc = get_cache(f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv",
                       lambda: _extract_hand(tiles))
tf = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
            "soil_fraction"]].merge(tf0, on="tile_path", how="left")
assert len(tf) == len(tiles) and not tf[list(CFG.control_feats)].isna().any().any()
print(f"hand-built control features: {tf0.shape} from {_hsrc}")

# ---- code cell 14 ----
print(" >>> cell 14", flush=True)
"""Cell F3 - the resolution-matched blur, derived from the manifests exactly as E1/E3 did.

These views exist only for the report-only ruler in section N. Nothing in the gate and
nothing in the submission reads them.
"""
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
TRAIN_TILES = tiles[tiles.split == "train"].copy()


def _sig_for(camera, tgt):
    return RES_SIGMA.get((camera.split(" ")[0], tgt), 0.0)


for tgt, tag in zip(CFG.res_match_targets, VIEWS[1:]):
    fkey = f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}_iPhone{tgt.split()[-1]}.csv"
    _, s = get_cache(fkey, lambda tt=TRAIN_TILES, t=tgt:
                     _extract_hand(tt, lambda c, t=t: _sig_for(c, t)))
    print(f"  view '{tag}' hand features from {s}")

# ---- code cell 15 ----
print(" >>> cell 15", flush=True)
"""Cell F4 - load E2's tile embeddings instead of computing them.

Replaces E2's F4 entirely. E2 ran ~20,000 frozen forward passes over four crops and two
backbones to build these arrays; E3 reads them from disk. Every array was already provenance-
checked in C3, and the loader re-checks per call, so a wrong file cannot reach the model.

EMB keys match E2's (backend, crop, view) so the aggregation code below is the same code.
"""
EMB = {}
for _b, _c, _v in estore.REQUIRED:
    rec = estore.load_one(EMBED_STORE, _b, _c, _v, META, CFG.tile_size_px, CONFIG_HASH,
                          CFG.embed_seed)
    EMB[(_b, _c, _v)] = (rec["arr"], rec["names"], "E2 cache")
print(f"loaded {len(EMB)} embedding tables")
for _k in sorted(EMB, key=str):
    print(f"  {str(_k):34s} {EMB[_k][0].shape}")

_real = EMB[("dinov2", 256, "real")][0]
assert np.allclose(np.linalg.norm(_real, axis=1), 1.0, atol=1e-6), (
    "DINOv2 rows are not L2-normalised - these are not the arrays backbones.embed_tiles wrote")
_dv = [k for k in EMB if k[0] == "dinov2"]
_assert = {k[1] for k in _dv} == {256, 128, 64, 32}
assert _assert, "not every crop of the sweep is present, so the sensitivity arms would be wrong"
print("crop-256 rows are unit-norm and all four crops are present")

# ---- code cell 16 ----
print(" >>> cell 16", flush=True)
"""Cell G1 - tile -> image -> soil for the base arms, then the fused and permuted arms.

E2's build_arm and tile_matrix, with the torch surface replaced by fusion.py's column names.
Every arm ends up in SOIL/IMG for all three views, because the reused mechanism and CAM+RES
cells iterate views unconditionally.
"""
ECOLS = fu.emb_cols(CFG.embed_dim)
ARM_FEATS = {"M1": list(CFG.control_feats)}
for _b in ("dinov2", "vit_random"):
    for _c in CFG.crops:
        if (_b, _c, "real") in EMB:
            ARM_FEATS[f"{_b}@c{_c}"] = ECOLS
SOIL, IMG = {}, {}


def tile_matrix(arm, view):
    """A tile_path-indexed numeric DataFrame with exactly ARM_FEATS[arm] columns."""
    if arm == "M1":
        cols = list(CFG.control_feats)
        if view == "real":
            f = tf
        else:
            f = pd.read_csv(CACHE / f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}"
                            f"_iPhone{view[-2:]}.csv")
        out = f.drop_duplicates("tile_path").set_index("tile_path")[cols]
    else:
        backend, crop = arm.split("@c")
        arr, names, _src = EMB[(backend, int(crop), view)]
        out = pd.DataFrame(arr, index=names, columns=ARM_FEATS[arm])
    assert not out.index.duplicated().any(), f"{arm}/{view}: duplicated tile_path"
    return out


def build_arm(arm):
    for view in VIEWS:
        backend_crop = None
        if arm != "M1":
            b, c = arm.split("@c")
            backend_crop = (b, int(c), view)
            if view != "real" and backend_crop not in EMB:
                continue
        Mat = tile_matrix(arm, view)
        sub = GOOD if view == "real" else GOOD[GOOD.split == "train"]
        sub = sub[sub.tile_path.isin(Mat.index)]
        d = sub[["tile_path", "sample_id", "split", "camera_fam",
                 "parent_image_path"]].merge(
            Mat.rename_axis("tile_path").reset_index(), on="tile_path", how="inner")
        assert len(d) == len(sub), f"{arm}/{view}: lost {len(sub)-len(d)} tiles in the join"
        assert not d[ARM_FEATS[arm]].isna().to_numpy().any(), f"{arm}/{view}: NaN features"
        im = d.groupby(["sample_id", "camera_fam",
                        "parent_image_path"])[ARM_FEATS[arm]].median()
        IMG[(arm, view)] = im
        SOIL[(arm, view)] = im.groupby("sample_id")[ARM_FEATS[arm]].median()


for _base in ["M1"] + [a for a in ARM_FEATS if a != "M1"]:
    build_arm(_base)

BASE_ARMS = list(ARM_FEATS)


def build_fused(name, hand_arm, emb_arm, permute=False):
    """Concatenate two aggregated tables, per view, and register them as a first-class arm."""
    for view in VIEWS:
        h = SOIL[(hand_arm, view)]
        e = SOIL[(emb_arm, view)]
        hc, ec = list(CFG.control_feats), ECOLS
        if view != "real":
            # res views cover train soils only; both blocks must still be the same soils
            common = h.index.intersection(e.index)
            h, e = h.loc[common], e.loc[common]
        if permute:
            e, _vec = fu.permute_rows(e, CFG.permutation_seed)
        SOIL[(name, view)] = fu.fuse(h, e, hc, ec)
        if (hand_arm, view) in IMG and (emb_arm, view) in IMG:
            hi, ei = IMG[(hand_arm, view)], IMG[(emb_arm, view)]
            if view != "real":
                common = hi.index.intersection(ei.index)
                hi, ei = hi.loc[common], ei.loc[common]
            if permute:
                ei, _ = fu.permute_rows(ei, CFG.permutation_seed)
            IMG[(name, view)] = fu.fuse(hi, ei, hc, ec)
    ARM_FEATS[name] = list(CFG.control_feats) + ECOLS


build_fused("fuse@c256", "M1", "dinov2@c256")
build_fused("shuf@c256", "M1", "dinov2@c256", permute=True)
build_fused("fr@c256", "M1", "vit_random@c256")
for _c in (128, 64, 32):
    if f"dinov2@c{_c}" in ARM_FEATS:
        build_fused(f"fuse@c{_c}", "M1", f"dinov2@c{_c}")

ARMS_LIST = [a for a in ["M1", "dinov2@c256", "vit_random@c256", "fuse@c256", "shuf@c256",
                         "fr@c256", "fuse@c128", "fuse@c64", "fuse@c32"] if a in ARM_FEATS]
missing = [a for a in ARMS_LIST_EXPECTED if a not in ARMS_LIST]
assert not missing, f"expected arms absent: {missing}"
print(f"\narms ({len(ARMS_LIST)}): {ARMS_LIST}")

print("\nsoil-level tables:")
for k in sorted(SOIL, key=str):
    print(f"  {str(k):34s} {SOIL[k].shape}")

_f = SOIL[("fuse@c256", "real")]
_s = SOIL[("shuf@c256", "real")]
print(f"\nfused width {SOIL[('fuse@c256','real')].shape[1]} = 12 hand + {CFG.embed_dim} emb")
assert _f.shape[1] == _s.shape[1] == 396
assert not np.allclose(_f[ECOLS].to_numpy(), _s[ECOLS].to_numpy()), (
    "SHUF's embedding block is IDENTICAL to the real one - the permutation did nothing and "
    "the control would be measuring itself. This is the failure E3's contract test was "
    "written to catch.")
assert np.allclose(_f[list(CFG.control_feats)].to_numpy(),
                   _s[list(CFG.control_feats)].to_numpy()), "the hand blocks differ"
for _c, _lbl in ((float(_f[ECOLS].to_numpy().std(0).mean()), "real"),
                 (float(_s[ECOLS].to_numpy().std(0).mean()), "shuf")):
    print(f"  embedding block mean column sd, {_lbl}: {_c:.6f}")
assert abs(float(_f[ECOLS].to_numpy().std(0).mean())
           - float(_s[ECOLS].to_numpy().std(0).mean())) < 1e-12, "permutation changed column sd"
print("SHUF preserves every column statistic and destroys only the correspondence.")
pd.DataFrame([{"arm": a, "n_feat": len(ARM_FEATS[a]),
               "kind": ("control" if a == "M1" else
                        "backbone" if a in ("dinov2@c256", "vit_random@c256") else
                        "permuted control" if a == "shuf@c256" else
                        "capacity control" if a == "fr@c256" else
                        "fusion" ),
               "gated": a == GATED} for a in ARMS_LIST
              ]).to_csv(OUT_DIR / "arms_config.csv", index=False)

# ---- code cell 17 ----
print(" >>> cell 17", flush=True)
"""Cell G2 - prove the control arm IS Experiment 3's matrix."""
S = SOIL[("M1", "real")]
_chk = e3_soil                      # already indexed by sample_id in cell C1
_drift = {c: float(np.nanmax(np.abs(S[c] - _chk[c].reindex(S.index)))) for c in CFG.control_feats}
_mx = max(_drift.values())
assert len(S) == len(e3_soil) == 34, f"row count {len(S)} vs E3's {len(e3_soil)}"
print(f"max absolute drift vs Experiment 3 across {len(CFG.control_feats)} features: {_mx:.3e}")
assert _mx < 1e-6, f"control drifted from E3 by {_mx} - the experiments are not comparable"
print("REPRODUCED: the control arm is Model 1 E3's feature matrix.")
pd.concat([S, e3_soil[["split"]].reindex(S.index)],
          axis=1).to_csv(OUT_DIR / "features_soil_M1_control.csv")

fams = e3_fam.set_index("sample_id").cv_family
assert set(fams.index) == set(tr_ids)
GROUPS = sorted(fams.unique())
e3_fam.to_csv(OUT_DIR / "cv_families.csv", index=False)
print(f"{len(GROUPS)} CV families over {len(fams)} soils, copied unchanged from E1/E2/E3.")

# ---- code cell 18 ----
print(" >>> cell 18", flush=True)
"""Cell H1 - model primitives, byte-for-byte Model 1's."""
rng = np.random.default_rng(SEED)


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


XREAL = {a: SOIL[(a, "real")] for a in ARMS_LIST}
Ymap = {s: Y[i] for i, s in enumerate(tr_ids)}
TR_ARR = np.array(tr_ids)


def logo_errors(arm, alpha, eval_family):
    X = XREAL[arm].loc[tr_ids, ARM_FEATS[arm]].to_numpy(float)
    keep = np.array([fams[s] != eval_family for s in tr_ids])
    assert not any(fams[s] == fams[t] for s in TR_ARR[~keep] for t in TR_ARR[keep]), \
        "a same-family soil leaks the held-out label"
    P = fit_predict(X[keep], Y[keep], X[~keep], alpha)
    ev = TR_ARR[~keep]
    return {ev[i]: emd_pair(P[i], Ymap[ev[i]]) for i in range(len(ev))}


print(f"arms scored: {ARMS_LIST}")
print(f"{len(CFG.alpha_grid)} alphas x {len(GROUPS)} folds = "
      f"{len(CFG.alpha_grid)*len(GROUPS)} fits per arm")

# ---- code cell 19 ----
print(" >>> cell 19", flush=True)
"""Cell H2 - the structural error table, the sweep, and the nested procedure score."""
LERR, SWEEP, NESTED, per_soil, fold_rows = {}, {}, {}, {}, []
_t0 = time.time()
for _a in ARMS_LIST:
    LERR[_a] = {(al, F): logo_errors(_a, al, F) for al in CFG.alpha_grid for F in GROUPS}
    sw = {al: float(np.mean([e for F in GROUPS for e in LERR[_a][(al, F)].values()]))
          for al in CFG.alpha_grid}
    SWEEP[_a] = sw
    _or = min(sw.values()); _oa = min(sw, key=sw.get)
    _ok = [al for al in CFG.alpha_grid if sw[al] <= _or + CFG.ORDINAL_FLOOR]
    _pa = max(_ok)
    NESTED[_a] = {"arm": _a, "n_feat": len(ARM_FEATS[_a]),
                  "crop": CFG.tile_size_px if _a == "M1" else int(_a.split("@c")[1]),
                  "backend": "M1" if _a == "M1" else _a.split("@c")[0],
                  "oracle_alpha": _oa, "oracle_emd": _or, "parsimony_alpha": _pa,
                  "parsimony_emd": sw[_pa],
                  "parsimony_is_grid_max": bool(_pa == CFG.alpha_grid[-1]),
                  "grid_spread": sw[max(sw, key=sw.get)] - _or}
    vec = {}
    for F in GROUPS:
        inner = [G for G in GROUPS if G != F]
        sc = {al: float(np.mean([e for G in inner for e in LERR[_a][(al, G)].values()]))
              for al in CFG.alpha_grid}
        ba = min(sc, key=sc.get)
        errs = LERR[_a][(ba, F)]
        vec.update(errs)
        fold_rows.append(dict(arm=_a, fold_family=F, n_eval_soils=len(errs),
                              alpha_selected=ba, fold_emd=float(np.mean(list(errs.values()))),
                              worst_soil=max(errs, key=errs.get),
                              worst_emd=float(max(errs.values()))))
    per_soil[_a] = vec
    print(f"  {_a:22s} nested {np.mean(list(vec.values())):6.2f}  "
          f"({time.time()-_t0:.0f}s cumulative)", flush=True)
FOLDS = pd.DataFrame(fold_rows)
FOLDS.to_csv(OUT_DIR / "cv_per_fold.csv", index=False)
RES = pd.DataFrame(per_soil)
RES.index.name = "sample_id"
RES["cv_family"] = [fams[s] for s in RES.index]
RES["D50_mm"] = [D50_MM[tr_ids.index(s)] for s in RES.index]
RES["soil_group"] = [cg.soil_group(d) for d in RES["D50_mm"]]
RES.to_csv(OUT_DIR / "cv_per_soil.csv")

# ---- code cell 20 ----
print(" >>> cell 20", flush=True)
"""Cell H3 - THE REGRESSION CHECK. All three base arms must reproduce E2.

Three arms, not one, because E3 depends on all three: M1 anchors the comparison, D is the
block being fused, and R is what the FR arm and the capacity reading depend on. Passing this
check is also the only proof that the 15 harvested .npy files really are the arrays E2 scored.

Two levels, in descending strength:
  1. per-soil vectors against E2's own cv_per_soil.csv (24 numbers per arm)
  2. arm means against E2's pinned constants, if that file is not attached
"""
print("regression check vs Model 2 E2\n")
_map = {"M1": "M1", "dinov2@c256": "dinov2@c256", "vit_random@c256": "vit_random@c256"}
_pin = {"M1": "M1", "dinov2@c256": "D", "vit_random@c256": "R"}
_have = not e2_cv.empty
print(f"  check level: {E2_LEVEL}\n")
_ok_all = True
for _arm, _col in _map.items():
    a = np.array([per_soil[_arm][s] for s in tr_ids])
    if _have:
        ref = e2_cv[_col].reindex(tr_ids).to_numpy(float)
        mean_gap = float(abs(a.mean() - ref.mean()))
        max_gap = float(np.abs(a - ref).max())
        tail = f"max per-soil gap {max_gap:.4f} over {len(ref)} soils"
    else:
        mean_gap = float(abs(a.mean() - E2_PINNED[_pin[_arm]]))
        tail = "mean-only check"
    good = mean_gap <= CFG.REGRESSION_TOL
    _ok_all &= good
    print(f"  [{'PASS' if good else '**FAIL**'}] {_arm:16s} vs E2:  "
          f"E3 {a.mean():7.3f}  mean gap {mean_gap:.4f}  {tail}")
assert _ok_all, (
    f"a base arm does not reproduce Model 2 E2 within {CFG.REGRESSION_TOL} EMD. Then E3's "
    "harvested embeddings or its aggregation are not E2's, and every fusion number below "
    "would be attributable to that difference rather than to the feature matrix. Fix the "
    "inputs; do NOT interpret the gap as a result.")
print(f"\nPASS: E3 reproduces E2's three base arms ({E2_LEVEL}). Everything below is a")
print("feature-composition effect and nothing else.")

# ---- code cell 21 ----
print(" >>> cell 21", flush=True)
"""Cell I1 - the primary result table over all nine arms."""
# NESTED is keyed by arm AND carries an "arm" field, so after .T the arm name exists both as
# the index and as a column; reset_index(names=...) would raise "cannot insert arm, already
# exists". Drop the redundant column first, then promote the index.
NESTED_DF = pd.DataFrame(NESTED).T
NESTED_DF.index.name = "arm"
NESTED_DF = NESTED_DF.drop(columns=["arm"]).reset_index()
NESTED_DF["label"] = [LABEL.get(a, a) for a in NESTED_DF.arm]
NESTED_DF["nested_in_domain"] = [float(np.mean(list(per_soil[a].values())))
                                 for a in NESTED_DF.arm]
NESTED_DF["median"] = [float(np.median(list(per_soil[a].values()))) for a in NESTED_DF.arm]
NESTED_DF["worst"] = [float(max(per_soil[a].values())) for a in NESTED_DF.arm]
NESTED_DF = NESTED_DF.sort_values("nested_in_domain")
print("nested in-domain LOGO-CV by arm (lower is better):\n")
for r in NESTED_DF.itertuples(index=False):
    star = "  <-- THE GATED ARM" if r.arm == GATED else ""
    print(f"  {r.label:9s} {str(r.arm):16s} n_feat {r.n_feat:4d}  "
          f"nested {r.nested_in_domain:7.2f}  oracle alpha {r.oracle_alpha:g}{star}")
print(f"\n  control M1 = {NESTED_DF[NESTED_DF.arm=='M1'].nested_in_domain.iloc[0]:.2f}   "
      f"no-image floor {FLOOR_NO_IMAGE:.2f}   rank-3 ceiling 7.35")
NESTED_DF.to_csv(OUT_DIR / "sweep_results.csv", index=False)
assert len(NESTED_DF) == 9, f"expected nine arms, got {len(NESTED_DF)}"
_sat = NESTED_DF[NESTED_DF.parsimony_is_grid_max]
if len(_sat):
    print(f"  NOTE: parsimony alpha sits at the grid top for {list(_sat.arm)}")

# ---- code cell 22 ----
print(" >>> cell 22", flush=True)
"""Cell J1 - the per-soil difference profile for the gated arm.

E2's J1 computed a Spearman rho against crop px; E3 has no crop variable to trend, so that
cell is removed rather than repurposed. What replaces it is the question a mean cannot answer:
is a gain broad-based across soils or produced by two outliers?
"""
_m1 = np.array([per_soil["M1"][s] for s in tr_ids])
_fu = np.array([per_soil[GATED][s] for s in tr_ids])
_diff = _m1 - _fu                                  # positive = fusion better
_order = np.argsort(-_diff)
JROWS = []
for s, d in sorted(zip(tr_ids, _diff), key=lambda t: -t[1]):
    JROWS.append(dict(sample_id=s, m1=per_soil["M1"][s], fuse=per_soil[GATED][s],
                      gain=d, cv_family=fams[s], soil_group=cg.soil_group(
                          D50_MM[tr_ids.index(s)])))
JD = pd.DataFrame(JROWS)
print("per-soil gain of FUSE@256 over M1, best first (positive = fusion better):\n")
for r in JD.itertuples(index=False):
    print(f"  {r.sample_id}  M1 {r.m1:7.2f}  FUSE {r.fuse:7.2f}  gain {r.gain:+8.2f}  "
          f"{r.cv_family} {r.soil_group}")
_n_imp = int((_diff > 0).sum()); _n_wor = int((_diff < 0).sum())
_top2 = float(np.sort(_diff)[-2:].sum())
print(f"\n  improved {len(_diff)-_n_wor-_n_imp}+{_n_imp} soils, worsened {_n_wor}")
print(f"  mean gain {float(_diff.mean()):+.3f} EMD; the two best soils contribute "
      f"{_top2:+.2f} of that")
if abs(float(_diff.mean())) > 1e-9:
    print(f"  share of mean gain from the top two soils: "
          f"{100*_top2/(len(_diff)*float(_diff.mean())):.0f}%")
JD.to_csv(OUT_DIR / "per_soil_gain.csv", index=False)

# ---- code cell 23 ----
print(" >>> cell 23", flush=True)
"""Cell K1 - the leak canary, reused from E2's logic: is a gain in the wrong soils?

Magnification was E2's mechanism and it could only act on mid/coarse soils. E3's mechanism is
different - the embedding is information about grain size in general - so this table is a
diagnostic, not a gate clause. It is printed because E2 established the discipline: a gain
concentrated in the ten fines is physically hard to explain and would indicate leakage.
"""
GRP = []
for _a in ("M1", GATED, "shuf@c256", "dinov2@c256", "fr@c256"):
    v = np.array([per_soil[_a][s] for s in tr_ids])
    g = np.array([cg.soil_group(D50_MM[tr_ids.index(s)]) for s in tr_ids])
    GRP.append(dict(arm=_a, label=LABEL.get(_a, _a), overall=float(v.mean()),
                    fines=float(v[g == "fines"].mean()) if (g == "fines").any() else np.nan,
                    mid=float(v[g == "mid"].mean()) if (g == "mid").any() else np.nan,
                    coarse=float(v[g == "coarse"].mean()) if (g == "coarse").any() else np.nan,
                    n_fines=int((g == "fines").sum()), n_mid=int((g == "mid").sum()),
                    n_coarse=int((g == "coarse").sum())))
GR = pd.DataFrame(GRP)
print("in-domain EMD by D50 group (fines < 0.11 mm, mid <= 1.5, coarse > 1.5):\n")
for r in GR.itertuples(index=False):
    print(f"  {r.label:9s} overall {r.overall:7.2f}  fines {r.fines:7.2f} (n={r.n_fines})  "
          f"mid {r.mid:7.2f} (n={r.n_mid})  coarse {r.coarse:7.2f} (n={r.n_coarse})")
_num = ["overall", "fines", "mid", "coarse"]
_d = (GR[GR.arm == GATED].iloc[0][_num] - GR[GR.arm == "M1"].iloc[0][_num])
print(f"\n  FUSE minus M1: fines {_d.fines:+.2f}  mid {_d.mid:+.2f}  coarse {_d.coarse:+.2f} "
      "(negative = fusion better)")
print("  A gain concentrated in the fines would be physically suspicious, as established in")
print("  E2 section 6; here it is reported, and it is NOT a gate clause.")
GR.to_csv(OUT_DIR / "per_group_effects.csv", index=False)

# ---- code cell 24 ----
print(" >>> cell 24", flush=True)
"""Cell L1 - the four paired contrasts that matter, each with a soil bootstrap CI."""
PAIRED_ON = sorted(set.intersection(*[set(per_soil[a]) for a in ARMS_LIST]))
MVEC = {a: np.array([per_soil[a][s] for s in PAIRED_ON]) for a in ARMS_LIST}


def boot_ci(diff, n=CFG.n_boot):
    boots = np.empty(n)
    idx = np.arange(len(diff))
    r = np.random.default_rng(SEED)
    for b in range(n):
        boots[b] = diff[r.choice(idx, size=len(idx), replace=True)].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


CON = []
for _a, _b, _q in ((GATED, "M1", "FUSE - M1"), ("shuf@c256", "M1", "SHUF - M1"),
                   (GATED, "shuf@c256", "FUSE - SHUF"), (GATED, "dinov2@c256", "FUSE - D"),
                   ("fr@c256", "M1", "FR - M1"), (GATED, "vit_random@c256", "FUSE - R")):
    d = MVEC[_a] - MVEC[_b]
    lo, hi = boot_ci(d)
    CON.append(dict(contrast=_q, arm_a=_a, arm_b=_b, estimate=float(d.mean()),
                    ci_lo=lo, ci_hi=hi,
                    significant=bool((lo > 0) or (hi < 0)),
                    a_better=bool(d.mean() < 0),
                    wins_a=int((d < 0).sum()), wins_b=int((d > 0).sum()),
                    n_soils=len(d)))
CONEFF = pd.DataFrame(CON)
print("paired soil-bootstrap contrasts (negative estimate = arm A better):\n")
for r in CONEFF.itertuples(index=False):
    sig = "SIGNIFICANT" if r.significant else "not significant"
    print(f"  {r.contrast:15s} {r.estimate:+8.2f}  CI [{r.ci_lo:+8.2f}, {r.ci_hi:+8.2f}]  "
          f"{sig:16s} wins {r.wins_a:2d}/{r.n_soils}")
CONEFF.to_csv(OUT_DIR / "fusion_effects.csv", index=False)
assert set(CONEFF.contrast) >= {"FUSE - M1", "SHUF - M1", "FUSE - SHUF"}

# ---- code cell 25 ----
print(" >>> cell 25", flush=True)
"""Cell M1 - camera shift and blur shift per arm, plus the cosine separation ratios."""
def r_rows(arm, view, camera):
    return IMG[(arm, view)].xs(camera, level="camera_fam").groupby("sample_id")[
        ARM_FEATS[arm]].median()


_ids = sorted(set(r_rows("M1", "real", "Motorola").index)
              & set(r_rows("M1", "real", "Samsung").index))
MECH, SEP = [], []
for _a in ARMS_LIST:
    FE = ARM_FEATS[_a]
    X = XREAL[_a].loc[tr_ids, FE]
    sd = X.std().replace(0, np.nan)
    A, B = r_rows(_a, "real", "Motorola"), r_rows(_a, "real", "Samsung")
    sh = sorted(set(A.index) & set(B.index) & set(_ids))
    cam = ((A.loc[sh, FE] - B.loc[sh, FE]).abs() / sd).mean()
    C = r_rows(_a, "res14", "Motorola")
    sh2 = sorted(set(A.index) & set(C.index) & set(_ids))
    sharp = ((A.loc[sh2, FE] - C.loc[sh2, FE]).abs() / sd).mean()
    MECH.append(dict(arm=_a, n_feat=len(FE), camera_shift_z=float(cam.mean()),
                     sharpness_shift_z=float(sharp.mean())))

    def _cos(u, v):
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
    within = float(np.mean([1 - _cos(A.loc[s].to_numpy(float), B.loc[s].to_numpy(float))
                            for s in sh]))
    betw = float(np.mean([1 - _cos(A.loc[s].to_numpy(float), A.loc[t].to_numpy(float))
                          for i, s in enumerate(sh) for t in sh[i + 1:]]))
    blur = float(np.mean([1 - _cos(A.loc[s].to_numpy(float), C.loc[s].to_numpy(float))
                          for s in sh2]))
    degenerate = betw < 1e-6
    SEP.append(dict(arm=_a, camera_distance=within, soil_distance=betw,
                    camera_over_soil=(np.nan if degenerate else within / betw),
                    blur_over_soil=(np.nan if degenerate else blur / betw),
                    degenerate=degenerate))
MECH_DF = pd.DataFrame(MECH).set_index("arm")
SEP_DF = pd.DataFrame(SEP).set_index("arm")
MECH_DF.to_csv(OUT_DIR / "mechanism_table.csv")
SEP_DF.to_csv(OUT_DIR / "embedding_separation.csv")
SHOW = MECH_DF.join(SEP_DF[["camera_over_soil", "blur_over_soil", "degenerate"]])
print(SHOW.round(4).to_string())
print("\n  E1 reference: D camera/soil 0.209, blur/soil 0.029; M1 1.105 and 0.818.")
print("  'degenerate' means between-soil distance is ~0, so the ratio is a quotient of")
print("  noise and is reported as NaN rather than as a number that looks interpretable.")
_d = SEP_DF.loc[[i for i in SEP_DF.index if i.startswith("dinov2")], "camera_over_soil"].dropna()
if len(_d):
    print(f"\n  P6: D stays below 0.35 at every crop -> "
          f"{'HOLDS' if _d.max() < 0.35 else 'FAILS'} (max {_d.max():.3f} over {len(_d)} crops)")
else:
    print("\n  P6: not evaluated - no dinov2 arms in this run")

# ---- code cell 26 ----
print(" >>> cell 26", flush=True)
"""Cell N1 - CAM+RES per arm at the arm's own parsimony alpha."""
CAM_RES, CAM_ONLY = [], []
for _src in CFG.holdout_cameras:
    _tgt = [c for c in CFG.holdout_cameras if c != _src][0]
    for _v in VIEWS:
        _c = dict(name=f"{_src[:3]}>{_tgt[:3]}:{_v}", src=_src, tgt=_tgt, view=_v)
        (CAM_ONLY if _v == "real" else CAM_RES).append(_c)
ALL_CONDS = CAM_ONLY + CAM_RES
for _c in ALL_CONDS:
    assert set(r_rows("M1", _c["view"], _c["tgt"]).index) >= set(_ids)
print(f"CAM+RES conditions ({len(CAM_RES)}), CAM ({len(CAM_ONLY)}), "
      f"{len(_ids)} dual-camera soils")


def cond_errors(arm, alpha, cond, eval_family):
    A = r_rows(arm, "real", cond["src"]); B = r_rows(arm, cond["view"], cond["tgt"])
    ids = [s for s in _ids if s in A.index and s in B.index]
    keep = np.array([fams[s] != eval_family for s in ids])
    if keep.all() or not (~keep).any():
        return {}
    FE = ARM_FEATS[arm]
    P = fit_predict(A.loc[ids, FE].to_numpy(float)[keep],
                    np.array([Ymap[s] for s in ids])[keep],
                    B.loc[ids, FE].to_numpy(float)[~keep], alpha)
    return {ids[j]: emd_pair(P[i], Ymap[ids[j]]) for i, j in enumerate(np.where(~keep)[0])}


RULER = []
for _a in ARMS_LIST:
    al = NESTED[_a]["parsimony_alpha"]
    cr = [e for F in GROUPS for c in CAM_RES for e in cond_errors(_a, al, c, F).values()]
    cm = [e for F in GROUPS for c in CAM_ONLY for e in cond_errors(_a, al, c, F).values()]
    RULER.append(dict(arm=_a, alpha=al, camres=float(np.mean(cr)) if cr else np.nan,
                      cam=float(np.mean(cm)) if cm else np.nan,
                      in_domain=float(np.mean(list(per_soil[_a].values())))))
RU = pd.DataFrame(RULER).set_index("arm")
RU.to_csv(OUT_DIR / "camres_report_only.csv")
print(RU.round(2).to_string())
print("\n  NOT USED TO SELECT. E1's probe is the calibration point that matters: CAM+RES")
print("  said 47.08 for arm D and the leaderboard returned 60.56.")

# ---- code cell 27 ----
print(" >>> cell 27", flush=True)
"""Cell O1 - the gate. G1, G2, G3 + PROVISO + VALID + COMPLETE."""
CLAUSES = []


def clause(text, ok, detail=""):
    CLAUSES.append(dict(clause=text, passed=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}" + (f"   {detail}" if detail else ""))


COMPLETE = {"M1", "dinov2@c256", "vit_random@c256"} <= set(per_soil)
EVALUABLE = bool(COMPLETE and GATED in per_soil and not CONEFF.empty)


def _c(contrast):
    r = CONEFF[CONEFF.contrast == contrast]
    return r.iloc[0] if len(r) else None


# Two corrections made AFTER run 1, both restoring this cell to the clause text frozen in
# Model 2/Model 2 Experiment 3/instructions.txt and model_plan_m2e3.md section 8. No clause,
# threshold or comparator was changed; the code simply disagreed with the spec.
#   (a) _cfs was hard-wired to None, so G2 evaluated nothing. The real contrast is FUSE-SHUF.
#   (b) VALID was written as (SHUF - M1) < 0, which is SHUF BEATING M1 - the exact inverse of
#       the spec, "VALID: in_domain(SHUF@256) > in_domain(M1)". Run 1 therefore mislabelled a
#       healthy instrument as INSTRUMENT INVALID.
_cfm = _c("FUSE - M1")
_cfsm = _c("SHUF - M1")
_cfs = _c("FUSE - SHUF")
_cfd = _c("FUSE - D")
_idm = {a: float(np.mean(list(per_soil[a].values()))) for a in ARMS_LIST}
_alpha = NESTED[GATED]["oracle_alpha"]
PROVISO = bool(CFG.alpha_grid[0] < _alpha < CFG.alpha_grid[-1])
VALID = bool(_cfsm is not None and _cfsm.estimate > 0)      # SHUF must be WORSE than M1

G1 = bool(_cfm is not None and _cfm.estimate < 0 and _cfm.significant)
G2 = bool(_cfs is not None and _cfs.estimate < 0 and _cfs.significant)
G3 = bool(_cfd is not None and _cfd.estimate < 0)

clause(f"G1 in_domain(FUSE@{CFG.gated_crop}) < in_domain(M1) with CI excluding zero",
       G1, f"estimate {_cfm.estimate:+.2f} CI [{_cfm.ci_lo:+.2f}, {_cfm.ci_hi:+.2f}]"
       if _cfm is not None else "contrast unavailable")
clause("G2 FUSE beats SHUF with CI excluding zero (gain exceeds what noise columns achieve)",
       G2, f"estimate {_cfs.estimate:+.2f} CI [{_cfs.ci_lo:+.2f}, {_cfs.ci_hi:+.2f}]"
       if _cfs is not None else "contrast unavailable")
clause("G3 FUSE beats D alone (fusion beats either family alone)", G3,
       f"estimate {_cfd.estimate:+.2f}" if _cfd is not None else "contrast unavailable")
clause("PROVISO FUSE's oracle alpha is strictly inside the grid", PROVISO,
       f"alpha {_alpha:g} on a grid of {len(CFG.alpha_grid)}")
clause("VALID SHUF does not beat M1 (the CV is not simply rewarding capacity)", VALID,
       f"SHUF - M1 {_cfsm.estimate:+.2f}" if _cfsm is not None else "unavailable")
clause("COMPLETE M1, D and R all present (Model 2 capacity-control rule)", COMPLETE,
       f"arms: {sorted(per_soil)}")

SUBMIT = bool(G1 and G2 and G3 and PROVISO and VALID and COMPLETE and EVALUABLE)
if not PROVISO:
    VERDICT = "INCONCLUSIVE (alpha at the grid edge)"
elif not VALID:
    VERDICT = "INSTRUMENT INVALID (noise columns beat the control)"
elif not COMPLETE:
    VERDICT = "NOT EVALUABLE (a base arm is missing)"
elif not G1:
    VERDICT = "REFUTED (fusion does not beat the hand features)"
elif not G2:
    VERDICT = "CAPACITY-NOT-INFORMATION (gain no larger than the permuted block's)"
elif not G3:
    VERDICT = "REFUTED (fusion does not beat D alone)"
else:
    VERDICT = "SUBMIT"

GAIN = float(-_cfm.estimate) if _cfm is not None else float("nan")
GATE = dict(submit=SUBMIT, verdict=VERDICT, evaluable=EVALUABLE, gated_arm=GATED,
            clauses=CLAUSES, in_domain=_idm,
            fusion_gain_vs_M1=GAIN,
            practically_meaningful_band=CFG.ORDINAL_FLOOR,
            oracle_alpha_fuse=_alpha, proviso=PROVISO, valid=VALID,
            regression_level=E2_LEVEL,
            trend_note="E3 has no crop trend; the gated arm is fixed at crop "
                       f"{CFG.gated_crop}")
json.dump(GATE, open(OUT_DIR / "gate.json", "w"), indent=2, default=str)
print(f"\nGATE -> {VERDICT}   submit={SUBMIT}")
print(f"  fusion gain over M1: {GAIN:+.2f} EMD; the band the ruler can resolve is "
      f"{CFG.ORDINAL_FLOOR}")
if not SUBMIT:
    print("  This is the predicted outcome (P3) and it is a complete answer, not an")
    print("  unfinished run. If G1 failed, Model 2 closes at three experiments: E1 tested")
    print("  replacement, E2 tested magnification, E3 tested complementarity.")

# ---- code cell 28 ----
print(" >>> cell 28", flush=True)
"""Cell P1 - fit the submission arm and predict the 10 test soils."""
SUB_ARM = GATED if SUBMIT else "M1"
WRITES_FILE = bool(SUBMIT or (CFG.probe_arm and CFG.probe_arm in per_soil))
PLUMBING_ONLY = not WRITES_FILE
FE = ARM_FEATS[SUB_ARM]
S = SOIL[(SUB_ARM, "real")]
alpha_f = NESTED[SUB_ARM]["parsimony_alpha"]
Xtr = S.loc[tr_ids, FE].to_numpy(float)
Xte = S.loc[te_ids, FE].to_numpy(float)
assert Xtr.shape == (24, len(FE)) and Xte.shape == (10, len(FE))
assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()
P_test = fit_predict(Xtr, Y, Xte, alpha_f)
assert P_test.shape == (10, 11)
_sat = 100.0 * float(np.mean((P_test <= 1e-6) | (P_test >= 100)))
_e3p = e3_sub[TARGET_COLS].to_numpy(float)
print({"gate": "SUBMISSION - the gate fired",
       "probe": "DECLARED PROBE - the gate did not fire",
       "plumb": "PLUMBING CHECK on the control arm (no file will be written)"}
      ["gate" if SUBMIT else ("probe" if WRITES_FILE else "plumb")])
print(f"  arm {SUB_ARM}: {len(FE)} features, crop {CFG.tile_size_px}px "
      f"(patch {cg.patch_mm(CFG.tile_size_px, TARGET_PPM):.4f} mm), alpha {alpha_f:g}")
print(f"  columns pinned at 0/100: this {_sat:.1f}% (real labels 27%)")
print(f"  mean EMD of predictions vs E3's submitted curves: "
      f"{float(np.mean([emd_pair(a, b) for a, b in zip(P_test, _e3p)])):.2f}")
if NESTED[SUB_ARM]["parsimony_is_grid_max"]:
    print("  NOTE: alpha is the top of the grid, so these curves are heavily shrunk.")
print(f"  probe_arm = {CFG.probe_arm!r}: E3 pre-authorises no probe.")

# ---- code cell 29 ----
print(" >>> cell 29", flush=True)
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

# ---- code cell 30 ----
print(" >>> cell 30", flush=True)
"""Cell Q1 - the contrast figure: gain per soil, coloured by D50 group."""
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(range(len(JD)), JD.gain,
          color=["#b33" if v > 0 else "#369" for v in JD.gain])
ax[0].axhline(0, color="k", lw=0.8)
ax[0].axhline(float(-_cfm.estimate), color="g", ls="--", lw=1.2,
              label=f"mean {_cfm.estimate:+.2f}")
ax[0].set_xlabel("soil (sorted by gain)")
ax[0].set_ylabel("M1 - FUSE  (positive = fusion better)")
ax[0].set_title("per-soil gain of the 384-column block")
ax[0].legend(frameon=False, fontsize=8)
_x = np.arange(len(GR))
ax[1].bar(_x - 0.2, GR.overall, 0.4, label="overall")
ax[1].bar(_x + 0.2, GR.fines, 0.4, label="fines")
ax[1].set_xticks(_x); ax[1].set_xticklabels(GR.label, rotation=30, ha="right")
ax[1].set_ylabel("in-domain EMD"); ax[1].set_title("arms by size group")
ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
print("The left panel is the one to read: a mean gain carried by two soils is not a result.")

# ---- code cell 31 ----
print(" >>> cell 31", flush=True)
"""Cell Q2 - the base arms against each other, per soil."""
fig, ax = plt.subplots(figsize=(7, 5))
for _a, _col in (("M1", "#369"), ("dinov2@c256", "#b33"), ("vit_random@c256", "#888"),
                 ("fuse@c256", "#2a6"), ("shuf@c256", "#c80")):
    v = np.array([per_soil[_a][s] for s in tr_ids])
    ax.plot(np.arange(len(v)), v, marker="o", ms=3, color=_col,
            label=f"{LABEL[_a]} mean {v.mean():.2f}")
ax.set_yscale("log"); ax.set_ylabel("in-domain EMD per soil (log)")
ax.set_xlabel("soil, sorted by M1 error")
ax.set_title("E3 five key arms, per soil")
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
_shown = ["M1", "dinov2@c256", "vit_random@c256", "fuse@c256", "shuf@c256"]
_means = [float(np.mean([per_soil[a][s] for s in tr_ids])) for a in _shown]
print(f"plotted {len(tr_ids)} soils x {len(_shown)} arms; arm means span "
      f"{max(_means) - min(_means):.2f} EMD ({min(_means):.2f} to {max(_means):.2f})")
print("cell 31 executed and drew; an empty cell here would mean the plot silently skipped,")
print("which is exactly E1's failure mode.")

# ---- code cell 32 ----
print(" >>> cell 32", flush=True)
"""Cell Q3 - the submission curves, if one exists."""
if not WRITES_FILE:
    print("No submission curves: the gate did not fire and no probe is armed.")
else:
    _fig, _axes = plt.subplots(2, 5, figsize=(19, 7.2))
    for i, sid in enumerate(te_ids):
        ax = _axes.ravel()[i]
        ax.plot(DL, MEAN_CURVE, ":", c="#999", label="train mean")
        ax.plot(DL, _e3p[i], "x--", c="#dd6b20", lw=1.2, label="E3 (61.24)")
        ax.plot(DL, P_test[i], "o-", c="#2b6cb0", lw=2, label=f"E2 {SUB_ARM}")
        ax.set_title(sid, fontsize=8); ax.set_ylim(-5, 105); ax.tick_params(labelsize=7)
    _axes.ravel()[0].legend(fontsize=7)
    plt.suptitle("Model 2 E2 predicted cumulative curves vs the best Model 1 submission")
    plt.tight_layout(); plt.show()

# ---- code cell 33 ----
print(" >>> cell 33", flush=True)
"""Cell R1 - the pre-registered predictions, evaluated mechanically."""
PREDS = []


def pred(pid, text, blind, status, evidence):
    PREDS.append(dict(id=pid, prediction=text, blind="yes" if blind else "no",
                      status=status, evidence=evidence))


_c = lambda name: CONEFF[CONEFF.contrast == name]
pred("P1", f"M1, D@256 and R@256 reproduce E2's {E2_LEVEL.split('(')[0].strip()} within "
     f"{CFG.REGRESSION_TOL} EMD", False, "CONFIRMED",
     "asserted in cell H3; the notebook would have stopped otherwise")
_shuf = _c("SHUF - M1")
pred("P2", "SHUF@256 is WORSE than M1 (384 noise columns cost accuracy at n=24)", True,
     "CONFIRMED" if _shuf.iloc[0].estimate > 0 else "REFUTED",
     f"SHUF - M1 {_shuf.iloc[0].estimate:+.2f} CI [{_shuf.iloc[0].ci_lo:+.2f}, "
     f"{_shuf.iloc[0].ci_hi:+.2f}]")
_fuse = _c("FUSE - M1")
pred("P3", "FUSE@256 does NOT beat M1 with a CI excluding zero (H3 refuted)", True,
     "REFUTED - H3 SURVIVES" if (_fuse.iloc[0].estimate < 0 and _fuse.iloc[0].significant)
     else "CONFIRMED - H3 refuted",
     f"FUSE - M1 {_fuse.iloc[0].estimate:+.2f} CI [{_fuse.iloc[0].ci_lo:+.2f}, "
     f"{_fuse.iloc[0].ci_hi:+.2f}]")
pred("P4", "if FUSE beats M1, the gain is under ~2 EMD (far below the 15.0 band)", True,
     "n/a - no gain" if _fuse.iloc[0].estimate >= 0 else
     ("CONFIRMED" if -_fuse.iloc[0].estimate < 2.0 else "REFUTED"),
     f"gain {-_fuse.iloc[0].estimate:+.2f} EMD")
_S = SHOW if "SHOW" in globals() else pd.DataFrame()
_ratio = (float(_S.loc["fuse@c256", "camera_over_soil"]) if "fuse@c256" in _S.index
          else float("nan"))
pred("P5", "FUSE's camera/soil ratio lands between D's 0.209 and M1's 1.105", True,
     "not evaluated" if _ratio != _ratio else
     ("CONFIRMED" if 0.209 <= _ratio <= 1.105 else "REFUTED"),
     f"fuse camera/soil {_ratio}")
_fr = _c("FR - M1")
pred("P6", "FR@256 behaves like FUSE@256, so any gain is columns rather than pretraining",
     True, "not evaluated" if _fr.empty else
     ("CONFIRMED" if abs(_fr.iloc[0].estimate - _fuse.iloc[0].estimate) < 2.0 else "REFUTED"),
     f"FR - M1 {_fr.iloc[0].estimate:+.2f} vs FUSE - M1 {_fuse.iloc[0].estimate:+.2f}")
_sens = [a for a in ARMS_LIST if a.startswith("fuse@") and a != GATED]
_best_sens = min((float(np.mean(list(per_soil[a].values()))) for a in _sens),
                 default=float("nan"))
pred("P7", "no fused sensitivity crop beats the gated fusion at crop 256", True,
     "CONFIRMED" if _best_sens >= _idm[GATED] else "REFUTED",
     f"fuse@c256 {_idm[GATED]:.2f} vs best other {min(_sens, key=lambda a: _idm[a]) if _sens else '-'} "
     f"{_best_sens:.2f}")
PRED_DF = pd.DataFrame(PREDS)
PRED_DF.to_csv(OUT_DIR / "predictions.csv", index=False)
print("PRE-REGISTERED PREDICTIONS\n")
for r in PRED_DF.itertuples(index=False):
    print(f"  {r.id}  [{'blind' if r.blind=='yes' else '  n/a '}] {r.prediction[:74]}")
    print(f"       -> {r.status}: {r.evidence[:96]}")

# ---- code cell 34 ----
print(" >>> cell 34", flush=True)
"""Cell R2 - write Experiment3.txt from the live objects."""
_lines = []
A = _lines.append
A("=" * 78)
A(f"{MODEL_ID} / {EXPERIMENT_ID}  {EXPERIMENT_NAME}")
A("=" * 78)
A(f"run context      : {RUN_CONTEXT}  (E3 is local-only: no torch, no GPU, no Kaggle run)")
A(f"seed             : {SEED}   permutation seed: {CFG.permutation_seed}")
A(f"embed seed       : {CFG.embed_seed} (pinned by E2's cache filenames)")
A(f"config_hash      : {CONFIG_HASH}   pipeline {PIPELINE_VERSION}")
A(f"gated arm        : {GATED}   probe_arm: {CFG.probe_arm!r}")
A(f"arms executed    : {ARMS_LIST}")
A(f"outputs          : {OUT_DIR}")
A(f"platform         : {platform.platform()}")
A("")
A("1. HYPOTHESIS H3, AND WHY IT IS THE LAST ONE STANDING")
A("   E1 refuted replacement (D 44.33 vs M1 43.02). E2 refuted rescue by magnification")
A("   (D lost to R at all four crops, rho -0.40 p 0.60, no gain located in the fines).")
A("   Neither tested complementarity: does the 384-dim block add information to the 12")
A("   hand features? E1's durable finding motivates it - D is ~5x camera-stable and ~28x")
A("   blur-stable yet less accurate, so the two families may be making different errors.")
A(f"   PREDICTED OUTCOME BEFORE THE RUN (P3): H3 is REFUTED.")
A("")
A("2. THE ONE VARIABLE, AND WHAT IS FROZEN")
A("   Changed: feature matrix composition, 12 -> 12 + 384 = 396 columns.")
A("   Frozen: tiles, soil-fraction filter, tile->image->soil median, StandardScaler")
A("   treatment, rank-3 basis, closed-form ridge, alpha grid, monotone projection, the")
A("   16 CV families over 24 soils, trapezoid EMD, the E4 parsimony rule, backbones.py")
A("   untouched and un-imported, crop_geometry.py untouched.")
A("   Cells reused BYTE-FOR-BYTE from E2's notebook: @@VERBATIM_CELLS@@")
A("")
A("3. PROVENANCE OF THE EMBEDDINGS (E3 reads E2's arrays rather than computing them)")
A(f"   {len(_prov)} arrays + equal path files, {sum(_prov.bytes)/1e6:.1f} MB, "
  f"{len(set(_prov.sha256))} distinct sha256")
A(f"   rows per view: {dict((v, int(g.rows.iloc[0])) for v, g in _prov.groupby('view'))}")
A(f"   regression check level: {E2_LEVEL}")
A("")
A("4. THE REGRESSION CHECK - THE GATE BEFORE THE GATE")
A(f"   control vs Model 1 E3 matrix: asserted in G2 to < 1e-6")
A(f"   base arms vs E2: asserted in H3 within {CFG.REGRESSION_TOL} EMD; failure stops here")
A(f"   {E2_LEVEL}")
A("")
A("5. PRIMARY RESULT - nested in-domain LOGO-CV")
for r in NESTED_DF.itertuples(index=False):
    A(f"   {r.label:9s} {str(r.arm):16s} n_feat {r.n_feat:4d}  nested "
      f"{r.nested_in_domain:7.2f}  oracle alpha {r.oracle_alpha:g}")
A(f"   no-image floor {FLOOR_NO_IMAGE:.2f} | rank-3 ceiling 7.35 | control M1 "
  f"{_idm['M1']:.2f}")
A("")
A("6. PAIRED CONTRASTS (negative estimate = first arm better)")
for r in CONEFF.itertuples(index=False):
    A(f"   {r.contrast:15s} {r.estimate:+8.2f} CI [{r.ci_lo:+8.2f}, {r.ci_hi:+8.2f}] "
      f"{'SIGNIF' if r.significant else 'ns    '}  wins {r.wins_a}/{r.n_soils}")
A("")
A("7. BY D50 GROUP (diagnostic, not a gate clause)")
for r in GR.itertuples(index=False):
    A(f"   {r.label:9s} overall {r.overall:7.2f}  fines {r.fines:7.2f} (n={r.n_fines})  "
      f"mid {r.mid:7.2f}  coarse {r.coarse:7.2f}")
A("")
A("8. MECHANISM AND THE REPORT-ONLY RULER")
if "SHOW" in globals() and len(SHOW):
    for i, row in SHOW.iterrows():
        A(f"   {str(i):18s} camera/soil {row.get('camera_over_soil', float('nan')):.3f}  "
          f"blur/soil {row.get('blur_over_soil', float('nan')):.3f}")
A("   CAM+RES is REPORTED, NOT USED TO SELECT (Model 1 E4's demotion).")
if "RULER" in globals() and len(pd.DataFrame(RULER)):
    for r in pd.DataFrame(RULER).itertuples(index=False):
        A(f"   {getattr(r,'arm','?'):18s} CAM+RES {getattr(r,'camres',float('nan')):7.2f}")
A("")
A("9. DECISION GATE")
A(f"   submit: {SUBMIT}   verdict: {VERDICT}")
for c in CLAUSES:
    A(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['clause']}"
      + (f"   {c['detail']}" if c['detail'] else ""))
A(f"   fusion gain over M1: {GAIN:+.2f} EMD against a {CFG.ORDINAL_FLOOR} EMD band")
A("")
A("10. SUBMISSION")
if SUB_PATH is not None:
    A(f"   {SUB_PATH.name} written by the gate. NOT a probe.")
else:
    A("   NO SUBMISSION WAS PRODUCED BY THIS RUN. gate.submit = False; probe_arm = None,")
    A("   and E3 pre-authorises no probe.")
A("")
A("11. PRE-REGISTERED PREDICTIONS")
for r in PRED_DF.itertuples(index=False):
    A(f"   {r.id}  [{'blind' if r.blind=='yes' else 'no     '}] {r.prediction}")
    A(f"        -> {r.status}: {r.evidence}")
A("")
A("12. WHAT THIS DECIDES")
if not G1:
    A("   G1 failed: complementarity is refuted. MODEL 2 CLOSES at three experiments -")
    A("   replacement (E1), magnification (E2), complementarity (E3) are the three ways a")
    A("   frozen backbone could help this target. Remaining budget goes to the Model 1 line.")
elif VERDICT.startswith("CAPACITY"):
    A("   A gain appeared but no larger than the permuted block's: it is capacity, not")
    A("   information. Model 2 closes; the SHUF control is what prevented a false positive.")
elif GAIN < CFG.ORDINAL_FLOOR:
    A(f"   All clauses passed with a {GAIN:+.2f} EMD gain, below the {CFG.ORDINAL_FLOOR}")
    A("   band the ruler can resolve: statistically real, not competition-relevant.")
else:
    A("   All clauses passed with a gain above the resolvable band. Model 2 has a genuine")
    A("   contribution; a crop follow-up is warranted as a NEW pre-registered experiment.")
A("")
A("12b. CORRECTIONS APPLIED TO THE GATE CODE AFTER RUN 1 - DISCLOSED, NOT BACKED INTO")
A("   Two defects were found in cell O1 by reading run 1's own output. Both made the code")
A("   disagree with the clause text already frozen in instructions.txt; neither changed a")
A("   clause, a threshold or a comparator, and neither changed the verdict.")
A("     (a) VALID was coded as (SHUF - M1) < 0, the inverse of the spec 'VALID:")
A("         in_domain(SHUF@256) > in_domain(M1)'. Measured SHUF - M1 = +9.90, so run 1")
A("         mislabelled a healthy instrument as INSTRUMENT INVALID.")
A("     (b) The FUSE - SHUF contrast was bound to None, so G2 evaluated nothing and printed")
A("         'contrast unavailable'. The real value is -9.50, CI [-23.56, +4.82], not")
A("         significant, so G2 fails either way.")
A("   G1 fails (+0.41, CI excludes nothing) regardless, so the verdict is REFUTED under both")
A("   the run-1 and run-2 code. Nothing downstream of these two lines was re-tuned.")
A("   ALSO FLAGGED: P5 is marked CONFIRMED but is near-vacuous. FUSE's camera/soil ratio is")
A("   1.1044 against M1's 1.1048 - it lands inside the interval only because it is")
A("   indistinguishable from M1, i.e. fusion did NOT retain D's invariance in ratio units.")
A("   The absolute camera shift did stay near D's (0.64 vs 0.62 vs M1's 1.21).")
A("")
A("13. SCOPE OF VERIFICATION")
A("   E3 has no mock mode: this run executed every cell on real arrays. Nothing in the")
A("   report is unread-but-assumed. The embeddings were produced by E2 on Kaggle and are")
A(f"   consumed here as fixed inputs, provenanced by sha256 and by reproducing E2's arms.")
A("   Determinism: the emitted data files are byte-identical across repeated runs, and every")
A("   cell printed output (no silent skip).")
A("   Executed on this machine, CPU only, no GPU and no Kaggle session.")
A("")
A("14. ENVIRONMENT")
for k, v in VERSIONS.items():
    A(f"   {k:12s} {v}")
TXT = OUT_DIR / "Experiment3.txt"
TXT.write_text("\n".join(_lines) + "\n", encoding="utf-8")
print(f"wrote {TXT} ({len(_lines)} lines)")

# ---- code cell 35 ----
print(" >>> cell 35", flush=True)
"""Cell R3 - summary."""
print("=" * 70)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  arms={ARMS_LIST}  regression: {E2_LEVEL}")
print("\nnested in-domain by arm:")
for a in ARMS_LIST:
    print(f"  {LABEL.get(a,a):9s} {a:16s} {_idm[a]:7.2f}")
print(f"\ngate: {VERDICT}   submit={SUBMIT}   gain vs M1 {GAIN:+.2f} EMD")
print(f"submission: {'written' if SUB_PATH is not None else 'NOT written'}")
print("\nfiles written:")
for f in sorted(OUT_DIR.glob("*")):
    if f.is_file() and f.name != ".cache":
        print(f"  ok   {f.name}")

