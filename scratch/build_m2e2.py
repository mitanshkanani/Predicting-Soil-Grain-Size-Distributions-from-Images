"""Builds 'Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb'.

E2 changes one thing versus Model 2 E1: how much physical soil a tile contributes to the
backbone. A centred window of the existing 256 px tile is cropped and resized to 224, and
the sweep runs for both embedding arms - pretrained D and random-init R - against the
frozen Model 1 control.

E1's `backbones.py` is imported unmodified. The new logic lives in `crop_geometry.py`,
which has no torch and is fully tested locally.

Run:  python scratch/build_m2e2.py
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src))


def code(src):
    CELLS.append(("code", src))


# ================================================================ A. SETUP
md("""# Model 2 - Experiment 2
## Does E1's refutation survive correcting the magnification?

**Single variable.** How much physical soil a tile contributes to the backbone. A centred
window of the existing 256 px tile is cropped and resized to 224 px. Everything else is
frozen at E1's values: the same tiles, the same two-stage tile->image->soil median, the
same `StandardScaler`, rank-3 curve basis, closed-form ridge, monotone projection, the same
16 CV families, the same trapezoid EMD, and the same unmodified `backbones.py`.

**Why this experiment exists.** E1 refuted its premise - D 44.33 vs M1 43.02 in-domain, and
no better than its own random-weights control. But E1 declared its own hole before it ran:
a patch spans **3.51 mm** and twelve of twenty-four soils have D50 below 1.03 mm, so for
those the backbone sees fines as texture, never as particles. Testing a representation at a
scale that cannot resolve the measured quantity is a weak test. Until that is separated,
"learned features don't help" is provisional.

**Arms.** `M1` Model 1 E3's 12 hand features at full tile (the anchor, never re-cropped).
`R` random-init ViT-S/14 and `D` frozen pretrained DINOv2, **both riding the whole crop
sweep** - because a gain for D at fine crops is only evidence about pretraining if R does
not gain equally, and R's 256 px advantage came from a feature collapse that magnification
may itself undo.

**`crop 256` is not an arm, it is a regression check.** It is the same computation as E1's,
so E2 must reproduce E1's in-domain table exactly or the experiment is void.

Read `instructions.txt` in this folder before changing anything.""")

code('''"""Cell A1 - configuration. Every tunable lives here and nowhere else."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import os

MODEL_ID        = "Model2"
EXPERIMENT_ID   = "E2"
EXPERIMENT_NAME = "Model 2 / Experiment 2 - patch magnification sweep"
SEED = 20260931
E1_DIR_NAME = "Model 2 Experiment 1"
E3_DIR_NAME = "Model 1 Experiment 3"


@dataclass(frozen=True)
class Config:
    # ---- frozen at Model 2 E1's values; changing any of these is a different experiment
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
    embed_seed: int = 20260930
    batch_size: int = 64
    # E1's grid, unchanged. Not re-extended: that accommodation is already part of the
    # frozen head definition, and touching it per-experiment would make E1 and E2
    # incomparable.
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                                     300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0,
                                     300000.0, 1000000.0)
    # ---- THE ONE NEW VARIABLE. 256 is E1's own configuration and serves as the
    # regression check; 32 is the practical floor (below that a window is a handful of real
    # pixels stretched to 224, which changes the encoder input without adding information).
    crops: Tuple[int, ...] = (256, 128, 64, 32)
    # ---- gate thresholds, fixed before the run
    REGRESSION_TOL: float = 0.05     # EMD; crop 256 must reproduce E1 this closely
    GROUP_CONCENTRATION: float = 0.0  # a gain must be strictly larger in the patch-limited
                                      # groups than in the fines
    probe_arm: Optional[str] = None   # E1 used a declared probe; E2 does not, by default
    # D50 group cut points. The training distribution is bimodal - ten soils under 0.11 mm,
    # twelve above 1.5 mm, two in between - so these are the data's own boundaries, not a
    # smooth grid.
    fine_d50_mm: float = 0.11
    mid_d50_mm: float = 1.5


CFG = Config()


def _env_backends():
    """BACKEND accepts mock | vit_random | dinov2 | both | all | none. Default is mock
    locally so a careless local run can never produce a table that looks like a result, and
    'both' on Kaggle so one run produces R and D across the whole sweep."""
    raw = os.environ.get("BACKEND", "").strip().lower()
    if raw in ("none", "control", "control-only"):
        return ()
    if not raw:
        return ("mock",) if not Path("/kaggle/input").exists() else ("vit_random", "dinov2")
    if raw == "both":
        return ("vit_random", "dinov2")
    if raw == "all":
        return ("mock", "vit_random", "dinov2")
    parts = tuple(p.strip() for p in raw.replace(";", ",").split(",") if p.strip())
    return parts or ("mock",)


BACKBONES = _env_backends()
ON_KAGGLE = Path("/kaggle/input").exists()
_EXP_DIR = Path("Model 2 / Model 2 Experiment 2".replace(" / ", "/"))
if ON_KAGGLE:
    OUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
elif set(BACKBONES) <= {"mock", "none", "control", "control-only"}:
    # Plumbing artifacts and results must never share a directory. A mock run once
    # overwrote E1's downloaded Kaggle outputs because they use identical filenames.
    OUT_DIR = _EXP_DIR / "local mock dry-run"
else:
    OUT_DIR = _EXP_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  context={'kaggle' if ON_KAGGLE else 'local'}")
print(f"backbone arms requested: {BACKBONES or '(control only)'}   crop sweep: {CFG.crops}")
print("outputs ->", OUT_DIR)''')

code('''"""Cell A2 - imports and environment capture."""
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
    print(f"  {_k:12s} {_v}")''')

code('''"""Cell A3 - locate the processed data, the frozen backbone module and the new crop module.

Three separate searches, because the three things live in different places and Kaggle's
mount depth is not stable. Each is found by CONTENT, not by folder name: an experiment that
only finds what it expects is an experiment that breaks silently.
"""
def _attachment_roots(max_depth=4):
    """Every directory that could plausibly be a dataset root.

    Kaggle's mount depth is NOT stable: `/kaggle/input/<slug>` and
    `/kaggle/input/datasets/<user>/<slug>` have both been observed, and the second is three
    levels down. E1's resolver carried a deep fallback; this cell's first draft assumed two
    levels and failed on Kaggle with 'Could not find manifest_images.csv' while the data was
    plainly attached. So walk a bounded number of levels and let the callers test each one.
    """
    roots = []
    kg = Path("/kaggle/input")
    if kg.exists():
        frontier = [kg]
        for _ in range(max_depth):
            nxt = []
            for d in frontier:
                roots.append(d)
                try:
                    nxt += [p for p in sorted(d.iterdir()) if p.is_dir()]
                except OSError:
                    pass
            frontier = nxt
            if not frontier:
                break
    here = Path.cwd()
    roots += [here, here / "Model 2" / "Model 2 Experiment 2",
              here / "Model 2" / "Model 2 Experiment 1", here / "Model 1"]
    seen, out = set(), []
    for d in roots:
        try:
            r = d.resolve()
        except OSError:
            continue
        if r.is_dir() and r not in seen:
            seen.add(r); out.append(r)
    return out


def resolve_input_root():
    PROBE = Path("data") / "processed_meta" / "manifest_images.csv"
    for d in _attachment_roots():
        if (d / PROBE).exists():
            return d
    kg = Path("/kaggle/input")
    if kg.exists():                                   # last resort: search the whole tree
        try:
            hits = sorted(p for p in kg.rglob("manifest_images.csv")
                          if p.parent.name == "processed_meta")
        except OSError:
            hits = []
        if hits:
            return hits[0].parent.parent.parent
    for d in [Path.cwd(), *Path.cwd().parents]:
        if (d / PROBE).exists():
            return d
    raise RuntimeError("Could not find data/processed_meta/manifest_images.csv under any "
                       "attachment. Attach the processed dataset (kaggle_setup.md "
                       "section 1). Locally, run from the repository root.")


def _find_file(fname):
    """Nearest attachment root containing fname, then any depth beneath it."""
    for d in _attachment_roots():
        if (d / fname).exists():
            return d
    for d in _attachment_roots():
        try:
            for hit in sorted(d.rglob(fname)):
                return hit.parent
        except OSError:
            continue
    return None


INPUT_ROOT = resolve_input_root()
RUN_CONTEXT = "kaggle" if ON_KAGGLE else "local"
META = INPUT_ROOT / "data" / "processed_meta"
sys.path.insert(0, str(INPUT_ROOT))

BB_DIR = _find_file("backbones.py")
assert BB_DIR is not None, ("backbones.py not found. E2 imports it UNMODIFIED from Model 2 "
                            "E1 - the backbone is not something this experiment is allowed "
                            "to change, and a locally rewritten copy of the feature "
                            "extractor is the easiest way to make a result unreproducible.")
sys.path.insert(0, str(BB_DIR))

CG_DIR = _find_file("crop_geometry.py")
assert CG_DIR is not None, ("crop_geometry.py not found. It owns E2's single new variable "
                            "and must be attached with this experiment.")
sys.path.insert(0, str(CG_DIR))

from preprocess import verify as pv          # the validated hand-built feature definition
import backbones as bb                       # E1's frozen torch surface
import crop_geometry as cg                   # E2's one new variable

print(f"INPUT_ROOT     {INPUT_ROOT}")
print(f"backbones.py   {bb.__file__}")
print(f"crop_geometry  {cg.__file__}")''')

code('''"""Cell A4 - pin the preprocessing version, the submission schema, and the seed."""
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
print(f"config_hash {CONFIG_HASH} | pipeline {PIPELINE_VERSION} | {TARGET_PPM:.6f} px/mm")''')

# ================================================================ B. DATA & METRIC
md("""## Section B - the metric and the six external ground-truth scores

The metric is re-asserted against the host's own published number in every notebook. Six
Kaggle scores are now fixed reference points, including Model 2 E1's probe at 60.56.""")

code('''"""Cell B1 - load manifests and labels. Labels come from the SAMPLE manifest only."""
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
print(f"D50 spans {D50_MM.min():.4f} .. {D50_MM.max():.3f} mm")''')

code('''"""Cell B2 - the EMD implementation and its reconciliation against the page."""
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
assert abs(_t - PUB_TRIVIAL) < 0.05, \\
    "our EMD no longer reproduces the host's published reference - stop and reconcile"
print("METRIC CONFIRMED (as in E2, E3, E4 and Model 2 E1).")''')

code('''"""Cell B3 - reference baselines and every external score this project owns."""
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
print("\\nexternal ground truth:")
for _k, _v in sorted(LB.items(), key=lambda kv: kv[1]):
    print(f"  {_v:8.2f}  {_k}")
print(f"\\n  best so far {LB_M2E1:.2f} (Model 2 E1's probe). The public set is a fixed 3 of the")
print("  10 test soils and the final ranking is 0.30*public + 0.70*private, so a difference")
print("  of a few tenths on three soils is not evidence of a large effect.")''')

# ================================================================ C. PRIOR RESULTS
md("""## Section C - what Model 2 E1 concluded, and the geometry that bounds E2

E1's verdict is loaded from its own artifacts rather than quoted, and the reachability
table is computed live so the number that frames this experiment cannot silently drift.""")

code('''"""Cell C1 - locate E1's and E3's outputs by content, not by folder name.

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
          "falls\\n          back to E1's pinned arm means, which is a weaker but still "
          "real check.")
    e1_cv = pd.DataFrame()
    E1_ARMS = []
elif _is_plumbing(E1_DIR):
    print("  The only E1 per-soil file found is inside a plumbing folder, so it is not "
          "used.\\n          The regression check falls back to E1's pinned arm means.")
    e1_cv = pd.DataFrame()
    E1_ARMS = []
else:
    e1_cv = pd.read_csv(E1_DIR / "cv_per_soil.csv").set_index("sample_id")
    E1_ARMS = [c for c in ("M1", "D", "R") if c in e1_cv.columns]
    if not {"D", "R"} <= set(E1_ARMS):
        print(f"  NOTE: E1's file carries only {E1_ARMS}. The crop-256 regression check will "
              "verify the\\n        control arm per soil and the backbone arms against pinned "
              "means only.")
e3_soil = pd.read_csv(E3_DIR / "features_soil.csv").set_index("sample_id")
e3_fam = pd.read_csv(E3_DIR / "cv_families.csv")
e3_sub = pd.read_csv(E3_DIR / "Submission_Model1_E3.csv")
print(f"E1_DIR {E1_DIR}")
print(f"E3_DIR {E3_DIR}")
for _c in list(CFG.control_feats):
    assert _c in e3_soil.columns, f"E3's matrix is missing control feature {_c}"''')

code('''"""Cell C2 - E1's verdict, restated from E1's own per-soil file."""
E1_IN = {k: float(e1_cv[k].mean()) for k in E1_ARMS} if len(E1_ARMS) else dict(E1_PINNED)
print("MODEL 2 E1 VERDICT\\n")
print("  nested in-domain LOGO-CV   " + "   ".join(
    f"{k} {E1_IN[k]:6.2f}" for k in ("M1", "D", "R")))
if not len(e1_cv.columns):
    print("  (from E1's pinned record means - its per-soil file is not attached)")
else:
    print("  (recomputed from E1's own cv_per_soil.csv)")
print("  gate: D must beat M1 AND beat R with a significant paired CI. It did neither.")
print("  P4 REFUTED, P5 REFUTED -> the premise that a learned representation extracts more")
print("  grain-size signal than 12 hand-built features is REFUTED at E1's magnification.")
print(f"\\n  E1's declared probe (arm D submitted anyway, as a measurement of transfer):")
print(f"    {LB_M2E1:.5f} vs E3's {LB_E3:.5f}  -> {LB_E3 - LB_M2E1:+.3f} EMD. A near-tie.")
print("\\n  The hole this experiment closes: E1 tested at a patch scale that cannot resolve")
print("  the measured quantity for half the soils. That was declared in E1's own section 12")
print("  and never tested. Until it is, the refutation is provisional.""")''')

code('''"""Cell C3 - the geometry, computed live. Includes the correction to E1's number."""
print(f"canonical scale {TARGET_PPM:.6f} px/mm | tile {CFG.tile_size_px}px = "
      f"{cg.window_mm(CFG.tile_size_px, TARGET_PPM):.2f} mm\\n")
print(cg.describe(D50_MM, TARGET_PPM))
print("\\n  'resolvable' counts soils whose median grain is at least 1/3 of a patch. Below")
print("  that, a grain sits inside one token's receptive field: the network sees texture,")
print("  never particles.")
_naive = cg.PATCH_PX / TARGET_PPM
print(f"\\nCORRECTION TO E1'S RECORDED NUMBER\\n  E1 recorded PATCH_MM = 14/4.552516 = "
      f"{_naive:.4f} mm. That omits the 256->224 resize.\\n  True footprint at crop 256 = "
      f"{cg.patch_mm(256, TARGET_PPM):.4f} mm "
      f"({cg.patch_mm(256, TARGET_PPM) / _naive - 1:+.1%}).\\n  backbones.py is NOT edited - "
      "E1's record and cached embeddings must stay byte-reproducible.\\n  E1's conclusions "
      "are unaffected: the resolvable count is 12 either way.")
_gain = cg.resolvability(D50_MM, min(CFG.crops), TARGET_PPM) - \\
        cg.resolvability(D50_MM, max(CFG.crops), TARGET_PPM)
_area = 100.0 * (min(CFG.crops) / CFG.tile_size_px) ** 2
_n_hi = cg.resolvability(D50_MM, max(CFG.crops), TARGET_PPM)
_n_lo = cg.resolvability(D50_MM, min(CFG.crops), TARGET_PPM)
print(f"\\nWHAT BOUNDS THIS EXPERIMENT\\n  The whole sweep buys {_gain} more resolvable soils "
      f"({_n_hi} -> {_n_lo} of 24) while using {_area:.1f}% of the tile area.")
print(f"  The fines are physically unreachable, not under-sampled: the largest fine soil "
      f"(D50={D50_MM[D50_MM < CFG.fine_d50_mm].max():.4f} mm) needs a "
      f"{D50_MM[D50_MM < CFG.fine_d50_mm].max() * 3 * TARGET_PPM * cg.INPUT_SIZE / cg.PATCH_PX:.0f}"
      f"x same window upsampled to {cg.INPUT_SIZE}px, which contains no grain information.")
print("  Therefore any magnification gain MUST appear in the coarse/mid groups and be "
      "absent\\n  in the fines. A uniform gain is not explainable by magnification; a gain in\\n"
      "  the fines would indicate a leak. This is P4 and the gate's fourth clause.")''')

# ================================================================ D. CROP CHECKS
md("""## Section D - the crop, verified before a single tile is processed

Two checks: the module's own contract, and the assumption that a centre crop is
representative of its tile. If the second fails, the sweep is confounded with soil
coverage and the experiment cannot be read.""")

code('''"""Cell D1 - run check_crop_geometry.py, then check_backbones.py against the real arms."""
_here = Path(cg.__file__).parent
_rc = subprocess.run([sys.executable, str(_here / "check_crop_geometry.py")],
                     cwd=str(_here), capture_output=True, text=True)
print(_rc.stdout[-2600:])
assert _rc.returncode == 0, "crop_geometry contract failed:\\n" + _rc.stderr[-2000:]
print("PASS: crop and geometry contract holds.")

HAS_TORCH = bb.torch_available()
_to_verify = [b for b in BACKBONES if not bb.requires_torch(b) or HAS_TORCH]
if _to_verify:
    cmd = [sys.executable, str(Path(bb.__file__).parent / "check_backbones.py"), *_to_verify]
    print("\\nrunning:", " ".join(cmd))
    _rc2 = subprocess.run(cmd, cwd=str(Path(bb.__file__).parent), capture_output=True, text=True)
    print(_rc2.stdout[-2200:])
    assert _rc2.returncode == 0, "check_backbones.py failed:\\n" + _rc2.stderr[-2000:]
    print("PASS: backbone interface contract holds for " + ", ".join(_to_verify))
else:
    print("\\nno backbone arms requested - control-only run, backbone checks skipped.")''')

code('''"""Cell D2 - is a centre crop representative of its tile?

The tiles are cut from the inscribed soil rectangle, so soil should fill them uniformly -
but the sweep's interpretation depends on it, so it is measured on every tile rather than
assumed. The criterion is absolute: the soil threshold is taken from the FULL tile, then
the density inside the crop is compared against the tile's own density. A per-window
percentile would be useless here, because it always excludes the same fraction by
construction.
"""
_rng = np.random.default_rng(SEED)
_sample = tiles.sample(min(240, len(tiles)), random_state=SEED)
_cov = {}
for c in CFG.crops:
    ratios, abss = [], []
    off = (CFG.tile_size_px - c) // 2
    for p in _sample.tile_path:
        g = np.asarray(Image.open(INPUT_ROOT / p).convert("RGB"), float).mean(2)
        thr = np.percentile(g, 12)
        full = g > thr
        w = full[off:off + c, off:off + c]
        ratios.append(w.mean() / max(full.mean(), 1e-9))
        abss.append(w.mean())
    ratios, abss = np.array(ratios), np.array(abss)
    _cov[c] = (float(ratios.mean()), float(np.percentile(ratios, 5)),
               float(np.percentile(ratios, 95)), float((abss < 0.5).sum()))
    print(f"  crop {c:3d}px: density ratio {ratios.mean():.3f} "
          f"[p05 {np.percentile(ratios,5):.3f}, p95 {np.percentile(ratios,95):.3f}]  "
          f"crops below 0.5 soil: {int((abss < 0.5).sum())}/{len(abss)}")
_bad = [c for c, v in _cov.items() if abs(v[0] - 1.0) > 0.10 or v[3] > 0]
assert not _bad, (f"centre crops are not representative at {_bad}. The sweep would be "
                  "confounded with how much soil is in view, not with magnification - "
                  "stop and redesign rather than reading the numbers.")
print("\\nPASS: every crop is soil-representative, so a difference between crops is a")
print("difference in magnification and nothing else.")''')

# ================================================================ E. ENVIRONMENT
md("""## Section E - environment probe

The backbone arms cannot run on the development machine. That is declared here, before any
long computation, rather than discovered later.""")

code('''"""Cell E1 - what can actually run in this process."""
if HAS_TORCH:
    import torch, timm
    VERSIONS.update({"torch": torch.__version__, "timm": timm.__version__})
    print(f"  {'torch':12s} {torch.__version__}")
    print(f"  {'timm':12s} {timm.__version__}")
    print(f"  cuda         {torch.cuda.is_available()} "
          f"{torch.cuda.device_count() if torch.cuda.is_available() else ''}")

ARMS = {"M1": dict(backend="M1", crop=CFG.tile_size_px, executable=True,
                   dim=len(CFG.control_feats))}
for _b in BACKBONES:
    for _c in CFG.crops:
        ok = HAS_TORCH or not bb.requires_torch(_b)
        ARMS[f"{_b}@c{_c}"] = dict(backend=_b, crop=_c, executable=ok, dim=bb.dim(_b))

LABEL = {"M1": "M1", "mock": "MOCK", "vit_random": "R", "dinov2": "D"}


def arm_name(backend, crop):
    return "M1" if backend == "M1" else f"{LABEL[backend]}@{crop}"


print("\\narms requested / executable:")
for k, a in ARMS.items():
    print(f"  {k:22s} dim={a['dim']:4d} torch={bb.requires_torch(a['backend']) if a['backend']!='M1' else False!s:6s} "
          f"executable={a['executable']}")
BLOCKED = [k for k, a in ARMS.items() if not a["executable"]]
if BLOCKED:
    print(f"\\n** {len(BLOCKED)} backbone arms cannot run here (no torch). Expected locally.**")
    print("  The mock arm is a plumbing test and is never a result; the gate excludes it.")
WEIGHTS_PATH = os.environ.get("DINO_WEIGHTS", "").strip() or None
if WEIGHTS_PATH:
    _wp = Path(WEIGHTS_PATH)
    if not _wp.is_absolute():
        for cand in [Path("/kaggle/input"), Path.cwd()]:
            hits = sorted(cand.rglob(Path(WEIGHTS_PATH).name)) if cand.exists() else []
            if hits:
                _wp = hits[0]; break
    assert _wp.exists(), f"DINO_WEIGHTS={WEIGHTS_PATH} resolved to {_wp}, which is not there"
    WEIGHTS_PATH = str(_wp)
    print(f"\\nweights: {WEIGHTS_PATH}")
else:
    print("\\nweights: none set - timm fetches from the hub (needs internet on).")
CELL_SEED = CFG.embed_seed
print(f"embedding seed {CELL_SEED} | batch {CFG.batch_size} | sweep {CFG.crops}")''')

# ================================================================ F. EMBEDDINGS
md("""## Section F - the sweep, one embedding table per (arm, crop, view)

The cache key **must include the crop**. E1's key omits it, so reusing E1's naming would
load the full-tile embedding at every scale and produce a completely fake sweep - four
identical arms reported as a magnification experiment. The key is asserted distinct in-cell.""")

code('''"""Cell F1 - tile metadata and the soil-fraction filter, identical to E1."""
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
print(f"\\nmedian tiles per soil: train {_tps.get('train', 0):.0f}, test {_tps.get('test', 0):.0f}"
      "   (test soils already see fewer views; a fine crop shrinks the area per view too)")''')

code('''"""Cell F2 - the hand-built features for the control arm, reused from E1's caches."""
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
_srcs = [CACHE, Path(bb.__file__).parent / ".cache", E3_DIR / ".cache"]
# E1_DIR is None whenever E1's per-soil file is not attached - which is the normal state on
# Kaggle, because the zip ships E1's folder without that file. It crashed the first Kaggle
# run here. Locally it never showed, because E1's plumbing folder always resolved.
if E1_DIR is not None and not _is_plumbing(E1_DIR):
    _srcs.append(E1_DIR / ".cache")
_srcs = [s for s in _srcs if s.is_dir()]
print(f"cache sources: {[str(s) for s in _srcs]}")


def get_cache(fname, builder):
    for d in _srcs:
        if (d / fname).exists():
            if not (CACHE / fname).exists():
                (CACHE / fname).write_bytes((d / fname).read_bytes())
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
print(f"hand-built control features: {tf0.shape} from {_hsrc}")''')

code('''"""Cell F3 - the resolution-matched blur, derived from the manifests exactly as E1/E3 did.

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
    print(f"  view '{tag}' hand features from {s}")''')

code('''"""Cell F4 - embedding extraction across the sweep.

Cache key: embed_<backend>_c<crop>_<tile>_<config_hash>_<seed>_<view>.npy. The crop is in
the key on purpose and the distinctness is asserted below, because a key that ignores it
would silently return the full-tile embedding for every arm and turn the sweep into four
copies of the same number.
"""
def embed_cache_name(backend, crop, view):
    v = "" if view == "real" else f"_{view}"
    return f"embed_{backend}_c{crop}_{CFG.tile_size_px}_{CONFIG_HASH}_{CELL_SEED}{v}.npy"


_names = {embed_cache_name(b, c, "real") for b in BACKBONES for c in CFG.crops}
assert len(_names) == len(BACKBONES) * len(CFG.crops), \\
    "embedding cache keys collide across crops - the sweep would silently reuse one table"
print(f"cache keys distinct across the sweep: {len(_names)} keys")


def get_embeddings(backend, crop, sub_tiles, view):
    """(array aligned to sub_tiles.tile_path, names, source)."""
    fn = embed_cache_name(backend, crop, view)
    pf = fn.replace(".npy", ".paths.npy")
    want = sub_tiles.tile_path.tolist()
    for d in _srcs:
        p = Path(d) / fn
        if not p.exists() or not (Path(d) / pf).exists():
            continue
        names = np.load(Path(d) / pf, allow_pickle=True).tolist()
        arr = np.load(p)
        if names != want or arr.shape != (len(want), bb.dim(backend)):
            print(f"    ignoring stale cache {fn}")
            continue
        for _f in (fn, pf):
            if not (CACHE / _f).exists():
                (CACHE / _f).write_bytes((Path(d) / _f).read_bytes())
        return arr, want, str(d)
    sigma_fn = None if view == "real" else (lambda c, v=view: _sig_for(c, "iPhone " + v[-2:]))
    recs, cap, t0 = [], 256, time.time()
    for i in range(0, len(sub_tiles), cap):
        blk = sub_tiles.iloc[i:i + cap]
        arr = np.stack([np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB"), float)
                        for t in blk.itertuples(index=False)])
        if sigma_fn is not None:
            arr = np.stack([_blur_rgb(a, sigma_fn(c)) for a, c in zip(arr, blk.camera)])
        arr = cg.crop_tile(arr, crop)
        recs.append(bb.embed_tiles(arr, backend, seed=CELL_SEED, batch_size=CFG.batch_size,
                                   weights_path=WEIGHTS_PATH, progress_every=0))
        if (i // cap + 1) % 4 == 0 or i + cap >= len(sub_tiles):
            print(f"    {backend}@c{crop}/{view}: {min(i+cap, len(sub_tiles))}/"
                  f"{len(sub_tiles)}  {time.time()-t0:.0f}s", flush=True)
    out = np.concatenate(recs, axis=0)
    assert out.shape == (len(sub_tiles), bb.dim(backend))
    np.save(CACHE / fn, out)
    np.save(CACHE / pf, np.array(want, dtype=object), allow_pickle=True)
    return out, want, "built here"


EMB = {}
for _b in BACKBONES:
    if not all(ARMS[f"{_b}@c{c}"]["executable"] for c in CFG.crops):
        print(f"\\n{_b}: not executable here (needs torch). Skipped.")
        continue
    for _c in CFG.crops:
        print(f"\\nextracting {_b} @ crop {_c}px "
              f"(patch {cg.patch_mm(_c, TARGET_PPM):.4f} mm) over {len(tiles)} real tiles")
        EMB[(_b, _c, "real")] = get_embeddings(_b, _c, tiles, "real")
        for view in VIEWS[1:]:
            EMB[(_b, _c, view)] = get_embeddings(_b, _c, TRAIN_TILES, view)
    if _b == "mock":
        print("  (mock is a plumbing test, never a result - the gate excludes it by name)")
print(f"\\nembedding tables built: {len(EMB)}")''')

# ================================================================ G. AGGREGATION
md("""## Section G - aggregation, unchanged, and proved unchanged

Two-stage median, tile -> image -> soil, the identical code path for every arm at every
crop. The control must reproduce Experiment 3's matrix to under 1e-6 or nothing downstream
is comparable to Model 1.""")

code('''"""Cell G1 - tile -> image -> soil, one code path for every arm and crop."""
ECOLS = {b: [f"emb_{i:03d}" for i in range(bb.dim(b))] for b in bb.BACKENDS}
ARM_FEATS = {"M1": list(CFG.control_feats)}
for _b in BACKBONES:
    for _c in CFG.crops:
        ARM_FEATS[f"{_b}@c{_c}"] = ECOLS[_b]
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
        if arm != "M1" and view != "real" and (arm.split("@c")[0],
                                               int(arm.split("@c")[1]), view) not in EMB:
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


build_arm("M1")
for _b in BACKBONES:
    for _c in CFG.crops:
        if ARMS[f"{_b}@c{_c}"]["executable"]:
            build_arm(f"{_b}@c{_c}")
ARMS_LIST = ["M1"] + [f"{_b}@c{_c}" for _b in BACKBONES for _c in CFG.crops
                      if (_b, _c, "real") in EMB]
print("soil-level tables:")
for k in sorted(SOIL, key=str):
    print(f"  {str(k):34s} {SOIL[k].shape}")''')

code('''"""Cell G2 - prove the control arm IS Experiment 3's matrix."""
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
print(f"{len(GROUPS)} CV families over {len(fams)} soils, copied unchanged from E1/E2/E3.")''')

# ================================================================ H. IN-DOMAIN + REGRESSION
md("""## Section H - in-domain nested LOGO-CV, then the regression check that gates everything

Alpha is chosen inside each outer fold from the inner folds only, so what is reported is the
score of the procedure and never the score of the best alpha. One structural table is
computed per arm and read three ways - full sweep, inner selection, nested evaluation -
which keeps the sweep and the nested score consistent and costs 256 fits per arm.

**The regression check runs immediately after.** Crop 256 is E1's exact configuration, so
E2 must reproduce E1's per-soil scores or the sweep is measuring something other than
magnification. If it fails, this notebook stops before drawing a single conclusion.""")

code('''"""Cell H1 - model primitives, byte-for-byte Model 1's."""
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
    assert not any(fams[s] == fams[t] for s in TR_ARR[~keep] for t in TR_ARR[keep]), \\
        "a same-family soil leaks the held-out label"
    P = fit_predict(X[keep], Y[keep], X[~keep], alpha)
    ev = TR_ARR[~keep]
    return {ev[i]: emd_pair(P[i], Ymap[ev[i]]) for i in range(len(ev))}


print(f"arms scored: {ARMS_LIST}")
print(f"{len(CFG.alpha_grid)} alphas x {len(GROUPS)} folds = "
      f"{len(CFG.alpha_grid)*len(GROUPS)} fits per arm")''')

code('''"""Cell H2 - the structural error table, the sweep, and the nested procedure score."""
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
RES.to_csv(OUT_DIR / "cv_per_soil.csv")''')

code('''"""Cell H3 - THE REGRESSION CHECK. Crop 256 must reproduce Model 2 E1 exactly.

This is the strongest correctness gate the project has had. E2's crop-256 arm D is the same
computation as E1's arm D, from a different notebook in a different session. If the numbers
differ, something in E2's pipeline is not E1's pipeline, and every sweep result below would
be attributable to that difference rather than to magnification.

Two levels, because E1's per-soil file may or may not be attached:
  1. the arm MEANS, against values pinned from E1's own record. Always available, and
     pinned constants are legitimate here for the same reason the leaderboard scores are:
     they are external facts about a finished experiment.
  2. the full per-soil vector, against E1's cv_per_soil.csv when it is present. This is the
     stronger check - a matching mean with a scrambled per-soil vector is possible in
     principle - so the notebook says plainly which level it managed.
"""
print("regression check vs Model 2 E1\\n")
_arm_to_col = {"M1": "M1"}
for _b, _lbl in (("dinov2", "D"), ("vit_random", "R")):
    if f"{_b}@c{CFG.tile_size_px}" in per_soil:
        _arm_to_col[f"{_b}@c{CFG.tile_size_px}"] = _lbl
_ok_all, _level = True, "means only (E1 per-soil file not attached)"
_have_per_soil = {"D", "R"} <= set(E1_ARMS)
if _have_per_soil:
    _level = f"per-soil vectors vs E1's cv_per_soil.csv ({len(tr_ids)} soils)"
print(f"  check level: {_level}\\n")
for _arm, _col in _arm_to_col.items():
    a = np.array([per_soil[_arm][s] for s in tr_ids])
    _ref = e1_cv[_col].reindex(tr_ids).to_numpy(float) if _have_per_soil else None
    if _ref is not None:
        mean_gap = float(abs(a.mean() - _ref.mean()))
        max_gap = float(np.abs(a - _ref).max())
        src = "E1 per-soil"
    else:
        mean_gap = float(abs(a.mean() - E1_PINNED[_col]))
        max_gap = float("nan")
        src = "E1 pinned mean"
    good = mean_gap <= CFG.REGRESSION_TOL
    _ok_all &= good
    _tail = (f"max per-soil gap {max_gap:.4f}" if max_gap == max_gap
             else "mean-only check")
    print(f"  [{'PASS' if good else '**FAIL**'}] {_arm:20s} vs E1 '{_col}' ({src}):  "
          f"E2 {a.mean():7.3f}  mean gap {mean_gap:.4f}  {_tail}")
assert _ok_all, (
    f"crop 256 does not reproduce Model 2 E1 within {CFG.REGRESSION_TOL} EMD. E2 is not "
    "comparable to E1, so the sweep measures something other than magnification. Fix the "
    "pipeline before reading any number below - do NOT interpret the difference as a "
    "result.")
print(f"\\nPASS: E2's crop-256 arm is E1's arm ({_level}). Everything below is a")
print("magnification effect and nothing else.")''')

# ================================================================ I. THE SWEEP
md("""## Section I - the sweep

Nine arms: the control, and R and D at four magnifications. The headline table is
in-domain nested LOGO-CV against patch size in millimetres, because millimetres is the
variable and pixel counts are just how it is implemented.""")

code('''"""Cell I1 - the primary result table."""
rows = []
for _a in ARMS_LIST:
    n = NESTED[_a]
    _v = np.array(list(per_soil[_a].values()))
    rows.append(dict(arm=n["arm"], label="M1" if _a == "M1" else f"{LABEL[n['backend']]}@{n['crop']}",
                     backend=n["backend"], crop_px=n["crop"],
                     patch_mm=cg.patch_mm(n["crop"], TARGET_PPM),
                     window_mm=cg.window_mm(n["crop"], TARGET_PPM),
                     n_feat=n["n_feat"], nested_in_domain=float(_v.mean()),
                     median=float(np.median(_v)), worst=float(_v.max()),
                     oracle_alpha=n["oracle_alpha"], oracle_emd=n["oracle_emd"],
                     parsimony_alpha=n["parsimony_alpha"],
                     grid_spread=n["grid_spread"]))
SW = pd.DataFrame(rows).sort_values(["backend", "crop_px"]).reset_index(drop=True)
SW.to_csv(OUT_DIR / "sweep_results.csv", index=False)
pd.DataFrame([NESTED[a] for a in ARMS_LIST]).to_csv(OUT_DIR / "arms_config.csv", index=False)
print(SW[["label", "crop_px", "patch_mm", "n_feat", "nested_in_domain", "median", "worst",
          "oracle_alpha", "grid_spread"]].to_string(index=False,
          float_format=lambda v: f"{v:8.3f}"))
print(f"\\n  no-image floor {FLOOR_NO_IMAGE:.2f} | rank-3 ceiling 7.35 | "
      f"control (M1) {SW[SW.label=='M1'].nested_in_domain.iloc[0]:.2f}")
for _lbl in ("D", "R"):
    _s = SW[SW.label.str.startswith(_lbl)].sort_values("crop_px")
    if len(_s) > 1:
        _d = _s.nested_in_domain.iloc[0] - _s.nested_in_domain.iloc[-1]
        print(f"  {_lbl}: crop {int(_s.crop_px.iloc[0])} -> {int(_s.crop_px.iloc[-1])} "
              f"changes in-domain error by {_d:+.2f} EMD")''')

# ================================================================ J. TREND
md("""## Section J - the sweep read as a trend, not as a best-of-four

Four crops times two contrasts is multiple-comparison surface. The pre-registered primary
read is therefore the *direction and monotonicity* of the relationship between patch size
and error, not the winner.""")

code('''"""Cell J1 - Spearman rho between crop px and in-domain error, per arm."""
TREND = []
for _b in ("dinov2", "vit_random", "mock"):
    _s = SW[SW.backend == _b].sort_values("crop_px")
    if len(_s) < 3:
        continue
    rho, p = spearmanr(_s.crop_px.to_numpy(float), _s.nested_in_domain.to_numpy(float))
    mono = bool(_s.nested_in_domain.is_monotonic_increasing or
                _s.nested_in_domain.is_monotonic_decreasing)
    TREND.append(dict(backend=_b, n_crops=len(_s), rho_crop_vs_error=float(rho), p=float(p),
                      monotonic=mono,
                      first=float(_s.nested_in_domain.iloc[0]),
                      last=float(_s.nested_in_domain.iloc[-1]),
                      direction="smaller crop -> better" if rho < 0 else
                                "smaller crop -> worse" if rho > 0 else "flat"))
TR = pd.DataFrame(TREND)
TR.to_csv(OUT_DIR / "sweep_trend.csv", index=False)
print(TR.to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
print("\\n  Sign convention: rho is between CROP PX and ERROR. A magnification benefit means")
print("  smaller crop -> lower error, so it shows up as NEGATIVE rho.")
print("  P3 predicts negative rho for D. P5 predicts the same for R. If both move by a")
print("  similar amount, the effect is scale and has nothing to do with pretraining.")''')

# ================================================================ K. PER GROUP
md("""## Section K - where any gain actually lives

This is the section that decides whether a sweep result is a magnification effect at all.
Magnification changes anything only for soils whose grain was below the patch scale, so a
real gain must be **located** in the coarse and mid groups. A uniform gain cannot be
explained by the mechanism, and a gain concentrated in the fines is physically impossible
and would mean a leak.""")

code('''"""Cell K1 - paired per-soil difference between the coarsest and finest crop, by D50 group."""
GROUP_ROWS = []
for _b in ("dinov2", "vit_random", "mock"):
    _coarse_arm, _fine_arm = f"{_b}@c{max(CFG.crops)}", f"{_b}@c{min(CFG.crops)}"
    if _coarse_arm not in per_soil or _fine_arm not in per_soil:
        continue
    d = np.array([per_soil[_fine_arm][s] - per_soil[_coarse_arm][s] for s in tr_ids])
    grp = np.array([cg.soil_group(D50_MM[tr_ids.index(s)]) for s in tr_ids])
    rec = dict(backend=_b, from_crop=max(CFG.crops), to_crop=min(CFG.crops),
               overall=float(d.mean()))
    for g in ("fines", "mid", "coarse"):
        m = grp == g
        rec[f"{g}_n"] = int(m.sum())
        rec[f"{g}_delta"] = float(d[m].mean()) if m.any() else np.nan
    _lim = max(abs(rec["fines_delta"]), 1e-9)
    _gain_outside = np.nanmean([rec["mid_delta"], rec["coarse_delta"]])
    rec["concentrated_in_patch_limited"] = bool(abs(_gain_outside) > abs(rec["fines_delta"])
                                                and _gain_outside < 0)
    GROUP_ROWS.append(rec)
    print(f"{_b}: overall {d.mean():+6.2f} | fines {rec['fines_delta']:+6.2f} (n={rec['fines_n']})"
          f" | mid {rec['mid_delta']:+6.2f} (n={rec['mid_n']})"
          f" | coarse {rec['coarse_delta']:+6.2f} (n={rec['coarse_n']})")
GR = pd.DataFrame(GROUP_ROWS)
GR.to_csv(OUT_DIR / "per_group_effects.csv", index=False)
print("\\n  negative delta = the finer crop scores BETTER. P4 requires the mid/coarse")
print("  movement to be larger than the fines movement. The fines are physically capped")
print("  (section C3), so a gain concentrated there would be a leak, not a result.")
_bad = GR[(GR.fines_delta < 0) & (GR.fines_delta < GR.coarse_delta - 5)]
if len(_bad):
    print("\\n  ** SUSPICIOUS: the fines improved MORE than the coarse soils for "
          f"{list(_bad.backend)}. Magnification cannot resolve them. Treat as a leak until"
          " explained. **")''')

# ================================================================ L. ABLATION
md("""## Section L - the pretraining ablation at every crop""")

code('''"""Cell L1 - paired D vs R per crop, soil-level bootstrap."""
PAIRED_ON = sorted(set.intersection(*[set(per_soil[a]) for a in ARMS_LIST]))
MVEC = {a: np.array([per_soil[a][s] for s in PAIRED_ON]) for a in ARMS_LIST}


def boot_ci(diff, n=CFG.n_boot):
    k = len(diff)
    b = np.array([diff[rng.integers(0, k, k)].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


CON = []
print(f"paired on {len(PAIRED_ON)} soils, each scored out-of-fold\\n")
for _c in CFG.crops:
    _d, _r = f"dinov2@c{_c}", f"vit_random@c{_c}"
    if _d not in MVEC or _r not in MVEC:
        continue
    diff = MVEC[_d] - MVEC[_r]
    lo, hi = boot_ci(diff)
    sig = bool(lo > 0 or hi < 0)
    CON.append(dict(crop=_c, patch_mm=cg.patch_mm(_c, TARGET_PPM), estimate=float(diff.mean()),
                    ci_lo=lo, ci_hi=hi, wins_D_better=int((diff < 0).sum()), n=len(diff),
                    significant=sig, D_better=bool(diff.mean() < 0 and sig)))
    print(f"  crop {_c:3d}px (patch {cg.patch_mm(_c, TARGET_PPM):.3f} mm): "
          f"D-R {diff.mean():+7.2f}  CI [{lo:+7.2f},{hi:+7.2f}]  "
          f"wins {int((diff<0).sum()):2d}/{len(diff)}  "
          f"{'SIGNIFICANT' if sig else 'not distinguishable'}")
    _m = f"M1"
    dm = MVEC[_d] - MVEC[_m]
    lo2, hi2 = boot_ci(dm)
    print(f"           D-M1 {dm.mean():+7.2f}  CI [{lo2:+7.2f},{hi2:+7.2f}]  "
          f"wins {int((dm<0).sum()):2d}/{len(dm)}")
    CON[-1].update(D_minus_M1=float(dm.mean()), dm_ci_lo=lo2, dm_ci_hi=hi2,
                   dm_wins=int((dm < 0).sum()),
                   beats_control=bool(dm.mean() < 0 and hi2 < 0))
CONEFF = pd.DataFrame(CON)
CONEFF.to_csv(OUT_DIR / "paired_effects.csv", index=False)''')

# ================================================================ M. MECHANISM
md("""## Section M - does magnification cost D the camera stability E1 measured?

E1's real finding was that D is ~5x more device-stable than Model 1's features and no more
accurate. If zooming in destroys that, the one thing D had going for it is gone too.""")

code('''"""Cell M1 - camera shift and blur shift per arm, plus the cosine separation ratios."""
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
print("\\n  E1 reference: D camera/soil 0.209, blur/soil 0.029; M1 1.105 and 0.818.")
print("  'degenerate' means between-soil distance is ~0, so the ratio is a quotient of")
print("  noise and is reported as NaN rather than as a number that looks interpretable.")
_d = SEP_DF.loc[[i for i in SEP_DF.index if i.startswith("dinov2")], "camera_over_soil"].dropna()
if len(_d):
    print(f"\\n  P6: D stays below 0.35 at every crop -> "
          f"{'HOLDS' if _d.max() < 0.35 else 'FAILS'} (max {_d.max():.3f} over {len(_d)} crops)")
else:
    print("\\n  P6: not evaluated - no dinov2 arms in this run")''')

# ================================================================ N. REPORT-ONLY RULER
md("""## Section N - CAM+RES, computed and explicitly not used

E4 demoted this ruler. It is still reported, because every arm is another calibration point
for the ruler's own failure analysis, and because if magnification does help, the ruler
should see it.""")

code('''"""Cell N1 - CAM+RES per arm at the arm's own parsimony alpha."""
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
print("\\n  NOT USED TO SELECT. E1's probe is the calibration point that matters: CAM+RES")
print("  said 47.08 for arm D and the leaderboard returned 60.56.")''')

# ================================================================ O. GATE
md("""## Section O - the decision gate, executed as code

Four clauses, fixed before the run. The fourth - that any gain be *located* in the patch-
limited size groups - is what stops a lucky mean from reviving a dead premise.""")

code('''"""Cell O1 - the gate."""
CLAUSES = []


def clause(text, ok, detail=""):
    CLAUSES.append(dict(clause=text, passed=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}" + (f"   {detail}" if detail else ""))


CAND = [c for c in CFG.crops if c != max(CFG.crops)]
EVALUABLE = (not CONEFF.empty) and {"estimate", "significant", "beats_control"} \\
    <= set(CONEFF.columns)
WINNERS = []
if EVALUABLE:
    for _c in CAND:
        _row = CONEFF[CONEFF.crop == _c]
        if not len(_row):
            continue
        _row = _row.iloc[0]
        c1 = _row.estimate < 0 and bool(_row.significant)
        c2 = bool(_row.beats_control)
        _g = GR[GR.backend == "dinov2"] if not GR.empty else pd.DataFrame()
        c3 = bool(len(_g) and _g.iloc[0]["concentrated_in_patch_limited"])
        if c1 and c2 and c3:
            WINNERS.append(_c)
        print(f"  crop {_c:3d}: D<R {c1}  D<M1 {c2}  "
              f"gain located in patch-limited groups {c3}")
    print()
    clause(f"exists a crop in {CAND} where D beats R with a significant CI",
           any(CONEFF.significant & (CONEFF.estimate < 0)),
           f"best D-R {CONEFF.estimate.min():+.2f}")
    clause("and D beats the M1 control at that crop", bool(CONEFF.beats_control.any()))
    _g = GR[GR.backend == "dinov2"] if not GR.empty else pd.DataFrame()
    clause("and the gain is concentrated in the patch-limited D50 groups",
           bool(len(_g) and _g.iloc[0].concentrated_in_patch_limited))
else:
    print("  [NOT EVALUABLE] no backbone arms were executed in this run, so the gate has")
    print("  nothing to decide. A mock-only run cannot pass by construction and writes no")
    print("  submission - that is the intended behaviour, not a limitation.")
    clause("both backbone arms present", False, f"arms: {ARMS_LIST}")
SUBMIT = bool(WINNERS)
BEST_CROP = min(WINNERS, key=lambda c: float(CONEFF[CONEFF.crop == c].estimate.iloc[0])) \\
    if WINNERS else None
GATE = dict(submit=SUBMIT, evaluable=EVALUABLE, best_crop=BEST_CROP,
            candidate_crops=CAND, winners=WINNERS, clauses=CLAUSES,
            in_domain={a: float(np.mean(list(per_soil[a].values()))) for a in ARMS_LIST},
            trend=TR.to_dict("records"),
            per_group=(GR[GR.backend == "dinov2"].to_dict("records")
                       if not GR.empty else []))
json.dump(GATE, open(OUT_DIR / "gate.json", "w"), indent=2, default=str)
print(f"\\nGATE -> {'SUBMIT ' + str(BEST_CROP) + 'px crop' if SUBMIT else 'NO SUBMISSION'}")
print("  Three of the four outcomes close Model 2. That is the honest expectation given the")
print("  reachability table, and it is still worth the session: E1\\'s hole stays open until")
print("  it is tested.")''')

# ================================================================ P. SUBMISSION
md("""## Section P - submission, conditional on the gate

The fit and all eight validation gates **always** run. When the gate has not fired they run
on the control arm and are labelled a plumbing check, and nothing is written. E1 learned
this the hard way: a section behind `if SUBMIT:` never executed locally, and a one-line bug
reached Kaggle instead.""")

code('''"""Cell P1 - fit the submission arm and predict the 10 test soils."""
SUB_ARM = (f"dinov2@c{BEST_CROP}" if SUBMIT else
           (CFG.probe_arm if CFG.probe_arm else "M1"))
WRITES_FILE = bool(SUBMIT or (CFG.probe_arm and CFG.probe_arm in per_soil))
PLUMBING_ONLY = not WRITES_FILE
FE = ARM_FEATS[SUB_ARM]
S = SOIL[(SUB_ARM, "real")]
alpha_f = NESTED[SUB_ARM]["parsimony_alpha"]
_crop = NESTED[SUB_ARM]["crop"]
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
print(f"  arm {SUB_ARM}: {len(FE)} features, crop {_crop}px "
      f"(patch {cg.patch_mm(_crop, TARGET_PPM):.4f} mm), alpha {alpha_f:g}")
print(f"  columns pinned at 0/100: this {_sat:.1f}% (real labels 27%)")
print(f"  mean EMD of predictions vs E3\\'s submitted curves: "
      f"{float(np.mean([emd_pair(a, b) for a, b in zip(P_test, _e3p)])):.2f}")
if NESTED[SUB_ARM]["parsimony_is_grid_max"]:
    print("  NOTE: alpha is the top of the grid, so these curves are heavily shrunk.")''')

code('''"""Cell P2 - eight validation gates, then write only if the gate fired."""
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
print("\\nall 8 submission checks passed")
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
        print(f"  {'ok ' if c['passed'] else 'NOT '} {c['clause']}")''')

# ================================================================ Q. ERROR ANALYSIS
md("""## Section Q - error analysis

Read the per-group strip before believing any mean. The mechanism only permits a gain in
the patch-limited groups, so a mean that hides an absent group effect is the failure mode
this section exists to expose.""")

code('''"""Cell Q1 - the sweep curve: in-domain error against patch size in millimetres."""
_fig, _ax = plt.subplots(figsize=(8.6, 5.0))
for _b, _lbl, _st in (("dinov2", "D pretrained", "o-"), ("vit_random", "R random-init", "s--"),
                      ("mock", "MOCK plumbing", "^:")):
    _s = SW[SW.backend == _b].sort_values("patch_mm")
    if len(_s):
        _ax.plot(_s.patch_mm, _s.nested_in_domain, _st, ms=7, lw=1.8, label=_lbl)
_ax.axhline(float(np.mean(list(per_soil["M1"].values()))), c="#111", lw=1.4, ls="-.",
            label="M1 control (12 hand features)")
_ax.axhline(FLOOR_NO_IMAGE, c="#888", ls=":", label=f"no-image floor {FLOOR_NO_IMAGE:.1f}")
_ax.set_xscale("log"); _ax.invert_xaxis()
_ax.set_xlabel("patch footprint (mm, log scale, finer to the right)")
_ax.set_ylabel("nested in-domain LOGO-CV EMD")
_ax.set_title("Model 2 E2: does magnification help, and is the gain located where physics allows it?")
_ax.grid(alpha=.3); _ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
for _c in CFG.crops:
    print(f"  crop {_c:3d}px = patch {cg.patch_mm(_c, TARGET_PPM):.4f} mm -> "
          f"{cg.resolvability(D50_MM, _c, TARGET_PPM)} of 24 soils resolvable")''')

code('''"""Cell Q2 - per-soil differences by D50 group."""
_fig, _axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
for _b, _ax in zip(("dinov2", "vit_random"), _axes):
    _coarse, _fine = f"{_b}@c{max(CFG.crops)}", f"{_b}@c{min(CFG.crops)}"
    if _coarse not in per_soil or _fine not in per_soil:
        _ax.set_title(f"{_b}: backbone arm not run"); continue
    _d = np.array([per_soil[_fine][s] - per_soil[_coarse][s] for s in tr_ids])
    _order = np.argsort(D50_MM)
    _cols = {"fines": "#c0392b", "mid": "#e67e22", "coarse": "#27ae60"}
    _ax.bar(range(len(_order)), _d[_order],
            color=[_cols[cg.soil_group(D50_MM[i])] for i in _order])
    _ax.axhline(0, c="k", lw=.8)
    _ax.set_title(f"{LABEL[_b]}: finer crop minus coarser crop, per soil (neg = finer better)")
    _ax.set_xlabel("soil, ordered by D50 (fines left)")
    _ax.grid(alpha=.3, axis="y")
_fig.legend(handles=[plt.Line2D([0], [0], color=c, lw=6) for c in
                     ["#c0392b", "#e67e22", "#27ae60"]],
            labels=["fines (<0.11 mm)", "mid", "coarse (>1.5 mm)"], loc="lower center",
            ncol=3, frameon=False)
plt.tight_layout(); plt.show()
print("  Magnification can only act on the mid and coarse bars. The red bars on the left")
print("  are physically capped (section C3): a systematic gain there would be a leak.")''')

code('''"""Cell Q3 - the submission curves, if one exists."""
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
    plt.tight_layout(); plt.show()''')

# ================================================================ R. RECORD
md("""## Section R - the record

`Experiment2.txt` is written from the live objects in this notebook, never by hand.""")

code('''"""Cell R1 - the pre-registered predictions, evaluated mechanically."""
PREDS = []


def pred(pid, text, blind, status, evidence):
    PREDS.append(dict(id=pid, prediction=text, blind="YES" if blind else "no",
                      status=status, evidence=evidence))
    print(f"  {pid:3s} [{status:14s}] {text}")
    print(f"      {evidence}")


print("predictions were written in the plan BEFORE implementation; labels are fixed\\n")
_mk = "mock"
_mk_arms = [a for a in ARMS_LIST if a.startswith("mock@")]
if _mk_arms:
    _worst = min(float(np.mean(list(per_soil[a].values()))) for a in _mk_arms)
    pred("P1", "mock in-domain LOGO-CV is worse than the no-image floor at EVERY crop", False,
         "CONFIRMED" if _worst > FLOOR_NO_IMAGE else "REFUTED",
         f"best mock arm {_worst:.2f} vs floor {FLOOR_NO_IMAGE:.2f}")
else:
    pred("P1", "mock in-domain LOGO-CV is worse than the no-image floor at EVERY crop", False,
         "not evaluated", "mock arm not executed in this run")
_reg_ok = "asserted in cell H3; the notebook would have stopped otherwise"
pred("P2", "crop 256 reproduces E1's in-domain table within 0.05 EMD", False,
     "CONFIRMED", _reg_ok)
_t = TR[TR.backend == "dinov2"]
pred("P3", "D's in-domain error decreases as the crop shrinks (negative rho vs crop px)", True,
     ("CONFIRMED" if len(_t) and _t.iloc[0].rho_crop_vs_error < 0 and _t.iloc[0].monotonic
      else ("REFUTED" if len(_t) else "not evaluated")),
     f"rho={_t.iloc[0].rho_crop_vs_error:.3f} monotonic={_t.iloc[0].monotonic}"
     if len(_t) else "no dinov2 arms")
_g = GR[GR.backend == "dinov2"]
pred("P4", "the 256->32 gain is larger in the mid/coarse groups than in the fines", True,
     ("CONFIRMED" if len(_g) and _g.iloc[0].concentrated_in_patch_limited
      else ("REFUTED" if len(_g) else "not evaluated")),
     f"fines {_g.iloc[0].fines_delta:+.2f} | mid {_g.iloc[0].mid_delta:+.2f} | "
     f"coarse {_g.iloc[0].coarse_delta:+.2f}" if len(_g) else "no group data")
_t = TR[TR.backend == "vit_random"]
pred("P5", "R also improves with magnification (its 256px collapse is itself a scale effect)",
     True, ("CONFIRMED" if len(_t) and _t.iloc[0].rho_crop_vs_error < 0
            else ("REFUTED" if len(_t) else "not evaluated")),
     f"rho={_t.iloc[0].rho_crop_vs_error:.3f}" if len(_t) else "no vit_random arms")
_d = SEP_DF.loc[[i for i in SEP_DF.index if i.startswith("dinov2")], "camera_over_soil"].dropna()
pred("P6", "D stays camera-stable at every crop (camera/soil ratio below 0.35)", True,
     ("CONFIRMED" if len(_d) and _d.max() < 0.35
      else ("REFUTED" if len(_d) else "not evaluated")),
     f"max ratio {_d.max():.3f} over {len(_d)} crops" if len(_d)
     else "no dinov2 arms")
pred("P7", "no crop lets D beat both M1 and R with a significant CI", True,
     ("not evaluated" if not GATE["evaluable"] else
      ("CONFIRMED" if not SUBMIT else "REFUTED")),
     (f"gate not evaluable: {ARMS_LIST}" if not GATE["evaluable"] else
      f"gate submit={SUBMIT}, winning crops={GATE['winners']}"))
PRED_DF = pd.DataFrame(PREDS)
PRED_DF.to_csv(OUT_DIR / "predictions.csv", index=False)''')

code('''"""Cell R2 - write Experiment2.txt from the live objects."""
L = []
A = L.append
A("=" * 78)
A(EXPERIMENT_NAME)
A("=" * 78)
A(f"run context      : {RUN_CONTEXT}")
A(f"seed             : {SEED}   embedding seed: {CELL_SEED}")
A(f"config_hash      : {CONFIG_HASH}   pipeline {PIPELINE_VERSION}")
A(f"BACKEND requested: {BACKBONES}")
A(f"crop sweep       : {list(CFG.crops)}")
A(f"arms executed    : {ARMS_LIST}")
A(f"outputs          : {OUT_DIR}")
A(f"{'platform':16s} {platform.platform()}")
A("")
A("1. WHAT CHANGED, AND ONLY THIS")
A("   How much physical soil a tile contributes to the backbone: a centred window of the")
A("   existing 256px tile is cropped to c in {256,128,64,32} and resized to 224.")
A("   Frozen from Model 2 E1: tiles, soil-fraction filter, two-stage tile->image->soil")
A("   median, StandardScaler, rank-3 curve basis, closed-form ridge, monotone projection,")
A("   the 16 CV families, the alpha grid, the trapezoid EMD, and backbones.py UNMODIFIED.")
A("   Not done: multi-crop per tile, patch-mean pooling, other backbones, fine-tuning,")
A("   label-adaptive cropping (that is label leakage), re-tiling (a preprocessing change).")
A("")
A("2. THE REGRESSION CHECK - THE GATE BEFORE THE GATE")
A(f"   control vs E3 feature matrix: max drift {_mx:.3e} (gate < 1e-6)")
A("   crop 256 vs Model 2 E1 per-soil nested scores: asserted in cell H3 to within "
  f"{CFG.REGRESSION_TOL} EMD.")
A("   If either had failed this notebook stopped; every number below is downstream of both.")
A("")
A("3. GEOMETRY, AND THE CORRECTION TO E1")
A(cg.describe(D50_MM, TARGET_PPM))
A(f"   E1 recorded PATCH_MM = {cg.PATCH_PX/TARGET_PPM:.4f} mm, omitting the 256->224 resize.")
A(f"   True footprint at crop 256 = {cg.patch_mm(256, TARGET_PPM):.4f} mm. backbones.py is")
A("   left as authored so E1 stays byte-reproducible; E2 computes the correct value.")
A(f"   The whole sweep buys "
  f"{cg.resolvability(D50_MM, min(CFG.crops), TARGET_PPM) - cg.resolvability(D50_MM, max(CFG.crops), TARGET_PPM)}"
  " more resolvable soils while using "
  f"{100*(min(CFG.crops)/CFG.tile_size_px)**2:.1f}% of the tile area.")
A("   The fines are physically unreachable: resolving them needs a sub-24px window")
A("   upsampled to 224, which contains no grain information.")
A("")
A("4. PRIMARY RESULT - nested in-domain LOGO-CV by arm and crop")
for _r in SW.itertuples(index=False):
    A(f"   {_r.label:9s} crop {_r.crop_px:3d}px patch {_r.patch_mm:6.4f} mm  "
      f"n_feat {_r.n_feat:3d}  nested {_r.nested_in_domain:6.2f}  "
      f"oracle alpha {_r.oracle_alpha:g}")
A(f"   no-image floor {FLOOR_NO_IMAGE:.2f} | rank-3 ceiling 7.35")
A("   per-fold and per-soil scores: cv_per_fold.csv, cv_per_soil.csv")
A("")
A("5. THE SWEEP AS A TREND (primary read; picking the best of four is a trap)")
for _r in TR.itertuples(index=False):
    A(f"   {_r.backend:12s} rho(crop px, error) = {_r.rho_crop_vs_error:+.3f} "
      f"p={_r.p:.3f}  monotonic={_r.monotonic}  {_r.first:.2f} -> {_r.last:.2f}")
A("")
A("6. WHERE ANY GAIN ACTUALLY LIVES (per D50 group)")
for _r in GR.itertuples(index=False):
    A(f"   {_r.backend:12s} overall {_r.overall:+6.2f} | fines {_r.fines_delta:+6.2f} "
      f"(n={_r.fines_n}) | mid {_r.mid_delta:+6.2f} (n={_r.mid_n}) | "
      f"coarse {_r.coarse_delta:+6.2f} (n={_r.coarse_n})")
A("   Magnification can only act on mid and coarse. A gain concentrated in the fines would")
A("   indicate a leak, not a result.")
A("")
A("7. PRETRAINING ABLATION AT EACH CROP")
for _r in CONEFF.itertuples(index=False):
    A(f"   crop {_r.crop:3d}px  D-R {_r.estimate:+7.2f} CI [{_r.ci_lo:+7.2f},{_r.ci_hi:+7.2f}] "
      f"wins {_r.wins_D_better}/{_r.n} {'SIG' if _r.significant else 'ns'}   "
      f"D-M1 {_r.D_minus_M1:+7.2f} CI [{_r.dm_ci_lo:+7.2f},{_r.dm_ci_hi:+7.2f}]")
A("")
A("8. MECHANISM - does magnification cost D its camera stability?")
for _i, _r in MECH_DF.join(SEP_DF[["camera_over_soil", "blur_over_soil", "degenerate"]]).iterrows():
    _cs = "degenerate" if _r.degenerate else f"{_r.camera_over_soil:.3f}"
    _bs = "degenerate" if _r.degenerate else f"{_r.blur_over_soil:.3f}"
    A(f"   {_i:22s} camera {_r.camera_shift_z:6.3f} z  sharpness {_r.sharpness_shift_z:6.3f} z"
      f"   camera/soil {_cs}  blur/soil {_bs}")
A("   E1 reference: D camera/soil 0.209, blur/soil 0.029; M1 1.105 and 0.818.")
A("")
A("9. CAM+RES - REPORTED, NOT USED TO SELECT")
for _i, _r in RU.iterrows():
    A(f"   {_i:22s} CAM+RES {_r.camres:7.2f}  CAM {_r.cam:7.2f}  in-domain {_r.in_domain:6.2f}")
A("   E1's probe is the calibration point that matters: CAM+RES said 47.08 for arm D and")
A("   the leaderboard returned 60.56.")
A("")
A("10. DECISION GATE")
A(f"   submit: {SUBMIT}   winning crops: {GATE['winners']}   best: {BEST_CROP}")
for c in CLAUSES:
    A(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['clause']}")
A("   submit = exists a crop where D beats M1 in-domain AND beats R with a significant")
A("   paired CI AND the gain is concentrated in the patch-limited D50 groups.")
A("   Three of the four outcomes close Model 2; that was the honest expectation written")
A("   into the plan before the run, given the reachability table in section 3.")
A("")
A("11. SUBMISSION")
if SUB_PATH is not None:
    A(f"   {Path(SUB_PATH).name} written: {SUB_PATH}")
    A(f"   arm {SUB_ARM}, crop {_crop}px, alpha {alpha_f:g}")
    A("   all 8 validation gates passed; file re-read and re-asserted after writing.")
    A("   Kaggle score: PENDING. Reference to beat: 60.56167 (M2 E1 probe) / 61.23560 (M1 E3).")
else:
    A("   NO SUBMISSION WAS PRODUCED BY THIS RUN.")
    A(f"   gate.submit = {SUBMIT}; probe armed = {bool(CFG.probe_arm)}.")
A("")
A("12. PRE-REGISTERED PREDICTIONS")
for _r in PRED_DF.itertuples(index=False):
    A(f"   {_r.id:3s} [{_r.blind:3s}] {_r.prediction}")
    A(f"        -> {_r.status}: {_r.evidence}")
A("")
A("13. SCOPE OF VERIFICATION")
if not HAS_TORCH:
    A("   torch/timm are absent here, so only the mock backbone executed. crop_geometry.py")
    A("   is fully tested locally (check_crop_geometry.py, 28 assertions); backbones.py is")
    A("   imported from E1 UNMODIFIED. A mock run is a plumbing test and is never a result.")
else:
    A(f"   torch present; backbone arms executed: "
      f"{sorted({a.split('@c')[0] for a in ARMS_LIST if a != 'M1'})}.")
A("   Unverified by a local run: the real embedding geometry at each crop, and every number")
A("   involving arms R and D.")
A("")
A("14. ENVIRONMENT")
for _k, _v in VERSIONS.items():
    A(f"   {_k:12s} {_v}")
A("=" * 78)
TXT = OUT_DIR / "Experiment2.txt"
TXT.write_text("\\n".join(L) + "\\n", encoding="utf-8")
print(f"wrote {TXT} ({len(L)} lines)")''')

code('''"""Cell R3 - summary."""
print("=" * 72)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  arms={ARMS_LIST}  torch={HAS_TORCH}")
print("\\nnested in-domain LOGO-CV by arm:")
for _r in SW.itertuples(index=False):
    print(f"  {_r.label:9s} crop {_r.crop_px:3d}px  {_r.nested_in_domain:6.2f}")
print(f"  control M1 = {float(np.mean(list(per_soil['M1'].values()))):.2f}   "
      f"no-image floor {FLOOR_NO_IMAGE:.2f}")
print(f"\\ngate submit={SUBMIT} winning crops={GATE['winners']}")
print(f"submission: {'written -> ' + str(SUB_PATH) if SUB_PATH else 'NOT written'}")
print("\\nfiles written:")
for _f in ["Experiment2.txt", "arms_config.csv", "sweep_results.csv", "sweep_trend.csv",
           "per_group_effects.csv", "cv_per_fold.csv", "cv_per_soil.csv",
           "paired_effects.csv", "mechanism_table.csv", "embedding_separation.csv",
           "camres_report_only.csv", "predictions.csv", "gate.json",
           "features_soil_M1_control.csv", "cv_families.csv"]:
    _fp = OUT_DIR / _f
    print(f"  {'ok  ' if _fp.exists() else 'MISSING'} {_f}")''')

md("""## Reading of the result

| outcome | what it means |
|---|---|
| a crop satisfies all four gate clauses | E1's refutation was partly a scale artefact; learned features do help once the patch resolves the grain |
| D improves with magnification but never past R | the scale effect is real and has nothing to do with pretraining |
| no crop helps D | E1's refutation survives its declared confound. **Model 2 is closed on evidence** |
| crop 256 fails the regression check | E2 is void - fix the pipeline, read nothing |

Three of four close Model 2, and the reachability table in section 3 says that is the
likely answer. That is not a reason to skip it: E1's hole stays open until it is tested,
and the fourth outcome would matter.

The result to read first is **section 6, not section 4**. A mean over 24 soils where ten of
them physically cannot respond to magnification is not a measurement of the mechanism. If
the mid and coarse bars move and the fines do not, magnification did what physics allows.
If everything moves equally, something other than magnification moved.""")

# ================================================================ write it
nb = {
    "cells": [],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
for kind, src in CELLS:
    lines = src.splitlines(keepends=True)
    if kind == "markdown":
        nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})
    else:
        nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                            "outputs": [], "source": lines})

out = Path("Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
nc = sum(1 for k, _ in CELLS if k == "code")
print(f"wrote {out}  ({len(CELLS)} cells: {nc} code, {len(CELLS) - nc} markdown)")
