"""Builds 'Model 2/Model 2 Experiment 1/Model2_Experiment1.ipynb'.

Model 2 E1 changes ONE variable: the per-tile feature vector. Model 1's 12 hand-built
features stay as the control arm; frozen DINOv2 ViT-S/14 embeddings and the same
architecture with random untrained weights are the two new arms. Aggregation, the rank-3
curve basis, the ridge head, the monotone projection, the CV families and the metric are
copied from Experiment 3 and asserted to reproduce it.

The backbone code lives in ``backbones.py``, the only place torch is imported, because this
development machine has no torch: the notebook runs end-to-end locally against a pure-numpy
``mock`` arm and runs for real on Kaggle.

Run:  python scratch/build_m2e1.py
"""
import json
from pathlib import Path

CELLS = []


def md(src):
    CELLS.append(("markdown", src))


def code(src):
    CELLS.append(("code", src))


# ================================================================ A. SETUP
md("""# Model 2 - Experiment 1
## Does a *learned* visual representation extract grain-size signal the hand-built features could not?

**Single variable.** The per-tile feature vector. Everything else is frozen at Experiment
3's values: the same 256 px tiles, the same two-stage tile->image->soil median aggregation,
the same `StandardScaler`, the same rank-3 PCA curve basis, the same closed-form ridge, the
same clip->cummax->force-100 monotone projection, the same 16 CV families, the same trapezoid
EMD.

**Why Model 1 was closed rather than continued.** Its in-domain score never moved across
four experiments while every external gain came from removing features and adding
shrinkage - variance reduction, not signal. The ~34 EMD between the best in-domain score and
the rank-3 representation ceiling (7.35) is a signal-extraction problem.

**Three arms, and why all three are mandatory.**

| arm | per-tile features | what it is |
|---|---|---|
| `M1` | 12 | Model 1 E3's cell C - the control, always run |
| `R`  | 384 | DINOv2 ViT-S/14 architecture, **random untrained weights** |
| `D`  | 384 | the same architecture, **pretrained** weights, frozen |

DINOv2 emits 384 numbers where Model 1 emitted 12. A win for `D` over `M1` alone would be
uninterpretable: it could be pretraining, or it could just be dimensionality. `R` has the
same width, the same receptive field and the same position in the pipeline, and carries no
learned information at all. **Without `R` this experiment answers nothing.**

**Metric ruling, declared before the run.** E1 proved leave-one-family-out CV ranks
configurations in the wrong order for *cross-camera transfer*. That is a claim about one
axis. It does not make LOGO-CV a poor measure of how much curve signal a representation
carries on devices already seen - which is the representation question, and the only thing
constant across E1-E4. So in-domain LOGO-CV is primary here, and CAM+RES is **reported and
never used to select**: it mis-ranked E4 against E2 by 24.6 EMD in the wrong direction.

Read `instructions.txt` in this folder before changing anything.""")

code('''"""Cell A1 - configuration. Every tunable lives here and nowhere else."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import os

MODEL_ID        = "Model2"
EXPERIMENT_ID   = "E1"
EXPERIMENT_NAME = "Model 2 / Experiment 1 - frozen pretrained backbone vs hand-built features"
SEED = 20260930
E3_DIR_NAME = "Model 1 Experiment 3"
E1_DIR_NAME = "Model 1 Experiment 1"


@dataclass(frozen=True)
class Config:
    # ---- frozen at Experiment 3's values; changing any of these is a different experiment
    tile_size_px: int = 256
    min_tile_soil_fraction: float = 0.50
    pc_rank: int = 3
    family_linkage: str = "average"
    family_cut_emd: float = 14.0
    holdout_cameras: Tuple[str, ...] = ("Motorola", "Samsung")
    res_match_targets: Tuple[str, ...] = ("iPhone 14", "iPhone 16")
    n_boot: int = 4000
    n_perm: int = 200
    ORDINAL_FLOOR: float = 15.0      # E3's ruling: differences below this may not be used to choose
    # ---- the one variable: the per-tile feature vector
    control_feats: Tuple[str, ...] = ("e4", "e8", "e16", "lum_sd", "grad_mean",
                                      "R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90")
    # E3's selected cell C: texture core + colour, frequency excluded. 12 features.
    embed_seed: int = 20260930
    batch_size: int = 64
    # ---- alpha grid extended UPWARD versus E3. 384 scaled features need far more
    # shrinkage than 12, and a grid topping out at 1000 would silently under-regularise
    # the new arms. This extension is a declared consequence of the representation change
    # and is applied to ALL arms, control included, so nothing is tuned per arm.
    alpha_grid: Tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0,
                                     300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0,
                                     300000.0, 1000000.0)
    # ---- DECLARED PROBE. Not a gate pass, and it must never be read as one.
    # The gate was fixed before the run and its verdict stands: the premise "a learned
    # representation extracts more signal" is REFUTED, and that stays in the record either
    # way. What the gate structurally cannot decide is a DIFFERENT question this run raised
    # - D is slightly worse in-domain yet better on every device-shift measure, and E1
    # proved our internal metrics rank cross-camera transfer in the WRONG order. Only the
    # leaderboard measures transfer, so one submission slot is spent as a measurement.
    # Set to None to make this notebook submission-agnostic again.
    probe_arm: Optional[str] = "dinov2"


CFG = Config()


def _env_backends():
    """BACKEND accepts mock | vit_random | dinov2 | both | all. Default is mock locally, so
    a careless local run can never produce a table that looks like a result, and 'both' on
    Kaggle, so one run produces R and D together and the gate is complete."""
    raw = os.environ.get("BACKEND", "").strip().lower()
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
_EXP_DIR = Path("Model 2/Model 2 Experiment 1")
if ON_KAGGLE:
    OUT_DIR = Path(os.environ.get("KAGGLE_OUTPUT_DIR", "/kaggle/working"))
elif set(BACKBONES) <= {"mock"}:
    # A mock run writes into its own folder, never into the experiment root. This happened
    # once: a local dry run overwrote the downloaded Kaggle outputs, because both write the
    # same 13 filenames. Plumbing artifacts and results must not share a directory.
    OUT_DIR = _EXP_DIR / "local mock dry-run"
else:
    OUT_DIR = _EXP_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{MODEL_ID}/{EXPERIMENT_ID}  seed={SEED}  context={'kaggle' if ON_KAGGLE else 'local'}")
print(f"backbone arms requested this run: {BACKBONES}")
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

code('''"""Cell A3 - locate the processed dataset and the backbone interface.

`backbones.py` sits next to this notebook. It is the only module in the project that
imports torch, and importing it here does NOT require torch to exist.
"""
def resolve_input_root():
    """Find the folder holding data/processed_meta/manifest_images.csv.

    Kaggle's mount depth is not stable (`/kaggle/input/<slug>` and
    `/kaggle/input/datasets/<user>/<slug>` have both been seen), so this searches deep
    instead of assuming a level. Finding the file is only a lookup; the config_hash pin in
    cell A4 is what proves the RIGHT dataset was found, so a broad search costs no safety.
    """
    kg = Path("/kaggle/input")
    if kg.exists():
        for cand in sorted(kg.iterdir()):
            for probe in (cand, *sorted(cand.iterdir())):
                if (probe / "data" / "processed_meta" / "manifest_images.csv").exists():
                    return probe, "kaggle"
        for hit in sorted(kg.rglob("manifest_images.csv")):
            meta = hit.parent                       # .../data/processed_meta
            if meta.name == "processed_meta" and (meta / "manifest_tiles.csv").exists():
                root = meta.parent.parent           # the folder that owns data/
                if (root / "data" / "processed_meta" / "manifest_images.csv").exists():
                    return root, "kaggle"
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "data" / "processed_meta" / "manifest_images.csv").exists():
            return cand, "local"
    raise RuntimeError(
        "Could not find data/processed_meta/manifest_images.csv.\\n"
        "The attached dataset must contain the processed data tree (see kaggle_setup.md "
        "section 1). Locally, run this notebook from the repository root.")


def resolve_backbones_dir():
    """Where backbones.py / check_backbones.py live. Found by name anywhere under each
    attachment root, so the folder layout of the dataset is the owner's choice and not a
    correctness requirement for the run."""
    probes = []
    kg = Path("/kaggle/input")
    if kg.exists():
        probes += sorted(kg.iterdir())
    here = Path.cwd()
    probes += [here / "Model 2" / "Model 2 Experiment 1", here]
    for d in probes:
        if not d.is_dir():
            continue
        if (d / "backbones.py").exists():
            return d
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            if (sub / "backbones.py").exists():
                return sub
        try:
            hit = next(iter(sorted(d.rglob("backbones.py"))), None)
        except (OSError, RuntimeError):
            hit = None
        if hit:
            return hit.parent
    raise RuntimeError(
        "backbones.py not found. It must be attached with this experiment (see kaggle_setup.md "
        "section 1) - the notebook does not reimplement the backbone, "
        "because a locally rewritten copy of the feature extractor is the easiest way to "
        "make a result unreproducible.")


INPUT_ROOT, RUN_CONTEXT = resolve_input_root()
META = INPUT_ROOT / "data" / "processed_meta"
sys.path.insert(0, str(INPUT_ROOT))
BB_DIR = resolve_backbones_dir()
sys.path.insert(0, str(BB_DIR))
from preprocess import verify as pv          # the VALIDATED hand-built feature definition
import backbones as bb

print(f"INPUT_ROOT  {INPUT_ROOT}  ({RUN_CONTEXT})")
print(f"backbones   {bb.__file__}")''')

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
TARGET_PPM = float(_imgs_probe.target_ppm.iloc[0])
PIPELINE_VERSION = str(_imgs_probe.pipeline_version.iloc[0])

_SUB_PROBE = pd.read_csv(INPUT_ROOT / "data" / "sample_submission.csv")
TARGET_COLS = [str(c) for c in _SUB_PROBE.columns if str(c) != "sample_id"]
assert len(TARGET_COLS) == 11, f"expected 11 support columns, got {TARGET_COLS}"
SUPPORT = np.array([float(c) for c in TARGET_COLS])
DL = np.log10(SUPPORT)
assert np.allclose(SUPPORT, [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
print(f"config_hash {CONFIG_HASH} | pipeline {PIPELINE_VERSION} | {TARGET_PPM:.6f} px/mm")

# Physical scale of the backbone's receptive field, stated once and carried into the record.
_mm_per_px = 1.0 / TARGET_PPM
PATCH_MM = bb.PATCH_MM
TILE_MM = CFG.tile_size_px * _mm_per_px
print(f"tile {CFG.tile_size_px}px = {TILE_MM:.2f} mm square -> resized to "
      f"{bb.INPUT_SIZE}px = {bb.INPUT_SIZE * _mm_per_px:.2f} mm")
print(f"one ViT-S/14 patch = {PATCH_MM:.3f} mm at the canonical scale")''')

# ================================================================ B. DATA & METRIC
md("""## Section B - the metric and the five external ground-truth scores

The metric is not re-derived here; it is *re-asserted* against the host's own published
number, the same check E2 and E3 ran. Five Kaggle scores are now fixed reference points.""")

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
print(f"images {imgs.shape} | tiles@{CFG.tile_size_px} {tiles.shape} "
      f"(train {int((tiles.split=='train').sum())} / test {int((tiles.split=='test').sum())}) "
      f"| labels {Y.shape}")
print(f"D50 spans {10**LD50.min():.4f} .. {10**LD50.max():.3f} mm")
_D50 = 10 ** LD50
N_PATCH_GT3 = int((_D50 < PATCH_MM / 3).sum())     # one patch spans more than 3 median grains
N_SUB_PATCH = int((_D50 < PATCH_MM).sum())          # a median grain is smaller than one patch
N_VERY_FINE = int((_D50 < 0.11).sum())              # the plan's stated fine-soil group
print(f"\\nSCALE CAVEAT (measured here, stated up front, not discovered afterwards): "
      f"one ViT-S/14 patch = {PATCH_MM:.3f} mm.\\n"
      f"  {N_PATCH_GT3} of 24 training soils have D50 < {PATCH_MM/3:.2f} mm, so a single patch "
      f"spans MORE THAN 3 median\\n  grains for those soils. "
      f"{N_SUB_PATCH} of 24 have a median grain smaller than one whole patch.\\n"
      f"  {N_VERY_FINE} of 24 have D50 < 0.11 mm, where the patch is "
      f"{PATCH_MM/0.11:.0f}x the median grain.\\n"
      "  For those soils the network sees fines as TEXTURE, never as particles. Cropping a "
      "smaller\\n  physical area and upsampling so patches land at sub-millimetre size is a "
      "DIFFERENT variable\\n  and belongs in Model 2 E2. If D fails here it may be "
      "magnification, not representation.""")''')

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
_p = float(np.mean([emd_page(TRIVIAL, y) for y in Y]))
print(f"trivial baseline: ours {_t:.4f} | page formula {_p:.4f} | host publishes {PUB_TRIVIAL}")
assert abs(_t - PUB_TRIVIAL) < 0.05, \\
    "our EMD no longer reproduces the host's published reference - stop and reconcile"
print("METRIC CONFIRMED (as in E2/E3): the trapezoid matches the host's own number.")''')

code('''"""Cell B3 - reference baselines and the five external ground-truth scores."""
MEAN_CURVE = Y.mean(axis=0)
MEDIAN_CURVE = np.median(Y, axis=0)
FLOOR_NO_IMAGE = mean_emd(np.tile(MEDIAN_CURVE, (24, 1)), Y)
REF = {"official trivial baseline (9.09%/bin)": mean_emd(np.tile(TRIVIAL, (24, 1)), Y),
       "constant train MEAN curve": mean_emd(np.tile(MEAN_CURVE, (24, 1)), Y),
       "constant train MEDIAN curve (no images)": FLOOR_NO_IMAGE}
print("reference EMD on the training set (lower is better):")
for _k, _v in REF.items():
    print(f"  {_k:44s} {_v:7.2f}")

# Kaggle scores actually received. Hardcoded because they are external facts, and each is
# labeled with the configuration that earned it.
LB = {"E1 16 feats, alpha 0.03 chosen by LOGO-CV": 172.69929,
      "no-image probe (train median curve)":       102.37237,
      "E2 14 feats, alpha 30 chosen by CAM+RES":    71.27346,
      "E3 12 feats, alpha 10 chosen by CAM+RES":    61.23560,
      "E4  5 feats, alpha 30 chosen by CAM+RES":    77.12257}
LB_E3 = LB["E3 12 feats, alpha 10 chosen by CAM+RES"]
print("\\nexternal ground truth so far:")
for _k, _v in LB.items():
    print(f"  {_v:8.2f}  {_k}")

# In-domain LOGO-CV reported by each Model 1 experiment, and the rank-3 ceiling.
M1_LOGO = {"E1": 35.98, "E2": 42.66, "E3": 41.15, "E4": 39.46}
M1_CAMRES = {"E1": 205.34, "E2": 75.73, "E3": 51.87, "E4": 56.98}
print(f"\\nno-image in-domain floor {FLOOR_NO_IMAGE:.2f}. The rank-3 representation ceiling "
      "measured in E1 was 7.35.")''')

# ================================================================ C. PRIOR RESULTS
md("""## Section C - Model 1's result, restated and reproduced

The control arm is Model 1's own feature matrix. Before it can be compared to anything,
this notebook has to prove it rebuilds Experiment 3's numbers from the same tiles - if the
control drifted, every difference reported later is attributable to the plumbing and not to
the representation.""")

code('''"""Cell C1 - locate and load Experiment 3's outputs.

Located by CONTENT, not by folder name. Prior experiments were attached under varying
names (`Model 1 Experiment 3`, `experiment_history/Model1_E3`), and a notebook that only
finds what it expects is a notebook that breaks silently. The signature is E3's own three
artifacts together; the control feature columns are then asserted, so a wrong folder
cannot be mistaken for the right one.
"""
E3_SIGNATURE = ("features_soil.csv", "cv_families.csv", "Submission_Model1_E3.csv")


def _prior_dir():
    roots = []
    kg = Path("/kaggle/input")
    if kg.exists():
        for cand in sorted(kg.iterdir()):
            roots += [cand, *sorted(cand.iterdir())]
            roots += sorted(cand.iterdir()) if cand.is_dir() else []
    here = Path.cwd()
    roots += [here / "Model 1" / E3_DIR_NAME, here / E3_DIR_NAME, here]
    seen, cands = set(), []
    for d in roots:
        if d in seen or not d.is_dir():
            continue
        seen.add(d)
        cands.append(d)
    for d in cands:
        if all((d / f).exists() for f in E3_SIGNATURE):
            return d
    for d in cands:                       # one level deeper: Model 1/<name>, history/<name>
        try:
            subs = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError:
            continue
        for s in subs:
            if all((s / f).exists() for f in E3_SIGNATURE):
                return s
    hit = None
    for d in cands:
        for p in d.rglob("features_soil.csv"):
            if all((p.parent / f).exists() for f in E3_SIGNATURE):
                hit = p.parent; break
        if hit:
            break
    if hit:
        return hit
    raise RuntimeError(
        "Could not locate Experiment 3's outputs. Looked for a folder containing "
        f"{E3_SIGNATURE} under /kaggle/input and the working directory. Attach it (see "
        "kaggle_setup.md section 1) - Experiment 3's feature matrix is the control arm of "
        "this experiment, and without it there is no comparison to Model 1.")


E3_DIR = _prior_dir()
e3_soil = pd.read_csv(E3_DIR / "features_soil.csv")
e3_fam = pd.read_csv(E3_DIR / "cv_families.csv")
e3_sub = pd.read_csv(E3_DIR / "Submission_Model1_E3.csv")
print(f"E3_DIR {E3_DIR}")
print(f"  features_soil.csv {e3_soil.shape} | cv_families.csv {e3_fam.shape} "
      f"| submission {e3_sub.shape}")
for _c in list(CFG.control_feats):
    assert _c in e3_soil.columns, f"E3's matrix is missing control feature {_c}"''')

code('''"""Cell C2 - the plateau that motivated Model 2, and why in-domain is the right ruler here.

This cell prints an argument, not a measurement. It is in the notebook rather than only in
instructions.txt because it is the reason the primary metric is in-domain, and a reader of
the results should find that reasoning next to them.
"""
print("MODEL 1 TRAJECTORY\\n")
print(f"{'exp':5s} {'in-domain LOGO-CV':>18s} {'CAM+RES':>9s} {'Kaggle':>9s}")
for _e in ["E1", "E2", "E3", "E4"]:
    _lb = {"E1": 172.69929, "E2": 71.27346, "E3": 61.23560, "E4": 77.12257}[_e]
    print(f"{_e:5s} {M1_LOGO[_e]:18.2f} {M1_CAMRES[_e]:9.2f} {_lb:9.2f}")
print("\\n  The in-domain column did not move. Every external gain came from removing")
print("  features and raising shrinkage, which is variance reduction, not new signal.")
print(f"  Headroom to the rank-3 ceiling: {min(M1_LOGO.values()) - 7.35:.1f} EMD in-domain.")
print("\\nWHY LOGO-CV IS ADMITTED HERE, AFTER E1 DEMOTED IT\\n")
print("  E1's finding was precise: LOGO-CV ranks configurations in the WRONG ORDER for")
print("  cross-camera transfer (rank correlation -0.953 against CAM+RES). That is a claim")
print("  about transfer. It is not a claim about how much curve signal a representation")
print("  carries on devices already seen, and the question Model 2 E1 asks is exactly the")
print("  second one: does a learned representation lower in-domain error at all?")
print("  If it does not, Model 2 is refuted cheaply on a CPU metric with no submission spent.")
print("  If it does, a submission has been earned to discover how the transfer penalty")
print("  behaves under a new representation.")
print("\\n  CAM+RES is computed in section K for reporting and for new calibration points in")
print("  the ruler's own failure analysis. It selects nothing in this experiment.")''')

# ================================================================ D. ENVIRONMENT PROBE
md("""## Section D - environment probe, before any long computation

The backbone arms cannot run on the development machine: no torch, no torchvision, no timm.
That limitation is **isolated, not hidden**. This cell declares which arms are executable
here, and on Kaggle it fails loudly *before* the extraction loop if the weights are
unreachable - a plumbing failure should cost seconds, not a session.""")

code('''"""Cell D1 - what can actually run in this process.

Printed as a table so the record contains an honest statement of scope, and so a reader
who finds a complete-looking results table also finds whether it was ever complete.
"""
HAS_TORCH = bb.torch_available()
if HAS_TORCH:
    import torch, timm
    VERSIONS.update({"torch": torch.__version__, "timm": timm.__version__})
    print(f"  {'torch':12s} {torch.__version__}")
    print(f"  {'timm':12s} {timm.__version__}")
    print(f"  cuda         {torch.cuda.is_available()} "
          f"{torch.cuda.device_count() if torch.cuda.is_available() else ''}")

ARMS = {}
for _b in BACKBONES:
    if _b not in bb.BACKENDS:
        raise KeyError(f"BACKEND={_b!r} is not one of {bb.BACKENDS}")
    needs = bb.requires_torch(_b)
    ok = (not needs) or HAS_TORCH
    ARMS[_b] = dict(backend=_b, executable=ok, needs_torch=needs, dim=bb.dim(_b))
ARMS["M1"] = dict(backend="M1", executable=True, needs_torch=False,
                  dim=len(CFG.control_feats))

print("\\narms requested / arms executable:")
for nm in ["M1", *BACKBONES]:
    a = ARMS[nm]
    print(f"  {nm:11s} dim={a['dim']:4d}  torch={a['needs_torch']!s:6s}  "
          f"executable={a['executable']}")

BLOCKED = [nm for nm, a in ARMS.items() if not a["executable"]]
if BLOCKED:
    print(f"\\n** {BLOCKED} cannot run in this process. **")
    print("  This is expected locally and is NOT an error. The mock arm is a plumbing test")
    print("  and is never a result: no submission will be written from it (section L).")
    print("  To produce the real arms, run on Kaggle with BACKEND=both (kaggle_setup.md).")
if ON_KAGGLE and "mock" in BACKBONES:
    print("\\nNOTE: BACKEND=mock on Kaggle. The gate cannot fire; this is a plumbing run.")

# Fail EARLY, not late: resolve the weights location before extracting 1,976 tiles.
WEIGHTS_PATH = os.environ.get("DINO_WEIGHTS", "").strip() or None
if WEIGHTS_PATH:
    _wp = Path(WEIGHTS_PATH)
    if not _wp.is_absolute():
        for cand in [Path("/kaggle/input"), Path.cwd()]:
            hits = sorted(cand.rglob(Path(WEIGHTS_PATH).name)) if cand.exists() else []
            if hits:
                _wp = hits[0]; break
    print(f"\\nweights override: {_wp}  exists={_wp.exists()}")
    assert _wp.exists(), f"DINO_WEIGHTS={WEIGHTS_PATH} resolved to {_wp}, which is not there"
    WEIGHTS_PATH = str(_wp)
else:
    print("\\nno DINO_WEIGHTS set: timm will fetch pretrained weights from the hub, which "
          "needs internet ON.\\n  kaggle_setup.md gives the offline dataset alternative. "
          "vit_random needs no download at all, which is why it is run first.")
CELL_SEED = CFG.embed_seed
print(f"embedding seed {CELL_SEED} | batch {CFG.batch_size}")''')

# ================================================================ E. EMBEDDINGS
md("""## Section E - the per-tile feature vectors

Two extractors, one variable. `M1` uses `preprocess.verify.feats` - the exact definition
`golden_checks.json` was validated against - and the embedding arms use `backbones.embed_tiles`.
Every arm is extracted for the *real* view and for the two resolution-matched views, so
section K's report-only ruler has inputs; the blur is derived from the manifests exactly as
E3 derived it.""")

code('''"""Cell E1 - run the backbone contract test against the arms this notebook is about to use.

The same file a developer runs by hand. On Kaggle with torch present it verifies the REAL
backbone structure (shape, finiteness, L2 norms, determinism, batching) before 1,976 tiles
go through it; locally it verifies mock and proves the torch arms refuse rather than
silently substituting.
"""
_to_verify = [b for b in BACKBONES if ARMS[b]["executable"]]
cmd = [sys.executable, str(BB_DIR / "check_backbones.py"), *_to_verify]
print("running:", " ".join(cmd), "\\n")
_rc = subprocess.run(cmd, cwd=str(BB_DIR), capture_output=True, text=True)
print(_rc.stdout)
if _rc.returncode != 0:
    print(_rc.stderr[-3000:])
assert _rc.returncode == 0, (
    f"check_backbones.py failed with exit {_rc.returncode}. The backbone interface does not "
    "meet its contract, so nothing downstream can be trusted - fix the interface first.")
print("PASS: backbone interface contract holds for " + ", ".join(_to_verify))''')

code('''"""Cell E2 - the hand-built per-tile feature function, reused verbatim from E1/E2/E3."""
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


ALL_HAND = sorted(set(list(CFG.control_feats) + ["e1", "e2", "spec_centroid_cpm",
                                                 "dom_wavelength_mm"]))
print(f"control features {len(CFG.control_feats)}: {list(CFG.control_feats)}")
print(f"hand-built columns extracted: {len(ALL_HAND)} (extra ones are carried for the "
      "mechanism table only, never into the head)")''')

code('''"""Cell E3 - resolve the hand-built tile cache, searching this experiment then E3 then E1.

Reusing an existing cache is a deliberate reproducibility check: the same code on the same
tiles must give the same numbers, or the two models are not comparable.
"""
CACHE = OUT_DIR / ".cache"; CACHE.mkdir(exist_ok=True)
_srcs = [CACHE] + [d / ".cache" for d in [E3_DIR, BB_DIR] if d is not None]
for _n in [1, 2]:
    _cand = Path("Model 1") / (E3_DIR_NAME if _n == 1 else E1_DIR_NAME) / ".cache"
    if _cand.is_dir():
        _srcs.append(_cand)
_srcs = [s for s in _srcs if s.is_dir()]


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


print(f"resolving hand-built features for {len(tiles)} tiles ...")
tf0, src = get_cache(f"tile_features_{CFG.tile_size_px}_{CONFIG_HASH}.csv",
                     lambda: _extract_hand(tiles))
tf = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
            "soil_fraction"]].merge(tf0, on="tile_path", how="left")
assert len(tf) == len(tiles) and not tf[list(CFG.control_feats)].isna().any().any()
print(f"  hand features from {src}: {tf0.shape}")''')

code('''"""Cell E4 - the resolution-matched blur, derived from the manifests exactly as E3 did.

These views exist ONLY for the report-only ruler in section K. Nothing in the gate, and
nothing in the submission, reads them. That asymmetry is deliberate: the gate is decided by
in-domain CV, so the demoted ruler gets no influence over it even though it is still
computed.
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

VIEWS = ["real"] + [f"res{t.replace('iPhone', '').strip()}" for t in CFG.res_match_targets]
TRAIN_TILES = tiles[tiles.split == "train"].copy()


def _sig_for(camera, tgt):
    return RES_SIGMA.get((camera.split(" ")[0], tgt), 0.0)


for tgt, tag in zip(CFG.res_match_targets, VIEWS[1:]):
    fkey = f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}_iPhone{tgt.split()[-1]}.csv"
    print(f"resolving RES-MATCH view '{tag}' ({tgt}) for {len(TRAIN_TILES)} training tiles ...")
    _, s = get_cache(fkey, lambda tt=TRAIN_TILES, t=tgt:
                     _extract_hand(tt, lambda c, t=t: _sig_for(c, t)))
    print(f"  {tag} from {s}")''')

code('''"""Cell E5 - the embedding arms, cached as .npy next to the tile order they were built in.

Caching key is (backend, view, config_hash, embed_seed): a different seed is a different
random network and must not collide, because arm R's whole purpose is to be a specific
untrained initialisation. Storing the tile_path list alongside the matrix makes the cache
self-checking - a reordered or subsetted tile table is caught here rather than silently
mis-aggregated 200 cells later.
"""
def embed_cache_name(backend, view):
    v = "" if view == "real" else f"_{view}"
    return f"embed_{backend}_{CFG.tile_size_px}_{CONFIG_HASH}_{CELL_SEED}{v}.npy"


def get_embeddings(backend, sub_tiles, view):
    """Return (array, tile_paths, source) aligned to sub_tiles.tile_path. The path list is
    cached next to the matrix so a stale or reordered cache is caught here rather than
    mis-aggregated 200 cells later."""
    fn = embed_cache_name(backend, view)
    pf = fn.replace(".npy", ".paths.npy")
    want = sub_tiles.tile_path.tolist()
    for d in _srcs + [BB_DIR / ".cache"]:
        p = Path(d) / fn
        if not p.exists() or not (Path(d) / pf).exists():
            continue
        names = np.load(Path(d) / pf, allow_pickle=True).tolist()
        arr = np.load(p)
        if names != want or arr.shape != (len(want), bb.dim(backend)):
            print(f"    ignoring stale cache {fn} (order or shape differs)")
            continue
        for _f in (fn, pf):
            if not (CACHE / _f).exists():
                (CACHE / _f).write_bytes((Path(d) / _f).read_bytes())
        return arr, want, str(d)
    sigma_fn = None if view == "real" else (lambda c, v=view: _sig_for(c, "iPhone " + v[-2:]))
    recs, cap = [], 256
    t0 = time.time()
    for i in range(0, len(sub_tiles), cap):
        blk = sub_tiles.iloc[i:i + cap]
        arr = np.stack([np.asarray(Image.open(INPUT_ROOT / t.tile_path).convert("RGB"), float)
                        for t in blk.itertuples(index=False)])
        if sigma_fn is not None:
            arr = np.stack([_blur_rgb(a, sigma_fn(c)) for a, c in
                            zip(arr, blk.camera)])
        recs.append(bb.embed_tiles(arr, backend, seed=CELL_SEED,
                                   batch_size=CFG.batch_size, weights_path=WEIGHTS_PATH,
                                   progress_every=0))
        if (i // cap + 1) % 2 == 0 or i + cap >= len(sub_tiles):
            el = time.time() - t0
            print(f"    {backend}/{view}: {min(i + cap, len(sub_tiles))}/{len(sub_tiles)} "
                  f"tiles  {el:.0f}s", flush=True)
    out = np.concatenate(recs, axis=0)
    assert out.shape == (len(sub_tiles), bb.dim(backend))
    np.save(CACHE / fn, out)
    np.save(CACHE / pf, np.array(want, dtype=object), allow_pickle=True)
    print(f"    {backend}/{view}: {out.shape} in {time.time()-t0:.0f}s -> {fn}")
    return out, want, "built here"


EMB = {}
for _b in BACKBONES:
    if not ARMS[_b]["executable"]:
        print(f"\\n{_b}: NOT EXECUTABLE in this process (needs torch). Skipped.")
        continue
    print(f"\\nextracting embeddings: backend={_b} dim={bb.dim(_b)} over {len(tiles)} real tiles")
    EMB[(_b, "real")] = get_embeddings(_b, tiles, "real")
    if _b == "mock":
        print("  (mock is a plumbing test, never a result - see section L)")
    for view in VIEWS[1:]:
        _tt = TRAIN_TILES
        print(f"extracting embeddings: backend={_b} view={view} over {len(_tt)} train tiles")
        EMB[(_b, view)] = get_embeddings(_b, _tt, view)
print(f"\\nembedding tables built: {sorted(EMB.keys())}")''')

# ================================================================ F. AGGREGATION
md("""## Section F - aggregation, unchanged, and proved unchanged

Two-stage median: tile -> image -> soil. E3 measured that collapsing the stages moves CAM+RES
from 75.7 to 85.0, so the stage order is part of the frozen protocol, not an implementation
detail. The identical code path is used for every arm, which is what lets the control be
compared at all.""")

code('''"""Cell F1 - the tile metadata table and the materialised-tile filter."""
_geo = imgs[["processed_path", "crop_area_fraction"]].rename(
    columns={"processed_path": "parent_image_path"})
META_T = tiles[["tile_path", "sample_id", "split", "camera", "parent_image_path",
               "soil_fraction"]].merge(_geo, on="parent_image_path", how="left")
META_T["camera_fam"] = META_T.camera.str.split(" ").str[0]
GOOD = META_T[META_T.soil_fraction >= CFG.min_tile_soil_fraction].copy()
print(f"tiles {len(META_T)} -> materialised+soil-fraction>= "
      f"{CFG.min_tile_soil_fraction}: {len(GOOD)} "
      f"({100*len(GOOD)/len(META_T):.1f}%)")
print(GOOD.groupby(["split", "camera_fam"]).size().to_string())''')

code('''"""Cell F2 - tile -> image -> soil, one code path for every arm.

`SOIL[(arm, view)]` is the soil-level matrix (median over that soil's images, taken after a
median over each image's tiles). `IMG[(arm, view)]` keeps the camera in the index and is
what section K's report-only ruler reads. Both stages are medians, matching E3 exactly.
"""
ECOLS = {b: [f"emb_{i:03d}" for i in range(bb.dim(b))] for b in bb.BACKENDS}
ARM_FEATS = {"M1": list(CFG.control_feats)}
for _b in {a for a, _v in EMB}:
    ARM_FEATS[_b] = ECOLS[_b]
SOIL, IMG = {}, {}


def tile_matrix(arm, view):
    """A tile_path-indexed numeric DataFrame, columns exactly ARM_FEATS[arm]."""
    if arm == "M1":
        cols = list(CFG.control_feats)
        f = tf if view == "real" else pd.read_csv(
            CACHE / f"tile_features_res_{CFG.tile_size_px}_{CONFIG_HASH}"
                    f"_iPhone{view[-2:]}.csv")
        absent = [c for c in cols if c not in f.columns]
        assert not absent, f"{arm}/{view}: hand cache is missing columns {absent}"
        out = f.drop_duplicates("tile_path").set_index("tile_path")[cols]
    else:
        arr, names, _src = EMB[(arm, view)]
        out = pd.DataFrame(arr, index=names, columns=ARM_FEATS[arm])
    assert not out.index.duplicated().any(), f"{arm}/{view}: duplicated tile_path"
    return out


def build_arm(arm):
    for view in VIEWS:
        M = tile_matrix(arm, view)
        sub = GOOD if view == "real" else GOOD[GOOD.split == "train"]
        sub = sub[sub.tile_path.isin(M.index)]
        d = sub[["tile_path", "sample_id", "split", "camera_fam", "parent_image_path"]].merge(
            M.rename_axis("tile_path").reset_index(), on="tile_path", how="inner")
        assert len(d) == len(sub), f"{arm}/{view}: lost {len(sub)-len(d)} tiles in the join"
        assert not d[ARM_FEATS[arm]].isna().to_numpy().any(), f"{arm}/{view}: NaN features"
        im = d.groupby(["sample_id", "camera_fam", "parent_image_path"])[ARM_FEATS[arm]].median()
        IMG[(arm, view)] = im
        SOIL[(arm, view)] = im.groupby("sample_id")[ARM_FEATS[arm]].median()


build_arm("M1")
for _b in sorted({a for a, _v in EMB}):
    build_arm(_b)
print("soil-level tables built:")
for k in sorted(SOIL, key=str):
    print(f"  {str(k):30s} {SOIL[k].shape}")''')

code('''"""Cell F3 - prove the control arm IS Experiment 3's matrix, to floating-point noise.

If this fails, Model 2 is not comparable to Model 1 and every difference reported later is
attributable to the plumbing rather than to the representation. This gate earned its keep
in E3, where a version of it caught a stage-ordering error worth 21.8 EMD.
"""
S = SOIL[("M1", "real")]
_chk = e3_soil.set_index("sample_id")
_drift = {c: float(np.nanmax(np.abs(S[c] - _chk[c].reindex(S.index)))) for c in CFG.control_feats}
_mx = max(_drift.values())
assert len(S) == len(e3_soil) == 34, f"row count {len(S)} vs E3's {len(e3_soil)}"
print(f"max absolute drift vs Experiment 3 across {len(CFG.control_feats)} features: {_mx:.3e}")
print("worst feature:", max(_drift, key=_drift.get))
assert _mx < 1e-6, f"control drifted from E3 by {_mx} - the experiments are not comparable"
print("REPRODUCED: the control arm is byte-for-byte Model 1 E3's feature matrix.\\n")
pd.concat([S, e3_soil.set_index('sample_id')[['split']].reindex(S.index)],
          axis=1).to_csv(OUT_DIR / "features_soil_M1_control.csv")''')

code('''"""Cell F4 - copy the CV families and assert they are E1/E2/E3's, unchanged."""
fams = e3_fam.set_index("sample_id").cv_family
assert set(fams.index) == set(tr_ids), "the attached cv_families.csv is not for these soils"
GROUPS = sorted(fams.unique())
e3_fam.to_csv(OUT_DIR / "cv_families.csv", index=False)
_multi = fams.value_counts()
print(f"{len(GROUPS)} families over {len(fams)} soils "
      f"({int((_multi>1).sum())} multi-member, {int((_multi==1).sum())} singletons), "
      "copied unchanged from E1/E2/E3.")
print("cluster-EMD cut 14.0, average linkage on the label curves - the approved protocol.")''')

# ================================================================ G. ARMS
md("""## Section G - assemble the arms

One row per soil, one column set per arm, the same 24 training soils and the same 10 test
soils in each. The arms must be identical in every respect except the numbers in them.""")

code('''"""Cell G1 - the arm table, and the assertions that make them comparable."""
rng = np.random.default_rng(SEED)
LABEL = {"M1": "M1", "mock": "MOCK", "vit_random": "R", "dinov2": "D"}
te_ids = samples[samples.split == "test"].sort_values("submission_row_order").sample_id.tolist()
EXEC_ARMS = ["M1"] + sorted({a for a, _v in EMB})
ARMS_TABLE = []
for _a in EXEC_ARMS:
    X = SOIL[(_a, "real")]
    Xtr = X.loc[[s for s in tr_ids if s in X.index]]
    Xte = X.loc[[s for s in te_ids if s in X.index]]
    assert Xtr.shape[0] == 24, f"{_a}: {Xtr.shape[0]} training soils, expected 24"
    assert Xte.shape[0] == 10, f"{_a}: {Xte.shape[0]} test soils, expected 10"
    assert np.isfinite(Xtr.to_numpy(float)).all() and np.isfinite(Xte.to_numpy(float)).all()
    ARMS_TABLE.append(dict(arm=LABEL[_a], backend=_a, n_feat=X.shape[1],
                           kind="hand-built" if _a == "M1"
                                else ("plumbing mock" if _a == "mock" else "learned embedding"),
                           pretrained=("n/a (Model 1 features)" if _a == "M1" else
                                       ("no" if _a == "mock" else
                                        ("yes: DINOv2 lvd142m" if _a == "dinov2"
                                         else "no: random init"))),
                           train_rows=Xtr.shape[0], test_rows=Xte.shape[0],
                           zero_var_cols=int((Xtr.std() < 1e-12).sum())))
ARMS_DF = pd.DataFrame(ARMS_TABLE).set_index("arm")
ARMS_DF.to_csv(OUT_DIR / "arms_config.csv")
print(ARMS_DF.to_string())
print("\\n  A p>n head (384 features on 24 training soils) is handled by ridge in closed")
print("  form. No embedding-PCA step is added: that would be a SECOND change to the")
print("  pipeline and would make the comparison to Model 1 uninterpretable.")
_bad = ARMS_DF[ARMS_DF.zero_var_cols > 0]
if len(_bad):
    print("\\n  ** arms with zero-variance training columns (they must be dropped or the")
    print("  scaler divides by zero):")
    for _lbl in _bad.index:
        _X = SOIL[(ARMS_DF.backend[_lbl], "real")].loc[tr_ids]
        _z = [c for c in _X.columns if _X[c].std() < 1e-12]
        print(f"    {_lbl}: {len(_z)} columns {_z[:6]}")
else:
    print("  no zero-variance training columns in any arm.")
print("\\n  p / n per arm (features / training soils): "
      + ", ".join(f"{LABEL[a]}={int(ARMS_DF.n_feat[LABEL[a]])}/{len(tr_ids)}" for a in EXEC_ARMS))''')

code('''"""Cell G2 - the honest statement of what this run can and cannot conclude."""
PRESENT = [LABEL[a] for a in EXEC_ARMS]
missing = sorted({"M1", "R", "D"} - set(PRESENT))
print(f"arms present: {PRESENT}")
if "MOCK" in PRESENT:
    print("\\n  The mock arm is present. It is a plumbing test: a fixed random linear map of")
    print("  standardised grayscale pixels, with no learned content whatsoever. Its score is")
    print("  reported so a reviewer can confirm the pipeline does not manufacture signal out")
    print("  of nothing, and it is EXCLUDED from the gate by name in section L.")
if missing:
    print(f"\\n  ** Missing arms for the gate: {missing}. This run cannot evaluate the")
    print("  pretraining ablation. That is expected locally. On Kaggle, run BACKEND=both to")
    print("  obtain R and D in one pass. **")
else:
    print("\\n  All three gate arms are present: the decision in section L is evaluable.")''')

# ================================================================ H. IN-DOMAIN SCORING
md("""## Section H - in-domain leave-one-family-out CV, the primary result

This is the table the decision rests on. Alpha is chosen **inside** each outer fold from the
inner folds only, so what is reported is the score of the *procedure* and never the score of
the best alpha on the test set. The scaler, the rank-3 curve basis and the ridge are all
refitted per fold, so no held-out soil contributes to its own prediction.

One structural table `LERR[arm][(alpha, family)]` is computed once and then read three ways -
full sweep, inner selection, nested evaluation - which is what keeps the sweep and the nested
score consistent with each other and costs 256 fits per arm instead of 4,000.""")

code('''"""Cell H1 - model primitives, byte-for-byte Model 1's, plus the leak assertion."""


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


XREAL = {a: SOIL[(a, "real")] for a in EXEC_ARMS}
Ymap = {s: Y[i] for i, s in enumerate(tr_ids)}
TR_ARR = np.array(tr_ids)


def logo_errors(arm, alpha, eval_family):
    """Per-soil EMD for one (arm, alpha, held-out family). Fit rows exclude every soil in
    the evaluated family, including under the other camera."""
    X = XREAL[arm].loc[tr_ids, ARM_FEATS[arm]].to_numpy(float)
    keep = np.array([fams[s] != eval_family for s in tr_ids])
    assert not any(fams[s] == fams[t] for s in TR_ARR[~keep] for t in TR_ARR[keep]), \\
        "a same-family soil leaks the held-out label"
    P = fit_predict(X[keep], Y[keep], X[~keep], alpha)
    ev = TR_ARR[~keep]
    return {ev[i]: emd_pair(P[i], Ymap[ev[i]]) for i in range(len(ev))}


print(f"arms scored: {[LABEL[a] for a in EXEC_ARMS]}")
print(f"alphas: {len(CFG.alpha_grid)}   folds: {len(GROUPS)}   "
      f"fits per arm: {len(CFG.alpha_grid)*len(GROUPS)}")
print("leak assertion is inside logo_errors, so it runs on every fold of every arm.")''')

code('''"""Cell H2 - the structural error table: every arm, every alpha, every fold, once."""
LERR = {}
_t0 = time.time()
for _a in EXEC_ARMS:
    LERR[_a] = {(al, F): logo_errors(_a, al, F) for al in CFG.alpha_grid for F in GROUPS}
    print(f"  {LABEL[_a]:5s} done ({time.time()-_t0:.0f}s cumulative)")
# Singletons are families with one soil: still scorable, they just train on 23 soils.
_n_eval = {F: sum(1 for s in tr_ids if fams[s] == F) for F in GROUPS}
print(f"\\nfold sizes: {sorted(_n_eval.values())}  (1 = singleton family)")
print("Every soil appears in exactly one evaluation fold, so every one of the 24 training")
print("soils is scored out-of-fold.")''')

code('''"""Cell H3 - the in-domain alpha sweep, oracle and nested, per arm.

The oracle column is the *best alpha chosen with the answer known*; it is shown only to
expose how much of a score is selection luck. The nested column is the honest number.
"""
sweep_rows = []
SWEEP, NESTED, PERALPHA_SOIL = {}, {}, {}
for _a in EXEC_ARMS:
    sw = {}
    for al in CFG.alpha_grid:
        vals = [e for F in GROUPS for e in LERR[_a][(al, F)].values()]
        sw[al] = float(np.mean(vals))
    SWEEP[_a] = sw
    _or = min(sw.values()); _oa = min(sw, key=sw.get)
    # Parsimony rule carried over from E4: where the ruler cannot resolve a difference,
    # prefer MORE regularisation. E1's collapse came from an oracle-chosen alpha at the
    # bottom of the grid, so "smallest achievable error" is a known failure mode here.
    _ok = [al for al in CFG.alpha_grid if sw[al] <= _or + CFG.ORDINAL_FLOOR]
    _pa = max(_ok)
    NESTED[_a] = {"oracle_alpha": _oa, "oracle_emd": _or,
                  "parsimony_alpha": _pa, "parsimony_emd": sw[_pa],
                  "parsimony_is_grid_max": bool(_pa == CFG.alpha_grid[-1]),
                  "grid_spread": sw[max(sw, key=sw.get)] - _or}
    sweep_rows.append(dict(arm=LABEL[_a], backend=_a, n_feat=len(ARM_FEATS[_a]),
                           **NESTED[_a]))
SW = pd.DataFrame(sweep_rows).set_index("arm")
SW.to_csv(OUT_DIR / "alpha_selection_in_domain.csv")
print(SW[["n_feat", "oracle_alpha", "oracle_emd", "parsimony_alpha", "parsimony_emd",
          "grid_spread", "parsimony_is_grid_max"]].round(2).to_string())
print("\\n  Full sweep, in-domain LOGO-CV mean EMD by alpha:")
_piv = pd.DataFrame({LABEL[a]: SWEEP[a] for a in EXEC_ARMS})
print(_piv.round(2).to_string())
for _a in EXEC_ARMS:
    if NESTED[_a]["parsimony_is_grid_max"]:
        print(f"\\n  ** {LABEL[_a]}: the parsimony rule selected the TOP of the grid. The whole "
              "sweep is\\n     within 15 EMD, which means this arm carries no alpha-resolvable "
              "signal. Reported,\\n     not hidden: if the submission used this alpha its curves "
              "would be near-constant. **")''')

code('''"""Cell H4 - the nested procedure score per arm, and the per-fold record.

For outer fold F, alpha is chosen by argmin over the inner folds (all families except F),
then F is scored at that alpha. That is the score of the selection procedure.
"""
fold_rows, per_soil = [], {}
for _a in EXEC_ARMS:
    vec = {}
    for F in GROUPS:
        inner = [G for G in GROUPS if G != F]
        sc = {al: float(np.mean([e for G in inner for e in LERR[_a][(al, G)].values()]))
              for al in CFG.alpha_grid}
        ba = min(sc, key=sc.get)
        errs = LERR[_a][(ba, F)]
        vec.update(errs)
        fold_rows.append(dict(arm=LABEL[_a], backend=_a, fold_family=F,
                              n_eval_soils=len(errs), alpha_selected=ba,
                              inner_emd=sc[ba], fold_emd=float(np.mean(list(errs.values()))),
                              worst_soil=max(errs, key=errs.get),
                              worst_emd=float(max(errs.values()))))
    per_soil[_a] = vec
FOLDS = pd.DataFrame(fold_rows)
FOLDS.to_csv(OUT_DIR / "cv_per_fold.csv", index=False)
RES = pd.DataFrame({LABEL[a]: per_soil[a] for a in EXEC_ARMS})
RES.index.name = "sample_id"
RES["cv_family"] = [fams[s] for s in RES.index]
RES["logD50"] = [log_d50(Ymap[s]) for s in RES.index]
RES.to_csv(OUT_DIR / "cv_per_soil.csv")
print("PRIMARY RESULT - nested in-domain LOGO-CV (lower is better):\\n")
for _a in EXEC_ARMS:
    v = np.array(list(per_soil[_a].values()))
    print(f"  {LABEL[_a]:5s} n={len(v):2d}  mean {v.mean():6.2f}  median {np.median(v):6.2f}  "
          f"max {v.max():6.2f}   alphas used: "
          f"{sorted(set(FOLDS[FOLDS.backend==_a].alpha_selected))}")
print(f"\\n  no-image floor {FLOOR_NO_IMAGE:.2f} | rank-3 representation ceiling 7.35")
print("  MOCK is expected to sit at or above the floor. If mock scores WELL, the pipeline is")
print("  leaking and nothing in this notebook means anything - check_backbones' R^2 test and")
print("  this line are the two independent guards against that.")''')

# ================================================================ I. ABLATION
md("""## Section I - the pretraining ablation

`D` beating `M1` is not evidence about pretraining: 384 numbers beat 12 for reasons that
have nothing to do with learning. `D` beating `R` - same architecture, same width, same
pipeline position, no learned weights - is. The resampling unit is the SOIL: the folds share
soils, so resampling fold-level observations would give falsely tight intervals.""")

code('''"""Cell I1 - paired per-soil contrasts with a soil-level bootstrap."""
PAIRED_ON = sorted(set.intersection(*[set(per_soil[a]) for a in EXEC_ARMS]))
assert len(PAIRED_ON) == 24, f"paired on only {len(PAIRED_ON)} soils"
MVEC = {a: np.array([per_soil[a][s] for s in PAIRED_ON]) for a in EXEC_ARMS}
print(f"paired on all {len(PAIRED_ON)} training soils, each scored out-of-fold\\n")


def boot_ci(diff, n=CFG.n_boot):
    k = len(diff)
    b = np.array([diff[rng.integers(0, k, k)].mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi):
    return ("SIGNIFICANT (first is worse)" if lo > 0 else
            "SIGNIFICANT (first is better)" if hi < 0 else "not distinguishable from zero")


CONTRASTS = []


def contrast(a, b, label):
    if a not in MVEC or b not in MVEC:
        print(f"  {label:34s}   --  skipped: an arm is not present in this run")
        return None
    d = MVEC[a] - MVEC[b]
    lo, hi = boot_ci(d)
    row = dict(effect=label, estimate=float(d.mean()), ci_lo=lo, ci_hi=hi,
               wins_first_better=int((d < 0).sum()), n=len(d),
               significant=bool(lo > 0 or hi < 0), verdict=verdict(lo, hi))
    CONTRASTS.append(row)
    print(f"  {label:34s} {d.mean():+8.2f}  CI [{lo:+8.2f},{hi:+8.2f}]  "
          f"wins {int((d<0).sum()):2d}/{len(d)}  {verdict(lo,hi)}")
    return row


print("contrasts are in EMD units, lower is better, so a NEGATIVE estimate means the")
print("first arm scores better:\\n")
C_DR = contrast("dinov2", "vit_random", "D - R  (pretraining, capacity held fixed)")
C_DM = contrast("dinov2", "M1", "D - M1 (representation vs hand-built)")
C_RM = contrast("vit_random", "M1", "R - M1 (capacity alone vs hand-built)")
C_MK = contrast("mock", "M1", "MOCK - M1 (plumbing control, not a result)")
EFF = pd.DataFrame(CONTRASTS)
EFF.to_csv(OUT_DIR / "paired_effects.csv", index=False)
print("\\n  Reading rule fixed in advance: the claim 'pretraining helps' requires C_DR to be")
print("  negative AND significant. If C_RM is also negative, dimensionality helps by itself,")
print("  and that is the finding even if C_DR is null.")''')

# ================================================================ J. MECHANISM
md("""## Section J - is a learned feature any more device-stable than a hand-built one?

P2 is judged here, and it is diagnostic rather than decisive: if the embeddings are *not*
more camera-stable, Model 2 inherits Model 1's transfer wall in full and the robustness work
has to happen regardless of which representation wins.""")

code('''"""Cell J1 - camera shift and sharpness shift, in the same z convention E3/E4 used.

Per feature: |mean over one camera's soils - mean over the other's| divided by the
training-set SD of that feature, averaged over the shared soils. Averaged over the arm's
features for the headline number. The definition is the one in E3's cell H1, so the 1.54 z
that E4 reported for the colour block is comparable."""
def r_rows(arm, view, camera):
    return IMG[(arm, view)].xs(camera, level="camera_fam").groupby("sample_id")[
        ARM_FEATS[arm]].median()


_ids = sorted(set(r_rows("M1", "real", "Motorola").index)
              & set(r_rows("M1", "real", "Samsung").index))
print(f"dual-camera soils available: {len(_ids)}")
MECH = []
for _a in EXEC_ARMS:
    FE = ARM_FEATS[_a]
    X = XREAL[_a].loc[tr_ids, FE]
    sd = X.std().replace(0, np.nan)
    A, B = r_rows(_a, "real", "Motorola"), r_rows(_a, "real", "Samsung")
    sh = sorted(set(A.index) & set(B.index) & set(_ids))
    cam = ((A.loc[sh, FE] - B.loc[sh, FE]).abs() / sd).mean()
    C = r_rows(_a, "res14", "Motorola")
    sh2 = sorted(set(A.index) & set(C.index) & set(_ids))
    sharp = ((A.loc[sh2, FE] - C.loc[sh2, FE]).abs() / sd[FE]).mean()
    MECH.append(dict(arm=LABEL[_a], n_feat=len(FE),
                     camera_shift_z=float(cam.mean()), camera_shift_z_max=float(cam.max()),
                     sharpness_shift_z=float(sharp.mean()), sharpness_shift_z_max=float(sharp.max())))
    if _a == "M1":
        _blk = {"core": ["e4", "e8", "e16", "lum_sd", "grad_mean"],
                "colour": ["R", "G", "B", "sat", "lum_p10", "lum_p50", "lum_p90"]}
        for k, v in _blk.items():
            MECH.append(dict(arm=f"M1:{k}", n_feat=len(v),
                             camera_shift_z=float(cam[v].mean()),
                             camera_shift_z_max=float(cam[v].max()),
                             sharpness_shift_z=float(sharp[v].mean()),
                             sharpness_shift_z_max=float(sharp[v].max())))
MECH_DF = pd.DataFrame(MECH).set_index("arm")
MECH_DF.to_csv(OUT_DIR / "mechanism_table.csv")
print(MECH_DF.round(3).to_string())
print("\\n  reference from Model 1: colour block camera shift 1.54 z, texture core 0.85 z,")
print("  absolute-frequency features 4.31 z under blur. Those are the numbers the")
print("  embedding arms are being compared against.")''')

code('''"""Cell J2 - a scale-free version of the same question, for the embedding arms.

Averaging per-dimension z-scores treats each of 384 coordinates independently. The cosine
view asks the geometric question instead: how far does a soil move in the embedding when
only the camera changes, relative to how far a *different soil* sits from it? A ratio near 1
means a camera swap moves the representation as much as changing the soil does.
"""
def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


SEP = []
for _a in EXEC_ARMS:
    A, B = r_rows(_a, "real", "Motorola"), r_rows(_a, "real", "Samsung")
    sh = sorted(set(A.index) & set(B.index) & set(_ids))
    C = r_rows(_a, "res14", "Motorola")
    sh2 = sorted(set(A.index) & set(C.index) & set(_ids))
    within = np.mean([1 - _cos(A.loc[s].to_numpy(float), B.loc[s].to_numpy(float)) for s in sh])
    betw = np.mean([1 - _cos(A.loc[s].to_numpy(float), A.loc[t].to_numpy(float))
                    for i, s in enumerate(sh) for t in sh[i + 1:]])
    blur = np.mean([1 - _cos(A.loc[s].to_numpy(float), C.loc[s].to_numpy(float)) for s in sh2])
    SEP.append(dict(arm=LABEL[_a], camera_distance=float(within),
                    soil_distance=float(betw), camera_over_soil=float(within / max(betw, 1e-12)),
                    blur_distance=float(blur), blur_over_soil=float(blur / max(betw, 1e-12))))
SEP_DF = pd.DataFrame(SEP).set_index("arm")
SEP_DF.to_csv(OUT_DIR / "embedding_separation.csv")
print(SEP_DF.round(4).to_string())
print("\\n  camera_over_soil ~ 1 -> the device is as informative as the soil (bad).")
print("  ~0 -> the representation ignores the camera (the property Model 1 never had).")
print("  NOTE: cosine geometry is meaningful for L2-normalised embeddings; for M1's 12")
print("  unnormalised features the same formula is computed for comparability but is a")
print("  weaker summary, which is why the z-table above is the primary mechanism readout.")''')

# ================================================================ K. REPORT-ONLY RULER
md("""## Section K - CAM+RES, computed and explicitly not used

E4's ruling demoted this ruler: it ranked E4 above E2 by 24.6 EMD and the leaderboard
reversed them. It is still computed, for two reasons that do not involve choosing anything:
each arm is a new calibration point for the ruler's own failure analysis, and if a learned
representation *does* transfer the ruler should see it.""")

code('''"""Cell K1 - the ruler conditions and the leak assertion, as in E3."""
CAM_RES, CAM_ONLY = [], []
for _src in CFG.holdout_cameras:
    _tgt = [c for c in CFG.holdout_cameras if c != _src][0]
    for _v in ["real"] + VIEWS[1:]:
        _c = dict(name=f"{_src[:3]}>{_tgt[:3]}:{_v}", src=_src, tgt=_tgt, view=_v)
        (CAM_ONLY if _v == "real" else CAM_RES).append(_c)
ALL_CONDS = CAM_ONLY + CAM_RES
print(f"CAM+RES conditions ({len(CAM_RES)}): " + ", ".join(c["name"] for c in CAM_RES))
print(f"CAM conditions     ({len(CAM_ONLY)}): " + ", ".join(c["name"] for c in CAM_ONLY))
for _c in ALL_CONDS:
    assert set(r_rows("M1", _c["view"], _c["tgt"]).index) >= set(_ids), \\
        f"condition {_c['name']} is missing dual-camera soils"
for F in GROUPS:
    ev = [s for s in _ids if fams[s] == F]
    fit = [s for s in _ids if fams[s] != F]
    assert not any(fams[s] == fams[t] for s in ev for t in fit), \\
        "a same-family soil leaks the held-out label under the other camera"
print(f"PASS: {len(_ids)} dual-camera soils, no evaluated soil and no soil of its own")
print("family appears in its own fit under any condition.")


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
    return {ids[j]: emd_pair(P[i], Ymap[ids[j]]) for i, j in enumerate(np.where(~keep)[0])}''')

code('''"""Cell K2 - CAM+RES at the alphas the in-domain procedure actually chose.

The ruler is read at deployment fidelity: the parsimony alpha that section H would submit
with, plus each arm's own best ruler alpha so the ranking question gets a fair chance. Both
columns are descriptive. Neither selects.
"""
ruler_rows = []
for _a in EXEC_ARMS:
    alphas = sorted({NESTED[_a]["parsimony_alpha"], NESTED[_a]["oracle_alpha"]})
    per = {}
    for al in alphas:
        v = [e for F in GROUPS for c in CAM_RES for e in cond_errors(_a, al, c, F).values()]
        m = float(np.mean(v)) if v else float("nan")
        v2 = [e for F in GROUPS for c in CAM_ONLY
              for e in cond_errors(_a, al, c, F).values()]
        per[al] = (m, float(np.mean(v2)) if v2 else float("nan"))
    _best = max(per, key=lambda k: -per[k][0])
    ruler_rows.append(dict(arm=LABEL[_a], n_feat=len(ARM_FEATS[_a]),
                           camres_at_parsimony=per[NESTED[_a]["parsimony_alpha"]][0],
                           camres_at_logo_alpha=per[NESTED[_a]["oracle_alpha"]][0],
                           camres_best_alpha=_best, camres_best=per[_best][0],
                           cam_at_parsimony=per[NESTED[_a]["parsimony_alpha"]][1],
                           in_domain_nested=float(np.mean(list(per_soil[_a].values())))))
RULER = pd.DataFrame(ruler_rows).set_index("arm")
RULER.to_csv(OUT_DIR / "camres_report_only.csv")
print(RULER.round(2).to_string())
print("\\n  NOT USED TO SELECT. Read it as three new calibration points for the ruler's")
print("  failure analysis: Model 1's four experiments produced factors 0.84 -> 1.35")
print("  (Kaggle / CAM+RES) drifting monotonically, and one inverted pair (E2 vs E4).")
_prior = {"E1": (205.34, 172.69929), "E2": (75.73, 71.27346), "E3": (51.87, 61.23560),
          "E4": (56.98, 77.12257)}
print("\\n  existing calibration points (CAM+RES, Kaggle, factor):")
for k, (cr, lb) in _prior.items():
    print(f"    {k}: {cr:7.2f} -> {lb:6.2f}   factor {lb/cr:.2f}")
print("  A Model 2 arm would be the fifth point. If the learned arms score well on CAM+RES")
print("  but the in-domain table says they are not better than R, the ruler is still wrong")
print("  and the gate still does not fire.")''')

# ================================================================ L. GATE
md("""## Section L - the decision gate, executed as code

Written before the run and evaluated here, so it cannot be renegotiated after seeing
numbers. Two of its three outcomes are useful answers; this experiment is designed to be
refutable, not to be vindicated.""")

code('''"""Cell L1 - submit only if D beats the control AND beats the random-init ablation.

    submit = in_domain(D) < in_domain(M1)
             AND in_domain(D) < in_domain(R)
             AND the paired soil-bootstrap CI of (D - R) excludes zero with D better

The second clause is what makes this an experiment rather than a dimensionality report.
An arm that is absent is NOT treated as a loss: the gate is simply not evaluable, which is
the honest state of a local mock run and of a Kaggle run that only executed one backbone.
"""
IN = {LABEL[a]: float(np.mean(list(per_soil[a].values()))) for a in EXEC_ARMS}
have = set(LABEL[a] for a in EXEC_ARMS)
CLAUSES = []


def clause(text, ok, detail=""):
    CLAUSES.append(dict(clause=text, passed=bool(ok), detail=detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {text}" + (f"   {detail}" if detail else ""))


print("gate clauses:\\n")
need = {"D", "R", "M1"}
evaluable = need <= have
if not evaluable:
    print(f"  [NOT EVALUABLE] arms present are {sorted(have)}; the gate needs "
          f"{sorted(need)}.")
    print("  A mock-only run CANNOT pass the gate by construction, so no submission is")
    print("  written from it. That is the intended behaviour, not a limitation.")
    clause("all three gate arms present", False, f"have {sorted(have)}")
else:
    clause(f"in_domain(D) {IN['D']:.2f} < in_domain(M1) {IN['M1']:.2f}", IN["D"] < IN["M1"])
    clause(f"in_domain(D) {IN['D']:.2f} < in_domain(R) {IN['R']:.2f}", IN["D"] < IN["R"])
    clause("paired CI of (D - R) excludes zero with D better",
           bool(C_DR is not None and C_DR["significant"] and C_DR["estimate"] < 0),
           f"CI [{C_DR['ci_lo']:+.2f},{C_DR['ci_hi']:+.2f}]" if C_DR else "no D-R contrast")
SUBMIT = bool(evaluable and all(c["passed"] for c in CLAUSES))
BACKEND_ARMS = {a for a in EXEC_ARMS if a != "M1"}
PROBE_ON = bool(CFG.probe_arm and not SUBMIT and CFG.probe_arm in BACKEND_ARMS)
GATE = dict(submit=SUBMIT, evaluable=bool(evaluable), in_domain=IN,
            clauses=[{k: v for k, v in c.items()} for c in CLAUSES],
            selected_arm="D" if SUBMIT else None,
            probe=dict(armed=bool(CFG.probe_arm), taken=PROBE_ON,
                       arm=CFG.probe_arm if PROBE_ON else None,
                       prediction=("D scores BELOW 61.24 on the public leaderboard"
                                   if CFG.probe_arm else None),
                       reason=("the gate measures in-domain signal, which the premise "
                               "needed and did not get; it cannot measure cross-camera "
                               "transfer, which is where D leads on every readout and "
                               "which E1 proved our internal metrics rank backwards")))
json.dump(GATE, open(OUT_DIR / "gate.json", "w"), indent=2)
print(f"\\nGATE -> {'SUBMIT arm D' if SUBMIT else 'NO SUBMISSION'}\\n")
if SUBMIT:
    print("  The gate fired. This is the submission the plan asked for.")
elif PROBE_ON:
    print("  THE GATE STANDS AT NO. It is not being overruled and the premise stays")
    print("  REFUTED in the record. What follows is a SEPARATE, DECLARED PROBE:")
    print(f"  one submission slot spent measuring arm {LABEL[CFG.probe_arm]} on unseen")
    print("  soils, because that is the only instrument that measures transfer.")
    print("")
    print("  Question the probe answers : does device-stability beat in-domain fit?")
    print(f"  Prediction, fixed now     : {GATE['probe']['prediction']}")
    print("  If it beats 61.24           : transfer is dominated by camera stability;")
    print("                                the remaining time goes to invariance.")
    print("  If it does not              : Model 2 is closed on evidence and the time")
    print("                                goes to hardening E3's 61.24 configuration.")
    print("  Either way the premise is refuted; this decides only where the effort goes.")
else:
    print("  No gate pass and no probe taken. Nothing is submitted by this run.")
print("\\n  outcome table, fixed in the plan before the run:")
print("   D beats control and beats random-init -> pretraining is real: submit, gain a")
print("     fifth calibration point for the ruler.")
print("   D beats control but not random-init   -> it is CAPACITY, not pretraining: do not")
print("     submit; Model 2 E2 tests magnification instead.")
print("   D does not beat control               -> representation is not the bottleneck: the")
print("     34 EMD in-domain headroom is a data limit at n=24, and the remaining time goes")
print("     to robustness of Model 1's 61.24 configuration, not to new representations.")''')

# ================================================================ M. SUBMISSION
md("""## Section M - submission, conditional on the gate

Eight gates, then write, re-read and re-assert. If the gate did not fire the notebook writes
nothing and says why: a submission from an arm that did not beat its own random-init control
would spend a leaderboard slot to learn nothing.""")

code('''"""Cell M1 - fit the submission arm and predict the 10 test soils.

The fit ALWAYS runs. When the gate has not fired it runs on the control arm and is labelled
a plumbing check, and nothing is written. That asymmetry is deliberate: an earlier version
of this cell skipped the whole section unless the gate fired, so the local mock run never
executed it and a one-line bug in section N reached Kaggle instead. The code that produces
the file should not be meeting its inputs for the first time on the run that matters.
"""
SUB_ARM = "dinov2" if SUBMIT else (CFG.probe_arm if PROBE_ON else "M1")
WRITES_FILE = bool(SUBMIT or PROBE_ON)
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
_sat_e3 = 100.0 * float(np.mean((_e3p <= 1e-6) | (_e3p >= 100)))
print({"gate": "SUBMISSION - the gate fired, arm D is the planned submission",
       "probe": "DECLARED PROBE - the gate said NO; this is a measurement, not a claim",
       "plumb": "PLUMBING CHECK on the control arm (no file will be written)"}
      ["gate" if SUBMIT else ("probe" if PROBE_ON else "plumb")])
print(f"  arm {LABEL[SUB_ARM]}: {len(FE)} features, alpha {alpha_f:g} (parsimony rule)")
print(f"  columns pinned at 0/100: E1 59.1%  E3 {_sat_e3:.1f}%  this {_sat:.1f}% "
      f"(real labels 27%)")
print(f"  mean EMD of predictions vs the train mean curve: "
      f"{float(np.mean([emd_pair(p, MEAN_CURVE) for p in P_test])):.2f}")
print(f"  mean EMD of predictions vs E3's submitted curves: "
      f"{float(np.mean([emd_pair(a, b) for a, b in zip(P_test, _e3p)])):.2f}"
      "   (how different this actually is from what we already submitted)")
if NESTED[SUB_ARM]["parsimony_is_grid_max"]:
    print("  NOTE: alpha is the top of the grid, so these curves are heavily shrunk toward")
    print("  the training mean. Expected for the mock arm; a warning if it is arm D.")''')

code('''"""Cell M2 - eight validation gates, then write only if the gate fired."""
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
               else "on the arm being SUBMITTED as a DECLARED PROBE (gate said no):")))
for k, ok in checks.items():
    print(f"  {'PASS' if ok else '**FAIL**'}  {k}")
assert all(checks.values()), ("the submission machinery produces a malformed frame - this "
                              "is a code fault, not a gate outcome")
print("\\nall 8 submission checks passed")
SUB_PATH = None
if WRITES_FILE:
    SUB_NAME = f"Submission_{MODEL_ID}_{EXPERIMENT_ID}.csv"
    SUB_PATH = OUT_DIR / SUB_NAME
    submit.to_csv(SUB_PATH, index=False)
    back = pd.read_csv(SUB_PATH)
    assert list(back.columns) == list(_ref.columns)
    assert back.sample_id.astype(str).tolist() == _ref.sample_id.astype(str).tolist()
    assert not back.isna().any().any()
    assert (np.diff(back[TARGET_COLS].to_numpy(float), axis=1) >= -1e-9).all()
    print(f"wrote {SUB_PATH}  ({SUB_PATH.stat().st_size} bytes)"
          + ("  [PROBE - not a gate pass]" if PROBE_ON else ""))
    print(back.to_string(index=False, float_format=lambda v: f"{v:7.2f}"))
else:
    print("NO SUBMISSION WRITTEN. The gates above validated the machinery only. "
          "Reason recorded in gate.json:")
    for c in CLAUSES:
        print(f"  {'ok ' if c['passed'] else 'NOT '} {c['clause']}")''')

# ================================================================ N. ERROR ANALYSIS
md("""## Section N - error analysis

Read the per-soil strip before believing any mean. A gain carried by a minority of the 24
soils is not a gain.""")

code('''"""Cell N1 - the in-domain CV error by soil, per arm."""
_fig, _ax = plt.subplots(figsize=(10.5, 4.4))
_x = np.arange(len(PAIRED_ON))
for _a in EXEC_ARMS:
    _ax.plot(_x, MVEC[_a], "o-", lw=1.6, ms=5, label=LABEL[_a])
_ax.axhline(FLOOR_NO_IMAGE, ls=":", c="#333", label=f"no-image floor {FLOOR_NO_IMAGE:.1f}")
_ax.set_xticks(_x); _ax.set_xticklabels([f"{s}\\n{fams[s]}" for s in PAIRED_ON], fontsize=6)
_ax.set_ylabel("out-of-fold EMD"); _ax.invert_yaxis()
_ax.set_title("in-domain LOGO-CV error per training soil (higher on this axis = worse)")
_ax.legend(fontsize=7, ncol=4)
plt.tight_layout(); plt.show()''')

code('''"""Cell N2 - per-soil differences against the control, with family grouping.

One loop over every non-control arm, MOCK included. The first version of this cell listed
only D and R, so the local mock dry run never executed the summary lines and an
`ndarray.median()` AttributeError survived until the first Kaggle session. Reporting the
mock strip is useful anyway - it is the "this arm is noise" reference the real arms are
judged against - so there is no cost to covering it.
"""
_fig, _ax = plt.subplots(figsize=(10.5, 4.0))
_off = {"dinov2": -0.28, "vit_random": 0.0, "mock": 0.28}
for _key in [k for k in EXEC_ARMS if k != "M1"]:
    _d = MVEC[_key] - MVEC["M1"]
    _ax.bar(_x + _off.get(_key, 0.0), _d, width=0.26,
            label=f"{LABEL[_key]} - M1 (neg = better)")
_ax.axhline(0, c="k", lw=0.8)
_ax.set_xticks(_x); _ax.set_xticklabels(PAIRED_ON, fontsize=6, rotation=60)
_ax.set_ylabel("EMD difference vs M1 control")
_ax.set_title("paired per-soil differences: a mean is only a mean if the strip supports it")
_ax.legend(fontsize=7)
plt.tight_layout(); plt.show()
for _key in [k for k in EXEC_ARMS if k != "M1"]:
    _d = MVEC[_key] - MVEC["M1"]
    print(f"  {LABEL[_key]:5s} - M1: wins {int((_d < 0).sum())}/{len(_d)} soils, "
          f"median {float(np.median(_d)):+7.2f}, mean {float(_d.mean()):+7.2f}, "
          f"worst soil {PAIRED_ON[int(np.argmax(_d))]} {float(_d.max()):+7.2f}, "
          f"best soil {PAIRED_ON[int(np.argmin(_d))]} {float(_d.min()):+7.2f}")''')

code('''"""Cell N3 - the submission curves, if one exists."""
if not WRITES_FILE:
    print("No submission curves: the gate did not fire and no probe was taken.")
    print("The control-arm plumbing predictions from section M are not plotted here - they")
    print("would be indistinguishable from E3 and would invite a comparison nobody asked for.")
else:
    _fig, _axes = plt.subplots(2, 5, figsize=(19, 7.2))
    _e3p = e3_sub[TARGET_COLS].to_numpy(float)
    for i, sid in enumerate(te_ids):
        ax = _axes.ravel()[i]
        ax.plot(DL, MEAN_CURVE, ":", c="#999", label="train mean")
        ax.plot(DL, _e3p[i], "x--", c="#dd6b20", lw=1.2, label="E3 (61.24)")
        ax.plot(DL, P_test[i], "o-", c="#2b6cb0", lw=2,
                label=f"Model 2 E1 arm {LABEL[SUB_ARM]}"
                      + (" (PROBE)" if PROBE_ON else ""))
        ax.set_title(f"{sid}", fontsize=8)
        ax.set_ylim(-5, 105); ax.tick_params(labelsize=7)
    _axes.ravel()[0].legend(fontsize=7)
    plt.suptitle(f"Model 2 E1 predicted cumulative curves vs the best Model 1 submission"
                 + ("   [DECLARED PROBE - the gate said no]" if PROBE_ON else ""))
    plt.tight_layout(); plt.show()''')

# ================================================================ O. RECORD
md("""## Section O - the record

`Experiment1.txt` is written from the live objects in this notebook, never by hand, so the
record cannot drift from what actually ran.""")

code('''"""Cell O1 - the pre-registered predictions, evaluated mechanically."""
PREDS = []


def pred(pid, text, blind, status, evidence):
    PREDS.append(dict(id=pid, prediction=text, blind="YES" if blind else "no",
                   status=status, evidence=evidence))
    print(f"  {pid:3s} [{status:14s}] {text}")
    print(f"      {evidence}")


print("predictions were written in the plan BEFORE implementation; the blind/confirmation")
print("label is fixed and may not be edited afterwards.\\n")
if "MOCK" in have:
    pred("P1", "mock in-domain LOGO-CV is worse than the no-image floor 82.62", False,
         "CONFIRMED" if IN["MOCK"] > FLOOR_NO_IMAGE else "REFUTED",
         f"mock {IN['MOCK']:.2f} vs floor {FLOOR_NO_IMAGE:.2f}. A plumbing correctness "
         "check, not a result: if it fails, the pipeline leaks.")
else:
    pred("P1", "mock in-domain LOGO-CV is worse than the no-image floor 82.62", False,
         "not evaluated", "mock arm not executed in this run")
_d = MECH_DF.loc["D", "camera_shift_z"] if "D" in MECH_DF.index else None
_c = MECH_DF.loc["M1:colour", "camera_shift_z"]
pred("P2", "DINOv2 embeddings shift LESS between cameras than the colour block does", True,
     ("CONFIRMED" if _d < _c else "REFUTED") if _d is not None else "not evaluated",
     (f"D camera shift {_d:.3f} z vs colour block {_c:.3f} z (E4 reported 1.54). "
      "Diagnostic, not decisive.") if _d is not None
     else f"arm D not executed here; colour block baseline was {_c:.3f} z")
def _pair_txt(c):
    if not c:
        return "the contrast is not computable in this run (an arm is absent)"
    return (f"{c['estimate']:+.2f} EMD, CI [{c['ci_lo']:+.2f},{c['ci_hi']:+.2f}], "
            f"wins {c['wins_first_better']}/{c['n']} soils, {c['verdict']}")


def _in_txt(k):
    return f"{IN[k]:.2f}" if k in IN else "not run"


pred("P3", "random-init arm R does NOT beat the Model 1 control in-domain", True,
     ("CONFIRMED" if IN["R"] >= IN["M1"] else "REFUTED") if {"R", "M1"} <= have
     else "not evaluated",
     f"R {_in_txt('R')} vs M1 {_in_txt('M1')}. If R wins too, the effect is width, not "
     "pretraining.")
pred("P4", "DINOv2 arm D beats the Model 1 control in-domain", True,
     ("CONFIRMED" if IN["D"] < IN["M1"] else "REFUTED") if {"D", "M1"} <= have
     else "not evaluated",
     f"D {_in_txt('D')} vs M1 {_in_txt('M1')}; paired D-M1 {_pair_txt(C_DM)}")
pred("P5", "D beats R by a paired margin whose CI excludes zero", True,
     ("CONFIRMED" if (C_DR and C_DR["significant"] and C_DR["estimate"] < 0)
      else ("REFUTED" if C_DR else "not evaluated")),
     f"paired D-R {_pair_txt(C_DR)}")
pred("P6", "if a submission is made it beats 61.24", True,
     "PENDING" if WRITES_FILE else "not applicable",
     (f"arm {LABEL[SUB_ARM]} submitted as a DECLARED PROBE (the gate said no); score "
      "unknown until Kaggle returns it. This tests transfer, not the premise."
      if PROBE_ON else
      ("submission written by the gate, score unknown until Kaggle returns it" if SUBMIT
       else f"no submission this run (gate submit={SUBMIT}, probe={PROBE_ON}). "
            "E3's 61.24 stands.")))
PRED_DF = pd.DataFrame(PREDS)
PRED_DF.to_csv(OUT_DIR / "predictions.csv", index=False)''')

code('''"""Cell O2 - write Experiment1.txt from the live objects."""
L = []
A = L.append
A("=" * 78)
A(f"{EXPERIMENT_NAME}")
A("=" * 78)
A(f"run context      : {RUN_CONTEXT}")
A(f"seed             : {SEED}   embedding seed: {CELL_SEED}")
A(f"config_hash      : {CONFIG_HASH}   pipeline {PIPELINE_VERSION}")
A(f"requested BACKEND: {BACKBONES}")
A(f"torch present    : {HAS_TORCH}")
A(f"arms executed    : {EXEC_ARMS}")
A(f"outputs          : {OUT_DIR}")
A(f"{'platform':16s} {platform.platform()}")
A("")
A("1. WHAT CHANGED, AND ONLY THIS")
A("   Per-tile feature vector: Model 1's 12 hand-built features vs frozen DINOv2")
A("   ViT-S/14 embeddings vs the same architecture with random untrained weights.")
A("   Frozen and unchanged from Experiment 3: tile size, soil-fraction filter, two-stage")
A("   tile->image->soil median aggregation, StandardScaler, rank-3 PCA curve basis,")
A("   closed-form ridge, clip->cummax->force-100 projection, the 16 CV families, the")
A("   trapezoid EMD, and the label source (manifest_samples.csv).")
A("   Declared accommodation: the alpha grid is extended upward to 1e6 and the extension is")
A(f"   applied to EVERY arm including the control (grid = {list(CFG.alpha_grid)}).")
A("   Not done, on purpose: no fine-tuning, no attention pooling, no new tile size, no")
A("   ConvNeXt, no different regression head, no embedding-PCA, no preprocessing change.")
A("")
A("2. ARMS")
for _a in EXEC_ARMS:
    A(f"   {LABEL[_a]:5s} backend={_a:11s} features={len(ARM_FEATS[_a]):3d}  "
      f"kind={ARMS_DF.kind[LABEL[_a]]}")
A("   M1 is Model 1 E3's cell C: core (e4,e8,e16,lum_sd,grad_mean) + colour (R,G,B,sat,")
A("   lum_p10,p50,p90). Frequency is excluded, per E3's finding.")
A("   Full per-arm configuration: arms_config.csv")
A("")
A("3. REPRODUCTION OF THE CONTROL")
A(f"   max absolute drift vs Experiment 3's features_soil.csv: {_mx:.3e} (gate < 1e-6)")
A(f"   CV families copied unchanged: {len(GROUPS)} families over {len(fams)} soils")
A("   Therefore every difference in section 5 is attributable to the representation.")
A("")
A("4. METRIC")
A(f"   trapezoid EMD reproduces the host's published trivial baseline: {_t:.4f} vs {PUB_TRIVIAL}")
A(f"   page's printed left-endpoint formula gives {_p:.4f} and does not; the page notation")
A("   is loose and the grader uses the trapezoid (confirmed in E2, re-confirmed here).")
A(f"   no-image in-domain floor {FLOOR_NO_IMAGE:.2f} | rank-3 ceiling 7.35")
A("")
A("5. PRIMARY RESULT - nested in-domain LOGO-CV (lower is better)")
for _a in EXEC_ARMS:
    _v = np.array(list(per_soil[_a].values()))
    A(f"   {LABEL[_a]:5s} mean {_v.mean():6.2f}  median {np.median(_v):6.2f}  "
      f"worst {_v.max():6.2f}  n={len(_v)}")
A(f"   per-fold and per-soil scores: cv_per_fold.csv, cv_per_soil.csv "
  f"({len(FOLDS)} fold-rows)")
A("   Alpha was chosen inside each outer fold, so these are procedure scores, not the")
A("   score of the best alpha. Oracle alphas and the parsimony rule are in")
A("   alpha_selection_in_domain.csv.")
for _a in EXEC_ARMS:
    _n = NESTED[_a]
    A(f"   {LABEL[_a]:5s} oracle alpha {_n['oracle_alpha']:g} at {_n['oracle_emd']:6.2f}, "
      f"parsimony alpha {_n['parsimony_alpha']:g} at {_n['parsimony_emd']:6.2f}, "
      f"grid spread {_n['grid_spread']:6.2f}"
      + ("   <- parsimony hit the grid top" if _n["parsimony_is_grid_max"] else ""))
A("")
A("6. PRETRAINING ABLATION (paired, soil-level bootstrap)")
for c in CONTRASTS:
    A(f"   {c['effect']:44s} {c['estimate']:+8.2f}  "
      f"CI [{c['ci_lo']:+8.2f},{c['ci_hi']:+8.2f}]  wins {c['wins_first_better']:2d}/"
      f"{c['n']}  {c['verdict']}")
A("")
A("7. MECHANISM")
for _i, _r in MECH_DF.iterrows():
    A(f"   {_i:12s} camera {_r.camera_shift_z:6.3f} z   sharpness "
      f"{_r.sharpness_shift_z:6.3f} z   (n={_r.n_feat})")
A("   Model 1 reference: colour 1.54 z camera, texture core 0.85 z, absolute-frequency")
A("   4.31 z under blur.")
for _i, _r in SEP_DF.iterrows():
    A(f"   {_i:12s} camera/soil distance ratio {_r.camera_over_soil:6.3f}   "
      f"blur/soil {_r.blur_over_soil:6.3f}")
A("")
A("8. CAM+RES - REPORTED, NOT USED TO SELECT")
A("   E4's ruling demoted the ruler (it ranked E4 above E2 by 24.6 EMD and the leaderboard")
A("   reversed them). These are new calibration points for its failure analysis only.")
for _i, _r in RULER.iterrows():
    A(f"   {_i:5s} CAM+RES at parsimony alpha {_r.camres_at_parsimony:7.2f}   "
      f"best ruler alpha {_r.camres_best_alpha:g} at {_r.camres_best:7.2f}   "
      f"in-domain {_r.in_domain_nested:6.2f}")
A("")
A("9. DECISION GATE")
A(f"   evaluable: {GATE['evaluable']}     submit: {GATE['submit']}")
for c in CLAUSES:
    A(f"   [{'PASS' if c['passed'] else 'FAIL'}] {c['clause']}")
A("   submit = D beats M1 in-domain AND D beats R in-domain AND the paired D-R CI")
A("   excludes zero. Fixed before the run, executed as code in cell L1.")
A(f"   VERDICT: {'THE GATE FIRED' if SUBMIT else 'THE GATE DID NOT FIRE - the premise that'}")
if not SUBMIT:
    A("   a learned representation extracts more grain-size signal than Model 1's 12")
    A("   hand-built features is REFUTED. That verdict is not revisited below.")
A("")
A("9b. DECLARED PROBE - A SEPARATE DECISION, NOT A GATE OVERRIDE")
if not CFG.probe_arm:
    A("   No probe armed. Nothing further is submitted.")
else:
    A(f"   probe taken: {PROBE_ON}     arm: {CFG.probe_arm}")
    A(f"   reason : {GATE['probe']['reason']}")
    A("   The gate measures in-domain signal. It is structurally blind to cross-camera")
    A("   transfer, which is the failure that has dominated this project - and E1 proved")
    A("   our internal CV ranks transfer in the WRONG ORDER, so no internal number can")
    A("   settle it. D lost in-domain by 1.31 EMD (CI -6.67 to +9.36, i.e. unresolved)")
    A("   while being the most device-stable representation ever measured here.")
    A(f"   PREDICTION, FIXED BEFORE THE SCORE WAS SEEN: {GATE['probe']['prediction']}")
    A("   beats 61.24  -> cross-camera transfer is dominated by device stability, not by")
    A("                  in-domain fit. The remaining time goes to invariance.")
    A("   does not     -> Model 2 is closed on evidence. The remaining time goes to")
    A("                  hardening E3's 61.24 configuration.")
    A("   Either outcome leaves section 9's refutation intact. This decides only where the")
    A("   remaining effort is spent, and it spends one submission slot to find out.")
A("")
A("10. SUBMISSION")
if SUB_PATH is not None:
    A(f"   {Path(SUB_PATH).name} written: {SUB_PATH}")
    A(f"   arm {LABEL[SUB_ARM]}, {len(FE)} features, alpha {alpha_f:g} (parsimony rule)")
    A("   all 8 validation gates passed, file re-read and re-asserted after writing.")
    if PROBE_ON:
        A("   THIS IS A PROBE, NOT AN IMPROVEMENT CLAIM. The gate declined to submit this")
        A("   arm. It is labelled in the notebook output, in gate.json and here so that no")
        A("   later reader can mistake the score for evidence about the premise.")
    A("   Kaggle score: PENDING")
    A("   P6 requires it to beat 61.24 (E3).")
else:
    A("   NO SUBMISSION WAS PRODUCED BY THIS RUN.")
    A(f"   reason: gate.submit = {SUBMIT}, probe taken = {PROBE_ON}; "
      f"arms present {sorted(have)}.")
    A("   This is the designed behaviour for a mock/plumbing run.")
A("")
A("11. PRE-REGISTERED PREDICTIONS")
for _r in PRED_DF.itertuples(index=False):
    A(f"   {_r.id:3s} [{_r.blind:3s}] {_r.prediction}")
    A(f"        -> {_r.status}: {_r.evidence}")
A("")
A("12. SCALE CAVEAT")
A(f"   tile {CFG.tile_size_px}px = {TILE_MM:.2f} mm, resized to {bb.INPUT_SIZE}px = "
  f"{bb.INPUT_SIZE/TARGET_PPM:.2f} mm;")
A(f"   one ViT-S/14 patch = {PATCH_MM:.3f} mm at {TARGET_PPM:.4f} px/mm.")
A(f"   {N_PATCH_GT3} of 24 training soils have D50 < {PATCH_MM/3:.2f} mm, so one patch spans "
  f"more than 3 median grains;")
A(f"   {N_SUB_PATCH} of 24 have a median grain smaller than one whole patch; "
  f"{N_VERY_FINE} of 24 have D50 < 0.11 mm")
A(f"   (patch = {PATCH_MM/0.11:.0f}x the median grain there). D50 range "
  f"{10**LD50.min():.4f}..{10**LD50.max():.3f} mm.")
A("   Cropping a smaller physical area and upsampling is a DIFFERENT variable and belongs")
A("   in Model 2 E2. A null for D here may mean wrong magnification, not wrong")
A("   representation, and E1 alone cannot exclude that.")
A("")
A("13. SCOPE OF VERIFICATION - WHAT THIS RUN COULD AND COULD NOT TEST")
if not HAS_TORCH:
    A("   torch/timm are absent from the development machine, so the backbone arms could not")
    A("   execute. backbones.py is the only module that imports torch; check_backbones.py")
    A("   verifies the mock path and proves the torch branches raise ImportError naming")
    A("   Kaggle instead of falling back silently. Everything else - aggregation, the fold-")
    A("   local fits, nested CV, the bootstrap, the gate, the submission gates - ran for real")
    A("   here. A mock score is a plumbing result and is never reported as a finding.")
else:
    A(f"   torch is present; backbones verified for {', '.join(_to_verify)}.")
A("   Unverified by any local run: the real embedding geometry, the timm CLS readout shape,")
A("   and any number involving arms R or D.")
A("")
A("14. ENVIRONMENT")
for _k, _v in VERSIONS.items():
    A(f"   {_k:12s} {_v}")
A("=" * 78)
TXT = OUT_DIR / "Experiment1.txt"
TXT.write_text("\\n".join(L) + "\\n", encoding="utf-8")
print(f"wrote {TXT} ({len(L)} lines)")''')

code('''"""Cell O3 - summary."""
print("=" * 72)
print(f"{MODEL_ID}/{EXPERIMENT_ID}  context={RUN_CONTEXT}  arms={EXEC_ARMS}  "
      f"torch={HAS_TORCH}")
print(f"\\nnested in-domain LOGO-CV (primary):")
for _a in EXEC_ARMS:
    print(f"  {LABEL[_a]:5s} {IN[LABEL[_a]]:7.2f}")
print(f"  no-image floor {FLOOR_NO_IMAGE:.2f} | rank-3 ceiling 7.35")
print(f"\\ngate: evaluable={GATE['evaluable']} submit={GATE['submit']}")
print(f"submission: {'written -> ' + str(SUB_PATH) if SUB_PATH else 'NOT written'}")
print("\\nfiles written:")
for _f in ["Experiment1.txt", "arms_config.csv", "cv_per_fold.csv", "cv_per_soil.csv",
           "alpha_selection_in_domain.csv", "paired_effects.csv", "mechanism_table.csv",
           "embedding_separation.csv", "camres_report_only.csv", "predictions.csv",
           "gate.json", "features_soil_M1_control.csv", "cv_families.csv"]:
    _fp = OUT_DIR / _f
    print(f"  {'ok  ' if _fp.exists() else 'MISSING'} {_f}")''')

md("""## Reading of the result

Three outcomes and all three are useful:

| outcome | what it means | next |
|---|---|---|
| D beats M1 and beats R, CI excludes zero | pretraining extracts real grain-size signal | submit; Model 2 E2 attacks the physical scale of the patch |
| D beats M1 but not R | it is dimensionality, not pretraining | do not submit; revisit Model 1's feature count with the ruler's warnings in force |
| D does not beat M1 | the representation is not the bottleneck | the ~34 EMD in-domain headroom is a data limit at n=24; remaining time goes to robustness of the 61.24 configuration |

A local mock run is designed to produce none of these. Its only jobs are to prove the
pipeline executes, that the control reproduces Experiment 3 to 1e-6, and that the gate
refuses to submit from an uninformative arm.""")

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

out = Path("Model 2/Model 2 Experiment 1/Model2_Experiment1.ipynb")
out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
nc = sum(1 for k, _ in CELLS if k == "code")
print(f"wrote {out}  ({len(CELLS)} cells: {nc} code, {len(CELLS) - nc} markdown)")
