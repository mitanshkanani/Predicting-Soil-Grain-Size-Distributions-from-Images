"""Single source of every constant used by the preprocessing pipeline.

Nothing here is a guess: the tolerances and the canonical scale are justified in
the approved plan, and TARGET_PPM is *derived* at runtime by geometry.py rather
than typed here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# --- read-only inputs: never written to, enforced in safe_open() -------------
SOURCE_DIRS = {
    "train": os.path.join(DATA_DIR, "Training"),
    "test": os.path.join(DATA_DIR, "Test"),
}
LABELS_CSV = os.path.join(DATA_DIR, "Training_labels_updated.csv")
SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
PPM_CSV = os.path.join(DATA_DIR, "ppm_updated.csv")

# --- outputs -----------------------------------------------------------------
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_meta")
MASK_DIR = os.path.join(PROCESSED_DIR, "masks")
QC_DIR = os.path.join(DATA_DIR, "qc")
OUT_DIRS = {"train": os.path.join(DATA_DIR, "training_down"),
            "test": os.path.join(DATA_DIR, "testing_down")}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Canonical physical scale. Populated by geometry.run() as
# min(effective_ppm over all cameras); deliberately None here so that anything
# reading it before the geometry stage fails loudly instead of silently using a
# stale number.
_TARGET_PPM: float | None = None

TARGET_PPM_CANDIDATE = 4.58          # the value proposed in the brief, for comparison only

# Tolerances (rationale in plan STEP 3 / STEP 4).
TOL_AXIS_DISAGREEMENT = 0.005        # 0.5%; 8x above Samsung's measured 0.062% anisotropy,
                                     # 240x below the 122% error a naive width/width pairing gives.
TOL_ASPECT_MISMATCH = 0.005          # catches cropping, which the ppm check cannot see
TOL_FOV_MM = 0.5                     # per-axis FOV drift allowed through resampling
TOL_PPM_REL = 5e-5                   # canvas_px / fov_mm must match TARGET_PPM this closely
TOL_ACTUAL_PPM = 0.005               # audit: |actual_ppm_after - target| / target
# Crop policy. Measured tray fraction ranges 14.5% of a Motorola frame to 41.6% of
# an iPhone 14 frame, so a single small cap manufactures a train/test asymmetry:
# test tiles would keep roughly three times as much empty tray as training tiles.
# Capping at 30% lets the tray-heavy cameras be de-trayed while the 40% rejection
# guard still catches the mask failures that confidently want half the frame.
MAX_CROP_FRACTION = 0.30             # per axis: expand a too-small rect back to this
MAX_ACCEPT_REMOVAL = 0.40            # per axis: beyond this the mask is distrusted
MIN_CROP_SOIL_FRACTION = 0.85        # retained rect must be this soil-dense internally
# A row/col is kept when its smoothed soil density reaches this fraction of the
# image's OWN interior density. Relative, not absolute: measured frame soil density
# ranges 0.755 (Samsung) to 0.943 (iPhone 16) for the same physical tray, so any
# absolute threshold is a camera decision dressed up as a soil decision. An earlier
# absolute 0.98 was simply unreachable -- max per-row coverage is 0.945 on Motorola
# and 0.951 on Samsung, because soil always contains dark gaps between grains --
# which silently returned full frames for exactly the cameras that need cropping.
ROW_REL_COVERAGE = 0.85
COVERAGE_SMOOTH_FRAC = 0.05          # local-density window as a fraction of axis length
MIN_COMPONENT_FRACTION = 0.005       # drop mask blobs smaller than 0.5% of area
PROXY_SHORT_SIDE = 480               # mask computed on a proxy; rect needs mm-level, not px-level, accuracy

TILE_SIZES = (128, 224, 256, 384)
MATERIALIZED_TILE_SIZE = 256         # only this size is written as image files
TILE_STRIDE_FRAC = 0.0               # non-overlapping; kept in the manifest so overlap is a config edit

PNG_COMPRESS_LEVEL = 6
PIPELINE_VERSION = "1.0.0"

# Label columns exactly as spelled in Training_labels_updated.csv and
# sample_submission.csv -- note the bare "2", "20", "63", "200".
TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63",
                  "2", "6.3", "20", "63", "200"]
SUPPORT_MM = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0]


def target_ppm() -> float:
    if _TARGET_PPM is None:
        raise RuntimeError(
            "TARGET_PPM is derived by the geometry stage and is not available yet. "
            "Run `python -m preprocess --stage geometry` first."
        )
    return _TARGET_PPM


def set_target_ppm(value: float) -> None:
    global _TARGET_PPM
    _TARGET_PPM = float(value)


def load_target_ppm() -> float | None:
    path = os.path.join(PROCESSED_DIR, "target_ppm.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return float(json.load(fh)["target_ppm"])


def _stable(value):
    """Render a value into something whose str() is process-independent."""
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _stable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable(v) for v in value]
    return value


def config_hash() -> str:
    """Fingerprint of every knob that can change pipeline output.

    Sets must be sorted before serialising. Python randomises string hashing per
    process, so str({"​.jpg", ".jpeg", ".png"}) comes out in a different order in
    different runs -- four consecutive processes with an identical config produced
    three different hashes here before that was fixed. With a hash that moved,
    every stage looked stale on every invocation, so the invalidation machinery
    rebuilt the whole corpus each time and never actually tested anything.
    """
    payload = {k: _stable(v) for k, v in sorted(globals().items())
               if k.isupper() and isinstance(
                   v, (int, float, str, bool, tuple, list, dict, set, frozenset, type(None)))}
    payload.pop("REPO_ROOT", None)
    payload.pop("_TARGET_PPM", None)
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def stage_path(name: str) -> str:
    return os.path.join(PROCESSED_DIR, f"stage_{name}.csv")


def state_path() -> str:
    return os.path.join(PROCESSED_DIR, "run_state.json")


def load_state() -> dict:
    p = state_path()
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def record_stage_done(name: str) -> None:
    """Stamp a completed stage with the config that produced it.

    Without this, changing a constant such as MAX_CROP_FRACTION leaves the old
    resampled PNGs on disk while the manifests and audit are regenerated, so a
    normal (non-forced) run silently produces a corpus whose pixels and metadata
    describe different pipelines.
    """
    import datetime
    st = load_state()
    st[name] = {"config_hash": config_hash(),
                "pipeline_version": PIPELINE_VERSION,
                "finished_at": datetime.datetime.now(datetime.timezone.utc)
                               .isoformat(timespec="seconds")}
    with open(state_path(), "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)


def stage_is_stale(name: str, order: list[str]) -> bool:
    """True if this stage, or anything it depends on, was built under other config."""
    st = load_state()
    h = config_hash()
    if name not in order:
        return True
    for prev in order[:order.index(name) + 1]:
        rec = st.get(prev)
        if (not rec or rec.get("config_hash") != h
                or rec.get("pipeline_version") != PIPELINE_VERSION):
            return True
    return False


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, MASK_DIR, QC_DIR, os.path.join(QC_DIR, "sheets")):
        os.makedirs(d, exist_ok=True)
    for d in OUT_DIRS.values():
        os.makedirs(d, exist_ok=True)


def assert_writable(path: str) -> str:
    """Refuse any write that resolves inside a read-only source directory.

    Belt and braces on the plan's 'untouched, enforced not assumed' rule: a stray
    shutil.move or open(..., 'w') in a later refactor is caught here rather than
    discovered by the audit after the data is already gone.
    """
    ap = os.path.abspath(path)
    for name, src in SOURCE_DIRS.items():
        src_abs = os.path.abspath(src)
        if ap == src_abs or ap.startswith(src_abs + os.sep):
            raise RuntimeError(
                f"refusing to write inside read-only source dir {name}: {path}"
            )
    if os.path.abspath(path).startswith(os.path.abspath(PPM_CSV)):
        raise RuntimeError(f"refusing to overwrite {path}")
    for csv in (LABELS_CSV, SUBMISSION_CSV, PPM_CSV):
        if os.path.abspath(path) == os.path.abspath(csv):
            raise RuntimeError(f"refusing to overwrite input CSV: {path}")
    return path


@dataclasses.dataclass(frozen=True)
class Flag:
    """A named, machine-checkable audit condition."""
    name: str
    detail: str
