"""m7a_data.py - read the materialised tiles, compute A1-A4, aggregate exactly as the pipeline does.

Tiles are READ. Nothing here regenerates a tile, writes into data/, or re-runs preprocessing; the
qualifying-tile filter is copied from the frozen Model 5 loader so the pixel set is identical. The
only new computation is the four statistics in spatial_features.py.

WHAT THIS MODULE ADDS ON TOP OF THE BRIEF'S ASSEMBLY is the census, and the census exists because
three different facts were being confused:

  (1) INVALID CALL - tile_spatial raised a ValueError (or anything else unexpected) on a real tile,
      OR tile_spatial wrote NaN into a key whose own registered definition returns a number on that
      same tile. The second case is a disagreement inside the implementation rather than a fact
      about the tile, so it is raised as a ValueError too: scan_tiles names the tile in the context
      it leaves behind, records the reason, and lets the exception propagate. It is never counted as
      degeneracy, never converted to NaN, never skipped. The per-tile loop catches one type only,
      sf.TileDegenerate; there is no broad except, and a RuntimeError from the statistics layer
      reaches the caller (check_m7a_data.py forces both).
  (2) VALID TILE, DEGENERATE STATISTIC - sf.TileDegenerate for the whole tile (status
      `degenerate_tile`, all four keys NaN), or a NaN in one key of an otherwise fine tile whose own
      definition refuses it the same way (status `nan_keys`). Both counted, with per-key NaN tallies
      and the reason string captured.
  (3) VALID FINITE STATISTIC - status `ok`, counted with percentiles.

Nothing is clamped, transformed, winsorised or dropped on the way to the census: outcome (3)'s
percentiles are the numbers tile_spatial returned. The extreme-outlier block MEASURES them and
reports; it does not act. Because |value - median| / MAD is one-sided for a key that never changes
sign - no value below a positive median can reach z 10 once median / MAD is under 10, which is the
whole of A3's lower tail - the block reports a second rule on the log scale, where 0.1x and 10x the
median are the same distance from it. A zero MAD is its own reported condition rather than a count
of zero, because "nothing flagged" and "nothing could be flagged" are different sentences. That
second look is here because one synthetic tile once answered A2 with 1.2e+13, and a single such
value in a standardised column dominates a rank-3 fit.

`normalized_ppm` is carried for one purpose - it is the physical scale argument tile_spatial needs -
and is dropped from both design frames. `camera`, `camera_fam`, `cv_group`, `source_effective_ppm`,
`source_image_width` and `source_image_height` are dropped at load, so a design assembled here
cannot be selected on them later.

Aggregation is sf.image_block then sf.soil_block, the functions check_spatial_features.py tests, so
the spec's claim that the block is "aggregated by the same median rules as the existing 12" is
executed by the same code that is checked. The contract for that is behavioural: the checker spies
on the calls, feeds the blocks an inf (which the tested path refuses and a plain median would
publish) and an all-NaN column, and re-derives both frames from the blocks directly.

Warnings are recorded for all three stages of a pass - the tile loop, the image aggregation and the
soil aggregation - because sf._median_column warns at the aggregation stage, which is where a dead
column actually surfaces. Recording them is not silencing them: the census counts them per stage and
the contract treats a non-zero tally as a failure, so "warnings: 0" is a claim about the whole pass.

Artifacts written by run_census: data/tile_census.csv (one row per tile), data/census.json (the
aggregate), data/image_features_m7a.csv and data/soil_features_m7a.csv (the two assembled design
frames, so the matrix a fit would consume can be audited without re-running anything). No
wall-clock timestamp is stored in any of them, so they are byte-comparable between runs and the
determinism claim can be tested on the files themselves. Nothing non-finite is ever written to
census.json: publish() looks for one on the aggregate before serialising and refuses, rather than
letting the JSON writer's sanitiser turn it into null.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT / "Model 5" / "Model 5 Experiment 1", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import spatial_features as sf  # noqa: E402
import transfer_eval as tv  # noqa: E402

KEYS: Tuple[str, ...] = tuple(sf.SPATIAL)
STATUS_OK = "ok"
STATUS_DEGENERATE = "degenerate_tile"
STATUS_NAN_KEYS = "nan_keys"
STATUS_INVALID = "invalid_call"          # never reaches the CSV: outcome (1) aborts the pass
CENSUS_STATUSES = (STATUS_OK, STATUS_DEGENERATE, STATUS_NAN_KEYS)
WARNING_STAGES = ("tiles", "image_block", "soil_block")

ART_DIR = HERE / "data"
CSV_NAME = "tile_census.csv"
JSON_NAME = "census.json"
IMG_CSV_NAME = "image_features_m7a.csv"
SOIL_CSV_NAME = "soil_features_m7a.csv"
CENSUS_COLUMNS: List[str] = ["tile_path", "sample_id", "split", "normalized_ppm", *KEYS, "status",
                             *["%s_nan" % k for k in KEYS], "n_nan", "reason"]
REASON_MAX = 320
OUTLIER_Z = 10.0
OUTLIER_MAGNITUDE = 1e6
DROPPED_AT_LOAD = ("camera", "camera_fam", "cv_group", "source_effective_ppm",
                   "source_image_width", "source_image_height")
# One meaning, everywhere this word appears in an artifact: a value is non-finite when it is NaN or
# +-inf. Anything that counts them says which unit it counts in (a tile, or a cell), because the
# first version of this census used the same key for a tile count and an inf count.
NON_FINITE_MEANING = ("a value that is NaN or +-inf; *_tiles counts tiles carrying at least one, "
                      "*_cells and per-key tallies count values")

NAN_REPORT: Dict[str, object] = {}
_CACHE: Dict[Tuple[str, bool, int], Dict[str, object]] = {}
# The statistic tile_spatial can leave NaN, and the module-internal function that raises the
# TileDegenerate explaining why. Used ONLY to explain a key that came back NaN and to decide whether
# that explanation agrees with tile_spatial; it never produces a value, and it is never on the path
# that computes one.
_KEY_PROBE = {"A2": "_a2", "A3": "_a3", "A4": "_a4"}
_KEY_ARGS = {"A2": lambda g, m, e, ppm: (e, m), "A3": lambda g, m, e, ppm: (g, m, ppm),
             "A4": lambda g, m, e, ppm: (g, m, ppm)}


# --------------------------------------------------------------------------- the frozen pixel set
def find_input_root() -> Path:
    """The repository root, which is where data/ and the model folders live."""
    for cand in (ROOT, Path.cwd(), Path("/kaggle/input")):
        if (cand / "data" / "processed_meta" / "manifest_images.csv").exists():
            return cand
    raise AssertionError("no data/processed_meta/manifest_images.csv found from any known root")


def tile_table(root) -> pd.DataFrame:
    t = pd.read_csv(Path(root) / "data" / "processed_meta" / "manifest_tiles.csv")
    q = t[(t.tile_size_px == tv.TILE_PX) & t.materialized
          & (t.soil_fraction >= tv.MIN_SOIL_FRACTION)].copy()
    # dropped, not merely unused: the manifest carries camera and cv_group, and a design that cannot
    # see them cannot accidentally be selected on them later
    q = q.drop(columns=[c for c in DROPPED_AT_LOAD if c in q.columns])
    if "normalized_ppm" not in q.columns:
        raise AssertionError("normalized_ppm missing; the statistics have no physical scale")
    if q.empty:
        raise AssertionError("no qualifying tiles; the frozen filter did not match")
    return q


# --------------------------------------------------------------------------- the three outcomes
def _trunc(text: str, limit: int = REASON_MAX) -> str:
    return str(text)[:limit]


def _read_tile(path: Path) -> np.ndarray:
    """One tile, as float. Closed through a context manager so PIL cannot hand back a resource
    warning: under ruling 2 a warning is a failure, and an unclosed handle would be a false one."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), float)


def compute_tile_row(a: np.ndarray, ppm: float):
    """The four statistics for one tile: (row, status, nan_keys, reason).

    TileDegenerate is NOT caught here - a whole-tile degeneracy has to reach the caller as itself,
    which is what makes the split between outcomes (1) and (2) checkable from outside.
    """
    row = sf.tile_spatial(a, ppm)
    nan_keys = tuple(k for k in KEYS if not np.isfinite(row[k]))
    return row, (STATUS_OK if not nan_keys else STATUS_NAN_KEYS), nan_keys, ""


def _key_reason(a: np.ndarray, ppm: float, key: str) -> Tuple[str, bool]:
    """Why tile_spatial wrote NaN into this one key, as text, and whether that answer DISAGREES.

    tile_spatial swallows the TileDegenerate that produced a per-key NaN, so the reason is
    recovered by re-running that key's registered definition on the same mask. Diagnostic only: it
    runs on tiles that already came back NaN, and it contributes no value to the census. The second
    half of the answer is the load-bearing part. A definition that REFUSES the tile is agreement;
    a definition that returns a NUMBER where tile_spatial wrote NaN is a disagreement inside the
    implementation, and the caller must treat it as outcome (1) rather than tallying it as (2).
    """
    g, m, e = sf.mask_and_field(a, ppm)
    name = _KEY_PROBE.get(key)
    fn = getattr(sf, name, None) if name else None
    if fn is None:
        return ("%s is NaN and this module has no registered undefined branch for it; A1 has no "
                "NaN path, so that is a bug rather than a degeneracy" % key, True)
    try:
        val = float(fn(*_KEY_ARGS[key](g, m, e, ppm)))
    except sf.TileDegenerate as exc:
        return str(exc), False
    return ("tile_spatial wrote NaN but %s returned %r, which is a bug to report, not a "
            "degeneracy to count" % (name, val), True)


def classify_tile(a: np.ndarray, ppm: float) -> Dict[str, object]:
    """Instrumented single-tile call. This is where the three outcomes are separated.

    Raises ValueError when the statistics layer disagrees with itself: a key tile_spatial left NaN
    while its own registered definition measures it is not a degenerate statistic, and counting one
    would launder a bug into outcome (2).
    """
    try:
        row, status, nan_keys, reason = compute_tile_row(a, ppm)
    except sf.TileDegenerate as exc:
        # TileDegenerate is the ONLY thing a bad tile may raise. It is not an AssertionError and
        # not a ValueError, so a systemic error can never be logged here as degeneracy.
        return {"values": {k: float("nan") for k in KEYS}, "status": STATUS_DEGENERATE,
                "nan_keys": list(KEYS), "reason": _trunc(str(exc))}
    if status == STATUS_NAN_KEYS:
        parts, disagreed = [], []
        for k in nan_keys:
            text, bad = _key_reason(a, ppm, k)
            parts.append("%s: %s" % (k, text))
            if bad:
                disagreed.append(k)
        if disagreed:
            raise ValueError("%s came back NaN from tile_spatial but its own registered definition "
                             "measures this tile: %s" % (",".join(disagreed), "; ".join(parts)))
        reason = _trunc("; ".join(parts))
    return {"values": row, "status": status, "nan_keys": list(nan_keys), "reason": reason}


def new_ctx() -> Dict[str, object]:
    return {"records": [], "warnings": [], "errors": [], "current": None, "aborted": False,
            "stage": None}


def _warning_hook(ctx: Dict[str, object]):
    """A showwarning replacement that attributes every warning to the row and stage in flight."""
    def hook(message, category, filename, lineno, file=None, line=None):
        cur = ctx.get("current") or {}
        ctx["warnings"].append({"stage": ctx.get("stage"), "tile_path": cur.get("tile_path"),
                                "row": cur.get("row"),
                                "parent_image_path": cur.get("parent_image_path"),
                                "sample_id": cur.get("sample_id"), "category": category.__name__,
                                "message": _trunc(str(message), 200), "filename": str(filename),
                                "lineno": int(lineno or 0)})
    return hook


@contextlib.contextmanager
def _sink(ctx: Dict[str, object], stage: str):
    """Record every warning raised inside this block against this stage, then hand the process back.

    Installed around each of the three stages a pass has (see WARNING_STAGES), because the
    aggregation is where a dead column warns and a census that only watched the tile loop would
    report zero warnings for a run whose stderr was not silent.
    """
    ctx.setdefault("warnings", [])
    outer = warnings.showwarning
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _warning_hook(ctx)
        ctx["stage"] = stage
        try:
            yield
        finally:
            ctx["stage"] = None
            warnings.showwarning = outer


def _record_abort(ctx: Dict[str, object], exc: BaseException, stage: Optional[str] = None) -> None:
    """Outcome (1), as far as the census is concerned: name the tile, keep the reason, propagate.

    Called from a finally block, so it records without catching. The exception keeps going. The
    stage is passed in rather than read back off the context, because the sink that set it has
    already exited by the time this runs and resets it to None.
    """
    cur = ctx.get("current") or {}
    kind = "missing_tile" if (isinstance(exc, AssertionError) and "missing tile" in str(exc)) \
        else type(exc).__name__
    ctx["errors"].append({"stage": stage or ctx.get("stage") or "tiles",
                          "tile_path": cur.get("tile_path"), "row": cur.get("row"), "kind": kind,
                          "reason": _trunc("%s: %s" % (kind, exc), 400)})
    ctx["aborted"] = True


def scan_tiles(root, q: Optional[pd.DataFrame] = None, scrambled: bool = False,
               seed: int = 90001, ctx: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """One instrumented pass over the qualifying tiles, in manifest order.

    Warnings are recorded, not silenced and not raised: the census has to say WHICH tile produced
    WHICH warning. The `finally` records an escaping exception as an invalid call naming its tile,
    and the exception propagates - an invalid call aborts the pass rather than becoming a row.
    """
    root = Path(root)
    q = tile_table(root) if q is None else q
    ctx = new_ctx() if ctx is None else ctx
    ctx.update({"n_tiles": int(len(q)), "scrambled": bool(scrambled), "seed_base": int(seed)})
    try:
        with _sink(ctx, "tiles"):
            for r in q.itertuples():
                # named BEFORE the file is touched: an abort has to accuse the tile it was working
                # on, and the structured tile_path/row fields are what the artifact publishes
                ctx["current"] = {"tile_path": r.tile_path, "row": int(r.Index),
                                  "sample_id": str(r.sample_id), "split": str(r.split)}
                path = root / r.tile_path
                if not path.exists():
                    raise AssertionError("manifest references a missing tile: %s" % r.tile_path)
                a = _read_tile(path)
                if scrambled:
                    a = sf.phase_scrambled(a, seed=seed + int(r.Index))
                rec = classify_tile(a, float(r.normalized_ppm))
                rec.update({"tile_path": r.tile_path, "sample_id": str(r.sample_id),
                            "split": str(r.split), "normalized_ppm": float(r.normalized_ppm),
                            "row": int(r.Index), "tile_w_mm": float(r.tile_w_mm),
                            "seed_used": (int(seed) + int(r.Index)) if scrambled else None})
                ctx["records"].append(rec)
            ctx["current"] = None
    finally:
        exc = sys.exc_info()[1]
        if exc is not None:
            _record_abort(ctx, exc, stage="tiles")
    return ctx


def report_from(ctx: Dict[str, object]) -> Dict[str, object]:
    """The one tally surface for a pass. Every non-finite word here means NON_FINITE_MEANING.

    Tiles and cells are named separately because the first version used one key for both: `non_finite`
    counted tiles in one place and infs in another. A rate carries the unit of its own numerator.
    """
    recs, errs, warns = ctx["records"], ctx["errors"], ctx.get("warnings") or []
    n = len(recs)
    cells = [[r["values"].get(k, float("nan")) for k in KEYS] for r in recs]
    finite = [[bool(np.isfinite(v)) for v in row] for row in cells]
    isnan = [[bool(np.isnan(v)) for v in row] for row in cells]
    tiles_nf = sum(1 for row in finite if not all(row))
    tiles_nan = sum(1 for row in isnan if any(row))
    per_key = {k: int(sum(1 for r in recs if k in r["nan_keys"])) for k in KEYS}
    per_key_nf = {k: int(sum(1 for row in cells if not np.isfinite(row[i])))
                  for i, k in enumerate(KEYS)}
    by_stage = {s: sum(1 for w in warns if w.get("stage") == s) for s in WARNING_STAGES}
    return {"meaning": NON_FINITE_MEANING, "n_tiles": int(n),
            "degenerate_mask": int(sum(1 for r in recs if r["status"] == STATUS_DEGENERATE)),
            "non_finite": int(tiles_nf), "tiles_with_nan": int(tiles_nan),
            "non_finite_cells": int(sum(int(not f) for row in finite for f in row)),
            "nan_cells": int(sum(int(f) for row in isnan for f in row)),
            # the brief's nan_rate is the fraction of tiles carrying a non-finite value, and it is
            # kept at that formula: non_finite and nan_rate were always the same measurement here.
            # The NaN-only fraction is published beside it under a name that says so.
            "nan_rate": float(tiles_nf / n) if n else 0.0,
            "nan_only_rate": float(tiles_nan / n) if n else 0.0,
            "non_finite_rate": float(tiles_nf / n) if n else 0.0,
            "per_key_nan": per_key, "per_key_nonfinite": per_key_nf,
            "invalid_calls": int(len(errs)), "warnings": int(len(warns)),
            "warnings_by_stage": by_stage,
            "scrambled": bool(ctx.get("scrambled", False)),
            "seed_base": int(ctx.get("seed_base", 0))}


def _compute(root, q: pd.DataFrame, scrambled: bool = False, seed: int = 90001,
             ctx: Optional[Dict[str, object]] = None) -> List[Dict[str, float]]:
    """The brief's row-list engine: one pass, four numbers per tile, NAN_REPORT filled in."""
    ctx = scan_tiles(root, q, scrambled=scrambled, seed=seed, ctx=ctx)
    rows = [r["values"] for r in ctx["records"]]
    if not scrambled:
        NAN_REPORT.update(report_from(ctx))
    return rows


# --------------------------------------------------------------------------- the frozen aggregation
def image_table(q: pd.DataFrame, ctx: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Median of a soil's tile rows, through sf.image_block. Grouped without camera, by design.

    With a ctx the stage's warnings are recorded against the image that produced them; without one
    they behave the way any other warning does. Both aggregations are inside the sink the tile loop
    gets, because a dead column warns HERE and not in the loop.
    """
    out = {}
    # grouped by image and soil only, and camera is not in q at all by the time we get here
    sink = _sink(ctx, "image_block") if ctx is not None else contextlib.nullcontext()
    with sink:
        for (img, sid), grp in q.groupby(["parent_image_path", "sample_id"]):
            if img in out:
                raise AssertionError("one image path carries two soils, so the assembly would "
                                     "silently drop one: %s" % img)
            if ctx is not None:
                ctx["current"] = {"tile_path": None, "row": None, "parent_image_path": img,
                                  "sample_id": str(sid)}
            out[img] = dict(sf.image_block(list(grp._row)), sample_id=str(sid))
        if ctx is not None:
            ctx["current"] = None
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = "parent_image_path"
    df["sample_id"] = df["sample_id"].astype(str)
    return df[list(KEYS) + ["sample_id"]]


def soil_table(img: pd.DataFrame, ctx: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Median of an image's rows, through sf.soil_block. NOT a second pandas median."""
    out = {}
    sink = _sink(ctx, "soil_block") if ctx is not None else contextlib.nullcontext()
    with sink:
        for sid, grp in img.groupby("sample_id"):
            if ctx is not None:
                ctx["current"] = {"tile_path": None, "row": None, "parent_image_path": None,
                                  "sample_id": str(sid)}
            out[sid] = sf.soil_block([r for r in grp.to_dict("records")])
        if ctx is not None:
            ctx["current"] = None
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = "sample_id"
    return df[list(KEYS)]


def tile_frame(records: List[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"tile_path": r["tile_path"], "sample_id": r["sample_id"], "split": r["split"],
               "normalized_ppm": r["normalized_ppm"]}
        for k in KEYS:
            row[k] = r["values"][k]
            row["%s_nan" % k] = k in r["nan_keys"]
        row["status"] = r["status"]
        row["n_nan"] = len(r["nan_keys"])
        row["reason"] = r["reason"]
        rows.append(row)
    return pd.DataFrame(rows, columns=CENSUS_COLUMNS)


# --------------------------------------------------------------------------- the assembly
def assembly(root=None, scrambled: bool = False, seed: int = 90001) -> Dict[str, object]:
    """The pass plus both frames, memoised per (root, scrambled, seed) within this process.

    The memo exists so a contract file can ask the same question twice without paying three minutes
    for it. It is not how determinism is tested: clear_assembly_cache() forces a real recomputation,
    and the third run happens in a process that never had the cache at all.
    """
    root = Path(root) if root is not None else find_input_root()
    key = (str(root), bool(scrambled), int(seed))
    if key in _CACHE:
        return _CACHE[key]
    q = tile_table(root)
    ctx = new_ctx()
    rows = _compute(root, q, scrambled=scrambled, seed=seed, ctx=ctx)
    q = q.assign(_row=rows)
    img, tf = image_table(q, ctx=ctx), tile_frame(ctx["records"])
    soil = soil_table(img, ctx=ctx)
    counts = {k: int(v) for k, v in tf.status.value_counts().items()}
    asm = {"root": root, "q": q, "records": ctx["records"], "tile_frame": tf, "image": img,
           "soil": soil, "status_counts": counts, "warnings": ctx["warnings"],
           "errors": ctx["errors"], "aborted": bool(ctx["aborted"]),
           "degenerate_examples": [{"tile_path": r["tile_path"], "status": r["status"],
                                    "nan_keys": r["nan_keys"], "reason": r["reason"]}
                                   for r in ctx["records"] if r["status"] != STATUS_OK][:12],
           "hashes": (frame_hash(tf), frame_hash(img), frame_hash(soil)),
           "scrambled": bool(scrambled), "seed": int(seed), "report": report_from(ctx)}
    _CACHE[key] = asm
    return asm


def clear_assembly_cache() -> None:
    _CACHE.clear()


def spatial_image_table(root, scrambled: bool = False, seed: int = 90001) -> pd.DataFrame:
    """One row per image: A1-A4 (tile median) plus sample_id. Indexed by parent_image_path."""
    return assembly(root, scrambled=scrambled, seed=seed)["image"].copy()


def spatial_soil_table(root, scrambled: bool = False, seed: int = 90001) -> pd.DataFrame:
    """Soil rows via sf.soil_block, NOT a second pandas median.

    The tested aggregation is the only one allowed, otherwise the spec's claim that the block is
    "aggregated by the same median rules as the existing 12" would be checked by one function and
    executed by another.
    """
    return assembly(root, scrambled=scrambled, seed=seed)["soil"].copy()


# --------------------------------------------------------------------------- the census
def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def key_stats(col: np.ndarray) -> Dict[str, object]:
    """One key's distribution over a frame, with NaN and +-inf counted apart and then together.

    The percentiles are the raw numbers tile_spatial returned, over the finite values only. Counting
    the non-finite cells is the census's job; borrowing a percentile from them would be an act.
    """
    v = np.asarray(col, float)
    fin = _finite(v)
    n_nan, n_inf = int(np.count_nonzero(np.isnan(v))), int(np.count_nonzero(np.isinf(v)))
    stats: Dict[str, object] = {"nan_count": n_nan, "inf_count": n_inf,
                                "nonfinite_count": n_nan + n_inf, "n_finite": int(fin.size)}
    if fin.size:
        pct = np.percentile(fin, [0.0, 5.0, 50.0, 95.0, 100.0])
        stats.update({f: float(x) for f, x in zip(("min", "p5", "median", "p95", "max"), pct)})
    else:
        stats.update({f: None for f in ("min", "p5", "median", "p95", "max")})
    return stats


def vector_summary(col: np.ndarray) -> Dict[str, object]:
    """The same five percentiles over one assembled frame's column, at the level the model is
    actually fitted on. Reported so the census shows what the standardiser will meet."""
    return key_stats(col)


def finiteness(asm: Dict[str, object]) -> Dict[str, object]:
    """Non-finite cells counted directly off the assembled arrays, before any writer sees them.

    The strict-JSON leg used to be the finiteness check, and it could not fail: the writer converted
    anything non-finite to null on the way out, so an inf in a reported number arrived as a clean
    artifact. These counts come off the numpy arrays themselves and are published next to the hashes.
    """
    out: Dict[str, object] = {"meaning": NON_FINITE_MEANING}
    for level, frame in (("tile", asm["tile_frame"]), ("image", asm["image"]),
                         ("soil", asm["soil"])):
        arr = frame[list(KEYS)].to_numpy(float)
        out[level] = {"cells": int(arr.size), "nan_cells": int(np.count_nonzero(np.isnan(arr))),
                      "inf_cells": int(np.count_nonzero(np.isinf(arr))),
                      "non_finite_cells": int(np.count_nonzero(~np.isfinite(arr)))}
        out[level]["tiles_with_nonfinite"] = int(np.count_nonzero(
            ~np.isfinite(arr).any(axis=1)))
    return out


def outlier_stats(col: np.ndarray) -> Dict[str, object]:
    """Two robust looks at one key's spread - absolute, and multiplicative. Reports, never acts.

    |value - median| / MAD is one-sided for a key that never changes sign. A2's median over these
    tiles is about 1.0 and its MAD about 0.2, so a value below the median can only ever reach z ~ 5
    no matter how close to zero it is: the low tail was structurally invisible, and A3's whole lower
    tail - including the tiles tied at its minimum - sat under the same cap. The log rule measures
    the same distance multiplicatively, which is how a ratio, a lag fraction and a log eigenvalue
    ratio are built, and it is symmetric by construction. A key whose finite values straddle zero
    has no multiplicative scale, so the log rule says NOT APPLICABLE rather than inventing one, and
    a key whose MAD is zero says so instead of reporting a clean count of zero.

    The 1e6 question is not a tuned threshold: A1 is a log of an eigenvalue ratio, A3 a lag
    fraction, A4 a difference of two fractions, and A2 a ratio of two lengths each at least one full
    lag step long, so every one of them is bounded orders of magnitude below 1e6 by its own
    definition. A distribution sitting above 1e6 would be a units bug, which is what one synthetic
    tile's A2 = 1.2e+13 looked like before the A2 ruling.
    """
    v = np.asarray(col, float)
    fin = _finite(v)
    out: Dict[str, object] = {"n_values": int(v.size), "n_finite": int(fin.size),
                              "mad_z_threshold": OUTLIER_Z, "log_z_threshold": OUTLIER_Z,
                              "magnitude_floor": OUTLIER_MAGNITUDE}
    if not fin.size:
        return {**out, "mad": None, "mad_is_zero": False, "n_distinct": 0, "median": None,
                "count_beyond_10_mad": None, "max_abs_z": None, "max_abs_value": None,
                "n_abs_above_1e6": 0, "whole_distribution_above_1e6": False,
                "log_two_sided_applicable": False, "log_of": None, "log_note": None,
                "max_abs_z_log": None, "count_beyond_10_mad_log": None,
                "interpretation": "no finite value for this key, so neither rule has an opinion"}
    med = float(np.percentile(fin, 50.0))
    mad = float(np.percentile(np.abs(fin - med), 50.0))
    absz = np.abs(fin - med)
    if mad > 0.0:
        z = absz / mad
        beyond, max_z = int(np.count_nonzero(z > OUTLIER_Z)), float(np.max(z))
        interpret = "the absolute rule is armed on this key"
    else:
        beyond, max_z = None, None      # a zero robust scale is reported, not divided by
        interpret = ("MAD is zero, so |value - median| / MAD has no scale: a zero count here would "
                     "be not a claim of clean")
    mag = np.abs(fin)
    positive, negative = bool((fin > 0.0).all()), bool((fin < 0.0).all())
    applicable = positive or negative
    log_note = None
    if positive:
        log_note = "the key is strictly positive, so the multiplicative rule reads its values"
    elif negative:
        log_note = ("the key is strictly negative, so this look is on the MAGNITUDE (log of |x|): "
                    "a second-order reading of how far apart the contrasts are, not a ratio of the "
                    "signed statistic, and it is reported separately for that reason")
    if applicable:
        s = np.log(np.abs(fin))
        med_s = float(np.percentile(s, 50.0))
        mad_s = float(np.percentile(np.abs(s - med_s), 50.0))
        if mad_s > 0.0:
            z_log = np.abs(s - med_s) / mad_s
            max_z_log, count_log = float(np.max(z_log)), int(np.count_nonzero(z_log > OUTLIER_Z))
        else:
            max_z_log, count_log = None, None
    else:
        max_z_log, count_log = None, None
    return {**out, "mad": mad, "mad_is_zero": bool(mad == 0.0),
            "n_distinct": int(len(np.unique(fin))), "median": med,
            "count_beyond_10_mad": beyond, "max_abs_z": max_z,
            "log_two_sided_applicable": applicable,
            "log_of": ("value" if positive else "abs" if negative else None),
            "log_note": log_note,
            "max_abs_z_log": max_z_log, "count_beyond_10_mad_log": count_log,
            "max_abs_value": float(np.max(mag)), "n_abs_above_1e6": int(np.count_nonzero(
                mag > OUTLIER_MAGNITUDE)),
            "whole_distribution_above_1e6": bool(np.all(mag > OUTLIER_MAGNITUDE)),
            "interpretation": interpret}


def find_nonfinite(obj: object, path: str = "$") -> List[str]:
    """Every place a NaN or +-inf is hiding in a report structure, by its JSON path.

    Runs on the aggregate BEFORE it is serialised, so a non-finite number is a named defect at a
    named path instead of a null the reader is expected to trust.
    """
    hits: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            hits += find_nonfinite(v, "%s.%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            hits += find_nonfinite(v, "%s[%d]" % (path, i))
    elif isinstance(obj, (bool, np.bool_)):
        return hits
    elif isinstance(obj, (float, np.floating)):
        if not np.isfinite(float(obj)):
            hits.append(path)
    elif isinstance(obj, np.ndarray):
        if obj.size and not np.all(np.isfinite(obj)):
            hits.append(path)
    return hits


def require_publishable(cen: Dict[str, object], asm: Optional[Dict[str, object]] = None) -> None:
    """Refuse to publish a report that contains a non-finite number, or a frame that carries one.

    Two different surfaces are guarded. The aggregate's own floats must all be finite, because the
    writer used to convert a non-finite to null and the strict-JSON check then passed. And the frames
    must be clean in the way the census claims they are: an +-inf is never legal, a design frame
    (image or soil) never holds a NaN, and a NaN in the tile frame is legal only where the pass
    counted that tile as degenerate and recorded a reason for it.
    """
    bad = find_nonfinite(cen)
    if bad:
        raise ValueError("census aggregate carries %d non-finite value(s) at %s; the artifact is "
                         "not written over them" % (len(bad), ", ".join(bad[:8])))
    if asm is None:
        return
    inf_rows = [r["tile_path"] for r in asm["records"]
                if any(np.isinf(v) for v in r["values"].values())]
    if inf_rows:
        raise ValueError("the tile pass returned %d row(s) carrying +-inf (%s); an infinite "
                         "statistic is a bug, not a degeneracy, so no artifact is written"
                         % (len(inf_rows), ", ".join(str(x) for x in inf_rows[:3])))
    for level in ("image", "soil"):
        frame = asm[level]
        col = frame[list(KEYS)].to_numpy(float)
        rows = np.isinf(col).any(axis=1)
        if int(rows.sum()):
            labels = ", ".join(str(x) for x in np.asarray(frame.index)[rows][:3])
            raise ValueError("the %s frame carries %d +-inf cell(s) across %d row(s) (%s); no "
                             "artifact is written" % (level, int(np.count_nonzero(np.isinf(col))),
                                                      int(rows.sum()), labels))
    for level in ("image", "soil"):
        frame = asm[level]
        col = frame[list(KEYS)].to_numpy(float)
        rows = np.isnan(col).any(axis=1)
        if int(rows.sum()):
            labels = ", ".join(str(x) for x in np.asarray(frame.index)[rows][:3])
            raise ValueError("the %s design frame carries %d NaN cell(s) across %d row(s) (%s); a "
                             "fitted design may not hold one, so no artifact is written"
                             % (level, int(np.count_nonzero(np.isnan(col))), int(rows.sum()), labels))
    uncounted = [r for r in asm["records"]
                 if any(not np.isfinite(v) for v in r["values"].values())
                 and (r["status"] == STATUS_OK or not r["reason"])]
    if uncounted:
        raise ValueError("%d tile(s) hold a non-finite value the pass filed as neither degenerate "
                         "nor reasoned: %s" % (len(uncounted),
                                               ", ".join(r["tile_path"] for r in uncounted[:3])))


def aggregate(asm: Dict[str, object]) -> Dict[str, object]:
    tf, q, img, soil = asm["tile_frame"], asm["q"], asm["image"], asm["soil"]
    counts = asm["status_counts"]
    total = int(len(tf))
    ok = int(counts.get(STATUS_OK, 0))
    rep = report_from(asm_records_ctx(asm))
    fin = finiteness(asm)
    return {
        "generated_by": "m7a_data.run_census",
        "task": "M7-A task 3: tile and image assembly census",
        "scrambled": bool(asm["scrambled"]), "seed": int(asm["seed"]),
        "total_tiles": total,
        "successful_tiles": ok,
        "tiles_with_any_nan": int(total - ok),
        "non_finite_tiles": int(rep["non_finite"]),
        "degenerate_tiles": int(counts.get(STATUS_DEGENERATE, 0)),
        "nan_key_tiles": int(counts.get(STATUS_NAN_KEYS, 0)),
        "invalid_calls": int(len(asm["errors"])),
        "aborted": bool(asm["aborted"]),
        "per_status": dict(counts),
        "per_key": {k: key_stats(tf[k].to_numpy(float)) for k in KEYS},
        "image_percentiles": {k: vector_summary(img[k].to_numpy(float)) for k in KEYS},
        "soil_percentiles": {k: vector_summary(soil[k].to_numpy(float)) for k in KEYS},
        "finiteness": fin,
        "extreme_outlier": {"level": "tile", "rule": "two robust looks at each key: |value - median| "
                                                     "/ MAD, and the same rule on log|value| where "
                                                     "the key keeps one sign. A zero MAD and a "
                                                     "non-applicable log rule are reported as such. "
                                                     "Report only: no clamp, no transform, no "
                                                     "winsorise, no row dropped",
                             "mad_z_threshold": OUTLIER_Z, "log_z_threshold": OUTLIER_Z,
                             "magnitude_floor": OUTLIER_MAGNITUDE,
                             "keys": {k: outlier_stats(tf[k].to_numpy(float)) for k in KEYS},
                             "soil_level": {k: outlier_stats(soil[k].to_numpy(float))
                                            for k in KEYS}},
        "images": int(len(img)), "soils": int(len(soil)),
        "tiles_by_split": {str(k): int(v) for k, v in q.split.value_counts().items()},
        "images_by_split": {str(k): int(v) for k, v in
                            q.groupby("parent_image_path")["split"].first().value_counts().items()},
        "train_soils": int(q[q.split == "train"].sample_id.nunique()),
        "test_soils": int(q[q.split == "test"].sample_id.nunique()),
        "normalized_ppm_values": sorted(float(x) for x in q.normalized_ppm.unique()),
        "tile_w_mm": float(q.tile_w_mm.iloc[0]),
        "camera_columns_present": [c for c in DROPPED_AT_LOAD
                                   if c in img.columns or c in soil.columns or c in q.columns],
        "columns": {"image_table": [str(c) for c in img.columns],
                    "soil_table": [str(c) for c in soil.columns],
                    "tile_census": CENSUS_COLUMNS},
        "warnings_count": int(rep["warnings"]), "warnings_by_stage": rep["warnings_by_stage"],
        "errors_count": int(len(asm["errors"])),
        "warning_records": asm["warnings"][:50], "error_records": asm["errors"][:50],
        "degenerate_examples": asm["degenerate_examples"],
        "nan_report": rep,
        "determinism_hashes": {"tile_vector": asm["hashes"][0], "image_vector": asm["hashes"][1],
                              "soil_vector": asm["hashes"][2]},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "pandas": pd.__version__},
    }


def asm_records_ctx(asm: Dict[str, object]) -> Dict[str, object]:
    """The report shape from a cached assembly, so NAN_REPORT survives a memoised second call."""
    return {"records": asm["records"], "errors": asm["errors"], "warnings": asm["warnings"],
            "scrambled": asm["scrambled"], "seed_base": asm["seed"]}


def _json_safe(obj):
    """Convert what JSON genuinely cannot hold (numpy scalars), and REFUSE what it must not hide.

    The first version of this function rewrote NaN and +-inf to null, which made the artifact's
    strictness a property of the writer instead of a property of the measurement: an inf could not
    fail the strict-JSON check because it never reached the text. require_publishable() now looks
    for the non-finite numbers before this function is ever called, and this refuses to launder one
    if it somehow arrives afterwards.
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not np.isfinite(f):
            raise ValueError("refusing to write %r into a JSON artifact; a non-finite number is a "
                             "defect to name, not a null to hide" % f)
        return f
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def frame_hash(df: pd.DataFrame) -> str:
    """sha256 over a frame's index, column names and raw float bytes. Byte-identical or nothing."""
    h = hashlib.sha256()
    h.update(repr(list(df.columns)).encode("utf-8"))
    h.update(str(df.index.name).encode("utf-8"))
    for i in df.index.astype(str):
        h.update(str(i).encode("utf-8"))
        h.update(b"\x00")
    for c in df.columns:
        s = df[c]
        if np.issubdtype(s.dtype, np.number):
            h.update(np.ascontiguousarray(s.to_numpy(float)).tobytes())
        elif s.dtype == bool or str(s.dtype) == "bool":
            h.update(np.ascontiguousarray(s.to_numpy(bool)).view(np.uint8).tobytes())
        else:
            h.update("|".join(str(x) for x in s.tolist()).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def determinism_hashes(root=None, scrambled: bool = False, seed: int = 90001) -> List[str]:
    """Force a real pass in THIS process and report the three vector hashes."""
    clear_assembly_cache()
    return list(assembly(root, scrambled=scrambled, seed=seed)["hashes"])


def artifact_paths(out_dir=None) -> Dict[str, Path]:
    """The four published files. Named by filename so the artifact map in census.json is readable."""
    out = Path(out_dir) if out_dir is not None else ART_DIR
    return {CSV_NAME: out / CSV_NAME, JSON_NAME: out / JSON_NAME,
            IMG_CSV_NAME: out / IMG_CSV_NAME, SOIL_CSV_NAME: out / SOIL_CSV_NAME}


def read_census_csv(path) -> pd.DataFrame:
    """Re-read the census artifact EXACTLY.

    pandas' default float parser is not correctly rounded: on this corpus a plain read_csv moved A2
    by 1.78e-15, so re-derived statistics would not be the statistics that were measured.
    float_precision="round_trip" makes the artifact's text recover the same doubles tile_spatial
    returned, which is what lets a check compare the file against memory at all.
    """
    df = pd.read_csv(Path(path), float_precision="round_trip")
    for k in KEYS:
        if "%s_nan" % k in df.columns:
            df["%s_nan" % k] = df["%s_nan" % k].astype(bool)
    return df


def read_feature_csv(path) -> pd.DataFrame:
    """Re-read one published design frame, exactly, with its index restored."""
    return pd.read_csv(Path(path), index_col=0, float_precision="round_trip")


def write_frames(asm: Dict[str, object], paths: Dict[str, Path]) -> None:
    """Publish both assembled design frames, so the matrix a fit would consume is auditable off
    disk. Indexed by parent_image_path and sample_id; A1-A4 and sample_id only, never a camera."""
    asm["image"].to_csv(paths[IMG_CSV_NAME], na_rep="NaN", lineterminator="\n")
    asm["soil"].to_csv(paths[SOIL_CSV_NAME], na_rep="NaN", lineterminator="\n")


def publish(asm: Dict[str, object], cen: Optional[Dict[str, object]] = None, out_dir=None,
            write: bool = True) -> Dict[str, object]:
    """Check the numbers, then write. A non-finite value stops the write instead of being sanitised.

    The order is the point: require_publishable() runs before the output directory is even created,
    so a refused publish leaves the previous artifacts untouched rather than half-replaced.
    """
    cen = aggregate(asm) if cen is None else cen
    require_publishable(cen, asm)
    if not write:
        return cen
    paths = artifact_paths(out_dir)
    paths[JSON_NAME].parent.mkdir(parents=True, exist_ok=True)
    asm["tile_frame"].to_csv(paths[CSV_NAME], index=False, na_rep="NaN", lineterminator="\n")
    write_frames(asm, paths)
    paths[JSON_NAME].write_text(json.dumps(_json_safe(cen), indent=2) + "\n", encoding="utf-8",
                                 newline="\n")
    cen["artifact_bytes"] = {name: int(p.stat().st_size) for name, p in paths.items()}
    cen["artifact_sha256"] = {name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for name, p in paths.items()}
    return cen


def run_census(root=None, scrambled: bool = False, seed: int = 90001, write: bool = True,
               out_dir=None) -> Dict[str, object]:
    """Assemble everything, then write the census artifacts and hand back the aggregate."""
    return publish(assembly(root, scrambled=scrambled, seed=seed), out_dir=out_dir, write=write)


def data_fingerprint(root=None) -> Tuple[int, int, str]:
    """(file count, total bytes, hash of path/size/mtime) over the repo's read-only data/."""
    root = Path(root) if root is not None else find_input_root()
    base = root / "data"
    h = hashlib.sha256()
    n = total = 0
    for dirpath, dirs, files in os.walk(base):
        dirs.sort()
        for f in sorted(files):
            p = Path(dirpath) / f
            st = p.stat()
            h.update(("%s|%d|%d\n" % (p.relative_to(root), st.st_size, st.st_mtime_ns)).encode())
            n += 1
            total += int(st.st_size)
    return n, total, h.hexdigest()[:16]
