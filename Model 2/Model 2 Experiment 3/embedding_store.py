"""embedding_store.py - reads the tile embeddings Model 2 E2 already computed, and refuses
to read anything it cannot prove is E2's.

E3 changes one thing: the feature matrix. It does not change how embeddings are produced,
and it has no torch in it. The 384-dimensional blocks it fuses were computed by E2 on a
Kaggle GPU and cached as .npy files; E3 loads them. That makes E3 a CPU-only experiment,
and it makes the provenance of those files the single point where E3 could go silently
wrong: a stale cache, a mis-keyed crop, or a tile list in a different order would all
produce clean-looking numbers for the wrong data.

So this module is mostly suspicion. Every loader call re-derives the tile list E2 would have
used, straight from the manifests, and requires the file to match it exactly -- name, shape,
dtype, row order and path list. Nothing is hard-coded as a row count.

One implementation, shared by scratch/verify_e2_embeddings.py and the E3 notebook, so the
check that runs before the experiment and the code that runs during it cannot diverge.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

EMBED_DIM: int = 384
VIEWS: Sequence[str] = ("real", "res14", "res16")
BACKENDS: Sequence[str] = ("dinov2", "vit_random")

# (backend, crop, view) triples E3 actually consumes. dinov2 at every crop because the
# sensitivity arms fuse at 128/64/32; vit_random at crop 256 only, for the FR arm.
REQUIRED: Tuple[Tuple[str, int, str], ...] = tuple(
    [(b, c, v) for b in ("dinov2",) for c in (256, 128, 64, 32) for v in VIEWS]
    + [(b, c, v) for b in ("vit_random",) for c in (256,) for v in VIEWS]
)


def cache_name(backend: str, crop: int, view: str, tile_px: int, config_hash: str,
               embed_seed: int) -> str:
    """E2's exact naming: embed_<backend>_c<crop>_<tile>_<hash>_<seed><view>.npy.

    The crop is in the key on purpose. A key that ignored it would return the full-tile
    embedding for every arm and turn E3's sensitivity arms into four copies of one number.
    """
    if backend not in BACKENDS:
        raise AssertionError(f"unknown backend {backend!r}")
    if view not in VIEWS:
        raise AssertionError(f"unknown view {view!r}")
    v = "" if view == "real" else f"_{view}"
    return f"embed_{backend}_c{crop}_{tile_px}_{config_hash}_{embed_seed}{v}.npy"


def expected_tile_list(meta_dir: Path, tile_px: int, view: str) -> List[str]:
    """The tile list E2 embedded, re-derived rather than trusted.

    E2 embedded ``tiles`` for the real view and ``TRAIN_TILES`` for the two
    resolution-matched views, where ``tiles`` is every materialised tile of ``tile_px`` in
    manifest order. Reproducing that filter here is what lets the loader demand an exact
    path-list match instead of hoping the order survived.
    """
    t = pd.read_csv(Path(meta_dir) / "manifest_tiles.csv")
    tiles = t[(t.tile_size_px == int(tile_px)) & t.materialized]
    sub = tiles if view == "real" else tiles[tiles.split == "train"]
    return sub.tile_path.tolist()


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_one(store_dir: Path, backend: str, crop: int, view: str, meta_dir: Path,
             tile_px: int, config_hash: str, embed_seed: int,
             allow_pickle: bool = True) -> Dict[str, object]:
    """Load and prove one embedding array. Raises with the reason, never returns partials."""
    store_dir = Path(store_dir)
    fn = cache_name(backend, crop, view, tile_px, config_hash, embed_seed)
    p, pp = store_dir / fn, store_dir / fn.replace(".npy", ".paths.npy")
    if not p.exists() or not pp.exists():
        miss = [str(x) for x in (p, pp) if not x.exists()]
        raise FileNotFoundError(
            f"missing {len(miss)} file(s) for {backend}@c{crop}/{view}: {miss}\n"
            f"  Expected in {store_dir}\n"
            f"  These are Model 2 E2's Kaggle outputs in /kaggle/working/.cache. See "
            f"model_plan_m2e3.md section 10 for the three ways to retrieve them.")

    names = np.load(pp, allow_pickle=allow_pickle).tolist()
    if not isinstance(names, list) or not all(isinstance(s, str) for s in names):
        raise AssertionError(f"{fn}: .paths.npy is not a list of str")
    want = expected_tile_list(meta_dir, tile_px, view)
    if names != want:
        same_set = set(names) == set(want)
        raise AssertionError(
            f"{fn}: tile list does not match the manifest."
            + (" identical members in a DIFFERENT ORDER - rows would be silently "
               "mis-paired" if same_set else
               f" {len(set(want) - set(names))} expected tiles absent, "
               f"{len(set(names) - set(want))} unexpected present"))

    arr = np.load(p)
    if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
        raise AssertionError(f"{fn}: shape {arr.shape} is not (N, {EMBED_DIM})")
    if arr.shape[0] != len(want):
        raise AssertionError(f"{fn}: {arr.shape[0]} rows for {len(want)} tiles")
    if arr.dtype != np.float64:
        raise AssertionError(f"{fn}: dtype {arr.dtype}, expected float64 - "
                            "backbones.embed_tiles returns L2-normalised float64")
    if not np.isfinite(arr).all():
        raise AssertionError(f"{fn}: non-finite values present")
    std = arr.std(axis=0)
    dead = [i for i, s in enumerate(std) if not s > 0.0]
    if dead:
        raise AssertionError(
            f"{fn}: {len(dead)} zero-variance embedding dimensions (first {dead[:5]}). "
            "A collapsed feature must be an error here, not a strange score three cells "
            "later - E1 measured exactly this failure mode in the random-weights arm.")

    return {"file": fn, "backend": backend, "crop": int(crop), "view": view,
            "rows": int(arr.shape[0]), "cols": int(arr.shape[1]), "dtype": str(arr.dtype),
            "bytes": int(p.stat().st_size + pp.stat().st_size),
            "sha256": _sha256(p), "paths_sha256": _sha256(pp),
            "min_abs_row_norm": float(np.abs(np.linalg.norm(arr, axis=1)).min()),
            "arr": arr, "names": names}


def verify_all(store_dir: Path, meta_dir: Path, tile_px: int, config_hash: str,
               embed_seed: int) -> Tuple[List[dict], List[str]]:
    """Load every required array, collecting failures so one run reports all of them.

    Collecting rather than failing on the first is deliberate: these files arrive from a
    manual Kaggle download, and a person should learn about every missing one at once.
    """
    rows, fails = [], []
    for backend, crop, view in REQUIRED:
        try:
            rows.append(load_one(store_dir, backend, crop, view, meta_dir,
                                 tile_px, config_hash, embed_seed))
        except Exception as exc:                              # noqa: BLE001
            fails.append(f"{backend}@c{crop}/{view}: {type(exc).__name__}: {exc}")
    return rows, fails


def manifest_frame(rows: Sequence[dict]) -> pd.DataFrame:
    cols = ["file", "backend", "crop", "view", "rows", "cols", "dtype", "bytes",
            "min_abs_row_norm", "sha256", "paths_sha256"]
    return pd.DataFrame([{k: r[k] for k in cols} for r in rows], columns=cols)
