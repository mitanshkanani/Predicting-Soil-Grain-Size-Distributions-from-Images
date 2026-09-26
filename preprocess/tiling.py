"""STEP 7 - physical tiling.

Tiles partition the *cropped* canonical image, so a tile's physical extent is
exactly tile_px / TARGET_PPM millimetres in every image from every camera. Nothing
is ever whole-image-resized: a 128x128 tile is a 28 mm window of real soil, which
is the whole point of putting the corpus on a common physical scale first.

Manifest rows are emitted for all four requested sizes, but only one size is
written as image files. Materialising all four would be ~20k files, each a
redundant re-encode of pixels already stored losslessly in the canonical image --
pure disk and pure artefact amplification. `tile_path` is always the deterministic
path the tile *would* occupy, plus a `materialized` flag, so changing size later is
a config edit rather than a re-run.

Tiles are deliberately NOT filtered by soil fraction at generation time: that
threshold is a modelling decision, regeneration costs a full pass, and a filter
column costs nothing.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = None


def grid(n_px: int, tile: int, stride: int) -> list[int]:
    """Non-overlapping tile origins, partial edges dropped."""
    if stride <= 0:
        stride = tile
    if n_px < tile:
        return []
    return list(range(0, n_px - tile + 1, stride))


def rows_for_image(row, tile: int, mask: np.ndarray | None) -> list[dict]:
    x0, y0, x1, y1 = [int(v) for v in str(row.crop_rect_canonical).split(",")]
    cw, ch = x1 - x0, y1 - y0
    stride = int(round(tile * (1 - config.TILE_STRIDE_FRAC)))
    xs, ys = grid(cw, tile, stride), grid(ch, tile, stride)
    dropped_w = cw - (len(xs) * stride if xs else 0)
    dropped_h = ch - (len(ys) * stride if ys else 0)
    out = []
    for ty, gy in enumerate(ys):
        for tx, gx in enumerate(xs):
            ax, ay = x0 + gx, y0 + gy
            sf = float(mask[ay:ay + tile, ax:ax + tile].mean()) if mask is not None else float("nan")
            stem = os.path.splitext(row.filename)[0]
            out.append({
                "tile_id": f"{row.image_id}_t{tile}_{tx}_{ty}",
                "tile_size_px": tile,
                "tile_w_mm": tile / config.target_ppm(),
                "tile_h_mm": tile / config.target_ppm(),
                "tile_x": int(ax), "tile_y": int(ay),
                "grid_col": tx, "grid_row": ty,
                "parent_image_path": row.processed_path,
                "sample_id": row.sample_id, "sample_slug": row.sample_slug,
                "split": row.split, "camera": row.camera,
                "cv_group": row.sample_id,
                "source_effective_ppm": round(float(row.effective_ppm), 6),
                "normalized_ppm": round(config.target_ppm(), 6),
                "source_image_width": int(row.canonical_w),
                "source_image_height": int(row.canonical_h),
                "soil_fraction": round(sf, 5),
                "tile_path": (f"data/tiles/{row.split}/{row.sample_slug}/"
                              f"{row.sample_slug}__{stem}__x{ax}_y{ay}__t{tile}.png"),
                "materialized": tile == config.MATERIALIZED_TILE_SIZE,
            })
    meta = {"tile_size_px": tile, "n_tiles": len(out),
            "edge_dropped_px_w": int(max(0, dropped_w)),
            "edge_dropped_px_h": int(max(0, dropped_h)),
            "edge_dropped_mm_w": round(max(0, dropped_w) / config.target_ppm(), 3),
            "edge_dropped_mm_h": round(max(0, dropped_h) / config.target_ppm(), 3),
            "cropped_w_px": cw, "cropped_h_px": ch}
    return out, meta


def run(force: bool = False) -> int:
    geo = pd.read_csv(config.stage_path("geometry"))
    res = pd.read_csv(config.stage_path("resample"))
    msk = pd.read_csv(config.stage_path("soilmask"))
    df = geo.merge(res, on="image_id").merge(msk, on="image_id")

    all_rows, metas = [], []
    written = 0
    for row in df.itertuples(index=False):
        mpath = os.path.join(config.REPO_ROOT, row.mask_path)
        mask = np.asarray(Image.open(mpath).convert("L"), dtype=np.float32) / 255.0 \
            if os.path.exists(mpath) else None
        for tile in config.TILE_SIZES:
            rows, meta = rows_for_image(row, tile, mask)
            metas.append({**meta, "image_id": row.image_id, "camera": row.camera,
                          "sample_id": row.sample_id})
            all_rows.extend(rows)
        if len(all_rows) % 5000 < 100:
            print(f"  {df.image_id.tolist().index(row.image_id) + 1}/{len(df)} images")

    tiles = pd.DataFrame(all_rows)
    metas = pd.DataFrame(metas)
    tiles.to_csv(os.path.join(config.PROCESSED_DIR, "manifest_tiles.csv"), index=False)
    metas.to_csv(config.stage_path("tiling"), index=False)

    # materialise exactly one size
    mat = tiles[tiles.materialized]
    root = os.path.join(config.DATA_DIR, "tiles")
    by_img: dict[str, list] = {}
    for t in mat.itertuples(index=False):
        by_img.setdefault(t.parent_image_path, []).append(t)
    for i, (ppath, items) in enumerate(sorted(by_img.items())):
        src = os.path.join(config.REPO_ROOT, ppath)
        im = Image.open(src).convert("RGB")
        for t in items:
            dest = os.path.join(config.REPO_ROOT, t.tile_path)
            config.assert_writable(dest)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            im.crop((t.tile_x, t.tile_y, t.tile_x + t.tile_size_px,
                     t.tile_y + t.tile_size_px)).save(
                dest, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL)
            written += 1
        if (i + 1) % 40 == 0:
            print(f"  materialised {i + 1}/{len(by_img)} images")

    # Prune materialised tiles left over from a previous crop/tile policy. Without
    # this, changing TILE_SIZES or the mask cap silently leaves stale tiles on disk
    # whose geometry is not in the manifest, and anything globbing the directory
    # picks them up. Only files under the tile root that the manifest does not name
    # are removed; nothing outside it is touched.
    tile_root = os.path.join(config.DATA_DIR, "tiles")
    intended = set(tiles[tiles.materialized].tile_path)
    pruned = 0
    for dirpath, _dirs, files in os.walk(tile_root):
        for fn in files:
            rel = os.path.relpath(os.path.join(dirpath, fn),
                                  config.REPO_ROOT).replace(os.sep, "/")
            if rel not in intended:
                config.assert_writable(os.path.join(config.REPO_ROOT, rel))
                os.remove(os.path.join(config.REPO_ROOT, rel))
                pruned += 1

    print(f"\ntiles manifest rows: {len(tiles)}")
    print(f"stale tiles pruned: {pruned}")
    print(tiles.groupby("tile_size_px").size().rename("n_tiles").to_string())
    print(f"\nmaterialised files written: {written} "
          f"(size {config.MATERIALIZED_TILE_SIZE}px)")
    tmm = config.TILE_SIZES[0] / config.target_ppm()
    for s in config.TILE_SIZES:
        print(f"  {s}px tile = {s / config.target_ppm():.2f} mm square")
    print(f"\nsmallest tile count comes from the cropped region; "
          f"images yielding 0 tiles: "
          f"{int((metas.groupby('image_id').n_tiles.sum() == 0).sum())}")
    print(f"edge dropped (mm, width) max: {metas.edge_dropped_mm_w.max():.2f}")
    print(f"tiles with soil_fraction < 0.05 (pure tray, expected 0): "
          f"{int((tiles.soil_fraction < 0.05).sum())}")
    print(f"tiles with null sample_id (expected 0): {int(tiles.sample_id.isna().sum())}")
    return 0
