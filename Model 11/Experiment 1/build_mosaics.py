"""build_mosaics.py - Model 11 / Experiment 1, Task 3 step S1: build the VLM input mosaics.

WHAT THIS DOES
  For every one of the 162 images (train AND test, treated identically), assemble a 3x2 mosaic of
  up to 6 of its qualifying 256 px tiles, exactly as plan.md T3 specifies: `soil_fraction >= 0.50`,
  sorted by tile_path, indices `round(linspace(0, n-1, 6))`. Tiles are pasted at NATIVE 256 px --
  no resizing, no cropping, no colour transform -- so nothing here alters the texture statistics
  the VLM is being asked to read. Each square is therefore exactly 56.23 mm of real soil, which is
  what the frozen prompt tells the model.

THE SPEC GAP THIS MODULE HAS TO CLOSE (flagged, not buried)
  plan.md says "up to 6 tiles" but gives one index rule. Measured on the real corpus: 3 of 162
  images have FEWER than 6 qualifying tiles (two have 4, one has 5) and NO image has exactly 6.
  For n < 6 the rule `round(linspace(0, n-1, 6))` emits duplicate indices, which would silently
  repeat a tile without recording that it did.
  RESOLVED AS: use all n distinct tiles, then fill the remaining grid cells by cycling through
  those same tiles in sorted order, and record the count in `n_cells_filled_by_repeat` in the
  manifest for every mosaic.
  WHY REPEAT RATHER THAN PAD: a grey or black pad cell is a fake material in an image whose whole
  purpose is to be judged for grain content, and it would read as "no grains here". Repeating a
  real tile keeps every pixel genuine soil. A variable grid (2x2 for n=4) was rejected because the
  frozen prompt asserts "each square is exactly 56 mm wide", which only holds if all 162 mosaics
  share one geometry.

BLINDNESS
  The neutral id is sha256(SALT + parent_image_path)[:12]: random-looking, reproducible, and
  revealing nothing about soil, site, camera or split without the mapping file. The manifest
  contains NO sample_id, camera, split, ppm or path information. `LOCAL_ONLY_mosaic_map.csv` is
  the only place the identity appears and it must NOT be uploaded to Kaggle.

DETERMINISM
  No randomness. Fixed salt, fixed grid order, fixed PNG compression. Two cold runs must produce
  identical mosaic bytes and an identical manifest.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]          # Model 11/Experiment 1 -> Model 11 -> repo root
OUT = EXP / "mosaics"
MANIFEST = EXP / "mosaic_manifest.csv"
MAPFILE = EXP / "LOCAL_ONLY_mosaic_map.csv"

SALT = "m11e1-blind-v1"          # fixed; changing it changes every id
TILE_PX = 256
GRID_COLS, GRID_ROWS = 3, 2
N_CELLS = GRID_COLS * GRID_ROWS   # 6
MIN_SOIL_FRACTION = 0.50
PNG_COMPRESS_LEVEL = 6


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def neutral_id(parent_image_path: str) -> str:
    return hashlib.sha256((SALT + "|" + parent_image_path).encode("utf-8")).hexdigest()[:12]


def pick_indices(n: int) -> tuple:
    """The plan's rule for n >= 6; the declared repeat-pad rule for n < 6."""
    if n >= N_CELLS:
        idx = np.rint(np.linspace(0, n - 1, N_CELLS)).astype(int)
        assert len(set(idx.tolist())) == N_CELLS, "duplicate tile index at n=%d" % n
        return idx.tolist(), 0
    idx = list(range(n))
    repeats = N_CELLS - n
    fill = [idx[i % n] for i in range(repeats)]
    return idx + fill, repeats


def build() -> int:
    mt = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_tiles.csv")
    q = mt[(mt.tile_size_px == TILE_PX) & (mt.materialized == True)
           & (mt.soil_fraction >= MIN_SOIL_FRACTION)].copy()
    print("=" * 78)
    print("MODEL 11 / EXPERIMENT 1  Task 3 S1 -- build the blind VLM mosaics")
    print("=" * 78)
    print("  qualifying tiles: %d (soil_fraction >= %.2f, 256 px, materialised)"
          % (len(q), MIN_SOIL_FRACTION))

    missing = [p for p in q.tile_path.unique() if not (ROOT / p).exists()]
    if missing:
        print("  *** %d tile files absent on disk, e.g. %r" % (len(missing), missing[:3]))
        return 1
    print("  all tile files present on disk: yes")

    if OUT.exists():
        for f in OUT.iterdir():
            if f.suffix == ".png":
                f.unlink()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, maprows = [], []
    for parent, g in q.groupby("parent_image_path"):
        g = g.sort_values("tile_path")            # the plan's ordering rule
        paths = g.tile_path.tolist()
        n = len(paths)
        idx, repeats = pick_indices(n)
        mid = neutral_id(str(parent))

        canvas = Image.new("RGB", (GRID_COLS * TILE_PX, GRID_ROWS * TILE_PX))
        for cell, i in enumerate(idx):
            t = Image.open(ROOT / paths[i]).convert("RGB")
            if t.size != (TILE_PX, TILE_PX):
                raise AssertionError("tile %s is %s, expected (%d,%d)"
                                     % (paths[i], t.size, TILE_PX, TILE_PX))
            canvas.paste(t, ((cell % GRID_COLS) * TILE_PX, (cell // GRID_COLS) * TILE_PX))
        dest = OUT / ("%s.png" % mid)
        canvas.save(dest, format="PNG", compress_level=PNG_COMPRESS_LEVEL)

        rows.append({"mosaic_id": mid, "mosaic_path": "mosaics/%s.png" % mid,
                     "sha256_mosaic": sha(dest), "n_tiles_available": n,
                     "n_cells_filled_by_repeat": repeats,
                     "grid": "%dx%d" % (GRID_COLS, GRID_ROWS), "tile_px": TILE_PX})
        maprows.append({"mosaic_id": mid, "parent_image_path": parent,
                        "sample_id": g.sample_id.iloc[0], "split": g.split.iloc[0],
                        "camera": g.camera.iloc[0], "n_tiles_available": n,
                        "n_cells_filled_by_repeat": repeats,
                        "tile_paths": " ; ".join(paths[i] for i in idx)})

    man = pd.DataFrame(rows).sort_values("mosaic_id").reset_index(drop=True)
    man.to_csv(MANIFEST, index=False)
    mp = pd.DataFrame(maprows).sort_values("mosaic_id").reset_index(drop=True)
    mp.to_csv(MAPFILE, index=False)

    total = sum(f.stat().st_size for f in OUT.glob("*.png"))
    print("\n  mosaics written: %d  (expected 162)" % len(man))
    print("  grid: %dx%d of %d px tiles -> %dx%d px per mosaic, each square = 56.23 mm of soil"
          % (GRID_COLS, GRID_ROWS, TILE_PX, GRID_COLS * TILE_PX, GRID_ROWS * TILE_PX))
    print("  unique mosaic sha256: %d (collisions: %d)"
          % (man.sha256_mosaic.nunique(), len(man) - man.sha256_mosaic.nunique()))
    print("  total payload: %.1f MB (%.2f GB)" % (total / 1e6, total / 1e9))
    rep = man[man.n_cells_filled_by_repeat > 0]
    print("\n--- the declared n<6 rule, and how often it fired ---")
    print("  mosaics with repeated cells: %d of %d (%.2f%%)"
          % (len(rep), len(man), 100.0 * len(rep) / len(man)))
    for _, r in rep.iterrows():
        print("    %s  n_tiles_available=%d  repeated_cells=%d"
              % (r.mosaic_id, r.n_tiles_available, r.n_cells_filled_by_repeat))

    print("\n--- blindness audit of the UPLOADED artifacts (manifest + mosaics) ---")
    leaky = [c for c in man.columns if c in ("sample_id", "split", "camera", "ppm", "site")]
    print("  identity columns present in mosaic_manifest.csv: %s"
          % (leaky if leaky else "NONE"))
    print("  any path-like value in the manifest: %s"
          % any("/" in str(v) and "mosaics/" not in str(v) for v in man.values.ravel()))
    bad_meta = []
    for f in sorted(OUT.glob("*.png")):
        im = Image.open(f)
        if im.getexif() or im.info.get("icc_profile"):
            bad_meta.append(f.name)
    print("  mosaics carrying EXIF or an ICC profile: %d" % len(bad_meta))
    print("  mosaic filenames reveal soil identity: %s"
          % "NO (12-hex neutral ids only)")
    print("\n  LOCAL ONLY, DO NOT UPLOAD: %s" % MAPFILE.name)
    print("  (this is the only file linking mosaic_id to soil / camera / split)")
    print("=" * 78)
    ok = (len(man) == 162 and not leaky and not bad_meta
          and man.sha256_mosaic.nunique() == len(man))
    print("S1: %s" % ("PASS" if ok else "PROBLEM -- inspect above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(build())
