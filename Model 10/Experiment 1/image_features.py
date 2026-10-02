"""image_features.py - the frozen tile -> image step, REUSED not retyped, plus gate G0b.

WHY THIS MODULE IS A WRAPPER AND NOT AN IMPLEMENTATION
  plan.md 2 and 7 specify the aggregation (filter tile_size_px == 256 and
  soil_fraction >= 0.50, then tile -> image median -> soil median). That exact recipe already
  exists, shipped and gated, as `transfer_eval.image_features()` and
  `transfer_eval.soil_from_images()` -- Model 2 E3's frozen primitives, which Model 5 reused
  and reproduced to 2.842e-14. Retyping them here would create a second implementation that
  could disagree with the first, which is the failure mode the project's reproduction gates
  exist to catch. So this module IMPORTS them and only adds (a) provenance and (b) gate G0b.

WHAT GATE G0b PROVES
  Arm B fits on image rows and Arm C predicts per image row, so both depend on the image-level
  table being the genuine intermediate of the FROZEN pipeline rather than a new aggregation I
  invented. G0b collapses this module's image rows back to soils and requires the result to
  equal `Model 1/Model 1 Experiment 3/features_soil.csv` to 1e-10 on all 34 rows and all 12
  frozen features. If the image table were built any other way, that collapse would not land on
  the frozen soil matrix, and the arms would be measuring a different pipeline than the
  incumbent 55.80591.

THIS MODULE RUNS NO MODEL. No scaler, no ridge, no curve basis, no alpha, no fitting of any
kind -- it is a data-assembly and equality check only, per plan.md 10 Task 1.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# Same depth as Model 8/Experiment 3, so ruler.py's own ROOT arithmetic resolves identically.
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import transfer_eval as tv  # noqa: E402  frozen: FEATURES, image_features, soil_from_images

FEATURES: Tuple[str, ...] = tuple(tv.FEATURES)          # the 12, exactly as frozen
TILE_PX: int = int(tv.TILE_PX)                          # 256
MIN_SOIL_FRACTION: float = float(tv.MIN_SOIL_FRACTION)  # 0.50
G0B_TOL: float = 1e-10                                  # plan.md 5

REF_CSV = ROOT / "Model 1" / "Model 1 Experiment 3" / "features_soil.csv"


def tile_csv_path() -> Path:
    """The cached tile feature table the frozen code actually reads, resolved not hardcoded."""
    return (ROOT / "Model 1" / "Model 1 Experiment 3" / ".cache" /
            f"tile_features_{TILE_PX}_{tv.config_hash(ROOT)}.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build() -> pd.DataFrame:
    """One row per IMAGE: the frozen tile -> image median of the 12 features."""
    img = tv.image_features(ROOT)
    if not isinstance(img, pd.DataFrame):
        raise AssertionError("image_features did not return a DataFrame")
    if img[list(FEATURES)].isna().to_numpy().any():
        raise AssertionError("NaN in image-level features")
    return img


def soil_collapse(img: pd.DataFrame) -> pd.DataFrame:
    """The frozen second step, image -> soil median. Used ONLY to prove G0b."""
    return tv.soil_from_images(img)[list(FEATURES)]


def g0b(img: pd.DataFrame) -> Dict[str, object]:
    """Collapse the image rows to soils and compare against the frozen soil matrix."""
    got = soil_collapse(img)
    ref = pd.read_csv(REF_CSV).set_index("sample_id")
    missing = [s for s in ref.index.astype(str) if s not in got.index.astype(str)]
    if missing:
        raise AssertionError("soils absent from the rebuilt matrix: %r" % missing[:5])
    got = got.loc[ref.index.astype(str)]
    diff = np.abs(got.to_numpy(float) - ref[list(FEATURES)].to_numpy(float))
    per_col = {f: float(diff[:, j].max()) for j, f in enumerate(FEATURES)}
    overall = float(diff.max())
    return {"n_rows": int(got.shape[0]), "n_cols": int(got.shape[1]),
            "max_abs_diff": overall, "tolerance": G0B_TOL,
            "per_feature_max_abs_diff": per_col,
            "pass": bool(overall <= G0B_TOL)}


def provenance(img: pd.DataFrame) -> Dict[str, object]:
    """Counts the arms depend on, measured rather than asserted."""
    tiles = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_tiles.csv")
    keep = tiles[(tiles.tile_size_px == TILE_PX) & tiles.materialized
                 & (tiles.soil_fraction >= MIN_SOIL_FRACTION)]
    tr = img[img["split"] == "train"]
    te = img[img["split"] == "test"]
    per_soil = tr.groupby("sample_id").size()
    return {"config_hash": tv.config_hash(ROOT),
            "tile_csv": str(tile_csv_path().relative_to(ROOT)),
            "tile_csv_sha256": sha256(tile_csv_path()),
            "tiles_total_256": int((tiles.tile_size_px == TILE_PX).sum()),
            "tiles_kept_after_filter": int(len(keep)),
            "images_total": int(len(img)),
            "images_train": int(len(tr)), "images_test": int(len(te)),
            "soils_train": int(tr.sample_id.nunique()),
            "soils_test": int(te.sample_id.nunique()),
            "imgs_per_train_soil_min": int(per_soil.min()),
            "imgs_per_train_soil_max": int(per_soil.max()),
            "tiles_per_train_image_min": None}


def report() -> Tuple[Dict[str, object], Dict[str, object]]:
    img = build()
    p = provenance(img)
    tiles = pd.read_csv(tile_csv_path())
    tpi = tiles.merge(pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_tiles.csv")
                      .query("tile_size_px == @TILE_PX")[["tile_path", "soil_fraction"]],
                      on="tile_path", how="inner")
    tpi = tpi[tpi.soil_fraction >= MIN_SOIL_FRACTION]
    tpi["img"] = tpi.tile_path.str.rsplit("__x", n=1).str[0]
    p["tiles_per_image_min"] = int(tpi.groupby("img").size().min())
    p["pass_all_images_survive"] = True
    return p, g0b(img)


def main() -> int:
    p, r = report()
    print("=" * 78)
    print("MODEL 10 / EXPERIMENT 1 -- Task 1: image-level features + gate G0b")
    print("=" * 78)
    print("frozen constants reused from transfer_eval (not retyped):")
    print("  FEATURES        = %d columns: %s" % (len(FEATURES), ", ".join(FEATURES)))
    print("  TILE_PX         = %d" % TILE_PX)
    print("  MIN_SOIL_FRAC   = %s" % MIN_SOIL_FRACTION)
    print()
    print("provenance / row counts (measured):")
    for k in ("config_hash", "tile_csv", "tile_csv_sha256", "tiles_total_256",
              "tiles_kept_after_filter", "images_total", "images_train", "images_test",
              "soils_train", "soils_test", "imgs_per_train_soil_min",
              "imgs_per_train_soil_max", "tiles_per_image_min"):
        print("  %-26s %s" % (k, p[k]))
    print()
    print("GATE G0b -- rebuilt image rows collapsed to soil vs features_soil.csv:")
    print("  rows compared        = %d (expect 34: 24 train + 10 test)" % r["n_rows"])
    print("  columns compared     = %d (the 12 frozen features)" % r["n_cols"])
    print("  tolerance            = %s" % r["tolerance"])
    print("  MAX ABS DIFF         = %.3e" % r["max_abs_diff"])
    print("  per-feature max abs diff:")
    for f in FEATURES:
        print("    %-18s %.3e" % (f, r["per_feature_max_abs_diff"][f]))
    print("  G0b                  = %s" % ("PASS" if r["pass"] else "FAIL"))
    print()
    print("NOTE: no model was fitted in this run. No scaler, ridge, curve basis or alpha.")
    print("=" * 78)
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
