"""STEP 9 - the audit report and the fail-loud assertion set.

Exits non-zero on any failed assertion. The reasoning is asymmetric: a silently
garbage 233 Mpx output costs a whole modelling phase before anyone notices, while
a crash costs nothing. Several assertions are also *regression guards* that are
known not to fire on this corpus -- they are kept because they would fire if the
derivation or a future camera broke it, and the audit says so explicitly rather
than letting a never-firing check masquerade as evidence of health.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from PIL import Image

from . import config

Image.MAX_IMAGE_PIXELS = None


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sources_untouched() -> tuple[bool, list[str]]:
    """Prove data/Training and data/Test are byte-identical to the recorded baseline."""
    problems: list[str] = []
    bp = os.path.join(config.PROCESSED_DIR, "baseline_sources.json")
    if not os.path.exists(bp):
        return False, ["baseline_sources.json missing -- inventory stage never ran"]
    base = json.load(open(bp, encoding="utf-8"))
    now = {}
    for split, root in config.SOURCE_DIRS.items():
        for dirpath, _dirs, files in os.walk(root):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in config.IMAGE_EXTS:
                    continue
                fp = os.path.join(dirpath, fn)
                rel = os.path.relpath(fp, config.REPO_ROOT).replace(os.sep, "/")
                now[rel] = _sha_file(fp)
    if set(now) != set(base["files"]):
        added = set(now) - set(base["files"])
        gone = set(base["files"]) - set(now)
        if added:
            problems.append(f"{len(added)} image(s) ADDED to source tree: {sorted(added)[:3]}")
        if gone:
            problems.append(f"{len(gone)} image(s) MISSING from source tree: {sorted(gone)[:3]}")
    changed = [p for p, d in base["files"].items() if p in now and now[p] != d]
    if changed:
        problems.append(f"{len(changed)} source image(s) CHANGED: {changed[:3]}")
    return not problems, problems


def run(force: bool = False) -> int:
    img_path = os.path.join(config.PROCESSED_DIR, "manifest_images.csv")
    if not os.path.exists(img_path):
        print("manifest_images.csv missing -- run manifests stage")
        return 1
    img = pd.read_csv(img_path)
    samp = pd.read_csv(os.path.join(config.PROCESSED_DIR, "manifest_samples.csv"))
    tiles = pd.read_csv(os.path.join(config.PROCESSED_DIR, "manifest_tiles.csv")) \
        if os.path.exists(os.path.join(config.PROCESSED_DIR, "manifest_tiles.csv")) else None

    checks: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    # --- source integrity ----------------------------------------------------
    ok_src, src_problems = check_sources_untouched()
    add("source tree byte-identical to baseline", ok_src,
        "; ".join(src_problems) or "162 files, 0 changed")

    # --- counts --------------------------------------------------------------
    add("image row count == 162", len(img) == 162, f"{len(img)}")
    add("train images == 127", int((img.split == "train").sum()) == 127)
    add("test images == 35", int((img.split == "test").sum()) == 35)
    add("no duplicate image_id", not img.image_id.duplicated().any())
    missing_paths = [p for p in img.source_path
                     if not os.path.exists(os.path.join(config.REPO_ROOT, p))]
    add("every source_path exists", not missing_paths, f"{len(missing_paths)} missing")
    missing_out = [p for p in img.processed_path
                   if not os.path.exists(os.path.join(config.REPO_ROOT, p))]
    add("every processed_path exists", not missing_out, f"{len(missing_out)} missing")

    # --- scale ---------------------------------------------------------------
    add("ppm_flag count == 0 (regression guard, cannot fire on this corpus)",
        int((img.ppm_flag != "ok").sum()) == 0,
        f"{int((img.ppm_flag != 'ok').sum())} flagged")
    # Re-derive the guarantee from raw pixel counts rather than trusting the label,
    # so a bug in how resample_direction is computed cannot hide a real upscale.
    grew = ((img.canvas_long_px > img.source_long_px)
            | (img.canvas_short_px > img.source_short_px))
    add("zero upsamples (canvas <= source on both axes)",
        not bool(grew.any()),
        f"{int(grew.sum())} upscaled; labels "
        f"{img.resample_direction.value_counts().to_dict()}")
    label_says_identity = img.resample_direction == "identity"
    pixels_are_identity = ((img.canvas_long_px == img.source_long_px)
                           & (img.canvas_short_px == img.source_short_px))
    add("resample_direction labels match pixel comparison",
        bool((label_says_identity == pixels_are_identity).all()),
        str(img.resample_direction.value_counts().to_dict()))
    ppm_dev = (img[["actual_ppm_short", "actual_ppm_long"]]
               .apply(lambda c: (c - img.target_ppm).abs() / img.target_ppm)).max(axis=1)
    add("actual_ppm within 0.5% of target", float(ppm_dev.max()) < config.TOL_ACTUAL_PPM,
        f"max deviation {float(ppm_dev.max()) * 100:.4f}%")
    fov_err = img[["fov_error_mm_short", "fov_error_mm_long"]].abs().max(axis=1)
    add("FOV preserved within 0.5mm", float(fov_err.max()) < config.TOL_FOV_MM,
        f"max {float(fov_err.max()):.4f} mm")
    spread = img.groupby("camera").effective_ppm.agg(["min", "max"])
    add("effective_ppm constant within each camera",
        bool(((spread["max"] / spread["min"] - 1) < 1e-9).all()),
        "per-camera spread max "
        f"{float((spread['max'] / spread['min'] - 1).max()):.10f}")

    # --- provenance ----------------------------------------------------------
    add("camera resolved for all images", not img.camera.isna().any())
    add("exif/filename disagreements == 0",
        int((img.exif_agrees_with_filename == False).sum()) == 0)  # noqa: E712
    add("has_exif == 38", int(img.has_exif.sum()) == 38, f"{int(img.has_exif.sum())}")
    add("icc_present == 38", int(img.icc_present.sum()) == 38)
    add("icc conversions == 38", int(img.icc_converted.sum()) == 38)
    add("unknown camera == 0", int(img.camera.isna().sum()) == 0)
    add("unparseable filenames == 0",
        int((img.filename_parse_status == "unparseable").sum()) == 0)

    # --- pixels --------------------------------------------------------------
    if "clipping_introduced_frac" not in img.columns:
        add("resample stage carries clipping-introduced column", False,
            "stage_resample.csv is stale -- re-run the resample stage")
    else:
        add("clipping introduced by resample < 1e-3",
            float(img.clipping_introduced_frac.max()) < 1e-3,
            f"max {float(img.clipping_introduced_frac.max()):.8f}")
    rot = img.rotation_applied_deg % 90
    add("all rotations are multiples of 90deg (lossless)", bool((rot == 0).all()))

    # re-open every written file: catches a silent format/quality regression
    bad_dims = []
    for r in img.itertuples(index=False):
        p = os.path.join(config.REPO_ROOT, r.processed_path)
        if not os.path.exists(p):
            bad_dims.append((r.processed_path, "absent"))
            continue
        with Image.open(p) as im:
            if (im.width, im.height) != (int(r.canonical_w), int(r.canonical_h)):
                bad_dims.append((r.processed_path, f"{im.size} != "
                               f"{(int(r.canonical_w), int(r.canonical_h))}"))
    add("every output re-opens with manifest dims", not bad_dims,
        f"{len(bad_dims)} mismatched")

    # --- masking -------------------------------------------------------------
    add("mask_status legal", set(img.mask_status) <= {"ok", "capped", "fallback_fullframe"},
        str(img.mask_status.value_counts().to_dict()))
    cap_pct = int(round(config.MAX_CROP_FRACTION * 100))
    add(f"crop never exceeds {cap_pct}% per axis",
        bool((img.crop_area_fraction >= 1 - 2 * config.MAX_CROP_FRACTION - 1e-6).all()),
        f"min crop_area_fraction {float(img.crop_area_fraction.min()):.4f} "
        f"(floor {(1 - config.MAX_CROP_FRACTION) ** 2:.4f})")

    # --- tiling / hierarchy --------------------------------------------------
    if tiles is not None:
        add("every tile has a sample_id", not tiles.sample_id.isna().any())
        add("tile sample_ids are a subset of image sample_ids",
            set(tiles.sample_id) <= set(img.sample_id))
        add("tile_w_mm == tile_size_px / target_ppm",
            bool(np.allclose(tiles.tile_w_mm,
                             tiles.tile_size_px / img.target_ppm.iloc[0], atol=1e-6)))
        pure_tray = tiles[tiles.soil_fraction < 0.05]
        frac = len(pure_tray) / max(len(tiles), 1)
        # Pure-tray tiles are a direct consequence of the crop cap, not an
        # unrelated defect: the cap refuses to remove more than cap_pct per axis, so
        # a little rim survives and can form a tile. Report the linkage explicitly so
        # nobody reads a low count as evidence the mask is working.
        if len(pure_tray):
            srcs = set(pure_tray.parent_image_path)
            contrib = img[img.processed_path.isin(srcs)]
            n_capped = int((contrib.mask_status == "capped").sum())
            detail = (f"{len(pure_tray)}/{len(tiles)} ({frac * 100:.2f}%) across "
                      f"{len(srcs)} images, {n_capped} of them cap-bound")
        else:
            detail = "none"
        add("pure-tray tiles below 2% of all tiles", frac < 0.02, detail)
        add("crop cap bindingness reported", True,
            f"{int((img.mask_status == 'capped').sum())}/{len(img)} images hit the "
            f"{int(config.MAX_CROP_FRACTION * 100)}% cap")
        mat = int(tiles.materialized.sum())
        on_disk = len([p for p in tiles[tiles.materialized].tile_path
                       if os.path.exists(os.path.join(config.REPO_ROOT, p))])
        add("materialised tiles present on disk", mat == on_disk, f"{on_disk}/{mat}")

    # --- labels --------------------------------------------------------------
    tr = samp[samp.split == "train"]
    add("24 train samples", len(tr) == 24)
    add("10 test samples", int((samp.split == "test").sum()) == 10)
    add("all label rows monotonic", bool(tr.all_target_cols_monotonic.all()))
    add("all label rows have 200mm == 100", bool(tr.col_200_equals_100.all()))
    sub = pd.read_csv(config.SUBMISSION_CSV)
    want = sorted(sub.sample_id.astype(str))
    got = sorted(samp[samp.split == "test"].submission_id.dropna().astype(str))
    add("test submission_ids reproduce sample_submission.csv exactly",
        got == want, f"{len(set(got) & set(want))}/{len(want)} matched")
    unpaired = sorted(samp[(samp.split == "train") & (~samp.camera_paired)].sample_id)
    add("single-camera train samples are exactly H366/H374/H637",
        unpaired == ["H366", "H374", "H637"], str(unpaired))

    # --- write report --------------------------------------------------------
    passed = sum(1 for _n, o, _d in checks if o)
    failed = [(n, d) for n, o, d in checks if not o]

    cam_tbl = img.groupby("camera").agg(
        n_images=("image_id", "size"),
        native=("native_width", lambda s: f"{int(s.iloc[0])}x{int(img.loc[s.index, 'native_height'].iloc[0])}"),
        native_ppm=("native_ppm", "first"),
        delivered=("delivered_w", lambda s: f"{int(s.iloc[0])}x{int(img.loc[s.index, 'delivered_h'].iloc[0])}"),
        effective_ppm=("effective_ppm", "first"),
        fov_long_mm=("fov_long_mm", "first"),
        fov_short_mm=("fov_short_mm", "first"),
        canvas=("canvas_long_px", lambda s: f"{int(s.iloc[0])}x{int(img.loc[s.index, 'canvas_short_px'].iloc[0])}"),
        actual_ppm=("actual_ppm_long", "mean"),
        crop_area_fraction=("crop_area_fraction", "mean"),
        soil_fraction=("soil_fraction", "mean"),
    ).reset_index()

    lines = ["# Preprocessing audit", ""]
    lines += [f"- pipeline_version `{config.PIPELINE_VERSION}`  config_hash "
              f"`{config.config_hash()}`",
              f"- TARGET_PPM (derived) `{float(img.target_ppm.iloc[0]):.6f}` px/mm",
              f"- checks passed **{passed}/{len(checks)}**", ""]
    if tiles is not None:
        tc = tiles.groupby("tile_size_px").size()
        lines += ["## Tile counts", "", "| tile px | mm square | tiles |", "|---|---|---|"]
        t0 = float(img.target_ppm.iloc[0])
        for size, n in tc.items():
            lines.append(f"| {int(size)} | {int(size) / t0:.2f} | {int(n)} |")
        lines += [""]
    lines += ["## Per-camera geometry", "",
              "```", cam_tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
              "```", "", "## Assertions", "",
              "| check | pass | detail |", "|---|---|---|"]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else '**FAIL**'} | {detail} |")
    lines += ["", "## Notes on guards that cannot fire on this corpus", "",
              "- `ppm_flag` axis-agreement: max measured disagreement is 0.0625% "
              "(Samsung's delivered width is 1599 rather than 1600). Tolerance is 0.5%. "
              "This is a regression guard for a future camera whose native and delivered "
              "aspects differ, not a data filter. Without short/long axis pairing the same "
              "guard fires on all 25 stored-portrait Motorola files with a bogus 122% error.",
              "- `effective_ppm` is exactly constant within each camera, so there is no "
              "per-image scale variation to model: one calibration per camera suffices.",
              "- The camera texture gap (same soil, +63% to +124% texture energy between the "
              "two training cameras, 21/21 same sign) is NOT fixable by any colour "
              "normalisation and is deliberately left in place, recorded, and attributable "
              "via the `camera` column rather than papered over.", ""]

    md = os.path.join(config.PROCESSED_DIR, "audit.md")
    js = os.path.join(config.PROCESSED_DIR, "audit.json")
    config.assert_writable(md)
    config.assert_writable(js)
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    json.dump({"pipeline_version": config.PIPELINE_VERSION,
               "config_hash": config.config_hash(),
               "target_ppm": float(img.target_ppm.iloc[0]),
               "checks": [{"name": n, "pass": o, "detail": d} for n, o, d in checks],
               "failed": [n for n, _ in failed]},
              open(js, "w", encoding="utf-8"), indent=1)
    cam_tbl.to_csv(os.path.join(config.PROCESSED_DIR, "camera_table.csv"), index=False)

    print(f"checks passed: {passed}/{len(checks)}")
    if failed:
        print("\nFAILED:")
        for n, d in failed:
            print(f"  - {n}: {d}")
        return 1
    print("all assertions passed")
    return 0
