"""STEP 3 - effective PPM, physical field of view, and the derived canonical scale.

Two things happen here that matter more than they look.

First, axis pairing by sorted side rather than by width/height. 25 of the 69
Motorola Edge files are stored portrait (720x1600) with no EXIF to say so, so a
naive width-over-width comparison reports a 122% mismatch on those files. Pairing
short-with-short and long-with-long is immune to storage rotation.

Second, TARGET_PPM is derived as min(effective_ppm) instead of being typed. The
brief proposed 4.58 px/mm while also forbidding upscaling, but the Samsung A52's
effective PPM is ~4.553, so 4.58 would force a 0.57% upscale on all 55 Samsung
images. Taking the minimum makes every image identity-or-downsample and leaves
0.07% residual cross-camera scale error -- which is what lets later phases blame
any remaining difference on the camera pipeline rather than on geometry.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np
import pandas as pd

from . import config


def _round_half_up_even(x: float) -> int:
    """Round half-up, breaking exact .5 ties toward even.

    Python's round() is banker's rounding, so 0.5 cases would depend on the
    parity of the neighbour rather than on the geometry. Ties go to even, which
    slightly favours a future 2x pooling layer. Values that are already close to
    an integer are *not* forced to even -- doing that would push the Samsung
    canvas from an exact 1599 to 1598 and turn a genuine no-op resample into a
    1-pixel distortion.
    """
    floor = math.floor(x)
    frac = x - floor
    if abs(frac - 0.5) < 1e-9:
        return floor if floor % 2 == 0 else floor + 1
    return floor + 1 if frac > 0.5 else floor


def derive(row) -> dict:
    """Per-image geometry from native/delivered size and native PPM."""
    out: dict = {}
    nw, nh = row.native_width, row.native_height
    dw, dh = row.delivered_w, row.delivered_h
    ppm0 = row.native_ppm

    if not all(isinstance(v, (int, float)) and v for v in (nw, nh, dw, dh, ppm0)):
        out.update(eff_ppm_short=None, eff_ppm_long=None, effective_ppm=None,
                   fov_short_mm=None, fov_long_mm=None,
                   eff_ppm_disagreement_pct=None, aspect_mismatch_pct=None,
                   ppm_flag="unresolved_metadata")
        return out

    native_short, native_long = sorted((float(nw), float(nh)))
    # EXIF orientation 5-8 means the *displayed* image is transposed relative to
    # the stored buffer; sorting sides makes that irrelevant to the ratio.
    deliv_short, deliv_long = sorted((float(dw), float(dh)))

    f_s = deliv_short / native_short
    f_l = deliv_long / native_long
    eff_s = ppm0 * f_s
    eff_l = ppm0 * f_l
    mean = 0.5 * (eff_s + eff_l)
    disagreement = abs(eff_s - eff_l) / mean if mean else float("nan")

    native_aspect = native_long / native_short
    stored_aspect = deliv_long / deliv_short
    aspect_mismatch = abs(native_aspect - stored_aspect) / native_aspect

    flags = []
    if disagreement > config.TOL_AXIS_DISAGREEMENT:
        flags.append("ppm_axis_disagreement")
    if aspect_mismatch > config.TOL_ASPECT_MISMATCH:
        flags.append("aspect_mismatch")

    out.update(
        eff_ppm_short=eff_s,
        eff_ppm_long=eff_l,
        # geometric mean: the scale-preserving average of two axis estimates
        effective_ppm=math.sqrt(eff_s * eff_l),
        downscale_factor_short=f_s,
        downscale_factor_long=f_l,
        eff_ppm_disagreement_pct=disagreement * 100.0,
        stored_aspect=stored_aspect,
        native_aspect=native_aspect,
        aspect_mismatch_pct=aspect_mismatch * 100.0,
        # per-axis FOV from per-axis ppm, so the Samsung's delivered width of 1599
        # (not 1600) cannot leak into a physical extent.
        fov_short_mm=deliv_short / eff_s,
        fov_long_mm=deliv_long / eff_l,
        ppm_flag=";".join(flags) or "ok",
        stored_portrait=bool(dh > dw),
        exif_transpose_applied=bool(row.exif_orientation in (5, 6, 7, 8)),
        rotation_180_applied=bool(row.exif_orientation == 3),
        gravity_known=bool(row.has_exif and row.exif_orientation is not None
                           and pd.notna(row.exif_orientation)),
    )
    return out


def apply_target(df: pd.DataFrame, target: float) -> pd.DataFrame:
    """Canvas dimensions and resample direction for a given canonical scale."""
    short_px = (df.fov_short_mm * target).map(_round_half_up_even)
    long_px = (df.fov_long_mm * target).map(_round_half_up_even)
    df = df.copy()
    df["target_ppm"] = target
    df["canvas_short_px"] = short_px
    df["canvas_long_px"] = long_px
    df["canvas_short_exact_px"] = df.fov_short_mm * target
    df["canvas_long_exact_px"] = df.fov_long_mm * target
    df["actual_ppm_short"] = short_px / df.fov_short_mm
    df["actual_ppm_long"] = long_px / df.fov_long_mm
    df["fov_error_mm_short"] = short_px / target - df.fov_short_mm
    df["fov_error_mm_long"] = long_px / target - df.fov_long_mm
    df["resample_factor_short"] = df.eff_ppm_short / target
    df["resample_factor_long"] = df.eff_ppm_long / target
    # Direction is decided by comparing ACTUAL pixel counts in the canonical frame,
    # not by thresholding the ppm ratios. The ratio form labelled every Samsung
    # image "identity" because its long axis is the one that defines TARGET_PPM and
    # so has a factor of exactly 1.0 -- but the short axis still shrinks 1200 ->
    # 1199, so a real resample happens. Pixel comparison is what the resampler
    # actually branches on, so it cannot disagree with the guarantee below.
    src_long = df[["delivered_w", "delivered_h"]].max(axis=1)
    src_short = df[["delivered_w", "delivered_h"]].min(axis=1)
    grew = (df.canvas_long_px > src_long) | (df.canvas_short_px > src_short)
    unchanged = (df.canvas_long_px == src_long) & (df.canvas_short_px == src_short)
    df["resample_direction"] = np.where(grew, "up", np.where(unchanged, "identity", "down"))
    df["source_long_px"] = src_long
    df["source_short_px"] = src_short
    return df


def restore_target_ppm() -> float | None:
    v = config.load_target_ppm()
    if v is not None:
        config.set_target_ppm(v)
    return v


def run(force: bool = False) -> int:
    cam = pd.read_csv(config.stage_path("cameras"))
    inv = pd.read_csv(config.stage_path("inventory"))
    df = inv.merge(cam, on="image_id", validate="one_to_one")

    geo = pd.DataFrame([derive(r) for r in df.itertuples(index=False)], index=df.index)
    df = pd.concat([df, geo], axis=1)

    # --- misfiling check: the folder's sample_id must appear in the filename ----
    def skel(s: str) -> str:
        s = s.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
        return re.sub(r"[^0-9a-z]+", "", s)

    # Test folders use the umlaut and a comma ('Münster_BS6_9,0-10m') while the
    # filenames use the same characters, so both sides need identical folding.
    df["sample_id_in_filename"] = [
        skel(sid) in skel(os.path.splitext(fn)[0])
        for sid, fn in zip(df.sample_id, df.filename)
    ]

    unresolved = df[df.ppm_flag == "unresolved_metadata"]
    if len(unresolved):
        print(f"!! {len(unresolved)} images have unresolved native metadata -- cannot proceed")
        print(unresolved[["sample_id", "filename", "camera"]].to_string(index=False))
        return 1

    # --- derive the canonical scale --------------------------------------------
    per_camera = df.groupby("camera").agg(
        n_images=("image_id", "size"),
        native_width=("native_width", "first"),
        native_height=("native_height", "first"),
        native_ppm=("native_ppm", "first"),
        delivered_w=("delivered_w", lambda s: int(s.mode().iloc[0])),
        delivered_h=("delivered_h", lambda s: int(s.mode().iloc[0])),
        eff_ppm_short=("eff_ppm_short", "min"),
        eff_ppm_long=("eff_ppm_long", "min"),
        effective_ppm=("effective_ppm", "min"),
        eff_ppm_max=("effective_ppm", "max"),
        disagreement_pct=("eff_ppm_disagreement_pct", "max"),
        fov_short_mm=("fov_short_mm", "min"),
        fov_long_mm=("fov_long_mm", "min"),
    ).reset_index()
    per_camera["ppm_spread_within_camera"] = per_camera.eff_ppm_max / per_camera.effective_ppm - 1

    target = float(df[["eff_ppm_short", "eff_ppm_long"]].min().min())
    config.set_target_ppm(target)
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    config.assert_writable(os.path.join(config.PROCESSED_DIR, "target_ppm.json"))
    with open(os.path.join(config.PROCESSED_DIR, "target_ppm.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"target_ppm": target,
                   "derivation": "min(effective_ppm) over all cameras and axes",
                   "brief_proposed_ppm": config.TARGET_PPM_CANDIDATE,
                   "would_upscale_cameras_at_proposed": sorted(
                       df[df.effective_ppm < config.TARGET_PPM_CANDIDATE].camera.unique().tolist()),
                   "config_hash": config.config_hash()}, fh, indent=1)

    df = apply_target(df, target)

    # Hard stop here rather than in resample: this stage writes only metadata and
    # costs a fraction of a second, so an impossible canonical scale should be
    # caught before 233 Mpx are resampled the wrong way.
    ups = df[df.resample_direction == "up"]
    if len(ups):
        print(f"!! {len(ups)} images would be UPSCALED to reach {target:.6f} px/mm")
        print(ups[["sample_id", "filename", "camera", "source_short_px",
                   "canvas_short_px", "source_long_px", "canvas_long_px"]]
              .head(10).to_string(index=False))
        return 1
    print(f"zero-upscale guarantee holds: no image's canvas exceeds its source "
          f"pixel count on either axis")

    # --- flags ------------------------------------------------------------------
    flags = df[df.ppm_flag != "ok"][["image_id", "sample_id", "filename", "camera",
                                     "ppm_flag", "eff_ppm_disagreement_pct",
                                     "aspect_mismatch_pct"]]
    flags.to_csv(os.path.join(config.PROCESSED_DIR, "flags.csv"), index=False)
    per_camera.to_csv(os.path.join(config.PROCESSED_DIR, "camera_table.csv"), index=False)
    df.to_csv(config.stage_path("geometry"), index=False)

    # --- report -----------------------------------------------------------------
    print(f"rows: {len(df)}")
    print(f"\nderived TARGET_PPM = {target:.6f} px/mm")
    print(f"brief proposed      = {config.TARGET_PPM_CANDIDATE:.6f} px/mm")
    up_at_proposed = df[df.effective_ppm < config.TARGET_PPM_CANDIDATE]
    print(f"images that 4.58 would force to upscale: {len(up_at_proposed)} "
          f"({sorted(set(up_at_proposed.camera))})")
    print(f"\nresample direction: {df.resample_direction.value_counts().to_dict()} "
          "(expected: no 'up')")
    print(f"ppm flags: {len(flags)} (expected 0 -- this is a regression guard for future "
          f"cameras, not a data filter)")
    print(f"max axis disagreement anywhere: {df.eff_ppm_disagreement_pct.max():.4f}% "
          f"(tolerance {config.TOL_AXIS_DISAGREEMENT * 100:.1f}%)")
    print(f"max FOV error after canvas rounding: "
          f"{df[['fov_error_mm_short', 'fov_error_mm_long']].abs().values.max():.4f} mm "
          f"(tolerance {config.TOL_FOV_MM} mm)")
    print(f"sample_id absent from filename: "
          f"{int((~df.sample_id_in_filename).sum())}")
    print("\nper-camera effective PPM:")
    print(per_camera[["camera", "n_images", "native_width", "native_height", "native_ppm",
                      "delivered_w", "delivered_h", "effective_ppm",
                      "ppm_spread_within_camera", "fov_long_mm", "fov_short_mm"]]
          .to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("\nstored-portrait files (rotation-blind, no EXIF): "
          f"{int(df.stored_portrait.sum())}")
    print("gravity known from EXIF: "
          f"{int(df.gravity_known.sum())}")
    return 0
