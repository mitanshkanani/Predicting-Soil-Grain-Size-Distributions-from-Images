"""STEP 4/6 - colour handling.

Two operations with very different epistemic status, deliberately treated
differently:

* Display P3 -> sRGB is a **correctness fix**. 38 of the 162 files (all 35 test
  images and all 3 H374 images) carry an embedded Display P3 profile; the other
  124 carry no profile and decode as sRGB. Leaving that unconverted means the
  train/test split is partly a colourspace split. It uses zero fitted parameters
  and makes no assumption about the soil population, so it is baked in.

* Gray-world white balance is a **choice**. Its gains depend on the soil mask,
  and the mask policy is expected to change during this phase. So the gains are
  computed and stored as three columns and deliberately NOT applied -- baking
  them now would silently bake a mask decision too.

The conversion must be applied *before* the resample, not after: the P3->sRGB
transform is a nonlinear per-channel tone curve followed by a matrix, so it does
not commute with a linear anti-alias filter. Converting afterwards would leave
the 38 profiled files with a measurably different effective blur kernel than the
other 124, manufacturing exactly the camera artefact this phase exists to
isolate.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageCms

from . import config

# Relative colorimetric: maps in-gamut colours exactly and clips the small number
# of P3 primaries outside sRGB. Preferred over perceptual here because a
# compression curve would redistribute contrast in a way that varies per image and
# silently biases texture statistics. Soil colours are near-neutral, so almost
# nothing is actually clipped -- clipped_pixel_frac is asserted downstream.
INTENT = ImageCms.Intent.RELATIVE_COLORIMETRIC

_srgb = ImageCms.createProfile("sRGB")
_transform_cache: dict[bytes, ImageCms.ImageCmsTransform] = {}


def to_srgb(im: Image.Image, icc: bytes | None) -> tuple[Image.Image, bool]:
    """Convert a profiled image to sRGB. Returns (image, converted)."""
    if not icc:
        return im, False
    key = icc
    t = _transform_cache.get(key)
    if t is None:
        src = ImageCms.getOpenProfile(io.BytesIO(icc))
        t = ImageCms.buildTransform(src, _srgb, "RGB", "RGB", renderingIntent=INTENT)
        _transform_cache[key] = t
    return ImageCms.applyTransform(im, t, inPlace=False), True


def gray_world_gains(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float, float]:
    """Channel gains that equalise mean channel response over the given region.

    Computed on the soil region rather than the whole frame: border content is
    camera-dependent, so frame-level gains would differ between cameras for
    reasons that have nothing to do with white balance.
    """
    px = arr[mask] if mask is not None else arr.reshape(-1, arr.shape[-1])
    if px.shape[0] < 64:
        return 1.0, 1.0, 1.0
    means = px.mean(axis=0)
    ref = float(means.mean())
    gains = [ref / float(m) if m > 1e-6 else 1.0 for m in means]
    # normalise so the geometric mean gain is 1: this is a chroma/white-balance
    # correction, not a brightness correction, and must not silently rescale luma.
    prod = float(np.prod(gains))
    if prod > 0:
        norm = prod ** (1.0 / 3.0)
        gains = [g / norm for g in gains]
    return tuple(round(g, 6) for g in gains)


def apply_gains(arr: np.ndarray, gains) -> np.ndarray:
    out = arr.astype(np.float32) * np.asarray(gains, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def canonical_path(row) -> str:
    return os.path.join(config.OUT_DIRS[row["split"]], row["sample_slug"],
                        os.path.splitext(row["filename"])[0] + ".png")


def run(force: bool = False) -> int:
    """Compute and store gray-world gains. Applies nothing."""
    geo = pd.read_csv(config.stage_path("geometry"))
    out = config.stage_path("colour")
    if os.path.exists(out) and not force:
        print("output exists, using it (--force to redo)")
        return 0

    recs = []
    for row in geo.itertuples(index=False):
        d = row._asdict()
        cpath = canonical_path(d)
        gains = (1.0, 1.0, 1.0)
        soil_px = 0
        if os.path.exists(cpath):
            arr = np.asarray(Image.open(cpath).convert("RGB"))
            mpath = os.path.join(config.MASK_DIR, d["split"], d["sample_slug"],
                                 os.path.splitext(d["filename"])[0] + "_mask.png")
            mask = None
            if os.path.exists(mpath):
                mask = np.asarray(Image.open(mpath).convert("L")) > 127
                if mask.shape[:2] != arr.shape[:2]:
                    mask = None
            gains = gray_world_gains(arr, mask)
            soil_px = int(mask.sum()) if mask is not None else int(arr.shape[0] * arr.shape[1])
        recs.append({"image_id": d["image_id"], "gw_gain_r": gains[0],
                     "gw_gain_g": gains[1], "gw_gain_b": gains[2],
                     "gw_soil_pixels": soil_px,
                     "icc_converted": bool(d["icc_present"])})
    df = pd.DataFrame(recs)
    df.to_csv(out, index=False)

    merged = geo.merge(df, on="image_id")
    print(f"rows: {len(df)}")
    print(f"P3->sRGB conversions baked: {int(merged.icc_converted.sum())} "
          f"(expected 38 = 35 test + 3 H374)")
    print("\ngray-world gains by camera (stored, not applied):")
    print(merged.groupby("camera")[["gw_gain_r", "gw_gain_g", "gw_gain_b"]].agg(
        ["mean", "std"]).to_string(float_format=lambda v: f"{v:7.4f}"))
    print("\nREJECTED normalisations, with reasons (recorded in the audit):")
    for line in [
        "per-channel standardisation to fixed mean/SD -- measured destructive: "
        "Spearman(mid-scale texture energy, log D50) falls 0.73 -> 0.38",
        "any colour op aimed at the texture gap -- the same soil differs 63-124% between "
        "the two training cameras at matched scale, unchanged by colour ops; it is MTF/tone, "
        "not chroma",
        "CLAHE / histogram equalisation -- manufactures mid-scale contrast the sensor never "
        "recorded and breaks the monotone contrast<->grain-size link",
    ]:
        print(f"  - {line}")
    return 0
