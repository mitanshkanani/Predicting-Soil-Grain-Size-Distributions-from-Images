"""STEP 4 - resample every image onto the canonical physical-scale grid.

The goal is NOT equal pixel dimensions; it is equal physical scale, so that 1 mm
of soil subtends the same number of pixels in all 162 images. Canvas sizes are
computed from each image's measured field of view, which is why the field of view
itself is preserved per image and deliberately not equalised across images (it
genuinely differs, 289 mm to 351 mm, and equalising it would mean cropping or
padding real data).

Canonical frame is "long side horizontal", applied uniformly, rather than
honouring EXIF. Reason: 32 of the 38 EXIF-bearing files carry orientation 6, so
honouring EXIF would make the 35 test images portrait while the 124 EXIF-less
training images are a 44/25 landscape/portrait mix with no recoverable up
direction. A geometry-defined rule is deterministic and camera-independent, and
90 degree rotations preserve the pixel multiset exactly, so nothing is lost.
`gravity_known` and `rotation_applied_deg` are recorded so a gravity-aligned
variant is rebuildable from metadata alone.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from . import colour, config

Image.MAX_IMAGE_PIXELS = None

RESAMPLE = Image.Resampling.LANCZOS
# Pillow rescales the Lanczos kernel support by the ratio, so this is genuinely
# band-limited at our non-integer factors (0.9994, 0.3652, 0.3265, 0.2332). BOX is
# only equivalent at exact integer factors; NEAREST would alias and is the failure
# mode verify.py's stop-band check exists to catch.


def load_source(path: str) -> tuple[Image.Image, bytes | None]:
    """Open, apply EXIF orientation, and hand back the embedded ICC profile."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
    # exif_transpose copies info wholesale; strip the metadata that no longer
    # describes these pixels. Propagating it would be an outright lie downstream.
    for k in ("exif", "dpi", "jfif", "jfif_unit", "jfif_version",
              "adobe", "adobe_transform"):
        im.info.pop(k, None)
    return im, icc


def canonicalise(im: Image.Image) -> tuple[Image.Image, int]:
    """Force long side horizontal. Returns (image, extra rotation applied)."""
    if im.height > im.width:
        return im.transpose(Image.Transpose.ROTATE_270), 90
    return im, 0


def to_canvas(im: Image.Image, long_px: int, short_px: int) -> Image.Image:
    if (im.width, im.height) == (long_px, short_px):
        return im
    return im.resize((long_px, short_px), RESAMPLE)


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_one(row, dry: bool = False) -> dict:
    src = os.path.join(config.REPO_ROOT, row.source_path)
    im, icc = load_source(src)
    arr_before = np.asarray(im, dtype=np.float32)
    luma_before = float(arr_before.mean())

    im, converted = colour.to_srgb(im, icc)
    im, extra = canonicalise(im)

    want = (int(row.canvas_long_px), int(row.canvas_short_px))
    if im.size != want:
        # Enlarging is forbidden: it would invent detail the sensor never recorded.
        # factor > 1 means the canvas is bigger than the source, i.e. an upscale.
        factor = max(want[0] / im.width, want[1] / im.height)
        if factor > 1.0 + 1e-9:
            raise RuntimeError(
                f"refusing to upscale {row.source_path} by {factor:.4f}x "
                f"({im.width}x{im.height}->{want[0]}x{want[1]}); "
                f"TARGET_PPM derivation is broken")
    out = to_canvas(im, *want)

    arr = np.asarray(out, dtype=np.float32)
    clipped_after = float(((arr <= 0.5) | (arr >= 254.5)).mean())
    # Measure clipping *introduced*, not clipping *present*. The two are very
    # different claims: the Samsung A52's own tone curve blows ~1.6% of pixels in
    # the source, which is a property of the soil photo and not this pipeline's
    # doing. What matters is whether the resample created new saturated pixels,
    # which Lanczos can do via ringing overshoot at high-contrast edges.
    arr_pre = np.asarray(im, dtype=np.float32)
    clipped_before = float(((arr_pre <= 0.5) | (arr_pre >= 254.5)).mean())
    rec = {
        "image_id": row.image_id,
        "processed_path": os.path.relpath(
            colour.canonical_path(row._asdict()), config.REPO_ROOT).replace(os.sep, "/"),
        "canonical_w": out.width,
        "canonical_h": out.height,
        "rotation_applied_deg": int(extra),
        "exif_transpose_applied": bool(row.exif_transpose_applied),
        "icc_converted": bool(converted),
        "clipped_pixel_frac": clipped_after,
        "clipped_pixel_frac_source": round(clipped_before, 6),
        "clipping_introduced_frac": round(max(0.0, clipped_after - clipped_before), 6),
        "mean_luma_before": round(luma_before, 4),
        "mean_luma_after": round(float(arr.mean()), 4),
        "sd_luma_after": round(float(arr.std()), 4),
        "actual_ppm_w": out.width / float(row.fov_long_mm),
        "actual_ppm_h": out.height / float(row.fov_short_mm),
    }
    if not dry:
        dest = colour.canonical_path(row._asdict())
        config.assert_writable(dest)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # PNG, not JPEG: the file on disk should be exactly the array verify.py
        # measured. No EXIF/ICC propagated -- geometry is baked.
        out.save(dest, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL)
        rec["out_sha256"] = _sha(dest)
    return rec


def run(force: bool = False) -> int:
    geo = pd.read_csv(config.stage_path("geometry"))
    out = config.stage_path("resample")
    if os.path.exists(out) and not force:
        print("output exists, using it (--force to redo)")
        return 0

    recs = []
    for i, row in enumerate(geo.itertuples(index=False)):
        recs.append(process_one(row))
        if (i + 1) % 40 == 0 or i == len(geo) - 1:
            print(f"  {i + 1}/{len(geo)}")
    df = pd.DataFrame(recs)
    df.to_csv(out, index=False)

    print(f"images written: {len(df)}")
    print(f"icc conversions applied: {int(df.icc_converted.sum())} (expected 38)")
    print(f"rotated to canonical frame: "
          f"{int((df.rotation_applied_deg > 0).sum())}")
    print(f"max clipping present in output: {df.clipped_pixel_frac.max():.6f}")
    print(f"max clipping INTRODUCED by resample: {df.clipping_introduced_frac.max():.8f} "
          f"(tolerance 1e-3)")
    print(f"mean source clipping by camera (a property of the photo, not this pipeline):")
    print(geo.merge(df, on="image_id").groupby("camera").clipped_pixel_frac_source
          .mean().to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"mean |luma change| from colour+geometry: "
          f"{(df.mean_luma_after - df.mean_luma_before).abs().mean():.3f} /255")
    total_px = int((df.canonical_w * df.canonical_h).sum())
    print(f"canonical canvas total: {total_px / 1e6:.1f} Mpx")
    return 0
