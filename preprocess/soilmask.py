"""STEP 5 - detect the soil region and choose the largest safe crop rectangle.

Why a rectangle and not a pixel mask, which is the direct answer to "how do we
avoid deleting dark soil grains": deleting a *pixel* only requires that pixel to
be dark, whereas deleting a *row* requires the entire row to be non-soil. An
isolated dark grain, a shadow, or a gap between clasts therefore cannot be removed
by construction. The mask is used to choose a rectangle, never to erase pixels
from the delivered image.

The threshold is applied to the ratio of each pixel to its own large-scale
neighbourhood rather than to absolute brightness. That is what lets a genuinely
dark grain survive: it is dark relative to the frame but not relative to its
surroundings, while the tray rim is dark in both senses.

Measured border profiles that make a fixed-percentage crop wrong for three of five
cameras simultaneously (illumination-relative dark fraction, outer band -> interior):
    Motorola Edge      0.48 -> 0.34   (no rim; the tray fills the frame)
    Edge 60 Fusion     0.37 -> 0.35   (no rim)
    Samsung A52        0.82 -> 0.32   (strong rim)
    iPhone 14 / 16     0.68 / 0.79 -> 0.33 / 0.32   (strong rim)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

from . import config

Image.MAX_IMAGE_PIXELS = None


def otsu_from_hist(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold computed from a histogram (no scikit-image available)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    counts, edges = np.histogram(finite, bins=bins)
    counts = counts.astype(np.float64)
    total = counts.sum()
    if total == 0:
        return float(np.median(finite))
    p = counts / total
    mid = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * mid)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    k = int(np.nanargmax(sigma_b))
    return float(mid[k])


def disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def soil_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (boolean soil mask, illumination-normalised diagnostic image)."""
    h, w = gray.shape
    sigma = max(2.0, 0.08 * min(h, w))
    surround = ndi.gaussian_filter(gray, sigma)
    flat = gray / (surround + 1e-6)
    thr = otsu_from_hist(flat)
    mask = flat > thr
    el = disk(max(2, int(0.004 * min(h, w))))
    mask = ndi.binary_closing(mask, el)
    mask = ndi.binary_opening(mask, el)
    mask = ndi.binary_fill_holes(mask)
    lab, n = ndi.label(mask)
    if n > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = np.flatnonzero(sizes >= config.MIN_COMPONENT_FRACTION * mask.size)
        keep = keep[keep > 0]
        if keep.size:
            biggest = keep[int(np.argmax(sizes[keep]))]
            mask = lab == biggest
            mask = ndi.binary_fill_holes(mask)
    return mask, flat


def _smoothed_density(cov: np.ndarray) -> np.ndarray:
    k = max(3, int(config.COVERAGE_SMOOTH_FRAC * len(cov)) | 1)
    return ndi.uniform_filter1d(cov.astype(np.float64), k, mode="nearest")


def _longest_run(flags: np.ndarray) -> tuple[int, int]:
    best_s, best_e, cur_s = 0, 0, None
    for i, f in enumerate(flags):
        if f and cur_s is None:
            cur_s = i
        elif not f and cur_s is not None:
            if i - cur_s > best_e - best_s:
                best_s, best_e = cur_s, i
            cur_s = None
    if cur_s is not None and len(flags) - cur_s > best_e - best_s:
        best_s, best_e = cur_s, len(flags)
    return int(best_s), int(best_e)


def _axis_rect(cov: np.ndarray) -> tuple[int, int]:
    """Longest span whose local soil density is near this image's own interior level."""
    dens = _smoothed_density(cov)
    n = len(dens)
    centre = dens[n // 4: max(n // 4 + 1, 3 * n // 4)]
    ref = float(np.median(centre)) if centre.size else float(np.median(dens))
    if ref <= 1e-6:
        return 0, n
    return _longest_run(dens >= config.ROW_REL_COVERAGE * ref)


def inscribed_rect(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned rectangle of near-uniform local soil density.

    Works on smoothed row/column *densities* rather than on per-pixel coverage.
    Returning a rectangle rather than a pixel mask is the safety property: a single
    dark grain can dent a row's density but cannot delete the row.
    """
    x0, x1 = _axis_rect(mask.mean(axis=0))
    y0, y1 = _axis_rect(mask.mean(axis=1))
    return int(x0), int(y0), int(x1), int(y1)


def _slug_path(base: str, row) -> str:
    return os.path.join(base, row.split, row.sample_slug,
                        os.path.splitext(row.filename)[0] + "_mask.png")


def run(force: bool = False) -> int:
    geo = pd.read_csv(config.stage_path("geometry"))
    res = pd.read_csv(config.stage_path("resample"))
    df = geo.merge(res, on="image_id", validate="one_to_one")
    out = config.stage_path("soilmask")
    if os.path.exists(out) and not force:
        print("output exists, using it (--force to redo)")
        return 0

    recs = []
    for row in df.itertuples(index=False):
        cpath = os.path.join(config.REPO_ROOT, row.processed_path)
        im = Image.open(cpath).convert("RGB")
        arr = np.asarray(im, dtype=np.float32)
        gray = arr.mean(axis=2)
        H, W = gray.shape

        mask, flat = soil_mask(gray)
        x0, y0, x1, y1 = inscribed_rect(mask)
        if x1 - x0 < 16 or y1 - y0 < 16:
            x0, y0, x1, y1 = 0, 0, W, H

        # cap how much a mask decision is allowed to remove
        max_w = int(round(W * (1 - config.MAX_CROP_FRACTION)))
        max_h = int(round(H * (1 - config.MAX_CROP_FRACTION)))
        status = "ok"
        if (x1 - x0) < max_w or (y1 - y0) < max_h:
            # the rect is more than MAX_CROP_FRACTION away from the frame: trust the
            # cap rather than let a bad mask silently shrink the corpus.
            cx0, cy0 = x0, y0
            x0 = max(0, min(cx0, W - max_w))
            y0 = max(0, min(cy0, H - max_h))
            x1 = min(W, x0 + max_w)
            y1 = min(H, y0 + max_h)
            status = "capped"

        crop = mask[y0:y1, x0:x1]
        frac = float(crop.mean()) if crop.size else 0.0
        if frac < config.MIN_CROP_SOIL_FRACTION:
            x0, y0, x1, y1 = 0, 0, W, H
            frac = float(mask.mean())
            status = "fallback_fullframe"

        mpath = _slug_path(config.MASK_DIR, row)
        config.assert_writable(mpath)
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        Image.fromarray((mask * 255).astype(np.uint8)).save(
            mpath, format="PNG", compress_level=config.PNG_COMPRESS_LEVEL)

        recs.append({
            "image_id": row.image_id,
            "mask_path": os.path.relpath(mpath, config.REPO_ROOT).replace(os.sep, "/"),
            "crop_rect_canonical": f"{x0},{y0},{x1},{y1}",
            "crop_area_fraction": round((x1 - x0) * (y1 - y0) / (W * H), 6),
            "soil_fraction": round(frac, 6),
            "frame_soil_fraction": round(float(mask.mean()), 6),
            "mask_status": status,
        })
        if len(recs) % 40 == 0:
            print(f"  {len(recs)}/{len(df)}")

    m = pd.DataFrame(recs)
    m.to_csv(out, index=False)
    merged = df.merge(m, on="image_id")

    print(f"\nmask_status: {m.mask_status.value_counts().to_dict()}")
    print("\ncrop_area_fraction by camera -- the measured rim profiles predict Samsung and")
    print("iPhone crop meaningfully while Motorola and H374 barely move. If the reverse")
    print("happens, the mask is broken:")
    print(merged.groupby("camera")[["crop_area_fraction", "soil_fraction",
                                    "frame_soil_fraction"]].mean()
          .to_string(float_format=lambda v: f"{v:8.4f}"))
    print(f"\nmin soil_fraction inside a crop: {m.soil_fraction.min():.4f}")
    print(f"images kept at full frame: {int((m.mask_status == 'fallback_fullframe').sum())}")
    return 0
