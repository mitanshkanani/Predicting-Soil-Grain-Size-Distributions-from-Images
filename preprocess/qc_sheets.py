"""STEP 10 - per-image contact sheets and the paired-camera sheets.

Numbers are necessary but not sufficient here: a pipeline can pass every statistic
and still visibly destroy the thing it is supposed to measure. Two panels exist
specifically to be looked at rather than asserted on:

* the illumination-normalised diagnostic, which is where you can actually see
  whether dark soil grains survived the mask; and
* the colour-conversion panel, which shows the same pixels with and without the
  Display P3 -> sRGB step so its effect is visible rather than assumed.

The 21 paired sheets are the highest-value artefact in the phase: the same soil,
both training cameras, identical physical scale, side by side. They are the only
visual proof that scale equalisation worked while the camera gap survived.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import colour, config, resample, soilmask

Image.MAX_IMAGE_PIXELS = None
PROXY_LONG = 420


def _proxy(im: Image.Image, long_side: int = PROXY_LONG) -> Image.Image:
    s = long_side / max(im.width, im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 14], fill=(0, 0, 0))
    d.text((2, 2), text[:int(out.width / 5.5)], fill=(255, 255, 255))
    return out


def sheet_for(row, tiles: pd.DataFrame | None) -> Image.Image:
    src_full = os.path.join(config.REPO_ROOT, row.source_path)
    with Image.open(src_full) as f:
        icc = f.info.get("icc_profile")
    im_raw, _ = resample.load_source(src_full)
    im_raw_only = _proxy(im_raw)

    can = Image.open(os.path.join(config.REPO_ROOT, row.processed_path)).convert("RGB")
    can_proxy = _proxy(can)

    # colour conversion shown, not assumed: same geometry, with and without P3->sRGB
    if icc:
        conv, _ = colour.to_srgb(im_raw, icc)
        nocv = im_raw_only
        cv = _proxy(conv)
        pair = Image.new("RGB", (nocv.width * 2 + 6, nocv.height), (30, 30, 30))
        pair.paste(nocv, (0, 0))
        pair.paste(cv, (nocv.width + 6, 0))
        p3 = _label(pair, "no ICC | ICC->sRGB")
    else:
        p3 = _label(can_proxy, "no embedded profile (sRGB)")

    gw = colour.apply_gains(np.asarray(can_proxy), (row.gw_gain_r, row.gw_gain_g, row.gw_gain_b))
    gwp = _label(Image.fromarray(gw), "gray-world (NOT saved)")

    mpath = os.path.join(config.REPO_ROOT, row.mask_path)
    if os.path.exists(mpath):
        arr = np.asarray(Image.open(os.path.join(config.REPO_ROOT, row.processed_path))
                         .convert("RGB"), dtype=np.float32).mean(axis=2)
        mask, flat = soilmask.soil_mask(arr)
        fmin, fmax = np.percentile(flat, [1, 99])
        diag = np.clip((flat - fmin) / max(fmax - fmin, 1e-6), 0, 1)
        combo = np.dstack([(diag * 255).astype(np.uint8),
                           (mask * 255).astype(np.uint8),
                           np.zeros_like(diag, dtype=np.uint8)])
        maskp = _label(_proxy(Image.fromarray(combo)), "illum-normalised + soil mask")
    else:
        maskp = _label(can_proxy, "no mask")

    x0, y0, x1, y1 = [int(v) for v in str(row.crop_rect_canonical).split(",")]
    ov = can_proxy.copy()
    d = ImageDraw.Draw(ov)
    sx = ov.width / can.width
    d.rectangle([x0 * sx, y0 * sx, x1 * sx, y1 * sx], outline=(255, 0, 0), width=2)
    cap = (f"{row.camera} | {row.sample_id} | eff {row.effective_ppm:.3f} ppm -> "
           f"{config.target_ppm():.4f} | FOV {row.fov_long_mm:.0f}x{row.fov_short_mm:.0f}mm "
           f"| soil {row.soil_fraction:.2f} | {row.mask_status}")
    ovl = _label(ov, cap)

    if tiles is not None:
        mine = tiles[(tiles.parent_image_path == row.processed_path)
                     & (tiles.materialized)]
        panels = []
        if len(mine):
            # deliberately include the tile nearest the crop boundary: highest risk
            ordered = mine.assign(bd=(mine.tile_x - x0).abs()
                                  + (mine.tile_y - y0).abs()).sort_values("bd")
            picks = [ordered.iloc[0]] + [mine.iloc[i] for i in
                                         np.linspace(0, len(mine) - 1, 3).astype(int)]
            seen = set()
            for t in picks:
                if t.tile_id in seen:
                    continue
                seen.add(t.tile_id)
                tp = os.path.join(config.REPO_ROOT, t.tile_path)
                if os.path.exists(tp):
                    panels.append(_label(Image.open(tp).convert("RGB").resize(
                        (int(t.tile_size_px * 0.9), int(t.tile_size_px * 0.9))),
                        f"tile x{t.tile_x} y{t.tile_y}"))
        if panels:
            w = sum(p.width for p in panels) + 6 * len(panels)
            h = max(p.height for p in panels)
            strip = Image.new("RGB", (w, h), (30, 30, 30))
            xx = 0
            for p in panels:
                strip.paste(p, (xx, 0))
                xx += p.width + 6
            tilep = _label(strip, f"materialised {config.MATERIALIZED_TILE_SIZE}px tiles")
        else:
            tilep = _label(can_proxy, "no materialised tiles")
    else:
        tilep = _label(can_proxy, "tiles not generated yet")

    cells = [ovl, _label(can_proxy, "canonical (resampled)"), p3,
             gwp, maskp, tilep]
    w = max(c.width for c in cells)
    h = sum(c.height for c in cells) + 6 * len(cells)
    out = Image.new("RGB", (w, h), (20, 20, 20))
    yy = 0
    for c in cells:
        out.paste(c, (0, yy))
        yy += c.height + 6
    return out


def paired_sheet(row_m, row_s) -> Image.Image:
    a = _proxy(Image.open(os.path.join(config.REPO_ROOT, row_m.processed_path)).convert("RGB"))
    b = _proxy(Image.open(os.path.join(config.REPO_ROOT, row_s.processed_path)).convert("RGB"))
    h = max(a.height, b.height)
    out = Image.new("RGB", (a.width + b.width + 8, h + 18), (20, 20, 20))
    out.paste(a, (0, 18))
    out.paste(b, (a.width + 8, 18))
    d = ImageDraw.Draw(out)
    d.text((2, 4), f"{row_m.sample_id}  Motorola {row_m.effective_ppm:.3f}ppm  |  "
                   f"Samsung {row_s.effective_ppm:.3f}ppm  (same soil, same scale)",
           fill=(255, 255, 255))
    return out


def run(force: bool = False) -> int:
    img = pd.read_csv(os.path.join(config.PROCESSED_DIR, "manifest_images.csv"))
    tpath = os.path.join(config.PROCESSED_DIR, "manifest_tiles.csv")
    tiles = pd.read_csv(tpath) if os.path.exists(tpath) else None
    root = os.path.join(config.QC_DIR, "sheets")
    config.assert_writable(root)
    os.makedirs(root, exist_ok=True)

    made = 0
    for i, row in enumerate(img.itertuples(index=False)):
        dest = os.path.join(root, row.split, row.sample_slug,
                            os.path.splitext(row.filename)[0] + ".png")
        config.assert_writable(dest)
        if os.path.exists(dest) and not force:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            sheet_for(row, tiles).save(dest)
            made += 1
        except Exception as exc:
            print(f"  sheet failed for {row.source_path}: {exc!r}")
        if (i + 1) % 30 == 0:
            print(f"  {i + 1}/{len(img)}")

    # paired-camera sheets
    proot = os.path.join(config.QC_DIR, "paired")
    os.makedirs(proot, exist_ok=True)
    n_paired = 0
    for sample, grp in img[img.split == "train"].groupby("sample_id"):
        m = grp[grp.camera == "Motorola Edge"]
        s = grp[grp.camera == "Samsung A52"]
        if m.empty or s.empty:
            continue
        dest = os.path.join(proot, f"{sample}.png")
        config.assert_writable(dest)
        if os.path.exists(dest) and not force:
            continue
        paired_sheet(m.iloc[0], s.iloc[0]).save(dest)
        n_paired += 1

    # deterministic review set: not hand-picked, so the same 20 every run
    sel = []
    sel += img.nsmallest(5, "crop_area_fraction").image_id.tolist()
    sel += img.nsmallest(5, "soil_fraction").image_id.tolist()
    sel += img[(img.camera == "Motorola Edge") & (img.stored_portrait)] \
        .head(5).image_id.tolist()
    sel += img[img.mask_status == "fallback_fullframe"].head(3).image_id.tolist()
    sel = list(dict.fromkeys(sel))
    pd.DataFrame({"image_id": sel,
                  "reason": ["review-set selection"] * len(sel)}).to_csv(
        os.path.join(config.QC_DIR, "qc_selection.csv"), index=False)

    print(f"\nsheets written: {made}, paired sheets: {n_paired}")
    print(f"review set: {len(sel)} unique sheets listed in data/qc/qc_selection.csv")
    print("  (drawn from 5 smallest crops + 5 lowest soil fraction + 5 stored-portrait "
          "Motorola + 3 full-frame fallbacks; de-duplicated, so the count can be < 18)")
    return 0
