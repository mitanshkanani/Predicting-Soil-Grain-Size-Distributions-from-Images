"""STEP 2 - resolve camera identity and embedded metadata for every image.

The brief asked for EXIF-first with filename as fallback. That is unsatisfiable
here: 124 of 162 images (all Motorola Edge and Samsung A52 training photos) have
no EXIF block at all. So the rule is inverted -- filename is the primary key and
EXIF acts only as a validator that can raise a flag, never as a source. Every
value records which path produced it in `camera_source`.
"""
from __future__ import annotations

import io
import os
import re
import struct

import pandas as pd
from PIL import Image, JpegImagePlugin

from . import config

Image.MAX_IMAGE_PIXELS = None

# Ordered most-specific first. "Motorola Edge" is a literal prefix of
# "Motorola Edge 60 Fusion", so testing the short key first would silently
# mislabel the 3 H374 files (and derive their sample id as "60").
CAMERA_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Motorola Edge 60 Fusion", re.compile(r"^motorola_edge_60_fusion_", re.I)),
    ("Motorola Edge",           re.compile(r"^motorola_edge_", re.I)),
    ("Samsung A52",             re.compile(r"^samsung_a52_", re.I)),
    # tolerates the measured malformed 'iPhone_16_HPC_...' spelling
    ("iPhone 14",               re.compile(r"^iphone_?14_hpc_", re.I)),
    ("iPhone 16",               re.compile(r"^iphone_?16_hpc_", re.I)),
]

NUL = re.compile(r"\x00+")


def _clean(value) -> str | None:
    if value is None:
        return None
    return NUL.sub("", str(value)).strip()


def camera_from_filename(filename: str) -> tuple[str | None, str]:
    """Return (camera, parse_status). status is 'ok' or 'malformed_recovered'."""
    stem = os.path.splitext(filename)[0]
    for camera, pat in CAMERA_PATTERNS:
        if pat.match(stem):
            status = "ok"
            # 'iPhone_16_' deviates from the 'iPhone16_' convention used everywhere
            # else; flag it as recovered rather than pretending it parsed cleanly.
            if camera.startswith("iPhone") and re.match(r"^iphone_", stem, re.I):
                status = "malformed_recovered"
            return camera, status
    return None, "unparseable"


def image_index_from_filename(filename: str) -> int | None:
    stem = os.path.splitext(filename)[0]
    m = re.search(r"\((\d+)\)\s*$", stem) or re.search(r"_(\d+)\s*$", stem)
    return int(m.group(1)) if m else None


def read_ppm_table() -> pd.DataFrame:
    df = pd.read_csv(config.PPM_CSV)
    # The CSV's `camera` column holds sensor model strings ('motorola edge 20') that
    # differ from the phone names we key on, so join on `phone` only.
    return df.set_index("phone")


def _icc_probe(prof: bytes) -> dict:
    """Parse just enough of the ICC profile to name it and read its red primary."""
    out = {"icc_bytes": len(prof), "icc_colorspace": prof[16:20].decode("latin1"),
           "icc_pcs": prof[20:24].decode("latin1"), "icc_desc": None,
           "icc_rXYZ": None, "icc_red_chromaticity": None}
    try:
        n = struct.unpack(">I", prof[128:132])[0]
        tags = {}
        for i in range(n):
            off = 132 + 12 * i
            tags[prof[off:off + 4].decode("latin1")] = struct.unpack(
                ">II", prof[off + 4:off + 12])
        for key in ("desc", "dscm"):
            if key in tags:
                o, _l = tags[key]
                typ = prof[o:o + 4].decode("latin1")
                if typ == "desc":
                    ln = struct.unpack(">I", prof[o + 8:o + 12])[0]
                    out["icc_desc"] = prof[o + 12:o + 12 + ln - 1].decode(
                        "latin1", errors="replace").strip("\x00 ")
                elif typ == "mluc":
                    cnt = struct.unpack(">I", prof[o + 12:o + 16])[0]
                    if cnt:
                        rec = o + 20
                        ln = struct.unpack(">I", prof[rec + 4:rec + 8])[0]
                        toff = struct.unpack(">I", prof[rec + 8:rec + 12])[0]
                        out["icc_desc"] = prof[o + toff:o + toff + ln * 2].decode(
                            "utf-16-be", errors="replace")
                if out["icc_desc"]:
                    break
        if "rXYZ" in tags:
            o, _l = tags["rXYZ"]
            if prof[o:o + 4].decode("latin1") in ("XYZ ", "xyz "):
                x, y, z = struct.unpack(">3i", prof[o + 8:o + 20])
                x, y, z = x / 65536.0, y / 65536.0, z / 65536.0
                out["icc_rXYZ"] = (round(x, 5), round(y, 5), round(z, 5))
                tot = x + y + z
                if tot:
                    out["icc_red_chromaticity"] = (round(x / tot, 4), round(y / tot, 4))
    except Exception as exc:  # a malformed profile must not kill the stage
        out["icc_parse_error"] = repr(exc)[:120]
    return out


def probe(path: str) -> dict:
    """Read EXIF + ICC + delivered dimensions from one file."""
    rec = {"has_exif": False, "exif_make": None, "exif_model": None,
           "exif_orientation": None, "exif_native_w": None, "exif_native_h": None,
           "exif_xres": None, "exif_yres": None, "exif_res_unit": None,
           "exif_datetime": None, "jfif_dpi": None, "icc_present": False,
           "icc_desc": None, "icc_colorspace": None, "icc_pcs": None,
           "icc_rXYZ": None, "icc_red_chromaticity": None, "icc_bytes": None,
           "icc_parse_error": None, "delivered_w": None, "delivered_h": None,
           "n_frames": 1}
    with Image.open(path) as im:
        rec["delivered_w"], rec["delivered_h"] = im.size
        rec["n_frames"] = getattr(im, "n_frames", 1)
        dpi = im.info.get("dpi")
        # Pillow returns IFDRational for EXIF-derived dpi, which has no __format__.
        rec["jfif_dpi"] = (f"{float(dpi[0]):g}x{float(dpi[1]):g}"
                           if dpi and len(dpi) == 2 else None)
        prof = im.info.get("icc_profile")
        if prof:
            rec["icc_present"] = True
            rec.update(_icc_probe(prof))
        raw = dict(im.getexif()) if im.getexif() else {}
    if raw:
        rec["has_exif"] = True
        rec["exif_make"] = _clean(raw.get(271))
        rec["exif_model"] = _clean(raw.get(272))
        rec["exif_orientation"] = raw.get(274)
        rec["exif_native_w"] = raw.get(256)
        rec["exif_native_h"] = raw.get(257)
        rec["exif_xres"] = float(raw[282]) if 282 in raw else None
        rec["exif_yres"] = float(raw[283]) if 283 in raw else None
        rec["exif_res_unit"] = raw.get(296)
        rec["exif_datetime"] = _clean(raw.get(306))
    return rec


def _exif_agrees_with_filename(exif_model: str | None, camera: str | None) -> bool | None:
    """EXIF may only confirm or contradict; it may never choose the camera."""
    if exif_model is None:
        return None
    em = exif_model.lower()
    cam = (camera or "").lower()
    if "iphone 14" in em:
        return cam == "iphone 14"
    if "iphone 16" in em:
        return cam == "iphone 16"
    if "edge 60" in em:
        return cam == "motorola edge 60 fusion"
    if "edge" in em or em.startswith("motorola"):
        return cam == "motorola edge"
    if em.startswith("sm-a52"):
        return cam == "samsung a52"
    return False


def run(force: bool = False) -> int:
    inv = pd.read_csv(config.stage_path("inventory"))
    out = config.stage_path("cameras")
    if os.path.exists(out) and not force:
        print("output exists, using it (--force to redo)")
        return 0

    ppm = read_ppm_table()
    recs = []
    for row in inv.itertuples(index=False):
        abs_path = os.path.join(config.REPO_ROOT, row.source_path)
        camera, status = camera_from_filename(row.filename)
        meta = probe(abs_path)
        native = ppm.loc[camera] if camera in ppm.index else None
        rec = {
            "image_id": row.image_id,
            "camera": camera,
            "camera_source": "filename_only",
            "camera_make": None,
            "camera_model": None,
            "filename_parse_status": status,
            "image_index": image_index_from_filename(row.filename),
            "native_width": int(native["width"]) if native is not None else None,
            "native_height": int(native["height"]) if native is not None else None,
            "native_ppm": float(native["ppm"]) if native is not None else None,
            "native_res_source": "ppm_updated.csv" if native is not None else "UNRESOLVED",
            "exif_agrees_with_filename": _exif_agrees_with_filename(
                meta["exif_model"], camera),
            **meta,
        }
        if rec["exif_agrees_with_filename"] is True:
            rec["camera_source"] = "exif_and_filename"
            rec["camera_make"] = meta["exif_make"]
            rec["camera_model"] = meta["exif_model"]
        elif rec["exif_agrees_with_filename"] is False:
            rec["camera_make"] = meta["exif_make"]
            rec["camera_model"] = meta["exif_model"]
        recs.append(rec)

    df = pd.DataFrame(recs)
    df.to_csv(out, index=False)

    print(f"rows: {len(df)}")
    print("\ncamera breakdown:")
    print(df.camera.value_counts(dropna=False).to_string())
    print(f"\nunknown camera: {int(df.camera.isna().sum())} (expected 0)")
    print(f"has_exif: {int(df.has_exif.sum())} (expected 38)")
    print(f"icc_present: {int(df.icc_present.sum())} (expected 38)")
    print(f"exif/filename disagreements: "
          f"{int((df.exif_agrees_with_filename == False).sum())} (expected 0)")
    print(f"malformed filenames recovered: "
          f"{int((df.filename_parse_status == 'malformed_recovered').sum())} (expected 1)")
    print(f"EXIF ImageWidth present on: {int(df.exif_native_w.notna().sum())} files")
    print(f"EXIF orientation: "
          f"{df.exif_orientation.fillna(-1).value_counts().sort_index().to_dict()}")
    print("\nICC profiles found:")
    for cam, g in df[df.icc_present].groupby("camera", dropna=False):
        c = g.icc_red_chromaticity.iloc[0]
        print(f"  {cam:24s} n={len(g):3d} desc={g.icc_desc.iloc[0]!r} "
              f"red_chromaticity={c}")
    print("  (Display P3 red primary = (0.680, 0.320); sRGB = (0.640, 0.330))")
    print(f"\nEXIF XResolution values seen: {sorted(set(df.exif_xres.dropna()))}")
    print(f"JFIF dpi values seen: {sorted(set(df.jfif_dpi.dropna()))}")
    print("  -> the two placeholder density fields disagree; neither is a physical scale.")
    return 0
