"""build_m11_mosaic_zip.py - Model 11 / Experiment 1: the Kaggle upload archive for T3.

Exactly 163 entries: mosaic_manifest.csv at the root plus the 162 mosaics under mosaics/.
ZIP_STORED, because PNG is already compressed and any re-encode would alter the texture the VLM
is being asked to read; the bytes must be identical to the files S1 wrote.

The identity map (LOCAL_ONLY_mosaic_map.csv) is never read into the archive, and the archive is
checked against it in the OTHER direction: every sample id, camera string and split value that
appears in the map is searched for in the entry names, so a leak would be caught rather than
merely not-intended.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1] / "Model 11" / "Experiment 1"
ZIP = EXP / "mosaics_upload.zip"
MANIFEST = EXP / "mosaic_manifest.csv"
MAPFILE = EXP / "LOCAL_ONLY_mosaic_map.csv"
COLS = ["mosaic_id", "mosaic_path", "sha256_mosaic", "n_tiles_available",
        "n_cells_filled_by_repeat", "grid", "tile_px"]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def build() -> int:
    man = pd.read_csv(MANIFEST)
    expect = {"mosaic_manifest.csv"} | set(man.mosaic_path.tolist())
    print("=" * 78)
    print("MODEL 11 / EXPERIMENT 1 -- build mosaics_upload.zip (ZIP_STORED)")
    print("=" * 78)
    print("  manifest rows: %d | expected entries: %d" % (len(man), len(expect)))

    missing = [p for p in man.mosaic_path if not (EXP / p).exists()]
    if missing or not MANIFEST.exists():
        print("  *** source files missing: %r" % missing[:3])
        return 1
    if len(man) != 162:
        print("  *** expected 162 mosaics, manifest lists %d" % len(man))
        return 1

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_STORED, compresslevel=None) as z:
        z.write(MANIFEST, "mosaic_manifest.csv")
        for p in man.mosaic_path:
            z.write(EXP / p, p)

    # ---------------- verification, from the archive itself ----------------
    fails = []
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        print("\n--- (a) entry count ---")
        print("  entries: %d (expected 163)" % len(names))
        if len(names) != 163:
            fails.append("entry count %d != 163" % len(names))
        extra = sorted(set(names) - expect)
        absent = sorted(expect - set(names))
        print("  unexpected entries: %d %s" % (len(extra), extra[:4] if extra else ""))
        print("  missing entries   : %d %s" % (len(absent), absent[:4] if absent else ""))
        if extra or absent:
            fails.append("entry set mismatch")
        if any(not n.replace("/", "").replace(".", "").replace("_", "").replace("-", "")
               .isascii() for n in names):
            fails.append("non-ascii entry name")
        if any("\\" in n for n in names):
            fails.append("backslash in entry name")
        print("  all entry names use forward slashes and are ASCII: %s"
              % all("\\" not in n and n.isascii() for n in names))

        print("\n--- (b) every PNG's sha256 inside the zip vs mosaic_manifest.csv ---")
        bad = 0
        for _, r in man.iterrows():
            raw = z.read(r.mosaic_path)
            got = sha_bytes(raw)
            disk = (EXP / r.mosaic_path).read_bytes()
            if got != r.sha256_mosaic or raw != disk:
                bad += 1
                print("    MISMATCH %s zip=%s manifest=%s bytes_equal=%s"
                      % (r.mosaic_id, got[:12], str(r.sha256_mosaic)[:12], raw == disk))
        print("  %d/%d PNGs byte-identical and hash-matching, %d problems"
              % (len(man) - bad, len(man), bad))
        if bad:
            fails.append("%d png hash/byte mismatches" % bad)
        mbytes = z.read("mosaic_manifest.csv")
        print("  manifest inside zip byte-identical to on-disk manifest: %s"
              % (mbytes == MANIFEST.read_bytes()))
        if mbytes != MANIFEST.read_bytes():
            fails.append("manifest altered in archive")
        compression = {i.compress_type for i in z.infolist()}
        print("  compress_type set: %s (0 = ZIP_STORED, no re-encode)" % compression)
        if compression != {zipfile.ZIP_STORED}:
            fails.append("archive is not ZIP_STORED")

        print("\n--- (c) no identity information in any entry name ---")
        joined = " | ".join(names).lower()
        print("  'LOCAL_ONLY' present in any entry name: %s" % ("local_only" in joined))
        if "local_only" in joined:
            fails.append("LOCAL_ONLY leaked into archive")
        mp = pd.read_csv(MAPFILE)
        banned = set()
        for v in mp.sample_id.astype(str):
            banned.add(v.lower())
            banned.add(v.lower().replace(" ", "_"))
        for v in mp.camera.astype(str).unique():
            for tok in v.lower().split():
                if len(tok) > 3:
                    banned.add(tok)
        for v in mp.split.astype(str).unique():
            banned.add(v.lower())
        banned |= {"motorola", "samsung", "iphone", "train", "test", "site", "cv_group"}
        hits = sorted(b for b in banned if b and b in joined)
        print("  identity/camera/split tokens searched: %d" % len(banned))
        print("  tokens found in entry names          : %s" % (hits if hits else "NONE"))
        if hits:
            fails.append("identity token in entry names: %s" % hits[:5])
        neutral = all(len(Path(n).stem) == 12 for n in names if n != "mosaic_manifest.csv")
        print("  every mosaic filename is a 12-hex neutral id: %s" % neutral)
        if not neutral:
            fails.append("a mosaic filename is not a 12-hex id")

        print("\n--- (d) manifest columns unchanged ---")
        cols = list(pd.read_csv(io.BytesIO(mbytes)).columns)
        print("  columns: %s" % cols)
        print("  unchanged: %s" % (cols == COLS))
        if cols != COLS:
            fails.append("manifest columns changed")

    size = ZIP.stat().st_size
    print("\n" + "=" * 78)
    print("zip        : %s" % ZIP.name)
    print("zip size   : %.2f MB (%d bytes)" % (size / 1e6, size))
    print("zip sha256 : %s" % sha_bytes(ZIP.read_bytes()))
    print("entries    : 163 (1 manifest + 162 PNGs), ZIP_STORED")
    print("VERIFICATION: %s" % ("ALL PASS" if not fails else "FAILURES"))
    for f in fails:
        print("   FAIL: %s" % f)
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(build())
