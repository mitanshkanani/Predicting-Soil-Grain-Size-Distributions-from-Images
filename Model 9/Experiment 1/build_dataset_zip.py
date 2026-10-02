"""build_dataset_zip.py - assemble final_kaggle_upload_m9e1.zip from data/ for Model 9 / Exp 1.

Self-contained: the archive carries everything the Kaggle notebook reads and nothing it does not.
Modelled on scratch/make_kaggle_zip.py (the builder every prior upload used), including its
round-trip md5 verification - the one guard that catches the 75 Munster tiles whose filenames
carry a non-ASCII u-umlaut and could silently break a lookup on another filesystem.

WHAT GOES IN (and why):
  data/tiles/**/*__t256.png        all 1976 materialized 256px tiles (1541 train + 435 test).
                                   The notebook applies soil_fraction>=0.50 at READ time (-> 405
                                   test actually consumed), exactly as the 55.80591 head did; the
                                   fraction filter lives in the notebook, not the archive, so it
                                   can be varied without a rebuild.
  data/processed_meta/manifest_*   images / tiles / samples. The tiles manifest carries every
                                   tile_path and soil_fraction; samples carries the submission_id
                                   join. No separate label file (that is how two runs end up
                                   disagreeing about one soil).
  data/sample_submission.csv       the 10-row board contract and id order.
  handcrafted_oof_train.csv        the 55.8 head's family-held-out OOF on the 24 soils -> honest
                                   blend-weight selection, no re-running the frozen head on Kaggle.
  train_labels_matrix.csv          the 24x11 labels so the notebook scores blends manifest-free.
  Submission_Model8_E2_E2A.csv     the 55.80591 TEST predictions - the blend's handcrafted side.

WHAT STAYS OUT: data/training_down, data/testing_down (445 MB of join-key images no notebook
  opens), raw sources, qc sheets, masks, extra label files - every pixel the run reads is a tile.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # Model 9/Experiment 1 -> repo root
META = ROOT / "data" / "processed_meta"
OUT  = ROOT / "final_kaggle_upload_m9e1.zip"

MANIFESTS = ["manifest_images.csv", "manifest_tiles.csv", "manifest_samples.csv"]
SIDE_FILES = [                                # (source, archive-name at root)
    (HERE / "handcrafted_oof_train.csv", "handcrafted_oof_train.csv"),
    (HERE / "train_labels_matrix.csv",   "train_labels_matrix.csv"),
    (ROOT / "Model 8" / "Experiment 2" / "Submission_Model8_E2_E2A.csv",
     "Submission_Model8_E2_E2A.csv"),
]
EXPECT_TILES = 1976                           # 1541 train + 435 test, materialized t256


def members():
    tiles = pd.read_csv(META / "manifest_tiles.csv")
    tiles = tiles[(tiles.tile_size_px == 256) & tiles.materialized]
    n = 0
    for p in tiles.tile_path:
        src = ROOT / p
        assert src.exists(), "manifest tile missing on disk: %s" % p
        yield src, p.replace("\\", "/")
        n += 1
    assert n == len(tiles) == EXPECT_TILES, "expected %d tiles, emitted %d" % (EXPECT_TILES, n)

    for f in MANIFESTS:
        yield META / f, "data/processed_meta/%s" % f
    yield ROOT / "data" / "sample_submission.csv", "data/sample_submission.csv"

    for src, arc in SIDE_FILES:
        assert src.exists(), "required side-file missing: %s" % src
        yield src, arc


def build() -> None:
    if OUT.exists():
        OUT.unlink()
    total, seen = 0, set()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for src, arc in members():
            assert arc not in seen, "duplicate member %s" % arc
            seen.add(arc)
            z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED
                    if src.suffix in {".csv", ".json", ".txt"} else zipfile.ZIP_STORED)
            total += src.stat().st_size
    print("wrote %s  %.1f MB (%.1f MB uncompressed, %d members)"
          % (OUT.name, OUT.stat().st_size / 1e6, total / 1e6, len(seen)))
    verify()


def verify() -> None:
    """Round-trip: every manifest tile out byte-identical, Munster non-ASCII names included."""
    import shutil, tempfile
    tiles = pd.read_csv(META / "manifest_tiles.csv")
    want = list(tiles[(tiles.tile_size_px == 256) & tiles.materialized].tile_path)
    want = [p.replace("\\", "/") for p in want]
    tmp = Path(tempfile.mkdtemp(prefix="m9zip_"))
    try:
        with zipfile.ZipFile(OUT) as z:
            assert z.testzip() is None, "corrupt entry in the archive"
            names = set(z.namelist())
            missing = [p for p in want if p not in names]
            assert not missing, "%d tiles absent, e.g. %s" % (len(missing), missing[:2])
            for _, arc in SIDE_FILES:
                assert arc in names, "side-file missing from archive: %s" % arc
            for f in MANIFESTS:
                assert "data/processed_meta/%s" % f in names, "manifest missing: %s" % f
            assert "data/sample_submission.csv" in names
            z.extractall(tmp)
        bad = [p for p in want
               if hashlib.md5((tmp / p).read_bytes()).hexdigest()
               != hashlib.md5((ROOT / p).read_bytes()).hexdigest()]
        assert not bad, "%d tiles differ after extraction, e.g. %s" % (len(bad), bad[:2])
        na = sum(1 for p in want if any(ord(c) > 127 for c in p))
        print("ROUND TRIP OK: %d tiles byte-identical after extract, incl. %d non-ASCII names"
              % (len(want), na))
        print("top level:", sorted({n.split("/")[0] for n in names}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    build()
