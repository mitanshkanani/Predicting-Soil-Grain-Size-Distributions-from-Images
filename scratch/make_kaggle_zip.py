"""Builds the single Kaggle dataset zip for Model 2 Experiment 1.

Why one zip and not two datasets: the notebook resolves every input by SEARCHING each
attachment (`resolve_input_root`, `resolve_backbones_dir`, `_prior_dir`), so one attachment
holding everything is strictly simpler than three, and there is no version-pinning to get
wrong.

What is deliberately NOT in here, and why:
  data/training_down, data/testing_down   the notebook never opens these. Their paths are
                                          join keys in manifest_images.csv only; every
                                          pixel it reads comes from data/tiles. 445 MB.
  data/Training, data/Test                raw sources. Forbidden to model notebooks; the
                                          absence is the guarantee.
  data/qc                                 contact sheets, nothing reads them.
  data/processed_meta/masks               Experiment 1 never reads masks; per-tile
                                          soil_fraction is already in manifest_tiles.csv.
  Training_labels_updated.csv, ppm_updated.csv, data/information.md
                                          labels come from manifest_samples.csv (the single
                                          source of truth) and the scale from the pinned
                                          config_hash. Shipping a second label file is how
                                          two experiments end up using different labels.
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(".")
OUT = ROOT / "final_kaggle_upload_m2e1.zip"
META = ROOT / "data" / "processed_meta"

META_FILES = ["manifest_images.csv", "manifest_tiles.csv", "manifest_samples.csv",
              "audit.json", "golden_checks.json", "target_ppm.json"]
E3 = ROOT / "Model 1" / "Model 1 Experiment 3"
E3_SKIP = {"Model1_Experiment3.ipynb"}     # 83 KB of notebook JSON nothing reads


def members():
    """(absolute source path, name inside the zip) for everything the notebook needs."""
    tiles = pd.read_csv(META / "manifest_tiles.csv")
    tiles = tiles[(tiles.tile_size_px == 256) & tiles.materialized]
    n = 0
    for p in tiles.tile_path:
        src = ROOT / p
        assert src.exists(), f"manifest tile is missing on disk: {p}"
        yield src, p.replace("\\", "/")
        n += 1
    assert n == len(tiles) == 1976, f"expected 1976 tiles, emitted {n}"

    for f in META_FILES:
        yield META / f, f"data/processed_meta/{f}"
    yield ROOT / "data" / "sample_submission.csv", "data/sample_submission.csv"

    for p in sorted((ROOT / "preprocess").rglob("*.py")):
        yield p, "preprocess/" + p.relative_to(ROOT / "preprocess").as_posix()

    m2 = ROOT / "Model 2" / "Model 2 Experiment 1"
    for f in ("backbones.py", "check_backbones.py"):
        yield m2 / f, f

    for p in sorted(E3.rglob("*")):
        if p.is_file() and p.name not in E3_SKIP and "__pycache__" not in p.parts:
            yield p, "Model 1 Experiment 3/" + p.relative_to(E3).as_posix()


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    total = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for src, arc in members():
            # PNGs are already compressed; deflating 276 MB of them buys ~0 and costs minutes.
            z.write(src, arc, compress_type=zipfile.ZIP_DEFLATED if src.suffix in
                    {".csv", ".json", ".txt", ".md", ".py"} else zipfile.ZIP_STORED)
            total += src.stat().st_size
    print(f"wrote {OUT.name}  {OUT.stat().st_size/1e6:.1f} MB "
          f"({total/1e6:.1f} MB uncompressed)")
    verify()


def verify() -> None:
    """Round-trip the archive and prove every manifest tile path survives extraction with
    its bytes intact - including the 75 Münster filenames that carry a non-ASCII u-umlaut,
    which is the one thing that could silently break cell B1."""
    import shutil
    import tempfile

    tiles = pd.read_csv(META / "manifest_tiles.csv")
    want = list(tiles[(tiles.tile_size_px == 256) & tiles.materialized].tile_path)
    tmp = Path(tempfile.mkdtemp(prefix="m2zip_"))
    try:
        with zipfile.ZipFile(OUT) as z:
            bad = z.testzip()
            assert bad is None, f"corrupt entry in the archive: {bad}"
            names = set(z.namelist())
            missing = [p for p in want if p not in names]
            assert not missing, f"{len(missing)} tiles absent, e.g. {missing[:2]}"
            z.extractall(tmp)
        diff = [p for p in want
                if hashlib.md5((tmp / p).read_bytes()).hexdigest()
                != hashlib.md5((ROOT / p).read_bytes()).hexdigest()]
        assert not diff, f"{len(diff)} tiles differ after extraction, e.g. {diff[:2]}"
        for f in [*(f"data/processed_meta/{m}" for m in META_FILES),
                  "data/sample_submission.csv", "preprocess/verify.py",
                  "preprocess/config.py", "backbones.py", "check_backbones.py",
                  "Model 1 Experiment 3/features_soil.csv",
                  "Model 1 Experiment 3/cv_families.csv",
                  "Model 1 Experiment 3/Submission_Model1_E3.csv",
                  "Model 1 Experiment 3/.cache/tile_features_256_010f44c36c74.csv"]:
            assert (tmp / f).exists(), f"required member missing: {f}"
        na = sum(1 for p in want if any(ord(c) > 127 for c in p))
        print(f"ROUND TRIP OK: {len(want)} tiles byte-identical after extract, "
              f"including {na} non-ASCII names")
        print("  every required manifest, the preprocess package, both backbone files and")
        print("  Experiment 3's signature artifacts are present.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
