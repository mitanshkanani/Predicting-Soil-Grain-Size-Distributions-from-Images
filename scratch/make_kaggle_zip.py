"""Builds the single Kaggle dataset zip for a given experiment.

    python scratch/make_kaggle_zip.py m2e1
    python scratch/make_kaggle_zip.py m2e2

Why one zip and not several attachments: the notebook resolves every input by searching
each attachment, and Kaggle searches them ALPHABETICALLY. An older dataset holding an older
copy of a side-car module silently wins over the fixed one - that cost a whole session once.
One self-contained dataset removes the class of mistake.

What is deliberately left out, and why this was measured rather than guessed:
  data/training_down, data/testing_down   445 MB the notebooks never open. Their paths are
                                          join keys in manifest_images.csv only; every
                                          pixel read comes from data/tiles.
  data/Training, data/Test                raw sources. Model notebooks are forbidden to
                                          read them; their absence is the guarantee.
  data/qc                                 contact sheets, nothing reads them.
  data/processed_meta/masks               never read; per-tile soil_fraction is already in
                                          manifest_tiles.csv.
  Training_labels_updated.csv,
  ppm_updated.csv                         labels come from manifest_samples.csv only. A
                                          second label file is how two experiments end up
                                          disagreeing about the same soil.
  any 'local mock dry-run' folder         plumbing artifacts with the same filenames as
                                          real results. Including them invites the
                                          notebook comparing an experiment against itself.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "processed_meta"
META_FILES = ["manifest_images.csv", "manifest_tiles.csv", "manifest_samples.csv",
              "audit.json", "golden_checks.json", "target_ppm.json"]
SKIP_DIRS = {"local mock dry-run", "__pycache__"}
SKIP_FILES = {"kaggle_setup.md"}          # documentation for the owner, not for the run
# Modules that ship at the archive root. A second copy nested inside a prior experiment's
# folder is the exact hazard that cost a session: attachments are searched alphabetically,
# so two copies of backbones.py means one of them can silently win.
ROOT_MODULES = {"backbones.py", "check_backbones.py", "crop_geometry.py",
                "check_crop_geometry.py"}


def _m2(dir_name):
    return ROOT / "Model 2" / dir_name


SPECS = {
    # experiment: (output zip name, extra root-level modules, prior-experiment folders)
    "m2e1": ("final_kaggle_upload_m2e1.zip",
             [_m2("Model 2 Experiment 1") / "backbones.py",
              _m2("Model 2 Experiment 1") / "check_backbones.py"],
             [ROOT / "Model 1" / "Model 1 Experiment 3"]),
    "m2e2": ("final_kaggle_upload_m2e2.zip",
             [_m2("Model 2 Experiment 1") / "backbones.py",
              _m2("Model 2 Experiment 1") / "check_backbones.py",
              _m2("Model 2 Experiment 2") / "crop_geometry.py",
              _m2("Model 2 Experiment 2") / "check_crop_geometry.py"],
             [ROOT / "Model 1" / "Model 1 Experiment 3",
              _m2("Model 2 Experiment 1")]),
}


def members(exp: str):
    tiles = pd.read_csv(META / "manifest_tiles.csv")
    tiles = tiles[(tiles.tile_size_px == 256) & tiles.materialized]
    n = 0
    for p in tiles.tile_path:
        src = ROOT / p
        assert src.exists(), f"manifest tile missing on disk: {p}"
        yield src, p.replace("\\", "/")
        n += 1
    assert n == len(tiles) == 1976, f"expected 1976 tiles, emitted {n}"

    for f in META_FILES:
        yield META / f, f"data/processed_meta/{f}"
    yield ROOT / "data" / "sample_submission.csv", "data/sample_submission.csv"

    for p in sorted((ROOT / "preprocess").rglob("*.py")):
        yield p, "preprocess/" + p.relative_to(ROOT / "preprocess").as_posix()

    out_zip, extra, prior_dirs = SPECS[exp]
    for p in extra:
        assert p.exists(), f"required module missing: {p}"
        yield p, p.name

    for d in prior_dirs:
        if not d.is_dir():
            print(f"  note: prior-experiment folder absent, skipped: {d.name}")
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file() or SKIP_DIRS & set(p.relative_to(d).parts):
                continue
            rel = p.relative_to(d).as_posix()
            if p.name in SKIP_FILES or p.suffix == ".ipynb":
                continue
            # Regenerable embedding caches: 17 MB of mock .npy per experiment, and a stale
            # one would be worse than none because the cache key would still match.
            if p.suffix == ".npy":
                continue
            # A module already shipped at the root must not appear a second time nested.
            if p.name in ROOT_MODULES and rel.count("/") == 0:
                continue
            yield p, f"{d.name}/{rel}"


def main(exp: str) -> None:
    out_zip = ROOT / SPECS[exp][0]
    if out_zip.exists():
        out_zip.unlink()
    total = 0
    seen = set()
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for src, arc in members(exp):
            assert arc not in seen, f"duplicate member {arc}"
            seen.add(arc)
            z.write(src, arc,
                    compress_type=zipfile.ZIP_DEFLATED if src.suffix in
                    {".csv", ".json", ".txt", ".md", ".py"} else zipfile.ZIP_STORED)
            total += src.stat().st_size
    print(f"wrote {out_zip.name}  {out_zip.stat().st_size/1e6:.1f} MB "
          f"({total/1e6:.1f} MB uncompressed, {len(seen)} members)")
    verify(out_zip)


def verify(out: Path) -> None:
    """Round-trip the archive: every manifest tile must come out byte-identical. The 75
    Munster filenames carry a non-ASCII u-umlaut, which is the one thing here that could
    silently break a notebook's tile lookup on a different filesystem."""
    import shutil
    import tempfile

    tiles = pd.read_csv(META / "manifest_tiles.csv")
    want = list(tiles[(tiles.tile_size_px == 256) & tiles.materialized].tile_path)
    tmp = Path(tempfile.mkdtemp(prefix="kgzip_"))
    try:
        with zipfile.ZipFile(out) as z:
            assert z.testzip() is None, "corrupt entry in the archive"
            names = set(z.namelist())
            missing = [p for p in want if p not in names]
            assert not missing, f"{len(missing)} tiles absent, e.g. {missing[:2]}"
            z.extractall(tmp)
        bad = [p for p in want
               if hashlib.md5((tmp / p).read_bytes()).hexdigest()
               != hashlib.md5((ROOT / p).read_bytes()).hexdigest()]
        assert not bad, f"{len(bad)} tiles differ after extraction, e.g. {bad[:2]}"
        for f in ["data/processed_meta/manifest_images.csv", "data/sample_submission.csv",
                  "preprocess/verify.py", "backbones.py"]:
            assert (tmp / f).exists(), f"required member missing: {f}"
        na = sum(1 for p in want if any(ord(c) > 127 for c in p))
        print(f"ROUND TRIP OK: {len(want)} tiles byte-identical after extract, "
              f"including {na} non-ASCII names")
        top = sorted({n.split("/")[0] for n in names})
        print(f"top level: {top}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    exp = (sys.argv[1] if len(sys.argv) > 1 else "m2e1").lower()
    assert exp in SPECS, f"unknown experiment {exp!r}; known: {sorted(SPECS)}"
    main(exp)
