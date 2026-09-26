"""STEP 1 - enumerate the image universe and record tamper-evidence for the
read-only source tree.

Establishes the 162-row corpus and a sha256 baseline that audit.py re-checks, so
"data/Training and data/Test were untouched" is a proved claim rather than an
assumption.
"""
from __future__ import annotations

import hashlib
import json
import os

import pandas as pd

from . import config


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(name: str) -> str:
    """ASCII directory name for a sample: umlaut transliterated, punctuation folded.

    Needed because sample_submission.csv spells the Munster site 'Muenster' with an
    underscore where the folder uses 'M' + u-umlaut and a comma. The same rule is
    what makes the 10/10 submission-id match work, and it keeps paths shell-safe on
    a Windows checkout where the umlaut round-trips as mojibake.
    """
    out = name.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    out = "".join(c if c.isalnum() else "_" for c in out)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def enumerate_images() -> tuple[list[dict], list[str]]:
    """Return (rows, skipped_non_image_paths)."""
    rows: list[dict] = []
    skipped: list[str] = []
    for split, root in sorted(config.SOURCE_DIRS.items()):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"missing source dir for split {split}: {root}")
        for sample in sorted(os.listdir(root)):
            pdir = os.path.join(root, sample)
            if not os.path.isdir(pdir):
                skipped.append(pdir)
                continue
            for fn in sorted(os.listdir(pdir)):
                fp = os.path.join(pdir, fn)
                ext = os.path.splitext(fn)[1].lower()
                if not os.path.isfile(fp):
                    continue
                if ext not in config.IMAGE_EXTS:
                    skipped.append(fp)          # catches the 24 per-folder labels.csv
                    continue
                rel = os.path.relpath(fp, config.REPO_ROOT).replace(os.sep, "/")
                rows.append({
                    "split": split,
                    "sample_id": sample,
                    "sample_slug": _slug(sample),
                    "filename": fn,
                    "source_path": rel,
                    "abs_path": fp,
                    "bytes": os.path.getsize(fp),
                })
    return rows, skipped


def run(force: bool = False) -> int:
    out = config.stage_path("inventory")
    rows, skipped = enumerate_images()
    if not rows:
        print("no images found -- aborting")
        return 1

    recs = []
    for r in rows:
        recs.append({**r, "src_sha256": _sha256(r["abs_path"])})
    df = pd.DataFrame(recs)
    df["image_id"] = df["source_path"].map(
        lambda s: hashlib.sha256(s.encode()).hexdigest()[:12])

    n_train = int((df.split == "train").sum())
    n_test = int((df.split == "test").sum())
    dup = df[df.duplicated("src_sha256", keep=False)]

    baseline = {
        "n_images": len(df),
        "n_train": n_train,
        "n_test": n_test,
        "n_samples_train": int(df[df.split == "train"].sample_id.nunique()),
        "n_samples_test": int(df[df.split == "test"].sample_id.nunique()),
        "total_bytes": int(df.bytes.sum()),
        "per_directory": {
            d: {"count": int((df.source_path.str.startswith(d)).sum())}
            for d in [os.path.relpath(v, config.REPO_ROOT).replace(os.sep, "/")
                      for v in config.SOURCE_DIRS.values()]
        },
        "files": {r["source_path"]: r["src_sha256"] for r in recs},
        "skipped_non_image": [os.path.relpath(s, config.REPO_ROOT).replace(os.sep, "/")
                             for s in skipped],
    }
    config.assert_writable(os.path.join(config.PROCESSED_DIR, "baseline_sources.json"))
    with open(os.path.join(config.PROCESSED_DIR, "baseline_sources.json"), "w",
              encoding="utf-8") as fh:
        json.dump(baseline, fh, indent=1, sort_keys=True)

    keep = ["image_id", "split", "sample_id", "sample_slug", "filename",
            "source_path", "bytes", "src_sha256"]
    df[keep].to_csv(out, index=False)

    print(f"images: {len(df)} (train {n_train} / test {n_test})")
    print(f"samples: train {baseline['n_samples_train']} / test {baseline['n_samples_test']}")
    print(f"total bytes: {baseline['total_bytes'] / 1e6:.1f} MB")
    print(f"non-image files skipped: {len(skipped)} "
          f"(expected 24 labels.csv: {len(skipped) == 24})")
    print(f"duplicate-content groups: {dup.src_sha256.nunique()} (expected 0)")
    print(f"unique image_id: {df.image_id.nunique()} (expected {len(df)})")
    return 0
