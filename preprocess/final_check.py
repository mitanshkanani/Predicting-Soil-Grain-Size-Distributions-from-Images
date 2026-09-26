"""Read-only final confirmation: hashes agree, checks pass, counts are consistent.

Run with:  python -m preprocess.final_check
Exits non-zero if any invariant is violated. Touches nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pandas as pd

from . import config


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    pm = config.PROCESSED_DIR
    problems: list[str] = []
    lines: list[str] = []

    def note(s=""):
        lines.append(s)
        print(s)

    note("=" * 72)
    note("SOIL-GSD PREPROCESSING -- FINAL VERIFICATION")
    note("=" * 72)

    # ---- 1. config_hash agreement across the four artifacts -----------------
    note("\n[1] config_hash agreement")
    found: dict[str, str | None] = {}
    tp = os.path.join(pm, "target_ppm.json")
    found["target_ppm.json"] = json.load(open(tp, encoding="utf-8")).get("config_hash") \
        if os.path.exists(tp) else None
    mi = os.path.join(pm, "manifest_images.csv")
    if os.path.exists(mi):
        d = pd.read_csv(mi)
        vals = set(d.config_hash.astype(str))
        found["manifest_images.csv"] = next(iter(vals)) if len(vals) == 1 else f"MULTIPLE:{vals}"
    else:
        found["manifest_images.csv"] = None
    aj = os.path.join(pm, "audit.json")
    found["audit.json"] = json.load(open(aj, encoding="utf-8")).get("config_hash") \
        if os.path.exists(aj) else None
    gj = os.path.join(pm, "golden_checks.json")
    found["golden_checks.json"] = json.load(open(gj, encoding="utf-8")).get("config_hash") \
        if os.path.exists(gj) else None

    live = config.config_hash()
    for name, h in found.items():
        ok = h == live
        note(f"  {'OK ' if ok else 'BAD'}  {name:22s} {h}")
        if not ok:
            problems.append(f"{name} config_hash {h!r} != live {live!r}")
    note(f"  ---  {'live config.config_hash()':22s} {live}")

    # ---- 2. audit and verify outcomes --------------------------------------
    note("\n[2] stage outcomes")
    if os.path.exists(aj):
        a = json.load(open(aj, encoding="utf-8"))
        n = len(a["checks"])
        failed = a.get("failed", [])
        note(f"  audit: {n - len(failed)}/{n} assertions passed")
        if failed:
            problems.append(f"audit failed: {failed}")
    else:
        problems.append("audit.json missing")

    from .run import STAGES
    state = json.load(open(config.state_path(), encoding="utf-8")) \
        if os.path.exists(config.state_path()) else {}
    absent = [s for s in STAGES if s not in state]
    note(f"  run_state stages recorded: {len(state)}/{len(STAGES)}"
         + (f"  MISSING: {absent}" if absent else ""))
    if absent:
        problems.append(f"stages with no run_state record: {absent}")
    if state:
        hashes = {v.get("config_hash") for v in state.values()}
        if len(hashes) != 1 or live not in hashes:
            problems.append(f"run_state has mixed hashes: {hashes}")
        else:
            note(f"  OK   all recorded stages share {live}")

    # ---- 3. counts ----------------------------------------------------------
    note("\n[3] counts")
    base = json.load(open(os.path.join(pm, "baseline_sources.json"), encoding="utf-8"))
    mismatch = [p for p, d in base["files"].items() if _sha(p) != d]
    note(f"  source images under data/Training + data/Test: {len(base['files'])}")
    note(f"  source sha256 mismatches vs baseline:          {len(mismatch)}")
    if mismatch:
        problems.append(f"{len(mismatch)} source images changed: {mismatch[:3]}")

    if os.path.exists(mi):
        d = pd.read_csv(mi)
        note(f"  manifest image rows: {len(d)}")
        n_png = 0
        for root in config.OUT_DIRS.values():
            for _dp, _d, fs in os.walk(root):
                n_png += sum(1 for f in fs if f.endswith(".png"))
        note(f"  canonical PNGs written: {n_png}")
        if n_png != len(d):
            problems.append(f"{n_png} canonical PNGs vs {len(d)} manifest rows")

    ms = pd.read_csv(os.path.join(pm, "manifest_samples.csv"))
    ntr = int((ms.split == "train").sum())
    nte = int((ms.split == "test").sum())
    note(f"  soil samples: {ntr} train / {nte} test")
    if (ntr, nte) != (24, 10):
        problems.append(f"sample counts {ntr}/{nte} != 24/10")
    if sorted(ms[ms.split == "test"].sort_values("submission_row_order").submission_id) != \
            sorted(pd.read_csv(config.SUBMISSION_CSV).sample_id.astype(str)):
        problems.append("submission ids do not match sample_submission.csv")

    mt = pd.read_csv(os.path.join(pm, "manifest_tiles.csv"))
    note(f"  tile manifest rows: {len(mt)}")
    for size, cnt in mt.groupby("tile_size_px").size().items():
        note(f"    {int(size):3d}px ({int(size) / float(mt.normalized_ppm.iloc[0]):5.2f}mm): {int(cnt):6d} tiles")
    n_tile_files = sum(len([f for f in fs if f.endswith('.png')])
                       for _dp, _d, fs in os.walk(os.path.join(config.DATA_DIR, "tiles")))
    note(f"  materialised tile files on disk: {n_tile_files}")
    intended = int(mt.materialized.sum())
    if n_tile_files != intended:
        problems.append(f"{n_tile_files} tile files on disk vs {intended} in manifest")

    note(f"  canonical target PPM: {float(json.load(open(tp, encoding='utf-8'))['target_ppm']):.6f} px/mm")
    if os.path.exists(mi):
        dirs = pd.read_csv(mi).resample_direction.value_counts().to_dict()
        note(f"  resample directions:  {dirs}")
        if set(dirs) - {"down", "identity"}:
            problems.append(f"upscale present: {dirs}")

    # ---- verdict ------------------------------------------------------------
    note("\n" + "=" * 72)
    if problems:
        note("FINAL CHECK: FAILED")
        for p in problems:
            note("  - " + p)
        return 1
    note("FINAL CHECK: PASSED -- all invariants hold")
    note("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
