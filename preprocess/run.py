"""CLI: python -m preprocess --stage geometry   (or --all, --dry-run)

Stages run in dependency order and each persists its own CSV, so any stage can
be re-run standalone against a stale tree -- which is what turns verify.py's
golden checks into regression tests.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from . import config

STAGES = [
    "inventory",
    "cameras",
    "geometry",
    "resample",     # exif-transpose -> P3->sRGB -> canonical grid -> PNG
    "soilmask",     # inscribed crop rect + soil_fraction
    "colour",       # gray-world gains, computed on the cropped soil region only
    "tiling",
    "manifests",
    "audit",
    "qc",
    "verify",
]

# Stages that only read source data; used by --dry-run.
METADATA_STAGES = {"inventory", "cameras", "geometry"}

# CLI stage name -> module name, where they differ.
MODULE_MAP = {"manifests": "manifest", "qc": "qc_sheets"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m preprocess")
    ap.add_argument("--stage", choices=STAGES, nargs="+", action="extend",
                    default=[], help="repeatable; also accepts several names at once")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="metadata-only stages; writes no pixels")
    ap.add_argument("--force", action="store_true",
                    help="re-run a stage even if its output already exists")
    args = ap.parse_args(argv)

    config.ensure_dirs()
    if config.load_target_ppm() is not None:
        from .geometry import restore_target_ppm
        restore_target_ppm()

    if args.dry_run:
        stages = sorted(METADATA_STAGES, key=STAGES.index)
    elif args.all:
        stages = STAGES
    elif args.stage:
        stages = sorted(args.stage, key=STAGES.index)
    else:
        ap.error("choose --all, --dry-run, or --stage NAME [NAME ...]")

    # geometry must precede anything that calls config.target_ppm()
    need = set(stages)
    if need - {"inventory", "cameras", "geometry"} and "geometry" not in need:
        if config.load_target_ppm() is None:
            stages = sorted(set(stages) | {"geometry"}, key=STAGES.index)

    # Pull in any stage whose recorded config_hash differs from the current one,
    # including upstream stages that produce pixels. Without this, editing a
    # constant and re-running `--stage manifests audit` would regenerate the
    # metadata over stale PNGs and report a clean audit over a corpus that no
    # longer matches it.
    last = max(STAGES.index(s) for s in stages)
    stale = [s for s in STAGES[:last + 1] if config.stage_is_stale(s, STAGES)]
    if stale:
        extra = [s for s in stale if s not in stages]
        if extra:
            print(f"config/pipeline changed -- also rebuilding upstream: {extra}")
        stages = sorted(set(stages) | set(stale), key=STAGES.index)

    failures: list[str] = []
    for name in stages:
        mod = importlib.import_module(
            f".{MODULE_MAP.get(name, name)}", __package__)
        t0 = time.time()
        force_this = args.force or name in stale
        print(f"\n=== {name} " + "=" * max(0, 58 - len(name)))
        if name in stale and not args.force:
            print("  (rebuilding: built under a different config)")
        try:
            rc = mod.run(force=force_this)
        except Exception as exc:  # surface the real error, keep going for the report
            import traceback
            traceback.print_exc()
            rc = 1
            failures.append(f"{name}: {exc}")
        dt = time.time() - t0
        if rc:
            failures.append(f"{name}: exit {rc}")
            print(f"--- {name} FAILED in {dt:.1f}s")
            break            # downstream would only compound the error
        config.record_stage_done(name)
        print(f"--- {name} ok in {dt:.1f}s")

    if failures:
        print("\n!!! pipeline failures:")
        for f in failures:
            print("   ", f)
        return 1
    print("\npipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
