"""Probe: does the new publish() frame guard have teeth, and does the old wiring pass?

Throwaway. Builds a tiny in-memory assembly (three tiles, one image, one soil) whose image frame
carries a NaN while its aggregate is perfectly finite, then runs it through the real publish() and
through the pre-delta wiring (aggregate-only guard) in the same process. Nothing in the experiment
folder is written: both publishes are aimed at a scratch probe directory that is deleted here.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APPROACH = Path("Model 7/Approach A")
sys.path.insert(0, str(APPROACH.resolve()))
sys.path.insert(0, str((APPROACH / "Model 5" / "Model 5 Experiment 1").resolve()))
import m7a_data as d  # noqa: E402
import spatial_features as sf  # noqa: E402

KEYS = list(sf.SPATIAL)
tile_rows = [{"A1": -0.2, "A2": 1.0, "A3": 0.05, "A4": 0.001},
             {"A1": -0.3, "A2": 1.1, "A3": 0.06, "A4": -0.002}]
tf = pd.DataFrame(tile_rows, columns=KEYS)
img = pd.DataFrame([[-0.25, float("nan"), 0.055, -0.0005]], columns=KEYS,
                   index=pd.Index(["i1.png"], name="parent_image_path"))
soil = pd.DataFrame([[-0.25, 1.05, 0.055, -0.0005]], columns=KEYS,
                    index=pd.Index(["S1"], name="sample_id"))
asm = {"tile_frame": tf, "image": img, "soil": soil, "q": pd.DataFrame({"split": ["train"]}),
       "records": [{"values": r, "status": "ok", "nan_keys": [], "reason": "", "tile_path": "t%d" % i}
                   for i, r in enumerate(tile_rows)],
       "errors": [], "warnings": [], "status_counts": {"ok": 2}, "hashes": ("a", "b", "c"),
       "scrambled": False, "seed": 90001, "degenerate_examples": [], "aborted": False}
clean_cen = {"per_key": {"A2": {"median": 1.05}}, "note": "the aggregate shows nothing"}
print("aggregate's own non-finite paths:", d.find_nonfinite(clean_cen))

probe_new = APPROACH / "scratch" / "_probe_new_guard"
probe_old = APPROACH / "scratch" / "_probe_old_guard"
try:
    try:
        d.publish(asm, clean_cen, out_dir=probe_new, write=True)
        print("CURRENT publish(): wrote %s" % sorted(p.name for p in probe_new.iterdir()))
    except (ValueError, AssertionError) as exc:
        print("CURRENT publish(): REFUSED -> %s" % str(exc)[:110])
    print("  directory created? %s" % probe_new.exists())

    real_guard = d.require_publishable

    def aggregate_only(cen, asm=None):
        """The pre-delta wiring: the report is checked, the frames are not."""
        return real_guard(cen, None)

    d.require_publishable = aggregate_only
    try:
        try:
            d.publish(asm, clean_cen, out_dir=probe_old, write=True)
            wrote = sorted(p.name for p in probe_old.iterdir()) if probe_old.exists() else []
            print("OLD wiring publish(): wrote %s" % wrote)
            printed = pd.read_csv(probe_old / d.IMG_CSV_NAME, index_col=0).to_string()
            print("  and the NaN is now on disk:\n%s" % printed)
        except (ValueError, AssertionError) as exc:
            print("OLD wiring publish(): REFUSED -> %s" % str(exc)[:110])
    finally:
        d.require_publishable = real_guard
finally:
    for p in (probe_new, probe_old):
        if p.exists():
            shutil.rmtree(p)
    print("probe directories removed: %s %s" % (not probe_new.exists(), not probe_old.exists()))
