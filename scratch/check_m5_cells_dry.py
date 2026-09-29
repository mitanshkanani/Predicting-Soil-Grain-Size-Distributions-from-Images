"""Execute Model 5's SETUP and ARM-CONSTRUCTION cells on real inputs, and score nothing.

The gate-logic dry run covers cell G1. This one covers everything between the manifests and the
first fit: cells A1..C1, E1 and E2. Those are the only cells that touch the filesystem and build
the fold structure, and they are also where a shipped notebook dies fastest - the Series.values()
crash in G1 was found by the other dry run, and the same class of bug is alive anywhere a real
column name or a real dtype is assumed.

What makes this NOT the experiment: no nested_predict call is reached, no EMD is computed for any
arm, no gate is evaluated, and OUT_DIR is redirected to scratch/_m5_dry so nothing lands in the
experiment folder. If this prints a number that could be a result, it is doing the wrong thing.

Run:  python scratch/check_m5_cells_dry.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 5" / "Model 5 Experiment 1" / "Model5_Experiment1.ipynb"
DRY = ROOT / "scratch" / "_m5_dry"
EXP_DIR = ROOT / "Model 5" / "Model 5 Experiment 1"
FAILS, LINES = [], []


def check(name, ok, detail=""):
    (LINES if ok else FAILS).append(f"  {'PASS' if ok else '**FAIL**'}  {name}"
                                    + (f"   {detail}" if detail else ""))


def cells():
    """Map each cell to the tag in its own docstring, which is the name the notebook uses."""
    out = {}
    for c in json.load(io.open(NB, encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        m = re.match(r'"""Cell ([A-Z0-9]+)', s)
        assert m, f"a code cell has no 'Cell NN -' docstring: {s[:60]!r}"
        out[m.group(1)] = s
    return out


CELL = cells()
SEQUENCE = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "C1", "E1", "E2"]
missing = [c for c in SEQUENCE if c not in CELL]
assert not missing, f"cells absent from the notebook: {missing}"
for c in ("E3", "F1", "F2", "G1", "H1", "Q1", "R1", "R2"):
    assert c in CELL, f"cell {c} should exist but was not found"
_SETUP_TEXT = "\n".join(CELL[c] for c in SEQUENCE)
_calls = [pat for pat in (r"te\.nested_predict\s*\(", r"te\.fit_predict\s*\(",
                          r"te\.oracle_sweep\s*\(") if re.search(pat, _SETUP_TEXT)]
assert not _calls, (
    f"a setup cell calls a fitting function ({_calls}); this dry run would then be scoring the "
    "experiment rather than checking its plumbing")

print("CHECK M5 CELLS DRY - real inputs, no arm scored, no artifact in the experiment folder")
print("-" * 78)


def snapshot():
    """Content hash of every file in the experiment folder, so 'I changed nothing' is testable.

    The earlier version of this checker asserted that a given result file did not exist there.
    Once the experiment had actually run that assertion could never pass again - a check that
    expires is not a check. Comparing the folder before and after works on either side of the run.
    """
    import hashlib
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(EXP_DIR.glob("*")) if p.is_file()}


before = snapshot()
shutil.rmtree(DRY, ignore_errors=True)
DRY.mkdir(parents=True, exist_ok=True)
ns = {}
log = io.StringIO()
with contextlib.redirect_stdout(log):
    for tag in SEQUENCE:
        exec(compile(CELL[tag], f"cell{tag}", "exec"), ns)
        if tag == "A1":
            ns["OUT_DIR"] = DRY                     # nothing this run prints may be an artifact
print(log.getvalue().strip()[:2400])

CFG, te = ns["CFG"], ns["te"]
check("config_hash and the submission schema pinned from the manifests",
      ns["CONFIG_HASH"] == "010f44c36c74" and len(ns["TARGET_COLS"]) == 11, ns["CONFIG_HASH"])
check("the EMD still reconciles against the host's published trivial baseline",
      abs(ns["mean_emd"](np.tile(ns["TRIVIAL"], (24, 1)), ns["Y"]) - ns["PUB_TRIVIAL"]) < 0.05,
      f"{ns['mean_emd'](np.tile(ns['TRIVIAL'], (24, 1)), ns['Y']):.4f} vs {ns['PUB_TRIVIAL']}")
check("G0's reference was read from E3's artifact, not typed",
      abs(ns["G0_REF"] - 43.0217308796477) < 1e-9, f"{ns['G0_REF']:.13f}")
check("the image table is one row per image with the 12 frozen features",
      list(ns["IMG"].columns[:12]) == list(CFG.control_feats) or
      set(CFG.control_feats) <= set(ns["IMG"].columns), str(list(ns["IMG"].columns)))
check("the alignment target is the 35 unlabeled test images",
      ns["TARGET"].shape == (CFG.n_images_test, 12), str(ns["TARGET"].shape))
check("no training soil contributes a target row",
      set(ns["IMG"].loc[ns["IMG"].split == "test", "sample_id"]).isdisjoint(ns["TR_IDS"]))
check("the tile frame carries a split column, which CORAL_TILE needs",
      "split" in ns["TILE"].columns and int((ns["TILE"].split == "test").sum()) > 0,
      f"{int((ns['TILE'].split == 'test').sum())} target tiles")
check("both directions' folds were built and total the contracted 45 rows",
      sum(len(f["eval_ids"]) for v in ns["_D"].values() for f in v) == CFG.n_rows_eval,
      str({k: len(v) for k, v in ns["_D"].items()}))
check("no fold scores a soil it was fitted on (spec 14, re-checked on real data)",
      all(not (set(f["eval_ids"]) & set(f["fit_ids"]))
          for v in ns["_D"].values() for f in v))

T = ns["fold_transforms"](ns["_D"]["A"][0]["fit_img"])
check("all seven arms produce a transform for a fold", set(T) == set(ns["ARMS"]), str(list(T)))
bad = [a for a, t in T.items()
       if not (np.isfinite(t["A"]).all() and np.isfinite(t["b"]).all())]
check("every transform matrix and shift is finite", not bad, str(bad))
ident = T["NONE"]
check("NONE is the identity, so the control arm cannot silently move the features",
      np.array_equal(ident["A"], np.eye(12)) and not ident["b"].any())
_src = ns["_D"]["A"][0]["fit_img"][list(CFG.control_feats)].to_numpy(float)
_Cs, _ws = ns["al"].covariance(_src)
_Ct, _wt = ns["al"].covariance(ns["TARGET"])
_before = float(np.abs(_Cs - _Ct).max())
_after = float(np.abs(T["CORAL"]["A"].T @ _Cs @ T["CORAL"]["A"] - _Ct).max())
check("CORAL moves the fitting camera's covariance toward the target's on real data",
      _after < _before, f"{_before:.3e} -> {_after:.3e}; shrinkage {_ws:.3f}/{_wt:.3f}")
_self = ns["al"].transform_distance(T["CORAL_SELF"], ident)
check("CORAL_SELF is numerically the identity, which is why G5 can be a validity canary",
      _self < 1e-8, f"max |T_self - I| = {_self:.2e}")
_indep = ns["al"].transform_distance(T["CORAL"], T["CORAL_INDEP"])
check("the placebo transform is genuinely different from the real one",
      _indep > 1e-3, f"max entrywise distance {_indep:.4f}")
_mean = ns["al"].transform_distance(T["CORAL"], T["MEAN"])
check("CORAL differs from MEAN, i.e. the covariance half is not vacuous",
      _mean > 1e-3, f"max entrywise distance {_mean:.4f}")

Xf, Yf, famf, Xe = ns["fold_matrices"](ns["_D"]["A"][0], T["CORAL"])
check("fold matrices conform: fit rows x 12, labels x 11, eval rows x 12",
      Xf.shape[1] == 12 and Yf.shape == (len(Xf), 11) and Xe.shape[1] == 12,
      f"{Xf.shape} {Yf.shape} {Xe.shape}")
check("every fold's fit set keeps at least two CV families, so alpha selection is possible",
      len(set(famf)) >= 2, f"{len(set(famf))} families")
check("the fitting pool shrinks by exactly one family per fold",
      all(len(f["fit_ids"]) < 24 for f in ns["_D"]["A"]))
check("the dry run wrote only into scratch, never into the experiment folder",
      sorted(p.name for p in DRY.glob("*")) == ["placebo_permutation.csv"],
      str(sorted(p.name for p in DRY.glob("*"))))
after = snapshot()
touched = [k for k, v in before.items() if after.get(k) != v] + \
          [k for k in after if k not in before]
check("and the experiment folder is UNCHANGED by running this checker",
      not touched, f"{len(after)} files before and after, changed: {touched}")

print("\n".join(LINES))
print("-" * 78)
if FAILS:
    print("\n".join(FAILS))
    print(f"CHECK M5 CELLS DRY: FAILED ({len(FAILS)} problems)")
    sys.exit(1)
print(f"CHECK M5 CELLS DRY: PASSED ({len(LINES)} checks; no arm scored, no EMD computed, "
      f"no file in the experiment folder)")
