"""check_m10_resolvers.py - exec the REAL notebook cells against fabricated mount trees.

WHY THIS TEST EXISTS
  Model 2 E2 died twice on Kaggle with bugs a local run could never see, because locally the repo
  root IS the data root and every path resolution succeeds by accident. So this test:
    - runs with cwd set to an EMPTY temp directory, so the real repo cannot rescue a broken mount;
    - execs the actual cell source extracted from the shipped .ipynb, not a paraphrase of it;
    - drives the tree from a real extraction of the zip that will be uploaded.
  Scenarios cover the mount depth Kaggle actually uses, the dot-directory a host may not surface,
  an incomplete attachment, and the nothing-attached case.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "final_kaggle_upload_m10e1.zip"
NB = ROOT / "Model 10" / "Experiment 1" / "Model10_Experiment1.ipynb"
TMP = ROOT / "scratch" / "_m10_check"

FAILS = []


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cell_sources():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    out = {}
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        head = src.splitlines()[0]
        for tag in ("A1", "A2", "A3", "B1", "C1", "C2"):
            if "Cell %s " % tag in head:
                out[tag] = src
    missing = {"A1", "A2", "A3", "B1", "C1", "C2"} - set(out)
    if missing:
        raise AssertionError("could not identify cells %r in the shipped notebook" % missing)
    return out


def fresh_dataset(dest: Path, drop_dot_cache: bool = False, drop_one: str | None = None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(dest)
    if drop_dot_cache:
        shutil.rmtree(dest / "Model 1" / "Model 1 Experiment 3" / ".cache")
    if drop_one:
        (dest / drop_one).unlink()
    return dest


def run_in_empty_cwd(src: str, ns: dict, label: str):
    """exec a notebook cell with cwd forced to an empty directory."""
    empty = TMP / "cwd_empty"
    empty.mkdir(parents=True, exist_ok=True)
    for p in empty.iterdir():
        p.unlink() if p.is_file() else shutil.rmtree(p)
    old = Path.cwd()
    os.chdir(empty)
    try:
        exec(compile(src, label, "exec"), ns)
        return None
    except Exception as e:
        return "%s: %s" % (type(e).__name__, e)
    finally:
        os.chdir(old)


def main() -> int:
    if not ZIP.exists() or not NB.exists():
        print("build_m10_kaggle.py has not produced the zip and notebook yet")
        return 1
    S = cell_sources()
    print("identified cells: %s" % ", ".join(sorted(S)))
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True, exist_ok=True)
    a1, a2 = S["A1"], S["A2"]

    # ---- 1: three-deep mount, the layout Kaggle actually uses ------------------
    d = TMP / "input_kaggle" / "datasets" / "mitansh" / "m10_e1_data"
    fresh_dataset(d)
    ns = {"os": os, "Path": Path, "shutil": shutil, "json": json, "__name__": "__main__"}
    os.environ["M10_TEST_INPUT"] = str(TMP / "input_kaggle")
    os.environ["M10_TEST_WORK"] = str(TMP / "work3")
    err = run_in_empty_cwd(a1, ns, "A1-3deep")
    if err or "DATA_ROOT" not in ns:
        FAILS.append("3-deep mount should resolve: %s" % err)
    else:
        print("[ok] 3-deep mount resolved to %s" % Path(ns["DATA_ROOT"]).name)
        err = run_in_empty_cwd(a2, ns, "A2-3deep")
        if err:
            FAILS.append("staging failed on 3-deep mount: %s" % err)
        else:
            run = Path(ns["RUNNER"])
            print("[ok] staged tree: %s" % run)
            r = subprocess.run([sys.executable, str(run / "build_m10e1.py"), "gates"],
                               capture_output=True, text=True, cwd=str(run))
            if r.returncode != 0:
                FAILS.append("gates stage failed in the staged tree: %s" % r.stdout[-600:])
            else:
                print("[ok] build_m10e1.py gates ran in the staged tree (exit 0)")
                for l in r.stdout.splitlines():
                    if "G0a = " in l or "G0b" in l and "->" in l or "G1 = " in l \
                       or "TASK 2 GATES" in l:
                        print("      " + l.strip())

    # ---- 2: one-deep mount ----------------------------------------------------
    d1 = TMP / "input_one" / "m10_e1_data"
    fresh_dataset(d1)
    ns2 = {"os": os, "Path": Path, "shutil": shutil, "json": json}
    os.environ["M10_TEST_INPUT"] = str(TMP / "input_one")
    os.environ["M10_TEST_WORK"] = str(TMP / "work1")
    err = run_in_empty_cwd(a1, ns2, "A1-1deep")
    print("[ok] 1-deep mount resolved" if not err else "[ok] 1-deep skipped: %s" % err)
    if err:
        FAILS.append("1-deep mount should resolve: %s" % err)

    # ---- 3: host did not surface .cache/ -- must restore from _alt/ -----------
    d3 = TMP / "input_dot" / "datasets" / "u" / "m10_e1_data"
    fresh_dataset(d3, drop_dot_cache=True)
    ns3 = {"os": os, "Path": Path, "shutil": shutil, "json": json}
    os.environ["M10_TEST_INPUT"] = str(TMP / "input_dot")
    os.environ["M10_TEST_WORK"] = str(TMP / "workdot")
    err = run_in_empty_cwd(a1, ns3, "A1-dot")
    if err:
        FAILS.append("A1 should accept a tree whose tile table is only in _alt/: %s" % err)
    else:
        err = run_in_empty_cwd(a2, ns3, "A2-dot")
        if err:
            FAILS.append("A2 should restore .cache/ from _alt/: %s" % err)
        else:
            # RUNNER is <work>/Model 10/Experiment 1, so the tree root is two levels up.
            tile = Path(ns3["EXP_ROOT"]) / "Model 1" / "Model 1 Experiment 3" / ".cache" / \
                "tile_features_256_010f44c36c74.csv"
            orig = ROOT / "Model 1" / "Model 1 Experiment 3" / ".cache" / \
                "tile_features_256_010f44c36c74.csv"
            if not tile.exists():
                FAILS.append("restored tile table is absent (looked at %s)" % tile)
            elif sha(tile) != sha(orig):
                FAILS.append("restored tile table is not byte-identical")
            else:
                print("[ok] dot-dir fallback restored the tile table byte-identically")

    # ---- 4: incomplete attachment must be REJECTED, not silently half-used -----
    d4 = TMP / "input_bad" / "datasets" / "u" / "m10_e1_data"
    fresh_dataset(d4, drop_one="Model 2/Model 2 Experiment 3/cv_families.csv")
    ns4 = {"os": os, "Path": Path, "shutil": shutil, "json": json}
    os.environ["M10_TEST_INPUT"] = str(TMP / "input_bad")
    err = run_in_empty_cwd(a1, ns4, "A1-incomplete")
    if err and "COULD NOT LOCATE" in err:
        print("[ok] incomplete attachment rejected loudly")
    else:
        FAILS.append("incomplete attachment should raise COULD NOT LOCATE, got %r" % err)

    # ---- 5: nothing attached at all -------------------------------------------
    empty_base = TMP / "input_empty"
    empty_base.mkdir(parents=True, exist_ok=True)
    ns5 = {"os": os, "Path": Path, "shutil": shutil, "json": json}
    os.environ["M10_TEST_INPUT"] = str(empty_base)
    err = run_in_empty_cwd(a1, ns5, "A1-nothing")
    if err and "COULD NOT LOCATE" in err:
        print("[ok] nothing-attached fails loudly, with the tried bases listed")
        print("      %s" % err[:160])
    else:
        FAILS.append("nothing-attached should fail loudly, got %r" % err)
    os.environ.pop("M10_TEST_INPUT", None)
    os.environ.pop("M10_TEST_WORK", None)

    print("\n" + "=" * 78)
    print("M10 KAGGLE RESOLVER CHECKS: %d failure(s)" % len(FAILS))
    for f in FAILS:
        print("  FAIL: %s" % f)
    print("=" * 78)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
