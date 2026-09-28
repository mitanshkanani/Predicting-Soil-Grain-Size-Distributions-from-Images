"""Proves Model 2 E2's cell A3 and C1 find their inputs on a Kaggle-shaped mount.

Why this exists: E2 shipped once and died in cell A3 with "Could not find
data/processed_meta/manifest_images.csv" while the dataset was plainly attached and
correct. Kaggle mounts at `/kaggle/input/datasets/<user>/<slug>` - three levels down - and
this notebook's first resolver only searched two. E1's resolver had a deep fallback; the
E2 rewrite silently dropped it. A local run can never catch that, because locally the
repository root IS the data root.

So this test runs the ACTUAL cell source, extracted from the generated notebook, against a
fake `/kaggle/input` with the same depth Kaggle uses. If a future edit narrows the search
again, this fails here instead of costing a session.

Run:  python scratch/check_m2e2_resolvers.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile
import types

REPO = pathlib.Path(__file__).resolve().parent.parent
NB = REPO / "Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb"
DEEP = pathlib.Path("datasets") / "mitanshkanani" / "final_kaggle_upload_m2e2"
MOUNTED = ("Model 1 Experiment 3", "Model 2 Experiment 1")

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def cell_source(marker):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"]
            if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, f"expected one cell containing {marker!r}, found {len(hits)}"
    return "".join(hits[0]["source"])


MAP: dict = {}
CWD: dict = {}          # override for the negative test: the repo root genuinely HAS data/


def fake_path(*args, **kw):
    a = list(args)
    if a and isinstance(a[0], str) and a[0] in MAP:
        a[0] = str(MAP[a[0]])
    return pathlib.Path(*a, **kw)


def _cwd():
    return CWD.get("here", pathlib.Path.cwd())


fake_path.cwd = staticmethod(_cwd)


def build_tree(tmp: pathlib.Path) -> pathlib.Path:
    """A dataset mounted exactly as Kaggle mounts it, with the files the resolvers need."""
    ds = tmp / DEEP
    (ds / "data" / "processed_meta").mkdir(parents=True)
    meta = REPO / "data" / "processed_meta"
    for f in ["manifest_images.csv", "manifest_tiles.csv", "manifest_samples.csv",
              "audit.json", "golden_checks.json", "target_ppm.json"]:
        shutil.copy(meta / f, ds / "data" / "processed_meta" / f)
    shutil.copy(REPO / "data" / "sample_submission.csv", ds / "data" / "sample_submission.csv")
    shutil.copytree(REPO / "preprocess", ds / "preprocess")
    m2e1 = REPO / "Model 2/Model 2 Experiment 1"
    m2e2 = REPO / "Model 2/Model 2 Experiment 2"
    shutil.copy(m2e1 / "backbones.py", ds / "backbones.py")
    shutil.copy(m2e1 / "check_backbones.py", ds / "check_backbones.py")
    shutil.copy(m2e2 / "crop_geometry.py", ds / "crop_geometry.py")
    shutil.copy(m2e2 / "check_crop_geometry.py", ds / "check_crop_geometry.py")
    # the two prior-experiment folders, as the zip ships them
    e3 = ds / "Model 1 Experiment 3"
    e3.mkdir()
    for f in ["features_soil.csv", "cv_families.csv", "Submission_Model1_E3.csv"]:
        shutil.copy(REPO / "Model 1/Model 1 Experiment 3" / f, e3 / f)
    e1 = ds / "Model 2 Experiment 1"
    e1.mkdir()
    # Copy E1's real outputs only where they exist. E1's per-soil file was lost locally, so
    # the fixture must not fabricate one: the optional-E1 branch is the state Kaggle will
    # actually be in, and it has to be exercised as such.
    for f in ["Experiment1.txt", "cv_per_soil.csv", "instructions.txt",
              "Submission_Model2_E1.csv"]:
        src = m2e1 / f
        if src.exists():
            shutil.copy(src, e1 / f)
    return ds


def run_cells(tmp: pathlib.Path) -> dict:
    """Exec the real cell chain A1 -> A2 -> A3 -> C1, in order, against the fake mount.

    Running A3 alone would hide exactly the class of bug this file exists to catch: a cell
    that works only because a previous cell defined something.
    """
    ns_name = "__m2e2_resolver_test__"
    mod = sys.modules.get(ns_name)
    if mod is None:
        mod = types.ModuleType(ns_name)
        sys.modules[ns_name] = mod
    ns = mod.__dict__
    ns["__name__"] = ns_name
    # @dataclass resolves annotations via sys.modules[cls.__module__], so the namespace has
    # to be a real registered module or cell A1's Config raises AttributeError on import.
    MAP.clear()
    MAP["/kaggle/input"] = tmp
    exec(compile(cell_source("Cell A1 - configuration"), "A1", "exec"), ns)
    exec(compile(cell_source("Cell A2 - imports"), "A2", "exec"), ns)
    # A1 rebinds Path from pathlib, so the mount redirect is installed after it and before
    # the cells under test. ON_KAGGLE is forced True because A1 computed it from the real
    # filesystem, which has no /kaggle - the point of the test is A3's search, and it must
    # run the way it runs on Kaggle.
    ns["Path"] = fake_path
    ns["ON_KAGGLE"] = True
    for marker in ["def resolve_input_root", "Cell A4 - pin the preprocessing version",
                   "PLUMBING = ("]:
        exec(compile(cell_source(marker), marker, "exec"), ns)
    return ns


def main() -> None:
    import pandas as pd     # noqa: F401 - the cells need it in the namespace
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="m2e2mount_"))
    syspath_backup = list(sys.path)
    try:
        ds = build_tree(tmp)
        print(f"fake mount : {tmp / DEEP}")
        ns = run_cells(tmp)
        got = pathlib.Path(ns["INPUT_ROOT"]).resolve()
        check("INPUT_ROOT resolved through 3 levels of Kaggle nesting",
              got == ds.resolve(), str(got))
        check("RUN_CONTEXT says kaggle, not the local fallback",
              ns["RUN_CONTEXT"] == "kaggle", ns["RUN_CONTEXT"])
        bbp = pathlib.Path(ns["bb"].__file__).resolve()
        check("backbones.py found at the dataset root, not a nested copy",
              bbp == (ds / "backbones.py").resolve(), str(bbp))
        cgp = pathlib.Path(ns["cg"].__file__).resolve()
        check("crop_geometry.py found", cgp == (ds / "crop_geometry.py").resolve(), str(cgp))
        check("preprocess imported off the resolved root", hasattr(ns.get("pv"), "feats"))
        e3 = pathlib.Path(ns["E3_DIR"]).resolve()
        check("E3 artifacts found by content signature",
              e3 == (ds / "Model 1 Experiment 3").resolve(), str(e3))
        e1 = ns.get("E1_DIR")
        check("E1 dir resolved without raising even though its files may be thin",
              e1 is None or pathlib.Path(e1).is_dir(), str(e1))
        check("config_hash pinned from the mounted manifests",
              ns["CONFIG_HASH"] == "010f44c36c74", ns["CONFIG_HASH"])

        print("\nshallow mount (/kaggle/input/<slug>), the other Kaggle layout")
        tmp2 = pathlib.Path(tempfile.mkdtemp(prefix="m2e2flat_"))
        try:
            flat = tmp2 / "my-slug"
            shutil.copytree(ds, flat)
            MAP.clear(); MAP["/kaggle/input"] = tmp2
            sys.path[:] = syspath_backup
            ns2 = run_cells(tmp2)
            check("the same resolver also handles the 1-level mount",
                  pathlib.Path(ns2["INPUT_ROOT"]).resolve() == flat.resolve(),
                  str(ns2["INPUT_ROOT"]))
            shutil.rmtree(tmp2 / "my-slug" / "data", ignore_errors=True)
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("\nnothing attached -> must fail loudly, not silently fall back")
        tmp3 = pathlib.Path(tempfile.mkdtemp(prefix="m2e2none_"))
        try:
            (tmp3 / "unrelated").mkdir()
            MAP.clear(); MAP["/kaggle/input"] = tmp3
            CWD["here"] = tmp3          # else the repo root's own data/ satisfies the probe
            sys.path[:] = syspath_backup
            try:
                run_cells(tmp3)
                check("raises when the data is genuinely absent", False, "it returned")
            except RuntimeError as exc:
                check("raises RuntimeError naming the missing file",
                      "manifest_images.csv" in str(exc), str(exc).splitlines()[0][:70])
        finally:
            CWD.pop("here", None)
            shutil.rmtree(tmp3, ignore_errors=True)
    finally:
        sys.path[:] = syspath_backup
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    if FAILS:
        print(f"RESOLVER CHECK: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("RESOLVER CHECK: PASSED")
    print("""
Scope: this runs the shipped cell A3 and C1 source verbatim against a fabricated Kaggle
mount. It proves the SEARCH works at both observed depths and refuses when the data is
truly absent. It does not prove anything about the science.""")


if __name__ == "__main__":
    main()
