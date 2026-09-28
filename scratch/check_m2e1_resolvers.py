"""Proves cell A3's input discovery works on Kaggle's mount layout, not just the local one.

Why this exists: the notebook locates `data/processed_meta/manifest_images.csv`,
`backbones.py` and Experiment 3's artifacts by searching. Kaggle's mount path has been
observed at two different depths (`/kaggle/input/<slug>` and
`/kaggle/input/datasets/<user>/<slug>`), and a resolver that only checks one level fails
silently on the other - which would cost an entire session at the worst possible moment.

This runs the ACTUAL cell A3 source, extracted from the generated notebook, inside a fake
`/kaggle/input` tree nested three deep. Nothing is copied or reimplemented: if the notebook's
cell changes, this test changes with it.

Run:  python scratch/check_m2e1_resolvers.py
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
NB = REPO / "Model 2/Model 2 Experiment 1/Model2_Experiment1.ipynb"
DEEP = pathlib.Path("datasets") / "mitanshkanani" / "soil-gsd-processed-final"


def cell_source(marker: str) -> str:
    """The code cell containing `marker`, exactly as it will run on Kaggle."""
    nb = json.loads(NB.read_text(encoding="utf-8"))
    hits = [c for c in nb["cells"]
            if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, f"expected exactly one cell containing {marker!r}, got {len(hits)}"
    return "".join(hits[0]["source"])


def build_tree(root: pathlib.Path) -> pathlib.Path:
    """A dataset mounted DEEP under the fake /kaggle/input, with only what A3 needs."""
    ds = root / DEEP
    (ds / "data" / "processed_meta").mkdir(parents=True)
    for f in ["manifest_images.csv", "manifest_tiles.csv", "manifest_samples.csv",
              "audit.json", "golden_checks.json"]:
        shutil.copy(REPO / "data" / "processed_meta" / f, ds / "data" / "processed_meta" / f)
    shutil.copy(REPO / "data" / "sample_submission.csv", ds / "data" / "sample_submission.csv")
    shutil.copytree(REPO / "preprocess", ds / "preprocess")
    m2 = REPO / "Model 2/Model 2 Experiment 1"
    for f in ["backbones.py", "check_backbones.py"]:
        shutil.copy(m2 / f, ds / f)
    e3 = REPO / "Model 1/Model 1 Experiment 3"
    dst = ds / "Model 1 Experiment 3"
    dst.mkdir()
    for f in ["features_soil.csv", "cv_families.csv", "Submission_Model1_E3.csv"]:
        shutil.copy(e3 / f, dst / f)
    return ds


MAP: dict = {}


def fake_path(*args, **kw):
    """Stands in for `pathlib.Path` inside the cell, redirecting only the literal
    '/kaggle/input'. WindowsPath cannot be subclassed, so this wraps rather than inherits -
    the cell only ever constructs paths and calls exists/iterdir/rglob/parent on them."""
    a = list(args)
    if a and isinstance(a[0], str) and a[0] in MAP:
        a[0] = str(MAP[a[0]])
    return pathlib.Path(*a, **kw)


fake_path.cwd = staticmethod(pathlib.Path.cwd)


def run_a3(tmp: pathlib.Path) -> dict:
    src = cell_source("def resolve_input_root")
    ns = {"Path": fake_path, "sys": __import__("sys")}
    MAP.clear()
    MAP["/kaggle/input"] = tmp
    exec(compile(src, "cell A3", "exec"), ns)          # noqa: S102 - the shipped cell, as-is
    return ns


def main() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="m2mount_"))
    syspath_backup = list(__import__("sys").path)
    try:
        ds = build_tree(tmp)
        ns = run_a3(tmp)
        ok = True
        got_root = str(ns["INPUT_ROOT"])
        got_bb = str(ns["BB_DIR"])
        print(f"fake mount : {tmp / DEEP}")
        print(f"INPUT_ROOT : {got_root}")
        print(f"BB_DIR     : {got_bb}")
        print(f"context    : {ns['RUN_CONTEXT']}")
        for label, got in [("INPUT_ROOT", got_root), ("BB_DIR", got_bb)]:
            good = pathlib.Path(got).resolve() == ds.resolve()
            print(f"  [{'PASS' if good else 'FAIL'}] {label} resolved through 3 levels "
                  f"of Kaggle nesting to the dataset root")
            ok &= good
        ok &= ns["RUN_CONTEXT"] == "kaggle"
        print(f"  [{'PASS' if ns['RUN_CONTEXT'] == 'kaggle' else 'FAIL'}] recognised as a "
              "kaggle context, not the local fallback")
        pv = ns.get("pv")
        good = pv is not None and hasattr(pv, "feats")
        print(f"  [{'PASS' if good else 'FAIL'}] `from preprocess import verify` worked off "
              "the resolved root")
        ok &= good
        good = ns.get("bb") is not None and ns["bb"].BACKENDS == ("mock", "vit_random", "dinov2")
        print(f"  [{'PASS' if good else 'FAIL'}] `import backbones` worked and exposes the "
              "three arms")
        ok &= good
        print("\n" + ("RESOLVER CHECK: PASSED" if ok else "RESOLVER CHECK: FAILED"))
        raise SystemExit(0 if ok else 1)
    finally:
        __import__("sys").path[:] = syspath_backup
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
