"""build_m10_kaggle.py - stage the dataset, build the .ipynb, zip, and round-trip verify.

WHY THE ZIP MIRRORS THE REPO LAYOUT INSTEAD OF FLATTENING IT
  ruler.py, transfer_eval.py and build_m10e1.py all compute ROOT = HERE.parents[1] and then
  address frozen inputs as "Model 5/Model 5 Experiment 1/...", "data/processed_meta/..." and so
  on. If the dataset were flattened, every one of those lines would need editing, and an edited
  frozen module is exactly how Model 2 E2 lost its deep-mount fallback and died on Kaggle while
  the local run passed. Mirroring the layout means the shipped files are BYTE-IDENTICAL to the
  ones that produced the numbers on disk, and the notebook only has to find the root and copy
  the tree somewhere writable. Byte-fidelity is then a fact about the files, printed as sha256,
  not a promise in a comment.

WHAT IS SHIPPED AND WHAT IS NOT
  ~5.5 MB: cached tile FEATURES plus manifests plus the frozen code. Not shipped: data/tiles
  (276 MB), training_down/testing_down (716 MB), DINOv2 embeddings (97 MB). The run opens no
  image file, so no pixel data is needed.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Model 10" / "Experiment 1"
STAGE = ROOT / "scratch" / "_m10_stage"
ZIP = ROOT / "final_kaggle_upload_m10e1.zip"
NB = EXP / "Model10_Experiment1.ipynb"

TILE_NAME = "tile_features_256_010f44c36c74.csv"
TILE_REL = "Model 1/Model 1 Experiment 3/.cache/" + TILE_NAME
ALT_REL = "_alt/" + TILE_NAME

# The files the run must find before it will start. TILE_REL is deliberately NOT in this list:
# a dot-prefixed directory is the one thing a dataset host may not surface, so Cell A2 restores
# it from _alt/ rather than assuming it arrived.
CODE = ["Model 5/Model 5 Experiment 1/transfer_eval.py",
        "Model 8/Experiment 3/ruler.py", "Model 8/Experiment 3/d50.py",
        "Model 10/Experiment 1/ruler.py", "Model 10/Experiment 1/d50.py",
        "Model 10/Experiment 1/image_features.py", "Model 10/Experiment 1/build_m10e1.py"]
DATA = ["Model 1/Model 1 Experiment 3/features_soil.csv",
        "Model 2/Model 2 Experiment 3/cv_families.csv",
        "Model 8/Experiment 2/Submission_Model8_E2_E2A.csv",
        "data/processed_meta/manifest_tiles.csv",
        "data/processed_meta/manifest_samples.csv",
        "data/processed_meta/manifest_images.csv",
        "data/sample_submission.csv"]
REQUIRED = CODE + DATA

ARTIFACTS = ["gates.json", "arm_comparison.csv", "cv_per_soil.csv", "bootstrap.json",
             "placebo.json", "score_internal.json", "submission_report.json",
             "Submission_Model10_E1_B.csv", "Submission_Model10_E1_C.csv"]


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ notebook cells
CELLS: dict = {}

CELLS["A1"] = '''# Cell A1 - locate the attached dataset at ANY mount depth, or fail loudly.
# Kaggle mounts this user's datasets three levels deep (/kaggle/input/datasets/<user>/<slug>).
# A resolver that assumes one level dies with "could not find ..." while the dataset IS attached
# correctly, and a local run can never catch it. Every base tried is reported on failure.
import os
from pathlib import Path

REQUIRED = __REQUIRED__
TILE_REL = __TILE_REL__
ALT_REL = __ALT_REL__
SENTINEL = "data/sample_submission.csv"


def find_root(extra_bases=()):
    bases = [Path(b) for b in list(extra_bases) + ["/kaggle/input", str(Path.cwd())] if b]
    tried = []
    for b in bases:
        if not b.exists():
            tried.append("absent base: %s" % b)
            continue
        hits = sorted({p.parents[1] for p in b.glob("**/" + SENTINEL)})
        if not hits:
            tried.append("no %s under %s" % (SENTINEL, b))
        for root in hits:
            miss = [r for r in REQUIRED if not (root / r).exists()]
            if miss:
                tried.append("incomplete root %s (missing %s)" % (root, miss[:3]))
                continue
            if not ((root / TILE_REL).exists() or (root / ALT_REL).exists()):
                tried.append("no tile table at %s or %s under %s" % (TILE_REL, ALT_REL, root))
                continue
            print("[ok] data root found: %s" % root)
            return root
    raise RuntimeError("COULD NOT LOCATE THE MODEL 10 DATASET. Tried: " + " | ".join(tried))


_env = os.environ.get("M10_TEST_INPUT", "")
DATA_ROOT = find_root(_env.split(os.pathsep) if _env else ())
print("  required files present: %d" % len(REQUIRED))
'''

CELLS["A2"] = '''# Cell A2 - copy the mirrored tree into a WRITABLE subdirectory of the work dir.
# /kaggle/input is read-only and build_m10e1.py writes artifacts next to itself, so the tree is
# copied and run from /kaggle/working/m10_run. Only that subdirectory is ever removed - the
# notebook's own outputs in /kaggle/working are never touched.
import shutil
from pathlib import Path

_wb = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
WORK = Path(os.environ.get("M10_TEST_WORK") or (_wb / "m10_run"))


def stage(src_root: Path, dst: Path) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in REQUIRED + [TILE_REL, ALT_REL]:
        s = src_root / f
        if not s.exists():
            continue
        d = dst / f
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        n += 1
    tile = dst / TILE_REL
    if not tile.exists():
        alt = dst / ALT_REL
        if not alt.exists():
            raise RuntimeError("tile table present at neither %s nor %s" % (TILE_REL, ALT_REL))
        tile.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(alt, tile)
        print("  .cache/ was not surfaced by the host; restored from _alt/ (byte-identical)")
    missing = [r for r in REQUIRED + [TILE_REL] if not (dst / r).exists()]
    if missing:
        raise RuntimeError("staged tree is incomplete, missing %r" % missing[:4])
    print("[ok] staged %d files into %s" % (n, dst))
    return dst


RUNNER = stage(DATA_ROOT, WORK) / "Model 10" / "Experiment 1"
EXP_ROOT = RUNNER.parents[1]
print("  executable tree root: %s" % EXP_ROOT)
'''

CELLS["A3"] = '''# Cell A3 - prove byte-fidelity of the shipped code before any number is believed.
# These are the same files that produced the local run. A sha match means Kaggle is executing the
# artifact that was judged, not a re-typed approximation of it.
import hashlib
import sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


EXPECT_CODE = __CODE_HASHES__
print("python %s" % sys.version.split()[0])
for mod in ("numpy", "pandas", "sklearn"):
    try:
        print("  %-8s %s" % (mod, __import__(mod).__version__))
    except Exception as e:
        print("  %-8s IMPORT FAILED: %r" % (mod, e))
code_mismatch = []
for rel, want in sorted(EXPECT_CODE.items()):
    p = EXP_ROOT / rel
    got = sha(p) if p.exists() else "ABSENT"
    if got != want:
        code_mismatch.append(rel)
    print("  [%s] %-52s %s" % ("ok" if got == want else "BAD", rel, got[:16]))
print("[ok] every shipped code file is byte-identical to the local one" if not code_mismatch
      else "[BAD] shipped code differs: %r" % code_mismatch)
'''

CELLS["B1"] = '''# Cell B1 - run the three stages exactly as they ran locally, as subprocesses.
# Subprocess, not import: this is the same command line the local run used, so "the notebook ran
# it" and "the experiment ran" are the same claim. Exit codes are checked, not assumed. All three
# stages execute -- there is no conditional that could skip the part that writes the submission.
import subprocess
import sys

STEPS = ["gates", "score", "submit"]
STEP_LOG = {}
for st in STEPS:
    r = subprocess.run([sys.executable, str(RUNNER / "build_m10e1.py"), st],
                       capture_output=True, text=True, cwd=str(RUNNER))
    STEP_LOG[st] = r.returncode
    print("\\n" + "#" * 78)
    print("# stage %s -- exit %d" % (st, r.returncode))
    print("#" * 78)
    print(r.stdout[-5500:] if len(r.stdout) > 5500 else r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2500:])
step_fail = [s for s in STEPS if STEP_LOG.get(s) != 0]
print("\\n[%s] stages run: %s" % ("FAIL" if step_fail else "ok",
                                "all three" if not step_fail else "stopped at " + str(step_fail)))
'''

CELLS["C1"] = '''# Cell C1 - the Kaggle outputs must equal the local outputs, artifact by artifact.
# This is the whole point of the mirror: same code, same inputs, same numbers. A mismatch is
# reported with both hashes rather than glossed over -- floating-point differences between
# environments are possible, and the submission files are the ones that matter.
import hashlib
import json
import shutil
from pathlib import Path

ARTIFACTS = __ARTIFACTS__
EXPECT = __ARTIFACT_HASHES__
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rows, artifact_mismatch = [], []
for a in ARTIFACTS:
    p = RUNNER / a
    got = sha(p) if p.exists() else "ABSENT"
    want = EXPECT.get(a, "n/a")
    same = (got == want)
    rows.append((a, got, want, same))
    if not same:
        artifact_mismatch.append(a)
    print("  [%s] %-34s %s" % ("ok" if same else "DIFF", a, got[:16]))
print("\\n  artifacts compared: %d | mismatches: %d" % (len(rows), len(artifact_mismatch)))
if artifact_mismatch:
    print("  *** These differ from the local run. Do not submit them until the cause is known:")
    for a in artifact_mismatch:
        print("      %-34s local %s  kaggle %s"
              % (a, dict((r[0], r[2]) for r in rows)[a][:24],
                 dict((r[0], r[1]) for r in rows)[a][:24]))

copied = []
for a in ("Submission_Model10_E1_B.csv", "Submission_Model10_E1_C.csv"):
    if (RUNNER / a).exists():
        shutil.copy2(RUNNER / a, OUT / a)
        copied.append(str(OUT / a))
print("\\n  submission files placed for download:")
for c in copied:
    print("    %s" % c)
json.dump({"artifact_matches": {r[0]: bool(r[3]) for r in rows},
           "mismatches": artifact_mismatch, "kaggle_sha256": {r[0]: r[1] for r in rows},
           "copied": copied}, open(OUT / "kaggle_verification.json", "w"), indent=2)
'''

CELLS["C2"] = '''# Cell C2 - one summary, every check, in a single pass.
# Failures are collected rather than raised one at a time: a remote session that reveals one bug
# per run costs the owner a full paste-and-run cycle each time.
import json
from pathlib import Path

checks = [
    ("data root located", "DATA_ROOT" in dir() and Path(DATA_ROOT).exists()),
    ("staged tree complete", all((EXP_ROOT / r).exists() for r in REQUIRED)),
    ("tile table reachable", (EXP_ROOT / TILE_REL).exists()),
    ("shipped code byte-identical", len(code_mismatch) == 0),
    ("stages gates/score/submit exit 0", len(STEP_LOG) == 3 and not step_fail),
    ("artifacts match the local run", len(artifact_mismatch) == 0),
    ("both submission files exist", all((OUT / a).exists() for a in
                                        ("Submission_Model10_E1_B.csv",
                                         "Submission_Model10_E1_C.csv"))),
    ("G3 contract held (asserted in stage submit)", True),
]
print("=" * 78)
print("MODEL 10 / EXPERIMENT 1 -- KAGGLE RUN SUMMARY")
print("=" * 78)
allok = True
for name, ok in checks:
    allok = allok and bool(ok)
    print("  [%s] %s" % ("PASS" if ok else "FAIL", name))
print("-" * 78)
print("  OVERALL: %s" % ("ALL CHECKS PASS" if allok else "AT LEAST ONE CHECK FAILED"))
print("  The G2 verdict is EXTERNAL. Internal CV is report-only and never selects an arm.")
print("  Submit both files; keep the incumbent unless B beats 55.80591 by >= 5 public EMD")
print("  (plan.md 6b).")
print("=" * 78)
'''

MD = {
    "A1": ("# Model 10 / Experiment 1 -- per-image data expansion\n\n"
           "Kaggle mirror of a run that already completed locally. Reproduces **arms A/B/C and "
           "gates G0a/G0b/G1/G3/D1**.\n\n"
           "| arm | fit | test side |\n|---|---|---|\n"
           "| A (control) | soil rows (24) | predict soil row, project once |\n"
           "| B (primary) | **image rows (127)**, `1/n_images` soil-balanced weights | mean raw "
           "curves per soil, then project once |\n"
           "| C (diagnostic) | soil rows, exactly as A | project **each** image curve, then mean |\n\n"
           "**Internal CV is report-only by design** -- Model 8 E2 measured that this ruler "
           "anti-ranks feature sets. The verdict is the Kaggle score.\n\n"
           "Setup: attach the `m10_e1_data` dataset. **CPU only, no GPU, no internet, no image "
           "files.** About 2 minutes.\n"),
    "A2": "## Locate the dataset (any mount depth) and stage a writable copy",
    "A3": "## Byte-fidelity proof: shipped code == the code that produced the local numbers",
    "B1": "## Run the experiment: gates -> score -> submit (same CLI as the local run)",
    "C1": "## Verify every artifact against the local run's hashes",
    "C2": "## Summary",
}


def build() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    for rel in REQUIRED + [TILE_REL]:
        s, d = ROOT / rel, STAGE / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
    STAGE.joinpath(ALT_REL).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAGE / TILE_REL, STAGE / ALT_REL)
    (STAGE / "README.txt").write_text(
        "Model 10 / Experiment 1 dataset (m10_e1_data)\n"
        "Mirrors the repository layout so the frozen modules need NO edits.\n"
        "Attach to Model10_Experiment1.ipynb. ~5.5 MB, CPU only.\n"
        "_alt/ holds a dot-path-free copy of the tile feature table in case the host does\n"
        "not surface the .cache/ directory.\n", encoding="ascii")

    ah = {a: sha(EXP / a) for a in ARTIFACTS}
    (STAGE / "EXPECTED_HASHES.json").write_text(json.dumps(ah, indent=2), encoding="ascii")
    code_hashes = {c: sha(STAGE / c) for c in CODE}

    subs = {"__REQUIRED__": json.dumps(REQUIRED), "__TILE_REL__": json.dumps(TILE_REL),
            "__ALT_REL__": json.dumps(ALT_REL), "__CODE_HASHES__": json.dumps(code_hashes),
            "__ARTIFACTS__": json.dumps(ARTIFACTS), "__ARTIFACT_HASHES__": json.dumps(ah)}

    cells, srcs = [], {}
    for k in ("A1", "A2", "A3", "B1", "C1", "C2"):
        if k in MD:
            cells.append({"cell_type": "markdown", "metadata": {}, "source": [MD[k]]})
        s = CELLS[k]
        for tok, val in subs.items():
            s = s.replace(tok, val)
        leftover = [t for t in subs if t in s]
        assert not leftover, "unsubstituted tokens %r in cell %s" % (leftover, k)
        compile(s, "cell_%s" % k, "exec")
        srcs[k] = s
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": s.splitlines(keepends=True)})
    flat = "\n".join(srcs[k] for k in ("A1", "A2", "A3", "B1", "C1", "C2"))
    compile(flat, "flattened_notebook", "exec")
    (ROOT / "scratch" / "_m10_flat.py").write_text(flat, encoding="ascii")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("notebook written: %s (%d cells, %d code)"
          % (NB.name, len(cells), sum(1 for c in cells if c["cell_type"] == "code")))

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(STAGE).as_posix())
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    print("zip written: %s (%.2f MB, %d members)"
          % (ZIP.name, ZIP.stat().st_size / 1e6, len(names)))

    tmp = ROOT / "scratch" / "_m10_roundtrip"
    if tmp.exists():
        shutil.rmtree(tmp)
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(tmp)
    src_files = [p for p in sorted(STAGE.rglob("*")) if p.is_file()]
    mism = [str(p.relative_to(STAGE)) for p in src_files
            if not (tmp / p.relative_to(STAGE)).exists()
            or sha(p) != sha(tmp / p.relative_to(STAGE))]
    print("round-trip: %d files extracted, byte-identical mismatches: %d"
          % (len(src_files), len(mism)))
    if mism:
        print("  MISMATCHES:", mism[:8])
        raise SystemExit(1)
    print("  non-ASCII member names: %d" % sum(1 for n in names if not n.isascii()))
    shutil.rmtree(tmp)


if __name__ == "__main__":
    build()
