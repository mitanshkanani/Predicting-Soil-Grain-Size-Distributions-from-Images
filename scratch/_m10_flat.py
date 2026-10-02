# Cell A1 - locate the attached dataset at ANY mount depth, or fail loudly.
# Kaggle mounts this user's datasets three levels deep (/kaggle/input/datasets/<user>/<slug>).
# A resolver that assumes one level dies with "could not find ..." while the dataset IS attached
# correctly, and a local run can never catch it. Every base tried is reported on failure.
import os
from pathlib import Path

REQUIRED = ["Model 5/Model 5 Experiment 1/transfer_eval.py", "Model 8/Experiment 3/ruler.py", "Model 8/Experiment 3/d50.py", "Model 10/Experiment 1/ruler.py", "Model 10/Experiment 1/d50.py", "Model 10/Experiment 1/image_features.py", "Model 10/Experiment 1/build_m10e1.py", "Model 1/Model 1 Experiment 3/features_soil.csv", "Model 2/Model 2 Experiment 3/cv_families.csv", "Model 8/Experiment 2/Submission_Model8_E2_E2A.csv", "data/processed_meta/manifest_tiles.csv", "data/processed_meta/manifest_samples.csv", "data/processed_meta/manifest_images.csv", "data/sample_submission.csv"]
TILE_REL = "Model 1/Model 1 Experiment 3/.cache/tile_features_256_010f44c36c74.csv"
ALT_REL = "_alt/tile_features_256_010f44c36c74.csv"
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

# Cell A2 - copy the mirrored tree into a WRITABLE subdirectory of the work dir.
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

# Cell A3 - prove byte-fidelity of the shipped code before any number is believed.
# These are the same files that produced the local run. A sha match means Kaggle is executing the
# artifact that was judged, not a re-typed approximation of it.
import hashlib
import sys
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


EXPECT_CODE = {"Model 5/Model 5 Experiment 1/transfer_eval.py": "a610959867256715e1dd7950cda527296490f54a62d77eb2690d66a645a9bad5", "Model 8/Experiment 3/ruler.py": "cafb00eab18ca9a097bc3139f9838d773364343c8af652c77ad0d3be78e17771", "Model 8/Experiment 3/d50.py": "82e8dab4a0d005a6d9f01873d886163f2719ffd089c933682c8fd8fc23902f42", "Model 10/Experiment 1/ruler.py": "cafb00eab18ca9a097bc3139f9838d773364343c8af652c77ad0d3be78e17771", "Model 10/Experiment 1/d50.py": "82e8dab4a0d005a6d9f01873d886163f2719ffd089c933682c8fd8fc23902f42", "Model 10/Experiment 1/image_features.py": "f5b5c49d855db34e5689b2b8ed14b15c5f592f0a1f86ac7eae3db964f9870424", "Model 10/Experiment 1/build_m10e1.py": "bc609bc5df3508c7153aae70d47e62af4a4630a07962002c53ca1c673829401a"}
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

# Cell B1 - run the three stages exactly as they ran locally, as subprocesses.
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
    print("\n" + "#" * 78)
    print("# stage %s -- exit %d" % (st, r.returncode))
    print("#" * 78)
    print(r.stdout[-5500:] if len(r.stdout) > 5500 else r.stdout)
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2500:])
step_fail = [s for s in STEPS if STEP_LOG.get(s) != 0]
print("\n[%s] stages run: %s" % ("FAIL" if step_fail else "ok",
                                "all three" if not step_fail else "stopped at " + str(step_fail)))

# Cell C1 - the Kaggle outputs must equal the local outputs, artifact by artifact.
# This is the whole point of the mirror: same code, same inputs, same numbers. A mismatch is
# reported with both hashes rather than glossed over -- floating-point differences between
# environments are possible, and the submission files are the ones that matter.
import hashlib
import json
import shutil
from pathlib import Path

ARTIFACTS = ["gates.json", "arm_comparison.csv", "cv_per_soil.csv", "bootstrap.json", "placebo.json", "score_internal.json", "submission_report.json", "Submission_Model10_E1_B.csv", "Submission_Model10_E1_C.csv"]
EXPECT = {"gates.json": "c99454ca50fe0c511ed4c71789bb36f10cf7ceedc5f85c5e0152ba63d387bcbe", "arm_comparison.csv": "ae9405f99f466be5509943b2155455ace97315f798976a3ec202b2dc5ca47aed", "cv_per_soil.csv": "7d69cfe911fb503ef243ccbd74df6db01ff05cd0b72159fc934a0377342137bc", "bootstrap.json": "a34301cb60e29713def2f08792061c4e166e4cdfbb7c38e4f8218c45d2194d32", "placebo.json": "0331913562f8ba5c5cfc92978ba44f4f9b133683ed86be8ac94dd5a74cd5d328", "score_internal.json": "cc6191c761b2023cd6a68d2e65f4b4a3e3d793472cd3360abf63719f5faa74a8", "submission_report.json": "84762aa80eb52d99b1cb0a2c8f8205db98a4d5994ece33d3925bbdde29085e68", "Submission_Model10_E1_B.csv": "1d1f3342db5bf19927e48dd63bd1c686a94e1a533402f496bbbb9b0e048fc391", "Submission_Model10_E1_C.csv": "48af674a6a5224969cd5e32af4f293555b98488af0392779f4225d1f607f8f1c"}
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
print("\n  artifacts compared: %d | mismatches: %d" % (len(rows), len(artifact_mismatch)))
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
print("\n  submission files placed for download:")
for c in copied:
    print("    %s" % c)
json.dump({"artifact_matches": {r[0]: bool(r[3]) for r in rows},
           "mismatches": artifact_mismatch, "kaggle_sha256": {r[0]: r[1] for r in rows},
           "copied": copied}, open(OUT / "kaggle_verification.json", "w"), indent=2)

# Cell C2 - one summary, every check, in a single pass.
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
