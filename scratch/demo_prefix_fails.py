"""Demonstrate that the Task 3 (pre-fix) implementation fails the new I1-I6 legs.

Loads the pre-fix m7a_data.py, recovered verbatim from the Task 3 review package, under the module
name the checker imports - written into the experiment folder so its HERE/ROOT paths resolve - and
runs the selected check_m7a_data legs against it. The real m7a_data.py on disk is never touched,
the legs chosen are the ones that need no full corpus pass (so no artifact is rewritten), and the
copy this makes is deleted at the end.
"""
import importlib.util
import io
import os
import sys

APPROACH = os.path.join("Model 7", "Approach A")
DIFF = os.path.join(".superpowers", "sdd", "model_plan_m7a", "task-3-diff.txt")
TARGET = "## Model 7/Approach A/m7a_data.py"
PREFIX = os.path.join(APPROACH, "_prefix_demo_m7a_data.py")
FILTERS = ["missing_tile", "named_even_when", "fabricated", "probe_flags", "real_degeneracy",
           "outlier_rule", "zero_mad", "strict_json", "nonfinite_paths", "routes_through",
           "systemic_error", "assembler_opens", "assembler_calls", "namespace", "installs_no_sink",
           "warning_sink", "tile_stage"]

lines = io.open(DIFF, encoding="utf-8").read().splitlines()
start = next(i for i, line in enumerate(lines) if line.startswith(TARGET))
nxt = [j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")]
body = lines[start:nxt[0] if nxt else len(lines)]
hunk = next(x for x, line in enumerate(body) if line.startswith("@@"))
out = [line[1:] for line in body[hunk + 1:] if line.startswith("+")]
io.open(PREFIX, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
print("pre-fix implementation recovered from the review package: %d lines" % len(out))

sys.path.insert(0, os.path.abspath(APPROACH))
sys.path.insert(0, os.path.join(os.path.abspath(os.curdir), "Model 5", "Model 5 Experiment 1"))
spec = importlib.util.spec_from_file_location("m7a_data", PREFIX)
mod = importlib.util.module_from_spec(spec)
sys.modules["m7a_data"] = mod
spec.loader.exec_module(mod)
print("loaded as 'm7a_data'. symbols the new legs need:")
for n in ("publish", "find_nonfinite", "require_publishable", "WARNING_STAGES",
          "NON_FINITE_MEANING", "IMG_CSV_NAME", "read_feature_csv", "artifact_paths"):
    print("  %-22s %s" % (n, "present" if hasattr(mod, n) else "ABSENT"))

os.chdir(APPROACH)
sys.argv = ["check_m7a_data.py"] + FILTERS
import check_m7a_data  # noqa: E402  picks up the pre-fix module through sys.modules
rc = check_m7a_data.main()
os.remove(PREFIX)
print("\npre-fix run exit code: %d  (a non-zero exit is the demonstration)" % rc)
