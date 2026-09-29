"""Regenerate the Model 5 E1 notebook from its builder, then compile-check it two ways.

Per-cell compile first, because a cell with a broken string terminator is a cell-level failure
that a flattened compile can report many lines away, in a different cell than the one that caused
it. That is not hypothetical here: build_m5e1.py shipped seven cells closed with \"\"\" instead of
''' , which made the builder itself unparseable rather than producing seven short cells. The
builder now pins the exact ordered cell list, and this script re-verifies each cell parses alone.

Then flatten the code cells into scratch/run_m5e1.py so the notebook can be executed headlessly,
exactly as Model 2 E1/E2/E3 did.

This script BUILDS AND CHECKS. It never runs the experiment - scratch/run_nb_m5e1.py does that.
"""
import io
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 5" / "Model 5 Experiment 1" / "Model5_Experiment1.ipynb"
OUT = ROOT / "scratch" / "run_m5e1.py"

subprocess.run([sys.executable, str(ROOT / "scratch" / "build_m5e1.py")], check=True)

nb = json.load(io.open(NB, encoding="utf-8"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
bad, names = [], []
for i, c in enumerate(cells):
    src = "".join(c["source"])
    m = re.match(r'"""Cell ([A-Z0-9]+)', src)
    name = m.group(1) if m else "?"
    names.append(name)
    try:
        compile(src, f"cell{i + 1}({name})", "exec")
    except SyntaxError as exc:
        bad.append(f"cell {i + 1} ({name}): {exc.msg} at line {exc.lineno}")

print(f"\n{len(cells)} code cells, {len(nb['cells'])} cells total")
if bad:
    print("PER-CELL COMPILE: FAILED")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print(f"PER-CELL COMPILE: PASSED (all {len(cells)} cells parse independently)")
print("cell order:", " ".join(names))

buf = io.StringIO()
buf.write('import matplotlib; matplotlib.use("Agg")\n\n')
for i, c in enumerate(cells):
    buf.write(f'# ---- code cell {i + 1} ----\nprint(" >>> cell {i + 1}", flush=True)\n')
    buf.write("".join(c["source"]) + "\n\n")
flat = buf.getvalue()
compile(flat, str(OUT), "exec")
io.open(OUT, "w", encoding="utf-8").write(flat)
print(f"flattened to {OUT.name}, compiles clean")

nonascii = {ln for ln in flat.splitlines() if any(ord(ch) > 127 for ch in ln)}
print(f"  {'ok  ' if not nonascii else 'WARN'} ASCII-only printed source "
      f"({len(nonascii)} non-ascii lines)")
for pat, why in [("import torch", "no torch"), ("import timm", "no timm"),
                 ("BACKEND=", "no mock switch"), ("/kaggle/input", "no Kaggle mount path"),
                 ("CORAL_PERM", "no inert row-permutation placebo")]:
    hits = flat.count(pat)
    print(f"  {'ok  ' if hits == 0 else 'WARN'} {pat!r} appears {hits}x  ({why})")

# the gate must be implemented, not merely described
gate = {k: flat.count(k) for k in
        ["G0 = bool(", "G1 = bool(", "G2 = bool(", "G3 = bool(", "G4 = bool(", "G5 = bool(",
         "G6 = bool(", "G7 = True", "SUBMIT = bool(G0 and G1 and G2 and G3 and G4 and G5 and G6"
         " and G7)", "VERDICT = ", "dir_boundary_status", "paired_contrast"]}
missing = [k for k, v in gate.items() if v == 0]
if missing:
    print("GATE IMPLEMENTATION CHECK: FAILED - not present in the shipped notebook:")
    for k in missing:
        print("  -", k)
    sys.exit(1)
print("GATE IMPLEMENTATION CHECK: PASSED  (G0-G7 each assigned in code, the spec's submit "
      "conjunction verbatim, verdict branches, per-direction boundary rule, paired contrasts)")

# ---- order-aware free-name scan -----------------------------------------------
# A NameError in cell R2 costs the whole run, and the two dry runs cannot reach past the
# gate without scoring arms. This resolves every loaded name against the namespace the
# PRECEDING cells would actually have built, which is exactly what the flattened run does.
import ast                                                        # noqa: E402

BUILTIN = set(dir(__builtins__)) | {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "format", "int", "len", "list",
    "map", "max", "min", "print", "range", "repr", "reversed", "round", "set", "sorted", "str",
    "sum", "tuple", "zip", "Exception", "FileNotFoundError", "ValueError", "AssertionError",
    "RuntimeError", "KeyError", "__file__", "__name__", "True", "False", "None"}


def bound_names(tree):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Import):
            out.update((a.asname or a.name.split(".")[0]) for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            out.update((a.asname or a.name) for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out.update(n.names)
    return out


seen = set(BUILTIN)
name_problems = []
for i, c in enumerate(cells):
    src = "".join(c["source"])
    tag = re.match(r'"""Cell ([A-Z0-9]+)', src).group(1)
    tree = ast.parse(src)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    own = bound_names(tree)
    free = sorted(used - seen - own)     # within-cell ordering is Python's business, not ours
    if free:
        name_problems.append(f"cell {tag}: {free}")
    seen |= own
if name_problems:
    print("FREE-NAME SCAN: FAILED - a name used before any preceding cell binds it")
    for p in name_problems:
        print("  -", p)
    sys.exit(1)
print(f"FREE-NAME SCAN: PASSED  ({len(cells)} cells in run order; every loaded name is bound by "
      f"a preceding cell, an import, or a builtin)")

# ---- contract <-> code agreement ------------------------------------------------
# The written rule and the executed rule are separate files, and this project has already been
# bitten by a doc that said one thing and code that did another. Every marker below is a literal
# string taken from the document it is checked against, so a rewording that drops the rule fails
# here rather than passing on a paraphrase.
EXP = ROOT / "Model 5" / "Model 5 Experiment 1"
SPEC = io.open(EXP / "model_spec_m5e1.md", encoding="utf-8").read()
INSTR = io.open(EXP / "instructions.txt", encoding="utf-8").read()
MARKERS = [
    ("spec: submit is iff SUPPORTED", SPEC,
     ['SUBMIT = TRUE  <=>  VERDICT == "SUPPORTED"',
      '`SUBMIT = TRUE` **if and only if** `VERDICT == "SUPPORTED"`']),
    ("spec: every caveat verdict is listed as no-submission", SPEC,
     ['| `SUPPORTED-WITH-BOUNDARY-CAVEAT` | none |',
      '| `REFUTED-WITH-BOUNDARY-CAVEAT` | none, and no permanent closure |']),
    ("spec: a single documented precedence exists", SPEC, ["### 6a. Verdict precedence"]),
    ("instructions: submit is iff SUPPORTED", INSTR,
     ["SUBMIT = TRUE IF AND ONLY IF VERDICT == SUPPORTED"]),
    ("instructions: every caveat verdict is listed as no-submission", INSTR,
     ["SUPPORTED-WITH-BOUNDARY-CAVEAT -> NO submission",
      "REFUTED-WITH-BOUNDARY-CAVEAT   -> NO submission, and no permanent closure"]),
    ("instructions: a single documented precedence exists", INSTR, ["VERDICT PRECEDENCE"]),
    ("notebook: the identity is asserted, not just written down", flat,
     ['assert SUBMIT == (VERDICT == "SUPPORTED")']),
    ("notebook: the caveat is folded into SUBMIT", flat, ["SUBMIT = bool(SUBMIT and not CAVEAT)"]),
]
gaps = [name for name, text, ms in MARKERS if any(m not in text for m in ms)]
if gaps:
    print("CONTRACT AGREEMENT CHECK: FAILED - the rule is missing from:")
    for g in gaps:
        print("  -", g)
    sys.exit(1)
for name, _, ms in MARKERS:
    print(f"  ok  {name}  ({len(ms)} marker{'s' if len(ms) > 1 else ''} present)")
print("CONTRACT AGREEMENT CHECK: PASSED  (spec, instructions and notebook all state SUBMIT <=> "
      "VERDICT == SUPPORTED, list both caveats as no-submission, and name one precedence)")


