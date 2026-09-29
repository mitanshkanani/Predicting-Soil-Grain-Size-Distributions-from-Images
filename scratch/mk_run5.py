"""Regenerate the E3 notebook from its builder, then compile-check it two ways.

Per-cell compile first, because a cell with a broken string terminator is a cell-level
failure that a flattened compile can misreport as an error many lines away, in a different
cell than the one that caused it. E2's generator only flattened, and the H3 cell's stray
delimiter in E3 would have surfaced as a syntax error 500 lines from its cause.

Then flatten the code cells into scratch/run_m2e3.py so the notebook can be executed
headlessly, exactly as Model 2 E1/E2 did.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 2" / "Model 2 Experiment 3" / "Model2_Experiment3.ipynb"
OUT = ROOT / "scratch" / "run_m2e3.py"

subprocess.run([sys.executable, str(ROOT / "scratch" / "build_m2e3.py")], check=True)

nb = json.load(io.open(NB, encoding="utf-8"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
bad = []
for i, c in enumerate(cells):
    src = "".join(c["source"])
    name = ""
    for line in src.splitlines():
        if line.startswith('"""Cell '):
            name = line.split("Cell ")[1].split(" -")[0]
            break
    try:
        compile(src, f"cell{i + 1}({name or '?'})", "exec")
    except SyntaxError as exc:
        bad.append(f"cell {i + 1} ({name or '?'}): {exc.msg} at line {exc.lineno}")

print(f"\n{len(cells)} code cells, {len(nb['cells'])} cells total")
if bad:
    print("PER-CELL COMPILE: FAILED")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print(f"PER-CELL COMPILE: PASSED (all {len(cells)} cells parse independently)")

buf = io.StringIO()
buf.write('import matplotlib; matplotlib.use("Agg")\n\n')
for i, c in enumerate(cells):
    buf.write(f'# ---- code cell {i + 1} ----\nprint(" >>> cell {i + 1}", flush=True)\n')
    buf.write("".join(c["source"]) + "\n\n")
flat = buf.getvalue()
compile(flat, str(OUT), "exec")
io.open(OUT, "w", encoding="utf-8").write(flat)
print(f"flattened to {OUT.name}, compiles clean")

for pat, why in [("import torch", "no torch"), ("import timm", "no timm"),
                 ("BACKEND=", "no mock switch"), ("/kaggle/input", "no Kaggle mount path")]:
    hits = flat.count(pat)
    print(f"  {'ok  ' if hits == 0 else 'WARN'} {pat!r} appears {hits}x  ({why})")
