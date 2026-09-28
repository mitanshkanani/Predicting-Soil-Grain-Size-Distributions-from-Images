"""Regenerate the Model 2 E2 notebook from its builder and flatten the code cells into an
executable script, so the notebook and the verified local run cannot diverge."""
import json, io, subprocess, sys

subprocess.run([sys.executable, "scratch/build_m2e2.py"], check=True)
NB = "Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
o = io.StringIO(); o.write('import matplotlib; matplotlib.use("Agg")\n\n')
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    n += 1
    o.write(f'# ---- code cell {n} ----\nprint(" >>> cell "+str({n}), flush=True)\n'
            + "".join(c["source"]) + "\n\n")
src = o.getvalue()
compile(src, "run_m2e2.py", "exec")
open("scratch/run_m2e2.py", "w", encoding="utf-8").write(src)
print(f"regenerated + extracted {n} code cells, compiles clean")
