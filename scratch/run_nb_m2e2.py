"""Flatten the SHIPPED E2 notebook into an executable script and run it under mock.

run_m2e2.py is generated from the builder; this reads the .ipynb itself, so a green run is
evidence about the artifact that actually goes to Kaggle, not about its source.
"""
import json, io, os, subprocess, sys

NB = "Model 2/Model 2 Experiment 2/Model2_Experiment2.ipynb"
OUT = "scratch/_nb_m2e2.py"

nb = json.load(io.open(NB, encoding="utf-8"))
parts = ['import matplotlib; matplotlib.use("Agg")']
n = 0
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    n += 1
    parts.append(f'\n# ---- code cell {n} ----\nprint(" >>> cell {n}", flush=True)')
    parts.append("".join(c["source"]))
src = "\n".join(parts)
io.open(OUT, "w", encoding="utf-8").write(src)
compile(src, OUT, "exec")
print(f"flattened {n} code cells, compile OK", flush=True)
env = dict(os.environ, BACKEND="mock", PYTHONIOENCODING="utf-8")
r = subprocess.run([sys.executable, OUT], env=env, cwd=os.getcwd())
print("exit =", r.returncode)
sys.exit(r.returncode)
