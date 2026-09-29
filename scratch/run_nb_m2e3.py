"""Execute the SHIPPED E3 notebook, not the builder's output.

E1 and E2 each cost a Kaggle round trip because the artifact judged was not the artifact
verified. This flattens Model2_Experiment3.ipynb itself and runs that, so a green run is
evidence about the file that produced the record.

E3 has no mock mode: BACKEND is ignored, and the run aborts at cell C3 listing every missing
embedding file until E2's cache is harvested into Model 2/Model 2 Experiment 3/embeddings/.
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 2" / "Model 2 Experiment 3" / "Model2_Experiment3.ipynb"
OUT = ROOT / "scratch" / "_nb_m2e3.py"

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
compile(src, str(OUT), "exec")
print(f"flattened {n} code cells from the shipped notebook, compile OK", flush=True)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
env.pop("BACKEND", None)          # E3 has no mock mode; a stale shell value must not leak in
r = subprocess.run([sys.executable, str(OUT)], env=env, cwd=str(ROOT))
print("exit =", r.returncode)
sys.exit(r.returncode)
