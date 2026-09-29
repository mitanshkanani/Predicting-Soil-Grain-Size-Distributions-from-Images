"""Execute the SHIPPED Model 5 E1 notebook, not the builder's output.

Every Model 2 round trip cost extra because the artifact judged was not the artifact verified.
This flattens Model5_Experiment1.ipynb itself and runs that, so a green run is evidence about the
file that produced the record - and a re-run reproduces byte-identical artifacts, which is the
only reason determinism checks are possible at all.

Model 5 has no mock mode and no Kaggle run: it reads cached tile features from Model 1 E3 and the
sample manifests, so it runs on a laptop CPU. There is no BACKEND switch to forget to unset, but
one is popped anyway so a stale shell value from a Model 2 session cannot mean anything here.

Usage:  python scratch/run_nb_m5e1.py            (from the repository root)
"""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 5" / "Model 5 Experiment 1" / "Model5_Experiment1.ipynb"
OUT = ROOT / "scratch" / "_nb_m5e1.py"

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
env.pop("BACKEND", None)          # Model 5 has no mock mode; a stale shell value must not leak in
r = subprocess.run([sys.executable, str(OUT)], env=env, cwd=str(ROOT))
print("exit =", r.returncode)
sys.exit(r.returncode)
