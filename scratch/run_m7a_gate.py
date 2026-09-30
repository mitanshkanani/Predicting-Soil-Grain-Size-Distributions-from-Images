"""Task 6 registered gate run: compute ag.separability twice in two cold interpreters and compare.

Nothing is written. No CV arm is fitted, no alpha is selected, no control block (P-COLS / P-SEL) runs,
no prediction is made, no submission exists. The two runs exist to show the four numbers survive a
process that never saw a memoised tile assembly.

Run:  python scratch/run_m7a_gate.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "Model 7" / "Approach A"
ROOT = HERE.parents[1]
CODE = ("import json, sys; sys.path.insert(0, %r); sys.path.insert(0, %r);"
        "import alias_gate as ag; print(json.dumps(ag.separability(%r)))"
        % (str(HERE), str(ROOT / "Model 5" / "Model 5 Experiment 1"), str(ROOT)))

pairs_sha = hashlib.sha256((HERE / "data" / "alias_pairs.csv").read_bytes()).hexdigest()
print("frozen pair set before the gate : %s" % pairs_sha[:16])

runs = []
for n in (1, 2):
    proc = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                          cwd=str(HERE), timeout=5400)
    if proc.returncode != 0:
        print("run %d FAILED rc=%d\n%s" % (n, proc.returncode, proc.stderr[-3000:]))
        sys.exit(1)
    runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))
    print("run %d done" % n)

print("\n== the registered pre-gate numbers (alias pairs) ==")
for key in ("base12", "block16", "rand16", "shuf16", "gain_vs_base", "rand_gain_vs_base",
            "shuf_gain_vs_base"):
    print("  %-20s run1 %+.8f   run2 %+.8f   identical %s"
          % (key, runs[0][key], runs[1][key], runs[0][key] == runs[1][key]))
print("  %-20s run1 %s   run2 %s" % ("gate_passed", runs[0]["gate_passed"],
                                     runs[1]["gate_passed"]))
print("\n== control pairs, reported only and NOT gate-relevant ==")
for key in ("ctrl_base12", "ctrl_block16", "ctrl_gain_vs_base"):
    print("  %-20s run1 %+.8f   run2 %+.8f" % (key, runs[0][key], runs[1][key]))
print("  n_alias %s  n_control %s  block %s  seeds rand/shuf %s/%s"
      % (runs[0]["n_alias"], runs[0]["n_control"], runs[0]["block_columns"],
         runs[0]["seed_rand"], runs[0]["seed_shuf"]))
print("\n== the two cold interpreters agree exactly ==")
print("  identical JSON : %s" % (json.dumps(runs[0], sort_keys=True)
                                 == json.dumps(runs[1], sort_keys=True)))
after = hashlib.sha256((HERE / "data" / "alias_pairs.csv").read_bytes()).hexdigest()
print("  frozen pair set after the gate : %s  unchanged: %s" % (after[:16], after == pairs_sha))
print("\nVERDICT INPUT: gate_passed=%s -> %s"
      % (runs[0]["gate_passed"],
         "M7-A PROCEEDS to the CV arms (Task 7)" if runs[0]["gate_passed"]
         else "M7-A CLOSES at the pre-gate: REFUTED-PRE-GATE"))
print(json.dumps(runs[0], indent=1, sort_keys=True))
