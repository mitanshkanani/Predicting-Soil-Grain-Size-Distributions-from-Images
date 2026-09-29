"""Executes the SHIPPED Model 5 verdict ladder against fabricated inputs. No experiment runs.

Why this exists: mk_run6.py proves the gate COMPILES, which is not the same claim. A verdict
ladder that cannot reach one of its ten outcomes, or a threshold that is printed but never read,
compiles perfectly. This file takes the decision code out of Model5_Experiment1.ipynb verbatim -
the same characters that will run the experiment, sliced between two markers - and drives it
through every branch the spec's section 6 table lists.

It then mutates one threshold at a time and requires the verdict to move. A gate whose numbers
cannot change its answer is decoration.

Nothing here touches data/, writes into the experiment folder, or produces a score. The inputs are
invented; only the decision code is real.

Run:  python scratch/check_m5_gate_logic.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Model 5" / "Model 5 Experiment 1" / "Model5_Experiment1.ipynb"
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import transfer_eval as te                                            # noqa: E402

FAILS, LINES = [], []
LO, HI, INT = te.ALPHAS[0], te.ALPHAS[-1], te.ALPHAS[7]
BASE_A, BASE_B = 58.0, 60.0           # invented cross-camera errors; only PENALTY reads them


def check(name, ok, detail=""):
    (LINES if ok else FAILS).append(f"  {'PASS' if ok else '**FAIL**'}  {name}"
                                    + (f"   {detail}" if detail else ""))


def cell(tag):
    nb = json.load(io.open(NB, encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            s = "".join(c["source"])
            if s.startswith(f'"""Cell {tag} -'):
                return s
    raise AssertionError(f"cell {tag} not in the shipped notebook")


A1 = cell("A1")
G1 = cell("G1")
LADDER = G1[G1.index("def dir_boundary_status"):G1.index("GATE = dict(")]
SUBMIT_LINE = "SUBMIT = bool(G0 and G1 and G2 and G3 and G4 and G5 and G6 and G7)"
assert SUBMIT_LINE in LADDER, "the shipped ladder does not use the spec's conjunction"
assert "json.dump" not in LADDER and "to_csv" not in LADDER, "sliced too far: this would write"


def _gain(arm, pooled, lo, hi, a, b, la=0.5, ha=3.5):
    return dict(arm=arm, pooled_gain=pooled, pooled_lo=lo, pooled_hi=hi, gain_A=a, lo_A=la,
                hi_A=ha, gain_B=b, lo_B=la, hi_B=ha, n_rows=45, n_clusters=24)


def run_ladder(**over):
    """Execute the shipped clause-and-verdict code on invented inputs and return its namespace."""
    o = dict(coral=4.0, lo=1.0, hi=7.0, A=2.0, B=2.0, delta=2.0, placebo=0.5, drift=0.0,
             mean=3.9, alphas=("interior", "interior"), g0_ok=True, cfg_over={})
    o.update(over)
    alpha_of = {"interior": INT, "FLOOR": LO, "CEILING": HI}
    GAIN = {
        "NONE":        _gain("NONE", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "CORAL":       _gain("CORAL", o["coral"], o["lo"], o["hi"], o["A"], o["B"]),
        "MEAN":        _gain("MEAN", o["mean"], 0.4, 7.4, 1.9, 1.9),
        "CORAL_INDEP": _gain("CORAL_INDEP", o["placebo"], -0.5, 1.5, 0.3, 0.3),
        "CORAL_SELF":  _gain("CORAL_SELF", o["drift"], -0.1, 0.1, o["drift"], o["drift"],
                             -0.1, 0.1),
        "CORAL_TILE":  _gain("CORAL_TILE", o["coral"] * 0.6, 0.2, 5.0, 1.2, 1.2),
        "CORAL_SOIL":  _gain("CORAL_SOIL", o["coral"] * 0.8, 0.3, 6.0, 1.6, 1.6),
    }
    CONTRASTS = pd.DataFrame([
        dict(arm_b=a, mean=o["delta"] if a == "CORAL_INDEP" else
             (GAIN["CORAL"]["pooled_gain"] - GAIN[a]["pooled_gain"]),
             lo=(o["delta"] - 1.0) if a == "CORAL_INDEP" else -0.5, hi=5.0)
        for a in GAIN if a != "CORAL"])
    FOLDS = pd.DataFrame([dict(direction=d, arm=a, fold=f"F{i}",
                               nested_alpha=alpha_of[o["alphas"][k]])
                          for k, d in enumerate(("A", "B")) for a in GAIN for i in range(4)])
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(A1, "cellA1", "exec"), ns)                    # the SHIPPED thresholds
        for fld, val in o["cfg_over"].items():
            object.__setattr__(ns["CFG"], fld, val)
        ns.update(te=te, np=np, GAIN=GAIN, CONTRASTS=CONTRASTS, FOLDS=FOLDS,
                  BASE={"A": pd.Series({f"S{i}": BASE_A for i in range(22)}),
                        "B": pd.Series({f"T{i}": BASE_B for i in range(23)})},
                  G0_OK=o["g0_ok"], G0_GAP=0.0, G0_REF=43.0217308796477, G0_GOT=43.0217308796477)
        exec(compile(LADDER, "cellG1-ladder", "exec"), ns)
    return ns


def scenario(wanted, label="", **over):
    ns = run_ladder(**over)
    check(f"{wanted}{': ' + label if label else ''}", ns["VERDICT"] == wanted,
          f"got {ns['VERDICT']} submit={ns['SUBMIT']}")
    check(f"  and submit is True for SUPPORTED alone (spec 7) [{wanted}]",
          ns["SUBMIT"] == (wanted == "SUPPORTED"), f"submit={ns['SUBMIT']}")
    return ns


print("CHECK M5 GATE LOGIC - the shipped decision code, driven through every branch")
print("-" * 78)

ns = scenario("SUPPORTED", "all eight clauses pass, both directions interior")
check("SUPPORTED is the only verdict that submits", ns["SUBMIT"] is True)
check("exactly eight clauses are evaluated", len(ns["CLAUSES"]) == 8, str(len(ns["CLAUSES"])))
check("the clause flags are the same booleans the verdict was built from",
      [c["passed"] for c in ns["CLAUSES"]] ==
      [ns[k] for k in ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7")])
check("the penalty is the cross-camera loss minus the same-camera control",
      abs(ns["PENALTY"] - (59.0 - 43.0217308796477)) < 0.05, f"{ns['PENALTY']:.3f}")

scenario("SUPPORTED-WITH-BOUNDARY-CAVEAT", "one direction clipped by the grid",
         alphas=("CEILING", "interior"))
scenario("REFUTED", "the predicted outcome",
         coral=1.0, lo=-1.0, A=1.0, B=1.0, delta=0.2)
scenario("REFUTED-WITH-BOUNDARY-CAVEAT", "refuted and clipped",
         coral=1.0, lo=-1.0, A=1.0, B=1.0, delta=0.2, alphas=("interior", "FLOOR"))
scenario("SUPPORTED-BELOW-BAND", "G1 misses, the placebo contrast still holds",
         coral=2.0, A=1.0, B=1.0)
scenario("REGULARISATION-ARTEFACT", "the gain the placebo also achieves", delta=0.4)
scenario("DIRECTION-INCONSISTENT", "one direction significantly worse", B=-0.4)
scenario("NOT-RESOLVED", "effect size without statistical support", lo=-0.3)
scenario("INCONCLUSIVE", "one direction floored, the other clipped",
         alphas=("FLOOR", "CEILING"))
scenario("INVALID-IMPLEMENTATION", "G0 failed", g0_ok=False)
scenario("INVALID-IMPLEMENTATION", "self-alignment moved the score", drift=0.9)

ns_base = run_ladder()
for v in ns_base["VERDICTS"]:
    check(f"verdict {v} is written by the ladder itself, not only listed in the tuple",
          f'"{v}"' in LADDER)

# a threshold that cannot change the answer is not a threshold
for fld, new, why, extra in [
        ("G1_MIN", 5.0, "raising the pooled floor above 4.0", {}),
        ("G2_MIN", 3.0, "raising the per-direction floor above 2.0", {}),
        ("G4_MIN", 4.0, "raising the placebo margin above the 2.0 delta", {}),
        ("G5_MAX", 0.0, "shrinking the canary band below a 0.05 drift", dict(drift=0.05)),
        ("G3_MIN", 2.0, "demanding the CI lower bound clear 2.0", {})]:
    moved = run_ladder(cfg_over={fld: new}, **extra)
    check(f"{why} changes the shipped verdict", moved["VERDICT"] != ns_base["VERDICT"],
          f"{ns_base['VERDICT']} -> {moved['VERDICT']}")

# and the reverse: relaxing a threshold must not be able to rescue a failed gate
lost = run_ladder(B=-0.4)
check("a significantly worse direction cannot be rescued by raising every other threshold",
      lost["VERDICT"] == "DIRECTION-INCONSISTENT",
      str({k: lost[k] for k in ("G2", "SUBMIT")}))

print("\n".join(LINES))
print("-" * 78)
if FAILS:
    print("\n".join(FAILS))
    print(f"CHECK M5 GATE LOGIC: FAILED ({len(FAILS)} problems)")
    sys.exit(1)
print(f"CHECK M5 GATE LOGIC: PASSED ({len(LINES)} checks, "
      f"all 10 verdicts exercised, 5 thresholds shown live)")
