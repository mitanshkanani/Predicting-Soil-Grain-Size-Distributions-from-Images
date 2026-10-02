"""anchor_check.py - Model 11 / Experiment 1, Task 1: gate G0a.

Reproduces the two registered internal anchors BEFORE anything else in the experiment is
allowed to run, per the project's standing rule that a reproduction claim must be a number
and not an assertion:

  nested  43.453225201811563   strictly family-honest nested LOFO (Model 7+ baseline)
  oracle  41.1475158656982032  the same design scored at alpha = 3.0

Both are INTERNAL CV numbers on 24 labelled soils. They are not Kaggle scores and are never
subtracted from one. Nothing here fits a new model, reads an image, or predicts a test soil:
ruler.design() rebuilds the frozen 24 x 12 matrix from the cached tile features and the two
scores are computed with the frozen protocol transcribed in ruler.py.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "Model 5" / "Model 5 Experiment 1", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ruler as R  # noqa: E402  byte-copy of Model 8 E3's transcription

NESTED_ANCHOR = 43.453225201811563
ORACLE_ANCHOR = 41.1475158656982032
ALPHA_ORACLE = 3.0
TOL_NESTED = 1e-6      # Model 8 E3's own tolerances, not new ones
TOL_ORACLE = 1e-9


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    print("=" * 78)
    print("MODEL 11 / EXPERIMENT 1  Task 1 -- scaffold + gate G0a")
    print("=" * 78)

    src = ROOT / "Model 8" / "Experiment 3"
    copies = {}
    for name in ("ruler.py", "d50.py"):
        s, d = src / name, HERE / name
        hs = sha(s)
        copies[name] = {"src_sha256": hs, "dst_sha256": sha(d), "identical": hs == sha(d)}
        print("  %-9s src %s" % (name, hs))
        print("  %-9s dst %s  -> identical: %s" % (" " * 9, copies[name]["dst_sha256"],
                                                   copies[name]["identical"]))
    te = ROOT / "Model 5" / "Model 5 Experiment 1" / "transfer_eval.py"
    print("  transfer_eval.py (imported read-only, not copied) sha %s" % sha(te)[:16])

    ok_copies = all(c["identical"] for c in copies.values())
    print("  byte-for-byte copies verified: %s" % ok_copies)

    X, Y, fams, ids = R.design()
    print("\n  design: X %s  Y %s  families %d  soils %d"
          % (X.shape, Y.shape, len(set(fams)), len(ids)))
    if X.shape != (24, 12) or len(set(fams)) != 16:
        print("  *** unexpected design shape or family count -- stopping")
        return 1

    nested = float(R.lofo_errors(X, Y, fams).mean())
    oracle = float(R.alpha_curve(X, Y, fams)[ALPHA_ORACLE])
    g_n, g_o = abs(nested - NESTED_ANCHOR), abs(oracle - ORACLE_ANCHOR)
    rep = R.fold_report(fams)

    print("\n--- GATE G0a ---")
    print("  nested %.16f  anchor %.16f  gap %.3e  (tol %s)" % (nested, NESTED_ANCHOR, g_n,
                                                                TOL_NESTED))
    print("  oracle %.16f  anchor %.16f  gap %.3e  (tol %s)" % (oracle, ORACLE_ANCHOR, g_o,
                                                                TOL_ORACLE))
    print("  folds %d | soils scored %d | family leaks %d"
          % (rep["n_folds"], rep["n_scored"], rep["family_leaks"]))
    ok = bool(g_n <= TOL_NESTED and g_o <= TOL_ORACLE and ok_copies
              and rep["family_leaks"] == 0 and rep["n_folds"] == 16 and rep["n_scored"] == 24)
    print("  G0a = %s" % ("PASS" if ok else "FAIL"))

    lines = ["=" * 78,
             "MODEL 11 / EXPERIMENT 1 -- reproduce_anchor.txt",
             "Generated: 2026-10-02 | Task 1 (G0a only). No model fitted, no prediction made.",
             "=" * 78, ""]
    for name, c in copies.items():
        lines.append("%-9s src %s" % (name, c["src_sha256"]))
        lines.append("%-9s dst %s  identical=%s" % (" " * 9, c["dst_sha256"], c["identical"]))
    lines.append("transfer_eval.py (read-only import) sha %s" % sha(te))
    lines.append("")
    lines.append("design: X %s  Y %s  families %d  soils %d"
                 % (X.shape, Y.shape, len(set(fams)), len(ids)))
    lines.append("nested %.16f  anchor %.16f  gap %.3e" % (nested, NESTED_ANCHOR, g_n))
    lines.append("oracle %.16f  anchor %.16f  gap %.3e" % (oracle, ORACLE_ANCHOR, g_o))
    lines.append("folds %d  scored %d  family_leaks %d"
                 % (rep["n_folds"], rep["n_scored"], rep["family_leaks"]))
    lines.append("G0a %s" % ("PASS" if ok else "FAIL"))
    lines.append("")
    lines.append("NEGATIVE CONFIRMATIONS (Task 1 scope)")
    lines.append("  - No new model fitted. No Weibull fit. No VLM call. No external data read.")
    lines.append("  - No image or tile pixel file opened.")
    lines.append("  - No CV arm scored beyond the anchor reproduction itself.")
    lines.append("  - No prediction, no submission CSV, no upload, no zip, no git commit.")
    lines.append("  - No Model 1-10 file modified; Model 9 not read.")
    (HERE / "reproduce_anchor.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    (HERE / "task1.json").write_text(json.dumps(
        {"copies": copies, "nested": nested, "oracle": oracle,
         "gap_nested": g_n, "gap_oracle": g_o, "fold_report": {k: v for k, v in rep.items()
                                                              if k != "fold_sizes"},
         "pass": ok}, indent=2), encoding="ascii")
    print("\nwrote reproduce_anchor.txt, task1.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
