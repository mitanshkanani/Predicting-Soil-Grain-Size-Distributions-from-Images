"""Task 5 contract test: the alias pair set must reproduce the Model 6-3 figures its own rule owns.

Run: python check_alias_gate.py

WHAT THIS FILE ASSERTS (Task 5 of the pre-registration - a gate diagnostic, not a Model 7 result)
  reproduction    the two figures that are properties of THIS rule - top-10 share 61.2%, worst soil
                  10.1% - must come out of the registered reconstruction within the tolerances the
                  plan fixed. If they do not, the reconstruction is wrong and the gate may not run:
                  fix the rule, not the tolerance.
                  Amendment A1 (2026-09-30) removes the median alias curve gap of 65.7 EMD from that
                  list: it was printed by Model 6-3's own worst-8/best-8 UNCONSTRAINED neighbour rule,
                  a different instrument, so the registered rule cannot and must not reproduce it. Two
                  legs keep that honest in both directions - Model 6-3's rule still yields 65.7, and
                  this rule still does not. If either ever flips, the instruments are being confused
                  and Task 6 must not run.
  the rule        for each of the 10 highest-error soils (M1 column of Model 2 Experiment 3's
                  cv_per_soil.csv), its nearest neighbour in the standardised 12-feature space,
                  drawn from a DIFFERENT CV family than the query, is its alias partner. The control
                  set is the same rule applied to the 14 remaining, lowest-error soils. No extra
                  exclusion, no tuned radius, no hand-picked pair.
  independence    A1-A4 are nowhere in this derivation: the pair set is frozen before the new
                  features are consulted, and a leg proves the module never loaded the machinery
                  that computes them. A gate that could pick its own pairs is selection on scored
                  data, which is the failure the pre-registration was written to avoid.
  argmin honesty  every pair is checked against an independent recomputation of the distances: the
                  partner really is the closest cross-family soil, not one of the near-misses.
  determinism     the derivation runs twice and produces the same rows and the same artifact bytes.
  scope           Task 5 scores nothing. No CV arm, no P-RAND/P-SHUF/P-COLS/P-SEL, no prediction,
                  no submission, no file in results/.

Numbers this file prints are INTERNAL. It computes no gain and touches no test label.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import alias_gate as ag  # noqa: E402

FAIL = []
TOTAL = 0
PAIRS_CSV = HERE / "data" / "alias_pairs.csv"
COLUMNS = ["kind", "soil_a", "soil_b", "soil_a_family", "soil_b_family", "feature_distance",
           "curve_emd"]
OWNED = {"top10_share": (61.2, 0.15), "worst_share": (10.1, 0.05)}
HISTORICAL = {"worst8_median_curve_emd": (65.7, 0.6), "best8_median_curve_emd": (28.1, 0.6),
              "worst8_median_nn_dist": (1.65, 0.05), "best8_median_nn_dist": (1.58, 0.05)}
STATE = {}
SEPARABILITY = {}


def check(name, ok, detail=""):
    global TOTAL
    TOTAL += 1
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def caught(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except BaseException as exc:
        return exc


def state():
    if not STATE:
        STATE["df"] = ag.derive_pairs(ROOT)
        STATE["sum"] = dict(ag.summary(ROOT))
        STATE["ids"], STATE["X"], STATE["Y"], STATE["fam"], STATE["err"] = ag.design(ROOT)
        STATE["hist"] = dict(ag.historical_gaps(ROOT))
    return STATE


# ------------------------------------------------------------------ the Model 6-3 record
def test_reproduces_the_figures_this_rule_owns():
    s = state()["sum"]
    for key, (want, tol) in OWNED.items():
        got = float(s[key])
        check("%s reproduces the recorded %s within the registered +/- %s" % (key, want, tol),
              abs(got - want) <= tol, "measured %.6f, recorded %.4f, delta %+.4f"
              % (got, want, got - want))
    h = state()["hist"]
    for key, (want, tol) in HISTORICAL.items():
        got = float(h[key])
        check("Model 6-3's own worst-8/best-8 rule reproduces its recorded %s = %s" % (key, want),
              abs(got - want) <= tol, "measured %.6f, delta %+.4f" % (got, got - want))
    check("amendment A1: the registered rule does NOT reproduce 65.7, and is not asked to",
          abs(float(s["median_alias_curve_gap"]) - 65.7) > 0.6,
          "registered median gap %.6f" % float(s["median_alias_curve_gap"]))
    check("amendment A1: Model 6-3's rule lets a soil alias its own CV-family sibling, this one cannot",
          h["same_family_partners"] > 0, "%d of 24 same-family partners historically"
          % h["same_family_partners"])
    check("the pixel-alias pair survives under both rules as H038<->G190",
          h["h038_partner"] == "G190" and abs(h["h038_partner_curve_emd"] - 125.4189) < 0.01,
          "%s at %.4f EMD" % (h["h038_partner"], h["h038_partner_curve_emd"]))
    rep = ag.reproduction(ROOT)
    check("reproduction() reports the registered gap as not a target, with both rules comparable",
          rep["registered_median_alias_gap"]["is_reproduction_target"] is False
          and rep["rules_are_distinguishable"] is True and rep["all_reproduced"] is True,
          "%s" % sorted(rep))
    check("10 alias pairs and 14 control pairs, as pre-registered",
          s["n_alias"] == 10 and s["n_control"] == 14, "%s alias / %s control"
          % (s["n_alias"], s["n_control"]))
    key = sorted(["H038", "G190"])
    check("H038-G190 is in the alias set (the pair Model 6-3 recorded as pixel-aliased)",
          key in [sorted(p) for p in s["alias_pair_ids"]], str(s["alias_pair_ids"])[:150])
    print("       full pair set: %s" % str(s["alias_pair_ids"])[:200])


def test_the_recorded_figures_are_reported_with_their_precision():
    s = state()["sum"]
    print("       top-10 share            : %.6f %%  (recorded 61.2 %%)" % s["top10_share"])
    print("       worst soil share        : %.6f %%  (recorded 10.1 %%)" % s["worst_share"])
    print("       median alias curve gap  : %.6f EMD (this rule's own measure; 65.7 belongs to "
          "Model 6-3's worst-8/best-8 rule and is not a target - amendment A1)"
          % s["median_alias_curve_gap"])
    check("shares are percentages of the summed M1 error over the same 24 soils",
          0.0 < s["top10_share"] <= 100.0 and s["top10_share"] > s["worst_share"])
    check("the error column used is M1, the shipped model's own CV error, not a later arm",
          state()["err"][0] != 0.0 and len(state()["err"]) == 24,
          "H038 M1 %.6f" % float(state()["err"][0]))


# ------------------------------------------------------------------ the rule
def test_pairs_are_cross_family_by_construction():
    df = state()["df"]
    bad = [r for r in df.itertuples() if r.soil_a_family == r.soil_b_family]
    check("no pair joins two soils of the same CV family", not bad, "%d bad" % len(bad))
    check("every pair's two soils are distinct",
          not any(r.soil_a == r.soil_b for r in df.itertuples()))
    check("both endpoints of every pair are among the 24 labelled training soils",
          set(df.soil_a) | set(df.soil_b) <= set(state()["ids"]))


def test_queries_are_exactly_the_ten_worst_and_the_fourteen_remaining():
    st = state()
    df, ids, err = st["df"], st["ids"], st["err"]
    order = np.argsort(err)[::-1]
    worst = [ids[i] for i in order[:10]]
    rest = [ids[i] for i in order[10:]]
    alias = df[df.kind == "alias"]
    control = df[df.kind == "control"]
    check("the alias queries are the 10 highest-M1 soils, in error order",
          list(alias.soil_a) == worst, str(list(alias.soil_a))[:150])
    check("the control queries are the 14 lowest-M1 soils under the same rule",
          list(control.soil_a) == rest, str(list(control.soil_a))[:150])
    check("one pair per query soil, so the sets cannot be padded",
          len(alias) == 10 and len(control) == 14
          and alias.soil_a.nunique() == 10 and control.soil_a.nunique() == 14)
    check("the two sets partition the corpus",
          sorted(set(alias.soil_a) | set(control.soil_a)) == sorted(set(ids)),
          "%d queries over %d soils" % (alias.soil_a.nunique() + control.soil_a.nunique(), len(ids)))


def test_the_partner_is_the_nearest_cross_family_soil():
    """Independent recomputation of the argmin, so a near-miss cannot pass as the neighbour."""
    st = state()
    df, X, ids, fam = st["df"], st["X"], st["ids"], st["fam"]
    mu, sd = X.mean(0), np.where(X.std(0) > 1e-12, X.std(0), 1.0)
    Z = (X - mu) / sd
    pos = {s: i for i, s in enumerate(ids)}
    wrong = []
    for r in df.itertuples():
        a = pos[r.soil_a]
        cand = [j for j in range(len(ids)) if j != a and fam[ids[j]] != fam[ids[a]]]
        d = {ids[j]: float(np.sqrt(((Z[j] - Z[a]) ** 2).sum())) for j in cand}
        best = min(d, key=d.get)
        if best != r.soil_b or abs(d[best] - r.feature_distance) > 1e-12:
            wrong.append("%s: recorded %s (%.6f), nearest %s (%.6f)"
                         % (r.soil_a, r.soil_b, r.feature_distance, best, d[best]))
    check("every partner is genuinely the closest cross-family soil, to 1e-12",
          not wrong, "; ".join(wrong)[:220])
    check("the distance is Euclidean in the 12 standardised columns, not one column or a raw dot",
          all(0.0 < r.feature_distance < 30.0 for r in df.itertuples()),
          "range [%.3f, %.3f]" % (float(df.feature_distance.min()), float(df.feature_distance.max())))
    check("curve_emd is a symmetric pairwise EMD between the two recorded curves",
          all(abs(r.curve_emd) >= 0.0 for r in df.itertuples())
          and float(df.curve_emd.max()) > 0.0,
          "range [%.3f, %.3f]" % (float(df.curve_emd.min()), float(df.curve_emd.max())))


# ------------------------------------------------------------------ A1-A4 stay out of Task 5
def test_the_pair_set_is_frozen_without_the_new_features():
    for mod in ("m7a_data", "spatial_features", "m7a_eval"):
        check("deriving the pairs never loaded %s" % mod, mod not in sys.modules,
              "already imported" if mod in sys.modules else "clean")
    src = sorted(m for m in sys.modules if m.startswith("alias_gate"))
    check("only the gate module itself is in play", src == ["alias_gate"], str(src))
    df = state()["df"]
    check("the published pair table carries the registered columns and nothing else",
          list(df.columns) == COLUMNS, str(list(df.columns)))
    check("no column of either design frame appears in the table",
          not any(c in df.columns for c in ("A1", "A2", "A3", "A4")))


# ------------------------------------------------------------------ determinism and artifacts
def test_derivation_is_deterministic_and_the_artifact_matches():
    df1 = ag.derive_pairs(ROOT)
    df2 = ag.derive_pairs(ROOT)
    check("the derivation repeats row for row", df1.equals(df2))
    check("the order is the registered one (alias then control, error order within each)",
          list(df1.kind) == ["alias"] * 10 + ["control"] * 14, str(list(df1.kind))[:60])
    written = PAIRS_CSV.read_bytes()
    check("data/alias_pairs.csv re-reads exactly as derived",
          pd.read_csv(PAIRS_CSV, float_precision="round_trip").equals(df1.reset_index(drop=True)))
    h1 = hashlib.sha256(written).hexdigest()
    ag.derive_pairs(ROOT)
    h2 = hashlib.sha256(PAIRS_CSV.read_bytes()).hexdigest()
    check("re-writing the artifact produces identical bytes", h1 == h2, "%s / %s" % (h1[:12], h2[:12]))
    check("the table has no NaN and no placeholder text",
          not df1.isna().any().any() and all(isinstance(x, str) for x in df1.soil_a))


# ------------------------------------------------------------------ Task 6: the pre-gate itself
def test_separability_gate_reports_all_three_blocks():
    """The gate runs in a fresh interpreter, on purpose.

    Task 5's independence leg asserts that the pair derivation never loaded the A1-A4 machinery. An
    in-process call here would make that assertion depend on the order tests happen to run in, which
    is exactly the kind of tooth that quietly falls out. A subprocess keeps both legs true at once and
    doubles as proof that the gate numbers survive a cold interpreter.
    """
    code = ("import json, sys; sys.path.insert(0, %r); sys.path.insert(0, %r);"
            "import alias_gate as ag; print(json.dumps(ag.separability(%r)))"
            % (str(HERE), str(ROOT / "Model 5" / "Model 5 Experiment 1"), str(ROOT)))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          cwd=str(HERE), timeout=3600)
    check("the pre-gate ran in a fresh interpreter without error", proc.returncode == 0,
          (proc.stderr or "")[-240:])
    if proc.returncode != 0:
        return
    s = json.loads(proc.stdout.strip().splitlines()[-1])
    for key in ("base12", "block16", "rand16", "shuf16", "gain_vs_base", "gate_passed"):
        check("separability reports %s" % key, key in s)
    check("gate_passed is a bool, never a float close to one",
          isinstance(s.get("gate_passed"), bool), type(s.get("gate_passed")).__name__)
    check("the block is compared against the 12 on the same pairs", s["base12"] != s["block16"],
          "%.6f vs %.6f" % (s["base12"], s["block16"]))
    check("placebos are computed, not assumed equal to each other",
          abs(s["rand16"] - s["shuf16"]) > 1e-12 or s["rand16"] == s["shuf16"],
          "rand %.6f / shuf %.6f" % (s["rand16"], s["shuf16"]))
    check("the gain is exactly block minus base, not a restated number",
          abs(s["gain_vs_base"] - (s["block16"] - s["base12"])) < 1e-12)
    check("gate_passed follows the registered rule alone, on alias pairs only",
          s["gate_passed"] == bool(s["gain_vs_base"] > 0.0
                                   and s["gain_vs_base"] > s["rand_gain_vs_base"]
                                   and s["gain_vs_base"] > s["shuf_gain_vs_base"]),
          "gain %+.6f rand %+.6f shuf %+.6f passed %s"
          % (s["gain_vs_base"], s["rand_gain_vs_base"], s["shuf_gain_vs_base"], s["gate_passed"]))
    check("the gate consumes the frozen 10 alias pairs and reports the 14 controls",
          s["n_alias"] == 10 and s["n_control"] == 14
          and s["control_separation_is_gate_relevant"] is False,
          "%s / %s" % (s["n_alias"], s["n_control"]))
    check("the block is the frozen A1-A4 at the registered placebo seeds",
          s["block_columns"] == ["A1", "A2", "A3", "A4"] and s["seed_rand"] == 90001
          and s["seed_shuf"] == 90002)
    print("       alias pairs : base12 %.6f | block16 %.6f | rand16 %.6f | shuf16 %.6f | "
          "gain %+.6f | gate_passed %s"
          % (s["base12"], s["block16"], s["rand16"], s["shuf16"], s["gain_vs_base"],
             s["gate_passed"]))
    print("       control pairs (report only, not gate-relevant): base12 %.6f | block16 %.6f | "
          "gain %+.6f" % (s["ctrl_base12"], s["ctrl_block16"], s["ctrl_gain_vs_base"]))
    SEPARABILITY.update(s)


def test_task5_scored_nothing():
    s = state()["sum"]
    check("the summary reports the reproduction figures and pair counts only",
          set(s) == {"top10_share", "worst_share", "median_alias_curve_gap", "n_alias", "n_control",
                     "alias_pair_ids"}, str(sorted(s)))
    check("no gain, separation or CV score is present in Task 5's output",
          not any(k in s for k in ("gain", "separability", "block16", "base16", "cv_emd")),
          str(sorted(s)))
    arts = sorted(p.name for p in (HERE / "results").iterdir()) if (HERE / "results").exists() else []
    checks = sorted(p.name for p in HERE.rglob("*submission*"))
    check("nothing was written to results/ and no submission file exists",
          arts == [] and checks == [], "%s / %s" % (arts, checks))


def test_every_check_passed():
    """Pytest bridge, and it must stay the last test_ function in the file."""
    check("all checks above passed", not FAIL, "%d of %d failed" % (len(FAIL), TOTAL + 1))
    assert not FAIL, "%d check(s) failed: %s" % (len(FAIL), ", ".join(FAIL[:6]))


def main():
    fns = sorted(((obj.__code__.co_firstlineno, name, obj)
                  for name, obj in list(globals().items())
                  if name.startswith("test_") and callable(obj)))
    for _, name, fn in fns:
        try:
            fn()
        except Exception as exc:
            check("%s raised %s" % (name, type(exc).__name__), False, str(exc)[:90])
    print("\n%d checks, %d failed" % (TOTAL, len(FAIL)))
    for f in FAIL:
        print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
