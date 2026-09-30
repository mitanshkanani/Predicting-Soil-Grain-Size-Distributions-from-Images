"""Contract test for m7a_eval.py. Run: python check_m7a_g0.py

WHAT THIS FILE ASSERTS (Task 4 of the pre-registration: G0, G6, G7, fold integrity - nothing else)

  Terminology, because two internal numbers look similar and mean different things:
     43.0217308796477   HISTORICAL internal G0 anchor, produced by the frozen
                        Model 2/Model 5 no_holdout_control_score protocol. Not a Kaggle score.
     43.453225201811563 MODEL 7 internal no-block baseline under the strictly nested protocol this
                        run implements. Also not a Kaggle score. Gains use THIS one.
     60.56167           a previous EXTERNAL Kaggle submission result, a different scale entirely.
     future Kaggle score unknown until a submission CSV is generated and submitted.
  G0 anchor        the frozen protocol still reproduces 43.0217308796477 with deviation EXACTLY 0.0,
                   measured by calling the frozen Model 5 function, not a transcription of it. This
                   gate is about the historical anchor and is left exactly as registered.
  G6 inertness     as ruled: same 12 features, same folds, same alphas, same imported machinery,
                   max |prediction difference| <= 1e-12 and max/mean |EMD difference| <= 1e-12.
                   Bit-for-bit equality is NOT required, and the head was not modified to force it;
                   the measured 2.84e-14 residual is floating-point noise from four zero-variance
                   columns re-entering the scaler and the SVD.
  G7 honesty       the design carries no camera, device, ppm, sample-id or fold-label column; the
                   family labels are used as fold keys only; the block is the four Task 3 statistics
                   read through the published frame, and no test-split soil is in the matrix.
  fold integrity   16 CV families, 24 labelled soils, every soil scored exactly once, and no fold
                   ever fits on the family it scores.
  finiteness       every design cell finite for every block that Task 4 constructs.
  determinism      each design is rebuilt a second time and compared bit for bit, and a fresh
                   interpreter (started in main(), running while these legs run) hashes all three
                   designs identically.
  scope            the with-block arm is CONSTRUCTED and shape-checked, never SCORED. The gain, the
                   alias pre-gate, the four controls and the bootstrap belong to Tasks 5-8, and this
                   file proves by call-counting that lofo_errors was never asked about the real block.
                   Nothing printed here is a Kaggle claim; these are internal CV means on 24 labelled
                   soils, measured with a ruler whose per-soil SD is 22.75 EMD.

  Fitted arms in this run: "none" and "zero" only, plus the anchor's own nested protocol.
  Nothing here writes a model, a prediction, a submission or a gate verdict.

  Head imported, never reimplemented: tv.FEATURES, tv.ALPHAS, tv.PC_RANK, tv.fit_predict,
  tv.emd_pair, tv.labels_matrix, tv.image_features, tv.soil_from_images,
  tv.no_holdout_control_score. The only difference between the anchor and the gated arm is the
  presence of four columns, which is what makes the later verdicts distinguishable at all.

Cost: the frozen 12-feature assembly and the 1,946-tile A1-A4 assembly are each imported and cached
once (~3 min each, first call), then every arm reuses them; ~7 minutes wall clock total.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import m7a_data as d  # noqa: E402
import spatial_features as sf  # noqa: E402
import transfer_eval as tv  # noqa: E402

FAIL = []
TOTAL = 0
ANCHOR = 43.0217308796477
BLOCK = list(sf.SPATIAL)
FORBIDDEN_COLUMNS = ("camera", "camera_fam", "cv_group", "normalized_ppm", "tile_x", "tile_y",
                     "grid_col", "grid_row", "source_effective_ppm", "sample_id", "cv_family")
FRESH = None
SCORED = []
ARM_CALLS = []
STATE = {}


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
    """The expensive shared context: the three designs and the two fitted arms, built once."""
    import m7a_eval as ev
    if not STATE:
        t0 = time.time()
        STATE["X0"], STATE["Y"], STATE["fams"], STATE["ids"] = ev.soil_design(ROOT, "none")
        STATE["Xr"] = ev.soil_design(ROOT, "real")[0]
        STATE["Xz"] = ev.soil_design(ROOT, "zero")[0]
        t_design = time.time() - t0
        t1 = time.time()
        e0, a0 = ev.lofo_errors(STATE["X0"], STATE["Y"], STATE["fams"], return_alphas=True)
        ez, az = ev.lofo_errors(STATE["Xz"], STATE["Y"], STATE["fams"], return_alphas=True)
        STATE["e_none"] = record_score("none", e0)
        STATE["e_zero"] = record_score("zero", ez)
        STATE["a_none"], STATE["a_zero"] = a0, az
        STATE["p_none"] = ev.lofo_predictions(STATE["X0"], STATE["Y"], STATE["fams"], a0)
        STATE["p_zero"] = ev.lofo_predictions(STATE["Xz"], STATE["Y"], STATE["fams"], az)
        STATE["t_arms"] = time.time() - t1
        STATE["t_design"] = t_design
        STATE["g0_raw"] = float(tv.no_holdout_control_score(ROOT))
        STATE["g0"] = abs(STATE["g0_raw"] - ANCHOR)
        STATE["anchor_lofo"] = float(np.mean(STATE["e_none"]))
    return STATE


def record_score(block, errs):
    """Every lofo_errors call in this run is counted, so 'the real arm was not scored' is a fact."""
    SCORED.append(block)
    return errs


def arm_instead_of_none(ev):
    """Count every lofo_errors call, so 'the real arm was never scored' is observed, not claimed."""
    real = ev.lofo_errors

    def counted(X, Y, fams, *a, **k):
        ARM_CALLS.append(X.shape[1])
        return real(X, Y, fams, *a, **k)
    return counted, real


# ------------------------------------------------------------------ G0: the frozen anchor
def test_g0_anchor_is_exact():
    st = state()
    check("G0 anchor deviation is exactly 0.0 (not 1e-15, not 'close')", st["g0"] == 0.0,
          "gap %.2e  anchor %.15f" % (st["g0"], st["g0_raw"]))
    check("the anchor constant in m7a_eval is the registered 43.0217308796477",
          float(__import__("m7a_eval").ANCHOR) == ANCHOR, str(__import__("m7a_eval").ANCHOR))
    ev = __import__("m7a_eval")
    check("the strictly nested 12-column arm IS the Model 7 internal baseline, to 15 digits",
          repr(st["anchor_lofo"]) == repr(ev.INTERNAL_BASELINE),
          "measured %.15f, recorded %.15f" % (st["anchor_lofo"], ev.INTERNAL_BASELINE))
    check("internal baseline and historical anchor are different numbers from different protocols, "
          "and the 0.431 between them is recorded rather than hidden",
          ev.INTERNAL_BASELINE != ANCHOR
          and abs((st["anchor_lofo"] - ANCHOR) - 0.431494322163864) < 1e-12,
          "baseline - anchor = %+.15f" % (st["anchor_lofo"] - ANCHOR))
    check("a gain baseline is defined, and it is the nested one, never the historical anchor",
          getattr(ev, "GAIN_BASELINE", ev.INTERNAL_BASELINE) == ev.INTERNAL_BASELINE,
          str(getattr(ev, "GAIN_BASELINE", ev.INTERNAL_BASELINE)))
    check("exactly two arms were fitted in this run, and they are none and zero",
          SCORED.count("none") == 1 and SCORED.count("zero") == 1 and "real" not in SCORED
          and len(ARM_CALLS) == 2, "SCORED %s, lofo_errors calls %s" % (SCORED, ARM_CALLS))
    check("the 16-column design was built (the with-block arm exists) but never scored",
          st["Xr"].shape == (24, 16) and ARM_CALLS.count(16) == 1, str(ARM_CALLS))


# ------------------------------------------------------------------ the design
def test_design_shapes_and_types():
    st = state()
    ev = __import__("m7a_eval")
    check("no-block design is 24 x 12", st["X0"].shape == (24, 12), str(st["X0"].shape))
    check("with-block design is 24 x 16 (K = 4, exactly four columns added)",
          st["Xr"].shape == (24, 16), str(st["Xr"].shape))
    check("zero-block design is 24 x 16", st["Xz"].shape == (24, 16), str(st["Xz"].shape))
    check("labels are 24 x 11 (the frozen DIN-ISO supports)", st["Y"].shape == (24, 11),
          str(st["Y"].shape))
    check("24 distinct soils", len(set(st["ids"])) == 24, str(len(set(st["ids"]))))
    check("16 CV families over 24 soils", len(set(st["fams"])) == 16, str(len(set(st["fams"]))))
    check("designs are float arrays, so no label or fold key rode along as a column",
          all(arr.dtype.kind == "f" for arr in (st["X0"], st["Xr"], st["Xz"])),
          str([arr.dtype for arr in (st["X0"], st["Xr"], st["Xz"])]))
    check("the head is the imported 12-feature tuple, in order, not a transcription",
          tuple(ev.HEAD_COLUMNS) == tuple(tv.FEATURES), str(tuple(ev.HEAD_COLUMNS)))
    check("the block columns are exactly A1-A4", tuple(ev.BLOCK_COLUMNS) == tuple(BLOCK),
          str(tuple(ev.BLOCK_COLUMNS)))
    check("every design cell is finite in all three arms",
          all(bool(np.isfinite(arr).all()) for arr in (st["X0"], st["Xr"], st["Xz"])))
    Y = st["Y"]
    steps = np.diff(Y, axis=1)
    check("labels are cumulative curves: every cell finite", bool(np.isfinite(Y).all()))
    check("and monotone non-decreasing across the 11 supports", bool((steps >= -1e-12).all()),
          "min step %.3g" % float(steps.min()))
    check("and bounded within [0, 100]", float(Y.min()) >= 0.0 and float(Y.max()) <= 100.0,
          "range [%.4f, %.4f]" % (float(Y.min()), float(Y.max())))
    check("and every curve reaches exactly 100 at the last support",
          bool(np.all(Y[:, -1] == 100.0)), str(Y[:, -1][:3].tolist()))
    check("and no curve is degenerate: the first support is below 100 and the spread is real",
          bool((Y[:, 0] < 100.0).all()) and float((Y[:, -1] - Y[:, 0]).min()) > 0.0,
          "min first support %.3f, min spread %.3f"
          % (float(Y[:, 0].min()), float((Y[:, -1] - Y[:, 0]).min())))


def test_real_block_is_the_census_frame_and_nothing_else():
    st = state()
    ev = __import__("m7a_eval")
    published = pd.read_csv(HERE / "data" / "soil_features_m7a.csv", index_col=0,
                             float_precision="round_trip")
    in_process = d.spatial_soil_table(ROOT)
    check("the block in the design is the Task 3 published soil frame, bit for bit",
          np.array_equal(st["Xr"][:, 12:], published.loc[st["ids"], BLOCK].to_numpy(float)))
    check("and it equals the in-process assembly too (no silent recomputation drift)",
          np.array_equal(st["Xr"][:, 12:], in_process.loc[st["ids"], BLOCK].to_numpy(float)))
    check("the zero arm's block is exactly four columns of zeros",
          np.array_equal(st["Xz"][:, 12:], np.zeros((24, 4))))
    check("the zero arm's head is exactly the no-block head",
          np.array_equal(st["Xz"][:, :12], st["X0"]))
    check("the block columns are not all identical to one another or constant",
          all(st["Xr"][:, 12 + i].std() > 0 for i in range(4))
          and len({tuple(st["Xr"][:, 12 + i]) for i in range(4)}) == 4)


# ------------------------------------------------------------------ G6: inertness
def test_g6_zeroing_the_block_is_inert_within_the_registered_tolerance():
    """G6 as ruled: six conditions and a 1e-12 numerical tolerance, not bit-for-bit equality.

    The head, the PCA, the ridge and the projection were NOT touched to force bit equality. The
    measured residual is 2.84e-14, which is floating-point noise from four zero-variance columns
    re-entering the scaler and the SVD; the columns, folds, alphas and machinery are all identical,
    and that is what makes the tolerance a statement about numerics rather than about methodology.
    """
    st = state()
    ev = __import__("m7a_eval")
    e0, ez = st["e_none"], st["e_zero"]
    check("G6 (1/6) the same 12 original features, in the same order, bit for bit",
          np.array_equal(st["Xz"][:, :12], st["X0"])
          and tuple(ev.HEAD_COLUMNS) == tuple(tv.FEATURES))
    check("G6 (2/6) the same family folds: identical keys, identical membership, 16 of them",
          st["fams"] == STATE["fams"] and sorted(st["a_zero"]) == sorted(st["a_none"])
          and len(set(st["fams"])) == 16, str(len(set(st["fams"]))))
    check("G6 (3/6) the same alpha for every one of the 16 folds",
          st["a_zero"] == st["a_none"],
          "first folds %s" % str(list(st["a_none"].items())[:4])[:110])
    check("G6 (4/6) the head machinery is the imported one, not a copy of it",
          _routes_through_imported_head(ev))
    pmax = float(np.abs(st["p_zero"] - st["p_none"]).max())
    check("G6 (5/6) max absolute PREDICTION difference <= 1e-12", pmax <= 1e-12,
          "max |dcurve| %.3e over %d cells" % (pmax, st["p_zero"].size))
    emax = float(np.abs(ez - e0).max())
    check("G6 (6/6) per-soil EMD difference <= 1e-12 and mean difference <= 1e-12",
          emax <= 1e-12 and abs(float(ez.mean()) - float(e0.mean())) <= 1e-12,
          "max |dEMD| %.3e, |dmean| %.3e" % (emax, abs(float(ez.mean()) - float(e0.mean()))))
    check("the residual really is numerics: the two means agree when rounded to 12 digits",
          round(float(e0.mean()), 12) == round(float(ez.mean()), 12),
          "%.15f vs %.15f" % (float(e0.mean()), float(ez.mean())))
    check("and G6 asserts what the ruling says: both arms reproduce the INTERNAL baseline",
          abs(float(ez.mean()) - ev.INTERNAL_BASELINE) <= 1e-12
          and abs(float(e0.mean()) - ev.INTERNAL_BASELINE) <= 1e-12,
          "zero %.15f, none %.15f, baseline %.15f"
          % (float(ez.mean()), float(e0.mean()), ev.INTERNAL_BASELINE))
    print("       Model 7 internal baseline (12 cols, nested) : %.13f  [NOT a Kaggle score]"
          % float(e0.mean()))
    print("       same arm with the four columns zeroed       : %.13f" % float(ez.mean()))
    print("       historical G0 anchor (frozen protocol)      : %.13f" % ANCHOR)
    print("       previous EXTERNAL Kaggle submission result  : 60.56167  [different scale]")


def _routes_through_imported_head(ev):
    """Patch transfer_eval.fit_predict with a spy: if the arm still works and the spy fired once per
    fold, the head machinery really is the imported one. A local copy would not trip this."""
    seen = []
    real = tv.fit_predict

    def spy(Xtr, Ytr, Xte, al):
        seen.append(int(Xtr.shape[1]))
        return real(Xtr, Ytr, Xte, al)
    tv.fit_predict = spy
    try:
        ev.lofo_predictions(STATE["X0"], STATE["Y"], STATE["fams"], STATE["a_none"])
    finally:
        tv.fit_predict = real
    return len(seen) == 16 and set(seen) == {12}


# ------------------------------------------------------------------ G7: honesty / leakage
def test_g7_no_forbidden_column_reaches_any_design():
    st = state()
    ev = __import__("m7a_eval")
    names = list(ev.HEAD_COLUMNS) + list(ev.BLOCK_COLUMNS)
    hits = sorted({c for c in names for f in FORBIDDEN_COLUMNS if f in c.lower()})
    check("no camera, device, ppm, position, sample-id or fold-label name is a design column",
          not hits, str(hits or names))
    check("the family label list is a fold key, never a column",
          len(st["fams"]) == 24 and all(isinstance(f, str) for f in st["fams"]))
    check("no design column is constant across soils (a dead column would be a silent no-op)",
          all(st["X0"].std(axis=0) > 0) and all(st["Xr"][:, 12:].std(axis=0) > 0))
    q = d.tile_table(ROOT)
    test_soils = sorted(set(q[q.split == "test"].sample_id.astype(str)))
    check("no test-split soil is in the fitted design",
          not (set(test_soils) & set(st["ids"])), "%d test soils excluded" % len(test_soils))
    check("the design is the 24 labelled training soils and nothing else",
          len(st["ids"]) == 24 and set(st["ids"]) == set(q[q.split == "train"].sample_id
                                                         .astype(str)))
    check("labels come from the frozen labels_matrix, not a re-derived curve table",
          [str(i) for i in tv.labels_matrix(ROOT)[0]] == list(st["ids"]))


def test_fold_integrity():
    st = state()
    ev = __import__("m7a_eval")
    per, seen = ev.fold_report(st["Xr"], st["Y"], st["fams"])
    check("every soil appears in exactly one held-out fold",
          sorted(seen) == list(range(24)) and len(set(seen)) == 24, "%d scored" % len(seen))
    check("no fold ever fits on the family it scores", per["family_leaks"] == 0, str(per))
    check("16 folds for 16 families, and each fold's fit set excludes exactly its own family",
          per["n_folds"] == 16, str(per))
    F = np.asarray(st["fams"])
    sizes = sorted(int((F == g).sum()) for g in set(F))
    check("family sizes are the frozen 24/16 split: one four-soil family, five pairs, ten singletons",
          sizes == [1] * 10 + [2] * 5 + [4], str(sizes))
    check("14 soils have a sibling in the fitting set and 10 are alone in their family",
          sum(x for x in sizes if x > 1) == 14 and sum(1 for x in sizes if x == 1) == 10,
          str(sizes))
    check("every held-out fold leaves at least two families to fit on (alpha needs a selection set)",
          all(len({F[i] for i in range(24) if F[i] != g}) >= 2 for g in set(F)), str(len(set(F))))


# ------------------------------------------------------------------ determinism
def test_designs_are_rebuilt_identically():
    ev = __import__("m7a_eval")
    st = state()
    for block, key in (("none", "X0"), ("real", "Xr"), ("zero", "Xz")):
        X, Y, fams, ids = ev.soil_design(ROOT, block)
        check("the %s design rebuilds bit for bit in this process" % block,
              np.array_equal(X, st[key]) and np.array_equal(Y, st["Y"]) and fams == st["fams"]
              and ids == st["ids"])
    other = ev.soil_design(ROOT, "real")[0]
    check("the design hash is order-sensitive, so a column permutation is not silently tolerated",
          hashlib.sha256(other.tobytes()).hexdigest()
          != hashlib.sha256(other[:, ::-1].tobytes()).hexdigest())
    check("an unknown block name raises instead of defaulting to something safe",
          isinstance(caught(ev.soil_design, ROOT, "shuffle"), AssertionError),
          str(caught(ev.soil_design, ROOT, "shuffle"))[:80])


def test_fresh_interpreter_hashes_the_same_designs():
    global FRESH
    if FRESH is None:
        check("a fresh interpreter was started for the second-process check", False, "not started")
        return
    out, err = FRESH.communicate(timeout=1800)
    text = out.decode("utf-8", "replace").strip()
    check("the fresh interpreter exited 0", FRESH.returncode == 0, (err or b"").decode()[-300:])
    if not text.startswith("M7A-G0"):
        check("the fresh interpreter reported the design hashes", False, text[:120])
        return
    got = dict(kv.split("=") for kv in text.split("|")[1:])
    want = {k: hashlib.sha256(state()[k].tobytes()).hexdigest()[:16] for k in ("X0", "Xr", "Xz")}
    for k in ("X0", "Xr", "Xz"):
        check("fresh-interpreter design %s matches this process" % k, got.get(k) == want[k],
              "%s vs %s" % (got.get(k, "missing"), want[k]))
    check("the three designs hash differently from one another (they are three distinct matrices)",
          len(set(want.values())) == 3, str(want))


# ------------------------------------------------------------------ the last check stays last
def test_every_check_passed():
    check("all checks above passed", not FAIL, "%d of %d failed" % (len(FAIL), TOTAL + 1))
    assert not FAIL, "%d check(s) failed: %s" % (len(FAIL), ", ".join(FAIL[:6]))


def summary():
    st = STATE
    line = "=" * 78
    print("\n" + line)
    print("M7-A TASK 4 - G0 anchor, G6 inertness, G7 design integrity, fold integrity")
    print(line)
    import m7a_eval as _ev
    print("HISTORICAL G0 anchor (internal, frozen protocol) : %.15f" % ANCHOR)
    print("HISTORICAL G0 recomputed                         : %.15f" % st["g0_raw"])
    print("HISTORICAL G0 deviation                          : %.2e  (must be exactly 0.0)"
          % st["g0"])
    print("MODEL 7 internal baseline (nested, 12 cols)      : %.15f"
          % float(st["e_none"].mean()))
    print("MODEL 7 internal baseline recorded in m7a_eval   : %.15f" % _ev.INTERNAL_BASELINE)
    print("protocol difference between the two              : %+.15f"
          % (float(st["e_none"].mean()) - ANCHOR))
    print("zero-block arm mean EMD                          : %.15f" % float(st["e_zero"].mean()))
    print("G6 max |dcurve| / max |dEMD| / |dmean|           : %.3e / %.3e / %.3e"
          % (float(np.abs(st["p_zero"] - st["p_none"]).max()),
             float(np.abs(st["e_zero"] - st["e_none"]).max()),
             abs(float(st["e_zero"].mean()) - float(st["e_none"].mean()))))
    print("NOTE 43.0217 and 43.4532 are INTERNAL CV numbers; the only external evidence is a")
    print("      Kaggle submission. 60.56167 was an external Kaggle result, on a different scale.")
    print("designs                         : none 24x12, real 24x16 (built, NOT scored), zero 24x16")
    print("arms fitted in this run          : %s (lofo_errors called %d times: %s columns)"
          % (SCORED, len(ARM_CALLS), ARM_CALLS))
    print("CV protocol                      : leave-one-CV-family-out, 16 families / 24 soils,")
    print("                                   alpha re-selected inside each outer training set")
    print("per-soil errors (no-block arm)   : %s" % np.round(st["e_none"], 4).tolist())
    print("per-soil errors (zeroed block)   : %s" % np.round(st["e_zero"], 4).tolist())
    print("timing                           : designs %.0f s, fitted arms %.0f s"
          % (st["t_design"], st["t_arms"]))
    print(line)


def main():
    global FRESH
    wanted = [a.lower() for a in sys.argv[1:]]
    if not wanted:
        snippet = ("import hashlib,sys,numpy as np;sys.path.insert(0,r'%s');"
                   "sys.path.insert(0,r'%s');import m7a_eval as ev;"
                   "print('M7A-G0','|'.join('%%s=%%s'%%(k,hashlib.sha256("
                   "ev.soil_design(ev.ROOT,b)[0].tobytes()).hexdigest()[:16]) "
                   "for k,b in (('X0','none'),('Xr','real'),('Xz','zero'))),sep='|')"
                   % (str(HERE), str(ROOT / "Model 5" / "Model 5 Experiment 1")))
        FRESH = subprocess.Popen([sys.executable, "-c", snippet], cwd=str(HERE),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ev = __import__("m7a_eval")
    real_lofo = ev.lofo_errors
    ev.lofo_errors = arm_instead_of_none(ev)[0]
    fns = sorted(((obj.__code__.co_firstlineno, name, obj)
                  for name, obj in list(globals().items())
                  if name.startswith("test_") and callable(obj)))
    if wanted:
        fns = [f for f in fns if any(w in f[1].lower() for w in wanted)]
        print("FILTERED RUN: %d selected by %s" % (len(fns), wanted))
    try:
        for _, name, fn in fns:
            try:
                fn()
            except Exception as exc:
                check("%s raised %s" % (name, type(exc).__name__), False, str(exc)[:90])
    finally:
        ev.lofo_errors = real_lofo
    if not wanted:
        summary()
    print("\n%d checks, %d failed" % (TOTAL, len(FAIL)))
    for f in FAIL:
        print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
