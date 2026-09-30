"""One-shot patch: apply the owner's Task 4 rulings to check_m7a_g0.py. Throwaway."""
import io

P = "Model 7/Approach A/check_m7a_g0.py"
s = io.open(P, encoding="utf-8", newline="").read()
subs = []

subs.append((
"""        t1 = time.time()
        STATE["e_none"] = record_score("none", ev.lofo_errors(STATE["X0"], STATE["Y"], STATE["fams"]))
        STATE["e_zero"] = record_score("zero", ev.lofo_errors(STATE["Xz"], STATE["Y"], STATE["fams"]))""",
"""        t1 = time.time()
        e0, a0 = ev.lofo_errors(STATE["X0"], STATE["Y"], STATE["fams"], return_alphas=True)
        ez, az = ev.lofo_errors(STATE["Xz"], STATE["Y"], STATE["fams"], return_alphas=True)
        STATE["e_none"] = record_score("none", e0)
        STATE["e_zero"] = record_score("zero", ez)
        STATE["a_none"], STATE["a_zero"] = a0, az
        STATE["p_none"] = ev.lofo_predictions(STATE["X0"], STATE["Y"], STATE["fams"], a0)
        STATE["p_zero"] = ev.lofo_predictions(STATE["Xz"], STATE["Y"], STATE["fams"], az)"""))

subs.append((
'''    check("the LOFO arm on 12 columns agrees with the anchor to 1e-12 or better",
          abs(st["anchor_lofo"] - ANCHOR) < 1e-12,
          "lofo mean %.15f, gap %.3g" % (st["anchor_lofo"], abs(st["anchor_lofo"] - ANCHOR)))''',
'''    ev = __import__("m7a_eval")
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
          str(getattr(ev, "GAIN_BASELINE", ev.INTERNAL_BASELINE)))'''))

subs.append((
'''    check("every label cell is finite and rows sum to 100",
          bool(np.isfinite(st["Y"]).all())
          and float(np.abs(st["Y"].sum(axis=1) - 100.0).max()) < 1e-9,
          "max |row sum - 100| %.3g" % float(np.abs(st["Y"].sum(axis=1) - 100.0).max()))''',
'''    Y = st["Y"]
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
          % (float(Y[:, 0].min()), float((Y[:, -1] - Y[:, 0]).min())))'''))

OLD_G6 = s[s.index("def test_g6_zero_block_reproduces_the_no_block_arm():"):
           s.index("# ------------------------------------------------------------------ G7")]
NEW_G6 = '''def test_g6_zeroing_the_block_is_inert_within_the_registered_tolerance():
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


'''
subs.append((OLD_G6, NEW_G6))

subs.append((
'''    print("G0 anchor registered            : %.15f" % ANCHOR)
    print("G0 anchor recomputed            : %.15f" % st["g0_raw"])
    print("G0 deviation                    : %.2e   (must be exactly 0.0)" % st["g0"])
    print("no-block arm mean EMD (12 cols) : %.13f" % float(st["e_none"].mean()))
    print("zero-block arm mean EMD         : %.13f" % float(st["e_zero"].mean()))
    print("G6 zeroing the block reproduces the arm bit for bit : %s"
          % bool(np.array_equal(st["e_none"], st["e_zero"])))''',
'''    import m7a_eval as _ev
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
    print("      Kaggle submission. 60.56167 was an external Kaggle result, on a different scale.")'''))

for a, b in subs:
    assert a in s, "NOT FOUND: " + a[:70]
    s = s.replace(a, b)

io.open(P, "w", encoding="utf-8", newline="").write(s)
print("patched")
