"""Contract test for crop_geometry.py - run from the experiment folder:

    python check_crop_geometry.py

E1's lesson, learned the hard way: a section guarded by a condition the dry run never
satisfies is untested code, and an ``ndarray.median()`` survived straight to Kaggle because
of it. So this file pins the numbers that the plan's central finding rests on. If someone
edits ``CROPS`` or the geometry arithmetic, these assertions fail locally instead of quietly
changing what the experiment measures.

No pytest (not installed). Plain asserts, printed labels, non-zero exit on failure.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

import crop_geometry as cg

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


PPM = 4.552516


def test_no_torch_in_the_module():
    """The module's whole reason to exist separately is that it loads without torch.
    Checked against the source, not by trying to import it, because the import would
    succeed on Kaggle either way and the check would stop meaning anything."""
    src = (pathlib.Path(__file__).with_name("crop_geometry.py")).read_text(encoding="utf-8")
    bad = [ln.strip() for ln in src.splitlines()
           if ln.strip().startswith(("import torch", "from torch"))]
    check("crop_geometry.py contains no torch import", not bad, str(bad))
    check("INPUT_SIZE and PATCH_PX are pinned", (cg.INPUT_SIZE, cg.PATCH_PX) == (224, 14))


def test_sweep_is_what_the_plan_says():
    check("CROPS is exactly the pre-registered sweep", cg.CROPS == (256, 128, 64, 32),
          str(cg.CROPS))
    check("every crop is a multiple of 4 (the mock box-downsample)",
          all(c % 4 == 0 for c in cg.CROPS))
    check("every crop fits inside a 256 px tile", all(c <= 256 for c in cg.CROPS))
    check("crop 256 is present - it is E1's regression check, not an arm",
          256 in cg.CROPS)


def test_centred_and_exact():
    a = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(1, 8, 8, 3)
    c4 = cg.crop_tile(a, 4)
    check("crop shape is (N,c,c,3)", c4.shape == (1, 4, 4, 3), str(c4.shape))
    check("dtype is preserved (no silent cast)", c4.dtype == a.dtype, str(c4.dtype))
    check("crop is centred, not top-left",
          np.array_equal(c4[0, 0, 0], a[0, 2, 2]),
          "offset should be (8-4)//2 = 2")
    check("full-size crop is the identity", np.array_equal(cg.crop_tile(a, 8), a))
    big = np.arange(2 * 6 * 6 * 3, dtype=np.float64).reshape(2, 6, 6, 3)
    out = cg.crop_tile(big, 4)
    check("crop works on a float batch unchanged", out.dtype == big.dtype and out.shape == (2, 4, 4, 3))
    check("crop of a batch keeps every row independent",
          np.array_equal(out[0], big[0, 1:5, 1:5]) and np.array_equal(out[1], big[1, 1:5, 1:5]))


def test_rejects_bad_crops():
    a = np.zeros((1, 8, 8, 3), dtype=np.uint8)

    def raises(fn, *args):
        try:
            fn(*args)
        except Exception:
            return True
        return False

    check("2-D input rejected", raises(cg.crop_tile, np.zeros((8, 8)), 4))
    check("crop larger than the tile rejected", raises(cg.crop_tile, a, 16))
    check("non-multiple-of-4 crop rejected", raises(cg.crop_tile, a, 5))
    check("zero crop rejected", raises(cg.crop_tile, a, 0))


def test_geometry_matches_the_measured_sweep():
    """These are the numbers the plan's section 1 measured on real tiles. They are the
    experiment's premise: magnification barely helps."""
    want = {256: 3.5145, 128: 1.7573, 64: 0.8786, 32: 0.4393}
    for c, mm in want.items():
        got = cg.patch_mm(c, PPM)
        check(f"patch at crop {c:3d} is {mm:.4f} mm", abs(got - mm) < 5e-4, f"got {got:.4f}")
    check("window at crop 256 is the tile's 56.23 mm", abs(cg.window_mm(256, PPM) - 56.23) < 0.01)


def test_e1_recorded_the_wrong_number_on_purpose():
    """Pinned so nobody 'fixes' it back. E1 recorded 3.0752 mm by omitting the 256->224
    resize; backbones.py is left as authored so E1 stays byte-reproducible, and E2 computes
    the correct value here instead."""
    naive = 14 / PPM
    check("E1's naive value reproduces 3.0752", abs(naive - 3.0752) < 1e-3, f"{naive:.4f}")
    check("the true value is materially larger", cg.patch_mm(256, PPM) - naive > 0.4,
          f"gap {cg.patch_mm(256, PPM) - naive:.4f} mm")
    check("the gap is exactly the resize factor", abs(cg.patch_mm(256, PPM) / naive - 256 / 224)
          < 1e-9, f"ratio {cg.patch_mm(256, PPM) / naive:.4f}")


def test_reachability_is_bounded():
    """The finding that bounds this whole experiment, asserted rather than narrated."""
    d50 = np.array([0.021, 0.022, 0.030, 0.030, 0.035, 0.059, 0.062, 0.068, 0.099,
                    0.103, 0.211, 0.814, 1.795, 2.434, 2.535, 3.066, 3.674, 4.337,
                    4.585, 4.791, 4.792, 5.339, 5.911, 6.126])
    got = {c: cg.resolvability(d50, c, PPM) for c in cg.CROPS}
    check("crop 256 resolves 12 of 24 soils", got[256] == 12, str(got))
    check("crop 32 resolves 14 of 24 soils", got[32] == 14, str(got))
    check("the whole sweep gains at most 3 soils", got[32] - got[256] <= 3,
          f"gain {got[32] - got[256]}")
    check("resolvability never decreases as the crop shrinks",
          all(got[cg.CROPS[i]] >= got[cg.CROPS[i - 1]] for i in range(1, len(cg.CROPS))))
    fine = int((d50 < 0.11).sum())
    check("ten soils are fines and stay unresolvable across the sweep", fine == 10,
          f"{fine} soils below 0.11 mm")
    need_px = d50[d50 < 0.11].max() * 3 * PPM * cg.INPUT_SIZE / cg.PATCH_PX
    check("resolving the largest fine soil needs a sub-24 px window", need_px < 24,
          f"{need_px:.1f} px -> {need_px * need_px:.0f} real pixels upsampled to 224")


def test_soil_groups_match_the_data_s_bimodality():
    d50 = [0.021, 0.103, 0.211, 0.814, 1.795, 6.126]
    groups = [cg.soil_group(d) for d in d50]
    check("group cut points are as declared",
          groups == ["fines", "fines", "mid", "mid", "coarse", "coarse"], str(groups))


def test_describe_is_computed_not_hardcoded():
    d50 = np.array([0.021, 0.103, 0.211, 0.814, 1.795, 6.126])
    txt = cg.describe(d50, PPM)
    check("describe() emits one line per crop plus a header",
          len(txt.strip().splitlines()) == len(cg.CROPS) + 1)
    check("describe() carries the corrected patch size", "3.5145" in txt)


print("\ncrop_geometry contract")
for fn in [test_no_torch_in_the_module, test_sweep_is_what_the_plan_says,
           test_centred_and_exact, test_rejects_bad_crops,
           test_geometry_matches_the_measured_sweep,
           test_e1_recorded_the_wrong_number_on_purpose, test_reachability_is_bounded,
           test_soil_groups_match_the_data_s_bimodality,
           test_describe_is_computed_not_hardcoded]:
    print(f"\n{fn.__name__}")
    fn()

print("\n" + "=" * 72)
if FAILS:
    print(f"check_crop_geometry: {len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("check_crop_geometry: ALL CHECKS PASSED")
print("""
Scope: this verifies the crop and the arithmetic, on synthetic arrays, with no torch. It
does not verify anything about the backbone - that is check_backbones.py's job, and both
are run by the notebook before a single tile is processed.""")
