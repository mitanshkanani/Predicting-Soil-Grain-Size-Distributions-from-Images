"""Contract test for fusion.py - run from the experiment folder:

    python check_fusion.py

E3's entire interpretation rests on two operations in fusion.py: a concatenation that must
not silently drop or duplicate a block, and a row permutation that must destroy the
soil-to-embedding correspondence while leaving every column's statistics untouched. Both
are positional, so a reordered index would pass a set comparison and pair every soil with
the wrong embedding - the kind of bug that produces a clean-looking wrong answer.

E1's lesson applies: a section guarded by a condition the dry run never satisfies is
untested code. This file pins the numbers the plan's claims rest on, so an edit to the
fusion width, the column convention or the permutation seed fails locally instead of
quietly changing what E3 measures.

No pytest (not installed). Plain asserts, printed labels, non-zero exit on failure.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import numpy as np
import pandas as pd

import fusion as fu

FAILS = []
HAND = [f"h{i:02d}" for i in range(12)]
PERM_SEED = 20260933


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def _tables(n=24, dim=384, seed=7):
    rng = np.random.default_rng(seed)
    idx = [f"S{i:02d}" for i in range(n)]
    hand = pd.DataFrame(rng.normal(size=(n, 12)), index=idx, columns=HAND)
    emb = pd.DataFrame(rng.normal(size=(n, dim)), index=idx, columns=fu.emb_cols(dim))
    return hand, emb


def test_no_torch_in_the_module():
    """fusion.py exists to be checkable without a GPU. Verified against the source, not by
    importing, because the import would succeed on any machine and the check would stop
    meaning something."""
    src = (pathlib.Path(__file__).with_name("fusion.py")).read_text(encoding="utf-8")
    bad = [ln.strip() for ln in src.splitlines()
           if ln.strip().startswith(("import torch", "from torch", "import timm",
                                     "from timm"))]
    check("fusion.py contains no torch or timm import", not bad, str(bad))
    check("EMBED_DIM is pinned at 384", fu.EMBED_DIM == 384, str(fu.EMBED_DIM))


def test_column_convention_is_e2s():
    """E3 must reuse E2's embedding column names so a fused matrix can be diffed against
    E2's D matrix and nothing but the concatenation shows up."""
    cols = fu.emb_cols()
    check("384 columns", len(cols) == 384, str(len(cols)))
    check("first is emb_000", cols[0] == "emb_000", cols[0])
    check("last is emb_383", cols[-1] == "emb_383", cols[-1])
    check("names are unique", len(set(cols)) == len(cols))
    check("matches E2's f'emb_{i:03d}' convention",
          cols == [f"emb_{i:03d}" for i in range(384)])


def test_fuse_width_and_order():
    hand, emb = _tables()
    out = fu.fuse(hand, emb, HAND, fu.emb_cols())
    check("fused width is exactly 12 + 384 = 396", out.shape[1] == 396, str(out.shape[1]))
    check("fused rows unchanged", out.shape[0] == 24, str(out.shape[0]))
    check("first 12 columns are the hand block, in order",
          list(out.columns[:12]) == HAND)
    check("last 384 columns are the embedding block, in order",
          list(out.columns[12:]) == fu.emb_cols())
    check("hand values survived the join byte-exactly",
          np.array_equal(out[HAND].to_numpy(), hand[HAND].to_numpy()))
    check("embedding values survived the join byte-exactly",
          np.array_equal(out[fu.emb_cols()].to_numpy(), emb.to_numpy()))


def test_fuse_rejects_misalignment():
    """The guard that matters most: row ORDER, not row membership."""
    hand, emb = _tables()
    try:
        fu.fuse(hand, emb.iloc[:-1], HAND, fu.emb_cols())
        check("fuse rejects a row-count mismatch", False)
    except AssertionError:
        check("fuse rejects a row-count mismatch", True)
    shuffled = emb.iloc[::-1]
    try:
        fu.fuse(hand, shuffled, HAND, fu.emb_cols())
        check("fuse rejects a same-set, reordered index", False)
    except AssertionError:
        check("fuse rejects a same-set, reordered index", True)
    dup = emb.rename(columns={"emb_000": "h00"})
    try:
        fu.fuse(hand, dup, HAND, [c for c in dup.columns])
        check("fuse rejects a column-name collision", False)
    except AssertionError:
        check("fuse rejects a column-name collision", True)
    nan_emb = emb.copy()
    nan_emb.iloc[3, 5] = np.nan
    try:
        fu.fuse(hand, nan_emb, HAND, fu.emb_cols())
        check("fuse rejects a NaN in the matrix", False)
    except AssertionError:
        check("fuse rejects a NaN in the matrix", True)
    try:
        fu.align_two(hand, emb, ["nope"], fu.emb_cols())
        check("align_two rejects a missing column name", False)
    except AssertionError:
        check("align_two rejects a missing column name", True)


def test_permute_destroys_only_the_correspondence():
    hand, emb = _tables()
    perm, vec = fu.permute_rows(emb, PERM_SEED)
    check("permutation is a bijection over 24 rows",
          sorted(vec.tolist()) == list(range(24)))
    check("permutation is not the identity", not np.array_equal(vec, np.arange(24)))
    check("every row still exists exactly once",
          np.array_equal(np.sort(perm.to_numpy(), axis=0), np.sort(emb.to_numpy(), axis=0)))
    m0, s0 = emb.to_numpy().mean(0), emb.to_numpy().std(0)
    m1, s1 = perm.to_numpy().mean(0), perm.to_numpy().std(0)
    check("per-column means unchanged to 1e-12", np.abs(m1 - m0).max() < 1e-12,
          f"{np.abs(m1 - m0).max():.2e}")
    check("per-column stds unchanged to 1e-12", np.abs(s1 - s0).max() < 1e-12,
          f"{np.abs(s1 - s0).max():.2e}")
    check("column count still 384", perm.shape[1] == 384)
    check("index is preserved, so the table still joins by label",
          perm.index.equals(emb.index), str(list(perm.index)[:3]))
    check("data differs from the original, so the pairing really is broken",
          not np.array_equal(perm.to_numpy(), emb.to_numpy()))
    check("soil S00 no longer carries its own embedding row",
          not np.allclose(perm.iloc[0].to_numpy(), emb.iloc[0].to_numpy()))
    check("permutation is label-blind: it takes only a table and a seed",
          list(inspect.signature(fu.permute_rows).parameters) == ["df", "seed"],
          str(list(inspect.signature(fu.permute_rows).parameters)))
    # Determinism WITHIN a run is asserted; the exact vector is printed so the record can
    # reconstruct the arm even if a future numpy changes Generator behaviour.
    perm2, vec2 = fu.permute_rows(emb, PERM_SEED)
    check("same seed gives the same permutation in this environment",
          np.array_equal(vec, vec2), str(vec[:6].tolist()))
    _, vec3 = fu.permute_rows(emb, PERM_SEED + 1)
    check("a different seed gives a different permutation",
          not np.array_equal(vec, vec3))
    fused = fu.fuse(hand, perm, HAND, fu.emb_cols())
    check("the permuted table fuses at the same 396 width", fused.shape[1] == 396)
    real = fu.fuse(hand, emb, HAND, fu.emb_cols())
    check("the control arm's embedding block differs from the real arm's",
          not np.allclose(fused[fu.emb_cols()].to_numpy(), real[fu.emb_cols()].to_numpy()))
    check("the control arm's hand block is identical to the real arm's",
          np.allclose(fused[HAND].to_numpy(), real[HAND].to_numpy()))
    try:
        fu.permute_rows(emb.iloc[:1], PERM_SEED)
        check("permute_rows refuses a single-row table", False)
    except AssertionError:
        check("permute_rows refuses a single-row table", True)


def test_zero_variance_detection():
    """A collapsed embedding dimension must be an error at load time. E1 measured feature
    collapse in the random-weights arm; silently shipping 20 dead columns would change the
    meaning of every downstream number."""
    _, emb = _tables()
    check("no false positives on healthy data", fu.zero_variance_columns(emb,
                                                                         fu.emb_cols()) == [])
    dead = emb.copy()
    dead["emb_007"] = 1.0
    dead["emb_383"] = -3.5
    got = fu.zero_variance_columns(dead, fu.emb_cols())
    check("two constant columns are both found", sorted(got) == ["emb_007", "emb_383"],
          str(got))


def test_describe_blocks():
    hand, emb = _tables()
    d = fu.describe_blocks(hand, emb, HAND, fu.emb_cols())
    check("describe reports the pinned shape",
          (d["n_soils"], d["n_hand"], d["n_emb"], d["n_fused"]) == (24, 12, 384, 396),
          str(d))
    check("describe finds no dead column", d["emb_zero_var_cols"] == 0)
    check("describe returns both block scales",
          d["hand_block_std_mean"] > 0 and d["emb_block_std_mean"] > 0)


def main():
    tests = [test_no_torch_in_the_module, test_column_convention_is_e2s,
             test_fuse_width_and_order, test_fuse_rejects_misalignment,
             test_permute_destroys_only_the_correspondence, test_zero_variance_detection,
             test_describe_blocks]
    print("CHECK FUSION")
    print("-" * 78)
    for t in tests:
        print(f"\n{t.__name__}")
        try:
            t()
        except Exception as exc:                                  # noqa: BLE001
            check(f"{t.__name__} raised {type(exc).__name__}", False, str(exc)[:120])
    print("-" * 78)
    if FAILS:
        print(f"CHECK FUSION: FAILED ({len(FAILS)} problems)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("CHECK FUSION: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
