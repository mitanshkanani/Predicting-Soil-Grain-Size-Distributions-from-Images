"""fusion.py - the one new variable in Model 2 Experiment 3, isolated so it can be tested
before any experiment runs.

E3 changes exactly one thing: the composition of the feature matrix. Twelve hand-built
columns become twelve hand-built columns plus a 384-dimensional DINOv2 block. Everything
downstream of that matrix - the scaler, the rank-3 curve basis, the closed-form ridge, the
monotone projection, the CV families, the alpha grid, the metric - is inherited unchanged
from Model 2 E1 and E2.

This module exists separately from the notebook for the same reason crop_geometry.py did:
the arithmetic that defines the experiment must be checkable without torch, without the
data, and without running the experiment. It is also where the label-blind permutation
control lives. That control is what makes a fusion win attributable, so its correctness is
not a detail.

No torch, no timm, no file IO. numpy and pandas only.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

EMBED_DIM: int = 384


def emb_cols(dim: int = EMBED_DIM) -> List[str]:
    """Embedding column names, using Model 2 E2's existing convention (emb_000 ... emb_383).

    Reusing E2's names rather than inventing E3's is deliberate: a reviewer can diff E3's
    fused matrix against E2's D matrix and see that nothing but the concatenation happened.
    """
    return [f"emb_{i:03d}" for i in range(int(dim))]


def _same_rows(hand: pd.DataFrame, emb: pd.DataFrame, what: str) -> None:
    """Assert the two soil tables are the same soils in the same order.

    Row order is asserted, not just set equality. fuse() and permute_rows() are both
    positional, so a silently reordered index would pair every soil with the wrong
    embedding while still passing a set comparison.
    """
    if hand.shape[0] != emb.shape[0]:
        raise AssertionError(f"{what}: row count {hand.shape[0]} vs {emb.shape[0]}")
    if not hand.index.equals(emb.index):
        raise AssertionError(f"{what}: index differs in value or order - "
                             "reindex one table before fusing")


def align_two(hand: pd.DataFrame, emb: pd.DataFrame,
              hand_cols: Sequence[str], emb_cols_: Sequence[str]) -> Tuple[pd.DataFrame,
                                                                           pd.DataFrame]:
    """Return the two tables restricted to the named columns, rows aligned and asserted."""
    missing_h = [c for c in hand_cols if c not in hand.columns]
    missing_e = [c for c in emb_cols_ if c not in emb.columns]
    if missing_h or missing_e:
        raise AssertionError(f"align_two: missing columns hand={missing_h} emb={missing_e}")
    a = hand.loc[:, list(hand_cols)]
    b = emb.loc[:, list(emb_cols_)]
    _same_rows(a, b, "align_two")
    return a, b


def fuse(hand: pd.DataFrame, emb: pd.DataFrame,
         hand_cols: Sequence[str], emb_cols_: Sequence[str]) -> pd.DataFrame:
    """Concatenate the two blocks into one soil-indexed feature matrix.

    Guards three things the experiment's whole interpretation rests on: identical rows in
    identical order, no column-name collision (a collision would silently DROP a block in
    any later reindex or to_dict), and an output width that is exactly the sum of the
    inputs. 396 columns out, or this raises.
    """
    a, b = align_two(hand, emb, hand_cols, emb_cols_)
    clash = sorted(set(a.columns) & set(b.columns))
    if clash:
        raise AssertionError(f"fuse: column-name collision {clash[:5]}")
    out = pd.concat([a, b], axis=1)
    expected = len(list(hand_cols)) + len(list(emb_cols_))
    if out.shape[1] != expected:
        raise AssertionError(f"fuse: width {out.shape[1]} != hand {len(hand_cols)} + "
                             f"emb {len(emb_cols_)} = {expected}")
    if out.isna().to_numpy().any():
        raise AssertionError("fuse: NaN in the fused matrix")
    return out


def permute_rows(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """Destroy the soil-to-embedding correspondence and nothing else.

    This is E3's central control. Permuting the DATA rows while keeping the INDEX fixed
    leaves every column's mean, standard deviation and covariance structure exactly as they
    were, so the block still looks like 384 standardised DINOv2 dimensions to the ridge.
    What it no longer carries is any relationship to which soil produced it. If fusion still
    beats the control, the gain came from information; if it does not, the gain came from
    capacity.

    The index is deliberately left in place. ``df.iloc[perm]`` would be wrong and silently
    so: reordering rows preserves each label-to-row pairing, so the moment this table is
    joined to the hand block by label everything realigns and the "control" becomes
    identical to the real arm. Permuting must break the pairing, not the order.

    The permutation is label-blind: it never sees the targets, and it is generated from a
    pinned seed, so the arm is reproducible from the record. Returns the permutation vector
    as well as the table, because a control nobody can reconstruct is not a control.
    """
    n = df.shape[0]
    if n < 2:
        raise AssertionError(f"permute_rows: cannot permute {n} rows meaningfully")
    idx = np.random.default_rng(int(seed)).permutation(n)
    if sorted(idx.tolist()) != list(range(n)):
        raise AssertionError("permute_rows: permutation is not a bijection")
    if np.array_equal(idx, np.arange(n)):
        raise AssertionError("permute_rows: drew the identity permutation; the control "
                             "would be the real arm")
    arr = df.to_numpy(float)[idx]
    out = pd.DataFrame(arr, index=df.index, columns=df.columns)
    if out.index.equals(df.index) and np.array_equal(out.to_numpy(), df.to_numpy()):
        raise AssertionError("permute_rows: data unchanged - the control is not a control")
    return out, idx


def describe_blocks(hand: pd.DataFrame, emb: pd.DataFrame,
                    hand_cols: Sequence[str], emb_cols_: Sequence[str]) -> dict:
    """One-row summary of what the fused matrix is made of, for the printed record."""
    a, b = align_two(hand, emb, hand_cols, emb_cols_)
    return {
        "n_soils": int(a.shape[0]),
        "n_hand": int(a.shape[1]),
        "n_emb": int(b.shape[1]),
        "n_fused": int(a.shape[1] + b.shape[1]),
        "hand_block_std_mean": float(np.mean(a.to_numpy(float).std(axis=0))),
        "emb_block_std_mean": float(np.mean(b.to_numpy(float).std(axis=0))),
        "emb_zero_var_cols": int((b.to_numpy(float).std(axis=0) <= 0.0).sum()),
    }


def zero_variance_columns(emb: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    """Columns with no spread at all. A collapsed DINOv2 dimension shows up here.

    Checked deliberately: E1 measured feature collapse in the random-weights arm, so a dead
    column must be an error at load time rather than a strange score several cells later.
    """
    arr = emb.loc[:, list(cols)].to_numpy(float)
    std = arr.std(axis=0)
    return [c for c, s in zip(list(cols), std) if not s > 0.0]
