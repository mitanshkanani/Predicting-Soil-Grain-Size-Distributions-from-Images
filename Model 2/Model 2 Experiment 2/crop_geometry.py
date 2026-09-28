"""The one new variable in Model 2 Experiment 2: how much physical soil a tile contributes
to the backbone.

Model 2 E1 tested a frozen ViT-S/14 whose patch spans 3.51 mm against soils whose median
grain is often under 0.1 mm, and declared that confound in advance. E2 varies the
magnification and asks whether the refutation survives.

WHY THIS IS A SEPARATE MODULE
-----------------------------
Same reason ``backbones.py`` is. This is the single piece of new logic in the experiment
that has to be right, and the development machine has no torch, so it must be testable
without one. A module can be imported by both the notebook and ``check_crop_geometry.py``;
cell text cannot, and a helper retyped into a cell is a helper that can silently disagree
with the thing that is supposed to be verifying it.

There is no ``torch`` import here and there must never be one.

A CORRECTION, CARRIED HERE SO IT IS NOT SILENTLY INHERITED
----------------------------------------------------------
E1's ``backbones.PATCH_MM`` is ``14 / 4.552516 = 3.0752`` mm. That omits the 256 -> 224
resize: a 256 px tile is a 56.23 mm window squeezed into 224 px, so one 14 px patch
actually covers **3.5145 mm** - 14.3% more than recorded. ``backbones.py`` is left
untouched, because E1's record and its cached embeddings must stay byte-reproducible.
Every E2 number comes from ``patch_mm`` below, and the experiment prints both values so
the discrepancy is visible rather than buried.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# The sweep. 256 is E1's own configuration and doubles as the regression check: the
# notebook must reproduce E1's in-domain table from it before any other number here means
# anything. 32 is the practical floor - below that the window is a handful of real pixels
# stretched to 224, which changes the encoder's input without adding any information.
CROPS: Tuple[int, ...] = (256, 128, 64, 32)

# The backbone's input geometry, mirrored from backbones.py rather than imported: this
# module must be loadable on a machine with no torch, and importing backbones is safe but
# couples the geometry check to the torch surface it is meant to sit beside.
INPUT_SIZE: int = 224
PATCH_PX: int = 14

# The mock backend box-downsamples by 4 before its random map, so every crop must be a
# multiple of 4 or the mock dry run exercises a different code path than the real one.
CROP_MULTIPLE: int = 4


def crop_tile(rgb: np.ndarray, c: int) -> np.ndarray:
    """Centred square crop of side ``c`` from an ``(N,H,W,3)`` batch.

    Returns a view, not a copy, except when ``c == H == W`` where it returns the input
    unchanged. Dtype is preserved: the caller passes 0-255 floats and ``embed_tiles``
    divides by 255 itself, so a cast here would be a second normalisation.
    """
    a = np.asarray(rgb)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"expected an (N,H,W,3) batch, got {a.shape}")
    h, w = a.shape[1], a.shape[2]
    if c <= 0:
        raise ValueError(f"crop {c} is not positive")
    if c > min(h, w):
        raise ValueError(f"crop {c} does not fit a {h}x{w} tile")
    if c % CROP_MULTIPLE:
        raise ValueError(
            f"crop {c} is not a multiple of {CROP_MULTIPLE}; the mock backend box-"
            "downsamples by 4, so a non-multiple would make the local dry run test a "
            "different pipeline from the real arms")
    if c == h == w:
        return a
    o_y, o_x = (h - c) // 2, (w - c) // 2
    return a[:, o_y:o_y + c, o_x:o_x + c, :]


def window_mm(c: int, ppm: float) -> float:
    """Physical side length of a ``c`` canonical-pixel window."""
    return c / ppm


def patch_mm(c: int, ppm: float, input_size: int = INPUT_SIZE, patch: int = PATCH_PX) -> float:
    """Footprint of one backbone patch, in mm, after a ``c -> input_size`` resize.

    (window mm / input px) * patch px. The resize is the whole point of this function:
    omitting it is the error E1 shipped.
    """
    return window_mm(c, ppm) / input_size * patch


def resolvability(d50_mm: np.ndarray, c: int, ppm: float, divisor: float = 3.0) -> int:
    """How many soils have a median grain of at least ``1/divisor`` of a patch.

    A grain smaller than roughly a third of a patch is inside one token's receptive field
    and cannot be resolved as a particle - it enters only as texture. This is the quantity
    that bounds what magnification can achieve, and E2's headline finding is that it barely
    moves across the whole sweep.
    """
    d = np.asarray(d50_mm, dtype=float)
    return int((d >= patch_mm(c, ppm) / divisor).sum())


def soil_group(d50_mm: float) -> str:
    """Size group used for the per-group analysis, which is where E2's answer lives.

    Cut points are the data's own structure, not a smooth grid: the training D50 values are
    bimodal, ten soils under 0.11 mm and twelve above 1.5 mm, with only two in between.
    """
    if d50_mm < 0.11:
        return "fines"
    if d50_mm <= 1.5:
        return "mid"
    return "coarse"


def describe(d50_mm: np.ndarray, ppm: float) -> str:
    """A printed reachability table, computed rather than hardcoded, for the record."""
    d = np.asarray(d50_mm, float)
    lines = [f"{'crop px':>8} {'patch mm':>9} {'window mm':>10} {'resolvable':>11} "
             f"{'area used':>10}"]
    for c in CROPS:
        lines.append(f"{c:8d} {patch_mm(c, ppm):9.4f} {window_mm(c, ppm):10.2f} "
                     f"{resolvability(d, c, ppm):11d} {100.0 * (c / 256.0) ** 2:9.1f}%")
    return "\n".join(lines)
