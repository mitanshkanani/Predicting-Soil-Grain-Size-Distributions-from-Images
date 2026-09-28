"""Contract test for backbones.py - run from the experiment folder.

    python check_backbones.py              # local: mock verified, torch arms must refuse
    python check_backbones.py vit_random   # Kaggle: the REAL arm is verified too

This is not a test of the science. It is a test of the interface, and it exists because two
of the three arms cannot run on this machine. The design rule:

* the notebook will call ``embed_tiles`` on 1,976 tiles, where a shape or dtype mistake
  costs an entire session;
* if a backbone arm were ever to fall back to ``mock`` silently, the resulting table would
  look complete and be worthless.

So the test has two faces and picks its face from what torch can do, not from a hardcoded
expectation. When torch is absent the backbone arms must RAISE. When torch is present the
same structural checks run against the real network, which is why the notebook re-runs this
file before extracting anything.

No pytest here (not installed). Plain asserts, printed labels, non-zero exit on failure.
"""
from __future__ import annotations

import sys

import numpy as np

import backbones as bb

FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = ""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def _raises(exc, fn, *args, **kwargs) -> bool:
    """True when fn(...) raises exactly `exc` (or a subclass). Used for the negative
    tests, which are the point of this file: the arms that cannot run here must refuse."""
    try:
        fn(*args, **kwargs)
    except exc:
        return True
    except Exception:
        return False
    return False


def tiles(n: int = 7, size: int = 256, seed: int = 0) -> np.ndarray:
    """Synthetic RGB tiles with real spatial structure (grains), not pure noise."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, size, size, 3), dtype=np.uint8)
    coarse = (size // 8 + 1, size // 3 + 1)
    for i in range(n):
        base = rng.integers(60, 200, size=coarse)
        big = np.kron(base, np.ones((8, 3)))[:size, :size]
        out[i] = np.stack([big, big * 0.9, big * 0.8], axis=2).astype(np.uint8)
    return out


def structural(backend: str, rgb: np.ndarray) -> None:
    """The checks every backend must pass, run identically on mock and on a real backbone."""
    emb = bb.embed_tiles(rgb, backend)
    check(f"{backend}: output shape is (N, {bb.EMBED_DIM})",
          emb.shape == (len(rgb), bb.EMBED_DIM), str(emb.shape))
    check(f"{backend}: float dtype", np.issubdtype(emb.dtype, np.floating), str(emb.dtype))
    check(f"{backend}: all values finite", bool(np.isfinite(emb).all()))
    nrm = np.linalg.norm(emb, axis=1)
    check(f"{backend}: rows are L2-unit-norm", bool(np.allclose(nrm, 1.0, atol=1e-6)),
          f"max |1-norm| = {float(np.abs(nrm - 1).max()):.2e}")
    check(f"{backend}: not all-zero", float(np.abs(emb).max()) > 1e-9)
    check(f"{backend}: same input twice -> byte-identical",
          np.array_equal(emb, bb.embed_tiles(rgb, backend)))
    other = bb.embed_tiles(tiles(len(rgb), seed=7), backend)
    d_same = float(np.abs(emb - bb.embed_tiles(rgb, backend)).mean())
    d_diff = float(np.abs(emb - other).mean())
    check(f"{backend}: different tiles -> different embeddings",
          d_diff > 100 * max(d_same, 1e-15), f"d_same {d_same:.2e} vs d_diff {d_diff:.4f}")
    # A frozen network is not exactly brightness-invariant, so this is a soft check for the
    # torch arms: it reports, it does not fail.
    shifted = bb.embed_tiles(np.clip(rgb.astype(np.int32) + 40, 0, 255).astype(np.uint8),
                             backend)
    rel = float(np.abs(shifted - emb).max())
    if bb.requires_torch(backend):
        print(f"  [INFO] {backend}: max |delta| under a +40 brightness shift = {rel:.4f} "
              "(reported only; a frozen network is not a photometric normaliser)")
    else:
        check(f"{backend}: per-image standardisation kills a uniform brightness shift",
              rel < 1e-9, f"max |delta| = {rel:.2e}")


Torch_present = bb.torch_available()
print(f"\ntorch/timm present: {Torch_present}")

REQUESTED = [a for a in sys.argv[1:] if a != "mock"]
if not Torch_present:
    print("\n1. module surface")
    check("BACKENDS is exactly the three arms",
          bb.BACKENDS == ("mock", "vit_random", "dinov2"), str(bb.BACKENDS))
    check("EMBED_DIM == 384 for all arms", all(bb.dim(b) == 384 for b in bb.BACKENDS))
    check("dim() rejects unknown backends", _raises(KeyError, bb.dim, "clip"))
    check("requires_torch('mock') is False", bb.requires_torch("mock") is False)
    check("requires_torch('vit_random') is True", bb.requires_torch("vit_random") is True)
    check("requires_torch('dinov2') is True", bb.requires_torch("dinov2") is True)
    check("torch_available() is False, matching the import failure", Torch_present is False)

print("\n2. mock backend - the path that must run everywhere")
structural("mock", tiles())

print("\n3. determinism of the mock random map")
e0 = bb.embed_tiles(tiles(), "mock")
check("re-seeded mock differs", not np.allclose(e0, bb.embed_tiles(tiles(), "mock", seed=11)))

print("\n4. degenerate inputs do not produce NaN")
flat = np.full((2, 256, 256, 3), 127, dtype=np.uint8)
check("constant-colour tiles give finite output",
      bool(np.isfinite(bb.embed_tiles(flat, "mock")).all()),
      "a zero-variance tile would divide by zero without the guard")

print("\n5. input validation")
check("2-D input is rejected", _raises(ValueError, bb.embed_tiles, np.zeros((4, 4)), "mock"))
check("unknown backend is rejected", _raises(KeyError, bb.embed_tiles, tiles(), "clip"))

print("\n6. the mock arm is UNINFORMATIVE by construction")
# If this ever stops holding, the mock has accidentally learned something and every
# downstream plumbing conclusion drawn from it is suspect.
from sklearn.linear_model import Ridge                     # noqa: E402

X = bb.embed_tiles(tiles(24, seed=1), "mock")
y = np.linspace(0.0, 1.0, 24)                       # a target the tiles cannot know
pred = np.array([Ridge(alpha=1.0).fit(np.delete(X, i, 0), np.delete(y, i))
                 .predict(X[i][None, :])[0] for i in range(24)])
r2 = 1 - float(((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
check("mock embeddings do not predict an unrelated held-out target", r2 < 0.1,
      f"LOO R^2 = {r2:.3f}")

print("\n7. backbone arms")
if not Torch_present:
    for b in ("vit_random", "dinov2"):
        try:
            bb.embed_tiles(tiles(), b)
            check(f"{b} raises on this machine", False,
                  "it returned instead - either torch arrived mid-run or the branch fell "
                  "back to mock, and a silent fallback is the worst failure here")
        except ImportError as exc:
            msg = str(exc)
            check(f"{b} raises ImportError naming Kaggle", "Kaggle" in msg,
                  repr(msg.splitlines()[0]))
            check(f"{b} ImportError mentions the local mock option", "mock" in msg)
        except Exception as exc:
            check(f"{b} raises ImportError", False, f"raised {type(exc).__name__}: {exc}")
else:
    if not REQUESTED:
        FAILS.append("torch is present but no backend was requested")
        print("  [FAIL] torch is installed, so name the arm to verify: "
              "python check_backbones.py vit_random")
    for b in REQUESTED:
        if b not in bb.BACKENDS or b == "mock":
            FAILS.append(f"bad backend argument {b}")
            print(f"  [FAIL] {b!r} is not a backbone arm")
            continue
        print(f"  --- {b} (real) ---")
        structural(b, tiles())
        check(f"{b}: batch_size smaller than N concatenates correctly",
              bb.embed_tiles(tiles(64), b, batch_size=16).shape == (64, bb.EMBED_DIM))
    if "vit_random" in REQUESTED:
        a = bb.embed_tiles(tiles(), "vit_random", seed=1)
        b2 = bb.embed_tiles(tiles(), "vit_random", seed=2)
        check("vit_random: two seeds give two different networks", not np.allclose(a, b2))

print("\n" + "=" * 72)
if FAILS:
    print(f"check_backbones: {len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("check_backbones: ALL CHECKS PASSED")
covered = "mock" + (", " + ", ".join(REQUESTED) if Torch_present and REQUESTED else "")
print(f"verified backends: {covered}. Unverified by this run: "
      f"{', '.join(x for x in bb.BACKENDS if x not in covered and x != 'mock') or 'none'}.")
