"""Tests backbones._resolve_kwargs against a stub timm, because the real one only exists on
Kaggle and this is the code that decides whether the arms are comparable at all.

The failure it defends against is specific and already happened once: timm's
`vit_small_patch14_dinov2.lvd142m` defaults to a 518 px input and its patch embedding
asserts an exact size, so a 224 px forward raised `AssertionError: Input height (224)
doesn't match model (518)`. The tempting "fix" is to feed 518 - which silently changes the
physical magnification of a patch, the variable deliberately reserved for Model 2 E2, and
would make the arms uncomparable to Model 1. So the resolution must (a) find an argument
set that accepts 224, and (b) if none exists, REFUSE rather than drift.

Run:  python scratch/check_m2e1_backbone_kwargs.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                       / "Model 2" / "Model 2 Experiment 1"))
import backbones as bb                                        # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


class FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)

    def __getitem__(self, k):
        return self


class FakeParam:
    device = "cpu"


class FakeModel:
    def __init__(self, accept):
        self.accept = accept

    def eval(self):
        return self

    def parameters(self):
        return iter([FakeParam()])          # torch returns a generator, not a list

    def __call__(self, x):
        h = x.shape[-2] if hasattr(x, "shape") else 224
        if h != self.accept:
            raise AssertionError(f"Input height ({h}) doesn't match model ({self.accept}).")
        return FakeTensor((x.shape[0], bb.EMBED_DIM))


class FakeTorch:
    """Enough of torch for _probe: zeros(...), no_grad(), and a device attribute."""

    class _NG:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def no_grad(self):
        return self._NG()

    def zeros(self, *shape, device=None):
        return FakeTensor(shape)

    def manual_seed(self, s):
        pass


class FakeTimm:
    """Models a specific timm behaviour: `accept` is the input size the built network
    tolerates, given the construction kwargs it was handed."""

    __version__ = "stub"

    def __init__(self, accept_for):
        self.accept_for = accept_for          # kwargs tuple -> accepted input size
        self.built = []

    def create_model(self, name, pretrained=False, num_classes=0, **kw):
        self.built.append(tuple(sorted(kw.items())))
        return FakeModel(self.accept_for(kw))


def resolve(timm_stub):
    bb._KWARGS_RESOLVED.clear()
    return bb._resolve_kwargs(FakeTorch(), timm_stub)


print("1. the timm that produced the Kaggle failure")
# Pre-fix behaviour: without img_size/dynamic_img_size the network is 518-only.
KAGGLE_LIKE = lambda kw: (                                        # noqa: E731
    224 if (kw.get("dynamic_img_size") or kw.get("img_size") == 224) else 518)
got = resolve(FakeTimm(KAGGLE_LIKE))
check("resolution succeeds instead of raising", isinstance(got, dict), str(got))
check("it keeps the frozen 224 px definition",
      got.get("img_size") == 224 or got.get("dynamic_img_size") is True, str(got))
check("it does NOT fall back to the bare default (the 518 route)",
      got != {}, f"chosen={got}")

print("\n2. the winning argument set is shared by both arms")
t = FakeTimm(KAGGLE_LIKE)
first = resolve(t)                              # clears the cache, then resolves
second = bb._resolve_kwargs(FakeTorch(), t)     # as arm D would: cache already warm
check("second call returns the same kwargs", first == second, str(second))
check("the probe built the model once, not once per arm",
      len(t.built) == 1, f"built {len(t.built)}: {t.built}")

print("\n3. an older timm that has no dynamic_img_size argument at all")


def no_dyn(kw):
    if "dynamic_img_size" in kw:
        raise TypeError("create_model() got an unexpected keyword argument "
                        "'dynamic_img_size'")
    return 224 if kw.get("img_size") == 224 else 518


got = resolve(FakeTimm(no_dyn))
check("falls through to img_size-only rather than failing the run",
      got == {"img_size": 224}, str(got))

print("\n4. no argument set accepts 224 -> it must REFUSE, not drift")
STUBBORN = lambda kw: 518                                         # noqa: E731
try:
    resolve(FakeTimm(STUBBORN))
    check("raises instead of silently using 518", False, "it returned a kwargs set")
except RuntimeError as exc:
    msg = str(exc)
    check("raises RuntimeError, not AssertionError", True, msg.splitlines()[0][:70])
    check("the message names the reserved E2 variable", "E2" in msg, "")
    check("the message forbids the silent 518 switch", "518" in msg and "silently" in msg)
    check("the message lists every attempt that was tried",
          msg.count("->") >= len(bb._BUILD_CANDIDATES), f"{msg.count('->')} attempts logged")

print("\n5. a readout that is not the 384-dim pooled token is caught")
bb._KWARGS_RESOLVED.clear()


class WrongReadout(FakeModel):
    def __call__(self, x):
        class T:
            shape, ndim = (1, 1, 1, 384), 4
        return T()


class TimmBadReadout(FakeTimm):
    def create_model(self, *a, **kw):
        return WrongReadout(224)


try:
    resolve(TimmBadReadout(KAGGLE_LIKE))
    check("a 4-D readout is refused", False, "it passed")
except RuntimeError as exc:
    check("a 4-D readout is refused", "readout shape" in str(exc), str(exc)[:60])

print("\n6. the candidate list itself")
check("every candidate is tried with the 224 definition present or dynamic sizing on",
      all("img_size" in c or c.get("dynamic_img_size") for c in bb._BUILD_CANDIDATES[:-1]),
      str(bb._BUILD_CANDIDATES))
check("the bare default is last, so it can never mask a working 224 config",
      bb._BUILD_CANDIDATES[-1] == {}, str(bb._BUILD_CANDIDATES[-1]))
check("INPUT_SIZE is still 224", bb.INPUT_SIZE == 224, str(bb.INPUT_SIZE))
check("patch scale unchanged", abs(bb.PATCH_MM - 14 / 4.552516) < 1e-9,
      f"{bb.PATCH_MM:.4f} mm")

print("\n" + "=" * 70)
if FAILS:
    print(f"BACKBONE KWARGS CHECK: {len(FAILS)} FAILURE(S)")
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("BACKBONE KWARGS CHECK: PASSED")
print("""
Scope, stated plainly: this tests the DECISION logic with a stub, not timm itself. It proves
the code keeps 224 px when it can and refuses rather than drifting to 518 when it cannot.
Whether Kaggle's timm actually accepts dynamic_img_size is answered by cell E1 on the next
run, which prints the chosen argument set before any tile is processed.""")
