"""The only place in this project where ``torch`` is imported.

Model 2 Experiment 1 replaces the per-tile feature vector and nothing else. This module
owns that one variable. Everything downstream - aggregation, the rank-3 curve basis, the
ridge head, the CV families, the metric - lives in the notebook and is identical to Model 1
E3's.

WHY THIS FILE EXISTS SEPARATELY
-------------------------------
The development machine has no ``torch``, no ``torchvision`` and no ``timm``, so the
backbone arms cannot be executed locally at all. Rather than quietly simplifying the
experiment to fit the machine, the unrunnable code is isolated behind a three-function
interface and a pure-numpy ``mock`` backend exercises everything around it.

Consequences of that arrangement, both of them deliberate:

* ``mock`` is a plumbing test. It is never a result. It is built to be uninformative on
  purpose (a fixed random linear map of standardised grayscale pixels), so a mock arm that
  scored well in-domain would itself indicate a leak in the surrounding pipeline.
* the torch branches must fail loudly with a message naming Kaggle. A silent fallback to
  ``mock`` would let an untested path masquerade as a measurement, which is the single
  worst failure mode available here.

ARMS
----
``M1``  hand-built features, produced by the notebook, not here.
``R``   ``vit_random``  - DINOv2 ViT-S/14 architecture, random untrained weights.
``D``   ``dinov2``      - the same architecture, pretrained weights, frozen.

R exists because a 384-dimensional vector beating a 12-dimensional one proves nothing about
pretraining without a same-dimensionality control. See
``Model 2/Model 2 Experiment 1/instructions.txt``.

DETERMINISM
-----------
Every backend is a pure function of ``(rgb_batch, backend, seed)``. ``vit_random`` is
seeded immediately before weight initialisation, so a given seed always reproduces the same
random network.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np

BACKENDS: Tuple[str, ...] = ("mock", "vit_random", "dinov2")
EMBED_DIM: int = 384
MODEL_NAME: str = "vit_small_patch14_dinov2.lvd142m"
INPUT_SIZE: int = 224          # ViT-S/14 wants a multiple of its patch size
PATCH_MM: float = 14.0 / 4.552516   # one patch, in millimetres, at the canonical scale
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CACHE: Dict[str, object] = {}

_KAGGLE_HINT = (
    "torch/timm are not installed here, and that is expected: the development machine has "
    "no torch.\n"
    "The backbone arms run ONLY on Kaggle. See kaggle_setup.md for the attached dataset, "
    "the weights source and the accelerator setting.\n"
    "To exercise the surrounding pipeline locally instead, use BACKEND=mock. The mock arm "
    "is a plumbing test and must never be reported as a result."
)


def dim(backend: str) -> int:
    """Embedding width for every backend. Identical by design: the arms must differ in the
    information the vector carries, not in how many numbers the head receives."""
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    return EMBED_DIM


def requires_torch(backend: str) -> bool:
    """True for arms that cannot run on this machine. Used by the notebook's environment
    probe and by check_backbones.py's loud-failure assertion."""
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    return backend != "mock"


def torch_available() -> bool:
    """False on the development machine. The notebook's environment probe reports this so a
    local run declares which arms it could not execute, rather than looking like a complete
    experiment."""
    try:
        import torch  # noqa: F401
        import timm   # noqa: F401
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ mock backend
def _mock_matrix(n_in: int, seed: int) -> np.ndarray:
    h = hashlib.sha256(f"mock|{n_in}|{seed}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    return rng.standard_normal((n_in, EMBED_DIM)) / np.sqrt(n_in)


def _embed_mock(rgb: np.ndarray, seed: int) -> np.ndarray:
    """A fixed random linear map of per-image-standardised grayscale pixels.

    Deliberately uninformative: it discards colour, discards spatial layout beyond a 4x
    box downsample, and contains no learned features whatsoever. Its only job is to prove
    that tile loading, resizing, batching, caching, aggregation, fold-local fitting,
    nested CV and the submission gates all work, so that the first Kaggle run is not also
    the first time the plumbing has ever executed.

    Per-image standardisation is what makes the output invariant to a uniform brightness
    shift, which check_backbones.py asserts. The DINOv2 arms get that property from
    ImageNet normalisation plus the frozen network and must be checked for it separately.
    """
    a = np.asarray(rgb, dtype=np.float64)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {a.shape}")
    g = a.mean(axis=3)                                    # (N,H,W) grayscale
    n = g.shape[1]
    if n % 4:
        raise ValueError(f"tile edge {n} is not divisible by the 4x box factor")
    d = g.reshape(g.shape[0], n // 4, 4, n // 4, 4).mean(axis=(2, 4))   # (N,n/4,n/4)
    flat = d.reshape(d.shape[0], -1)
    mu = flat.mean(axis=1, keepdims=True)
    sd = flat.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-9, 1.0, sd)                    # constant tiles -> zeros, not NaN
    std_flat = (flat - mu) / sd
    out = std_flat @ _mock_matrix(std_flat.shape[1], seed)
    return _l2(out)


def _l2(x: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(nrm < 1e-12, 1.0, nrm)


# ------------------------------------------------------------------ torch backends
# timm's `vit_small_patch14_dinov2.lvd142m` is the RoPE variant whose default_cfg input
# size is 518, and its patch embedding asserts an exact height/width unless dynamic input
# sizing is enabled. The frozen feature definition in instructions.txt is 224 (patch =
# 3.075 mm at the canonical scale), and resizing tiles to 518 instead would silently
# change the physical magnification -- the variable deliberately reserved for E2. So the
# input size stays 224 and the construction is probed for the argument set that accepts it.
#
# The probe runs with pretrained=False, so no weights are downloaded per attempt, and the
# winning argument set is cached module-level so BOTH arms are built identically. Arm R is
# the architecture control; if R and D were constructed by different routes, the ablation
# would be comparing two different networks.
_BUILD_CANDIDATES = (
    {"img_size": INPUT_SIZE, "dynamic_img_size": True},
    {"img_size": INPUT_SIZE},
    {"dynamic_img_size": True},
    {},
)
_KWARGS_RESOLVED: Dict[str, dict] = {}


def _probe(model, torch) -> None:
    """One zero batch through the network. Raises if it rejects INPUT_SIZE or if the
    readout is not the 384-dim pooled token."""
    dev = next(model.parameters()).device
    with torch.no_grad():
        out = model(torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE, device=dev))
    if out.ndim != 2 or out.shape[1] != EMBED_DIM:
        raise RuntimeError(f"readout shape {tuple(out.shape)}, expected (1,{EMBED_DIM})")


def _resolve_kwargs(torch, timm) -> dict:
    """Pick the construction arguments that accept a 224 px input. Deterministic: the
    candidate list is tried in order and the probe is a zero-input forward pass."""
    if "kwargs" in _KWARGS_RESOLVED:
        return _KWARGS_RESOLVED["kwargs"]
    tried = []
    for cand in _BUILD_CANDIDATES:
        m = None
        try:
            m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, **cand)
            m.eval()
            _probe(m, torch)
        except Exception as exc:
            tried.append(f"    {cand} -> {type(exc).__name__}: {exc}")
            m = None
            continue
        print(f"  backbone construction accepted: {cand}")
        if tried:
            print("  rejected by this timm/model:")
            for t in tried:
                print(t)
        _KWARGS_RESOLVED["kwargs"] = dict(cand)
        return dict(cand)
    raise RuntimeError(
        f"{MODEL_NAME} in timm {timm.__version__} refuses a {INPUT_SIZE}px input under every "
        "construction tried:\n" + "\n".join(tried) +
        f"\n\nThe frozen feature definition requires {INPUT_SIZE}px (one patch = "
        f"{PATCH_MM:.3f} mm). Do NOT silently switch to the model's native 518px: that "
        "changes the physical magnification, which is the variable reserved for Model 2 E2, "
        "and it would make the arms uncomparable to Model 1. Either enable dynamic input "
        "sizing in the installed timm or report the version so the definition can be "
        "amended openly.")


def _get_model(backend: str, seed: int, weights_path: Optional[str]):
    """Build and cache the backbone. ``vit_random`` is created with pretrained=False
    immediately after seeding torch, so the random draw is reproducible."""
    import torch
    import timm

    key = f"{backend}|{seed}|{weights_path or ''}"
    if key in _CACHE:
        return _CACHE[key]
    if backend not in ("vit_random", "dinov2"):
        raise KeyError(f"{backend!r} does not use torch")
    build = _resolve_kwargs(torch, timm)
    if backend == "vit_random":
        torch.manual_seed(int(seed))
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0, **build)
    else:
        kwargs = dict(build)
        if weights_path:
            # offline: weights supplied as an attached Kaggle dataset
            kwargs["pretrained_cfg_overlay"] = dict(file=weights_path)
        model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)          # frozen is not a convention, it is enforced
    _probe(model, torch)                 # the weights build is probed too, not just the arch
    _CACHE[key] = model
    return model


def _embed_torch(rgb: np.ndarray, backend: str, seed: int, batch_size: int,
                 weights_path: Optional[str], device: Optional[str],
                 progress_every: int) -> np.ndarray:
    try:
        import torch
        import timm  # noqa: F401
    except Exception as exc:
        raise ImportError(
            f"backend {backend!r} needs torch and timm, which are unavailable here "
            f"({type(exc).__name__}).\n\n{_KAGGLE_HINT}"
        ) from exc

    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3), got {a.shape}")

    model = _get_model(backend, seed, weights_path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    outs = []
    n = a.shape[0]
    for i in range(0, n, batch_size):
        blk = torch.from_numpy(a[i:i + batch_size] / 255.0).permute(0, 3, 1, 2).to(dev)
        blk = torch.nn.functional.interpolate(
            blk, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
        blk = (blk - mean) / std
        with torch.no_grad():
            emb = model(blk)                       # num_classes=0 -> pooled CLS, (B,384)
        emb = emb.float().cpu().numpy()
        if emb.ndim != 2 or emb.shape[1] != EMBED_DIM:
            raise RuntimeError(
                f"{backend} returned shape {emb.shape}, expected (B,{EMBED_DIM}). The "
                "readout is the pooled CLS token; if timm changed the default global "
                "pooling for this model the arm definition has silently moved.")
        outs.append(emb)
        if progress_every and (i // batch_size + 1) % progress_every == 0:
            print(f"    {backend}: {min(i + batch_size, n)}/{n} tiles", flush=True)
    return _l2(np.concatenate(outs, axis=0).astype(np.float64))


# ------------------------------------------------------------------ public entry point
def embed_tiles(rgb_batch: np.ndarray, backend: str, seed: int = 20260930,
                batch_size: int = 64, weights_path: Optional[str] = None,
                device: Optional[str] = None, progress_every: int = 20) -> np.ndarray:
    """(N,256,256,3) float in 0-255 -> (N,384) float64, L2-normalised rows.

    The only function the notebook calls. Deterministic for a given
    ``(rgb_batch, backend, seed)``. Raises ``ImportError`` naming Kaggle rather than
    falling back to ``mock`` when a backbone is requested on a machine without torch.
    """
    if backend not in BACKENDS:
        raise KeyError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "mock":
        out = _embed_mock(rgb_batch, seed)
    else:
        out = _embed_torch(rgb_batch, backend, seed, batch_size, weights_path, device,
                           progress_every)
    if out.shape != (np.asarray(rgb_batch).shape[0], EMBED_DIM):
        raise RuntimeError(f"{backend}: expected ({len(rgb_batch)},{EMBED_DIM}), got {out.shape}")
    if not np.isfinite(out).all():
        raise RuntimeError(f"{backend}: non-finite values in the embedding matrix")
    return out
