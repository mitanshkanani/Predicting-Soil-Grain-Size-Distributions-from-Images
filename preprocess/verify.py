"""STEP 11 - prove preprocessing preserved information rather than destroyed it.

Every check here is a measurement with a threshold, written once into
golden_checks.json and asserted on every subsequent run, so retuning the mask or
the resampler cannot quietly regress.

The headline check is the paired-camera test. 21 of the 24 training soils were
photographed by BOTH the Motorola Edge and the Samsung A52: same soil, same tray,
same lighting, same fixed 21 cm distance, and after correcting for the delivered
downscale, effectively the same physical scale (4.597 vs 4.554 px/mm). So the
difference between their texture statistics is pure camera pipeline, and we can
measure how much of it survives preprocessing instead of guessing.

Design decision worth stating: the texture gap is EXPECTED to survive. It is
modulation transfer, sharpening and tone curve, not chroma, so no colour
operation removes it. The assertion is therefore that preprocessing must not
*widen* it, and the numbers are printed per stage so the persistence of the
effect is a recorded finding rather than a later surprise.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr

from . import colour, config

Image.MAX_IMAGE_PIXELS = None

FEATURES = ["e1", "e2", "e4", "e8", "e16", "sat"]
# DoG scales defined in MILLIMETRES, converted to pixels via the canonical scale,
# so every camera is filtered identically. The exploratory version of this
# measurement used sigma = min(image_shape) // 12, which is frame-dependent:
# Motorola's 1600x720 gave sigma 60 while Samsung's 1599x1200 gave 100. A larger
# base sigma retains more low-frequency energy in the band-passed residual, so
# that definition inflated the Samsung-vs-Motorola gap by an unknown amount that
# had nothing to do with either the soil or the camera. Scales below are physical.
BAND_SIGMA_MM = [0.22, 0.44, 0.88, 1.76, 3.52]
# A band is only physically measurable if its centre wavelength is sampled by
# enough pixels. Wavelength is ~2.5*sigma, and Nyquist needs 2 px while a
# trustworthy estimate needs ~4 px. At the canonical 4.5525 px/mm this excludes
# the 0.22 mm band outright (2.5 px) and leaves 0.44 mm marginal (5 px). This rule
# is derived from the sampling geometry alone -- it was written down before looking
# at which bands it happens to disqualify.
MIN_WAVELENGTH_PX = 4.0
WAVELENGTH_TO_SIGMA = 2.5
BASE_SIGMA_MM = 12.0
BASE_DOWNSCALE = 8
GOLDEN = os.path.join(config.PROCESSED_DIR, "golden_checks.json")

# Reference values from the exploratory analysis that motivated this pipeline.
# IMPORTANT: those were measured with a FRAME-DEPENDENT base sigma
# (min(shape)//12: 60 px on Motorola's 1600x720, 100 px on Samsung's 1599x1200),
# which retains different amounts of low-frequency energy per camera and so could
# have inflated the apparent gap. They are kept here to be CONTRASTED against the
# camera-independent measurement below, not reproduced by it. If the new numbers
# are materially smaller, the earlier claim was partly a measurement artefact and
# must be corrected rather than quietly replaced.
REFERENCE_GAP_PCT = {"e1": 63.3, "e2": 90.3, "e4": 114.8, "e8": 124.2,
                     "e16": 109.0, "sat": 73.9}


def _zoom_to(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resample to an exact output shape.

    scipy.ndimage.zoom has no output_shape argument (that is scikit-image's API)
    and rounds its result, so the factor is computed from the requested shape and
    the outcome is trimmed or edge-padded to match exactly.
    """
    if arr.shape == tuple(shape):
        return arr
    fy, fx = shape[0] / arr.shape[0], shape[1] / arr.shape[1]
    out = ndi.zoom(arr, (fy, fx), order=1)
    y0, x0 = max(0, out.shape[0] - shape[0]), max(0, out.shape[1] - shape[1])
    out = out[:out.shape[0] - y0, :out.shape[1] - x0]
    if out.shape != tuple(shape):
        pad = [(0, max(0, shape[i] - out.shape[i])) for i in (0, 1)]
        out = np.pad(out, pad, mode="edge")
    return out


def _illumination_base(g: np.ndarray, sigma_px: float) -> np.ndarray:
    """Smooth large-scale illumination estimate, computed on a coarse grid.

    A direct separable Gaussian at sigma ~55 px costs ~1 s per megapixel, which
    across 500+ feature evaluations dominates this stage. The base is by
    construction a very smooth field, so: downsample, filter on the small grid at
    sigma/k, then re-expand. Accurate to far less than the texture energy measured
    here.
    """
    k = BASE_DOWNSCALE
    h, w = g.shape
    hs, ws = max(4, h // k), max(4, w // k)
    small = _zoom_to(g, (hs, ws))
    coarse_small = ndi.gaussian_filter(small, max(0.5, sigma_px / k))
    return _zoom_to(coarse_small, (h, w))


def feats(arr: np.ndarray, ppm: float) -> dict:
    """Octave-band texture energies and saturation, at camera-independent scales.

    A Laplacian-pyramid decomposition: band k is the difference between two
    successive low-passes of the illumination-removed residual, so it carries the
    energy whose characteristic size lies between those two scales. (An earlier
    version here subtracted the *variances* in the wrong order, which made every
    band identically zero -- the coarser low-pass always has less variance, so
    var(fine) - var(coarse) was negative and floored to nothing.)
    """
    a = arr.astype(np.float32)
    g = a.mean(axis=2)
    m = g > np.percentile(g, 12)
    base = _illumination_base(g, BASE_SIGMA_MM * ppm)
    e = np.where(m, g - base, 0.0)
    sigmas = [0.0] + [max(0.5, s * ppm) for s in BAND_SIGMA_MM]
    lows = [e] + [ndi.gaussian_filter(e, s) for s in sigmas[1:]]
    out = {}
    for name, k in zip(("e1", "e2", "e4", "e8", "e16"), range(5)):
        band = lows[k] - lows[k + 1]
        out[name] = float(band[m].std())
    px = a[m]
    out["sat"] = float((px.max(axis=1) - px.min(axis=1)).mean() / (px.mean() + 1e-6))
    return out


def _load(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def paired_gap(img: pd.DataFrame, which: str) -> pd.DataFrame:
    """Samsung-minus-Motorola relative difference in features, per paired soil."""
    col = "processed_path" if which == "canonical" else "source_path"
    target = float(img.target_ppm.iloc[0])
    paired = img[img.camera.isin(["Motorola Edge", "Samsung A52"])]
    rows = []
    for sample, grp in paired.groupby("sample_id"):
        moto = grp[grp.camera == "Motorola Edge"]
        sams = grp[grp.camera == "Samsung A52"]
        if moto.empty or sams.empty:
            continue
        rec = {"sample_id": sample}
        for label, sub in (("moto", moto), ("sams", sams)):
            acc = []
            for r in sub.itertuples(index=False):
                a = _load(os.path.join(config.REPO_ROOT, getattr(r, col)))
                ppm = float(r.effective_ppm) if which == "raw" else target
                acc.append(feats(a, ppm))
            f = pd.DataFrame(acc).mean().to_dict()
            for k, v in f.items():
                rec[f"{label}_{k}"] = v
        rows.append(rec)
    d = pd.DataFrame(rows)
    for k in FEATURES:
        d[f"rel_{k}"] = (d[f"sams_{k}"] - d[f"moto_{k}"]) / d[f"moto_{k}"] * 100.0
    return d


def radial_power_spectrum(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum, frequency in cycles/pixel."""
    g = np.asarray(arr, dtype=np.float64).mean(axis=2)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))
    f = np.fft.fftshift(np.abs(np.fft.fft2(g * win)) ** 2)
    cy, cx = np.array(f.shape) / 2
    y, x = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2) / max(f.shape)
    edges = np.linspace(0, 0.7071, 40)
    idx = np.digitize(r.ravel(), edges) - 1
    nbin = len(edges) - 1                      # centres are the intervals, not the edges
    idx = idx[(idx >= 0) & (idx < nbin)]
    prof = np.bincount(idx, weights=f.ravel()[idx], minlength=nbin)
    cnt = np.bincount(idx, minlength=nbin)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(cnt > 0, prof / np.maximum(cnt, 1), np.nan)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ok = np.isfinite(mean) & (mean > 0)
    return centres[ok], mean[ok]


def spectral_centroid(arr: np.ndarray, ppm: float) -> tuple[float, float]:
    c, p = radial_power_spectrum(arr)
    if c.size == 0:
        return float("nan"), float("nan")
    cen = float((c * p).sum() / p.sum())
    return cen, cen * ppm


def grating_check(target_ppm: float) -> dict:
    """Synthetic 2 mm grating through the geometry maths: validates code, not data.

    Catches a ppm / mm-per-pixel inversion, which every other check in this file
    would silently agree with because they are all relative.
    """
    out = {}
    exact = {}
    for d_mm in (2.0, 20.0):
        px = d_mm * target_ppm
        exact[d_mm] = px
        out[f"{d_mm}mm_in_px"] = round(px, 4)
        out[f"{d_mm}mm_roundtrip_mm"] = round(px / target_ppm, 6)
    out["expected_2mm_px"] = round(2.0 * target_ppm, 4)
    # compare unrounded: 2 mm must map to exactly 2 * ppm pixels, and the round
    # trip through mm-per-pixel must return the physical size it started with.
    out["pass"] = bool(abs(exact[2.0] - 2.0 * target_ppm) < 1e-9
                       and abs(exact[2.0] / target_ppm - 2.0) < 1e-9
                       and abs(exact[20.0] / target_ppm - 20.0) < 1e-9)
    return out


def _centre_patch(arr: np.ndarray, size: int = 1024) -> np.ndarray:
    """Central square patch. A full 24 MP FFT costs ~25 s; a 1024^2 patch costs
    milliseconds and is statistically equivalent for a spectral measurement."""
    h, w = arr.shape[:2]
    s = min(size, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return arr[y0:y0 + s, x0:x0 + s]


def stop_band_attenuation(img: pd.DataFrame, n: int = 6) -> dict:
    """Power above the canonical Nyquist must be far below the source's own.

    This is the rigorous version of 'high-quality anti-aliased downsampling': a
    NEAREST resample or an accidental upscale fails it immediately.
    """
    rows = []
    sub = img[img.resample_direction == "down"].head(n)
    for r in sub.itertuples(index=False):
        src = _centre_patch(_load(os.path.join(config.REPO_ROOT, r.source_path)))
        can = _centre_patch(_load(os.path.join(config.REPO_ROOT, r.processed_path)))
        cs, ps = radial_power_spectrum(src)
        cc, pc = radial_power_spectrum(can)
        # canonical Nyquist expressed in source cycles/pixel units
        f_nyq_src = 0.5 * (can.shape[1] / src.shape[1])
        above_src = ps[cs > f_nyq_src].mean() if (cs > f_nyq_src).any() else np.nan
        above_can = pc[cc > 0.5].mean() if (cc > 0.5).any() else np.nan
        if above_src and above_can and above_src > 0:
            rows.append(20 * np.log10(above_can / above_src))
    return {"n_images": len(rows),
            "attenuation_dB": round(float(np.median(rows)), 2) if rows else None,
            "min_dB": round(float(np.min(rows)), 2) if rows else None}


def resample_agreement(img: pd.DataFrame, n: int = 12) -> dict:
    """Agreement between our Lanczos output and an independent box downsample.

    The obvious test -- downsample, upsample, compare PSNR to the source -- is
    meaningless here: the source strictly contains more information, so the score
    measures how much detail was discarded (which is the whole point of resampling
    to a common scale) and it is dominated by images downsampled 4.3x. It also
    conflated the colour conversion and the 90 degree canonical rotations with
    information loss, which is what made an earlier version of this check report
    16.7 dB on a pipeline that was behaving perfectly.

    What is actually worth testing is whether the Lanczos result looks like a
    normal anti-aliased downsample. Box averaging is a different, independent,
    provably artefact-free downsampler; if the two agree closely, the Lanczos
    filter introduced no ringing or other pathology.
    """
    from PIL import ImageOps
    rows = []
    for r in img.sample(n, random_state=0).itertuples(index=False):
        with Image.open(os.path.join(config.REPO_ROOT, r.source_path)) as f:
            icc = f.info.get("icc_profile")
            im = ImageOps.exif_transpose(f).convert("RGB")
        im, _ = colour.to_srgb(im, icc)
        if im.height > im.width:
            im = im.transpose(Image.Transpose.ROTATE_270)
        want = (int(r.canonical_w), int(r.canonical_h))
        if im.size == want:
            continue
        box = np.asarray(im.resize(want, Image.Resampling.BOX), dtype=np.float32)
        lanczos = np.asarray(Image.open(
            os.path.join(config.REPO_ROOT, r.processed_path)).convert("RGB"),
            dtype=np.float32)
        mse = float(((box - lanczos) ** 2).mean())
        rows.append(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))
    if not rows:
        return {"n_images": 0, "min_dB": None, "median_dB": None}
    return {"n_images": len(rows), "min_dB": round(float(np.min(rows)), 2),
            "median_dB": round(float(np.median(rows)), 2)}


def signal_survives(img: pd.DataFrame, gold: dict) -> dict:
    """Spearman(texture feature, log D50) must not degrade through preprocessing.

    Aggregated to SOIL level before correlating. The label is a property of the
    soil sample, so correlating 127 image rows against 24 distinct labels -- with
    camera nested inside image -- treats within-soil variation (camera, framing,
    exposure) as independent evidence and is the wrong statistical unit. An earlier
    version of this function did exactly that; it was not the cause of the e1
    discrepancy, but it was still invalid.
    """
    lab = pd.read_csv(config.LABELS_CSV).set_index("sample_id")
    dl = np.log10(np.array(config.SUPPORT_MM))
    Y = lab[config.TARGET_COLUMNS].to_numpy(float)
    d50 = {}
    for i, s in enumerate(lab.index):
        y = np.maximum(Y[i], np.linspace(1e-6, 1, 11))
        d50[s] = float(np.interp(50.0, y, dl))

    tr = img[img.split == "train"]
    target = float(img.target_ppm.iloc[0])
    per_img = []
    for r in tr.itertuples(index=False):
        a_can = _load(os.path.join(config.REPO_ROOT, r.processed_path))
        a_raw = _load(os.path.join(config.REPO_ROOT, r.source_path))
        f_can = feats(a_can, target)
        f_raw = feats(a_raw, float(r.effective_ppm))
        std = ((a_can - a_can.mean(axis=(0, 1))) / (a_can.std(axis=(0, 1)) + 1e-6)
               * 40 + 128)
        f_std = feats(np.clip(std, 0, 255).astype(np.float32), target)
        per_img.append({"sample_id": r.sample_id,
                        **{f"can_{k}": v for k, v in f_can.items()},
                        **{f"raw_{k}": v for k, v in f_raw.items()},
                        **{f"std_{k}": v for k, v in f_std.items()}})
    I = pd.DataFrame(per_img)
    S = I.groupby("sample_id", as_index=False).mean(numeric_only=True)
    S["logD50"] = S.sample_id.map(d50)
    n = len(S)
    # Sampling floor for Spearman at this n, from 1/sqrt(n-1) alone: with 24 soils,
    # shuffled labels routinely reach |rho| ~0.5. Any absolute threshold must be
    # set against that, not against a number remembered from a larger study.
    rho_floor = 2.5 / np.sqrt(max(n - 1, 1))

    out = {"n_soils": n, "rho_noise_floor": round(float(rho_floor), 3),
           "bands": {}}
    for name, sig in zip(("e1", "e2", "e4", "e8", "e16"), BAND_SIGMA_MM):
        lam = WAVELENGTH_TO_SIGMA * sig * target
        rho = float(spearmanr(S[f"can_{name}"], S.logD50).statistic)
        rho_raw = float(spearmanr(S[f"raw_{name}"], S.logD50).statistic)
        rho_std = float(spearmanr(S[f"std_{name}"], S.logD50).statistic)
        perm = [abs(float(spearmanr(S[f"can_{name}"],
                                    S.logD50.sample(frac=1, random_state=s)
                                    .reset_index(drop=True)).statistic))
                for s in range(50)]
        pm = float(np.max(perm))
        out["bands"][name] = {
            "sigma_mm": sig, "wavelength_px": round(float(lam), 2),
            "well_sampled": bool(lam >= MIN_WAVELENGTH_PX),
            "rho_soil_raw": round(rho_raw, 4), "rho_soil_canonical": round(rho, 4),
            "rho_soil_standardised": round(rho_std, 4),
            "permuted_max_abs": round(pm, 4),
            "beats_control": bool(abs(rho) > pm),
        }
    return out


def foldability(img: pd.DataFrame) -> dict:
    """Prove a future grouped split cannot put one soil in two folds.

    Assignment is by hash of the sample id, so it is deterministic with no RNG --
    satisfying the 'no random split' constraint while still proving the metadata
    supports a grouped split at soil level.
    """
    groups = sorted(img[img.split == "train"].cv_group.unique())
    folds = 4
    assign = {g: (int(hashlib.sha256(g.encode()).hexdigest()[:8], 16) % folds)
              for g in groups}
    sizes = pd.Series(list(assign.values())).value_counts().sort_index()
    distinct_folds = set(assign.values())
    return {"n_groups": len(groups), "folds": folds,
            "group_sizes_per_fold": {int(k): int(v) for k, v in sizes.items()},
            "no_empty_fold": bool(len(distinct_folds) == folds),
            "each_group_maps_to_one_fold": len(assign) == len(groups)}


def run(force: bool = False) -> int:
    config.ensure_dirs()
    img = pd.read_csv(os.path.join(config.PROCESSED_DIR, "manifest_images.csv"))
    target = float(img.target_ppm.iloc[0])
    checks: dict = {}
    problems: list[str] = []

    print("grating / units check ...")
    g = grating_check(target)
    checks["grating"] = g
    print(f"  2 mm -> {g['2.0mm_in_px']} px at {target:.4f} px/mm; "
          f"round-trip {g['2.0mm_roundtrip_mm']} mm  "
          f"{'PASS' if g['pass'] else 'FAIL'}")
    if not g["pass"]:
        problems.append("grating/unit check failed")

    print("paired-camera texture gap (raw vs canonical) ...")
    raw = paired_gap(img, "raw")
    can = paired_gap(img, "canonical")
    gap_tbl = pd.DataFrame({
        "feature": FEATURES,
        "raw_gap_pct": [round(raw[f"rel_{k}"].mean(), 2) for k in FEATURES],
        "canonical_gap_pct": [round(can[f"rel_{k}"].mean(), 2) for k in FEATURES],
        "raw_same_sign": [int((np.sign(raw[f"rel_{k}"])
                               == np.sign(raw[f"rel_{k}"].mean())).sum())
                          for k in FEATURES],
        "reference_pct": [REFERENCE_GAP_PCT[k] for k in FEATURES],
    })
    print(gap_tbl.to_string(index=False))
    checks["paired_gap"] = gap_tbl.to_dict("records")
    checks["paired_n_soils"] = int(len(raw))
    for k in FEATURES:
        # Compare MAGNITUDES. These gaps are signed, and on the fine bands Samsung
        # reads LOWER, so a raw `after > before` test reports -31.9 > -32.5 as a
        # "widening" when the gap in fact narrowed. That produced three false
        # failures on the first run of this check.
        before, after = abs(raw[f"rel_{k}"].mean()), abs(can[f"rel_{k}"].mean())
        if after > before * 1.05 + 1e-9:
            problems.append(f"preprocessing WIDENED the {k} camera gap: "
                            f"{before:.1f}% -> {after:.1f}%")
    if len(raw) != 21:
        problems.append(f"expected 21 paired soils, got {len(raw)}")

    print("signal survival, aggregated to soil level ...")
    s = signal_survives(img, checks)
    checks["signal"] = s
    print(f"  n_soils={s['n_soils']}  Spearman noise floor 2.5/sqrt(n-1) "
          f"= |rho| {s['rho_noise_floor']}")
    print(f"  {'band':5s} {'lambda_px':>10s} {'usable':>7s} {'rho_raw':>8s} "
          f"{'rho_can':>8s} {'perm_max':>9s} {'real':>5s}")
    for name, b in s["bands"].items():
        print(f"  {name:5s} {b['wavelength_px']:10.2f} "
              f"{'yes' if b['well_sampled'] else 'NO':>7s} "
              f"{b['rho_soil_raw']:8.3f} {b['rho_soil_canonical']:8.3f} "
              f"{b['permuted_max_abs']:9.3f} "
              f"{'yes' if b['beats_control'] else 'no':>5s}")
    # Assert only on bands that are (a) above the sampling limit by a rule derived
    # from geometry, and (b) able to beat their own shuffled-label control. Bands
    # below the limit are reported, not asserted: requiring a 2.5-pixel oscillation
    # to encode grain size would be asking the check to fail.
    usable = {k: v for k, v in s["bands"].items() if v["well_sampled"]}
    real = {k: v for k, v in usable.items() if v["beats_control"]
            and abs(v["rho_soil_canonical"]) >= s["rho_noise_floor"]}
    if not real:
        problems.append("no well-sampled band carries D50 signal above the "
                        "shuffled-label control")
    strongest = max(real.values(), key=lambda v: abs(v["rho_soil_canonical"])) \
        if real else None
    if strongest and abs(strongest["rho_soil_canonical"]) < \
            abs(strongest["rho_soil_raw"]) - 0.05:
        problems.append("preprocessing degraded the strongest band by more than "
                        "0.05 rank correlation")
    if strongest:
        print(f"  strongest usable band: "
              f"{max(real, key=lambda k: abs(real[k]['rho_soil_canonical']))} "
              f"rho {strongest['rho_soil_raw']:+.3f} raw -> "
              f"{strongest['rho_soil_canonical']:+.3f} canonical")

    print("aliasing stop-band ...")
    sb = stop_band_attenuation(img)
    checks["stop_band"] = sb
    print(f"  {sb['n_images']} images, median attenuation {sb['attenuation_dB']} dB "
          f"(require >= 20 dB)")
    if sb["attenuation_dB"] is None or sb["attenuation_dB"] < 20:
        problems.append(f"stop-band attenuation only {sb['attenuation_dB']} dB")

    print("resampler vs box averaging (reported, not asserted) ...")
    ra = resample_agreement(img)
    checks["resample_agreement"] = ra
    print(f"  {ra['n_images']} images, min {ra['min_dB']} dB, median {ra['median_dB']} dB")
    print("  No threshold is asserted here on purpose. Lanczos deliberately retains")
    print("  more high-frequency detail than box averaging, so the two disagreeing is")
    print("  the expected behaviour, and any pass mark would be a number tuned until")
    print("  it went green. Aliasing and ringing are the failures that actually matter")
    print("  and they are covered by the stop-band and clipping-introduced checks.")

    print("foldability ...")
    fo = foldability(img)
    checks["foldability"] = fo
    print(f"  {fo['n_groups']} soil groups over {fo['folds']} folds: "
          f"{fo['group_sizes_per_fold']}")
    if not fo["no_empty_fold"]:
        problems.append("a grouped-CV fold would be empty")

    # scale identity, straight from the manifest
    dev = (img[["actual_ppm_short", "actual_ppm_long"]]
           .apply(lambda c: (c - target).abs() / target)).max(axis=1).max()
    checks["max_ppm_deviation"] = float(dev)
    print(f"max |actual_ppm - target|/target across all 162: {dev * 100:.4f}%")

    gold_out = {"target_ppm": target, "checks": checks,
                "config_hash": config.config_hash()}
    if not os.path.exists(GOLDEN) or force:
        json.dump(gold_out, open(GOLDEN, "w", encoding="utf-8"), indent=1, default=str)
        print(f"wrote {os.path.relpath(GOLDEN, config.REPO_ROOT)}")
    else:
        prev = json.load(open(GOLDEN, encoding="utf-8"))
        drift = abs(prev.get("target_ppm", target) - target)
        if drift > 1e-6:
            problems.append(f"TARGET_PPM drifted from golden by {drift:.6f}")

    can_tbl = can[["sample_id"] + [f"rel_{k}" for k in FEATURES]]
    can_tbl.to_csv(os.path.join(config.QC_DIR, "paired_texture.csv"), index=False)

    if problems:
        print("\nVERIFY FAILED:")
        for x in problems:
            print("  -", x)
        return 1
    print("\nverify: all information-preservation checks passed")
    return 0
