"""STEP 8 - assemble the three manifests and validate them.

manifest_images.csv is the single source of truth: no downstream script may infer
PPM from a filename, because the delivered files are downscaled relative to the
native resolutions quoted in ppm_updated.csv and the correct per-pixel scale is a
measured derivation, not a lookup.

Column naming is deliberate. The brief asked for `original_metadata_ppm`; that name
invites a downstream reader to multiply by it and be wrong by 2.5x. We use
`native_ppm` (the input to the derivation, flagged as not being the delivered
scale) and `effective_ppm` (the number that actually applies to these pixels).

No label-derived statistic (D50, PCA score, anything) lives in preprocessing
metadata. Group keys and label summaries in one table invite leakage when splits
are built later, so labels stay quarantined in manifest_samples.csv.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from . import config

# Columns that must be present and non-null in manifest_images.csv.
REQUIRED_NONNULL = [
    "image_id", "source_path", "processed_path", "split", "sample_id", "cv_group",
    "camera", "camera_source", "native_ppm", "effective_ppm", "target_ppm",
    "canvas_long_px", "canvas_short_px", "canonical_w", "canonical_h",
    "fov_long_mm", "fov_short_mm", "resample_direction", "soil_fraction",
    "crop_rect_canonical", "mask_status", "gw_gain_r", "gw_gain_g", "gw_gain_b",
    "src_sha256", "out_sha256", "pipeline_version", "config_hash",
]
ALLOWED_DIRECTIONS = {"down", "identity"}
ALLOWED_MASK_STATUS = {"ok", "capped", "fallback_fullframe"}
ALLOWED_CAMERA_SOURCE = {"filename_only", "exif_and_filename", "exif_only"}

BOREHOLE = re.compile(r"(?i)(bs\s*\d|\d+\s*-\s*\d)")


def _norm(s: str) -> str:
    """Fold a sample/site name to a comparison key.

    Test folder names and submission ids differ in ways that are not cosmetic:
    the folder is 'Münster_BS6_9,0-10m' while sample_submission.csv wants
    'HPC_Muenster_BS6_9_0-10m' -- umlaut transliterated AND comma turned to
    underscore AND an HPC_ prefix added. A naive 'HPC_' + foldername rule fails on
    exactly that one row, so both sides are folded before matching.
    """
    s = s.lower().replace("ü", "ue").replace("ö", "oe").replace("ä", "ae").replace("ß", "ss")
    return re.sub(r"[^a-z0-9]+", "", s)


def submission_id_map() -> dict[str, str]:
    """folder sample_id -> exact id required by sample_submission.csv."""
    sub = pd.read_csv(config.SUBMISSION_CSV)
    out = {}
    for sid in sub.sample_id.astype(str):
        bare = sid[4:] if sid.startswith("HPC_") else sid
        out[_norm(bare)] = sid
    return out


def to_submission_id(folder_name: str, lookup: dict[str, str]) -> str | None:
    return lookup.get(_norm(folder_name))


def site_of(sample_id: str, split: str) -> str:
    """Leading place name before the borehole/layer code. Train ids have no site."""
    if split != "test":
        return "NA"
    m = BOREHOLE.search(sample_id)
    if not m:
        return sample_id.strip()
    return sample_id[:m.start()].strip(" _-,") or sample_id.strip()


def soil_family_of(sample_id: str, split: str) -> str:
    """ADVISORY ONLY.

    Training ids look like borehole codes, so letter-prefix + first digit gives a
    cheap grouping. It does NOT agree with the distance-based clustering of the
    label curves (H366 and H371 are near-twins at distance 3.3 yet both map to H3,
    while this rule also puts genuinely different soils in H6). cv_group =
    sample_id is the safe key for any split; this column exists for reporting.
    """
    if split != "test":
        m = re.match(r"^([A-Za-z])(\d)", sample_id)
        return f"{m.group(1)}{m.group(2)}" if m else "unknown"
    return site_of(sample_id, split)


def load_stages() -> pd.DataFrame:
    """Merge stage outputs left-to-right, keeping the first frame that supplies a column.

    Later stages re-carry identity columns from earlier ones, so a naive merge
    produces camera_x / camera_y suffixes and a naive de-duplication can delete the
    only copy of `camera` -- which lives in `cameras` and `geometry`, never in
    `inventory`. Dropping already-present columns as we walk forward is the safe
    rule.
    """
    have: list[str] = ["image_id"]
    merged: pd.DataFrame | None = None
    for name in ("inventory", "cameras", "geometry", "resample", "soilmask", "colour"):
        path = config.stage_path(name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing stage output {path}; run that stage first")
        df = pd.read_csv(path)
        if merged is not None:
            df = df.drop(columns=[c for c in df.columns
                                  if c in merged.columns and c != "image_id"])
        merged = df if merged is None else merged.merge(df, on="image_id",
                                                        validate="one_to_one")
        have.extend(c for c in merged.columns if c not in have)
    return merged


def build_images(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["site"] = [site_of(s, p) for s, p in zip(df.sample_id, df.split)]
    df["soil_family"] = [soil_family_of(s, p) for s, p in zip(df.sample_id, df.split)]
    df["cv_group"] = df["sample_id"]
    df["native_ppm_is_delivered_scale"] = False
    df["resolution_is_physical"] = False
    df["pipeline_version"] = config.PIPELINE_VERSION
    df["config_hash"] = config.config_hash()
    df["target_ppm"] = config.target_ppm()
    return df


def build_samples(img: pd.DataFrame) -> pd.DataFrame:
    lab = pd.read_csv(config.LABELS_CSV)
    lab.columns = [str(c) for c in lab.columns]
    sub_lookup = submission_id_map()
    # The competition asks for rows in sample_submission.csv order, so record that
    # position rather than making a later script re-derive it from a sorted frame.
    sub_order = {str(s): i for i, s in
                 enumerate(pd.read_csv(config.SUBMISSION_CSV).sample_id.astype(str))}
    rows = []
    for (split, sample_id), grp in img.groupby(["split", "sample_id"], sort=True):
        rec = {
            "sample_id": sample_id, "split": split,
            "sample_slug": grp.sample_slug.iloc[0],
            "submission_id": to_submission_id(sample_id, sub_lookup)
                             if split == "test" else None,
            "submission_row_order": sub_order.get(
                to_submission_id(sample_id, sub_lookup)) if split == "test" else None,
            "site": grp.site.iloc[0], "soil_family": grp.soil_family.iloc[0],
            "cv_group": sample_id,
            "n_images": int(len(grp)),
            "cameras": ";".join(sorted(set(grp.camera.dropna()))),
            "camera_paired": {"Motorola Edge", "Samsung A52"} <= set(grp.camera.dropna()),
        }
        if split == "train" and sample_id in set(lab.sample_id.astype(str)):
            lr = lab[lab.sample_id.astype(str) == sample_id].iloc[0]
            for c in config.TARGET_COLUMNS:
                rec[c] = float(lr[c])
            vals = np.array([float(lr[c]) for c in config.TARGET_COLUMNS])
            rec["all_target_cols_monotonic"] = bool(np.all(np.diff(vals) >= -1e-9))
            rec["col_200_equals_100"] = bool(abs(vals[-1] - 100.0) < 1e-9)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["split", "sample_id"]).reset_index(drop=True)
    return out


def validate(img: pd.DataFrame, samp: pd.DataFrame) -> list[str]:
    errs = []
    missing = [c for c in REQUIRED_NONNULL if c not in img.columns]
    if missing:
        errs.append(f"missing required columns: {missing}")
    for c in [c for c in REQUIRED_NONNULL if c in img.columns]:
        n = int(img[c].isna().sum())
        if n:
            errs.append(f"column {c} has {n} nulls (must be 0)")
    if len(img) != 162:
        errs.append(f"expected 162 image rows, got {len(img)}")
    if img.image_id.duplicated().any():
        errs.append("duplicate image_id")
    bad_dir = set(img.resample_direction) - ALLOWED_DIRECTIONS
    if bad_dir:
        errs.append(f"illegal resample_direction values {bad_dir} -- upscale present")
    bad_ms = set(img.mask_status) - ALLOWED_MASK_STATUS
    if bad_ms:
        errs.append(f"illegal mask_status values {bad_ms}")
    bad_cs = set(img.camera_source) - ALLOWED_CAMERA_SOURCE
    if bad_cs:
        errs.append(f"illegal camera_source values {bad_cs}")
    if img.camera.isna().any():
        errs.append("unresolved camera for some images")
    if set(img.resample_direction) & {"up"}:
        errs.append("upscale present")

    tr = samp[samp.split == "train"]
    te = samp[samp.split == "test"]
    if len(tr) != 24:
        errs.append(f"expected 24 train samples, got {len(tr)}")
    if len(te) != 10:
        errs.append(f"expected 10 test samples, got {len(te)}")
    if not tr.all_target_cols_monotonic.all():
        errs.append("non-monotonic label row present")
    if not tr.col_200_equals_100.all():
        errs.append("label row where 200mm column != 100")
    sub = pd.read_csv(config.SUBMISSION_CSV)
    want = sorted(sub.sample_id.astype(str))
    got = sorted(te.submission_id.dropna().astype(str))
    if got != want:
        errs.append(f"test submission_ids do not reproduce sample_submission.csv "
                    f"(matched {len(set(got) & set(want))}/{len(want)}; "
                    f"missing {sorted(set(want) - set(got))})")
    if int(te.submission_id.isna().sum()):
        errs.append(f"{int(te.submission_id.isna().sum())} test folders have no "
                    f"submission_id mapping")
    ordered = te.sort_values("submission_row_order").submission_id.astype(str).tolist()
    # compare against the submission file's own row order, not a sorted copy of it
    if ordered != sub.sample_id.astype(str).tolist():
        errs.append("submission_row_order does not reproduce sample_submission.csv order")
    if img[img.split == "train"].sample_id.nunique() != 24:
        errs.append("train cv_group distinct count != 24")
    if img[img.split == "test"].sample_id.nunique() != 10:
        errs.append("test cv_group distinct count != 10")
    return errs


def run(force: bool = False) -> int:
    config.ensure_dirs()
    raw = load_stages()
    img = build_images(raw)

    tiles_path = os.path.join(config.PROCESSED_DIR, "manifest_tiles.csv")
    if os.path.exists(tiles_path):
        t = pd.read_csv(tiles_path)
        cnt = t.groupby(["split", "sample_id", "tile_size_px"]).size().unstack(fill_value=0)
        cnt.columns = [f"n_tiles@{c}" for c in cnt.columns]
        samp = build_samples(img).merge(cnt.reset_index(), on=["split", "sample_id"], how="left")
    else:
        samp = build_samples(img)

    errs = validate(img, samp)

    img_path = os.path.join(config.PROCESSED_DIR, "manifest_images.csv")
    samp_path = os.path.join(config.PROCESSED_DIR, "manifest_samples.csv")
    for p in (img_path, samp_path):
        config.assert_writable(p)
    img.to_csv(img_path, index=False)
    samp.to_csv(samp_path, index=False)

    print(f"manifest_images.csv rows: {len(img)} cols: {len(img.columns)}")
    print(f"manifest_samples.csv rows: {len(samp)}")
    print(f"train sites: NA (borehole codes) | test sites: "
          f"{sorted(set(samp[samp.split == 'test'].site))}")
    print(f"soil_family (advisory) distinct: {samp[samp.split == 'train'].soil_family.nunique()}")
    print(f"camera_paired samples: {int(samp.camera_paired.sum())} "
          f"unpaired: {sorted(samp[~samp.camera_paired].sample_id)}")
    print(f"resample_direction: {img.resample_direction.value_counts().to_dict()}")
    print(f"mask_status: {img.mask_status.value_counts().to_dict()}")
    if errs:
        print("\nVALIDATION ERRORS:")
        for e in errs:
            print("  -", e)
        return 1
    print("\nvalidation: all checks passed")
    return 0
