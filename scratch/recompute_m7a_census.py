"""Independently recompute every number the M7-A Task 3 census publishes, from the artifacts only.

This script does not import m7a_data's aggregation or reporting helpers. It reads the four published
files with pandas, recomputes each statistic with numpy, and compares against data/census.json. Any
disagreement is printed with its magnitude. Exit 0 means every published number was reproduced by a
second, independent path.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APPROACH = Path("Model 7/Approach A")
DATA = APPROACH / "data"
KEYS = ("A1", "A2", "A3", "A4")
Z = 10.0
MAG = 1e6
BAD = []
N = 0


def cmp(label, got, want, tol=0.0):
    """Report one comparison. Floats compare with an absolute tolerance; counts exactly."""
    global N
    N += 1
    if isinstance(got, (list, tuple)) or isinstance(want, (list, tuple)):
        ok = list(got) == list(want)          # exact: these artifacts round-trip through repr
        gap = "sequence"
    elif isinstance(got, float) or isinstance(want, float):
        ok = got is not None and want is not None and abs(float(got) - float(want)) <= tol
        gap = "n/a" if (got is None or want is None) else "%.3g" % abs(float(got) - float(want))
    else:
        ok, gap = got == want, "n/a"
    print("%-5s %-58s reported %-14s recomputed %-14s gap %s"
          % ("PASS" if ok else "FAIL", label, want, got, gap))
    if not ok:
        BAD.append(label)


def pcts(v):
    fin = v[np.isfinite(v)]
    if not fin.size:
        return [None] * 5
    return [float(x) for x in np.percentile(fin, [0, 5, 50, 95, 100])]


def outlier(v):
    fin = v[np.isfinite(v)]
    med = float(np.percentile(fin, 50.0))
    mad = float(np.percentile(np.abs(fin - med), 50.0))
    if mad > 0.0:
        z = np.abs(fin - med) / mad
        beyond, maxz = int(np.count_nonzero(z > Z)), float(np.max(z))
    else:
        beyond, maxz = None, None
    same_sign = bool((fin > 0).all() or (fin < 0).all())
    if same_sign:
        s = np.log(np.abs(fin))
        mlog = float(np.percentile(s, 50.0))
        dlog = float(np.percentile(np.abs(s - mlog), 50.0))
        if dlog > 0.0:
            zl = np.abs(s - mlog) / dlog
            logb, logmax = int(np.count_nonzero(zl > Z)), float(np.max(zl))
        else:
            logb, logmax = None, None
    else:
        logb, logmax = None, None
    magv = np.abs(fin)
    return dict(n_finite=int(fin.size), mad=mad, mad_is_zero=bool(mad == 0.0),
                n_distinct=int(len(np.unique(fin))), median=med, count=beyond, maxz=maxz,
                applicable=same_sign, log_count=logb, log_maxz=logmax,
                max_abs=float(np.max(magv)), above_1e6=int(np.count_nonzero(magv > MAG)),
                whole=bool(np.all(magv > MAG)))


def main():
    j = json.loads((DATA / "census.json").read_text(encoding="utf-8"))
    csv = pd.read_csv(DATA / "tile_census.csv", float_precision="round_trip")
    img = pd.read_csv(DATA / "image_features_m7a.csv", index_col=0, float_precision="round_trip")
    soil = pd.read_csv(DATA / "soil_features_m7a.csv", index_col=0, float_precision="round_trip")
    man = pd.read_csv(Path("data/processed_meta/manifest_tiles.csv"))
    q = man[(man.tile_size_px == 256) & man.materialized & (man.soil_fraction >= 0.50)]

    print("== levels and partitions")
    cmp("total_tiles", len(csv), j["total_tiles"])
    cmp("csv rows equal the manifest's qualifying tiles", len(csv), len(q))
    counts = csv.status.value_counts().to_dict()
    cmp("successful_tiles", int(counts.get("ok", 0)), j["successful_tiles"])
    cmp("tiles_with_any_nan", len(csv) - int(counts.get("ok", 0)), j["tiles_with_any_nan"])
    cmp("degenerate_tiles", int(counts.get("degenerate_tile", 0)), j["degenerate_tiles"])
    cmp("nan_key_tiles", int(counts.get("nan_keys", 0)), j["nan_key_tiles"])
    cmp("invalid_calls", 0, j["invalid_calls"])
    cmp("per_status sums to the corpus", sum(counts.values()), j["total_tiles"])
    cmp("images", len(img), j["images"])
    cmp("soils", len(soil), j["soils"])
    by_split = q.split.value_counts().to_dict()
    cmp("tiles_by_split", {str(k): int(v) for k, v in by_split.items()}, j["tiles_by_split"])
    cmp("train_soils", int(q[q.split == "train"].sample_id.nunique()), j["train_soils"])
    cmp("test_soils", int(q[q.split == "test"].sample_id.nunique()), j["test_soils"])
    cmp("normalized_ppm_values", sorted(float(x) for x in q.normalized_ppm.unique()),
        j["normalized_ppm_values"])
    cmp("tile_w_mm", float(q.tile_w_mm.iloc[0]), j["tile_w_mm"], tol=0.0)

    print("\n== tile-level per-key statistics")
    for k in KEYS:
        v = csv[k].to_numpy(float)
        r = j["per_key"][k]
        cmp("%s nan_count" % k, int(np.count_nonzero(np.isnan(v))), r["nan_count"])
        cmp("%s inf_count" % k, int(np.count_nonzero(np.isinf(v))), r["inf_count"])
        cmp("%s nonfinite_count = nan + inf" % k,
            int(np.count_nonzero(np.isnan(v))) + int(np.count_nonzero(np.isinf(v))),
            r["nonfinite_count"])
        cmp("%s n_finite" % k, int(np.isfinite(v).sum()), r["n_finite"])
        for field, got, want in zip(("min", "p5", "median", "p95", "max"), pcts(v),
                                    (r["min"], r["p5"], r["median"], r["p95"], r["max"])):
            cmp("%s %s" % (k, field), got, want, tol=1e-12)
        cmp("%s NaN flags agree with the values" % k,
            int(csv["%s_nan" % k].astype(bool).sum()), r["nan_count"])
        cmp("%s n_nan columns sum to the NaN cells" % k,
            int(sum(csv["%s_nan" % kk].astype(bool).sum() for kk in KEYS)),
            int(sum(j["per_key"][kk]["nan_count"] for kk in KEYS)))

    print("\n== assembled levels")
    for level, frame, block in (("image", img, "image_percentiles"),
                                ("soil", soil, "soil_percentiles")):
        for k in KEYS:
            v = frame[k].to_numpy(float)
            r = j[block][k]
            cmp("%s %s min/median/max" % (level, k), pcts(v)[0::2],
                [r["min"], r["median"], r["max"]], tol=1e-12)
            cmp("%s %s nonfinite cells" % (level, k),
                int(np.count_nonzero(~np.isfinite(v))), r["nonfinite_count"])

    print("\n== extreme outliers, both rules, tile level")
    for k in KEYS:
        v = csv[k].to_numpy(float)
        o, r = outlier(v), j["extreme_outlier"]["keys"][k]
        cmp("tile %s mad" % k, o["mad"], r["mad"], tol=1e-12)
        cmp("tile %s mad_is_zero" % k, o["mad_is_zero"], r["mad_is_zero"])
        cmp("tile %s n_distinct" % k, o["n_distinct"], r["n_distinct"])
        cmp("tile %s median" % k, o["median"], r["median"], tol=1e-12)
        cmp("tile %s count beyond 10 MAD" % k, o["count"], r["count_beyond_10_mad"])
        cmp("tile %s max abs z" % k, o["maxz"], r["max_abs_z"], tol=1e-9)
        cmp("tile %s log rule applicable" % k, o["applicable"], r["log_two_sided_applicable"])
        cmp("tile %s count beyond 10 log MAD" % k, o["log_count"], r["count_beyond_10_mad_log"])
        cmp("tile %s max abs log z" % k, o["log_maxz"], r["max_abs_z_log"], tol=1e-9)
        cmp("tile %s max |value|" % k, o["max_abs"], r["max_abs_value"], tol=1e-12)
        cmp("tile %s values above 1e6" % k, o["above_1e6"], r["n_abs_above_1e6"])
        cmp("tile %s whole distribution above 1e6" % k, o["whole"],
            r["whole_distribution_above_1e6"])

    print("\n== extreme outliers, soil level")
    for k in KEYS:
        v = soil[k].to_numpy(float)
        o, r = outlier(v), j["extreme_outlier"]["soil_level"][k]
        cmp("soil %s count beyond 10 MAD" % k, o["count"], r["count_beyond_10_mad"])
        cmp("soil %s count beyond 10 log MAD" % k, o["log_count"], r["count_beyond_10_mad_log"])
        cmp("soil %s max abs z" % k, o["maxz"], r["max_abs_z"], tol=1e-9)
        cmp("soil %s log rule applicable" % k, o["applicable"], r["log_two_sided_applicable"])

    print("\n== finiteness of the published arrays")
    for level, frame in (("tile", csv), ("image", img), ("soil", soil)):
        arr = frame[list(KEYS)].to_numpy(float)
        r = j["finiteness"][level]
        cmp("%s cells" % level, int(arr.size), r["cells"])
        cmp("%s nan_cells" % level, int(np.count_nonzero(np.isnan(arr))), r["nan_cells"])
        cmp("%s inf_cells" % level, int(np.count_nonzero(np.isinf(arr))), r["inf_cells"])
        cmp("%s non_finite_cells" % level, int(np.count_nonzero(~np.isfinite(arr))),
            r["non_finite_cells"])
        cmp("%s tiles carrying a non-finite value" % level,
            int(np.count_nonzero(~np.isfinite(arr).any(axis=1))), r["tiles_with_nonfinite"])
    cmp("non_finite_tiles (surface)", j["finiteness"]["tile"]["tiles_with_nonfinite"],
        j["non_finite_tiles"])
    cmp("the published design holds no non-finite cell",
        sum(j["finiteness"][l]["non_finite_cells"] for l in ("tile", "image", "soil")), 0)

    print("\n== schema and leakage boundaries")
    cmp("image columns", list(img.columns), ["A1", "A2", "A3", "A4", "sample_id"])
    cmp("soil columns", list(soil.columns), list(KEYS))
    cmp("census columns", list(csv.columns), j["columns"]["tile_census"])
    cmp("no camera-derived column in any artifact",
        sorted({c for f in (img, soil, csv) for c in f.columns
                if any(t in str(c).lower() for t in ("camera", "cam", "device", "exif", "icc"))}),
        [])
    cmp("image index is the parent image path", str(img.index.name), "parent_image_path")
    cmp("soil index is the sample id", str(soil.index.name), "sample_id")
    cmp("every census tile path is in the manifest",
        int(csv.tile_path.isin(q.tile_path).sum()), len(csv))
    cmp("every image row's soil agrees with the manifest",
        sorted({str(s) for s in img.sample_id}),
        sorted({str(q[q.parent_image_path == p].sample_id.iloc[0]) for p in img.index}))

    print("\n== artifact self-description")
    cmp("warnings are counted by stage, all zero", sum(j["warnings_by_stage"].values()), 0)
    cmp("warnings_count agrees with the stage tally", j["warnings_count"],
        sum(j["warnings_by_stage"].values()))
    cmp("errors_count", j["errors_count"], 0)
    cmp("the meaning of non-finite is stated once", isinstance(j["finiteness"]["meaning"], str)
        and len(j["finiteness"]["meaning"]) > 20, True)
    cmp("the run was not scrambled", bool(j["scrambled"]), False)

    print("\n%d comparisons, %d disagreed" % (N, len(BAD)))
    for b in BAD:
        print("  - " + b)
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
