"""Contract test for m7a_data.py. Run: python check_m7a_data.py

WHAT THIS FILE ASSERTS (Task 3 of the pre-registration: assembly + census, nothing else)
  frozen filter  tile_table is exactly manifest_tiles.csv filtered to tile_size_px == tv.TILE_PX &
      materialized & soil_fraction >= tv.MIN_SOIL_FRACTION, which is the 1946 tiles (1541 train /
      405 test) the pre-registration counted. The filter uses tv's constants, not copied literals.
  pixel set      the image row set equals transfer_eval.image_features' row set exactly, so A1-A4
      are computed on the same pixels the frozen 12 are. The loader is imported, not reimplemented.
  dropped        camera, camera_fam, cv_group, source_effective_ppm, source_image_width and
      source_image_height are dropped at load. normalized_ppm survives only as the physical scale
      argument to tile_spatial; it is not in the image or soil design.
  THREE OUTCOMES (the heart of the census, and they must not be conflated):
    (1) invalid call    - a ValueError or any unexpected exception, INCLUDING a key tile_spatial
                          left NaN while its own registered definition measures it (a disagreement
                          inside the implementation is not a fact about a tile). ABORTS, with the
                          tile named in the structured fields. Never counted as degeneracy, never
                          converted to NaN, never skipped. Pinned by tests that force a ValueError,
                          a RuntimeError, a missing tile and a fabricated NaN.
    (2) valid tile,     - TileDegenerate for the whole tile, or a NaN in one key that the key's own
        degenerate      definition refuses on the same tile. Counted, per-key NaN tallies, reason
                          string captured.
    (3) valid finite    - counted, with percentiles.
  aggregation      sf.image_block then sf.soil_block, the tested path, and it is pinned
      behaviourally: a spy counts the calls, both frames are re-derived from the blocks and
      compared bit for bit, and a column the blocks refuse (an inf) is proved to be refused by the
      assembly rather than swallowed the way a plain median would.
  warnings         a warning is a FAILURE, and every stage of the pass is watched: the tile loop,
      the image aggregation and the soil aggregation. sf._median_column warns at aggregation, so a
      sink around the loop alone would have printed warnings: 0 over a run whose stderr was not
      silent. Forced-warning legs arm-test the sink, so the claim can fail.
  outliers         per key, two robust looks: |value - median| / MAD, and the same rule on
      log|value| where the key keeps one sign (a strictly positive key's lower tail is invisible to
      the absolute rule, which is the whole of A3's and much of A2's). A zero MAD and an
      inapplicable log rule are each reported as their own condition. Reported only: nothing is
      clamped, transformed, winsorised or dropped, and the artifact is compared against memory.
  finiteness       counted off the assembled numpy arrays before anything is serialised, and the
      writer refuses to launder a non-finite into null. A planted NaN/Infinity must be caught.
  artifacts        data/tile_census.csv, census.json, image_features_m7a.csv and
      soil_features_m7a.csv, the last two being the design frames so the matrix a fit would consume
      is auditable offline. Round-trip read and compared bit for bit against memory.
  determinism      the whole assembly runs twice in this process and once in a fresh interpreter;
      the three vectors (tile, image, soil) must hash byte-identically across all three, and the
      published tile census must be byte-identical to the standalone census run.
  boundaries       no CV, no fitting, no alpha, no predictions, no submission, no gate. Tested by
      watching which files pandas opens and by patching the model's own entry points to explode,
      not by scanning m7a_data's source for words. data/ is read-only: it is fingerprinted before
      and after.

WHAT THIS FILE DOES NOT DO: it does not read m7a_data's source text to decide whether the code is
right. Six checks used to, and one of them measurably shaped the implementation (the module routed
its percentiles through np.percentile specifically so a ban on the token ".median(" would not fire).
A test that the implementation can satisfy by rewording itself is not a test.

MEASURED (Task 3, real corpus, ppm = 4.552516 for every tile, tile_w_mm = 56.23264)
  d.find_input_root() -> the repository root; d.tile_table(root) -> 1946 rows, 1541 train / 405 test
  d.spatial_soil_table(root) -> 34 rows (24 train, 10 test), describe().loc[min / 50% / max]:
              A1        A2       A3        A4
    min   -0.264916   0.892047   0.015625  -0.000131
    50%   -0.184202   0.999912   0.044922   0.003160
    max   -0.103116   1.173272   0.468750   0.014159
  d.NAN_REPORT: {'n_tiles': 1946, 'degenerate_mask': 0, 'non_finite': 0, 'tiles_with_nan': 0,
    'nan_cells': 0, 'non_finite_cells': 0, 'nan_rate': 0.0, 'nan_only_rate': 0.0,
    'non_finite_rate': 0.0, 'per_key_nan': {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0},
    'per_key_nonfinite': {'A1': 0, 'A2': 0, 'A3': 0, 'A4': 0}, 'invalid_calls': 0, 'warnings': 0,
    'warnings_by_stage': {'tiles': 0, 'image_block': 0, 'soil_block': 0}, 'meaning': <one meaning>}

THE FULL-DATA CENSUS, published to data/ (1946 tiles read, none written):
  outcome (3) valid finite tile (all four keys) ...... 1946
  outcome (2) valid tile with a degenerate statistic .. 0  (degenerate_tile 0, nan_keys 0)
  outcome (1) invalid call ............................ 0  (a pass containing one aborts)
  warnings 0 at all three stages, errors 0, images 162 (127 train / 35 test), soils 34.
  Finiteness measured on the arrays: tile 7784 cells, image 648, soil 136 - zero NaN, zero inf.
  Every key: nan_count 0, inf_count 0. min / p5 / median / p95 / max over 1946 tiles:
    A1 -1.094692 / -0.506761 / -0.188583 / -0.051834 / -0.006885
    A2  0.170564 /  0.484315 /  1.000481 /  2.092122 /  9.227945
    A3  0.015625 /  0.015625 /  0.062500 /  0.382813 /  0.796875
    A4 -0.013417 / -0.003611 /  0.003752 /  0.019741 /  0.038608
  Extreme outliers, tile level, BOTH rules (threshold 10 on each; report only, nothing clamped,
  transformed, winsorised or dropped, and the artifact bytes equal the numbers tile_spatial gave):
    key   MAD        |x-med|/MAD > 10   max |z|   log rule      > 10   max log |z|   distinct
    A1    0.0851785         1            10.638   abs, on          0        6.897      1946
    A2    0.1997090        27            41.197   value, on         2       11.013      1946
    A3    0.0468750        38            15.667   value, on         0        3.672         88
    A4    0.0042331         0             8.234   straddles 0     n/a        n/a       1946
  The log rule is what makes the lower tail visible: A2's near-zero tiles reach 11.0 in
  multiplicative units and are flagged, where the absolute rule's cap below the median is about 5;
  A3's lower tail - the 297 tiles tied at its minimum - reaches only 3.67 log units, so the answer
  is that they are NOT extreme, and that answer is now measurable instead of structurally unaskable.
  No key reaches 1e6 in magnitude: max |A1| 1.0947, |A2| 9.2279, |A3| 0.7969, |A4| 0.0386, so the
  A2 = 1.2e+13 one synthetic tile produced before AMENDMENT 3 has no analogue on the corpus.
  At soil level (the matrix a rank-3 fit would standardise): A3 has 1 soil beyond 10 MAD
  (z 18.08, the 0.46875 soil) and nothing beyond 10 in log units (max 4.07); no other key flags.
  Determinism - tile / image / soil vector hashes, identical across two in-process passes and one
  fresh interpreter (and equal to the hashes the pre-fix Task 3 run published, which is how "an
  audit-layer change moved no measurement" is demonstrated rather than asserted):
    84e2b2c23e100b95e1b2b15e97933b81bde808b86602908676f002b97c54399e  1946 tiles
    778990e14303980a83d688ae739efb5a57fe548d2fdc1b72a9f9718ed87184b0  162 images
    41dbbd7e729cee22183cacfcc6409eaa5d6037a390e7afe41265a5df75df3eb6  34 soils
  sha256 of the four published files:
    tile_census.csv        a9ca9e0103c13086e1d2d94ba87051ce3590755cb4e86dc818b9813b7c4299cb
    census.json            17d2c66c225ed84a64ad2193ed1b117ffc19ba78d4b1fc47823473d33ffdd4a1
    image_features_m7a.csv b04d3bb7e80966a350076098793f1007d1724fd622e5a527ce659eeee2bf1d9b
    soil_features_m7a.csv  72bb91898d435dec27446f9ab54d8f45201218bba8006560dabde7360cc947b7
  Only tile_census.csv carries an expected-hash constant (FIRST_CSV_SHA, the pre-fix run's bytes).
  The two design frames are re-proved rather than pinned: read back against memory bit for bit, and
  re-published into a probe directory and byte-compared. They have no pre-fix baseline at all, since
  publishing them is what M8 added. census.json is a report, so its bytes move whenever a report
  field is added.

Cost: ~97 ms per 256 px tile single-threaded, ~190 s for the 1946 tiles, so this file runs the
instrumented pass twice in-process and once in a background subprocess (started in main() so the
fresh interpreter overlaps the in-process passes) - about 7 minutes wall clock, 324 checks.
Run a subset by passing name fragments: python check_m7a_data.py outlier_rule warning_sink.
A filtered run starts no fresh interpreter and prints no summary.

THINGS THIS FILE FOUND, AND HOW EACH WAS RESOLVED (all six were audit-layer; none changed a
measurement, a gate, a threshold, a control or an A1-A4 definition):
  I1 the missing-tile abort named the PREVIOUS tile, because ctx["current"] was set after the
     exists() check. Fixed by naming the tile before the file is touched; the test now uses two
     rows with the second deleted and asserts the published tile_path and row.
  I2 the outlier rule was mathematically one-sided for a positive key. Fixed by adding a second
     rule on log|value| with the same frozen threshold, reporting a zero MAD as its own condition
     instead of a count of zero, and refusing the log rule for a key that straddles zero.
  I3 the warning sink covered only the tile loop, while sf._median_column warns at aggregation, so
     "warnings: 0" was a claim about part of the run. Fixed by installing the same sink around both
     aggregations, tallying by stage, and arming each stage with a forced warning. The schema leg
     that checked an empty list is gone. The call site is pinned too, and pinned by mutation: with
     ctx=ctx deleted from assembly(), test_assembly_itself_instruments_both_aggregation_stages fails
     4 legs and the injected RuntimeWarning leaks to stderr; deleting only one of the two call sites
     fails 3 legs and leaks that stage's warning. The line was restored and m7a_data.py's sha256
     verified against scratch/protect.sha, which covers that one file: it is the file that was
     mutated. It is a POINT-IN-TIME record of that restore, not a live manifest: m7a_data.py has been
     edited again since (the round-2 frame-guard fix), so scratch/bytes_under_test.sha is the file
     that verifies against the current bytes, and the hash the restore produced was
     36c86672d3da555a2f86de1a0bd9f8d0e66becd22d1a92ed68beeedd9598f95f. Note that this leg reads
     state(), so a filtered run naming it is not cheap.
  I4 a key tile_spatial left NaN while its own definition measures it was TALLIED as degeneracy
     even though the code said in words it was a bug. Fixed: _key_reason returns the disagreement as
     data, classify_tile raises ValueError, and the pass aborts as outcome (1).
  I5 no check asserted tile-level finiteness, and the strict-JSON leg could not fail because the
     writer rewrote non-finite values to null. Fixed: counts are taken off the numpy arrays before
     serialisation, publish() refuses a non-finite aggregate before creating the output directory,
     _json_safe refuses to launder one, and a planted NaN/Infinity is now proved to be caught.
  I6 six checks read m7a_data's source text and one measurably shaped it (the module docstring
     recorded routing percentiles through np.percentile specifically so a ban on the token
     ".median(" would not fire). All six are gone, replaced by the recompute-and-compare legs plus
     spies: which block functions get called, which files pandas is asked to open, which model
     entry points fire, and whether the statistics namespace grows.
  M2 non_finite meant "tiles with any NaN key" in one place and "count of infs" in another. One
     meaning now (NaN or +-inf), with the unit in the name: *_tiles versus *_cells.
  M8 the assembled design existed only in memory. Both frames are published and round-tripped.
  Earlier: read_census_csv() parses with float_precision="round_trip" because pandas' default parser
  is not correctly rounded (it moved A2 by 1.78e-15 on re-read); and the strict-JSON check first
  used caught(), which returns the exception and not the value, so a clean parse read as None.
"""
import ast
import hashlib
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Model 5" / "Model 5 Experiment 1"))
import m7a_data as d  # noqa: E402
import spatial_features as sf  # noqa: E402
import transfer_eval as tv  # noqa: E402

FAIL = []
TOTAL = 0
PPM = 4.552516
ART_DIR = HERE / "data"
CSV_PATH = ART_DIR / "tile_census.csv"
JSON_PATH = ART_DIR / "census.json"
IMG_CSV = ART_DIR / "image_features_m7a.csv"
SOIL_CSV = ART_DIR / "soil_features_m7a.csv"
EXPECTED_ARTIFACTS = ("census.json", "image_features_m7a.csv", "soil_features_m7a.csv",
                      "tile_census.csv")
DROPPED = ("camera", "camera_fam", "cv_group", "source_effective_ppm", "source_image_width",
           "source_image_height")
CAMERA_TOKENS = ("camera", "cam", "device", "make", "model", "exif", "icc", "phone")
# the label, fold and fitted-model files the pipeline owns. Read by tests as PATHS THAT MAY NOT be
# opened, never as words to hunt for in m7a_data's source (I6: the token scan could not tell a
# forbidden call from a sentence about one, and it shaped the code to dodge it).
FORBIDDEN_FILES = ("labels", "sample_submission", "cv_families", "cv_per_soil", "curve",
                   "target", "submission")
FORBIDDEN_CALLS = ("nested_predict", "fit_predict", "oracle_sweep", "no_holdout_control_score",
                   "soil_from_images", "labels_matrix", "project", "emd_pair")
REQUIRED_JSON = ("total_tiles", "successful_tiles", "tiles_with_any_nan", "degenerate_tiles",
                 "invalid_calls", "per_key", "warnings_count", "errors_count", "extreme_outlier",
                 "finiteness", "non_finite_tiles", "warnings_by_stage")
REQUIRED_PER_KEY = ("nan_count", "inf_count", "nonfinite_count", "min", "p5", "median", "p95",
                    "max")
MAD_Z = 10.0
# tile_census.csv as the standalone census run left it, measured before this file rewrote it.
# Pinned because "the same numbers twice" is a weaker claim than "the same BYTES twice".
FIRST_CSV_SHA = "a9ca9e0103c13086e1d2d94ba87051ce3590755cb4e86dc818b9813b7c4299cb"
STATE = {}          # filled by the first expensive test; one instrumented pass serves the file
FRESH = None        # the fresh-interpreter determinism subprocess, started in main()


def check(name, ok, detail=""):
    global TOTAL
    TOTAL += 1
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def caught(fn, *args, **kw):
    """The exception a call raises, or None. Checks assert on the TYPE: the ruling's whole point is
    that TileDegenerate and ValueError are different facts about a tile."""
    try:
        fn(*args, **kw)
    except BaseException as exc:
        return exc
    return None


def state():
    """The one full instrumented pass this file needs, cached. Determinism does not read this cache:
    test_determinism clears it and recomputes, which is the only way the claim can mean anything."""
    if not STATE:
        STATE["data_fp_before"] = d.data_fingerprint(ROOT)
        STATE["asm"] = d.assembly(ROOT)
        STATE["census"] = d.run_census(ROOT, write=True)
    return STATE


def strict_json(path):
    """Load census.json rejecting NaN/Infinity, so a non-strict artifact cannot pass as JSON."""
    def bad(x):
        raise ValueError("non-strict JSON constant in the artifact: %s" % x)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=bad)


def num(x, fmt="%.6g"):
    """Format a reported number, including the None a key can carry when its MAD is zero."""
    return "n/a" if x is None else fmt % x


# ------------------------------------------------------------------ the frozen filter
def test_tile_selection_is_the_frozen_rule():
    t = d.tile_table(ROOT)
    check("only 256 px tiles", bool((t.tile_size_px == 256).all()))
    check("only materialized tiles", bool(t.materialized.astype(bool).all()))
    check("only soil_fraction >= 0.50 tiles", bool((t.soil_fraction >= 0.50).all()))
    check("1946 qualifying tiles, as measured for the pre-registration",
          len(t) == 1946, "%d" % len(t))
    raw = pd.read_csv(ROOT / "data" / "processed_meta" / "manifest_tiles.csv")
    ref = raw[(raw.tile_size_px == tv.TILE_PX) & raw.materialized
              & (raw.soil_fraction >= tv.MIN_SOIL_FRACTION)]
    check("tile set is the frozen loader's set, row for row",
          list(t.tile_path) == list(ref.tile_path), "%d vs %d" % (len(t), len(ref)))
    check("tv.TILE_PX and tv.MIN_SOIL_FRACTION are the filter's constants, not copies",
          tv.TILE_PX == 256 and abs(tv.MIN_SOIL_FRACTION - 0.50) < 1e-12)
    vc = t.split.value_counts().to_dict()
    check("1541 train / 405 test tiles, as pre-registered", vc == {"train": 1541, "test": 405},
          str(vc))


def test_camera_and_source_columns_are_dropped_at_load():
    t = d.tile_table(ROOT)
    for c in DROPPED:
        check("dropped at load: %s" % c, c not in t.columns, str(list(t.columns)))
    check("normalized_ppm is carried as the physical scale argument",
          "normalized_ppm" in t.columns and bool(np.isfinite(t.normalized_ppm).all()))
    check("every tile reads at one recorded ppm, so the scale is auditable",
          t.normalized_ppm.round(6).nunique() == 1, str(sorted(t.normalized_ppm.unique())))


# ------------------------------------------------------------------ the three outcomes
def test_invalid_call_aborts_and_is_never_counted_as_degeneracy():
    """Outcome (1): a ValueError must abort, name its tile, and stay out of the degeneracy count.

    Forced the honest way: a real tile read at an impossible recorded scale, which is exactly the
    ppm <= 0 case spatial_features calls an invalid CALL rather than a degenerate tile.
    """
    q = d.tile_table(ROOT).head(1).copy()
    q["normalized_ppm"] = 0.0
    ctx = {"records": [], "warnings": [], "errors": [], "current": None}
    err = caught(d.scan_tiles, ROOT, q, ctx=ctx)
    check("an invalid call raises instead of being logged", isinstance(err, ValueError),
          type(err).__name__ if err is not None else "returned normally")
    check("the abort names the tile that caused it",
          bool(ctx["errors"]) and ctx["errors"][0]["tile_path"] == q.tile_path.iloc[0],
          str(ctx["errors"][:1])[:120])
    check("the invalid call is recorded with its reason string",
          bool(ctx["errors"]) and "ppm" in ctx["errors"][0]["reason"],
          str(ctx["errors"][:1])[:160])
    check("the invalid call is recorded as a ValueError, not as a degeneracy kind",
          bool(ctx["errors"]) and ctx["errors"][0]["kind"] == "ValueError",
          str(ctx["errors"][:1])[:140])
    check("an invalid call is NEVER counted as a degenerate tile",
          not any(r["status"] == d.STATUS_DEGENERATE for r in ctx["records"]),
          str([r["status"] for r in ctx["records"]]))
    check("the aborted tile produced no statistic row: it was not skipped as if it were fine",
          len(ctx["records"]) == 0, str(len(ctx["records"])))
    check("the in-flight tile is known at the moment of the abort",
          ctx["current"] is not None and ctx["current"]["tile_path"] == q.tile_path.iloc[0],
          str(ctx["current"])[:120])


def test_missing_tile_path_is_an_invalid_call_not_a_skip():
    """I1: the error must name the tile that is MISSING, not the one that ran before it.

    Two rows, the second pointing at a file that does not exist. The structured tile_path and row
    fields are what the artifact publishes, so an error record naming the previous tile is a wrong
    accusation in a report, even though the free-text reason names the right file.
    """
    q = d.tile_table(ROOT).head(2).copy()
    q.loc[q.index[1], "tile_path"] = "data/tiles/does/not/exist__t256.png"
    ctx = d.new_ctx()
    err = caught(d.scan_tiles, ROOT, q, ctx=ctx)
    check("a manifest pointing at a missing tile aborts", isinstance(err, AssertionError),
          type(err).__name__ if err is not None else "returned normally")
    check("the missing tile is recorded as an error, not as degeneracy",
          bool(ctx["errors"]) and ctx["errors"][0]["kind"] == "missing_tile",
          str(ctx["errors"][:1])[:140])
    check("the error names the missing tile, not the tile that ran before it",
          len(ctx["errors"]) == 1 and ctx["errors"][0]["tile_path"] == q.tile_path.iloc[1],
          "recorded %s, missing is %s" % (str(ctx["errors"][:1])[:60], q.tile_path.iloc[1][:60]))
    check("the error carries the missing row index",
          len(ctx["errors"]) == 1 and ctx["errors"][0]["row"] == int(q.index[1]),
          str(ctx["errors"][:1])[:120])
    check("the free-text reason also names the path that is really missing",
          len(ctx["errors"]) == 1 and q.tile_path.iloc[1] in ctx["errors"][0]["reason"],
          str(ctx["errors"][:1])[:160])
    check("no record was written for the missing tile",
          len(ctx["records"]) == 1 and ctx["records"][0]["tile_path"] == str(q.tile_path.iloc[0]),
          str([r["tile_path"][:40] for r in ctx["records"]]))
    check("the tile before the missing one is still counted as a finished tile",
          len(ctx["records"]) == 1 and ctx["records"][0]["status"] == d.STATUS_OK,
          str([r["status"] for r in ctx["records"]]))


def test_a_missing_tile_is_named_even_when_it_is_the_first_row():
    """The attribution must not depend on a previous tile existing."""
    q = d.tile_table(ROOT).head(2).copy()
    q.loc[q.index[0], "tile_path"] = "data/tiles/does/not/exist__first.png"
    ctx = d.new_ctx()
    err = caught(d.scan_tiles, ROOT, q, ctx=ctx)
    check("the very first row can be the one that aborts", isinstance(err, AssertionError))
    check("an abort on the first row still names that row",
          len(ctx["errors"]) == 1 and ctx["errors"][0]["tile_path"] == q.tile_path.iloc[0]
          and ctx["errors"][0]["row"] == int(q.index[0]), str(ctx["errors"][:1])[:160])
    check("the first-row abort produced no records", ctx["records"] == [], str(ctx["records"]))


def test_degenerate_tile_is_outcome_two_not_one():
    """A genuinely degenerate TILE: TileDegenerate, all four keys NaN, reason captured."""
    a = np.full((64, 64, 3), 120.0)
    err = caught(sf.tile_spatial, a, PPM)
    check("the constant tile raises TileDegenerate (a tile fact, not a call fact)",
          isinstance(err, sf.TileDegenerate), type(err).__name__)
    row = caught(d.compute_tile_row, a, PPM)
    check("compute_tile_row lets a whole-tile degeneracy reach the caller as TileDegenerate",
          isinstance(row, sf.TileDegenerate), type(row).__name__)
    rec = d.classify_tile(a, PPM)
    check("a whole-tile degeneracy is status degenerate_tile",
          rec["status"] == d.STATUS_DEGENERATE, rec["status"])
    check("a degenerate tile is NaN in all four keys", rec["nan_keys"] == list(sf.SPATIAL),
          str(rec["nan_keys"]))
    check("the degeneracy reason string is captured", "degenerate mask" in rec["reason"],
          rec["reason"][:90])


def test_nan_key_is_outcome_two_with_a_per_key_reason():
    """A valid tile whose A2/A3/A4 are undefined: counted, keyed, reasoned; A1 stays finite."""
    rng = np.random.default_rng(0)
    a = np.full((64, 64, 3), 10.0)
    a[20:25, 20:25] = rng.uniform(100, 250, size=(5, 5, 3))
    rec = d.classify_tile(a, PPM)
    check("a valid tile with undefined statistics is status nan_keys",
          rec["status"] == d.STATUS_NAN_KEYS, rec["status"])
    check("the NaN keys are recorded individually", rec["nan_keys"] == ["A2", "A3", "A4"],
          str(rec["nan_keys"]))
    check("A1 stays finite on a nan_keys tile", np.isfinite(rec["values"]["A1"]),
          "%.6f" % rec["values"]["A1"])
    check("each NaN key carries its own reason", rec["reason"].count("A2:") == 1
          and rec["reason"].count("A3:") == 1 and rec["reason"].count("A4:") == 1,
          rec["reason"][:100])
    check("a nan_keys tile is never called a degenerate tile",
          rec["status"] != d.STATUS_DEGENERATE)


def test_wrong_shape_is_an_invalid_call():
    rec = caught(d.classify_tile, np.full((64, 64), 120.0), PPM)
    check("a 2-D array is an invalid call, not a degeneracy", isinstance(rec, ValueError),
          type(rec).__name__)


# ------------------------------------------------------------------ I4: disagreement is outcome (1)
def _nan_a2_stand_in(real):
    """A tile_spatial that writes NaN into A2 whenever the real one measured it."""
    def fake(a, ppm):
        out = dict(real(a, ppm))
        if np.isfinite(out["A2"]):
            out["A2"] = float("nan")
        return out
    return fake


def _blob_tile():
    """A valid tile whose only texture is a 5x5 patch: too few lines for A2/A3/A4, fine for A1."""
    rng = np.random.default_rng(0)
    a = np.full((64, 64, 3), 10.0)
    a[20:25, 20:25] = rng.uniform(100, 250, size=(5, 5, 3))
    return a


def test_the_probe_flags_a_key_that_has_no_business_being_nan():
    """I4: a NaN key whose own registered definition returns a NUMBER is a disagreement.

    _key_reason used to put that fact in a sentence and let the caller count the tile as degenerate
    anyway. It now reports the disagreement as data, which is the only form the caller can act on.
    """
    q = d.tile_table(ROOT).head(1)
    a = d._read_tile(ROOT / q.tile_path.iloc[0])
    ppm = float(q.normalized_ppm.iloc[0])
    real = sf.tile_spatial(a, ppm)
    check("the tile used here really does have a measurable A2", np.isfinite(real["A2"]),
          "%.6f" % real["A2"])
    out = d._key_reason(a, ppm, "A2")
    check("_key_reason answers a disagreement as a (reason, disagreed) pair",
          isinstance(out, tuple) and len(out) == 2, "%s: %s" % (type(out).__name__, str(out)[:90]))
    if isinstance(out, tuple) and len(out) == 2:
        reason, disagreed = out
        check("a key the definition can measure is reported as a disagreement",
              disagreed is True, str(reason)[:120])
        check("the disagreement reason quotes the number the probe returned",
              any(c.isdigit() for c in str(reason)), str(reason)[:140])
    a1 = d._key_reason(_blob_tile(), PPM, "A1")
    check("_key_reason keeps its (reason, disagreed) shape for A1 as well",
          isinstance(a1, tuple) and len(a1) == 2, "%s: %s" % (type(a1).__name__, str(a1)[:90]))
    if isinstance(a1, tuple) and len(a1) == 2:
        check("A1 has no undefined branch, so a NaN A1 is itself the disagreement",
              a1[1] is True, str(a1[0])[:120])


def test_a_real_degeneracy_is_not_reclassified_as_a_disagreement():
    """I4 must not turn outcome (2) into outcome (1): the probes have to agree with tile_spatial."""
    a, ppm = _blob_tile(), PPM
    out = sf.tile_spatial(a, ppm)
    check("the blob tile is a valid tile whose keys are undefined",
          all(not np.isfinite(out[k]) for k in ("A2", "A3", "A4")), str(out))
    for k in ("A2", "A3", "A4"):
        reason, disagreed = d._key_reason(a, ppm, k)
        check("%s is NaN because its own definition refuses it, not because of a bug" % k,
              disagreed is False, str(reason)[:120])
    rec = d.classify_tile(a, ppm)
    check("a genuine per-key degeneracy is still counted as nan_keys",
          rec["status"] == d.STATUS_NAN_KEYS, rec["status"])


def test_a_fabricated_nan_aborts_instead_of_being_counted():
    """End to end: tile_spatial writing NaN where it should not must abort the whole pass.

    Fault injected the only way it can be injected - a stand-in tile_spatial that NaNs a measurable
    key. The real module never reaches this state on this corpus (the census measured 0 nan_keys),
    which is exactly why the behaviour has to be forced to be observed.
    """
    q = d.tile_table(ROOT).head(2).copy()
    real = sf.tile_spatial
    try:
        sf.tile_spatial = _nan_a2_stand_in(real)
        ctx = d.new_ctx()
        err = caught(d.scan_tiles, ROOT, q, ctx=ctx)
    finally:
        sf.tile_spatial = real
    check("a fabricated NaN raises instead of being tallied", isinstance(err, ValueError),
          type(err).__name__ if err is not None else "returned normally")
    check("the fabricated NaN aborts the pass", ctx["aborted"] is True and err is not None)
    check("the abort is recorded as an invalid call, not as a degeneracy",
          len(ctx["errors"]) == 1 and ctx["errors"][0]["kind"] == "ValueError"
          and ctx["errors"][0]["stage"] == "tiles",
          str(ctx["errors"][:1])[:160])
    check("the error names the tile that triggered the disagreement",
          len(ctx["errors"]) == 1 and ctx["errors"][0]["tile_path"] == q.tile_path.iloc[0],
          str(ctx["errors"][:1])[:160])
    check("the offending key is named in the recorded reason",
          len(ctx["errors"]) == 1 and "A2" in ctx["errors"][0]["reason"],
          str(ctx["errors"][:1])[:180])
    check("a fabricated NaN never becomes a degenerate or nan_keys row",
          not ctx["records"] or all(r["status"] != d.STATUS_NAN_KEYS for r in ctx["records"]),
          str([r["status"] for r in ctx["records"]]))
    # report_from is called on the aborted context on purpose: an aborted pass must still be able to
    # say what it counted, and what it must say is that the failure was an invalid call, not a tile
    after = d.report_from(ctx)
    check("the report surface of an aborted pass counts the disagreement as an invalid call",
          after["invalid_calls"] == 1 and after["degenerate_mask"] == 0
          and after["per_key_nan"] == {k: 0 for k in sf.SPATIAL},
          str({k: after[k] for k in ("invalid_calls", "degenerate_mask", "non_finite")}))


# ------------------------------------------------------------------ the assembly (one pass)
def test_image_table_row_count_matches_the_frozen_pipeline():
    asm = state()["asm"]
    img = asm["image"]
    ref = tv.image_features(ROOT)
    check("same images as the frozen 12-feature assembly",
          sorted(img.index.astype(str)) == sorted(ref.index.astype(str)),
          "%d vs %d" % (len(img), len(ref)))
    check("columns are A1-A4 plus sample_id only",
          list(img.columns) == list(sf.SPATIAL) + ["sample_id"])
    check("no camera, ppm or sample-derived column is carried into the design",
          not any(c in img.columns for c in ("camera", "camera_fam", "normalized_ppm", "tile_x",
                                             "tile_y", "grid_col", "grid_row")))
    check("no NaN in any image-level statistic", not img[list(sf.SPATIAL)].isna().to_numpy().any())
    check("the image index is the parent image path", img.index.name == "parent_image_path")
    check("one image maps to one soil (no silent overwrite in the assembly)",
          img.groupby("sample_id").size().min() >= 1
          and asm["q"].groupby("parent_image_path").sample_id.nunique().max() == 1)


def test_soil_table_is_the_two_step_median_path():
    asm = state()["asm"]
    img, soil = asm["image"], asm["soil"]
    check("soil columns are exactly A1-A4", list(soil.columns) == list(sf.SPATIAL))
    check("soil index is sample_id", soil.index.name == "sample_id")
    check("34 soils in the corpus, 24 of them train", len(soil) == 34, str(len(soil)))
    tr = sorted(asm["q"][asm["q"].split == "train"].sample_id.unique())
    check("24 train soils and no test soil leaks into that set",
          len(tr) == 24 and set(tr) <= set(soil.index.astype(str)), str(len(tr)))
    # behaviour, not source: the tested block functions must reproduce the frames exactly
    rebuilt = {}
    for sid, grp in img.groupby("sample_id"):
        rebuilt[sid] = sf.soil_block([r for r in grp.to_dict("records")])
    rb = pd.DataFrame.from_dict(rebuilt, orient="index").loc[soil.index, list(sf.SPATIAL)]
    check("soil frame equals sf.soil_block applied to the image rows, bit for bit",
          np.array_equal(rb.to_numpy(float), soil.to_numpy(float), equal_nan=True),
          "max diff %.3g" % np.nanmax(np.abs(rb.to_numpy(float) - soil.to_numpy(float))))
    rebuilt_i = {}
    for (p, sid), grp in asm["q"].groupby(["parent_image_path", "sample_id"]):
        rebuilt_i[p] = dict(sf.image_block(list(grp._row)), sample_id=str(sid))
    ri = pd.DataFrame.from_dict(rebuilt_i, orient="index").loc[img.index, list(sf.SPATIAL)]
    check("image frame equals sf.image_block applied to the tile rows, bit for bit",
          np.array_equal(ri.to_numpy(float), img[list(sf.SPATIAL)].to_numpy(float),
                         equal_nan=True))


# ------------------------------------------------------------------ I6: the pins are behaviour
def test_the_assembler_routes_through_the_tested_blocks_not_a_substitute():
    """The recompute legs above pass for the wrong reason if the assembler never CALLS the blocks.

    A source scan used to assert that, and it shaped the implementation's wording instead of its
    behaviour. These legs watch the calls and the consequences: a spy proves the tested block is on
    the path, and a value the tested block refuses but a pandas median would swallow proves the
    refusal is the one the assembly inherits.
    """
    calls = {"image_block": 0, "soil_block": 0}
    real_i, real_s = sf.image_block, sf.soil_block

    def spy_i(rows):
        calls["image_block"] += 1
        return real_i(rows)

    def spy_s(rows):
        calls["soil_block"] += 1
        return real_s(rows)
    q = _tiny_q([{"A1": -0.2, "A2": 1.0, "A3": 0.05, "A4": 0.001},
                 {"A1": -0.3, "A2": 1.5, "A3": 0.06, "A4": -0.002}],
                ["i1.png", "i2.png"], ["S1", "S2"])
    img = pd.DataFrame([{"A1": -0.2, "A2": 1.0, "A3": 0.05, "A4": 0.001, "sample_id": "S1"},
                        {"A1": -0.3, "A2": 1.5, "A3": 0.06, "A4": -0.002, "sample_id": "S1"}],
                       index=pd.Index(["i1.png", "i2.png"], name="parent_image_path"))
    try:
        sf.image_block, sf.soil_block = spy_i, spy_s
        d.image_table(q)
        d.soil_table(img)
    finally:
        sf.image_block, sf.soil_block = real_i, real_s
    check("image_table calls the tested image block once per image",
          calls["image_block"] == 2, str(calls))
    check("soil_table calls the tested soil block once per soil",
          calls["soil_block"] == 1, str(calls))
    poison = _tiny_q([{"A1": -0.2, "A2": float("inf"), "A3": 0.05, "A4": 0.001},
                      {"A1": -0.3, "A2": 1.5, "A3": 0.06, "A4": -0.002}],
                     ["i1.png", "i1.png"], ["S1", "S1"])
    err = caught(d.image_table, poison)
    check("an inf tile value aborts the aggregation instead of becoming an inf row",
          isinstance(err, ValueError), type(err).__name__ if err is not None else "returned")
    rows = [{"A1": -0.2, "A2": float("nan"), "A3": float("nan"), "A4": float("nan")},
            {"A1": -0.3, "A2": float("nan"), "A3": float("nan"), "A4": float("nan")}]
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        dead = d.image_table(_tiny_q(rows, ["i1.png", "i1.png"], ["S1", "S1"]))
    check("an all-NaN key stays NaN after aggregation: no pandas fallback number appears",
          np.isnan(dead["A2"].iloc[0]) and np.isfinite(dead["A1"].iloc[0]),
          str(dead.to_dict("records"))[:140])
    check("the dead columns announce themselves, one warning per dead key, from the tested module",
          len(got) == 3 and {w.category.__name__ for w in got} == {"RuntimeWarning"}
          and all(Path(str(w.filename)).name == "spatial_features.py" for w in got),
          str([(w.category.__name__, Path(str(w.filename)).name) for w in got])[:160])


def test_a_systemic_error_is_never_swallowed_as_degeneracy():
    """No broad except: a RuntimeError from the statistics layer must reach the caller.

    The behaviour the old "except Exception" text scan was standing in for, and it can fail: an
    implementation that caught broadly would return records here instead of raising.
    """
    q = d.tile_table(ROOT).head(2).copy()
    real = sf.tile_spatial
    try:
        sf.tile_spatial = _raising_spatial(real, RuntimeError("systemic: an internal assert broke"))
        ctx = d.new_ctx()
        err = caught(d.scan_tiles, ROOT, q, ctx=ctx)
    finally:
        sf.tile_spatial = real
    check("a systemic error propagates out of the pass", isinstance(err, RuntimeError),
          type(err).__name__ if err is not None else "returned normally")
    check("a systemic error is recorded as an invalid call, never as a degenerate tile",
          len(ctx["errors"]) == 1 and ctx["errors"][0]["kind"] == "RuntimeError"
          and not any(r["status"] == d.STATUS_DEGENERATE for r in ctx["records"]),
          str(ctx["errors"][:1])[:160])
    check("a systemic error produces no statistic row for the tile that raised",
          len(ctx["records"]) == 0, str(len(ctx["records"])))
    check("the pass reports itself as aborted", ctx["aborted"] is True)


def _raising_spatial(real, exc):
    def fake(a, ppm):
        raise exc
    return fake


def test_the_assembler_opens_no_label_curve_or_fold_file():
    """I6 replaces a token scan with the thing that actually matters: which files are opened.

    Scanning m7a_data for the WORDS "labels_matrix" or "Ridge" cannot tell a forbidden call from a
    sentence about one, and it made the module's wording a test target. Recording the paths pandas
    is asked to read during a real pass tests the boundary itself.
    """
    read = []
    real_read = pd.read_csv

    def spy_read(*args, **kw):
        if args:
            read.append(str(args[0]))
        return real_read(*args, **kw)
    pd.read_csv = spy_read
    try:
        q = d.tile_table(ROOT).head(2)
        records = d.scan_tiles(ROOT, q, ctx=d.new_ctx())["records"]
        d.image_table(q.assign(_row=[r["values"] for r in records]))
    finally:
        pd.read_csv = real_read
    names = [p.replace("\\", "/").split("/")[-1] for p in read]
    hits = sorted({n for n in names for f in FORBIDDEN_FILES if f in n.lower()})
    check("the assembly opens no label, curve, fold or submission file", not hits, str(names))
    check("the only manifest it reads is the tile manifest",
          all("manifest_tiles" in n for n in names), str(names))
    check("the watched pass really did read something (the spy is armed)", bool(names), str(names))


def test_the_assembler_calls_no_fitted_model_entry_point():
    """The model's own functions are patched to explode: Task 3 must never reach them."""
    real = {n: getattr(tv, n) for n in FORBIDDEN_CALLS if hasattr(tv, n)}
    called = []

    def boom(name, fn):
        def f(*a, **k):
            called.append(name)
            raise AssertionError("%s must not be called by the assembly" % name)
        return f
    for n, fn in real.items():
        setattr(tv, n, boom(n, fn))
    try:
        q = d.tile_table(ROOT).head(2)
        d.scan_tiles(ROOT, q, ctx=d.new_ctx())
        err = None
    except Exception as exc:
        err = exc
    finally:
        for n, fn in real.items():
            setattr(tv, n, fn)
    check("no ridge, projection, EMD or nested-CV entry point fired", err is None and not called,
          str(err or called)[:160])
    check("the patch list was not empty (the leg is armed on real symbols)", len(real) >= 6,
          str(sorted(real)))


def test_assembly_leaves_the_statistics_namespace_and_imports_alone():
    """I6: the import boundary is structural; the namespace ban is watched, not scanned."""
    before = set(dir(sf))
    src = (HERE / "m7a_data.py").read_text(encoding="utf-8")
    imports = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    allowed = {"__future__", "sys", "os", "json", "hashlib", "warnings", "contextlib", "pathlib",
               "typing", "numpy", "pandas", "PIL", "spatial_features", "transfer_eval"}
    check("m7a_data imports only manifest/frame handling, the two local modules and the stdlib",
          imports <= allowed, str(sorted(imports - allowed)))
    q = d.tile_table(ROOT).head(1)
    d.scan_tiles(ROOT, q, ctx=d.new_ctx())
    check("m7a_data adds nothing to the spatial_features namespace", set(dir(sf)) == before,
          str(sorted(set(dir(sf)) - before)))
    check("no sklearn, torch or model library is imported anywhere in the module",
          not imports & {"sklearn", "torch", "scipy", "cv2", "tensorflow"}, str(sorted(imports)))


def test_schema_and_dimensionality():
    asm = state()["asm"]
    img, soil, tf = asm["image"], asm["soil"], asm["tile_frame"]
    check("image table is 162 rows x 5 columns (A1-A4 + sample_id)",
          img.shape == (162, 5), str(img.shape))
    check("soil table is 34 rows x 4 columns", soil.shape == (34, 4), str(soil.shape))
    check("tile census frame is 1946 rows x 4 statistics", len(tf) == 1946
          and all(k in tf.columns for k in sf.SPATIAL), str(tf.shape))
    check("every tile carries all four keys",
          not any(r["values"].keys() != set(sf.SPATIAL) for r in asm["records"]))
    check("aggregation is image_block then soil_block over the same rows the census counted",
          int(asm["q"].parent_image_path.nunique()) == len(img)
          and img.groupby("sample_id").size().sum() == len(img))
    cells = np.vstack([img[list(sf.SPATIAL)].to_numpy(float), soil.to_numpy(float)])
    check("no assembled image or soil cell is NaN, inf or -inf",
          bool(np.isfinite(cells).all()), "non-finite cells: %d" % int(np.count_nonzero(
              ~np.isfinite(cells))))
    vals = asm["tile_frame"][list(sf.SPATIAL)].to_numpy(float)
    reasoned = np.array([bool(r["reason"]) for r in asm["records"]])[:, None]
    check("every NaN cell in the tile matrix sits in a row with a recorded reason",
          bool((reasoned | ~np.isnan(vals)).all()),
          "NaN cells: %d" % int(np.count_nonzero(np.isnan(vals))))
    check("every non-ok tile record carries a reason string",
          all(bool(r["reason"]) for r in asm["records"] if r["status"] != d.STATUS_OK))


def test_no_camera_derived_column_survives_anywhere():
    asm = state()["asm"]
    names = ([str(c) for c in asm["q"].columns] + [str(c) for c in asm["image"].columns]
             + [str(c) for c in asm["soil"].columns] + [str(asm["image"].index.name),
                                                        str(asm["soil"].index.name)]
             + [str(c) for c in d.read_census_csv(CSV_PATH).columns])
    hits = sorted({n for n in names for t in CAMERA_TOKENS if t in n.lower()})
    check("no camera-derived column name anywhere in the assembly or the artifacts",
          not hits, str(hits))
    check("no tile-position column reaches the design",
          not any(c in asm["image"].columns or c in asm["soil"].columns
                  for c in ("tile_x", "tile_y", "grid_col", "grid_row", "normalized_ppm")))


# ------------------------------------------------------------------ the census artifacts
def test_census_artifacts_exist_and_are_strict_json():
    state()
    check("data/tile_census.csv exists", CSV_PATH.exists(), str(CSV_PATH))
    check("data/census.json exists", JSON_PATH.exists(), str(JSON_PATH))
    err = caught(strict_json, JSON_PATH)
    check("census.json is strict JSON (no NaN/Infinity tokens)", err is None,
          "" if err is None else str(err)[:120])
    csv = d.read_census_csv(CSV_PATH)
    check("tile_census.csv has one row per materialised tile", len(csv) == 1946, str(len(csv)))
    check("tile_census.csv tile paths are unique", csv.tile_path.nunique() == len(csv))
    need = (["tile_path", "sample_id", "split", "normalized_ppm", "status", "reason"]
            + list(sf.SPATIAL) + ["%s_nan" % k for k in sf.SPATIAL])
    check("tile_census.csv carries the required columns", all(c in csv.columns for c in need),
          str([c for c in need if c not in csv.columns]))


def test_census_json_has_every_required_field():
    j = strict_json(JSON_PATH)
    for f in REQUIRED_JSON:
        check("census.json field: %s" % f, f in j, str(sorted(j.keys())))
    for k in sf.SPATIAL:
        stats = j["per_key"].get(k, {})
        check("per_key %s has %s" % (k, list(REQUIRED_PER_KEY)),
              all(f in stats for f in REQUIRED_PER_KEY), str(sorted(stats.keys())))
    check("total_tiles is the frozen 1946", j["total_tiles"] == 1946, str(j["total_tiles"]))
    for block in ("image_percentiles", "soil_percentiles"):
        check("census.json also reports the assembled level: %s" % block,
              all(k in j.get(block, {}) for k in sf.SPATIAL), str(sorted(j.get(block, {}))))
    check("the artifact records the schema it assembled",
          j["columns"]["image_table"] == list(sf.SPATIAL) + ["sample_id"]
          and j["columns"]["soil_table"] == list(sf.SPATIAL), str(j["columns"]))
    check("the artifact records that no camera column was left in place",
          j["camera_columns_present"] == [], str(j["camera_columns_present"]))


def test_three_way_split_partitions_the_census():
    state()
    asm, j = state()["asm"], strict_json(JSON_PATH)
    counts = asm["status_counts"]
    allowed = {d.STATUS_OK, d.STATUS_DEGENERATE, d.STATUS_NAN_KEYS}
    check("no census row carries the invalid_call status (one would have aborted the run)",
          set(counts) <= allowed and j["invalid_calls"] == 0, str(counts))
    check("successful + any-NaN tiles partition the corpus",
          j["successful_tiles"] + j["tiles_with_any_nan"] == j["total_tiles"],
          "%d + %d = %d" % (j["successful_tiles"], j["tiles_with_any_nan"], j["total_tiles"]))
    check("degenerate tiles are a subset of the any-NaN tiles",
          j["degenerate_tiles"] <= j["tiles_with_any_nan"],
          "%d of %d" % (j["degenerate_tiles"], j["tiles_with_any_nan"]))
    check("degenerate_tiles equals the degenerate status count",
          j["degenerate_tiles"] == counts.get(d.STATUS_DEGENERATE, 0), str(counts))
    check("nan_keys tiles equal any-NaN minus whole-tile degeneracies",
          counts.get(d.STATUS_NAN_KEYS, 0) == j["tiles_with_any_nan"] - j["degenerate_tiles"])
    csv = d.read_census_csv(CSV_PATH)
    check("the CSV status column reproduces the aggregate",
          csv.status.value_counts().to_dict() == counts, str(csv.status.value_counts().to_dict()))


def test_per_key_nan_tallies_match_the_artifact():
    state()
    j = strict_json(JSON_PATH)
    csv = d.read_census_csv(CSV_PATH)
    for k in sf.SPATIAL:
        from_csv = int(csv["%s_nan" % k].astype(bool).sum())
        check("per_key %s nan_count matches the CSV tally" % k, j["per_key"][k]["nan_count"]
              == from_csv, "%s vs %d" % (j["per_key"][k]["nan_count"], from_csv))
        v = csv[k].to_numpy(float)
        finite = v[np.isfinite(v)]
        check("per_key %s counts NaN and inf separately and adds them honestly" % k,
              j["per_key"][k]["nan_count"] == int(np.count_nonzero(np.isnan(v)))
              and j["per_key"][k]["inf_count"] == int(np.count_nonzero(np.isinf(v)))
              and j["per_key"][k]["nonfinite_count"]
              == j["per_key"][k]["nan_count"] + j["per_key"][k]["inf_count"],
              str({f: j["per_key"][k][f] for f in ("nan_count", "inf_count", "nonfinite_count")}))
        check("per_key %s has no inf at all and its finite count is complete" % k,
              j["per_key"][k]["inf_count"] == 0
              and finite.size + j["per_key"][k]["nan_count"] == len(csv),
              "finite %d nan %d" % (finite.size, j["per_key"][k]["nan_count"]))
        q = np.percentile(finite, [0, 5, 50, 95, 100])
        names = ("min", "p5", "median", "p95", "max")
        check("per_key %s percentiles are the measured ones" % k,
              all(abs(j["per_key"][k][f] - a) <= 1e-12 + 1e-9 * abs(a)
                  for f, a in zip(names, q)),
              "reported %s measured %s" % ([j["per_key"][k][f] for f in names],
                                           [round(float(x), 8) for x in q]))
        check("per_key %s ordering min <= p5 <= median <= p95 <= max" % k,
              j["per_key"][k]["min"] <= j["per_key"][k]["p5"] <= j["per_key"][k]["median"]
              <= j["per_key"][k]["p95"] <= j["per_key"][k]["max"])


def test_warnings_and_errors_are_zero_on_the_real_pass():
    asm, j = state()["asm"], strict_json(JSON_PATH)
    w = asm["warnings"]
    check("no warning was emitted during the real pass, tiles and aggregation alike", len(w) == 0,
          str(w[:2])[:200])
    check("census.json warnings_count equals what the pass recorded",
          j["warnings_count"] == len(w), "%s vs %d" % (j["warnings_count"], len(w)))
    check("no error was recorded (an error would have aborted the pass)",
          j["errors_count"] == 0 and len(asm["errors"]) == 0, str(asm["errors"][:2])[:200])
    check("the artifact reports the warning tally per stage it instrumented",
          set(j["warnings_by_stage"]) == set(d.WARNING_STAGES), str(j["warnings_by_stage"]))
    check("every instrumented stage of the real pass reported zero warnings",
          all(v == 0 for v in j["warnings_by_stage"].values()) and
          set(j["warnings_by_stage"]) == set(d.WARNING_STAGES), str(j["warnings_by_stage"]))


# ------------------------------------------------------------------ I3: the sink covers aggregation
def _tiny_q(rows, images, soils):
    """A miniature tile table with the three columns the aggregation actually reads."""
    return pd.DataFrame({"parent_image_path": images, "sample_id": soils, "_row": rows})


def _nan_key_rows():
    """Two tile rows whose A2-A4 are NaN: a real all-NaN column, so sf warns for a real reason."""
    base = sf.tile_spatial(_blob_tile(), PPM)
    return [dict(base), dict(base)]


def test_the_warning_sink_covers_the_aggregation_stages():
    """I3: sf._median_column warns at AGGREGATION time, outside the tile loop.

    The sink used to wrap only the tile loop, so a RuntimeWarning from a dead column escaped to
    stderr while the census still printed warnings: 0. Under the owner's warnings-fail ruling that
    is a live gap: the number that certifies the run was silent was not covering the stage that can
    make the noise. The case provoked here is not synthetic: an all-NaN column is what a degenerate
    tile leaves behind, and _median_column is registered to warn about it.
    """
    ctx = d.new_ctx()
    outer = warnings.showwarning
    img = d.image_table(_tiny_q(_nan_key_rows(), ["i1.png", "i1.png"], ["S1", "S1"]), ctx=ctx)
    check("the aggregation sink records the warnings the aggregation emits",
          len(ctx["warnings"]) >= 1, "%d recorded" % len(ctx["warnings"]))
    cats = {w["category"] for w in ctx["warnings"]}
    check("the recorded warnings are the real numpy/pandas notices, not a summary",
          "RuntimeWarning" in cats, str(cats))
    check("every recorded warning names the stage it came from",
          all(w.get("stage") == "image_block" for w in ctx["warnings"]),
          str([w.get("stage") for w in ctx["warnings"]][:4]))
    check("a warning from aggregation is attributed to the image being aggregated",
          all(w.get("parent_image_path") == "i1.png" for w in ctx["warnings"]),
          str(ctx["warnings"][:1])[:160])
    check("the warning message is kept, not just counted",
          all(len(str(w["message"])) > 10 for w in ctx["warnings"]),
          str([str(w["message"])[:40] for w in ctx["warnings"]][:2]))
    check("the aggregation still returns the row an all-NaN column entitles it to",
          len(img) == 1 and np.isnan(img["A2"].iloc[0]), str(img.to_dict("records"))[:120])
    check("the sink restored the process showwarning on exit",
          warnings.showwarning is outer, str(warnings.showwarning))
    more = d.soil_table(img, ctx=ctx)
    check("the second aggregation stage is instrumented too",
          any(w["stage"] == "soil_block" for w in ctx["warnings"]),
          str([w["stage"] for w in ctx["warnings"]][-3:]))
    check("soil_block over an all-NaN column also returns NaN, not a substitute",
          len(more) == 1 and np.isnan(more["A2"].iloc[0]), str(more.to_dict("index"))[:120])
    rep = d.report_from(ctx)
    check("the report surface counts aggregation warnings, so warnings: 0 can be false",
          rep["warnings"] == len(ctx["warnings"]) and rep["warnings"] > 0, str(rep)[:200])
    check("the report splits the tally by stage",
          rep["warnings_by_stage"].get("image_block", 0) > 0
          and rep["warnings_by_stage"].get("soil_block", 0) > 0,
          str(rep["warnings_by_stage"]))


def test_the_tile_stage_sink_still_catches_a_forced_warning():
    """The tiles stage keeps working, and the sink is proven armed rather than assumed armed."""
    ctx = d.new_ctx()
    real = sf.tile_spatial
    try:
        sf.tile_spatial = _warning_stand_in(real)
        q = d.tile_table(ROOT).head(1)
        d.scan_tiles(ROOT, q, ctx=ctx)
    finally:
        sf.tile_spatial = real
    check("a warning raised inside the tile loop is recorded against its tile",
          len(ctx["warnings"]) == 1 and ctx["warnings"][0]["stage"] == "tiles",
          str(ctx["warnings"][:1])[:180])
    check("the tile warning is attributed to the tile that produced it",
          bool(ctx["warnings"]) and ctx["warnings"][0]["tile_path"] == q.tile_path.iloc[0],
          str(ctx["warnings"][:1])[:160])
    check("the forced warning did not become a row or an error",
          len(ctx["records"]) == 1 and not ctx["errors"], str(ctx["records"][:1])[:120])
    check("the census tally would have failed on that warning",
          d.report_from(ctx)["warnings"] == 1, str(d.report_from(ctx))[:160])


def _warning_stand_in(real):
    """tile_spatial with one RuntimeWarning added, to arm-test the sink."""
    def fake(a, ppm):
        warnings.warn("injected for the sink leg", RuntimeWarning, stacklevel=2)
        return real(a, ppm)
    return fake


def test_assembly_itself_instruments_both_aggregation_stages():
    """I3's real target is the production path, not a hand-built ctx.

    Every other aggregation leg calls image_table and soil_table with a ctx it passed in itself, so
    deleting ctx=ctx from assembly() would leave the suite green: the census would report zero
    warnings at all three stages while stderr was not silent, which is exactly the defect I3 raised.
    This leg goes through assembly() and nothing else.
    """
    real_i, real_s, real_tt = sf.image_block, sf.soil_block, d.tile_table
    base_q = d.tile_table(ROOT).head(3)
    try:
        sf.image_block = _warning_block(real_i)
        sf.soil_block = _warning_block(real_s)
        d.tile_table = lambda root: base_q
        asm = d.assembly(ROOT, scrambled=True, seed=4242)
    finally:
        sf.image_block, sf.soil_block, d.tile_table = real_i, real_s, real_tt
    stages = sorted({w["stage"] for w in asm["warnings"]})
    check("assembly passes the recording context to BOTH aggregation stages",
          "image_block" in stages and "soil_block" in stages, str(stages))
    check("the assembly's own warning tally is non-empty when aggregation warns",
          len(asm["warnings"]) >= 2 and asm["report"]["warnings"] == len(asm["warnings"]),
          "%d recorded, tally %s" % (len(asm["warnings"]), asm["report"]["warnings"]))
    check("the assembly's stage tally attributes the warnings where they came from",
          asm["report"]["warnings_by_stage"]["image_block"] >= 1
          and asm["report"]["warnings_by_stage"]["soil_block"] >= 1,
          str(asm["report"]["warnings_by_stage"]))
    check("an aggregation warning is attributed to a soil, not to a tile row",
          bool(asm["warnings"]) and all(w["tile_path"] is None and w["sample_id"]
                                        for w in asm["warnings"]
                                        if w["stage"] in ("image_block", "soil_block")),
          str(asm["warnings"][:1])[:180])
    check("the warnings from this probe never reached the published census",
          state()["census"]["warnings_count"] == 0
          and state()["census"]["scrambled"] is False, str(state()["census"]["warnings_count"]))


def _warning_block(real):
    """The tested block with one RuntimeWarning in front of the real call."""
    def f(rows):
        warnings.warn("injected to prove assembly() instruments this stage", RuntimeWarning,
                      stacklevel=2)
        return real(rows)
    return f


def test_the_real_pass_installs_no_sink_beyond_the_pass():
    """The forced-warning legs above must not leak into the corpus claim."""
    ref = warnings.showwarning
    ctx = d.new_ctx()
    q = d.tile_table(ROOT).head(1)
    d.scan_tiles(ROOT, q, ctx=ctx)
    check("an ordinary pass records no warning", ctx["warnings"] == [], str(ctx["warnings"])[:160])
    check("the pass leaves warnings.showwarning exactly as it found it",
          warnings.showwarning is ref, str(warnings.showwarning))
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        warnings.warn("control notice", RuntimeWarning)
        check("outside the pass a warning behaves normally instead of being captured",
              len(got) == 1, str([str(m.message) for m in got])[:120])


def test_outlier_rule_is_two_sided_where_one_sided_is_blind():
    """I2: |x - median| / MAD cannot see the lower tail of a positive key.

    With A2's median / MAD ratio, no value below the median can ever reach z 10, so the absolute
    rule flagged 27 high tiles and was structurally incapable of flagging the low ones. A ratio
    diagnostic on the log scale is symmetric: 0.1 x median and 10 x median are the same distance
    from the median in multiplicative terms, which is how these statistics are actually built.
    """
    v = np.array([10.0, 14.0, 6.0, 12.0, 8.0, 11.0, 9.0, 0.01, 900.0])
    s = d.outlier_stats(v)
    check("the absolute rule flags only the high member of this column",
          s["count_beyond_10_mad"] == 1, str(s.get("count_beyond_10_mad")))
    check("the log rule flags both tails of a positive key",
          s["log_two_sided_applicable"] is True
          and s["count_beyond_10_mad_log"] == 2, str(s.get("count_beyond_10_mad_log")))
    check("a one-sided diagnosis is recorded so the report says which rule saw what",
          "count_beyond_10_mad" in s and "count_beyond_10_mad_log" in s)
    low_only = np.array([10.0, 14.0, 6.0, 12.0, 8.0, 11.0, 9.0, 0.01])
    ls = d.outlier_stats(low_only)
    check("a column whose only extreme value is LOW is invisible to the absolute rule",
          ls["count_beyond_10_mad"] == 0, "max |z| %s" % ls["max_abs_z"])
    check("the log rule catches the low outlier the absolute rule cannot reach",
          ls["count_beyond_10_mad_log"] == 1, str(ls.get("count_beyond_10_mad_log")))
    check("the absolute rule ranks that low value under its own threshold, so it stayed invisible",
          ls["max_abs_z"] < MAD_Z, "max |z| %s" % ls["max_abs_z"])
    check("the same value is far past the threshold in multiplicative terms",
          ls["max_abs_z_log"] > MAD_Z, "max log |z| %s" % ls["max_abs_z_log"])
    mixed = np.array([-1.0, 0.5, 0.0, 2.0, -3.0])
    ms = d.outlier_stats(mixed)
    check("a key straddling zero reports the log rule as not applicable rather than inventing one",
          ms["log_two_sided_applicable"] is False
          and ms["count_beyond_10_mad_log"] is None, str(ms)[:200])
    check("a key straddling zero still gets the absolute rule", ms["mad"] is not None
          and isinstance(ms["count_beyond_10_mad"], int), str(ms)[:200])
    neg = np.array([-1.0, -1.05, -0.95, -1.02, -50.0, -0.009])
    ns = d.outlier_stats(neg)
    check("a strictly negative key is measured on the log of its magnitude",
          ns["log_two_sided_applicable"] is True and ns["log_of"] == "abs", str(ns)[:200])
    check("the log rule flags the near-zero member of a negative key",
          ns["count_beyond_10_mad_log"] >= 2, str(ns.get("count_beyond_10_mad_log")))
    check("a strictly positive key says it took the log of the value itself",
          ls["log_of"] == "value", str(ls["log_of"]))
    check("a key measured on its magnitude says so, so the report cannot be misread",
          isinstance(ns["log_note"], str) and "MAGNITUDE" in ns["log_note"]
          and isinstance(ls["log_note"], str) and "MAGNITUDE" not in ls["log_note"],
          str(ns["log_note"])[:120])
    notes = [(rep, rep["log_two_sided_applicable"]) for rep in (s, ls, ms, ns)]
    check("the multiplicative note is present exactly where the rule is applicable",
          all("log_note" in rep and (rep["log_note"] is not None) == ok
              for rep, ok in notes), str([(r["log_of"], r["log_note"] is not None)
                                         for r, _ in notes])[:160])
    empty = d.outlier_stats(np.array([float("nan"), float("inf")]))
    check("a key with no finite value keeps the whole field set, not a shorter dict",
          set(empty) == set(s), str(sorted(set(s) - set(empty))))
    check("an all-non-finite key reports itself as having no opinion",
          empty["n_finite"] == 0 and empty["count_beyond_10_mad"] is None
          and empty["log_two_sided_applicable"] is False, str(empty)[:200])


def test_zero_mad_is_reported_as_its_own_condition():
    """I2: a constant column must not be reported as "0 outliers"."""
    flat = np.full(40, 7.0)
    s = d.outlier_stats(flat)
    check("a zero robust scale is named, not silently counted as clean",
          s["mad_is_zero"] is True, str(s)[:180])
    check("a zero robust scale yields no absolute count, not a zero count",
          s["count_beyond_10_mad"] is None, str(s.get("count_beyond_10_mad")))
    check("a zero robust scale also yields no log count",
          s["count_beyond_10_mad_log"] is None, str(s.get("count_beyond_10_mad_log")))
    check("the artifact says so in words the reader cannot miss",
          "not a claim of clean" in s["interpretation"], str(s.get("interpretation")))
    near = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 30.0])
    ns = d.outlier_stats(near)
    check("a column whose MAD happens to be zero but which has one distinct value flags nothing "
          "and says why", ns["mad_is_zero"] is True and ns["n_distinct"] == 2, str(ns)[:200])


def test_extreme_outlier_block_is_reported_and_bounded():
    j = strict_json(JSON_PATH)
    blk = j["extreme_outlier"]
    check("the outlier block names its rule", blk.get("mad_z_threshold") == MAD_Z
          and blk.get("magnitude_floor") == 1e6, str({k: blk.get(k) for k in
                                                      ("mad_z_threshold", "magnitude_floor")}))
    check("the block states that the same threshold drives the absolute and the log rule",
          blk.get("log_z_threshold") == MAD_Z and blk.get("mad_z_threshold") == MAD_Z,
          str({k: blk.get(k) for k in ("mad_z_threshold", "log_z_threshold")}))
    for k in sf.SPATIAL:
        b = blk["keys"][k]
        check("outlier block complete for %s" % k,
              all(f in b for f in ("mad", "count_beyond_10_mad", "max_abs_z", "n_finite",
                                   "max_abs_value", "whole_distribution_above_1e6",
                                   "count_beyond_10_mad_log", "max_abs_z_log",
                                   "log_two_sided_applicable", "mad_is_zero", "n_distinct",
                                   "log_note", "interpretation")),
              str(sorted(b.keys())))
        check("no tile value for %s reaches 1e6 in magnitude (the definitions bound it far "
              "below)" % k, b["max_abs_value"] < 1e6 and not b["whole_distribution_above_1e6"],
              "max|x| %s  n>1e6 %d" % (num(b["max_abs_value"]), b["n_abs_above_1e6"]))
        check("the log rule reports which scale it used for %s" % k,
              b["log_of"] in ("value", "abs", None), str(b["log_of"]))
        if b["log_two_sided_applicable"]:
            check("%s is one-signed, so both tails were examined" % k,
                  b["count_beyond_10_mad_log"] is not None, str(b["count_beyond_10_mad_log"]))
        print("       outlier report %s: mad %s, beyond-10MAD %s (log %s), max |z| %s (log %s)"
              % (k, num(b["mad"]), b["count_beyond_10_mad"], b["count_beyond_10_mad_log"],
                 num(b["max_abs_z"], "%.3f"), num(b["max_abs_z_log"], "%.3f")))
    for k in sf.SPATIAL:
        b = blk["soil_level"][k]
        check("the soil-level block carries the same two-sided report for %s" % k,
              all(f in b for f in ("count_beyond_10_mad", "count_beyond_10_mad_log",
                                   "log_two_sided_applicable", "mad_is_zero")), str(b)[:160])
    csv = d.read_census_csv(CSV_PATH)
    rec = np.array([[r["values"][k] for k in sf.SPATIAL] for r in state()["asm"]["records"]], float)
    cv = csv[list(sf.SPATIAL)].to_numpy(float)
    gap = float(np.nan_to_num(np.abs(cv - rec)).max()) if cv.shape == rec.shape else float("nan")
    check("nothing was clamped, transformed or winsorised: the artifact holds the raw numbers the "
          "pass computed", np.array_equal(cv, rec, equal_nan=True),
          "rows %d vs %d, max |artifact - memory| %.3g" % (len(cv), len(rec), gap))
    check("the diagnostic did not filter rows either: every tile is still in the artifact",
          len(csv) == j["total_tiles"] == 1946, "%d" % len(csv))
    check("the diagnostic did not transform the columns it measured: percentiles still match",
          all(abs(j["per_key"][k]["median"] - float(np.nanmedian(csv[k].to_numpy(float)))) < 1e-12
              for k in sf.SPATIAL),
          str({k: j["per_key"][k]["median"] for k in sf.SPATIAL}))


# ------------------------------------------------------------------ I5: finiteness before writing
def test_strict_json_leg_can_actually_fail():
    """The strict-JSON check was un-failable: _json_safe turned every non-finite into null first.

    Proved failable two ways: a planted NaN in an artifact is caught, and the writer now refuses to
    sanitise one instead of quietly hiding it.
    """
    tmp = HERE / "scratch" / "_nan_probe.json"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp.write_text('{"a": NaN}\n', encoding="utf-8", newline="\n")
        err = caught(strict_json, tmp)
        check("a planted NaN in an artifact is caught by the strict loader",
              isinstance(err, ValueError), type(err).__name__ if err is not None
              else "the loader accepted a NaN")
        tmp.write_text('{"a": Infinity}\n', encoding="utf-8", newline="\n")
        check("a planted Infinity is caught too", isinstance(caught(strict_json, tmp), ValueError))
    finally:
        tmp.unlink(missing_ok=True)
    err_inf = caught(d._json_safe, {"a": float("inf")})
    check("the writer refuses to launder a non-finite into null",
          isinstance(err_inf, (ValueError, AssertionError)),
          "returned %r instead of raising" % (err_inf,) if err_inf is None else "")
    err_nan = caught(d._json_safe, {"a": [float("nan")]})
    check("the writer refuses to launder a NaN either",
          isinstance(err_nan, (ValueError, AssertionError)),
          "returned %r instead of raising" % (err_nan,) if err_nan is None else "")
    check("_json_safe still converts the types JSON genuinely cannot hold",
          d._json_safe({"n": np.int64(3), "f": np.float64(1.5), "b": np.bool_(True)})
          == {"n": 3, "f": 1.5, "b": True})


def test_nonfinite_paths_are_found_before_serialisation():
    """The aborting check runs on the aggregate while the bad value is still visible."""
    check("an inf in a nested report is located by path",
          d.find_nonfinite({"a": {"b": [1.0, float("inf")]}}) == ["$.a.b[1]"],
          str(d.find_nonfinite({"a": {"b": [1.0, float("inf")]}})))
    check("a NaN in a named field is located by name",
          d.find_nonfinite({"per_key": {"A2": {"median": float("nan")}}})
          == ["$.per_key.A2.median"], str(d.find_nonfinite(
              {"per_key": {"A2": {"median": float("nan")}}})))
    check("a clean aggregate yields no paths", d.find_nonfinite(
        {"x": 1.0, "y": [2.0, None], "z": "text"}) == [], str(d.find_nonfinite(
            {"x": 1.0, "y": [2.0, None]})))
    check("None is not treated as non-finite (it is the registered empty answer)",
          d.find_nonfinite({"mad": None, "count_beyond_10_mad": None}) == [])


def test_run_census_aborts_before_touching_the_artifacts():
    """A non-finite in the numbers to be published must stop the write, not be hidden by it."""
    asm = state()["asm"]
    j = strict_json(JSON_PATH)
    bad = json.loads(json.dumps(j))
    bad["per_key"]["A2"]["median"] = float("nan")
    check("the tampered aggregate is the case the guard exists for",
          d.find_nonfinite(bad) == ["$.per_key.A2.median"], str(d.find_nonfinite(bad)))
    probe = HERE / "scratch" / "publish_probe"
    err = caught(d.publish, asm, bad, out_dir=probe, write=True)
    check("publishing an aggregate with a NaN raises instead of sanitising it",
          isinstance(err, (ValueError, AssertionError)),
          type(err).__name__ if err is not None else "it wrote")
    check("the failed publish wrote nothing at all",
          not probe.exists() or sorted(p.name for p in probe.iterdir()) == [],
          str(sorted(p.name for p in probe.iterdir())) if probe.exists() else "no directory")
    err2 = caught(d.publish, asm, j, out_dir=None, write=False)
    check("the real aggregate passes the same guard", err2 is None,
          "no exception" if err2 is None else str(err2)[:160])
    check("a dry publish leaves the artifacts exactly as they were",
          sorted(p.name for p in ART_DIR.iterdir()) == sorted(EXPECTED_ARTIFACTS),
          str(sorted(p.name for p in ART_DIR.iterdir())))


def test_the_frames_are_guarded_as_well_as_the_report():
    """A clean census.json is not enough: the guard must look at the arrays that get written.

    Without this, a NaN could arrive in a design frame, the writer would print NaN, the JSON would
    stay clean, and the strict-JSON leg would still PASS. The wiring is tested through publish() as
    well as through the helper, because a correct guard that the writer never calls is no guard. Two
    boundaries must also stay OPEN: a NaN the pass COUNTED as a degenerate tile stays publishable
    (outcome (2) is a real outcome and forbidding it would be an act on the data), while an image
    whose aggregate is unusable refuses the whole publish rather than writing a frame a fit cannot
    consume.
    """
    cen = {"per_key": {"A1": {"median": 1.0}}, "note": "clean"}

    def asm_of(tile_rows, img_rows, soil_rows, statuses, reasons):
        tf = pd.DataFrame(tile_rows, columns=list(sf.SPATIAL))
        img = pd.DataFrame(img_rows, columns=list(sf.SPATIAL),
                           index=pd.Index(["i1.png"], name="parent_image_path"))
        soil = pd.DataFrame(soil_rows, columns=list(sf.SPATIAL),
                            index=pd.Index(["S1"], name="sample_id"))
        recs = [{"values": {k: v for k, v in zip(sf.SPATIAL, row)}, "status": st,
                 "nan_keys": [k for k, v in zip(sf.SPATIAL, row) if not np.isfinite(v)],
                 "reason": rs, "tile_path": "t%d" % i}
                for i, (row, st, rs) in enumerate(zip(tile_rows, statuses, reasons))]
        return {"tile_frame": tf, "image": img, "soil": soil, "records": recs}
    good = [[-0.2, 1.0, 0.05, 0.001], [-0.3, 1.1, 0.06, -0.002]]
    ok_img = [[-0.25, 1.05, 0.055, -0.0005]]
    ok_soil = [[-0.25, 1.05, 0.055, -0.0005]]
    clean = asm_of(good, ok_img, ok_soil, ["ok", "ok"], ["", ""])
    nan = float("nan")
    check("a clean assembly passes the frame guard",
          caught(d.require_publishable, cen, clean) is None,
          str(caught(d.require_publishable, cen, clean))[:160])
    inf_tile = asm_of([[-0.2, float("inf"), 0.05, 0.001], good[1]], ok_img, ok_soil,
                      ["nan_keys", "ok"], ["A2: refused"])
    err_inf = caught(d.require_publishable, cen, inf_tile)
    check("an +-inf in the tile frame stops the publish", isinstance(err_inf, ValueError),
          str(err_inf)[:160])
    check("and it names the tile that produced it, so the bug is locatable",
          "t0" in str(err_inf), str(err_inf)[:160])
    nan_img = asm_of(good, [[-0.25, nan, 0.055, -0.0005]], ok_soil, ["ok", "ok"], ["", ""])
    check("a NaN in the image design frame stops the publish",
          isinstance(caught(d.require_publishable, cen, nan_img), ValueError),
          str(caught(d.require_publishable, cen, nan_img))[:160])
    nan_soil = asm_of(good, ok_img, [[-0.25, 1.05, 0.055, nan]], ["ok", "ok"], ["", ""])
    check("a NaN in the soil design frame stops the publish",
          isinstance(caught(d.require_publishable, cen, nan_soil), ValueError))
    counted = asm_of([[-0.2, nan, 0.05, 0.001], good[1]], ok_img, ok_soil,
                     ["nan_keys", "ok"], ["A2: the 1/e crossing is unresolvable"])
    check("a NaN the pass counted as a degenerate key is still allowed to be published",
          caught(d.require_publishable, cen, counted) is None,
          str(caught(d.require_publishable, cen, counted))[:160])
    reasonless = asm_of([[-0.2, nan, 0.05, 0.001], good[1]], ok_img, ok_soil,
                        [d.STATUS_NAN_KEYS, "ok"], [""])
    check("a tile filed as degenerate WITHOUT a reason is refused too: the count would be real but "
          "the census could not say why",
          isinstance(caught(d.require_publishable, cen, reasonless), ValueError),
          str(caught(d.require_publishable, cen, reasonless))[:160])
    uncounted = asm_of([[-0.2, nan, 0.05, 0.001], good[1]], ok_img, ok_soil, ["ok", "ok"], ["", ""])
    check("a NaN filed as an ordinary statistic aborts the publish instead of vanishing",
          isinstance(caught(d.require_publishable, cen, uncounted), ValueError),
          str(caught(d.require_publishable, cen, uncounted))[:160])
    check("the guard reaches the tile path of the offending row, not just a count",
          "t0" in str(caught(d.require_publishable, cen, uncounted)),
          str(caught(d.require_publishable, cen, uncounted))[:160])
    dead = [[nan] * 4, [nan] * 4]
    wholly = asm_of(dead, [[nan] * 4], [[nan] * 4],
                    [d.STATUS_DEGENERATE, d.STATUS_DEGENERATE], ["degenerate mask"] * 2)
    err_dead = caught(d.require_publishable, cen, wholly)
    check("an image whose every tile is degenerate refuses the whole publish, rather than writing "
          "a design frame a rank-3 fit cannot consume",
          isinstance(err_dead, ValueError) and "image design frame" in str(err_dead),
          str(err_dead)[:160])
    check("and that refusal counts the cells it objecting to, so nothing is silently discarded",
          "4 NaN cell" in str(err_dead), str(err_dead)[:160])

    # now the wiring: publish() must run this guard, not merely be able to
    real = state()["asm"]
    broken = dict(real)
    bad_img = real["image"].copy()
    bad_img.iloc[0, 0] = nan
    broken["image"] = bad_img
    clean_cen = d.aggregate(real)
    check("the aggregate alone shows nothing wrong with the broken assembly",
          d.find_nonfinite(clean_cen) == [], str(d.find_nonfinite(clean_cen))[:120])
    probe = HERE / "scratch" / "frame_guard_probe"
    err = caught(d.publish, broken, clean_cen, out_dir=probe, write=True)
    try:
        check("publish() itself refuses a frame-only defect",
              isinstance(err, (ValueError, AssertionError)),
              type(err).__name__ if err is not None else "it wrote all four artifacts")
        check("the refused frame-only publish created nothing",
              not probe.exists() or sorted(p.name for p in probe.iterdir()) == [],
              str(sorted(p.name for p in probe.iterdir())) if probe.exists()
              else "no directory")
        err_ok = caught(d.publish, real, clean_cen, out_dir=None, write=False)
        check("the same call on the real assembly is not refused", err_ok is None,
              "no exception" if err_ok is None else str(err_ok)[:160])
    finally:
        if probe.exists():
            for pth in sorted(probe.iterdir()):
                pth.unlink()
            probe.rmdir()


def test_tile_frame_finiteness_is_measured_directly():
    """I5: assert finiteness on the assembled numbers, not on what the writer chose to print."""
    asm = state()["asm"]
    j = strict_json(JSON_PATH)
    tf = asm["tile_frame"][list(sf.SPATIAL)].to_numpy(float)
    img = asm["image"][list(sf.SPATIAL)].to_numpy(float)
    soil = asm["soil"][list(sf.SPATIAL)].to_numpy(float)
    check("no tile cell is +-inf", int(np.count_nonzero(np.isinf(tf))) == 0,
          "%d inf cells" % int(np.count_nonzero(np.isinf(tf))))
    check("no image or soil cell is +-inf",
          int(np.count_nonzero(np.isinf(img))) + int(np.count_nonzero(np.isinf(soil))) == 0)
    check("ON THIS CORPUS every tile cell is finite, and none is +-inf (a corpus with counted "
          "degenerates would fail this leg by design, not by defect)",
          int(np.count_nonzero(~np.isfinite(tf))) == int(np.count_nonzero(np.isnan(tf))) == 0,
          "%d non-finite, %d NaN" % (int(np.count_nonzero(~np.isfinite(tf))),
                                      int(np.count_nonzero(np.isnan(tf)))))
    fin = j["finiteness"]
    check("the artifact states once what non-finite means here",
          fin.get("meaning") == d.NON_FINITE_MEANING, str(fin.get("meaning"))[:120])
    check("the artifact reports the tile-level cell counts it measured",
          fin["tile"]["non_finite_cells"] == int(np.count_nonzero(~np.isfinite(tf)))
          and fin["tile"]["cells"] == int(tf.size), str(fin["tile"]))
    check("the artifact reports the image and soil levels too",
          fin["image"]["non_finite_cells"] == int(np.count_nonzero(~np.isfinite(img)))
          and fin["soil"]["non_finite_cells"] == int(np.count_nonzero(~np.isfinite(soil))))
    check("the published design is entirely finite at every level on this corpus",
          all(fin[l]["non_finite_cells"] == 0 for l in ("tile", "image", "soil")), str(fin))
    for k in sf.SPATIAL:
        col = asm["tile_frame"][k].to_numpy(float)
        s = j["per_key"][k]
        check("per_key %s: nonfinite_count is NaN plus inf, not one of them alone" % k,
              s["nonfinite_count"] == s["nan_count"] + s["inf_count"]
              and s["inf_count"] == int(np.count_nonzero(np.isinf(col)))
              and s["nan_count"] == int(np.count_nonzero(np.isnan(col))),
              str({f: s.get(f) for f in ("nan_count", "inf_count", "nonfinite_count")}))


def test_non_finite_has_one_meaning_in_every_surface():
    """M2: the same token meant a tile count in one place and an inf count in another."""
    asm, j = state()["asm"], strict_json(JSON_PATH)
    rep = d.report_from(d.asm_records_ctx(asm))
    tf = asm["tile_frame"][list(sf.SPATIAL)].to_numpy(float)
    tiles_nf = int(np.count_nonzero(~np.isfinite(tf).all(axis=1)))
    check("report non_finite counts TILES with at least one non-finite key",
          rep["non_finite"] == tiles_nf, "%s vs %d" % (rep["non_finite"], tiles_nf))
    check("report non_finite_cells counts CELLS, separately",
          rep["non_finite_cells"] == int(np.count_nonzero(~np.isfinite(tf))),
          str(rep["non_finite_cells"]))
    check("per-key nonfinite tallies sum to the cell count",
          sum(rep["per_key_nonfinite"].values()) == rep["non_finite_cells"],
          str(rep["per_key_nonfinite"]))
    check("the aggregate uses the same two names with the same two meanings",
          j["non_finite_tiles"] == tiles_nf and j["finiteness"]["tile"]["non_finite_cells"]
          == rep["non_finite_cells"], str(j["non_finite_tiles"]))
    tiles_nan = int(np.count_nonzero(np.isnan(tf).any(axis=1)))
    check("nan_rate keeps the brief's formula, the non-finite tile fraction, and says so beside it",
          abs(rep["nan_rate"] - tiles_nf / rep["n_tiles"]) < 1e-15
          and abs(rep["non_finite_rate"] - rep["nan_rate"]) < 1e-15
          and abs(rep["nan_only_rate"] - tiles_nan / rep["n_tiles"]) < 1e-15
          and rep["tiles_with_nan"] == tiles_nan,
          "nan_rate %s nan_only_rate %s" % (rep["nan_rate"], rep["nan_only_rate"]))
    check("a corpus with no NaN has both rates at zero, and they are different numbers to count",
          tiles_nan == tiles_nf == rep["nan_cells"] == rep["non_finite_cells"] == 0,
          str({k: rep[k] for k in ("nan_cells", "non_finite_cells", "tiles_with_nan",
                                   "non_finite")}))
    check("the tile-level count of NaN keys is still reported under its own name",
          j["tiles_with_any_nan"] == sum(1 for r in asm["records"]
                                         if r["status"] != d.STATUS_OK), str(j["tiles_with_any_nan"]))
    check("NAN_REPORT carries the meaning string so a reader cannot guess it",
          d.NAN_REPORT.get("meaning") == d.NON_FINITE_MEANING, str(d.NAN_REPORT)[:200])
    # the corpus is all-zero everywhere, which makes the two rates indistinguishable on real data;
    # this synthetic pass is what proves they are different measurements rather than two names
    mixed = {"records": [
        {"values": {"A1": -0.2, "A2": float("nan"), "A3": 0.05, "A4": 0.0}, "status": "nan_keys",
         "nan_keys": ["A2"], "reason": "A2: refused", "tile_path": "t0"},
        {"values": {"A1": -0.2, "A2": 1.0, "A3": float("inf"), "A4": 0.0}, "status": "nan_keys",
         "nan_keys": ["A3"], "reason": "A3: refused", "tile_path": "t1"},
        {"values": {"A1": -0.1, "A2": 1.1, "A3": 0.06, "A4": 0.001}, "status": "ok",
         "nan_keys": [], "reason": "", "tile_path": "t2"}],
        "errors": [], "warnings": [], "scrambled": False, "seed_base": 0}
    m = d.report_from(mixed)
    check("with a NaN tile and an inf tile, non_finite counts tiles carrying EITHER (3 fields)",
          m["non_finite"] == 2 and m["non_finite_cells"] == 2
          and m["per_key_nonfinite"] == {"A1": 0, "A2": 1, "A3": 1, "A4": 0}, str(m)[:200])
    check("while tiles_with_nan and nan_cells see only the NaN",
          m["tiles_with_nan"] == 1 and m["nan_cells"] == 1
          and m["per_key_nonfinite"] == {"A1": 0, "A2": 1, "A3": 1, "A4": 0}, str(m)[:200])
    check("per_key_nan counts the keys the pass FILED as undefined, which is the nan_keys record",
          m["per_key_nan"] == {"A1": 0, "A2": 1, "A3": 1, "A4": 0}
          and sum(m["per_key_nan"].values())
          == sum(len(r["nan_keys"]) for r in mixed["records"]), str(m["per_key_nan"]))
    check("an inf can only reach that field through a fabricated record, because a real tile is "
          "either measured, refused as degenerate, or refused as an invalid call",
          isinstance(caught(sf.tile_spatial, np.full((64, 64, 3), 120.0), PPM),
                     (sf.TileDegenerate, ValueError)),
          "the degenerate tile returned instead of refusing")
    check("the two rates are different numbers when the two conditions differ",
          abs(m["nan_rate"] - 2.0 / 3.0) < 1e-15 and abs(m["nan_only_rate"] - 1.0 / 3.0) < 1e-15,
          "nan_rate %s nan_only_rate %s" % (m["nan_rate"], m["nan_only_rate"]))
    check("the whole-tile degeneracy is still counted by its own field, not by a rate",
          m["degenerate_mask"] == 0, str(m["degenerate_mask"]))


def test_nan_report_surface_keeps_the_brief_fields():
    """The plan's NAN_REPORT keys survive the rename, with one meaning attached to each."""
    state()
    rep = dict(d.NAN_REPORT)
    for f in ("n_tiles", "degenerate_mask", "non_finite", "nan_rate", "per_key_nan",
              "invalid_calls", "warnings"):
        check("NAN_REPORT keeps the brief's field %s" % f, f in rep, str(sorted(rep.keys())))
    asm = state()["asm"]
    tf = asm["tile_frame"][list(sf.SPATIAL)].to_numpy(float)
    check("NAN_REPORT n_tiles is the corpus", rep["n_tiles"] == 1946, str(rep["n_tiles"]))
    check("NAN_REPORT degenerate_mask is the whole-tile count",
          rep["degenerate_mask"] == asm["status_counts"].get(d.STATUS_DEGENERATE, 0),
          str(rep["degenerate_mask"]))
    check("NAN_REPORT non_finite is the non-finite TILE count, the one meaning used here",
          rep["non_finite"] == int(np.count_nonzero(~np.isfinite(tf).any(axis=1))),
          str(rep["non_finite"]))
    check("NAN_REPORT nan_rate is the brief's non-finite tile fraction",
          abs(rep["nan_rate"] - rep["non_finite"] / rep["n_tiles"]) < 1e-15,
          str(rep["nan_rate"]))
    check("NAN_REPORT nan_only_rate is the NaN-tile fraction the name describes",
          abs(rep["nan_only_rate"] - int(np.count_nonzero(np.isnan(tf).any(axis=1)))
              / rep["n_tiles"]) < 1e-15, str(rep["nan_only_rate"]))
    check("NAN_REPORT per-key NaN tallies match the artifact",
          sum(rep["per_key_nan"].values()) == sum(strict_json(JSON_PATH)["per_key"][k]["nan_count"]
                                                   for k in sf.SPATIAL), str(rep["per_key_nan"]))
    check("NAN_REPORT per-key non-finite tallies are cell counts, not tile counts",
          sum(rep["per_key_nonfinite"].values())
          == strict_json(JSON_PATH)["finiteness"]["tile"]["non_finite_cells"],
          str(rep["per_key_nonfinite"]))
    check("NAN_REPORT warnings is the whole-pass tally across all three stages",
          rep["warnings"] == sum(rep["warnings_by_stage"].values())
          and set(rep["warnings_by_stage"]) == set(d.WARNING_STAGES), str(rep["warnings_by_stage"]))
    check("NAN_REPORT counts no invalid call, because one would have aborted the pass",
          rep["invalid_calls"] == 0, str(rep["invalid_calls"]))


# ------------------------------------------------------------------ M8: the design is auditable
def test_the_assembled_frames_are_persisted_and_round_trip():
    asm = state()["asm"]
    check("data/image_features_m7a.csv exists", IMG_CSV.exists(), str(IMG_CSV))
    check("data/soil_features_m7a.csv exists", SOIL_CSV.exists(), str(SOIL_CSV))
    img = d.read_feature_csv(IMG_CSV)
    soil = d.read_feature_csv(SOIL_CSV)
    check("the persisted image frame has the tested aggregation's rows",
          list(img.index.astype(str)) == list(asm["image"].index.astype(str)),
          "%d vs %d" % (len(img), len(asm["image"])))
    check("the persisted soil frame has the tested aggregation's rows",
          list(soil.index.astype(str)) == list(asm["soil"].index.astype(str)),
          "%d vs %d" % (len(soil), len(asm["soil"])))
    check("the persisted image frame equals the in-memory one bit for bit",
          np.array_equal(img[list(sf.SPATIAL)].to_numpy(float),
                         asm["image"][list(sf.SPATIAL)].to_numpy(float), equal_nan=True))
    check("the persisted soil frame equals the in-memory one bit for bit",
          np.array_equal(soil[list(sf.SPATIAL)].to_numpy(float),
                         asm["soil"].to_numpy(float), equal_nan=True))
    check("the persisted frames carry no camera, ppm or position column",
          not any(c in DROPPED + ("normalized_ppm", "tile_x", "tile_y")
                  for c in list(img.columns) + list(soil.columns)),
          str(list(img.columns) + list(soil.columns)))
    check("the soil frame carries exactly A1-A4", list(soil.columns) == list(sf.SPATIAL),
          str(list(soil.columns)))
    check("every published artifact is hashed by the run that wrote it",
          set(state()["census"]["artifact_sha256"])
          == {d.CSV_NAME, d.JSON_NAME, d.IMG_CSV_NAME, d.SOIL_CSV_NAME},
          str(sorted(state()["census"]["artifact_sha256"])))
    check("the hashes match the files on disk, byte for byte",
          all(state()["census"]["artifact_sha256"][name]
              == hashlib.sha256(p.read_bytes()).hexdigest()
              for name, p in d.artifact_paths().items()), "all four sha256 agree")
    check("every published artifact is listed with its size",
          set(state()["census"]["artifact_bytes"])
          == {d.CSV_NAME, d.JSON_NAME, d.IMG_CSV_NAME, d.SOIL_CSV_NAME},
          str(state()["census"]["artifact_bytes"]))
    # all four files, not just the census, must be a pure function of the assembly: republish into a
    # probe directory from the memoised frames and byte-compare against what is on disk
    probe = HERE / "scratch" / "frame_probe"
    try:
        again = d.publish(state()["asm"], out_dir=probe, write=True)
        same = all(again["artifact_sha256"][name]
                   == hashlib.sha256((ART_DIR / name).read_bytes()).hexdigest()
                   for name in d.artifact_paths())
        check("re-publishing the same assembly reproduces all four artifacts byte for byte", same,
              str({n: (again["artifact_sha256"][n][:8],
                       hashlib.sha256((ART_DIR / n).read_bytes()).hexdigest()[:8])
                   for n in d.artifact_paths() if again["artifact_sha256"][n]
                   != hashlib.sha256((ART_DIR / n).read_bytes()).hexdigest()})[:200])
        check("the probe publish wrote the same four filenames and nothing else",
              sorted(p.name for p in probe.iterdir()) == sorted(EXPECTED_ARTIFACTS),
              str(sorted(p.name for p in probe.iterdir())))
    finally:
        if probe.exists():
            for p in sorted(probe.iterdir()):
                p.unlink()
            probe.rmdir()
    check("the probe directory is cleaned up, so a re-run cannot read a stale frame",
          not probe.exists(), str(probe))


# ------------------------------------------------------------------ controls wired, not run
def test_scrambled_placebo_path_is_wired_but_not_censused():
    q = d.tile_table(ROOT).head(3)
    real = d.scan_tiles(ROOT, q, scrambled=False)["records"]
    sc1 = d.scan_tiles(ROOT, q, scrambled=True, seed=90001)["records"]
    sc2 = d.scan_tiles(ROOT, q, scrambled=True, seed=90001)["records"]
    check("the scrambled path runs on real tiles and returns all four keys",
          all(r["status"] in (d.STATUS_OK, d.STATUS_NAN_KEYS) for r in sc1))
    check("scrambling changes the statistics (it is not a no-op)",
          any(abs(real[i]["values"]["A2"] - sc1[i]["values"]["A2"]) > 1e-9 for i in range(3)),
          str([round(sc1[i]["values"]["A2"], 4) for i in range(3)]))
    check("the scrambled path is deterministic in the recorded seed",
          [r["values"] for r in sc1] == [r["values"] for r in sc2])
    check("the seed is the manifest row index plus the base, per tile",
          sc1[0]["seed_used"] == 90001 + int(q.index[0]) and sc1[2]["seed_used"] == 90001
          + int(q.index[2]), str([r["seed_used"] for r in sc1]))
    check("the census itself ran unscrambled", state()["census"]["scrambled"] is False)


def test_spatial_features_stays_a_numpy_only_module():
    """I6: the dataframe ban is read off the import graph, not off the module's prose."""
    imports = set()
    for node in ast.walk(ast.parse((HERE / "spatial_features.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    check("spatial_features imports no dataframe library", not imports & {"pandas"},
          str(sorted(imports)))
    check("no dataframe module is bound into spatial_features at runtime",
          not hasattr(sf, "pd") and not hasattr(sf, "DataFrame")
          and not any(getattr(v, "__name__", "") == "pandas" for v in sf.__dict__.values()),
          str([k for k in sf.__dict__ if k.startswith("pd")]))


def test_data_directory_is_never_written():
    before = state()["data_fp_before"]
    after = d.data_fingerprint(ROOT)
    check("repo data/ is byte-for-byte untouched by the whole pass", before == after,
          "before %s after %s" % (before, after))
    check("only the four published artifacts live in Model 7/Approach A/data",
          sorted(p.name for p in ART_DIR.iterdir()) == sorted(EXPECTED_ARTIFACTS),
          str(sorted(p.name for p in ART_DIR.iterdir())))


def test_determinism_three_runs_identical():
    h1 = state()["asm"]["hashes"]
    d.clear_assembly_cache()
    asm2 = d.assembly(ROOT)
    h2 = asm2["hashes"]
    STATE["runs"] = {"run 1, this process": tuple(h1)}
    check("the second in-process pass reproduces the first, byte for byte", h1 == h2,
          "%s / %s" % (h1, h2))
    STATE["runs"]["run 2, this process"] = tuple(h2)
    global FRESH
    if FRESH is None:
        check("a fresh interpreter was started for the third run", False, "not started")
        return
    out, err = FRESH.communicate(timeout=900)
    out = out.decode("utf-8", "replace").strip()
    check("the fresh interpreter exited 0", FRESH.returncode == 0, (err or b"").decode()[-300:])
    h3 = out.split("|")[1:] if out.startswith("M7A-HASH") else None
    check("the fresh interpreter reported the three vectors", h3 is not None and len(h3) == 3, out)
    STATE["runs"]["run 3, fresh interpreter"] = tuple(h3 or ())
    check("run 3 (fresh interpreter) equals run 1 and run 2 byte for byte",
          h3 is not None and tuple(h3) == tuple(h1) == tuple(h2),
          "\n       run1 %s\n       run2 %s\n       run3 %s" % (h1, h2, h3))
    check("the tile vector hash is not the image vector hash (three distinct vectors are hashed)",
          len(set(h1)) == 3, str(h1))
    # the artifacts on disk are byte-compared too: a re-run that changes a file is not deterministic
    STATE["csv_sha"] = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()
    check("the rewritten tile_census.csv is byte-identical to the standalone census run",
          STATE["csv_sha"] == FIRST_CSV_SHA, "%s vs %s" % (STATE["csv_sha"][:16],
                                                           FIRST_CSV_SHA[:16]))


def test_every_check_passed():
    """Pytest bridge, and it must stay the last test_ function in the file."""
    check("all checks above passed", not FAIL, "%d of %d failed" % (len(FAIL), TOTAL + 1))
    assert not FAIL, "%d check(s) failed: %s" % (len(FAIL), ", ".join(FAIL[:6]))


# ------------------------------------------------------------------ console summary
def print_summary():
    j = strict_json(JSON_PATH)
    asm = state()["asm"]
    counts = asm["status_counts"]
    line = "=" * 78
    print("\n" + line)
    print("M7-A TASK 3 CENSUS - 1946 materialised tiles, four frozen spatial statistics")
    print(line)
    print("outcome (3) valid finite tile (all four keys) : %d" % j["successful_tiles"])
    print("outcome (2) valid tile, degenerate statistic  : %d with any NaN key"
          % j["tiles_with_any_nan"])
    print("    of which whole-tile TileDegenerate        : %d" % j["degenerate_tiles"])
    print("    of which NaN in one or more keys only     : %d"
          % counts.get(d.STATUS_NAN_KEYS, 0))
    print("outcome (1) invalid call (aborts the pass)    : %d" % j["invalid_calls"])
    print("status counts                                 : %s" % dict(counts))
    print("images / soils (train soils)                  : %d / %d (%d train)"
          % (len(asm["image"]), len(asm["soil"]),
             len(asm["q"][asm["q"].split == "train"].sample_id.unique())))
    print("split tiles                                   : %s"
          % asm["q"].split.value_counts().to_dict())
    print("-" * 78)
    print("%-4s %10s %10s %12s %12s %12s %12s %12s"
          % ("key", "nan", "nonfinite", "min", "p5", "median", "p95", "max"))
    for k in sf.SPATIAL:
        s = j["per_key"][k]
        print("%-4s %10d %10d %12.6f %12.6f %12.6f %12.6f %12.6f"
              % (k, s["nan_count"], s["nonfinite_count"], s["min"], s["p5"], s["median"],
                 s["p95"], s["max"]))
    print("-" * 78)
    print("EXTREME OUTLIERS (report only; nothing clamped, transformed or winsorised)")
    print("  two rules: |x-median|/MAD (absolute) and the same rule on log|x| (multiplicative,")
    print("  symmetric). A zero MAD or an inapplicable log rule is reported as itself.")
    for level, field in (("tile", "keys"), ("soil", "soil_level")):
        for k in sf.SPATIAL:
            b = j["extreme_outlier"][field][k]
            print("  %-5s %-3s MAD %-10s abs>10MAD %4s  max|z| %8s | log>10MAD %4s  max|z| %8s "
                  "(log %s) | max|x| %s  whole dist > 1e6: %s"
                  % (level, k, num(b["mad"]), b["count_beyond_10_mad"], num(b["max_abs_z"], "%.3f"),
                     b["count_beyond_10_mad_log"], num(b["max_abs_z_log"], "%.3f"),
                     str(b["log_of"]), num(b["max_abs_value"]), b["whole_distribution_above_1e6"]))
    print("finiteness, measured on the arrays before anything was serialised")
    for level in ("tile", "image", "soil"):
        f = j["finiteness"][level]
        print("  %-6s cells %6d  NaN %d  inf %d  non-finite %d  (%s)"
              % (level, f["cells"], f["nan_cells"], f["inf_cells"], f["non_finite_cells"],
                 j["finiteness"]["meaning"][:48]))
    print("assembled soil table, min / median / max (what a rank-3 fit would standardise)")
    for k in sf.SPATIAL:
        s = j["soil_percentiles"][k]
        print("  %-3s n %2d  %s  %s  %s" % (k, s["n_finite"], num(s["min"]), num(s["median"]),
                                            num(s["max"])))
    print("warnings during the real pass, by stage                : %d %s"
          % (j["warnings_count"], j["warnings_by_stage"]))
    print("errors recorded (any would abort)                     : %d" % j["errors_count"])
    print("non-finite tiles / cells                             : %d / %d"
          % (j["non_finite_tiles"], j["finiteness"]["tile"]["non_finite_cells"]))
    for w in asm["warnings"][:10]:
        print("  WARNING tile %s | %s: %s" % (w["tile_path"], w["category"], w["message"]))
    for e in asm["errors"][:10]:
        print("  ERROR tile %s | %s: %s" % (e["tile_path"], e["kind"], e["reason"]))
    if asm["degenerate_examples"]:
        print("degenerate / NaN-key reasons (first %d):" % len(asm["degenerate_examples"]))
        for r in asm["degenerate_examples"]:
            print("  %s [%s] %s" % (r["tile_path"].split("/")[-1][:58], r["status"],
                                    r["reason"][:110]))
    else:
        print("degenerate / NaN-key reasons              : none on this corpus")
    print("determinism, three vectors per run")
    print("  vectors: 1 = %d tiles, 2 = %d images, 3 = %d soils"
          % (j["total_tiles"], j["images"], j["soils"]))
    runs = STATE.get("runs") or {"this pass, reused above": tuple(asm["hashes"])}
    for label, hs in runs.items():
        print("  %-24s %s" % (label, " | ".join(h[:16] for h in hs)))
    identical = len({tuple(h) for h in runs.values()}) == 1 if len(runs) > 1 else None
    if identical is not None:
        print("  all three runs byte-identical : %s" % identical)
    print("  tile_census.csv sha256       : %s" % STATE.get("csv_sha", "not yet hashed"))
    print("artifacts published (all four are byte-comparable between runs)")
    for name, p in sorted(d.artifact_paths().items()):
        print("  %-24s %8d bytes  %s" % (name, p.stat().st_size, p.relative_to(ROOT)))
    print(line)


def main():
    global FRESH
    wanted = [a.lower() for a in sys.argv[1:]]
    if not wanted:
        snippet = "import m7a_data as d; print('M7A-HASH', *d.determinism_hashes(), sep='|')"
        FRESH = subprocess.Popen([sys.executable, "-c", snippet], cwd=str(HERE),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fns = []
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            fns.append((obj.__code__.co_firstlineno, name, obj))
    fns.sort()
    if wanted:
        fns = [f for f in fns if any(w in f[1].lower() for w in wanted)]
        print("FILTERED RUN: %d of the file's tests selected by %s" % (len(fns), wanted))
    for _, name, fn in fns:
        try:
            fn()
        except Exception as exc:  # a test that crashes is a failure, not an aborted run
            check("%s raised %s" % (name, type(exc).__name__), False, str(exc)[:90])
    if not wanted:
        print_summary()
    print("\n%d checks, %d failed" % (TOTAL, len(FAIL)))
    for f in FAIL:
        print("  - " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
