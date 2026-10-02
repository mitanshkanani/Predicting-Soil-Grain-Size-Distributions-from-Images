"""verify_local.py - torch-free verification of everything Model 9 / Exp 1 ships to Kaggle.

The CNN needs a GPU and cannot run here. But every gate that does NOT need torch is checked
locally, reading from the ACTUAL final_kaggle_upload_m9e1.zip so it is the shipped bytes that are
verified, not a hopeful copy on disk:

  G0  the shipped handcrafted OOF reproduces the oracle alpha=3 anchor 41.1475 (head coherent
      with the 55.80591 submission) and the OOF / label row order match.
  G1  config_hash of the zipped manifest_images == 010f44c36c74.
  G2  the EMD metric reproduces the host trivial baseline 100.31 on the shipped labels.
  G4  submission contract on the 55.8 handcrafted CSV (10 rows, 11 cols, monotone, last=100,
      in range, 0 NaN) AND the manifest submission_id join reproduces the 10 board ids 1:1.
  tiles  the archive holds exactly 1976 materialized t256 tiles.

G3 (monotone-CDF head) and G5 (family-CV leak) require torch and are asserted INSIDE the notebook
(cells C2 and D1). The blend weight is selected on Kaggle once the CNN OOF exists; here we only
confirm its inputs are coherent.

An internal CV mean on 24 soils is NOT a Kaggle score and is never subtracted from one.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ZIP  = ROOT / "final_kaggle_upload_m9e1.zip"

CONFIG_HASH_EXPECTED = "010f44c36c74"
ORACLE_ANCHOR = 41.1475158656982032
SUP   = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
DCOLS = ["d_" + s for s in SUP]
DIAMS = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200]
DL    = np.log10(np.array(DIAMS, float))
_TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def emd_pair(p, t):
    return float(_TRAP(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))


def mean_emd(P, T):
    return float(np.mean([emd_pair(P[i], T[i]) for i in range(len(T))]))


def project_np(Fm):
    Fm = np.clip(np.atleast_2d(np.asarray(Fm, float)), 0, 100)
    Fm = np.maximum.accumulate(Fm, axis=1)
    Fm[:, -1] = 100.0
    return Fm


def _read_csv(z, name):
    with z.open(name) as f:
        return pd.read_csv(io.BytesIO(f.read()))


def main() -> int:
    assert ZIP.exists(), "ZIP not built yet - run build_dataset_zip.py first: %s" % ZIP
    fails = []

    def check(cond, label):
        print(("  [ok] " if cond else "  [FAIL] ") + label)
        if not cond:
            fails.append(label)

    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        assert z.testzip() is None, "corrupt archive"

        print("--- tiles ---")
        t256 = [n for n in names if n.startswith("data/tiles/") and n.endswith("__t256.png")]
        check(len(t256) == 1976, "1976 materialized t256 tiles present (got %d)" % len(t256))

        print("--- G1 config_hash ---")
        imgs = _read_csv(z, "data/processed_meta/manifest_images.csv")
        hashes = set(imgs.config_hash.astype(str))
        check(hashes == {CONFIG_HASH_EXPECTED}, "config_hash == %s (got %s)"
              % (CONFIG_HASH_EXPECTED, hashes))

        print("--- G0 handcrafted OOF anchor + row-order coherence ---")
        oof = _read_csv(z, "handcrafted_oof_train.csv")
        lab = _read_csv(z, "train_labels_matrix.csv")
        oof["sample_id"] = oof.sample_id.astype(str)
        lab["sample_id"] = lab.sample_id.astype(str)
        check(oof.sample_id.tolist() == lab.sample_id.tolist(),
              "OOF and label row order identical (24 soils)")
        check(len(oof) == 24, "24 train soils in OOF (got %d)" % len(oof))
        HAND = oof.set_index("sample_id").loc[oof.sample_id, DCOLS].to_numpy(float)
        Y    = lab.set_index("sample_id").loc[oof.sample_id, DCOLS].to_numpy(float)
        anchor = mean_emd(HAND, Y)
        check(abs(anchor - ORACLE_ANCHOR) < 1e-9,
              "OOF mean EMD %.10f == oracle anchor %.10f (gap %.2e)"
              % (anchor, ORACLE_ANCHOR, abs(anchor - ORACLE_ANCHOR)))
        check(bool(np.all(np.diff(Y, axis=1) >= -1e-9)) and bool(np.allclose(Y[:, -1], 100.0)),
              "shipped labels monotone and end at 100")

        print("--- G2 trivial baseline reproduces host 100.31 ---")
        trivial = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55,
                            63.64, 72.73, 81.82, 90.91, 100.0])
        triv = float(np.mean([emd_pair(trivial, y) for y in Y]))
        check(abs(triv - 100.31) < 0.1, "trivial baseline %.3f ~ 100.31" % triv)

        print("--- G4 submission contract (55.8 handcrafted CSV) + id join ---")
        ss = _read_csv(z, "data/sample_submission.csv")
        ss["sample_id"] = ss.sample_id.astype(str)
        ss_ids = ss.sample_id.tolist()
        check(len(ss_ids) == 10, "sample_submission has 10 rows (got %d)" % len(ss_ids))
        check(list(ss.columns) == ["sample_id"] + SUP, "sample_submission columns exact")

        samples = _read_csv(z, "data/processed_meta/manifest_samples.csv")
        samples["sample_id"] = samples.sample_id.astype(str)
        samples["submission_id"] = samples.submission_id.astype(str)
        sub_of = dict(zip(samples.sample_id, samples.submission_id))
        test_internal = samples[samples.split == "test"].sample_id.tolist() \
            if "split" in samples.columns else list(sub_of)
        mapped = [sub_of[s] for s in test_internal if sub_of.get(s) in set(ss_ids)]
        check(sorted(set(mapped)) == sorted(ss_ids),
              "manifest submission_id join covers all 10 board ids 1:1 (got %d)" % len(set(mapped)))

        hand = _read_csv(z, "Submission_Model8_E2_E2A.csv")
        hand["sample_id"] = hand.sample_id.astype(str)
        check(sorted(hand.sample_id) == sorted(ss_ids), "55.8 CSV ids == board ids")
        H = hand.set_index("sample_id").loc[ss_ids, SUP].to_numpy(float)
        check(H.shape == (10, 11), "55.8 CSV shape 10x11 (got %s)" % (H.shape,))
        check(bool(np.all(np.diff(H, axis=1) >= -1e-9)), "55.8 rows monotone")
        check(bool(np.allclose(H[:, -1], 100.0)), "55.8 last column == 100")
        check(bool(np.isfinite(H).all()) and bool((H >= -1e-6).all()) and bool((H <= 100 + 1e-6).all()),
              "55.8 values finite and in [0,100]")

        print("--- blend-input coherence (weight chosen on Kaggle once CNN OOF exists) ---")
        # identity sanity: projecting + scoring the hand OOF against itself at any w on hand = anchor
        check(abs(mean_emd(project_np(HAND), Y) - anchor) < 1e-9,
              "projection is a no-op on an already-valid curve (path sane)")
        # a worked blend with a deliberately-worse synthetic partner must never beat w=0 here
        synth = np.tile(trivial, (24, 1))                       # a bad 'CNN' stand-in
        grid = [round(i / 20, 2) for i in range(21)]
        best = min(mean_emd(project_np(w * synth + (1 - w) * HAND), Y) for w in grid)
        check(abs(best - anchor) < 1e-6, "with a worse partner the blend collapses to w=0 (=anchor)")

    print("\n%s" % ("ALL LOCAL GATES PASS" if not fails else "FAILURES: %s" % fails))
    print("(G3 monotone head + G5 family-CV leak are asserted inside the notebook on Kaggle.)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
