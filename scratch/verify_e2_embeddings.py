"""Plan Task 2, step 2-4: prove E2's embeddings are present and are E2's, and write the
committed provenance manifest.

    python scratch/verify_e2_embeddings.py

All the suspicion lives in Model 2/Model 2 Experiment 3/embedding_store.py, which the E3
notebook imports for the same calls, so this check and the real run cannot diverge. This
script only drives it and prints the result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Model 2" / "Model 2 Experiment 3"
sys.path.insert(0, str(EXP))

import embedding_store as es                                    # noqa: E402

STORE = EXP / "embeddings"
META = ROOT / "data" / "processed_meta"
TILE_PX = 256
EMBED_SEED = 20260930          # pinned by E2's cache filenames; changing it invalidates E3


def main() -> int:
    print("=" * 78)
    print("Model 2 E3 - embedding provenance check")
    print("=" * 78)
    if not META.exists():
        print(f"ABORT: {META} not found. E3 reads the frozen processed manifests.")
        return 2
    hash_ = sorted(set(pd.read_csv(META / "manifest_images.csv").config_hash.astype(str)))
    if len(hash_) != 1:
        print(f"ABORT: ambiguous config_hash {hash_}")
        return 2
    CONFIG_HASH = hash_[0]
    print(f"store        : {STORE}")
    print(f"tile_px      : {TILE_PX}   config_hash {CONFIG_HASH}   embed_seed {EMBED_SEED}")
    print(f"required     : {len(es.REQUIRED)} arrays "
          f"({len(es.REQUIRED) * 2} files counting .paths.npy)")

    if not STORE.is_dir():
        print(f"\nFAIL: {STORE} does not exist.")
        print("  E3 has no torch path and cannot embed these itself. Retrieve E2's Kaggle")
        print("  cache from /kaggle/working/.cache - model_plan_m2e3.md section 10 lists")
        print("  three ways. Do NOT quietly convert E3 into a GPU run instead; that changes")
        print("  the experiment's verification profile and is the user's decision.")
        return 1

    rows, fails = es.verify_all(STORE, META, TILE_PX, CONFIG_HASH, EMBED_SEED)
    total_mb = sum(r["bytes"] for r in rows) / 1e6
    for r in rows:
        print(f"  ok   {r['file'][:58]:58s} {r['rows']:5d}x{r['cols']} "
              f"{r['dtype']:8s} {r['bytes'] / 1e6:5.2f} MB  min|row| {r['min_abs_row_norm']:.3f}")
    for f in fails:
        print(f"  FAIL {f}")

    if not fails:
        man = EXP / "E2_embedding_manifest.csv"
        es.manifest_frame(rows).to_csv(man, index=False)
        print(f"\nwrote {man.relative_to(ROOT)} ({len(rows)} rows, committed; the .npy "
              f"themselves stay gitignored)")
        print(f"EMBEDDING PROVENANCE: PASSED "
              f"({len(rows)} arrays + {len(rows)} path files, {total_mb:.1f} MB)")
        return 0
    print(f"EMBEDDING PROVENANCE: FAILED ({len(fails)} problems)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
