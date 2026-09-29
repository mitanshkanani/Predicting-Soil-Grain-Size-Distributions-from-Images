"""Prove the E3 embedding loader is not vacuous, before the user spends time downloading
80 MB of real arrays.

    python scratch/check_e3_loader.py

E3 has no mock mode and no torch path, so the loader in embedding_store.py is the only
thing standing between a mis-keyed cache and a confident wrong answer. A guard nobody has
seen fire is not a guard. This file fabricates fixtures by renaming E2's LOCAL mock
embedding arrays -- same shapes, same dtype, same tile-path lists as the real dinov2 arrays
would have -- and then attacks the loader five ways it would fail in practice.

Nothing here writes into Model 2/Model 2 Experiment 3/embeddings/. Fixtures live in a
temporary folder under scratch/ and are deleted, so a fabricated array can never be
mistaken for E2's real one by a later run.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Model 2" / "Model 2 Experiment 3"
META = ROOT / "data" / "processed_meta"
sys.path.insert(0, str(EXP))

import embedding_store as es                                    # noqa: E402

TILE_PX, CONFIG_HASH, SEED = 256, "010f44c36c74", 20260930
SRC = sorted((ROOT / "Model 2" / "Model 2 Experiment 2" / "local mock dry-run" / ".cache")
             .glob("embed_mock_*.npy"))
FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def build_fixture(dst: Path) -> None:
    """Rename E2's mock arrays into the names E2's dinov2 / vit_random arrays carry."""
    dst.mkdir(parents=True, exist_ok=True)
    made = 0
    for p in SRC:
        stem = p.name[len("embed_mock_"):]                  # c256_256_<hash>_<seed>[_view].npy
        for backend in ("dinov2", "vit_random"):
            out = dst / f"embed_{backend}_{stem}"
            shutil.copyfile(p, out)
            made += 1
    print(f"  fixtures: {made} files from {len(SRC)} mock sources "
          f"(mock arrays share the real ones' shape, dtype and tile order)")


def run(store: Path):
    return es.verify_all(store, META, TILE_PX, CONFIG_HASH, SEED)


def main() -> int:
    print("=" * 78)
    print("E3 embedding loader self-test (fabricated fixtures, then five attacks)")
    print("=" * 78)
    if not SRC:
        print("ABORT: no mock embeddings found to fabricate fixtures from.")
        return 2
    tmp = Path(tempfile.mkdtemp(prefix="_e3fix_", dir=str(ROOT / "scratch")))
    try:
        store = tmp / "embeddings"
        build_fixture(store)

        print("\nbaseline: every required array present")
        rows, fails = run(store)
        check("all 15 required arrays load clean", not fails, f"{len(fails)} failures")
        check("15 arrays reported", len(rows) == 15, str(len(rows)))
        check("required set is dinov2x4crops + vit_random c256, 3 views",
              len(es.REQUIRED) == 15 and {b for b, _, _ in es.REQUIRED}
              == {"dinov2", "vit_random"}, str(sorted(set(b for b, _, _ in es.REQUIRED))))
        real_rows = {r["rows"] for r in rows if r["view"] == "real"}
        res_rows = {r["rows"] for r in rows if r["view"] != "real"}
        check("real view arrays all have the same row count", len(real_rows) == 1, str(real_rows))
        check("res view arrays all have the same row count", len(res_rows) == 1, str(res_rows))
        check("res arrays are strictly shorter than real arrays (train tiles only)",
              len(res_rows) == 1 and len(real_rows) == 1
              and max(res_rows) < min(real_rows)
              and max(res_rows) == 1541 and min(real_rows) == 1976,
              f"real {real_rows} vs res {res_rows}")
        check("expected tile lists recomputed live: 1976 / 1541",
              len(es.expected_tile_list(META, TILE_PX, "real")) == 1976
              and len(es.expected_tile_list(META, TILE_PX, "res14")) == 1541)
        man = es.manifest_frame(rows)
        check("manifest frame has 11 provenance columns", man.shape == (15, 11),
              str(man.shape))
        check("sha256 recorded per array", man.sha256.str.len().eq(64).all())

        def expect_fail(label, corrupt, repair, needle):
            corrupt()
            _rows, _fails = run(store)
            hit = any(needle.lower() in f.lower() for f in _fails)
            check(f"{label} is caught", bool(_fails) and hit,
                  f"{len(_fails)} failure(s): {(_fails[0][:78] if _fails else 'NONE')}")
            repair()
            _r2, _f2 = run(store)
            check(f"{label} repair restores a clean load", not _f2,
                  f"{len(_f2)} left")

        print("\nattack 1: one array deleted")
        victim = store / es.cache_name("dinov2", 128, "real", TILE_PX, CONFIG_HASH, SEED)
        keep = victim.read_bytes()
        expect_fail("missing array", victim.unlink, lambda: victim.write_bytes(keep),
                    "missing")

        print("\nattack 2: tile paths reordered (same members, wrong pairing)")
        pp = store / es.cache_name("vit_random", 256, "real", TILE_PX,
                                   CONFIG_HASH, SEED).replace(".npy", ".paths.npy")
        orig = np.load(pp, allow_pickle=True)
        swapped = list(orig.tolist())
        swapped[0], swapped[-1] = swapped[-1], swapped[0]

        def corrupt_paths():
            np.save(pp, np.array(swapped, dtype=object), allow_pickle=True)

        def repair_paths():
            np.save(pp, np.array(list(orig.tolist()), dtype=object), allow_pickle=True)

        expect_fail("reordered paths", corrupt_paths, repair_paths, "different order")

        print("\nattack 3: dtype silently downcast to float32")
        q = store / es.cache_name("dinov2", 64, "real", TILE_PX, CONFIG_HASH, SEED)
        keep32 = q.read_bytes()
        expect_fail("float32 array",
                    lambda: np.save(q, np.load(q).astype(np.float32)),
                    lambda: q.write_bytes(keep32), "float64")

        print("\nattack 4: one collapsed embedding dimension")
        q4 = store / es.cache_name("dinov2", 32, "real", TILE_PX, CONFIG_HASH, SEED)
        keep4 = q4.read_bytes()

        def corrupt_col():
            b = np.load(q4)
            b[:, 7] = 0.5
            np.save(q4, b)

        expect_fail("zero-variance column", corrupt_col,
                    lambda: q4.write_bytes(keep4), "zero-variance")

        print("\nattack 5: array trimmed by one row")
        q5 = store / es.cache_name("dinov2", 256, "res16", TILE_PX, CONFIG_HASH, SEED)
        keep5 = q5.read_bytes()
        expect_fail("short array",
                    lambda: np.save(q5, np.load(q5)[:-1]),
                    lambda: q5.write_bytes(keep5), "rows for")

        print("\nrecovery: all five repairs applied, run clean again")
        rows2, fails2 = run(store)
        check("loader passes after repairs", not fails2, f"{len(fails2)} left")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"\ncleaned up {tmp.name}; nothing written into the E3 embeddings folder")

    print("-" * 78)
    if FAILS:
        print(f"E3 LOADER SELF-TEST: FAILED ({len(FAILS)})")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("E3 LOADER SELF-TEST: PASSED  (5/5 corruptions caught)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
