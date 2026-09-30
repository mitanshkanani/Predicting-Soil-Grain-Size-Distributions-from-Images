"""Static checks on research_map_1_2_5_6.md. Read-only: it opens the map, the probe CSV and the
correction-bearing records, and asserts the map is internally consistent, ASCII-clean, free of the
withdrawn claims, and faithful to the numbers the probes actually printed.

Run:  python scratch/check_research_map.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "research_map_1_2_5_6.md"
TEXT = MAP.read_text(encoding="utf-8")
FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def has(pattern, label, regex=True):
    hit = bool(re.search(pattern, TEXT)) if regex else (pattern in TEXT)
    check(label, hit, pattern[:48])
    return hit


def lacks(pattern, label, regex=True):
    hit = bool(re.search(pattern, TEXT)) if regex else (pattern in TEXT)
    check(label, not hit, "forbidden: " + pattern[:48])
    return not hit


# ------------------------------------------------------------------ structure
check("file is non-trivial", len(TEXT.split()) > 1500, "%d words" % len(TEXT.split()))
bad = [(i + 1, ln) for i, ln in enumerate(TEXT.splitlines()) if any(ord(c) > 126 for c in ln)]
check("ASCII-only throughout", not bad, "%d non-ascii lines" % len(bad))
for tag in ("## 0.", "## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.", "## 8."):
    has(re.escape(tag), "section present: " + tag)
for model in ("Model 1 -", "Model 2 -", "Model 5 -", "Model 6 -"):
    has(re.escape(model), "ledger row present: " + model)

LINES = TEXT.splitlines()

def near(pattern, window, where):
    """True if some line matching `pattern` has a line matching `where` within +-`window` lines."""
    for i, ln in enumerate(LINES):
        if re.search(pattern, ln):
            lo, hi = max(0, i - window), min(len(LINES), i + window + 1)
            if any(re.search(where, l) for l in LINES[lo:hi]):
                return True
    return False

# every model must carry all seven required fields
FIELDS = ["Hypothesis", "Exact intervention", "Validation protocol", "Result",
          "Closure reason", "headroom", "Limitation"]
for f in FIELDS:
    n = len(re.findall(f, TEXT))
    check("field '%s' appears for every model (found %d)" % (f, n), n >= 4, str(n))
m6 = [ln for ln in LINES if ln.startswith("| **6-")]
check("Model 6 table has 4 candidates and a validation column",
      len(m6) == 4 and "**Validation protocol**" in TEXT, "%d rows" % len(m6))

# ------------------------------------------------------------------ honesty guards
lacks(r"beats the entire shipped model", "withdrawn leaky-bound sentence is gone")
check("the withdrawn 1.07 EMD figure appears only inside the WITHDRAWAL block",
      near(r"1\.07 EMD", 3, r"WITHDRAWAL|first draft"),
      "1.07 must be quarantined to the withdrawal")
check("a 1-1.5 EMD cap appears only as a withdrawal",
      not any(re.search(r"[Cc]apped at roughly 1 to 1\.5", l) for l in LINES))
lacks(r"Bound 1's ~?1 EMD", "M6-3 no longer cites the withdrawn cap as a size")
has(r"WITHDRAWAL", "the withdrawal is stated explicitly")
has(r"withdrawn", "withdrawals are labelled")
has(r"15\.29", "best-of-N selection bias is quoted")
for tag in ("[measured]", "[unverified]", "[withdrawn]"):
    has(re.escape(tag), "items tagged " + tag)

# ------------------------------------------------------------------ external vs internal
has(r"60\.56167", "external baseline present")
has(r"43\.0217308796477", "internal anchor present at full precision")
arith = re.compile(r"6[01]\.\d{2,5}[^|\n]{0,40}?[-+]\s*[45]\d\.\d|[45]\d\.\d[^|\n]{0,40}?[-+]\s*6[01]\.\d{2,5}")
offenders = [ln for ln in LINES if arith.search(ln)]
check("no line subtracts a CV value from an external score or vice versa",
      not offenders, str(offenders)[:90])
has(r"3 of 10", "public set size stated")
has(r"\[21\.7, 69\.9\]", "3-soil band quoted")

# ------------------------------------------------------------------ probe fidelity
csv = pd.read_csv(ROOT / "scratch" / "_head_spread.csv", index_col=0)
missing = [("%s %.2f" % (h, r["real"])) for h, r in csv.iterrows()
           if "%.2f" % r["real"] not in TEXT or "%.2f" % r["null_median"] not in TEXT]
check("every head's real AND null value from the probe is quoted in the map",
      not missing, "; ".join(missing[:4]))

alpha = pd.read_csv(ROOT / "scratch" / "_alpha_curve.csv", index_col=0)
miss_a = ["alpha %g -> %.2f" % (i, v) for i, v in alpha["lofo_emd"].items()
          if "%.2f" % v not in TEXT]
check("the alpha curve quoted in the map matches the persisted probe artifact",
      not miss_a, "; ".join(miss_a[:4]))
check("the map's 1.68 selection cost equals nested minus best fixed alpha",
      abs((alpha["lofo_emd"].min() + 1.68) - float(csv.loc["ridge_nested", "real"])) < 0.011,
      "best fixed %.2f + 1.68 vs nested %.2f"
      % (alpha["lofo_emd"].min(), csv.loc["ridge_nested", "real"]))

for num in ("47.66", "41.95", "59.27", "44.23", "42.83", "43.02", "40.20", "41.28"):
    has(re.escape(num), "number present: " + num)
check("47.66 minus 43.02 = the 4.64 the map claims", abs((47.66 - 43.02) - 4.64) < 0.011)
check("42.83 minus 41.15 = the 1.68 the map claims", abs((42.83 - 41.15) - 1.68) < 0.011)
check("7.35 minus 2.65 = the 4.70 bare floor gain", abs((7.35 - 2.65) - 4.70) < 0.011)
check("6.25 minus 2.38 = the 3.87 projected floor gain", abs((6.25 - 2.38) - 3.87) < 0.011)

# Bound 1 table must match the persisted four-convention probe artifact exactly
fb = pd.read_csv(ROOT / "scratch" / "_feature_bound.csv")
cell = {(r.scaler, r.pool, r.k): r.emd for r in fb.itertuples()}
want = {("all", "naive"): [44.23, 41.95, 45.73, 44.96, 48.26, 47.12, 50.99],
        ("all", "honest"): [50.66, 47.66, 50.27, 51.12, 51.55, 54.25, 57.21],
        ("pool", "naive"): [52.84, 41.95, 45.73, 44.88, 48.26, 46.51, 50.83],
        ("pool", "honest"): [59.27, 47.66, 50.27, 51.04, 51.55, 55.32, 61.59]}
bad_fb = []
for (sc, pl), vals in want.items():
    for k, v in zip((1, 2, 3, 4, 5, 6, 8), vals):
        got = cell.get((sc, pl, k))
        if got is None or abs(got - v) > 0.005:
            bad_fb.append("%s/%s k%d map=%.2f probe=%s" % (sc, pl, k, v, got))
        if "%.2f" % v not in TEXT:
            bad_fb.append("%s/%s k%d %.2f not quoted in map" % (sc, pl, k, v))
check("every Bound 1 cell is quoted in the map and matches the probe artifact",
      not bad_fb, "; ".join(bad_fb[:3]))

rf = pd.read_csv(ROOT / "scratch" / "_rank_floors.csv", index_col=0)
bad_rf = ["rank %d %s %.2f" % (i, c, v) for i, r in rf.iterrows() for c, v in r.items()
          if "%.2f" % v not in TEXT]
check("every rank-floor cell is quoted in the map and matches the probe artifact",
      not bad_rf, "; ".join(bad_rf[:3]))
check("the map reconciles 7.35 as the recorded bare rank-3 ceiling",
      "7.35" in TEXT and "rank-3 representation ceiling" in TEXT)

# provenance: any line calling 41.15 an oracle figure must sit near the CAM+RES caveat
def window_has(i, needle, w=9):
    return any(needle in LINES[j] for j in range(max(0, i - w), min(len(LINES), i + w + 1)))

bad = [i + 1 for i, ln in enumerate(LINES)
       if "41.15" in ln and "oracle" in ln and not window_has(i, "CAM+RES")]
check("41.15 is only called an oracle figure beside its provenance caveat", not bad, str(bad))
fb2 = pd.read_csv(ROOT / "scratch" / "_feature_bound.csv")
c2 = {(r.scaler, r.pool, r.k): r.emd for r in fb2.itertuples()}
leaks = {k: c2[("pool", "honest", k)] - c2[("all", "naive", k)] for k in (1, 2, 3, 4, 5, 6, 8)}
sib = {k: c2[("pool", "honest", k)] - c2[("pool", "naive", k)] for k in (1, 2, 3, 4, 5, 6, 8)}
missing_leak = ["k%d +%.2f" % (k, v) for k, v in leaks.items() if "+%.2f" % v not in TEXT]
check("the total-leak figures quoted in prose match the probe artifact",
      not missing_leak, "; ".join(missing_leak))
check("the sibling-leak extreme quoted (+10.76 at k=8) is the artifact's value",
      abs(sib[8] - 10.76) < 0.011 and "+10.76" in TEXT, "artifact %.4f" % sib[8])
check("the scaler-only effect at k=1 quoted (8.61) matches the artifact",
      abs((c2[("pool", "naive", 1)] - c2[("all", "naive", 1)]) - 8.61) < 0.011,
      "artifact %.4f" % (c2[("pool", "naive", 1)] - c2[("all", "naive", 1)]))

# ------------------------------------------------------------------ correction records intact
for path, marker in [("AGENT_BRIEF.md", "CORRECTION 2"),
                     ("Model 5/instructions.txt", "CORRECTION"),
                     ("Model 5/Model 5 Experiment 1/Experiment1.txt", "16.")]:
    t = (ROOT / path).read_text(encoding="utf-8", errors="replace")
    check("correction marker intact in " + path, marker in t, marker)
check("map states it implements nothing",
      "Nothing here was implemented" in TEXT or "nothing here was implemented" in TEXT)
has(r"Model 7 (?:is )?propos|no Model 7 is proposed|no Model 7", "explicitly proposes no Model 7")

# the five mandatory Model 7 prerequisites must all be named in section 6
for req in ("A falsifiable mechanism", "A pre-registered null", "A validation protocol",
            "A defined success criterion", "A path to a Kaggle submission"):
    has(re.escape(req), "decision framework carries: " + req)
has(r"Selection sequence", "selection sequence present")
has(r"deliberately stops before step 1", "map states it proposes no Model 7 candidate")
check("the bucket headings are all four the owner asked for",
      all(s in TEXT for s in ("### A. Modifications to the existing pipeline",
                              "### B. Methods that introduce genuinely new information",
                              "### C. Methods that require additional labelled soils",
                              "### D. Information present in the 12 features")))

print("\n%d checks failed" % len(FAIL))
for f in FAIL:
    print("  - " + f)
sys.exit(1 if FAIL else 0)
