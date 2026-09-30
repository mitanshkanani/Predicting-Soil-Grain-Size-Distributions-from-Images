"""Build the scoped review package for the I1-I6 (plus M2, M8) fix round.

Extracts the pre-fix text of both permitted files from the Task 3 review package, diffs it against
the current working tree with unified context, and writes one file a reviewer can read in a single
pass: the finding-by-finding statement, the stat summary, and the diff.
"""
import difflib
import io
import os

PKG = ".superpowers/sdd/model_plan_m7a/task-3-diff.txt"
OUT = ".superpowers/sdd/model_plan_m7a/task-3-fix-diff.txt"
FILES = ["Model 7/Approach A/m7a_data.py", "Model 7/Approach A/check_m7a_data.py"]

lines = io.open(PKG, encoding="utf-8").read().splitlines()


def recover(target):
    start = next(i for i, line in enumerate(lines) if line.startswith("## " + target))
    nxt = [j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")]
    body = lines[start:nxt[0] if nxt else len(lines)]
    hunk = next(x for x, line in enumerate(body) if line.startswith("@@"))
    return [line[1:] for line in body[hunk + 1:] if line.startswith("+")]


parts = ["""# M7-A Task 3 FIX ROUND - scoped review package

Baseline: the Task 3 implementation as reviewed (0 Critical, 6 Important: I1-I6, plus the two
queued audit items M2 and M8). Owner authorisation: apply I1-I6 inside m7a_data.py and
check_m7a_data.py only, test-first, and the two audit improvements because they stay in Task 3
scope. Nothing else may change: not instructions.txt, not model_spec_m7a.md, not the A1-A4
definitions, not the frozen constants, not gates G0-G7, not the thresholds, not the controls, not
any historical record, and no CV, pre-gate, fit, prediction or submission.

Findings under review, as the reviewer stated them:
  I1  the missing-tile abort named the PREVIOUS tile (ctx["current"] set after path.exists())
  I2  the outlier rule was one-sided for a positive key; mad == 0 reported "0 outliers"
  I3  the warning sink wrapped only the tile loop, not the aggregation stages; the sink schema leg
      checked an empty list so it could never fail
  I4  a NaN key without TileDegenerate was tallied as degeneracy although the code called it a bug
  I5  no check asserted tile-level finiteness; _json_safe rewrote non-finite to null, so the
      strict-JSON leg was a property of the writer
  I6  six checks pinned m7a_data's source text, and one measurably shaped the implementation
  M2  non_finite meant a tile count in one place and an inf count in another
  M8  the assembled image/soil design existed only in memory, so it could not be audited offline

Evidence the controller produced (reproduce, do not take on trust). The run below used the exact
bytes this package diffs, and the source hashes are stated so the match is checkable:
  m7a_data.py         sha256 36c86672d3da555a2f86de1a0bd9f8d0e66becd22d1a92ed68beeedd9598f95f
  check_m7a_data.py   sha256 f263b4bb04ac2692b2fbb766611db107931481100b2c8af24df5261c6564a726
  python "Model 7/Approach A/check_spatial_features.py"   510 checks, 0 failed, exit 0
  python "Model 7/Approach A/check_m7a_data.py"           324 checks, 0 failed, exit 0
  python scratch/recompute_m7a_census.py                   169 comparisons, 0 disagreed, exit 0
  (logs: Model 7/Approach A/scratch/sf_final3.txt, data_FINAL.txt, recompute_FINAL.txt;
  scratch/bytes_under_test.sha records the source bytes and was re-verified after the run)
The vector hashes and tile_census.csv sha256 are IDENTICAL to the pre-fix Task 3 run, which is how
"an audit-layer fix moved no measurement" is demonstrated rather than asserted:
    tile  84e2b2c23e100b95e1b2b15e97933b81bde808b86602908676f002b97c54399e
    image 778990e14303980a83d688ae739efb5a57fe548d2fdc1b72a9f9718ed87184b0
    soil  41dbbd7e729cee22183cacfcc6409eaa5d6037a390e7afe41265a5df75df3eb6
    csv   a9ca9e0103c13086e1d2d94ba87051ce3590755cb4e86dc818b9813b7c4299cb
Two new legs were themselves tested by mutation: with ctx=ctx deleted from assembly(), the I3
call-site leg fails 3 checks and the injected RuntimeWarning reaches stderr; the pre-fix
implementation fails 20 of the new legs. Both are recorded in the report.
"""]
for f in FILES:
    old = recover(f)
    new = io.open(f, encoding="utf-8").read().splitlines()
    diff = list(difflib.unified_diff(old, new, fromfile=f + " (Task 3 as reviewed)",
                                     tofile=f + " (after I1-I6, M2, M8)", lineterm="", n=3))
    add = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    parts.append("\n## %s\n--- %d lines removed, +++ %d added, %d -> %d lines\n"
                 % (f, rem, add, len(old), len(new)))
    parts.append("\n".join(diff))
io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(parts) + "\n")
print("wrote %s (%d bytes)" % (OUT, os.path.getsize(OUT)))
