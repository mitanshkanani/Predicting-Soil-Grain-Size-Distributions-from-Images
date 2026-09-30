"""Second pass on scratch/_m63_recovery.txt: keep only the strings that carry the Model 6-3
figures TOGETHER (the original probe's own printed output), so the command that produced them can
be identified. Read-only; writes one scratch file."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scratch" / "_m63_recovery.txt"
OUT = ROOT / "scratch" / "_m63_recovery_narrow.txt"

text = SRC.read_text(encoding="utf-8", errors="replace")
blocks = text.split("\n=== ")
keep = []
for b in blocks[1:]:
    header, _, body = b.partition("===\n")
    if "65.7" in body and ("28.1" in body or "1.65" in body or "1.37" in body):
        keep.append((header, body))
    elif ("neighbour" in body and ("1.65" in body or "1.58" in body)
          and "alias" in body.lower()):
        keep.append((header, body))

with OUT.open("w", encoding="utf-8") as fh:
    fh.write("%d narrow blocks of %d\n" % (len(keep), len(blocks) - 1))
    for header, body in keep:
        fh.write("\n=== %s ===\n%s\n" % (header, body[:8000]))
print("wrote %s : %d narrow blocks" % (OUT, len(keep)))
for header, body in keep[:12]:
    first = [ln for ln in body.splitlines() if "65.7" in ln or "1.65" in ln][:3]
    print("\n--- %s" % header)
    for ln in first:
        print("    " + ln[:200])
