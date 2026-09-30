"""Read-only recovery of the Model 6-3 forensic probe source from this project's own session
transcript. The map quotes figures (65.7 / 28.1 / 1.65 / 1.58 / 61.2 / 1.95 / 1.37) that no
committed script produces, so the only possible evidence for the ORIGINAL rule is the code that
actually printed them. This scans the recorded transcript for the commands and outputs that carry
those figures and dumps them verbatim to a text file for reading.

Writes nothing but this scratch dump; touches no experiment artifact.

Run:  python scratch/recover_m63_source.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = Path("C:/Users/mitansh/.qoder/projects")
OUT = ROOT / "scratch" / "_m63_recovery.txt"

MARKERS = ("65.7", "28.1", "1.65", "1.58", "61.2", "1.95", "1.37", "104.73", "125.4", "1.08",
           "neighbour", "alias", "max_d", "maxd", "|t|", "scatter")


def strings(node):
    """Yield every string in a nested JSON structure, with a path label."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from strings(v)


def main() -> None:
    files = sorted(PROJECTS.glob("*/*.jsonl")) + sorted(PROJECTS.glob("*/*/subagents/*.jsonl"))
    hits = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print("skip %s: %s" % (f, exc))
            continue
        if "neighbour" not in text and "alias" not in text:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if not any(m in line for m in MARKERS):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for s in strings(obj):
                if len(s) < 20:
                    continue
                if not any(m in s for m in MARKERS):
                    continue
                # keep only strings that look like code or printed output carrying the figures
                if not any(m in s for m in ("65.7", "28.1", "1.65", "1.95", "1.37", "104.73",
                                            "125.4", "neighbour", "alias_pair", "max_d")):
                    continue
                hits.append((f.name, lineno, obj.get("type", "?"), s))
    hits.sort(key=lambda h: (h[0], h[1]))
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("%d candidate strings\n" % len(hits))
        for name, lineno, typ, s in hits:
            fh.write("\n=== %s line %d type=%s len=%d ===\n" % (name, lineno, typ, len(s)))
            fh.write(s[:6000])
            fh.write("\n")
    print("wrote %s : %d hits" % (OUT, len(hits)))


if __name__ == "__main__":
    main()
