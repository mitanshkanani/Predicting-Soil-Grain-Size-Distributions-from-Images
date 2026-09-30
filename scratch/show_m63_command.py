"""Third pass: dump the transcript lines around the original Model 6-3 alias output so the exact
command (and its printed columns) can be read verbatim. Read-only, scratch output only.

Run:  python scratch/show_m63_command.py [start] [end]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SRC = Path("C:/Users/mitansh/.qoder/projects/C--Users-mitansh-Desktop-KAGGLE-HACKATHONS-Predicting-Soil-Grain-Size-Distributions-from-Images/c6165df8-94a3-4ed4-8272-b719ac03472c.jsonl")
OUT = Path(__file__).resolve().parents[1] / "scratch" / "_m63_command.txt"

start = int(sys.argv[1]) if len(sys.argv) > 1 else 11500
end = int(sys.argv[2]) if len(sys.argv) > 2 else 11575


def render(node, depth=0):
    """Flatten a message into readable text, keeping tool inputs verbatim."""
    lines = []
    if isinstance(node, dict):
        t = node.get("type")
        if t == "tool_use":
            lines.append("\n[TOOL_USE %s] input:" % node.get("name"))
            inp = node.get("input")
            lines.append(json.dumps(inp, indent=1, ensure_ascii=False) if not isinstance(inp, str)
                         else inp)
        elif t == "tool_result":
            lines.append("\n[TOOL_RESULT]")
            c = node.get("content")
            if isinstance(c, list):
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        lines.append(part["text"])
                    else:
                        lines.append(json.dumps(part, ensure_ascii=False)[:2000])
            else:
                lines.append(str(c))
        elif t == "text":
            lines.append("\n[TEXT]\n" + node.get("text", ""))
        else:
            for k, v in node.items():
                if k in ("type",):
                    continue
                lines.extend(render(v, depth + 1))
    elif isinstance(node, list):
        for v in node:
            lines.extend(render(v, depth + 1))
    elif isinstance(node, str):
        lines.append(node)
    return lines


with OUT.open("w", encoding="utf-8") as fh:
    for i, line in enumerate(SRC.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not (start <= i <= end):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        fh.write("\n########## line %d type=%s ##########\n" % (i, obj.get("type")))
        fh.write("\n".join(render(obj))[:20000])
print("wrote %s for lines %d-%d" % (OUT, start, end))
