"""Dump the verbatim Bash command text of specific transcript lines, wrapped at 100 columns so no
long line is omitted. Read-only; scratch output only.

Run:  python scratch/show_m63_lines.py 11551 11557
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

SRC = Path("C:/Users/mitansh/.qoder/projects/C--Users-mitansh-Desktop-KAGGLE-HACKATHONS-Predicting-Soil-Grain-Size-Distributions-from-Images/c6165df8-94a3-4ed4-8272-b719ac03472c.jsonl")
OUT = Path(__file__).resolve().parents[1] / "scratch" / "_m63_commands_verbatim.txt"
wanted = {int(a) for a in sys.argv[1:]}


def walk(node):
    if isinstance(node, dict):
        if node.get("type") == "tool_use":
            yield node
        else:
            for v in node.values():
                yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


with OUT.open("w", encoding="utf-8") as fh:
    for i, line in enumerate(SRC.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if i not in wanted:
            continue
        obj = json.loads(line)
        for tu in walk(obj):
            inp = tu.get("input") or {}
            fh.write("\n########## transcript line %d  tool=%s ##########\n" % (i, tu.get("name")))
            cmd = inp.get("command") or inp.get("prompt") or json.dumps(inp, ensure_ascii=False)
            fh.write(cmd if isinstance(cmd, str) else json.dumps(cmd, ensure_ascii=False))
            fh.write("\n")
print("wrote %s" % OUT)
