"""Static checks on model7_preregistrations.md: the file must be self-consistent, must match the
verified boundary numbers in scratch/_m7c_boundary.json, must not revive withdrawn or closed lines,
and must contain every element the owner required before any run.

Run:  python scratch/check_m7_prereg.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILE = ROOT / "model7_preregistrations.md"
SPEC_FILE = ROOT / "Model 7" / "Approach A" / "model_spec_m7a.md"
CONTRACT_FILE = ROOT / "Model 7" / "Approach A" / "instructions.txt"
MODEL_FILE = ROOT / "Model 7" / "instructions.txt"
TEXT = ROOT_FILE.read_text(encoding="utf-8")
SPEC = SPEC_FILE.read_text(encoding="utf-8") if SPEC_FILE.exists() else ""
CONTRACT = CONTRACT_FILE.read_text(encoding="utf-8") if CONTRACT_FILE.exists() else ""
MODELREC = MODEL_FILE.read_text(encoding="utf-8") if MODEL_FILE.exists() else ""
# M7-A moved into its experiment folder on registration; checks look at both
DOC = "\n".join([TEXT, SPEC, CONTRACT, MODELREC])
LINES = DOC.splitlines()
FACTS = json.loads((ROOT / "scratch" / "_m7c_boundary.json").read_text(encoding="utf-8"))
FAIL = []


def check(name, ok, detail=""):
    print("%-6s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + detail) if detail else ""))
    if not ok:
        FAIL.append(name)


def has(pat, label):
    ok = bool(re.search(pat, DOC))
    check(label, ok, pat[:44])
    return ok


def lacks(pat, label):
    ok = not re.search(pat, DOC)
    check(label, ok, ("forbidden " + pat[:40]) if not ok else "")
    return ok


# ---------------------------------------------------------------- structure
bad = [i + 1 for i, l in enumerate(LINES) if any(ord(c) > 126 for c in l)]
check("ASCII-only", not bad, "%d lines" % len(bad))
check("substantial", len(TEXT.split()) > 1200, "%d words" % len(TEXT.split()))
for s in ("## 1. Boundary rulings", "## 2. M7-A", "## 3. M7-C", "## 4. Side by side"):
    has(re.escape(s), "section present: " + s)

# ---------------------------------------------------------------- owner's required M7-A elements
for pat, label in [
    (r"A1 \|", "M7-A defines statistic A1"),
    (r"A2 \|", "M7-A defines statistic A2"),
    (r"A3 \|", "M7-A defines statistic A3"),
    (r"A4 \|", "M7-A defines statistic A4"),
    (r"K = 4", "M7-A fixes K explicitly"),
    (r"Random-feature placebo \(P-RAND\)", "M7-A random-feature placebo"),
    (r"Column-shuffled placebo \(P-SHUF\)", "M7-A column-shuffled placebo"),
    (r"Best-of-K selection null \(P-SEL\)", "M7-A best-of-K null"),
    (r"Feature-count control", "M7-A feature-count control"),
    (r"Alias-separability gate", "M7-A alias-separability gate"),
    (r"Leave-one-CV-family-out", "M7-A family-honest CV stated"),
    (r"Pass criteria", "M7-A pass criteria"),
    (r"Estimated headroom", "M7-A headroom labelled as estimate"),
    (r"Expected failure mode", "M7-A expected failure mode"),
]:
    has(pat, label)

# ---------------------------------------------------------------- owner's required M7-C elements
for pat, label in [
    (r"SEL-1SE", "M7-C names the exact selector"),
    (r"largest\*\* \(most regularised\) alpha", "M7-C fixes the tie-break direction"),
    (r"Explicitly out of scope", "M7-C excludes the sweep winner from the primary arm"),
    (r"Identity control", "M7-C identity control"),
    (r"Degenerate-parameter check", "M7-C inertness check"),
    (r"One-parameter null", "M7-C one-parameter null"),
    (r"Regularisation-artefact test", "M7-C artefact clause"),
    (r"no alpha clipping", "M7-C boundary rule reused from M5E1"),
]:
    has(pat, label)

# ---------------------------------------------------------------- boundary rulings
# ---------------------------------------------------------------- quote integrity
has(r"SUSPENDED, not killed by rule", "M7-T ruling answers the question rather than assuming it")
has(r"\*\*Answer: no, it does not\.\*\*", "M7-T: the closure scope is answered explicitly")
has(r"Ruling recorded, not assumed", "M7-T: the disposition is stated as a record, not a choice")
has(r"over-read all three records", "M7-T: the self-correction is disclosed in the document")
# Every passage the file introduces as a quotation from AGENT_BRIEF.md must appear there verbatim
# once line wrapping and blockquote markers are normalised. This is the check that catches a quote
# that was rewritten from memory rather than copied.
def flat(s):
    """Unwrap hard-wrapped markdown into a single line, applied identically to quotation and source.

    This is unwrapping, not a tolerance: a line break inside a paragraph is formatting, and a line
    that ENDS in a hyphen is a broken word, so it rejoins without a space ("pre-" + "registered").
    No character is otherwise removed, so a real misquote still fails.
    """
    s = re.sub(r"-\s*\n\s*", "-", s)
    return re.sub(r"\s+", " ", s).strip()


brief = flat((ROOT / "AGENT_BRIEF.md").read_text(encoding="utf-8"))
SOURCES = {
    "AGENT_BRIEF.md": brief,
    "Model 5/Model 5 Experiment 1/model_spec_m5e1.md": flat(
        (ROOT / "Model 5" / "Model 5 Experiment 1" / "model_spec_m5e1.md").read_text(
            encoding="utf-8")),
    "Model 5/instructions.txt": flat(
        (ROOT / "Model 5" / "instructions.txt").read_text(encoding="utf-8")),
}
quoted_blocks = re.findall(
    r"^> `(?:AGENT_BRIEF\.md|Model 5/Model 5 Experiment 1/model_spec_m5e1\.md|Model 5/instructions\.txt)`[^\n]*(?:\n>[^\n]*)*",
    TEXT, flags=re.M)
check("at least three source-quoted blocks present", len(quoted_blocks) >= 3,
      "%d found" % len(quoted_blocks))
bad_quotes = []
n_spans = 0
for b in quoted_blocks:
    key = re.match(r"^> `([^`]+)`", b).group(1)
    src = SOURCES.get(key, "")
    body = re.sub(r"(?m)^> ?", "", b)
    flat_block = flat(body)
    spans = re.findall(r"\*\*(.+?)\*\*", flat_block)
    spans += [s for s in re.findall(r'"([^"]{12,})"', flat_block) if "**" not in s]
    for inner in spans:
        quote = flat(inner).strip('"').strip("*")
        if len(quote) < 12:
            continue
        n_spans += 1
        if quote not in src:
            bad_quotes.append("%s: %s" % (key, quote[:56]))
check("every quoted or bolded passage is verbatim in the file it names", not bad_quotes,
      "; ".join(bad_quotes[:3]))
check("the verifier really inspected quotations (a vacuous pass is a failure)", n_spans >= 4,
      "%d spans inspected" % n_spans)
check("the transductive-use clause is quoted", "transductive use of the unlabeled test images"
      in flat(TEXT.replace("\n> ", " ")))
check("the adapting-to-test-distribution clause is quoted",
      "nor by adapting the features to the test distribution" in DOC.replace("\n> ", " "))
has(r"COLLIDES, recommend kill", "M7-E collision reported rather than routed around")
has(r"cross-tile", "M7-A records that the cross-tile variant is dropped")
has(r"ORACLE REGRET", "M7-C states 1.68 is oracle regret, not an achievable gain")
has(r"upper bound on the prize, not an expected gain", "M7-C ceiling language")

# ---------------------------------------------------------------- numbers must match the verified artifact
def quoted(x, label, fmt="%.2f"):
    s = fmt % x
    check("%s (%s) appears in the prereg" % (label, s), s in DOC)


quoted(FACTS["nested_honest"], "honest nested score")
quoted(FACTS["oracle_fixed"], "oracle best-fixed score")
quoted(FACTS["oracle_regret"], "oracle regret")
quoted(FACTS["adaptive_mean"], "adaptive arm score")
check("regret identity holds", abs(FACTS["nested_honest"] - FACTS["oracle_fixed"]
                                  - FACTS["oracle_regret"]) < 0.005)
check("fold-instability counts quoted (%d of %d)" % (FACTS["n_folds_choosing_oracle"],
      FACTS["n_folds"]),
      ("%d of %d" % (FACTS["n_folds_choosing_oracle"], FACTS["n_folds"])) in DOC)
check("adaptive penalty is below the frozen grid floor and that is stated",
      FACTS["adaptive_alpha_max"] < FACTS["grid_floor"]
      and "below the frozen grid" in TEXT,
      "max %.4f vs grid floor %.2f" % (FACTS["adaptive_alpha_max"], FACTS["grid_floor"]))

# ---------------------------------------------------------------- scale discipline
has(r"60\.56167", "external baseline present")
has(r"43\.0217308796477", "internal anchor at full precision")
check("anchor is required to be reproduced in both candidates",
      DOC.count("43.0217308796477") >= 4, "%d mentions" % DOC.count("43.0217308796477"))
arith = re.compile(r"6[01]\.\d{2,5}[^|\n]{0,40}?[-+]\s*[34]\d\.\d|[34]\d\.\d[^|\n]{0,40}?[-+]\s*6[01]\.\d{2,5}")
off = [l[:80] for l in LINES if arith.search(l)]
check("no line subtracts a CV value from an external score", not off, str(off)[:90])

# ---------------------------------------------------------------- withdrawn / closed lines
lacks(r"two-neighbour lookup beats", "leaky 1-NN claim not revived")
lacks(r"cap(?:ped)? at roughly 1 to 1\.5", "withdrawn 1-1.5 cap not revived")
lacks(r"resolution-matched|fit on test-blurred", "cancelled resolution-matched line not proposed")
lacks(r"granulometry as a feature|particle count", "refuted granulometry not revived")
check("the register says no candidate has been implemented or fitted yet",
      re.search(r"No M7 candidate has been\s+implemented", DOC) is not None)
check("the register records the owner's decision to run M7-A first",
      "run M7-A first" in DOC or "Register M7-A first" in DOC)

# ---------------------------------------------------------------- leakage prohibitions
for word in ("camera", "ppm", "EXIF", "ICC", "site", "sample-id"):
    check("prohibition on %s as an input is stated" % word,
          word.lower() in TEXT.lower())

print("\n%d checks failed" % len(FAIL))
for f in FAIL:
    print("  - " + f)
sys.exit(1 if FAIL else 0)
