"""build_m11_vlm_nb.py - emit Model 11/Experiment 1/vlm_inference.ipynb (Task 3, step S2).

The notebook is BLIND by construction: it reads only neutral 12-hex mosaic ids and PNG pixels.
It contains no label, sample-name, camera or split information anywhere, including comments, and
it writes exactly one artifact, vlm_raw.jsonl.

Cell sources live here so each one can be compile-checked before shipping and so the resolver can
be executed against a fabricated mount tree by check_m11_vlm_resolver.py. The prompt string is
injected with its sha256 computed at build time; the notebook asserts that hash at runtime, so a
later edit to the prompt cell fails loudly instead of passing silently.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "Model 11" / "Experiment 1"
NB = EXP / "vlm_inference.ipynb"

# ---- frozen configuration (owner-ruled 2026-10-02) -----------------------------
PROMPT = (
    "You are a geotechnical engineer. The image is a mosaic of up to 6 square close-up photos of "
    "the surface of ONE soil sample. Each square is exactly 56 mm wide. Estimate for the WHOLE "
    "sample (not only the visible surface): d10, d50, d90 in mm (diameters below which 10, 50, 90 "
    "percent of the dry mass lies) and the mass percentages of fines (<0.063 mm), sand "
    "(0.063-2 mm) and gravel (>2 mm), summing to 100. Return ONLY JSON with keys d10_mm, d50_mm, "
    "d90_mm, fines_pct, sand_pct, gravel_pct."
)
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
SEED = 42
MAX_NEW_TOKENS = 200
NATIVE_AREA = 768 * 512          # 393216 px, the mosaic geometry
MIN_PIXELS = 3136                # 4*28*28: effectively "no floor" at this resolution
MAX_PIXELS = NATIVE_AREA         # caps the lattice result at native area: never upscale, never downscale
SMOKE_N = 3
PARSE_FAIL_LIMIT = 0.05

C = {}

C["K0"] = '''# Cell 0 - frozen configuration. Nothing here may be edited after any output exists.
import hashlib, json, os, sys

MODEL_ID   = __MODEL_ID__
REVISION   = __REVISION__      # pinned HF commit; verified against the loaded model in cell 3
PROMPT     = __PROMPT__
PROMPT_SHA = __PROMPT_SHA__
SEED       = __SEED__
MAX_NEW_TOKENS = __MAX_NEW_TOKENS__
MIN_PIXELS = __MIN_PIXELS__
MAX_PIXELS = __MAX_PIXELS__    # == native mosaic area, so the image is never scaled
SMOKE_N    = __SMOKE_N__
PARSE_FAIL_LIMIT = __PARSE_FAIL_LIMIT__
MANIFEST_NAME = "mosaic_manifest.csv"

assert hashlib.sha256(PROMPT.encode("utf-8")).hexdigest() == PROMPT_SHA, \\
    "PROMPT cell has been edited after freeze - refusing to run"
print("prompt sha256 verified:", PROMPT_SHA[:16], "...")
print("model:", MODEL_ID, "@", REVISION[:12])
print("seed", SEED, "| do_sample=False | max_new_tokens", MAX_NEW_TOKENS,
      "| min_pixels", MIN_PIXELS, "| max_pixels", MAX_PIXELS)
'''

C["K1"] = '''# Cell 1 - locate the attached mosaic dataset at any mount depth, then verify every file by hash.
# Kaggle mounts datasets three levels deep, so a single-level assumption fails with the dataset
# correctly attached. On failure every base tried is printed.
import hashlib
from pathlib import Path

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

def find_input(known_file):
    tried, bases = [], [Path("/kaggle/input"), Path.cwd()]
    for b in bases:
        if not b.exists():
            tried.append("absent base %s" % b)
            continue
        for hit in sorted(b.glob("**/" + known_file)):
            root = hit.parent
            if (root / "mosaics").exists():
                print("[ok] dataset root:", root)
                return root
            tried.append("manifest at %s but no mosaics/ beside it" % root)
    raise RuntimeError("could not find %s with a mosaics/ folder. Tried: %s"
                       % (known_file, " | ".join(tried)))

DATA = find_input(MANIFEST_NAME)
import pandas as pd
man = pd.read_csv(DATA / MANIFEST_NAME).sort_values("mosaic_id").reset_index(drop=True)
print("mosaics listed:", len(man), "| columns:", list(man.columns))

bad = []
for _, r in man.iterrows():
    p = DATA / r.mosaic_path
    if not p.exists():
        bad.append((r.mosaic_id, "MISSING"))
    elif _sha(p) != r.sha256_mosaic:
        bad.append((r.mosaic_id, "HASH MISMATCH"))
print("integrity: %d files verified, %d problems" % (len(man), len(bad)))
if bad:
    for b in bad[:10]:
        print("   ", b)
    raise RuntimeError("uploaded dataset does not match mosaic_manifest.csv - refusing to run")
IDS = man.mosaic_id.tolist()
assert len(IDS) == len(set(IDS)) == 162, "expected 162 unique mosaic ids, got %d" % len(IDS)
print("blind order fixed: lexicographic by mosaic id (first %d used for the smoke test)" % SMOKE_N)
'''

C["K2"] = '''# Cell 2 - processor, and a MEASURED probe of what geometry it actually uses.
# The model's vision patch lattice is 14 px x a 2x2 merge = 28 px. 768 and 512 are not multiples
# of 28, so the image cannot be consumed at literally 768x512; the processor rounds each side to
# the nearest lattice point. min_pixels/max_pixels are set so that the ONLY geometric change is
# that rounding: no downscale and no upscale. This cell prints the true result rather than
# assuming it, and stops if the area changed by more than the lattice allows.
import torch
from PIL import Image
from transformers import AutoProcessor

print("torch", torch.__version__)
import transformers
print("transformers", transformers.__version__)

proc = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION,
                                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
ip = proc.image_processor
print("image_processor min_pixels=%s max_pixels=%s"
      % (getattr(ip, "min_pixels", "?"), getattr(ip, "max_pixels", "?")))

probe = Image.open(DATA / man.mosaic_path.iloc[0]).convert("RGB")
print("input mosaic size (WxH):", probe.size)
inp = proc(images=[probe], text=["placeholder"], return_tensors="pt")
grid = inp.get("image_grid_thw")
print("image_grid_thw (t,h,w in patches):", None if grid is None else grid.tolist())
if grid is not None:
    f = 28
    h_px, w_px = int(grid[0, 1]) * f, int(grid[0, 2]) * f
    print("effective consumed pixels: %dx%d = %d px vs native %d px (area ratio %.4f)"
          % (w_px, h_px, w_px * h_px, MAX_PIXELS, (w_px * h_px) / MAX_PIXELS))
    assert w_px * h_px <= MAX_PIXELS * 1.02, "processor UPSCALED the image - stop and report"
    assert w_px * h_px >= MAX_PIXELS * 0.90, "processor DOWNSCALED more than the lattice allows"
    print("[ok] geometry is lattice rounding only; identical for all 162 mosaics")
'''

C["K3"] = '''# Cell 3 - load the model: fp16, sdpa attention, sharded across both GPUs.
# T4 has no bf16 and no flash-attn, hence fp16 + sdpa. If the model class is unavailable in the
# installed transformers, this STOPS with the version printed - the model is never swapped.
import torch, transformers
from transformers import AutoModelForVision2Seq

print("transformers", transformers.__version__)
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLClass
    print("[ok] native Qwen2.5-VL class available")
except ImportError as e:
    print("*** Qwen2_5_VLForConditionalGeneration not importable:", e)
    print("*** STOP: upgrade transformers. Do not substitute another model class.")
    raise

print("cuda devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print("   cuda:%d %s  %.1f GB" % (i, p.name, p.total_memory / 1e9))

model = VLClass.from_pretrained(
    MODEL_ID,
    revision=REVISION,
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
    device_map="auto",
)
model.eval()
loaded = getattr(model.config, "_commit_hash", None)
print("loaded config._commit_hash:", loaded)
if loaded and loaded.lower()[:12] != REVISION.lower()[:12]:
    print("*** WARNING: loaded revision differs from the pinned revision")
print("dtype:", next(model.parameters()).dtype, "| class:", type(model).__name__)
print("device map (parameter -> cuda device):",
      {k: str(v) for k, v in sorted(getattr(model, "hf_device_map", {}).items())
       if isinstance(v, int)} or getattr(model, "hf_device_map", {}))
print("[ok] model loaded in fp16 with sdpa attention across the available GPUs")
'''

C["K4"] = '''# Cell 4 - one inference, and a parser that is tolerant of formatting but not of content.
# The parser accepts a JSON object wrapped in markdown fences or embedded in prose, because that
# is a formatting artefact of chat models, not a wrong answer. It never alters the prompt.
import json, re, torch

KEYS = ["d10_mm", "d50_mm", "d90_mm", "fines_pct", "sand_pct", "gravel_pct"]

def _candidates(text):
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\\s*", "", t)
    t = re.sub(r"\\s*```$", "", t)
    out = [t]
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start:i + 1])
    return out

def parse_reply(text):
    """Returns (parse_ok, parsed_dict_or_None, note). Structure only - values are recorded, not judged."""
    for cand in _candidates(text):
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        missing = [k for k in KEYS if k not in obj]
        if missing:
            continue
        vals, note = {}, []
        ok = True
        for k in KEYS:
            try:
                v = float(obj[k])
            except Exception:
                ok = False
                break
            if v != v:                      # NaN
                ok = False
                note.append("nan:" + k)
                break
            vals[k] = v
        if not ok:
            continue
        note.append("sum_pct=%.4f" % (vals["fines_pct"] + vals["sand_pct"] + vals["gravel_pct"]))
        note.append("monotone_d=%s" % (vals["d10_mm"] <= vals["d50_mm"] <= vals["d90_mm"]))
        return True, vals, ";".join(note)
    return False, None, "no JSON object with all 6 keys"

@torch.no_grad()
def ask(img):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inp = proc(images=[img], text=[text], return_tensors="pt")
    inp = inp.to(model.device)
    gen = model.generate(**inp, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                         pad_token_id=processor_pad_id)
    new = gen[0][inp["input_ids"].shape[1]:]
    return proc.tokenizer.decode(new, skip_special_tokens=True), int(new.shape[0])

processor_pad_id = proc.tokenizer.pad_token_id
if processor_pad_id is None:
    processor_pad_id = proc.tokenizer.eos_token_id
print("pad/eos token id:", processor_pad_id)
'''

C["K5"] = '''# Cell 5 - SMOKE TEST on the first 3 ids in the fixed blind order.
# Purpose is plumbing only: does the model load, produce output, and does the JSON parse.
# The prompt is NOT edited on the basis of what this prints. If output is empty, all-NaN or the
# parse fails on all 3, this STOPS and reports; the model, dtype and prompt are not swapped.
from PIL import Image

torch.manual_seed(SEED)
smoke = IDS[:SMOKE_N]
ok_n, notes = 0, []
for mid in smoke:
    img = Image.open(DATA / "mosaics" / (mid + ".png")).convert("RGB")
    raw, ntok = ask(img)
    good, parsed, note = parse_reply(raw)
    ok_n += int(good)
    notes.append(note)
    print("-" * 78)
    print("id", mid, "| tokens", ntok, "| parse_ok", good, "|", note)
    print("RAW >>>", raw[:700].replace(chr(10), " / "))
print("=" * 78)
print("smoke parse_ok: %d/%d" % (ok_n, SMOKE_N))
if ok_n == 0:
    raise RuntimeError("SMOKE TEST FAILED - no parsable JSON from any of %d mosaics. "
                       "Stop and report; do not alter model, dtype or prompt." % SMOKE_N)
if any("nan" in str(x) for x in notes):
    print("WARNING: NaN observed in smoke output - stop and report before the full run")
print("[ok] smoke test passed; proceeding to the full run")
'''

C["K6"] = '''# Cell 6 - the full run. Writes vlm_raw.jsonl and nothing else.
# One image at a time (no batching): padding-free single-item forwards are the most reproducible,
# and the line is flushed immediately so a crash leaves a usable partial record.
from PIL import Image
import time

OUT = "/kaggle/working/vlm_raw.jsonl"
torch.manual_seed(SEED)
t0 = time.time()
fails = retried_fired = retried_used = 0
with open(OUT, "w") as fh:
    for j, mid in enumerate(IDS):
        img = Image.open(DATA / "mosaics" / (mid + ".png")).convert("RGB")
        raw, ntok = ask(img)
        good, parsed, note = parse_reply(raw)
        retried = False
        if not good:
            retried = True
            retried_fired += 1
            raw2, ntok2 = ask(img)
            good2, parsed2, note2 = parse_reply(raw2)
            if good2:
                retried_used += 1
                raw, ntok, good, parsed, note = raw2, ntok2, True, parsed2, note2
        if not good:
            fails += 1
        fh.write(json.dumps({"mosaic_id": mid, "raw_reply": raw, "n_tokens": ntok,
                             "parse_ok": bool(good), "parsed": parsed, "parse_note": note,
                             "retry_fired": bool(retried),
                             "retry_recovered": bool(retried and good)}, sort_keys=True) + "\\n")
        fh.flush()
        if (j + 1) % 10 == 0 or j + 1 == len(IDS):
            print("  %3d/%d  elapsed %6.1fs  parse_fail %d  eta %5.1fmin"
                  % (j + 1, len(IDS), time.time() - t0, fails,
                     (time.time() - t0) / (j + 1) * (len(IDS) - j - 1) / 60.0))

n = len(IDS)
print("=" * 78)
print("rows written: %d | parse failures: %d (%.2f%%) | retries fired: %d | recovered: %d"
      % (n, fails, 100.0 * fails / n, retried_fired, retried_used))
print("GATE (plan.md T3): more than %.0f%% parse failures means the gate FAILS -> %s"
      % (100 * PARSE_FAIL_LIMIT, "FAIL" if fails / n > PARSE_FAIL_LIMIT else "within limit"))
print("elapsed %.1f min" % ((time.time() - t0) / 60.0))
'''

C["K7"] = '''# Cell 7 - final audit of the single artifact.
import hashlib, json

rows = [json.loads(l) for l in open(OUT)]
ids = [r["mosaic_id"] for r in rows]
blob = open(OUT, "rb").read()
print("file:", OUT, "| bytes:", len(blob))
print("sha256:", hashlib.sha256(blob).hexdigest())
print("lines:", len(rows), "| unique ids:", len(set(ids)),
      "| ids match manifest order:", ids == IDS)
print("parse_ok:", sum(r["parse_ok"] for r in rows), "/", len(rows))
print("retry_fired:", sum(r["retry_fired"] for r in rows),
      "| retry_recovered:", sum(r["retry_recovered"] for r in rows))
print("tokens: min %d median %g max %d"
      % (min(r["n_tokens"] for r in rows),
         sorted(r["n_tokens"] for r in rows)[len(rows) // 2],
         max(r["n_tokens"] for r in rows)))
print("empty replies:", sum(1 for r in rows if not r["raw_reply"].strip()))
print("fields per row:", sorted(rows[0].keys()))
print("NOTE: this file carries neutral ids only. Mapping back to anything is done offline.")
'''

MD = {
    "K0": ("# Model 11 / Experiment 1 / Task 3 — blind VLM inference\n\n"
           "Reads `mosaic_manifest.csv` + `mosaics/*.png`. **Contains no label, sample-name, "
           "camera or split information.** Writes exactly one artifact: `vlm_raw.jsonl`.\n\n"
           "**Requires internet ON** (weights download from Hugging Face, revision pinned).\n"
           "Accelerator: **GPU T4 x2**. Model: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898…`, "
           "fp16, `attn_implementation=\"sdpa\"`, `device_map=\"auto\"`, greedy decoding, "
           "seed 42, `max_new_tokens=200`.\n"),
    "K1": "## 1. Locate the dataset and verify every file against the manifest",
    "K2": "## 2. Processor, and a measured probe of the geometry it actually consumes",
    "K3": "## 3. Load the model (fp16 + sdpa, sharded across both T4s)",
    "K4": "## 4. Inference and parser",
    "K5": "## 5. SMOKE TEST — 3 mosaics, plumbing only, prompt never edited on this evidence",
    "K6": "## 6. Full run over all 162 mosaics",
    "K7": "## 7. Final audit",
}


def build() -> None:
    subs = {
        "__MODEL_ID__": json.dumps(MODEL_ID), "__REVISION__": json.dumps(REVISION),
        "__PROMPT__": json.dumps(PROMPT), "__PROMPT_SHA__": json.dumps(PROMPT_SHA),
        "__SEED__": str(SEED), "__MAX_NEW_TOKENS__": str(MAX_NEW_TOKENS),
        "__MIN_PIXELS__": str(MIN_PIXELS), "__MAX_PIXELS__": str(MAX_PIXELS),
        "__SMOKE_N__": str(SMOKE_N), "__PARSE_FAIL_LIMIT__": str(PARSE_FAIL_LIMIT),
    }
    cells, srcs = [], {}
    for k in ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7"):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [MD[k]]})
        s = C[k]
        for tok, val in subs.items():
            s = s.replace(tok, val)
        left = [t for t in subs if t in s]
        assert not left, "unsubstituted tokens %r in %s" % (left, k)
        compile(s, k, "exec")
        srcs[k] = s
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "outputs": [], "source": s.splitlines(keepends=True)})
    flat = "\n".join(srcs[k] for k in ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7"))
    compile(flat, "flattened", "exec")
    (EXP / "vlm_inference_cells.py").write_text(flat, encoding="ascii")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.11"},
                       "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

    low = flat.lower()
    banned = ["sample_id", "motorola", "samsung", "iphone", "camera", "cv_group",
              "manifest_samples", "training_labels", "d50_mm_true", "logd50"]
    hits = [b for b in banned if b in low]
    print("notebook written:", NB.name, "| cells:", len(cells),
          "| code cells:", len(srcs))
    print("pinned revision:", REVISION)
    print("prompt sha256:", PROMPT_SHA)
    print("identity-token scan of notebook code:", hits if hits else "CLEAN")


if __name__ == "__main__":
    build()
