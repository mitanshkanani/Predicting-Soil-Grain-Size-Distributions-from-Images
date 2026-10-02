"""build_notebook.py - emit Model9_Experiment1_kaggle.ipynb and compile-check every cell.

Local python has no torch, so the CNN cannot be RUN here. But compile() parses each cell without
importing anything, so every syntax / indentation error is caught locally before the notebook
ever reaches Kaggle. Cross-cell name references are not resolved by compile (that is intended -
compile validates syntax only).

Each cell is a raw triple-single-quoted string so the cells' own triple-double-quote docstrings
nest cleanly. No cell may itself contain a triple-single-quote.
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CELLS = [
# ============================================================ A1
r'''"""Model 9 / Experiment 1 - a camera-invariant CNN, blended with the 55.80591 head.

Runs on Kaggle with 2x T4 GPUs (set Accelerator = GPU T4 x2 in the notebook settings, and
Internet = ON so the pretrained backbone can download). Trains a pretrained backbone end-to-end,
per 256px tile, on the EXACT competition EMD metric, with camera colour-domain randomisation;
aggregates tiles to soils; blends with the frozen handcrafted head's shipped predictions.

Everything tunable lives in CFG and nowhere else. See plan.md for the pre-registration and the
honest split between what is measurable internally and what needs a submission.
"""
from dataclasses import dataclass
from pathlib import Path

SEED = 20261002
CONFIG_HASH_EXPECTED = "010f44c36c74"

@dataclass
class CFG:
    # data (frozen at the values every prior upload used)
    tile_size_px: int = 256
    min_tile_soil_fraction: float = 0.50
    img_size: int = 224
    # backbone
    backbone: str = "resnet34"       # timm name; torchvision fallback covers resnet*
    pretrained: bool = True
    # optimisation
    batch_size: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.10
    # camera colour-domain randomisation (training only) - THE lever
    chan_gain_lo: float = 0.80       # per-channel log-uniform gain; spans the measured camera gap
    chan_gain_hi: float = 1.20
    jitter_bcs: float = 0.30         # brightness / contrast / saturation
    jitter_hue: float = 0.08
    rrc_scale_lo: float = 0.65       # random-resized-crop lower scale
    # cross-validation (epoch choice + CNN OOF for the blend)
    do_cv: bool = True
    cv_epochs: int = 24              # probe up to here; the min-EMD epoch is chosen
    # final model
    n_seed_ensemble: int = 3
    epochs_fallback: int = 12        # used only if do_cv is False
    # inference
    tta: bool = True
    # blend weights swept on the 24 soils (0.00..1.00 step .05)
    blend_grid: tuple = tuple(round(i / 20, 2) for i in range(21))
    out_dir: str = "/kaggle/working"

cfg = CFG()
print("CFG:", cfg)
''',
# ============================================================ A2
r'''"""Cell A2 - imports, environment, GPU, numpy-version-safe trapezoid."""
import os, sys, json, time, math, random, platform
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# numpy 2.0 renamed trapz -> trapezoid; Kaggle may ship either. Pick whichever exists.
_TRAP = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPU = torch.cuda.device_count()
print("python     ", sys.version.split()[0])
print("numpy      ", np.__version__, "| trapezoid:", _TRAP.__name__)
print("torch      ", torch.__version__)
print("device     ", DEVICE, "| n_gpu", N_GPU)
if DEVICE == "cuda":
    for i in range(N_GPU):
        print("   gpu", i, torch.cuda.get_device_name(i))

try:
    import timm
    HAS_TIMM = True
    print("timm       ", timm.__version__)
except Exception as e:
    HAS_TIMM = False
    print("timm        NOT available ->", repr(e), "(torchvision fallback)")

import torchvision
from torchvision import transforms as T
print("torchvision", torchvision.__version__)
''',
# ============================================================ A3
r'''"""Cell A3 - locate the mounted dataset + the shipped side-files; pin config_hash (G1).

The dataset can mount at ANY depth - seen in the wild as
/kaggle/input/datasets/<user>/<slug>/data/processed_meta/... - so everything is found by a
RECURSIVE search for a basename (shallowest path wins), never by a fixed-depth directory walk."""
def _find_all(name):
    hits = []
    for base in [Path("/kaggle/input"), Path("."), Path("..")]:
        if base.exists():
            hits += list(base.rglob(name))
    return hits

def find_one(name):
    hits = _find_all(name)
    assert hits, "file not found anywhere under the mount: %s" % name
    return sorted(hits, key=lambda p: len(str(p)))[0]

# data/processed_meta at any nesting depth; prefer a hit that is really a processed_meta dir
_meta = [p.parent for p in _find_all("manifest_images.csv")
         if p.parent.name == "processed_meta" and (p.parent / "manifest_tiles.csv").exists()]
if not _meta:
    _meta = [p.parent for p in _find_all("manifest_images.csv")]
assert _meta, "could not find data/processed_meta/manifest_images.csv under the mount"
MAN = sorted(_meta, key=lambda p: len(str(p)))[0]
INPUT_ROOT = MAN.parent.parent
print("INPUT_ROOT ", INPUT_ROOT)
print("META       ", MAN)

_probe = pd.read_csv(MAN / "manifest_images.csv")
_hashes = set(_probe.config_hash.astype(str))
assert _hashes == {CONFIG_HASH_EXPECTED}, "config_hash mismatch: %s != {%s}" % (_hashes, CONFIG_HASH_EXPECTED)
print("[G1] config_hash pinned:", CONFIG_HASH_EXPECTED)

HAND_OOF_CSV  = find_one("handcrafted_oof_train.csv")
LABELS_CSV    = find_one("train_labels_matrix.csv")
HAND_TEST_CSV = find_one("Submission_Model8_E2_E2A.csv")
SAMPLE_SUB    = INPUT_ROOT / "data" / "sample_submission.csv"
if not SAMPLE_SUB.exists():
    SAMPLE_SUB = find_one("sample_submission.csv")
for nm, p in [("handcrafted_oof", HAND_OOF_CSV), ("labels", LABELS_CSV),
              ("handcrafted_test(55.8)", HAND_TEST_CSV), ("sample_submission", SAMPLE_SUB)]:
    print("   %-24s %s" % (nm, p))
''',
# ============================================================ B1
r'''"""Cell B1 - manifests, labels, family map, and the train/test tile tables."""
imgs      = pd.read_csv(MAN / "manifest_images.csv")
tiles_all = pd.read_csv(MAN / "manifest_tiles.csv")
samples   = pd.read_csv(MAN / "manifest_samples.csv")

DIAMS = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200]
SUP   = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
DCOLS = ["d_" + s for s in SUP]

oof_df = pd.read_csv(HAND_OOF_CSV)
lab_df = pd.read_csv(LABELS_CSV)
oof_df["sample_id"] = oof_df.sample_id.astype(str)
lab_df["sample_id"] = lab_df.sample_id.astype(str)

TRAIN_IDS = oof_df.sample_id.tolist()
assert len(TRAIN_IDS) == 24, "expected 24 train soils, got %d" % len(TRAIN_IDS)
fam_of   = dict(zip(oof_df.sample_id, oof_df.cv_family.astype(str)))
Y        = lab_df.set_index("sample_id").loc[TRAIN_IDS, DCOLS].to_numpy(float)
HAND_OOF = oof_df.set_index("sample_id").loc[TRAIN_IDS, DCOLS].to_numpy(float)
assert Y.shape == (24, 11) and HAND_OOF.shape == (24, 11)
assert np.all(np.diff(Y, axis=1) >= -1e-9) and np.allclose(Y[:, -1], 100.0)

tsel = tiles_all[(tiles_all.tile_size_px == cfg.tile_size_px) & (tiles_all.materialized)
                 & (tiles_all.soil_fraction >= cfg.min_tile_soil_fraction)].copy()
tsel["sample_id"] = tsel.sample_id.astype(str)
tsel["abs"] = [str(INPUT_ROOT / p) for p in tsel.tile_path]

tr_tiles = tsel[tsel.split == "train"].copy()
te_tiles = tsel[tsel.split == "test"].copy()
tr_tiles = tr_tiles[tr_tiles.sample_id.isin(set(TRAIN_IDS))].copy()
yidx = {s: i for i, s in enumerate(TRAIN_IDS)}
tr_tiles["yrow"] = [yidx[s] for s in tr_tiles.sample_id]
tr_tiles["fam"]  = [fam_of[s] for s in tr_tiles.sample_id]

_miss = [p for p in list(tr_tiles["abs"]) + list(te_tiles["abs"]) if not Path(p).exists()]
assert not _miss, "%d tile files missing, e.g. %s" % (len(_miss), _miss[:3])

print("train tiles %d over %d soils | test tiles %d over %d soils"
      % (len(tr_tiles), tr_tiles.sample_id.nunique(), len(te_tiles), te_tiles.sample_id.nunique()))
print("handcrafted OOF mean EMD (anchor 41.1475):", round(float(np.mean(
      [float(_TRAP(np.abs(HAND_OOF[i] - Y[i]), np.log10(DIAMS))) for i in range(24)])), 4))
''',
# ============================================================ B2
r'''"""Cell B2 - the EXACT EMD metric (numpy) + host trivial-baseline reconciliation (G2)."""
DL  = np.log10(np.array(DIAMS, float))
WID = np.diff(DL)

def emd_pair(p, t):
    return float(_TRAP(np.abs(np.asarray(p, float) - np.asarray(t, float)), DL))

def mean_emd(P, T):
    P, T = np.atleast_2d(P), np.atleast_2d(T)
    return float(np.mean([emd_pair(P[i], T[i]) for i in range(len(T))]))

def project_np(Fm):
    Fm = np.clip(np.atleast_2d(np.asarray(Fm, float)), 0, 100)
    Fm = np.maximum.accumulate(Fm, axis=1)
    Fm[:, -1] = 100.0
    return Fm

TRIVIAL = np.array([9.09, 18.18, 27.27, 36.36, 45.45, 54.55, 63.64, 72.73, 81.82, 90.91, 100.0])
_t = float(np.mean([emd_pair(TRIVIAL, y) for y in Y]))
print("[G2] trivial baseline: ours %.3f  (host publishes 100.31)" % _t)
assert abs(_t - 100.31) < 0.1, "EMD no longer matches the host reference - metric drift"
''',
# ============================================================ B3
r'''"""Cell B3 - EMD as a differentiable torch loss (identical trapezoid weighting to emd_pair)."""
DL_T  = torch.tensor(DL, dtype=torch.float32)
WID_T = (DL_T[1:] - DL_T[:-1])

def emd_loss(pred, true):
    # pred, true: (B, 11) in percent [0, 100]. Mean over batch of trapezoid(|pred-true|, DL).
    d = (pred - true).abs()
    w = WID_T.to(pred.device)
    trap = 0.5 * (d[:, :-1] + d[:, 1:]) * w
    return trap.sum(dim=1).mean()

# numeric parity check: torch loss == numpy mean_emd on the same random curves
_a = torch.rand(7, 11) * 100
_b = torch.rand(7, 11) * 100
_torch = float(emd_loss(_a, _b))
_np = mean_emd(_a.numpy(), _b.numpy())
print("EMD torch %.6f | numpy %.6f | diff %.2e" % (_torch, _np, abs(_torch - _np)))
assert abs(_torch - _np) < 1e-4, "torch EMD loss disagrees with the numpy metric"
''',
# ============================================================ C1
r'''"""Cell C1 - tile dataset + the camera colour-domain randomisation."""
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

class ChannelGain:
    """Per-channel log-uniform multiplicative gain - simulates residual camera white-balance.

    The measured gap is exactly this: Android grey-world gains ~ r0.90/b1.13-1.26 vs iPhone
    r0.98/b1.03. Randomising channel gains over [lo, hi] forces the net off the Android colour
    signature so an iPhone tile is in-distribution at test time."""
    def __init__(self, lo, hi):
        self.l = math.log(lo); self.h = math.log(hi)
    def __call__(self, x):      # x: (3, H, W) float tensor in [0, 1]
        g = torch.exp(torch.empty(3).uniform_(self.l, self.h))
        return (x * g[:, None, None]).clamp(0, 1)

def build_tf(train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(cfg.img_size, scale=(cfg.rrc_scale_lo, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomChoice([T.RandomRotation((0, 0)), T.RandomRotation((90, 90)),
                            T.RandomRotation((180, 180)), T.RandomRotation((270, 270))]),
            T.ColorJitter(cfg.jitter_bcs, cfg.jitter_bcs, cfg.jitter_bcs, cfg.jitter_hue),
            T.ToTensor(),
            ChannelGain(cfg.chan_gain_lo, cfg.chan_gain_hi),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([T.Resize((cfg.img_size, cfg.img_size)), T.ToTensor(),
                      T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class TileDS(Dataset):
    def __init__(self, df, train):
        self.abs = df["abs"].tolist()
        self.y   = df["yrow"].tolist() if "yrow" in df.columns else [-1] * len(df)
        self.tf  = build_tf(train)
    def __len__(self):
        return len(self.abs)
    def __getitem__(self, i):
        img = Image.open(self.abs[i]).convert("RGB")
        return self.tf(img), self.y[i]

def make_loader(df, train, bs=None):
    return DataLoader(TileDS(df, train), batch_size=bs or cfg.batch_size, shuffle=train,
                      num_workers=2, pin_memory=(DEVICE == "cuda"), drop_last=False)
''',
# ============================================================ C2
r'''"""Cell C2 - backbone + monotone-CDF head (correct by construction), + head self-test (G3)."""
class MonoCDFHead(nn.Module):
    """logits(11) -> softmax -> cumsum x100. Non-decreasing and ends at exactly 100 BY DESIGN.

    The 11 softmax masses are the grain fraction per diameter bin; the cumsum is the cumulative
    passing curve. No capacity is spent learning that a CDF is monotone."""
    def __init__(self, in_f, n=11):
        super().__init__()
        self.fc = nn.Linear(in_f, n)
    def forward(self, x):
        p = torch.softmax(self.fc(x), dim=1)
        return torch.cumsum(p, dim=1) * 100.0

def make_backbone():
    if HAS_TIMM:
        try:
            m = timm.create_model(cfg.backbone, pretrained=cfg.pretrained,
                                  num_classes=0, global_pool="avg")
            return m, m.num_features
        except Exception as e:
            print("timm backbone failed (%r) -> torchvision" % e)
    import torchvision.models as tvm
    ctor = getattr(tvm, cfg.backbone, None)
    if ctor is None or not cfg.backbone.startswith("resnet"):
        ctor = tvm.resnet34
    try:
        net = ctor(weights="DEFAULT" if cfg.pretrained else None)
    except Exception:
        net = ctor(pretrained=cfg.pretrained)
    in_f = net.fc.in_features
    net.fc = nn.Identity()
    return net, in_f

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, in_f = make_backbone()
        self.head = MonoCDFHead(in_f)
    def forward(self, x):
        return self.head(self.backbone(x))

def new_model():
    m = Net().to(DEVICE)
    if DEVICE == "cuda" and N_GPU > 1:
        m = nn.DataParallel(m)
    return m

# G3: head is monotone and ends at 100 for arbitrary input
_h = MonoCDFHead(8)
_o = _h(torch.randn(5, 8))
assert torch.all(_o[:, 1:] - _o[:, :-1] >= -1e-6), "head not monotone"
assert torch.allclose(_o[:, -1], torch.full((5,), 100.0)), "head does not end at 100"
print("[G3] monotone-CDF head verified: non-decreasing, ends at 100")
''',
# ============================================================ C3
r'''"""Cell C3 - train a model; aggregate per-tile predictions to soil curves."""
Y_T = torch.tensor(Y, dtype=torch.float32)

def train_model(df_tr, epochs, val_df=None, val_ids=None, seed=SEED):
    set_seed(seed)
    model = new_model()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loader = make_loader(df_tr, True)
    total = max(1, len(loader)) * max(1, epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=total,
                                                pct_start=cfg.warmup_frac)
    per_epoch = []
    for ep in range(epochs):
        model.train()
        for xb, ridx in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = Y_T[ridx].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = emd_loss(model(xb), yb)
            loss.backward()
            opt.step(); sched.step()
        if val_df is not None:
            per_epoch.append(soil_predictions(model, val_df, val_ids))
    return model, per_epoch

@torch.no_grad()
def predict_tiles(model, df):
    model.eval()
    out = []
    for xb, _ in make_loader(df, False):
        xb = xb.to(DEVICE, non_blocking=True)
        p = model(xb)
        if cfg.tta:
            p = (p + model(torch.flip(xb, dims=[3])) + model(torch.flip(xb, dims=[2]))) / 3.0
        out.append(p.detach().cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, 11))

def soil_predictions(model, df, ids):
    preds = predict_tiles(model, df)
    sid = df.sample_id.tolist()
    rows = []
    for s in ids:
        m = [preds[i] for i in range(len(sid)) if sid[i] == s]
        rows.append(np.median(np.stack(m, 0), axis=0) if m else np.full(11, np.nan))
    return project_np(np.stack(rows, 0))

def score_soils(P, ids):
    yy = np.stack([Y[TRAIN_IDS.index(s)] for s in ids], 0)
    return mean_emd(P, yy)
''',
# ============================================================ D1
r'''"""Cell D1 - leave-one-FAMILY-out CV: choose the epoch and collect the CNN OOF (G5).

No held-out family ever appears in a fold that scores it (asserted). The per-epoch held-out soil
predictions are stored so the OOF can be assembled at a single epoch chosen ACROSS folds."""
FAMS = sorted(set(fam_of[s] for s in TRAIN_IDS))
print("families:", len(FAMS))

CHOSEN_EPOCHS = cfg.epochs_fallback
cnn_oof = np.full((24, 11), np.nan)

if cfg.do_cv:
    t0 = time.time()
    fold_pred, fold_ids = {}, {}
    for gi, g in enumerate(FAMS):
        val_ids = [s for s in TRAIN_IDS if fam_of[s] == g]
        tr_df = tr_tiles[tr_tiles.fam != g]
        va_df = tr_tiles[tr_tiles.fam == g]
        assert g not in set(tr_df.fam), "G5 family leak in fold %s" % g      # G5
        _, pep = train_model(tr_df, cfg.cv_epochs, val_df=va_df, val_ids=val_ids, seed=SEED + gi)
        fold_pred[g], fold_ids[g] = pep, val_ids
        print("  fold %-10s soils %d  last-epoch EMD %.3f  [%.0fs]"
              % (g, len(val_ids), score_soils(pep[-1], val_ids), time.time() - t0))

    emd_by_epoch = []
    for e in range(cfg.cv_epochs):
        oof = np.full((24, 11), np.nan)
        for g in FAMS:
            for k, s in enumerate(fold_ids[g]):
                oof[TRAIN_IDS.index(s)] = fold_pred[g][e][k]
        emd_by_epoch.append(mean_emd(oof, Y))
    best_e = int(np.argmin(emd_by_epoch))
    CHOSEN_EPOCHS = best_e + 1
    for g in FAMS:
        for k, s in enumerate(fold_ids[g]):
            cnn_oof[TRAIN_IDS.index(s)] = fold_pred[g][best_e][k]

    print("  epoch EMD curve:", [round(x, 2) for x in emd_by_epoch])
    print("CNN CV: best epoch %d  OOF EMD %.4f   (handcrafted OOF %.4f)"
          % (CHOSEN_EPOCHS, emd_by_epoch[best_e], mean_emd(HAND_OOF, Y)))
else:
    print("CV skipped; using epochs_fallback =", CHOSEN_EPOCHS)
''',
# ============================================================ D2
r'''"""Cell D2 - final model on ALL 24 soils' tiles, seed-ensembled to cut variance."""
final_models = []
t0 = time.time()
for k in range(cfg.n_seed_ensemble):
    m, _ = train_model(tr_tiles, CHOSEN_EPOCHS, seed=SEED + 1000 + k)
    final_models.append(m)
    print("  final model %d/%d trained %d epochs  [%.0fs]"
          % (k + 1, cfg.n_seed_ensemble, CHOSEN_EPOCHS, time.time() - t0))
''',
# ============================================================ E1
r'''"""Cell E1 - test soil predictions: per-tile -> seed-mean -> soil-median -> project."""
TEST_IDS = te_tiles.sample_id.drop_duplicates().tolist()
print("test soils:", len(TEST_IDS))

ens = [soil_predictions(m, te_tiles, TEST_IDS) for m in final_models]
cnn_test = project_np(np.mean(np.stack(ens, 0), axis=0))   # (n_test, 11), TEST_IDS order
assert np.isfinite(cnn_test).all(), "a test soil had no tiles"
print("cnn_test:", cnn_test.shape)
''',
# ============================================================ E2
r'''"""Cell E2 - blend weight on the 24 soils (family-held-out OOF) + error-correlation report."""
if cfg.do_cv and np.isfinite(cnn_oof).all():
    e_cnn = np.array([emd_pair(cnn_oof[i], Y[i]) for i in range(24)])
    e_han = np.array([emd_pair(HAND_OOF[i], Y[i]) for i in range(24)])
    rho = float(np.corrcoef(e_cnn, e_han)[0, 1])
    print("per-soil OOF error: CNN mean %.4f | HAND mean %.4f | corr %.3f"
          % (e_cnn.mean(), e_han.mean(), rho))
    print("(blending helps only if corr is well below the 0.86-0.95 that killed E4's blend)")

    best_w, best_e, curve = 0.0, mean_emd(HAND_OOF, Y), []
    for w in cfg.blend_grid:
        e = mean_emd(project_np(w * cnn_oof + (1 - w) * HAND_OOF), Y)
        curve.append((w, e))
        if e < best_e - 1e-9:
            best_e, best_w = e, w
    print("blend grid (w*CNN + (1-w)*HAND  ->  internal LOFO EMD):")
    for w, e in curve:
        print("   w=%.2f  %.4f%s" % (w, e, "   <-- best" if abs(w - best_w) < 1e-9 else ""))
    print("chosen internal blend: w=%.2f  EMD=%.4f  (vs handcrafted OOF %.4f)"
          % (best_w, best_e, mean_emd(HAND_OOF, Y)))
else:
    best_w, rho = 0.5, float("nan")
    print("no CNN OOF (CV off) -> cannot select w internally; defaulting blend w=0.50")
''',
# ============================================================ E3
r'''"""Cell E3 - map internal ids to submission ids (via the manifest) and write CSVs (G4).

The internal -> board id join goes through manifest_samples.submission_id; NEVER string surgery.
(Muenster arrives mojibake-corrupted internally and is 'HPC_Muenster_BS6_9_0-10m' on the board.)"""
sub_of = dict(zip(samples.sample_id.astype(str), samples.submission_id.astype(str)))
ss = pd.read_csv(SAMPLE_SUB)
SS_IDS = ss.sample_id.astype(str).tolist()
assert len(SS_IDS) == 10

# handcrafted (55.80591) test predictions, aligned to the submission order
hdf = pd.read_csv(HAND_TEST_CSV)
hdf["sample_id"] = hdf.sample_id.astype(str)
HAND_TEST = hdf.set_index("sample_id").loc[SS_IDS, SUP].to_numpy(float)

# CNN test predictions, re-ordered from TEST_IDS (internal) into submission order
rows = []
for sid in SS_IDS:
    internal = [s for s in TEST_IDS if sub_of.get(s) == sid]
    assert len(internal) == 1, "id join failed for %s -> %s" % (sid, internal)
    rows.append(cnn_test[TEST_IDS.index(internal[0])])
CNN_TEST = project_np(np.stack(rows, 0))

def write_ss(mat, name):
    data = project_np(mat)
    assert data.shape == (10, 11), "shape %s" % (data.shape,)
    assert np.all(np.diff(data, axis=1) >= -1e-9), "non-monotone row"
    assert np.allclose(data[:, -1], 100.0), "last column != 100"
    assert np.isfinite(data).all() and (data >= -1e-6).all() and (data <= 100 + 1e-6).all()
    out = pd.DataFrame(data, columns=SUP)
    out.insert(0, "sample_id", SS_IDS)
    p = Path(cfg.out_dir) / name
    out.to_csv(p, index=False)
    print("   [G4 ok] wrote", p)
    return p

wtag = int(round(best_w * 100))
write_ss(CNN_TEST, "Submission_Model9_E1_CNN.csv")
write_ss(best_w * CNN_TEST + (1 - best_w) * HAND_TEST, "Submission_Model9_E1_BLEND_w%02d.csv" % wtag)
write_ss(0.5 * CNN_TEST + 0.5 * HAND_TEST, "Submission_Model9_E1_BLEND_w50.csv")
print("mean |CNN - HAND| on the 10 test soils:",
      round(float(np.mean([emd_pair(CNN_TEST[i], HAND_TEST[i]) for i in range(10)])), 3), "EMD")
''',
# ============================================================ E4
r'''"""Cell E4 - the honest summary and the submission recommendation."""
print("=" * 70)
print("MODEL 9 / EXPERIMENT 1 - SUMMARY")
print("=" * 70)
print("Backbone            :", cfg.backbone, "| img", cfg.img_size, "| ensemble", cfg.n_seed_ensemble)
print("Chosen epochs (CV)  :", CHOSEN_EPOCHS)
if cfg.do_cv and np.isfinite(cnn_oof).all():
    print("CNN  OOF (Android transfer, family-held-out):", round(mean_emd(cnn_oof, Y), 4))
    print("HAND OOF (the 55.8 head, same protocol)     :", round(mean_emd(HAND_OOF, Y), 4), "(anchor 41.1475)")
    print("best internal blend w, EMD                  :", best_w, round(mean_emd(
          project_np(best_w * cnn_oof + (1 - best_w) * HAND_OOF), Y), 4))
print("-" * 70)
print("HONEST READ:")
print(" - These are INTERNAL numbers on 24 Android soils. They are NOT Kaggle scores.")
print(" - The camera-augmentation benefit is UNMEASURABLE internally (0 iPhone train images).")
print("   Whether it beats 55.80591 can only be seen on the board - that is the bet.")
print("-" * 70)
print("SUBMIT (fewest slots that learn the most):")
print(" 1) Submission_Model9_E1_BLEND_w%02d.csv   (lowest-variance bet to beat 55.80591)" % wtag)
print(" 2) Submission_Model9_E1_CNN.csv          (diagnostic: did the representation transfer?)")
print(" The handcrafted control is the known 55.80591 and is NOT re-submitted.")
print("=" * 70)
''',
]


def main() -> int:
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3",
          "language": "python", "name": "python3"},
          "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    ok = 0
    for i, src in enumerate(CELLS):
        src = src.strip("\n")
        try:
            compile(src + "\n", "<cell %d>" % i, "exec")
            ok += 1
        except SyntaxError as e:
            print("SYNTAX ERROR in cell %d: %s (line %s)" % (i, e.msg, e.lineno))
            print("   >>>", (e.text or "").rstrip())
            return 1
        lines = src.split("\n")
        nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None,
                            "outputs": [], "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]})
    out = HERE / "Model9_Experiment1_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("[ok] all %d cells compiled cleanly" % ok)
    print("[ok] wrote", out, "(%d cells)" % len(nb["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
