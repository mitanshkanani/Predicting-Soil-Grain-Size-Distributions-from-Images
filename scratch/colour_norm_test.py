"""Does per-image colour normalisation remove the train->test camera domain gap?"""
import os, numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi

Image.MAX_IMAGE_PIXELS = None
D = "data"
EXT = {".jpg", ".jpeg", ".png"}
TARGET_PPM = 4.58
PPM = {"MotorolaEdge": 11.492 * 0.4, "SamsungA52": 26.33 * (1600 / 9248),
       "MotEdge60Fusion": 12.465, "iPhone14": 13.942, "iPhone16": 19.525}


def cam(f):
    fl = f.lower()
    if "samsung" in fl: return "SamsungA52"
    if "fusion" in fl: return "MotEdge60Fusion"
    if "motorola" in fl: return "MotorolaEdge"
    if "iphone14" in fl or "iphone_14" in fl: return "iPhone14"
    return "iPhone16"


def load(path, ppm):
    im = Image.open(path).convert("RGB")
    s = TARGET_PPM / ppm
    im = im.resize((max(int(im.width * s), 32), max(int(im.height * s), 32)))
    return np.asarray(im, float)


def gray_world(a):
    g = a.mean(axis=(0, 1))
    return a * (g.mean() / np.maximum(g, 1e-6))


def normalize(a, mode):
    if mode == "none":
        return a
    if mode == "gw":
        return gray_world(a)
    if mode == "gw+std":
        b = gray_world(a)
        return (b - b.mean((0, 1))) / (b.std((0, 1)) + 1e-6) * 40 + 128
    if mode == "gw+std+sat":
        b = gray_world(a)
        b = (b - np.percentile(b, 1, (0, 1))) / (np.percentile(b, 99, (0, 1)) - np.percentile(b, 1, (0, 1)) + 1e-6)
        return b * 255
    raise ValueError(mode)


def feats(a):
    g = a.mean(2)
    m = g > np.percentile(g, 12)
    base = ndi.gaussian_filter(g, min(g.shape) // 12)
    e = g - base
    out = {}
    for sc in [1, 2, 4, 8, 16, 32]:
        b = ndi.gaussian_filter(np.where(m, e, 0), sc)
        out[f"e{sc}"] = float(np.sqrt(b[m].var() + 1e-12))
    out["contrast"] = float(e[m].std())
    am = a[m]
    for k, c in enumerate("RGB"):
        out[c] = float(am[:, k].mean())
    out["sat"] = float((am.max(1) - am.min(1)).mean() / (am.mean() + 1e-6))
    return out


lab = pd.read_csv("data/Training_labels_updated.csv").set_index("sample_id")
dl = np.log10([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200])
logd50 = {s: np.interp(50, np.maximum(lab.loc[s].to_numpy(float), np.linspace(1e-6, 1, 11)), dl)
          for s in lab.index}

for mode in ["none", "gw", "gw+std", "gw+std+sat"]:
    rows = []
    for split, root in [("train", "Training"), ("test", "Test")]:
        for s in sorted(os.listdir(os.path.join(D, root))):
            p = os.path.join(D, root, s)
            if not os.path.isdir(p): continue
            fs = [f for f in sorted(os.listdir(p)) if os.path.splitext(f)[1].lower() in EXT]
            per = [feats(normalize(load(os.path.join(p, f), PPM[cam(f)]), mode)) for f in fs]
            r = pd.DataFrame(per).mean().to_dict()
            r.update(split=split, key=s)
            rows.append(r)
    F = pd.DataFrame(rows)
    tr, te = F[F.split == "train"], F[F.split == "test"]
    cols = [c for c in F.columns if c not in ("split", "key")]
    shifts = [(te[c].mean() - tr[c].mean()) / (tr[c].std() + 1e-9) for c in cols]
    sp = [pd.Series(tr[c].values).corr(pd.Series(tr.key.map(logd50).values), method="spearman") for c in cols]
    print(f"\n### normalisation = {mode}")
    print(f"  mean |train->test shift| over {len(cols)} features = {np.mean(np.abs(shifts)):.2f} SD   worst = {np.max(np.abs(shifts)):.2f} SD")
    for c, sh, s in zip(cols, shifts, sp):
        print(f"    {c:9s} shift={sh:6.2f} SD   spearman(feature,logD50)={s:6.3f}")
