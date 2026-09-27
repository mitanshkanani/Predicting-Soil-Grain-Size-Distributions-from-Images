"""Read-only reconnaissance for the Experiment 4 plan.

E4's treatment is: train on resolution-matched views so the model sees the test set's
sampling history. The question recon must answer is whether that helps under a ruler that
holds out a sharpness level the model has NEVER seen -- and whether the existing CAM+RES
ruler becomes circular once blur is part of training.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sup = ['0.002','0.0063','0.02','0.063','0.2','0.63','2','6.3','20','63','200']
DL = np.log10(np.array([float(c) for c in sup]))
smp = pd.read_csv('data/processed_meta/manifest_samples.csv')
imgs = pd.read_csv('data/processed_meta/manifest_images.csv')
tl = pd.read_csv('data/processed_meta/manifest_tiles.csv')
t256 = tl[(tl.tile_size_px == 256) & tl.materialized]
tr_ids = list(smp[smp.split == 'train'].sample_id)
Y = smp[smp.split == 'train'].set_index('sample_id')[sup].to_numpy(float)
Ymap = {s: Y[i] for i, s in enumerate(tr_ids)}
fams = pd.read_csv('Model 1/Model 1 Experiment 3/cv_families.csv').set_index('sample_id').cv_family
GROUPS = sorted(fams.unique())

CORE = ['e4','e8','e16','lum_sd','grad_mean']
FREQ = ['spec_centroid_cpm','dom_wavelength_mm']
COL  = ['R','G','B','sat','lum_p10','lum_p50','lum_p90']
CELLC = CORE + COL                      # E3's selected 12-feature set
CELLD = CORE + FREQ + COL               # E2's 14-feature set
FE = CELLC + FREQ                       # all 14 available
C = 'Model 1/Model 1 Experiment 3/.cache/'; H = '010f44c36c74'
geo = imgs[['processed_path','crop_area_fraction']].rename(columns={'processed_path':'parent_image_path'})
views = {}
for tag, fn in [('real', f'tile_features_256_{H}.csv'),
                ('res14', f'tile_features_res_256_{H}_iPhone14.csv'),
                ('res16', f'tile_features_res_256_{H}_iPhone16.csv')]:
    d = t256[['tile_path','sample_id','split','camera','parent_image_path','soil_fraction']].merge(
        pd.read_csv(C + fn), on='tile_path').merge(geo, on='parent_image_path')
    d = d[(d.soil_fraction >= 0.5) & (d.split == 'train')].copy()
    d['camera'] = d.camera.str.split(' ').str[0]
    im = d.groupby(['sample_id','camera','parent_image_path'])[FE].median()
    views[tag] = im.groupby(['sample_id','camera'])[FE].median()

def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F,float)),0,100); F = np.maximum.accumulate(F,axis=1); F[:,-1]=100.
    return F
def emd(p,t): return float(np.trapezoid(np.abs(p-t),DL))
def fp(Xtr,Ytr,Xte,a,rank=3):
    mu=Ytr.mean(0); _,_,Vt=np.linalg.svd(Ytr-mu,full_matrices=False); V=Vt[:rank]
    sc=StandardScaler().fit(Xtr); T=(Ytr-mu)@V.T
    return project(Ridge(alpha=a).fit(sc.transform(Xtr),T).predict(sc.transform(Xte))@V+mu)

def score(FEset, alpha, tv, tc, sv, sc_):
    """train on view tv camera tc; predict view sv camera sc_."""
    A = views[tv].xs(tc, level='camera'); B = views[sv].xs(sc_, level='camera')
    ids = sorted(set(A.index) & set(B.index) & set(tr_ids)); errs = []
    for F in GROUPS:
        k = np.array([fams[s] != F for s in ids])
        if k.all() or not (~k).any(): continue
        P = fp(A.loc[ids,FEset].to_numpy(float)[k], np.array([Ymap[s] for s in ids])[k],
               B.loc[ids,FEset].to_numpy(float)[~k], alpha)
        errs += [emd(P[i], Ymap[ids[j]]) for i,j in enumerate(np.where(~k)[0])]
    return float(np.mean(errs)) if errs else np.nan

PAIRS = [('Mot>Sam', 'Motorola', 'Samsung')]
print("E3's selected 12-feature set, alpha 10. Blur levels: real ~0.04 px, res14 0.84, res16 1.20\n")
print(f"{'training view':16s}{'-> real':>10}{'-> res14':>10}{'-> res16':>10}")
for trv in ['real','res14','res16']:
    row = [score(CELLC,10,trv,'Motorola',t,'Samsung') for t in ('real','res14','res16')]
    print(f"{trv:16s}" + "".join(f"{v:10.2f}" for v in row))
print("\n(the diagonal-ish entry 'res16 -> res16' is not shown; each row trains at one level")
print(" and predicts the OTHER camera at the column's level.)")

print("\nSame table, E2's 14-feature set (with frequency), alpha 30:")
for trv in ['real','res14','res16']:
    row = [score(CELLD,30,trv,'Motorola',t,'Samsung') for t in ('real','res14','res16')]
    print(f"{trv:16s}" + "".join(f"{v:10.2f}" for v in row))

print("\nDoes E3's 'drop frequency' conclusion survive blurred training?")
for trv, tv in [('real','res16'), ('res14','res16'), ('res16','res16')]:
    a = score(CELLC,10,trv,'Motorola',tv,'Samsung')
    b = score(CELLD,10,trv,'Motorola',tv,'Samsung')
    print(f"  train {trv:6s} -> {tv:6s}: 12-feat {a:6.2f}   14-feat(+freq) {b:6.2f}   freq costs {b-a:+6.2f}")
