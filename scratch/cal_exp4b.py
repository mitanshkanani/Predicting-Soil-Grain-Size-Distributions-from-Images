"""Does the tile->soil aggregation statistic have headroom? E1/E2/E3 all take a plain
median of tile features per image, then a median over images. A grain-size DISTRIBUTION
is being described by a single central value per feature. This measures what that costs."""
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
CORE = ['e4','e8','e16','lum_sd','grad_mean']; COL = ['R','G','B','sat','lum_p10','lum_p50','lum_p90']
FE = CORE + COL
C = 'Model 1/Model 1 Experiment 3/.cache/'; H = '010f44c36c74'
def load(fn):
    d = t256[['tile_path','sample_id','split','camera','parent_image_path','soil_fraction']].merge(
        pd.read_csv(C + fn), on='tile_path').merge(
        imgs[['processed_path','crop_area_fraction']].rename(columns={'processed_path':'parent_image_path'}),
        on='parent_image_path')
    d = d[(d.soil_fraction >= 0.5) & (d.split == 'train')].copy()
    d['camera'] = d.camera.str.split(' ').str[0]
    return d
def summarise(d, stats):
    g = d.groupby(['sample_id','camera','parent_image_path'])[FE]
    out = pd.concat({s: getattr(g, s)(**({'q':0.25},) if s=='quantile' else {}) for s in stats}, axis=1)
    out.columns = pd.MultiIndex.from_product([stats, FE])
    return out.groupby(['sample_id','camera']).median().stack(0)
def build(stats):
    d = load(f'tile_features_256_{H}.csv')
    g = d.groupby(['sample_id','camera','parent_image_path'])[FE]
    parts = {}
    for s in stats:
        parts[s] = g.median() if s == 'med' else (
            g.mean() if s == 'mean' else g.std() if s == 'sd' else
            g.quantile(0.1) if s == 'p10' else g.quantile(0.9) if s == 'p90' else g.count())
    T = pd.concat(parts, axis=1)
    T.columns = [f'{a}__{b}' for a, b in T.columns]
    return T.groupby('sample_id').median()
def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F,float)),0,100); F = np.maximum.accumulate(F,axis=1); F[:,-1]=100.
    return F
def emd(p,t): return float(np.trapezoid(np.abs(p-t),DL))
def fp(Xtr,Ytr,Xte,a,rank=3):
    mu=Ytr.mean(0); _,_,Vt=np.linalg.svd(Ytr-mu,full_matrices=False); V=Vt[:rank]
    sc=StandardScaler().fit(Xtr); T=(Ytr-mu)@V.T
    return project(Ridge(alpha=a).fit(sc.transform(Xtr),T).predict(sc.transform(Xte))@V+mu)
AG = [0.03,0.1,0.3,1,3,10,30,100,300,1000]
def logo(S):
    ids = [s for s in tr_ids if s in S.index]
    X = S.loc[ids].to_numpy(float); Yy = np.array([Ymap[s] for s in ids])
    g = np.array([fams[s] for s in ids]); best = 9e9
    for a in AG:
        e = []
        for F in GROUPS:
            k = g != F
            if not (~k).any(): continue
            P = fp(X[k], Yy[k], X[~k], a)
            e += [emd(P[i], Yy[j]) for i,j in enumerate(np.where(~k)[0])]
        best = min(best, float(np.mean(e)))
    return best, len(ids), S.shape[1]
for label, stats in [('median only (current protocol)', ['med']),
                     ('median+mean+sd',                  ['med','mean','sd']),
                     ('+ spread p10/p90',                ['med','mean','sd','p10','p90'])]:
    try:
        S = build(stats); S = S.replace([np.inf,-np.inf], np.nan).dropna()
        l, n, k = logo(S)
        print(f'{label:32s} {k:3d} cols  n={n}  LOGO-CV best {l:6.2f}')
    except Exception as ex:
        print(f'{label:32s} FAILED {type(ex).__name__}: {ex}')
print(f'\nreference: E3 cell C LOGO-CV 45.65 | rank-3 ceiling 7.35 | no-image floor {np.mean([emd(np.median(Y,axis=0),y) for y in Y]):.2f}')
