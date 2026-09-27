"""Read-only reconnaissance for the Experiment 3 plan. Sizes the candidate feature sets
under Experiment 2's validated ruler so E3's predictions are falsifiable. Writes nothing
outside scratch/."""
import numpy as np, pandas as pd, json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sup = ['0.002','0.0063','0.02','0.063','0.2','0.63','2','6.3','20','63','200']
DL = np.log10(np.array([float(c) for c in sup]))
E2 = pd.read_csv('Model 1/Model 1 Experiment 2/features_soil.csv')
smp = pd.read_csv('data/processed_meta/manifest_samples.csv')
tr_ids = list(smp[smp.split=='train'].sample_id)
Y = smp[smp.split=='train'].set_index('sample_id')[sup].to_numpy(float)
Ymap = {s: Y[i] for i, s in enumerate(tr_ids)}
fams = pd.read_csv('Model 1/Model 1 Experiment 2/cv_families.csv').set_index('sample_id').cv_family
GROUPS = sorted(fams.unique())

FE14 = ['e4','e8','e16','R','G','B','sat','lum_p10','lum_p50','lum_p90','lum_sd',
        'grad_mean','spec_centroid_cpm','dom_wavelength_mm']
COL = ['R','G','B','sat','lum_p10','lum_p50','lum_p90']
TEX = [c for c in FE14 if c not in COL]

# rebuild the soil x camera x view table from E2's caches, exactly as E2 assembled it
imgs = pd.read_csv('data/processed_meta/manifest_images.csv')
tl = pd.read_csv('data/processed_meta/manifest_tiles.csv')
t256 = tl[(tl.tile_size_px==256) & tl.materialized]
geo = imgs[['processed_path','crop_area_fraction']].rename(columns={'processed_path':'parent_image_path'})
C = 'Model 1/Model 1 Experiment 2/.cache/'
H = '010f44c36c74'
views = {}
for lab, fn in [('real', f'tile_features_256_{H}.csv'),
                ('res14', f'tile_features_res_256_{H}_iPhone14.csv'),
                ('res16', f'tile_features_res_256_{H}_iPhone16.csv')]:
    f = pd.read_csv(C + fn)
    d = t256[['tile_path','sample_id','split','camera','parent_image_path','soil_fraction']].merge(f, on='tile_path')
    d = d.merge(geo, on='parent_image_path')
    d = d[d.soil_fraction >= 0.5].copy()
    d['camera'] = d.camera.str.split(' ').str[0]
    if lab == 'real':
        cols = [c for c in d.columns if c in FE14]
        assert set(cols) == set(FE14), set(FE14) - set(cols)
    im = d[d.split=='train'].groupby(['sample_id','camera','parent_image_path'])[FE14].median()
    g = im.groupby(['sample_id','camera'])[FE14].median()   # two-stage, matching E2
    views[lab] = g
print('soil x camera rows per view:', {k: len(v) for k, v in views.items()})

def project(F):
    F = np.clip(np.atleast_2d(np.asarray(F,float)),0,100); F = np.maximum.accumulate(F,axis=1); F[:,-1]=100.
    return F
def emd(p,t): return float(np.trapezoid(np.abs(p-t),DL))
def fp(Xtr,Ytr,Xte,a,rank=3):
    mu = Ytr.mean(0); _,_,Vt = np.linalg.svd(Ytr-mu, full_matrices=False); V = Vt[:rank]
    sc = StandardScaler().fit(Xtr); T = (Ytr-mu)@V.T
    r = Ridge(alpha=a).fit(sc.transform(Xtr), T)
    return project(r.predict(sc.transform(Xte))@V + mu)

def score(FE, a, srcview, srccam, tgtview, tgtcam, skip=None):
    A = views[srcview].xs(srccam, level='camera'); B = views[tgtview].xs(tgtcam, level='camera')
    ids = sorted(set(A.index) & set(B.index) & set(tr_ids))
    errs = []
    for F in GROUPS:
        if F == skip: continue
        k = np.array([fams[s] != F for s in ids])
        if k.all() or not (~k).any(): continue
        P = fp(A.loc[ids,FE].to_numpy(float)[k], np.array([Ymap[s] for s in ids])[k],
               B.loc[ids,FE].to_numpy(float)[~k], a)
        errs += [emd(P[i], Ymap[ids[j]]) for i,j in enumerate(np.where(~k)[0])]
    return (float(np.mean(errs)), len(errs)) if errs else (np.nan, 0)

def logo(FE, a, skip=None):
    X = E2[E2.split=='train'].set_index('sample_id').loc[tr_ids, FE].to_numpy(float)
    errs = []
    for F in GROUPS:
        if F == skip: continue
        k = np.array([fams[s] != F for s in tr_ids])
        P = fp(X[k], Y[k], X[~k], a)
        errs += [emd(P[i], Y[j]) for i,j in enumerate(np.where(~k)[0])]
    return float(np.mean(errs)), len(errs)

CORE = ['e4','e8','e16','lum_sd','grad_mean']
SETS = {'CORE5': CORE, 'CORE+freq(TEX7)': CORE+['spec_centroid_cpm','dom_wavelength_mm'],
        'CORE+colour(12)': CORE+COL, 'CORE+both(14)': FE14,
        '16 (E1)': FE14+['soil_fraction','crop_area_fraction'], '14 (E2)': FE14,
        'TEX7': TEX, 'COL7': COL,
        'TEX5 (no freq)': [c for c in TEX if c not in ('spec_centroid_cpm','dom_wavelength_mm')],
        'TEX7+R': TEX+['R'], 'FREQ2': ['spec_centroid_cpm','dom_wavelength_mm']}
AG = [0.03,0.1,0.3,1,3,10,30,100,300,1000]
print(f"\n{'set':16s}{'n':>3} {'LOGO-CV':>8} {'CAM':>8} {'RES':>8} {'CAM+RES':>8} {'best a':>8}")
out = {}
for nm, FE in SETS.items():
    if nm == '16 (E1)':   # geometry is not in the per-camera table; approximate with 14
        FE = FE14; nm = '16 (E1)~14'
    best = min(((score(FE,a,'real','Motorola','real','Samsung')[0], a) for a in AG))
    cr = min(((score(FE,a,'real','Motorola','res14','Samsung')[0], a) for a in AG))
    rs = min(((score(FE,a,'real','Motorola','res14','Motorola')[0], a) for a in AG))
    lc = min((logo(FE,a)[0] for a in AG))
    out[nm] = dict(n=len(FE), logo=lc, cam=best[0], res=rs[0], camres=cr[0], a=cr[1])
    print(f"{nm:16s}{len(FE):>3} {lc:8.2f} {best[0]:8.2f} {rs[0]:8.2f} {cr[0]:8.2f} {cr[1]:8g}")
json.dump(out, open('scratch/cal_exp3.json','w'), indent=1)

# ---- paired simple-effects for the 2x2 ----
def per_soil(FE, a, tgtview, tgtcam, srccam='Motorola'):
    A = views['real'].xs(srccam, level='camera'); B = views[tgtview].xs(tgtcam, level='camera')
    ids = sorted(set(A.index) & set(B.index) & set(tr_ids)); d = {}
    for F in GROUPS:
        k = np.array([fams[s] != F for s in ids])
        if k.all() or not (~k).any(): continue
        P = fp(A.loc[ids,FE].to_numpy(float)[k], np.array([Ymap[s] for s in ids])[k],
               B.loc[ids,FE].to_numpy(float)[~k], a)
        for i, j in enumerate(np.where(~k)[0]): d[ids[j]] = emd(P[i], Ymap[ids[j]])
    return d

CELLS = {'CORE5': (CORE,10), 'TEX7': (CORE+['spec_centroid_cpm','dom_wavelength_mm'],30),
         'COL12': (CORE+COL,10), 'ALL14': (FE14,30)}
CONTRASTS = [('freq effect | no colour','TEX7','CORE5'), ('freq effect | +colour','ALL14','COL12'),
             ('colour effect | no freq','COL12','CORE5'), ('colour effect | +freq','ALL14','TEX7')]
rng = np.random.default_rng(20260927)
for ruler, tv, tc in [('CAM+RES','res14','Samsung'), ('CAM','real','Samsung'), ('RES','res14','Motorola')]:
    print("\n" + f"{ruler} ruler, paired simple effects (negative = the first cell is BETTER):")
    cache = {n: per_soil(FE,a,tv,tc) for n,(FE,a) in CELLS.items()}
    for nm, x, y in CONTRASTS:
        da, db = cache[x], cache[y]; sh = sorted(set(da)&set(db))
        d = np.array([da[s]-db[s] for s in sh])
        b = np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(4000)])
        lo,hi = np.percentile(b,[2.5,97.5])
        sign = int((d<0).sum())
        print(f"  {nm:22s} delta {d.mean():+7.2f}  CI [{lo:+7.2f},{hi:+7.2f}]  "
              f"{x} wins {sign}/{len(d)}  {'SIG' if hi<0 else ('SIG-worse' if lo>0 else 'ns')}")
