import numpy as np, pandas as pd, glob, os
from sklearn.linear_model import Ridge
sup=['0.002','0.0063','0.02','0.063','0.2','0.63','2','6.3','20','63','200']
L=np.log10(np.array([float(c) for c in sup]))
lab={os.path.basename(os.path.dirname(p)):pd.read_csv(p)[sup].to_numpy(float)[0]
     for p in sorted(glob.glob('data/Training/*/labels.csv'))}
t=pd.read_csv('Model 1/Model 1 Experiment 1/.cache/tile_features_256_010f44c36c74.csv')
sp=t.tile_path.str.extract(r'data/tiles/(train|test)/([^/]+)/')
t=t.assign(split=sp[0],soil=sp[1]); t['cam']=t.tile_path.str.split('__').str[1].str.split('_').str[0]
TILE=['e4','e8','e16','R','G','B','sat','lum_p10','lum_p50','lum_p90','lum_sd','grad_mean',
    'spec_centroid_cpm','dom_wavelength_mm']
GEOM=['soil_fraction','crop_area_fraction']; FE=TILE+GEOM
_mi=pd.read_csv('data/processed_meta/manifest_images.csv')
_mi=_mi.assign(cam=_mi.camera.str.split(' ').str[0].replace({'iPhone14':'iPhone','iPhone16':'iPhone'}))
_geo=_mi[_mi.split=='train'].groupby(['sample_id','cam'])[GEOM].mean()

G0=t[t.split=='train'].groupby(['soil','cam'])[TILE].mean()
G=pd.concat([G0,_geo.reindex(G0.index)],axis=1)
print('geometry join NaNs:',int(G.isna().to_numpy().sum()))
assert not G.isna().any().to_numpy().any(), G.isna().sum()[G.isna().sum()>0]
TEX=['e4','e8','e16','lum_sd','grad_mean','spec_centroid_cpm','dom_wavelength_mm']
COL=['R','G','B','sat','lum_p10','lum_p50','lum_p90']
CLEAN=[c for c in FE if c not in GEOM]
f=pd.read_csv('Model 1/Model 1 Experiment 1/features_soil.csv')
fam=pd.read_csv('Model 1/Model 1 Experiment 1/cv_families.csv').set_index('sample_id').cv_family
tr=f[f.split=='train'].set_index('sample_id')
# per-soil-per-camera feature table

ms=[s for s in sorted(set(G.index.get_level_values(0))) if len(G.loc[s])==2]
print('soils with BOTH training cameras: %d of 24'%len(ms))
Y={s:lab[s] for s in ms}
def emd(p,q): return float(np.trapezoid(np.abs(p-q),L))
def proj(F):
    F=np.clip(np.atleast_2d(np.asarray(F,float)),0,100);F=np.maximum.accumulate(F,axis=1);F[:,-1]=100.;return F
def basis(Ytr,r=3):
    mu=Ytr.mean(0);_,_,Vt=np.linalg.svd(Ytr-mu,full_matrices=False);return mu,Vt[:r]
def run(FEset,alpha,src,tgt,soils):
    X=pd.DataFrame({s:G.loc[(s,src),FEset].to_numpy() for s in soils}).T.loc[soils]
    Xt=pd.DataFrame({s:G.loc[(s,tgt),FEset].to_numpy() for s in soils}).T.loc[soils]
    Ym=np.array([Y[s] for s in soils]); groups=np.array([fam[s] for s in soils]); errs=[]
    for g in sorted(set(groups)):
        m=groups==g; ti=np.where(~m)[0]; vi=np.where(m)[0]
        mu,V=basis(Ym[ti],3); T=(Ym[ti]-mu)@V.T
        sc=X.iloc[ti].mean(); sd=X.iloc[ti].std()+1e-9
        r=Ridge(alpha=alpha).fit(((X.iloc[ti]-sc)/sd).to_numpy(),T)
        P=proj(r.predict(((Xt.iloc[vi]-sc)/sd).to_numpy())@V+mu)
        for i,k in enumerate(vi): errs.append(emd(P[i],Ym[k]))
    return float(np.mean(errs))
def testpred(FEset,alpha):
    f2=pd.read_csv('Model 1/Model 1 Experiment 1/features_soil.csv')
    trm=f2[f2.split=='train'].set_index('sample_id'); tem=f2[f2.split=='test'].set_index('sample_id')
    Ytr=np.array([lab[s] for s in trm.index]); mu,V=basis(Ytr,3); T=(Ytr-mu)@V.T
    sc=trm[FEset].mean(); sd=trm[FEset].std()+1e-9
    r=Ridge(alpha=alpha).fit(((trm[FEset]-sc)/sd).to_numpy(),T)
    P=proj(r.predict(((tem[FEset]-sc)/sd).to_numpy())@V+mu)
    return float(np.mean([emd(p,Ytr.mean(0)) for p in P])), 100*np.mean((P<=1e-6)|(P>=100))
print()
print('%-11s %-6s %10s %10s %10s %12s %9s'%('feats','alpha','M->M same','M->S trans','S->M trans','test vs mean','test sat'))
for nm,FS in [('16 (E1)',FE),('14 no-geom',CLEAN),('7 texture',TEX),('7 colour',COL)]:
    for a in [0.03,1,10,100,1000]:
        i=run(FS,a,'Motorola','Motorola',ms); x=run(FS,a,'Motorola','Samsung',ms)
        y=run(FS,a,'Samsung','Motorola',ms); tm,ts=testpred(FS,a)
        print('%-11s %-6.4g %10.1f %10.1f %10.1f %12.1f %8.1f%%'%(nm,a,i,x,y,tm,ts))
    print()
