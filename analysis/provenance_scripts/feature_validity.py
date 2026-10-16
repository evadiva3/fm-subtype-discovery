from __future__ import annotations
import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
from _paths import RP,out
import glob,json,re,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd

OUT=out("feature_validity")
from config import config
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression,RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold,GroupKFold,cross_val_predict
from sklearn.metrics import accuracy_score
from scipy import stats

OUT.mkdir(parents=True,exist_ok=True)
CONDS=[c.replace(" ","") for c in config.conditions]

ex=pd.read_csv(config.exclusionManifestPath)
keep=set(ex[ex.exclusion_reason.fillna("")==""].subject_id.astype(str))
clin=pd.read_csv(config.clinicalCsv);clin["subject_id"]=clin["subject_id"].astype(str)
print(f"analytic sample from manifest: {len(keep)} subjects")
print(f"clinical columns: {list(clin.columns)}")

iu=None
X,sub,cond=[],[],[]
for d in sorted(glob.glob(str(RP/"data/Subjects/sub-*"))):
    sid=re.sub(r"^sub-","",os.path.basename(d))
    if sid not in keep and os.path.basename(d) not in keep:
        continue
    key=sid if sid in keep else os.path.basename(d)
    for c in CONDS:
        f=os.path.join(d,f"{os.path.basename(d)}_FCMatrixCondition{c}.npy")
        if not os.path.exists(f):
            continue
        m=np.load(f).astype(np.float32)
        if iu is None:
            iu=np.triu_indices(m.shape[0],k=1)
        v=m[iu]
        v=np.arctanh(np.clip(v,-0.999999,0.999999))
        X.append(v);sub.append(key);cond.append(c)
X=np.asarray(X);sub=np.asarray(sub);cond=np.asarray(cond)
print(f"loaded {X.shape[0]} matrices from {len(np.unique(sub))} subjects, "
      f"{X.shape[1]:,} edge features each ({len(np.unique(cond))} conditions)")
X=np.nan_to_num(X)

res={}

def clf_cv(y,groups,name,chance,nsplits=5,stratified=True):
    pipe=make_pipeline(StandardScaler(),PCA(n_components=50,random_state=0),
                        LogisticRegression(max_iter=5000,C=1.0))
    cvo=(StratifiedGroupKFold(n_splits=nsplits,shuffle=True,random_state=0)
         if stratified else GroupKFold(n_splits=nsplits))
    pred=cross_val_predict(pipe,X,y,groups=groups,cv=cvo)
    acc=accuracy_score(y,pred)
    n=len(y);k=int(round(acc*n))
    p=stats.binomtest(k,n,chance,alternative="greater").pvalue
    lo,hi=stats.binomtest(k,n).proportion_ci(0.95)
    print(f"  {name:<26} acc {acc:.4f}  [95% CI {lo:.3f}, {hi:.3f}]  "
          f"chance {chance:.4f}  p={p:.3e}   {'ABOVE CHANCE' if p<0.05 else 'ns'}")
    res[name]={"accuracy":acc,"ci95":[lo,hi],"chance":chance,"p":float(p),"n":n}
    return acc

print()
print("(a) task condition -- subject-wise cv, subject identity cannot leak")
clf_cv(cond,sub,"task condition (7-way)",1/len(np.unique(cond)))

print()
print("(b) subject fingerprinting -- identify the subject from one condition")
subs=np.unique(sub)
hits=tot=0
for c in np.unique(cond):
    pi=np.where(cond==c)[0]
    gi=np.where(cond!=c)[0]
    gal=np.array([X[gi][sub[gi]==s].mean(axis=0) for s in subs])
    Pz=(X[pi]-X[pi].mean(0))/(X[pi].std(0)+1e-9)
    Gz=(gal-gal.mean(0))/(gal.std(0)+1e-9)
    sim=Pz@Gz.T
    guess=subs[np.argmax(sim,axis=1)]
    hits+=int((guess==sub[pi]).sum());tot+=len(pi)
acc=hits/tot;ch=1/len(subs)
p=stats.binomtest(hits,tot,ch,alternative="greater").pvalue
lo,hi=stats.binomtest(hits,tot).proportion_ci(0.95)
print(f"  {'subject identity':<26} acc {acc:.4f}  [95% CI {lo:.3f}, {hi:.3f}]  "
      f"chance {ch:.4f}  p={p:.3e}   {'ABOVE CHANCE' if p<0.05 else 'ns'}")
res["subject identity"]={"accuracy":acc,"ci95":[lo,hi],"chance":ch,
                          "p":float(p),"n":tot,"n_subjects":len(subs)}

print()
print("(c) diagnosis and demographics -- one row per subject (conditions averaged)")
Xs=np.array([X[sub==s].mean(axis=0) for s in subs])
meta=clin.set_index("subject_id").reindex(
    [s if s in set(clin.subject_id) else "sub-"+s for s in subs])
for col,kind in (("group","clf"),("sex","clf"),("age","reg")):
    if col not in meta.columns:
        print(f"  {col:<26} not in clinical table; skipped");continue
    y=meta[col].to_numpy()
    ok=pd.notna(y)
    if ok.sum()<20:
        print(f"  {col:<26} only {ok.sum()} labelled; skipped");continue
    Xi,yi=Xs[ok],y[ok]
    if kind=="clf":
        yi=pd.Series(yi).astype(str).to_numpy()
        maj=pd.Series(yi).value_counts(normalize=True).iloc[0]
        pipe=make_pipeline(StandardScaler(),PCA(n_components=min(20,len(yi)-2),
                            random_state=0),LogisticRegression(max_iter=5000))
        from sklearn.model_selection import StratifiedKFold
        pred=cross_val_predict(pipe,Xi,yi,
                                cv=StratifiedKFold(5,shuffle=True,random_state=0))
        a=accuracy_score(yi,pred);n=len(yi);k=int(round(a*n))
        p=stats.binomtest(k,n,maj,alternative="greater").pvalue
        print(f"  {col+' ('+str(n)+')':<26} acc {a:.4f}  majority {maj:.4f}  "
              f"p={p:.3f}   {'ABOVE CHANCE' if p<0.05 else 'ns'}")
        res[col]={"accuracy":a,"majority":float(maj),"p":float(p),"n":n}
    else:
        yi=pd.to_numeric(yi,errors="coerce");m2=pd.notna(yi)
        Xi,yi=Xi[m2],yi[m2].astype(float)
        from sklearn.model_selection import KFold
        pipe=make_pipeline(StandardScaler(),PCA(n_components=min(20,len(yi)-2),
                            random_state=0),RidgeCV(alphas=np.logspace(-3,4,20)))
        pred=cross_val_predict(pipe,Xi,yi,cv=KFold(5,shuffle=True,random_state=0))
        r,p=stats.pearsonr(pred,yi)
        print(f"  {col+' ('+str(len(yi))+')':<26} LOO-style r {r:+.4f}  p={p:.4f}   "
              f"{'ABOVE CHANCE' if p<0.05 else 'ns'}")
        res[col]={"r":float(r),"p":float(p),"n":int(len(yi))}

json.dump(res,open(OUT/"feature_validity.json","w"),indent=2)
print(f"\n  wrote {OUT/'feature_validity.json'}")
