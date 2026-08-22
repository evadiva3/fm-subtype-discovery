import os
for v in("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[v]="1"
import sys,json,time
import numpy as np,pandas as pd
import multiprocessing as mp
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_R=Path(__file__).resolve().parents[1]
for _p in(_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))

from config import config

o=config.resultsRoot
KR=config.kmeansKRange
NI=config.kmeansNInit
SD=config.randomSeed
FL=config.minClusterSizeFloor
FR=config.minClusterSizeFraction


def nrm(x):
    return x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)


def pr(x):
    xc=x-x.mean(0)
    s=np.linalg.svd(xc,compute_uv=False)
    ev=s**2
    ev=ev[ev>ev.max()*1e-12]
    return float(ev.sum()**2/(ev**2).sum())


def feats():
    lab=pd.read_csv(config.kLabelPath)
    subs=lab["Subject_Id"].tolist()
    iu=np.triu_indices(config.nNodes,k=1)
    rows=[]
    for s in subs:
        d=config.subjectDataFolder/s
        fs=sorted(f for f in os.listdir(d) if "FCMatrixCondition" in f and f.endswith(".npy"))
        assert len(fs)==config.nConditions,f"{s}: {len(fs)} matrices"
        m=np.mean([np.load(d/f).astype(np.float64) for f in fs],axis=0)
        rows.append(m[iu])
    return subs,np.asarray(rows)


def sel(d,rs):
    n=d.shape[0]
    ms=max(FL,round(FR*n))
    sils=[]
    mem=[]
    for k in KR:
        lb=KMeans(n_clusters=k,n_init=NI,random_state=rs).fit_predict(d)
        sils.append(silhouette_score(d,lb))
        mem.append(int(np.bincount(lb).min()))
    ok=[i for i in range(len(KR)) if mem[i]>=ms]
    b=max(ok,key=lambda i:sils[i]) if ok else int(np.argmax(sils))
    return sils[b],KR[b]


_G={}


def _init(mu,S,Vt,n,k):
    _G["mu"],_G["S"],_G["Vt"],_G["n"],_G["k"]=mu,S,Vt,n,k


def _one(idx):
    out=[]
    for i in idx:
        rng=np.random.default_rng(SD+i)
        r=len(_G["S"])
        z=rng.standard_normal((_G["n"],r))
        amb=_G["mu"]+(z*(_G["S"]/np.sqrt(_G["n"]-1)))@_G["Vt"]
        sph=nrm(amb)
        gs,_=sel(sph,i)
        as_,_=sel(amb,i)
        lb=KMeans(n_clusters=_G["k"],n_init=NI,random_state=i).fit_predict(sph)
        gf=silhouette_score(sph,lb)
        lb=KMeans(n_clusters=_G["k"],n_init=NI,random_state=i).fit_predict(amb)
        af=silhouette_score(amb,lb)
        out.append((af,gf,as_,gs))
    return out


def main(nd=1000,np_=6):
    subs,X=feats()
    Xn=nrm(X)
    obs,ok=sel(Xn,0)
    print(f"{X.shape[0]}x{X.shape[1]} pr={pr(X):.2f} prSphere={pr(Xn):.2f} obs={obs:.6f} k={ok}")

    mu=Xn.mean(0)
    U,S,Vt=np.linalg.svd(Xn-mu,full_matrices=False)
    keep=S>S.max()*1e-12
    S,Vt=S[keep],Vt[keep]

    ch=[list(range(i,min(i+25,nd))) for i in range(0,nd,25)]
    t0=time.perf_counter()
    res=[]
    with mp.Pool(np_,initializer=_init,initargs=(mu,S,Vt,len(subs),ok)) as pool:
        for r in pool.imap(_one,ch):
            res.extend(r)
    a=np.array(res)

    nm=["misspecified","geometry","selection","corrected"]
    d={"n":len(subs),"d":int(X.shape[1]),"pr":pr(X),"prSphere":pr(Xn),
       "observed":float(obs),"k":int(ok),"nDraws":nd,"rungs":{},
       "elapsed":round(time.perf_counter()-t0,1)}
    for c in range(4):
        s=a[:,c]
        p=(int((s>=obs).sum())+1)/(nd+1)
        d["rungs"][nm[c]]={"p":p,"nullMean":float(s.mean()),"nullSd":float(s.std(ddof=0))}
        print(f"  {nm[c]:<13} p={p:.4f} mean={s.mean():.4f} sd={s.std(ddof=0):.4f}")

    o.mkdir(parents=True,exist_ok=True)
    (o/"raw_feature_ladder.json").write_text(json.dumps(d,indent=2))
    return d


if __name__=="__main__":
    a=int(sys.argv[1]) if len(sys.argv)>1 else 1000
    b=int(sys.argv[2]) if len(sys.argv)>2 else 6
    main(a,b)
