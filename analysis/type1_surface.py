import json
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))

from config import config
from evaluate import cluster_evaluate

o=_R/"results"

CELLS=[(28,119,0.0),(28,119,"real")]
CELL_NAMES={0.0:"isotropic","real":"real_covariance"}

_RE=None

def _cov(d,al,rng):
    lm=np.arange(1,d+1,dtype=float)**(-al)
    lm=lm/lm.sum()*d
    q,_=np.linalg.qr(rng.standard_normal((d,d)))
    return (q*np.sqrt(lm))@q.T

def _wil(k,n,z=1.959963985):
    if n==0:
        return (float("nan"),float("nan"))
    p=k/n
    dn=1+z*z/n
    c=(p+z*z/(2*n))/dn
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/dn
    return (max(0.0,c-h),min(1.0,c+h))

def _pr(x):
    e=np.linalg.eigvalsh(np.cov(x,rowvar=False))
    e=e[e>1e-12]
    return float((e.sum()**2)/(e**2).sum()) if len(e) else float("nan")

def _draw(n,d,al,rng):
    global _RE
    if al=="real":
        if _RE is None:
            x=np.load(_R/"data"/"outputs"/"trained_fm_embeddings.npy").astype(float)
            xn=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
            _RE=(xn.mean(0),np.cov(xn,rowvar=False))
        mu,cv=_RE
        z=rng.multivariate_normal(mu,cv,size=n,method="svd")
    else:
        z=rng.standard_normal((n,d))@_cov(d,al,rng)
    return z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-8)

def _one(a):
    i,n,d,al,nd=a
    rng=np.random.default_rng(500_000+i)
    x=_draw(n,d,al,rng)
    e=cluster_evaluate()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _,k=e._select_best_silhouette(x,random_state=i,return_k=True)
        lb=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(x)
        pm=e.perm(x,lb,n_permutations=nd,random_state=i,
                  match_selection=False,match_geometry=False)
        pc=e.perm(x,lb,n_permutations=nd,random_state=i,
                  match_selection=True,match_geometry=True)
    return float(pm),float(pc),_pr(x)

def cell(n,d,al,nds,nd):
    a=[(i,n,d,al,nd) for i in range(nds)]
    with mp.Pool(min(nds,os.cpu_count() or 4)) as pl:
        rs=pl.map(_one,a)
    pm=np.array([r[0] for r in rs])
    pc=np.array([r[1] for r in rs])
    pr=float(np.mean([r[2] for r in rs]))
    km,kc=int((pm<0.05).sum()),int((pc<0.05).sum())
    lm,hm=_wil(km,nds)
    lc,hc=_wil(kc,nds)
    r={"n":n,"d":d,"alpha":al,"participation_ratio":pr,
       "n_datasets":nds,"n_draws":nd,
       "mis_reject":km/nds,"mis_ci":[lm,hm],
       "cor_reject":kc/nds,"cor_ci":[lc,hc],
       "mis_only":int(((pm<0.05)&(pc>=0.05)).sum()),
       "cor_only":int(((pc<0.05)&(pm>=0.05)).sum())}
    print(f"  n={n:<4d} a={str(al):<5s} PR={pr:6.2f} | mis {km/nds:.3f} "
          f"[{lm:.3f},{hm:.3f}] cor {kc/nds:.3f} [{lc:.3f},{hc:.3f}] "
          f"disc {r['mis_only']}/{r['cor_only']}",flush=True)
    return r

def main(nds=200,nd=200):
    cells={CELL_NAMES[a]:cell(n,dd,a,nds,nd) for n,dd,a in CELLS}
    d={"nominal_alpha":0.05,"cells":cells}
    o.mkdir(parents=True,exist_ok=True)
    (o/"type1_surface.json").write_text(json.dumps(d,indent=2))
    print(f"\nwrote {o}/type1_surface.json")
    return d

if __name__=="__main__":
    a=int(sys.argv[1]) if len(sys.argv)>1 else 200
    b=int(sys.argv[2]) if len(sys.argv)>2 else 200
    main(a,b)
