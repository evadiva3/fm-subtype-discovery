import sys
from pathlib import Path
_R=Path(__file__).resolve().parents[1]
for _p in (_R,_R/"src",_R/"models",_R/"analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0,str(_p))
import numpy as np
from sklearn.cluster import KMeans
from config import config
from evaluate import cluster_evaluate

def _l2(X):
    return X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-8)

def _lab(X,k):
    return KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=config.randomSeed).fit_predict(X)

def test_perm_significant_clusters():
    r=np.random.default_rng(0)
    c=np.array([[6,0,0,0,0],[0,6,0,0,0],[0,0,6,0,0]],dtype=float)
    X=np.vstack([m+r.normal(0,0.3,(15,5)) for m in c])
    Xn=_l2(X)
    p=cluster_evaluate().perm(Xn,_lab(Xn,3),n_permutations=200)
    assert p<0.05

def test_perm_null_on_blob():
    r=np.random.default_rng(1)
    X=r.normal(0,1,(45,5))
    Xn=_l2(X)
    p=cluster_evaluate().perm(Xn,_lab(Xn,3),n_permutations=200)
    assert p>0.05

def test_null_preserves_cov():
    r=np.random.default_rng(2)
    A=r.normal(0,1,(5,5))
    b=r.normal(0,1,(40,5))@A.T
    ev=cluster_evaluate()
    g=np.random.default_rng(config.randomSeed)
    d=np.vstack([ev._null_mvn(b,g) for _ in range(300)])
    ci=np.cov(b,rowvar=False)
    cn=np.cov(d,rowvar=False)
    rel=np.linalg.norm(cn-ci)/np.linalg.norm(ci)
    assert rel<0.15
    iu=np.triu_indices(5,1)
    assert np.corrcoef(ci[iu],cn[iu])[0,1]>0.9

def test_gap_deterministic():
    r=np.random.default_rng(3)
    X=r.normal(0,1,(40,6))
    g1=cluster_evaluate().gap_stat(X,k=[2,3,4])
    g2=cluster_evaluate().gap_stat(X,k=[2,3,4])
    for k in g1:
        assert abs(g1[k]["gap"]-g2[k]["gap"])<1e-12
        assert abs(g1[k]["s"]-g2[k]["s"])<1e-12
