# procedure matching does not establish calibration.
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from config import config

class cluster_evaluate:

    def __init__(self):
        pass

    def silhouette(self,embed,label):
        return silhouette_score(embed,label)

    def _null_mvn(self,embedN,rng):
        mu=embedN.mean(axis=0)
        cov=np.cov(embedN,rowvar=False)
        draws=rng.multivariate_normal(mu,cov,size=embedN.shape[0],method="svd")
        draws=draws/(np.linalg.norm(draws,axis=1,keepdims=True)+1e-08)
        return draws

    def _null_mvn_ambient(self,embedN,rng):
        mu=embedN.mean(axis=0)
        cov=np.cov(embedN,rowvar=False)
        return rng.multivariate_normal(mu,cov,size=embedN.shape[0],method="svd")

    def _select_best_silhouette(self,data,random_state=0,return_k=False):
        n=data.shape[0]
        minSize=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*n))
        sils=[]
        members=[]
        for k in config.kmeansKRange:
            lab=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=random_state).fit_predict(data)
            sils.append(self.silhouette(data,lab))
            members.append(int(np.bincount(lab).min()))
        passed=[i for i in range(len(config.kmeansKRange)) if members[i]>=minSize]
        if not passed:
            return (-1.0,None) if return_k else-1.0
        best=max(passed,key=lambda i:sils[i])
        if return_k:
            return (sils[best],int(config.kmeansKRange[best]))
        return sils[best]

    #Parametric diagnostic (historical name). Search uses identical seed/guard.
    #With match_selection=True, recompute the observed searched statistic.
    #Failed size guards give no rejection; null guard failures are counted at k=0.
    #Fixed-k mode compares supplied labels and is exploratory if labels were selected.
    def perm(self,embed,label,n_permutations=None,random_state=None,match_selection=True,match_geometry=True,return_k_counts=False,guard_fixed=True):
        n_permutations=config.nPermutations if n_permutations is None else n_permutations
        if not isinstance(n_permutations, (int,np.integer)) or n_permutations<1:
            raise ValueError("n_permutations must be a positive integer")
        embed=np.asarray(embed,dtype=float)
        if embed.ndim!=2 or not np.isfinite(embed).all() or np.any(np.linalg.norm(embed,axis=1)==0):
            raise ValueError("finite nonzero embedding rows required")
        if len(label)!=len(embed):
            raise ValueError("one label per embedding row required")
        k=len(np.unique(label))
        embedN=embed/(np.linalg.norm(embed,axis=1,keepdims=True)+1e-08)
        min_size=max(config.minClusterSizeFloor,round(config.minClusterSizeFraction*len(embedN)))
        if match_selection:
            real,realk=self._select_best_silhouette(embedN,random_state=0,return_k=True)
            if realk is None:
                return (1.0, {0:n_permutations}) if return_k_counts else 1.0
        else:
            labs,cnt=np.unique(label,return_counts=True)
            if guard_fixed and cnt.min()<min_size:
                return (1.0, {0:n_permutations}) if return_k_counts else 1.0
            real=self.silhouette(embedN,label)
        seed=config.randomSeed if random_state is None else random_state
        rng=np.random.default_rng(seed)
        c=0
        kc={int(kk):0 for kk in config.kmeansKRange}
        for i in range(0,n_permutations):
            null=self._null_mvn(embedN,rng) if match_geometry else self._null_mvn_ambient(embedN,rng)
            if match_selection:
                s,nk=self._select_best_silhouette(null,random_state=0,return_k=True)
            else:
                lab=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=0).fit_predict(null)
                s=self.silhouette(null,lab)
                if guard_fixed and np.bincount(lab).min()<min_size:
                    s=-1.0
                nk=k
            nk=0 if nk is None else int(nk)
            kc[nk]=kc.get(nk,0)+1
            if s>=real:
                c+=1
        p=(c+1)/(n_permutations+1)
        if return_k_counts:
            return (p,kc)
        return p

    def _inertia(self,x,k):
        km=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=0)
        km.fit(x)
        return km.inertia_

    def gap_stat(self,embed,k=[2,3,4],B=None,seed=None):
        B=getattr(config,"gapB",10) if B is None else B
        seed=config.randomSeed if seed is None else seed
        rng=np.random.default_rng(seed)
        embed=np.asarray(embed)
        lo=embed.min(axis=0)
        hi=embed.max(axis=0)
        out={}
        for i in k:
            wk=self._inertia(embed,i)
            logs=np.array([np.log(self._inertia(rng.uniform(lo,hi,embed.shape),i)) for _ in range(B)])
            gap=logs.mean()-np.log(wk)
            s=logs.std()*np.sqrt(1+1.0/B)
            out[i]= {"gap":float(gap),"s":float(s)}
        return out

    def gap_k(self,gaps,k_range):
        ks=list(k_range)
        for j in range(len(ks)-1):
            if gaps[ks[j]]["gap"]>=gaps[ks[j+1]]["gap"]-gaps[ks[j+1]]["s"]:
                return ks[j]
        return ks[-1]

    def gap_k_at_boundary(self,gaps,k_range):
        ks=list(k_range)
        for j in range(len(ks)-1):
            if gaps[ks[j]]["gap"]>=gaps[ks[j+1]]["gap"]-gaps[ks[j+1]]["s"]:
                return False
        return True
