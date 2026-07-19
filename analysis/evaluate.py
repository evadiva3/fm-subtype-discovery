import numpy as np
import diptest
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from config import config

class cluster_evaluate():

    def __init__(self):
        pass

    def silhouette(self,embed,label):
        return silhouette_score(embed,label)
    
    def _null_mvn(self,embed,rng):
        mu=embed.mean(axis=0)
        cov=np.cov(embed,rowvar=False)
        return rng.multivariate_normal(mu,cov,size=embed.shape[0],method="svd")

    def perm(self,embed,label,n_permutations=None,random_state=None):
        n_permutations=config.nPermutations if n_permutations is None else n_permutations
        embed=np.asarray(embed)
        k=len(np.unique(label))
        real=self.silhouette(embed,label)
        seed=config.randomSeed if random_state is None else random_state
        rng=np.random.default_rng(seed)
        c=0
        for i in range(0,n_permutations):
            null=self._null_mvn(embed,rng)
            lab=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=i).fit_predict(null)
            s=self.silhouette(null,lab)
            if s>real:
                c+=1
        p=c/n_permutations
        return p
    
    def _inertia(self,x,k):
        km=KMeans(n_clusters=k,n_init=config.kmeansNInit,random_state=0)
        km.fit(x)
        return km.inertia_

    def gap_stat(self,embed,k=[2,3,4],B=None):
        B=getattr(config,"gapB",10) if B is None else B
        embed=np.asarray(embed)
        lo=embed.min(axis=0)
        hi=embed.max(axis=0)
        out={}
        for i in k:
            wk=self._inertia(embed,i)
            logs=np.array([np.log(self._inertia(np.random.uniform(lo,hi,embed.shape),i)) for _ in range(B)])
            gap=logs.mean()-np.log(wk)
            s=logs.std()*np.sqrt(1+1.0/B)
            out[i]={"gap":float(gap),"s":float(s)}
        return out

    def gap_k(self,gaps,k_range):
        ks=list(k_range)
        for j in range(len(ks)-1):
            if gaps[ks[j]]["gap"]>=gaps[ks[j+1]]["gap"]-gaps[ks[j+1]]["s"]:
                return ks[j]
        return ks[-1]

    def dip_pcs(self,embed,n=5):
        embed=np.asarray(embed)
        pcs=PCA(n_components=n,random_state=config.randomSeed).fit_transform(embed)
        out={}
        for i in range(n):
            dp,pv=diptest.diptest(pcs[:,i])
            out[i+1]={"dip":float(dp),"p":float(pv)}
        return out

        


        