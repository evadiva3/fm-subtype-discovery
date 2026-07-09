import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from config import config  

class cluster_evaluate():

    def __init__(self):
        pass

    def silhouette(self,embed,label):
        return silhouette_score(embed,label)
    
    def _null_shuffle(self,embed):
        out=embed.copy()
        for j in range(out.shape[1]):
            np.random.shuffle(out[:,j])
        return out

    def perm(self,embed,label,n_permutations=None):
        n_permutations=config.nPermutations if n_permutations is None else n_permutations
        embed=np.asarray(embed)
        k=len(np.unique(label))
        real=self.silhouette(embed,label)
        c=0
        for i in range(0,n_permutations):
            null=self._null_shuffle(embed)
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

    #B refs, per-dim uniform box, s_k correction
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

    #1-SE rule: smallest k with gap(k)>=gap(k+1)-s(k+1)
    def gap_k(self,gaps,k_range):
        ks=list(k_range)
        for j in range(len(ks)-1):
            if gaps[ks[j]]["gap"]>=gaps[ks[j+1]]["gap"]-gaps[ks[j+1]]["s"]:
                return ks[j]
        return ks[-1]

        


        