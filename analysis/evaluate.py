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
    
    def gap_stat(self,embed,k=[2,3,4]):
        gaps={}
        for i in k:
            kmeans=KMeans(n_clusters=i)
            kmeans.fit(embed)
            real=kmeans.inertia_
            kmeans1=KMeans(n_clusters=i)
            ref=np.random.uniform(embed.min(),embed.max(),embed.shape)
            kmeans1.fit(ref)
            fake=kmeans1.inertia_
            gap=np.log(fake)-np.log(real)
            gaps[i]=gap
        return gaps

        


        