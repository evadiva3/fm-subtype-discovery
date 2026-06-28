#just methods
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class cluster_evaluate():
    def __init__(self):
        pass
    def silhouette(self,embed,label):
        return silhouette_score(embed,label)
    def perm(self,embed,label):
        real=self.silhouette(embed,label)
        c=0
        for i in range(0,1000):
            b=np.random.permutation(label)
            shuffled=self.silhouette(embed,b)
            if shuffled>real:
                c+=1
        p=c/1000
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

        


        