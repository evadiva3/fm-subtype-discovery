from config import config
import numpy as np
import pandas as pd
import os
import torch
import warnings

class syntheticEmbeddings():
    def __init__(self, p=None):
        if p is not None:
            try:
                self.emb=np.load(p)
            except FileNotFoundError:
                warnings.warn(f"path {p} not found, using default")
                self.emb=np.load(config.real2Synth)
        else:
            self.emb=np.load(config.real2Synth)

    def normCheck(self):
        chk=np.linalg.norm(self.emb, axis=1)
        if(np.all(np.abs(chk-1)<=config.normalizedTolerance)):
            raise RuntimeError("already normalized")

    def splitGroups(self):
        gl=[]
        rng=np.random.default_rng(config.synthSeedG)
        mn=max(config.minClusterSizeFloor, round(config.minClusterSizeFraction*config.subjectAmt))
        if(config.usePresets):
            gl=config.presetGroups.copy()
        while(len(gl)<config.numRandGroups):
            c=np.diff(np.insert(np.sort(rng.choice(np.arange(1,config.subjectAmt), size=config.clustersPerGroup-1, replace=False)),[0,config.clustersPerGroup-1], [0,config.subjectAmt])).tolist()
            if(min(c)>=mn):
                gl.append(c)
        return gl

    def calculate(self):
        gs=self.splitGroups()
        from clustering import cluster as clust
        from sklearn.metrics import adjusted_rand_score as randScore
        for g in gs:
            tc=clust(None, "nothing", [], [])
            for r in range(0,config.runsPerDelta):
                rng=np.random.default_rng(config.synthSeedG+r)
                rm=rng.standard_normal((128,4))
                Q, R=np.linalg.qr(rm, mode="reduced")
                pi=rng.permutation(len(self.emb))
                u=np.argsort(pi)
                ge=np.split(self.emb[pi],np.cumsum(g)[:-1])
                sg=[np.std(self.emb@Q[:,gi]) for gi in range(0,len(ge))]
                cl=[]
                tl=np.repeat(np.arange(len(g)),g)
                tl=tl[u]
                for i in range(0,len(ge)):
                    cu=np.stack(ge[i])
                    da=np.array(config.deltas).reshape(-1, 1, 1)
                    off=da * (sg[i]*Q[:,i]).reshape(1,1,128)
                    cl.append(cu+off)
                oe=np.concatenate(cl, axis=1)[:,u,:]
                for d in range(0,oe.shape[0]):
                    pk=tc.KMeansUse(torch.tensor(oe[d,:,:]),np.arange(0,len(oe[d,:,0])))
                    sk=pk[0]["k_selected_silhouette"].iloc[0]
                    ss, pp, so, bl=pk[0].loc[(pk[0]["k"]==sk),"silhouette_score"].iloc[0], pk[3], pk[4], pk[2]
                    ari=randScore(tl, bl)
                    rs=np.bincount(bl)
                    ko=pd.DataFrame({"silhouetteScore": ss, "selectedK": sk, "permutationScore": pp, "sizeOk":so, "ARI": ari, "recoveredSizes":rs, "clustersRecovered":len(rs),"group":"/".join(map(str,g)), "delta":config.deltas[d], "run":r})
                    t=os.path.join(config.syntheticOutputs,".".join(map(str,g)),str(config.deltas[d]))
                    os.makedirs(t, exist_ok=True)
                    ko.to_csv(os.path.join(t,f"run{r}.csv"))

    def main(self):
        self.normCheck()
        self.calculate()

if __name__ == "__main__":
    s=syntheticEmbeddings(None)
    s.main()
