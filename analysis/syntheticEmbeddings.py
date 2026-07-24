from config import config;
import numpy as np;
import pandas as pd;
import os;
import torch;
import warnings;
class syntheticEmbeddings():
    def __init__(self, path2Embeddings = None):
        if path2Embeddings is not None:
            try:
                self.embeddings = np.load(path2Embeddings);
            except FileNotFoundError:
                warnings.warn(f"Path Specified: {path2Embeddings} Does Not Exist - Check File Type. Using Default Path");
                self.embeddings = np.load(config.real2Synth);
        else:
            self.embeddings = np.load(config.real2Synth);
    def normCheck(self):
        check = np.linalg.norm(self.embeddings, axis=1);
        if(np.any(np.abs(check-1)<=config.normalizedTolerance)):
              raise RuntimeError(f"These Embeddings Have Already Been Normalized: Terminate");
    def splitGroups(self):
        groupList = [];
        rng = np.random.default_rng(config.synthSeedG);
        if(config.usePresets):
            groupList = config.presetGroups.copy();
        for _ in range(config.numRandGroups-len(groupList)):
            groupList.append(np.diff(np.insert(np.sort(rng.choice(np.arange(1,config.subjectAmt), size=config.clustersPerGroup-1, replace=False)),[0,config.clustersPerGroup-1], [0,config.subjectAmt])).tolist());
        return groupList;
    def calculate(self):
        groups = self.splitGroups();
        from clustering import cluster as clust;
        from sklearn.metrics import adjusted_rand_score as randScore;
        for group in groups:
            theClustering = clust(None, "nothing", [], []); 
            for run in range(0,config.runsPerDelta):
                rng = np.random.default_rng(config.synthSeedG+run);
                randomMatrix = rng.standard_normal((128,4));
                Q, R = np.linalg.qr(randomMatrix, mode="reduced");
                permutationIndex = rng.permutation(len(self.embeddings));
                unshuffle = np.argsort(permutationIndex);
                groupedEmbed = np.split(self.embeddings[permutationIndex],np.cumsum(group)[:-1]);
                sigmas = [np.std(self.embeddings@Q[:,groupIdx]) for groupIdx in range(0,len(groupedEmbed))];
                clusterList = [];
                trueLabels = np.repeat(np.arange(len(group)),group);
                trueLabels = trueLabels[unshuffle];
                for i in range(0,len(groupedEmbed)):
                    cluster = np.stack(groupedEmbed[i]);
                    deltaArray = np.array(config.deltas).reshape(-1, 1, 1);
                    offset = deltaArray * (sigmas[i]*Q[:,i]).reshape(1,1,128);
                    clusterList.append(cluster+offset);
                offsetEmbeds = np.concatenate(clusterList, axis=1)[:,unshuffle,:];
                for delta in range(0,offsetEmbeds.shape[0]):
                    package = theClustering.KMeansUse(torch.tensor(offsetEmbeds[delta,:,:]),np.arange(0,len(offsetEmbeds[delta,:,0])));
                    selectedK = package[0]["k_selected_silhouette"].iloc[0]
                    silhouetteScore, permP, sizeOk, bestLabels = package[0].loc[(package[0]["k"]==selectedK),"silhouette_score"].iloc[0], package[3], package[4], package[2];
                    ARI = randScore(trueLabels, bestLabels);
                    recoveredSizes = np.bincount(bestLabels);
                    kmeansOut = pd.DataFrame({"silhouetteScore": silhouetteScore, "selectedK": selectedK, "permutationScore": permP, "sizeOk":sizeOk, "ARI": ARI, "recoveredSizes":recoveredSizes, "clustersRecovered":len(recoveredSizes),"group":"/".join(map(str,group)), "delta":config.deltas[delta], "run":run});
                    target = os.path.join(config.syntheticOutputs,".".join(map(str,group)),str(config.deltas[delta]));
                    os.makedirs(target, exist_ok=True);
                    kmeansOut.to_csv(os.path.join(target,f"run{run}.csv"));
    def main(self):
        self.normCheck();
        self.claculate();
if __name__ == "__main__":
    synthesize = syntheticEmbeddings(None);
    synthesize.main();